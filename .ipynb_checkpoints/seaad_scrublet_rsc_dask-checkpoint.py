#!/usr/bin/env python3
# seaad_scrublet_rsc_dask_v2.py
# Multi-GPU Scrublet for g5.12xlarge (4x A10G) with RAPIDS + Dask-CUDA

import os, gc, time, math, glob, warnings
import numpy as np
import pandas as pd
import anndata as ad
import rapids_singlecell as rsc
import anndata as ad
import dask.array as da
import zarr
import os

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import delayed

# ====================================================
# CONFIG
# ====================================================
DATA_DIR      = "/mnt/data/seaad_dlpfc"
IN_ZARR       = os.path.join(DATA_DIR, "seaad_qc_cpu.zarr")      # your QC CPU Zarr
OUT_ZARR      = os.path.join(DATA_DIR, "seaad_scrublet_cpu.zarr")
CHUNK_DIR     = os.path.join(DATA_DIR, "scrublet_chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

PATIENT_COL   = "Donor ID"
COUNTS_LAYER  = None          # auto-detect 'counts' if present
EXPECTED_RATE = 0.045
SIM_RATIO     = 2.0
N_PCS         = 15
CHUNK_SIZE    = 100_000       # tune per-VRAM; try 60–100k on A10G (24 GB)
N_WORKERS     = 4             # g5.12xlarge has 4 GPUs

# UCX / NCCL env (the 'rc' warning is benign; we also fall back to TCP)
os.environ.setdefault("UCX_TLS", "rc,tcp,cuda_copy,cuda_ipc")
os.environ.setdefault("UCX_SOCKADDR_TLS_PRIORITY", "tcp")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("RMM_LOG_FILE", os.path.join(DATA_DIR, "rmm.log"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ====================================================
# HELPER: Robust Zarr reader (NO 'backed' kwarg)
# ====================================================

def open_zarr_lazy(path):
    """Open a Zarr-backed AnnData lazily (handles AnnData ≥0.10 layout)."""
    root = zarr.open_group(path, mode="r", use_consolidated=False)

    # ---- obs / var ----
    obs_cols = list(root["obs"].array_keys())
    var_cols = list(root["var"].array_keys())

    obs = pd.DataFrame({c: np.array(root["obs"][c]) for c in obs_cols})
    var = pd.DataFrame({c: np.array(root["var"][c]) for c in var_cols})

    # ---- index names stored as _index ----
    obs.index = np.array(root["obs"]["_index"])
    var.index = np.array(root["var"]["_index"])

    # ---- lazy load X ----
    X_dask = da.from_zarr(root["X"])
    adata = ad.AnnData(X=X_dask, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    print(f"✅ Opened lazily: {adata.shape}, X chunks: {adata.X.chunks}")
    return adata

# ====================================================
# WORKER TASK
# ====================================================
def _scrublet_chunk_task(zarr_path,
                         sel_idx,
                         donor,
                         patient_col,
                         counts_layer,
                         expected_rate,
                         sim_ratio,
                         n_pcs,
                         out_parquet):
    """Run Scrublet on one chunk inside a Dask worker and write a parquet result."""
    import gc, pandas as pd, numpy as np, dask.array as da
    import rapids_singlecell as rsc
    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    # Route CuPy allocations through RMM pool on the worker
    cp.cuda.set_allocator(rmm_cupy_allocator)

    adata = open_zarr_lazy(zarr_path)
    sel_idx = np.asarray(sel_idx, dtype=np.int64)
    if sel_idx.size == 0:
        return None

    sub = adata[sel_idx, :].copy()

    # Keep VRAM steady
    if isinstance(sub.X, da.Array):
        sub.X = sub.X.rechunk((512, -1))

    # Move to GPU and run Scrublet
    rsc.get.anndata_to_GPU(sub)
    rsc.pp.scrublet(
        sub,
        layer=counts_layer,
        expected_doublet_rate=expected_rate,
        sim_doublet_ratio=sim_ratio,
        n_prin_comps=n_pcs,
        log_transform=False,
        verbose=False,
        copy=False,
        random_state=0,
    )

    # Extract results (handle cudf/pandas)
    ds = sub.obs["doublet_score"]
    to_pd = getattr(ds, "to_pandas", None)
    doublet_score = (to_pd() if callable(to_pd) else ds).to_numpy(dtype=np.float32)

    pd_flag = hasattr(sub.obs["predicted_doublet"], "to_pandas")
    predicted_doublet = (sub.obs["predicted_doublet"].to_pandas()
                         if pd_flag else sub.obs["predicted_doublet"]).to_numpy(dtype=bool)

    df = pd.DataFrame({
        "barcode": np.asarray(sub.obs_names),
        "doublet_score": doublet_score,
        "predicted_doublet": predicted_doublet,
        patient_col: donor,
    })

    # Atomic write
    tmp = out_parquet + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_parquet)

    # Clean up VRAM
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    del adata, sub, df
    gc.collect()
    return out_parquet

scrublet_chunk = delayed(_scrublet_chunk_task)

# ====================================================
# CLUSTER SETUP
# ====================================================
def start_cluster():
    """Start a 4‑GPU cluster; fall back to TCP if UCX not available."""
    try:
        cluster = LocalCUDACluster(
            n_workers=N_WORKERS,
            protocol="ucx",
            rmm_pool_size="16GB",          # per worker
            device_memory_limit="18GB",    # per worker
            local_directory=f"{DATA_DIR}/dask-tmp",
            dashboard_address=":8787",
        )
    except Exception:
        cluster = LocalCUDACluster(
            n_workers=N_WORKERS,
            protocol="tcp",
            rmm_pool_size="16GB",
            device_memory_limit="18GB",
            local_directory=f"{DATA_DIR}/dask-tmp",
            dashboard_address=":8787",
        )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)
    return cluster, client

# ====================================================
# MAIN
# ====================================================
def main():
    t0 = time.time()
    cluster, client = start_cluster()

    print("📂 Loading AnnData …")
    adata = open_zarr_lazy(IN_ZARR)
    print(f"✅ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Auto-detect counts layer if present
    global COUNTS_LAYER
    if COUNTS_LAYER is None and "counts" in getattr(adata, "layers", {}):
        COUNTS_LAYER = "counts"
    print(f"Counts layer: {COUNTS_LAYER or 'X (default)'}")

    # Enumerate donors
    donors = pd.Series(adata.obs[PATIENT_COL].values).astype("category").cat.categories.tolist()
    print(f"Found {len(donors)} donors.")

    # Build tasks
    tasks = []
    for d_i, donor in enumerate(donors, 1):
        mask = (adata.obs[PATIENT_COL].values == donor)
        idx = np.where(mask)[0]
        donor_n = idx.size
        if donor_n == 0:
            continue

        n_chunks = math.ceil(donor_n / CHUNK_SIZE)
        print(f"[{d_i}/{len(donors)}] Donor {donor} → {donor_n:,} cells ({n_chunks} chunks)")

        for ci in range(n_chunks):
            start, end = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, donor_n)
            sel = idx[start:end]
            out_parquet = os.path.join(CHUNK_DIR, f"scr_{d_i:03d}_{donor}_{ci:05d}.parquet")
            if os.path.exists(out_parquet):
                continue
            tasks.append(scrublet_chunk(
                IN_ZARR, sel, donor, PATIENT_COL, COUNTS_LAYER,
                EXPECTED_RATE, SIM_RATIO, N_PCS, out_parquet
            ))

    # Submit and wait
    if tasks:
        print(f"🚀 Submitting {len(tasks)} chunk task(s) to {N_WORKERS} GPUs …")
        futures = client.compute(tasks)
        for i, f in enumerate(futures, 1):
            f.result()  # propagate any worker exceptions (e.g., OOM)
            if i % 16 == 0 or i == len(futures):
                print(f"  … {i}/{len(futures)} chunks done")

    # Merge per-chunk outputs
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "scr_*.parquet")))
    if not files:
        print("⚠️ No chunk files found — nothing to merge.")
    else:
        print(f"📦 Merging {len(files)} chunk(s)…")
        parts = [pd.read_parquet(p) for p in files]
        final = pd.concat(parts, ignore_index=True)
        final = final.drop_duplicates("barcode").set_index("barcode")
        # Align back to adata.obs order
        adata.obs["doublet_score"] = pd.Series(index=adata.obs_names, dtype="float32")
        adata.obs["predicted_doublet"] = pd.Series(index=adata.obs_names, dtype="boolean")
        adata.obs.loc[final.index, "doublet_score"] = final["doublet_score"].astype("float32").values
        adata.obs.loc[final.index, "predicted_doublet"] = final["predicted_doublet"].astype("boolean").values

    # Save (CPU)
    print("💾 Writing final CPU Zarr …")
    rsc.get.anndata_to_CPU(adata)  # ensure CPU-backed metadata & arrays
    adata.write_zarr(OUT_ZARR, chunks=(2000, 1000))
    print("✅ Saved:", OUT_ZARR)

    client.close(); cluster.close()
    print(f"⏱️ Total time: {(time.time()-t0)/60:.2f} min")

# ====================================================
if __name__ == "__main__":
    main()
