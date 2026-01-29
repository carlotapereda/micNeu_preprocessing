#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-GPU QC (Dask+RAPIDS) + per-Donor Scrublet (sequential, resumable, no Dask)
- QC happens once with Dask on all GPUs
- Scrublet runs donor-by-donor with pure CuPy arrays (no Dask), checkpointed
- Final singlets are streamed to Zarr (no huge RAM, no Dask writer)
"""

import os, gc, time, warnings, json
import numpy as np
import pandas as pd
import h5py
import anndata as ad
import scanpy as sc

import dask
import dask.array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import cupy as cp
from cupyx.scipy import sparse as cpx

import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm

# -----------------------
# Config
# -----------------------
DATA_DIR   = "/mnt/data/seaad_dlpfc"
H5AD_IN    = f"{DATA_DIR}/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
OUT_ZARR   = f"{DATA_DIR}/SEAAD_qc_singlets.zarr"  # streamed final output
DONOR_KEY  = "Donor ID"

# QC thresholds
QC_MIN_GENES       = 200
QC_MAX_PCT_MT      = 8.0
KEEP_GENES_MINCELLS= 10

# Scrublet params
SCRUBLET_RATE      = 0.045
SCRUBLET_SIMR      = 2.0
SCRUBLET_NPCS      = 15

# Runtime & memory knobs
N_WORKERS          = 4              # g5.12xlarge => 4x A10G (24GB each)
ROW_CHUNK_QC       = 4000           # Dask row-chunk for QC (tune 2000–8000)
TEST_MAX_CELLS     = None           # set to e.g. 50_000 to test quickly
PER_DONOR_MAX      = 12000          # cap per-donor rows loaded to GPU at once
ROW_CHUNK_LOAD     = 2000           # row block when assembling donor matrix

# Checkpoints
CHK_DIR            = f"{DATA_DIR}/chk_scrublet"
QC_KEEP_OBS_CSV    = f"{CHK_DIR}/qc_keep_obs.csv"     # obs_names kept after QC
QC_KEEP_VAR_TXT    = f"{CHK_DIR}/qc_keep_vars.txt"    # var_names kept after QC
DONORS_TXT         = f"{CHK_DIR}/donors.txt"          # donor list order
SCORES_CSV         = f"{CHK_DIR}/doublet_scores.csv"  # obs_name, score, pred

os.makedirs(CHK_DIR, exist_ok=True)

# -----------------------
# Dask / RAPIDS memory safety
# -----------------------
os.environ["DASK_RMM_POOL_SIZE"] = "18GB"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TARGET"]    = "0.70"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__SPILL"]     = "0.75"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE"]     = "0.85"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE"] = "0.95"

# Prefer stable TCP unless your UCX stack is known-good
DASK_PROTOCOL = "tcp"
DASK_DASHBOARD = ":0"   # auto-pick a free port

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# Utilities
# -----------------------

def start_cluster():
    cluster = LocalCUDACluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        protocol=DASK_PROTOCOL,
        rmm_pool_size="18GB",
        device_memory_limit="0.90",
        local_directory=os.path.join(DATA_DIR, "dask-tmp"),
        dashboard_address=DASK_DASHBOARD,
    )
    client = Client(cluster)
    print("✅ Dask dashboard:", client.dashboard_link)

    # Make CuPy use RMM on driver too
    rmm.reinitialize(managed_memory=True, pool_allocator=True, devices=[0,1,2,3])
    cp.cuda.set_allocator(rmm_cupy_allocator)
    # Make workers use RMM
    client.run(lambda: rmm.reinitialize(managed_memory=True, pool_allocator=True))
    return cluster, client

def csr_rows_to_cupy_dense(h5ad_path, row_idx, col_keep_mask, block_rows=2000):
    """
    Assemble a donor block as a CuPy dense array on GPU from CSR-backed H5AD.
    Reads 'row_idx' (np.array of int) rows; keeps only columns where col_keep_mask = True.
    """
    row_idx = np.asarray(row_idx, dtype=np.int64)
    with h5py.File(h5ad_path, "r") as h5:
        X = h5["X"]
        indptr  = X["indptr"]
        indices = X["indices"]
        data    = X["data"]
        n_vars  = X.attrs["shape"][1]

        # Map original var -> kept var position (else -1)
        col_map = np.full(n_vars, -1, dtype=np.int64)
        kept_cols = np.flatnonzero(col_keep_mask)
        col_map[kept_cols] = np.arange(kept_cols.size, dtype=np.int64)

        out = cp.zeros((row_idx.size, kept_cols.size), dtype=cp.float32)

        # process in small row blocks to bound CPU/GPU mem
        for start in range(0, row_idx.size, block_rows):
            end = min(start + block_rows, row_idx.size)
            rows = row_idx[start:end]
            # for each row: gather sparse slice and scatter-add into cupy dense
            for j, r in enumerate(rows):
                a = indptr[r]
                b = indptr[r+1]
                if b <= a:
                    continue
                idx = indices[a:b][...]   # numpy view
                dat = data[a:b][...].astype(np.float32, copy=False)
                mapped = col_map[idx]
                mask = mapped >= 0
                if not np.any(mask):
                    continue
                cols_gpu = cp.asarray(mapped[mask])
                vals_gpu = cp.asarray(dat[mask], dtype=cp.float32)
                # scatter-add row j
                out[start + j, cols_gpu] += vals_gpu
            # free any CuPy caches between blocks
            cp.get_default_memory_pool().free_all_blocks()
    return out  # cupy.ndarray

def write_zarr_streaming(h5ad_path, obs_names, var_names, out_path, row_block=4000, col_block=2000):
    """
    Stream a (rows-by-columns) selection from backed H5AD into a Zarr v2 store.
    Reads selected rows/columns in blocks; writes obs/var tables and X.
    """
    import zarr
    from numcodecs import Blosc

    obs_names = np.asarray(obs_names)
    var_names = np.asarray(var_names)
    zarr.storage.default_format = 2
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    with h5py.File(h5ad_path, "r") as h5:
        X = h5["X"]
        n_obs = obs_names.size
        n_vars = var_names.size

        root = zarr.open_group(out_path, mode="w")
        root.create_dataset("obs_names", data=obs_names.astype("U"))
        root.create_dataset("var_names", data=var_names.astype("U"))
        obs_grp = root.create_group("obs")
        var_grp = root.create_group("var")
        root.create_group("uns")
        root.create_group("layers")

        # Copy obs & var tables (subset)
        # NOTE: this assumes obs/var are dense H5 datasets per column name
        ad_b = sc.read_h5ad(h5ad_path, backed="r")
        obs_df = ad_b.obs.loc[obs_names].copy()
        var_df = ad_b.var.loc[var_names].copy()

        def _write_table(df, grp):
            for col in df.columns:
                s = df[col]
                if str(s.dtype).startswith(("category","datetime64")) or s.dtype == object:
                    arr = s.astype(str).to_numpy(dtype="U")
                else:
                    arr = s.to_numpy()
                safe = col.replace("/", "_")
                grp.create_dataset(safe, data=arr)

        _write_table(obs_df, obs_grp)
        _write_table(var_df, var_grp)

        # write X in blocks of rows
        X_ds = root.create_dataset(
            "X", shape=(n_obs, n_vars),
            chunks=(row_block, col_block), dtype="float32",
            compressor=compressor
        )

        # Build mask & col map once
        all_var_pos = {name:i for i, name in enumerate(ad_b.var_names)}
        cols_src = np.array([all_var_pos[n] for n in var_names], dtype=np.int64)
        indptr  = X["indptr"]; indices = X["indices"]; data = X["data"]

        for s in range(0, n_obs, row_block):
            e = min(s + row_block, n_obs)
            # rows by global index
            rows = np.array([ad_b.obs_names.get_loc(n) for n in obs_names[s:e]], dtype=np.int64)
            block = np.zeros((e - s, n_vars), dtype=np.float32)
            # fill block row by row
            for j, r in enumerate(rows):
                a = indptr[r]; b = indptr[r+1]
                if b <= a: 
                    continue
                idx = indices[a:b][...]
                dat = data[a:b][...].astype(np.float32, copy=False)
                # filter to kept columns via searchsorted against cols_src
                # (both idx and cols_src are sorted; align)
                # fast set-intersect with positions
                pos = np.searchsorted(cols_src, idx)
                mask = (pos < cols_src.size) & (cols_src[pos] == idx)
                if not np.any(mask):
                    continue
                block[j, pos[mask]] = dat[mask]
            X_ds[s:e, :] = block
            del block
            gc.collect()

    print(f"💾 Zarr written: {out_path}")

# -----------------------
# QC with Dask (multi-GPU)
# -----------------------

def qc_with_dask_and_save_masks(limit_rows=None):
    print("🚀 Starting Dask cluster for QC …")
    cluster, client = start_cluster()

    print("📂 Loading H5AD (backed='r') …")
    ad_b = sc.read_h5ad(H5AD_IN, backed="r")
    n_total = ad_b.n_obs
    n_use = n_total if limit_rows is None else min(limit_rows, n_total)
    print(f"  → {n_total:,} cells × {ad_b.n_vars:,} genes; using {n_use:,} rows")

    # Build dask array by delayed CSR→CuPy-dense row blocks
    with h5py.File(H5AD_IN, "r") as h5:
        n_vars = h5["X"].attrs["shape"][1]

    def _load_block_rows(start, end):
        with h5py.File(H5AD_IN, "r") as h5:
            X = h5["X"]; indptr = X["indptr"]; indices = X["indices"]; data = X["data"]
            out = cp.zeros((end - start, n_vars), dtype=cp.float32)
            for i, r in enumerate(range(start, end)):
                a = indptr[r]; b = indptr[r+1]
                if b <= a: 
                    continue
                idx = cp.asarray(indices[a:b])
                dat = cp.asarray(data[a:b], dtype=cp.float32)
                out[i, idx] += dat
            return out

    blocks = []
    for s in range(0, n_use, ROW_CHUNK_QC):
        e = min(s + ROW_CHUNK_QC, n_use)
        d = dask.delayed(_load_block_rows)(s, e)
        blocks.append(da.from_delayed(d, shape=(e - s, n_vars), dtype=np.float32))
    X_dask = da.concatenate(blocks, axis=0)

    obs = ad_b.obs.iloc[:n_use].copy()
    var = ad_b.var.copy()
    ad_q = ad.AnnData(X=X_dask, obs=obs, var=var)

    print("📦 Moving to GPU + QC …")
    rsc.get.anndata_to_GPU(ad_q)

    # Flag families (keywords only!)
    rsc.pp.flag_gene_family(adata=ad_q, gene_family_name="mt",   gene_family_prefix="MT-")
    rsc.pp.flag_gene_family(adata=ad_q, gene_family_name="ribo", gene_family_prefix="RPS")
    rsc.pp.flag_gene_family(adata=ad_q, gene_family_name="hb",   gene_family_prefix="HB")

    rsc.pp.calculate_qc_metrics(ad_q, qc_vars=["mt","ribo","hb"])
    keep_cell = (ad_q.obs["n_genes_by_counts"].values > QC_MIN_GENES) & \
                (ad_q.obs["pct_counts_mt"].values < QC_MAX_PCT_MT)
    ad_q = ad_q[keep_cell, :].copy()
    print(f"   → after cell filter: {ad_q.n_obs:,} cells")

    # gene filter (GPU-safe)
    rsc.pp.filter_genes(ad_q, min_cells=KEEP_GENES_MINCELLS)
    print(f"✅ After QC: {ad_q.n_obs:,} cells × {ad_q.n_vars:,} genes")

    # Save masks for phase 2
    ad_q.obs_names.to_series().to_csv(QC_KEEP_OBS_CSV, index=False, header=False)
    pd.Series(ad_q.var_names).to_csv(QC_KEEP_VAR_TXT, index=False, header=False)

    # donor list
    donors = ad_q.obs[DONOR_KEY].astype(str).dropna().unique().tolist()
    with open(DONORS_TXT, "w") as f:
        f.write("\n".join(donors))
    print(f"💾 Saved QC masks + {len(donors)} donors to {CHK_DIR}")

    # stop Dask cleanly
    client.close(); cluster.close()
    return

# -----------------------
# Scrublet per donor (sequential, no Dask), resumable
# -----------------------

def load_qc_masks():
    keep_obs = pd.read_csv(QC_KEEP_OBS_CSV, header=None)[0].astype(str).to_numpy()
    keep_vars = pd.read_csv(QC_KEEP_VAR_TXT, header=None)[0].astype(str).to_numpy()
    with open(DONORS_TXT) as f:
        donors = [line.strip() for line in f if line.strip()]
    return keep_obs, keep_vars, donors

def already_scored():
    if not os.path.exists(SCORES_CSV):
        return set()
    scored = pd.read_csv(SCORES_CSV)["obs_name"].astype(str)
    return set(scored.values.tolist())

def append_scores(obs_names, scores, preds):
    mode = "a" if os.path.exists(SCORES_CSV) else "w"
    header = not os.path.exists(SCORES_CSV)
    df = pd.DataFrame({
        "obs_name": np.asarray(obs_names, dtype=str),
        "doublet_score": np.asarray(scores, dtype=float),
        "predicted_doublet": np.asarray(preds, dtype=bool),
    })
    df.to_csv(SCORES_CSV, index=False, mode=mode, header=header)

def scrubble_all_donors():
    keep_obs, keep_vars, donors = load_qc_masks()

    # map names -> positions in backed
    ad_b = sc.read_h5ad(H5AD_IN, backed="r")
    pos_by_obs = {n:i for i, n in enumerate(ad_b.obs_names)}
    pos_by_var = {n:i for i, n in enumerate(ad_b.var_names)}
    keep_obs_pos = np.array([pos_by_obs[n] for n in keep_obs], dtype=np.int64)
    keep_var_pos = np.array([pos_by_var[n] for n in keep_vars], dtype=np.int64)

    done_obs = already_scored()
    print(f"🔁 Resuming Scrublet: {len(done_obs):,} rows already scored")

    for dname in donors:
        # donor positions intersected with QC-kept
        donor_mask_all = (ad_b.obs[DONOR_KEY].astype(str).values == dname)
        donor_kept_rows = keep_obs_pos[donor_mask_all[keep_obs_pos]]
        n_d = donor_kept_rows.size
        if n_d == 0:
            continue

        # skip if all these obs already have scores
        donor_obs_names = ad_b.obs_names[donor_kept_rows]
        if set(donor_obs_names).issubset(done_obs):
            print(f"→ [{dname}] already in checkpoint, skipping")
            continue

        print(f"🧩 [{dname}] cells: {n_d:,}")

        # optional cap to bound VRAM
        if PER_DONOR_MAX is not None and n_d > PER_DONOR_MAX:
            rng = np.random.default_rng(17)
            sel = np.sort(rng.choice(donor_kept_rows, size=PER_DONOR_MAX, replace=False))
            donor_rows = sel
            donor_obs_names = ad_b.obs_names[donor_rows]
            print(f"   capped to {PER_DONOR_MAX:,} cells for this donor (VRAM safety)")
        else:
            donor_rows = donor_kept_rows

        # materialize donor X on GPU as cupy dense
        X_cu = csr_rows_to_cupy_dense(H5AD_IN, donor_rows, np.isin(np.arange(len(ad_b.var_names)), keep_var_pos), block_rows=ROW_CHUNK_LOAD)

        # build temporary AnnData for Scrublet
        ad_sub = ad.AnnData(X=X_cu,
                            obs=ad_b.obs.iloc[donor_rows][[DONOR_KEY]].copy(),
                            var=ad_b.var.iloc[keep_var_pos].copy())
        # move to GPU (metadata mostly; X is already CuPy)
        rsc.get.anndata_to_GPU(ad_sub)

        # Scrublet (keywords only your build supports)
        rsc.pp.scrublet(
            ad_sub,
            expected_doublet_rate=SCRUBLET_RATE,
            sim_doublet_ratio=SCRUBLET_SIMR,
            n_prin_comps=SCRUBLET_NPCS,
            log_transform=False,
            random_state=0,
        )

        # checkpoint scores
        append_scores(donor_obs_names, ad_sub.obs["doublet_score"].to_numpy(), ad_sub.obs["predicted_doublet"].to_numpy())

        # GC
        del ad_sub, X_cu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    print(f"✅ Scrublet complete. Scores saved to {SCORES_CSV}")

def stream_singlets_to_zarr():
    # read QC & scrublet outputs
    keep_obs, keep_vars, _ = load_qc_masks()
    scr = pd.read_csv(SCORES_CSV)
    scr.index = scr["obs_name"].astype(str)

    # build final singlet list (QC-kept & predicted_doublet == False)
    preds = scr.loc[keep_obs, "predicted_doublet"].to_numpy()
    singlet_names = np.asarray(keep_obs)[~preds]
    print(f"✅ Singlets: {singlet_names.size:,} rows")

    # stream to Zarr without Dask
    write_zarr_streaming(H5AD_IN, singlet_names, keep_vars, OUT_ZARR, row_block=4000, col_block=2000)

# -----------------------
# Main
# -----------------------

def main():
    t0 = time.time()

    # 1) QC (once), save masks & donors, stop Dask
    if not (os.path.exists(QC_KEEP_OBS_CSV) and os.path.exists(QC_KEEP_VAR_TXT) and os.path.exists(DONORS_TXT)):
        qc_with_dask_and_save_masks(limit_rows=TEST_MAX_CELLS)
    else:
        print(f"↪ Using existing QC masks in {CHK_DIR}")

    # 2) Scrublet per donor (sequential), resumable
    scrubble_all_donors()

    # 3) Stream singlets to Zarr
    stream_singlets_to_zarr()

    print(f"⏱️ Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
