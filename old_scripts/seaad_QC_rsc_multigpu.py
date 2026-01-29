#!/usr/bin/env python
# seaad_QC_rsc_multigpu_lazy.py
# Multi-GPU QC + Scrublet (no Zarr) with lazy H5AD streaming and spill safety
# ENV: rapids_singlecell

import os, gc, time
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp, cudf
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask
import dask.array as da
import numpy as np
import anndata as ad
import h5py
from scipy import sparse

# -----------------------
# Config
# -----------------------
DATA_DIR  = "/mnt/data/seaad_dlpfc"
H5AD_IN   = f"{DATA_DIR}/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
H5AD_OUT  = f"{DATA_DIR}/SEAAD_qc_scrublet_multigpu_lazy.h5ad"

ROW_CHUNK = 2000        # per-block rows; lower if memory tight
EXPECTED_RATE = 0.045
SIM_RATIO     = 2.0
N_PCS         = 15

# -----------------------
# Dask spill + comm safety
# -----------------------
# Enable GPU memory spill-to-host (prevents worker OOM)
os.environ.setdefault("DASK_CUDA_MEMORY_POOL", "rmm")
os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__SPILL", "true")
os.environ.setdefault("DASK_DISTRIBUTED__COMM__RETRY__COUNT", "3")
# If you try UCX first below, these help too:
os.environ.setdefault("UCX_TLS", "tcp,cuda_copy,cuda_ipc")
os.environ.setdefault("UCX_SOCKADDR_TLS_PRIORITY", "tcp")

# ==========================================================
# Global helper: safe block loader for CSR-backed .h5ad
# ==========================================================
def load_block(file_path, start, end):
    """Load sparse rows [start:end) from a CSR-backed h5ad into dense float32."""
    with h5py.File(file_path, "r") as h5:
        X_grp = h5["X"]
        indptr  = X_grp["indptr"]
        indices = X_grp["indices"]
        data    = X_grp["data"]
        shape   = tuple(X_grp.attrs["shape"])
        row_ptr = indptr[start:end + 1]
        start_ptr, end_ptr = row_ptr[0], row_ptr[-1]
        block_idx  = indices[start_ptr:end_ptr]
        block_data = data[start_ptr:end_ptr]
        block_iptr = row_ptr - start_ptr
        csr = sparse.csr_matrix(
            (block_data, block_idx, block_iptr),
            shape=(end - start, shape[1])
        )
        # Use float16 to reduce VRAM pressure
        return csr.toarray().astype(np.float32)

# ==========================================================
# Main
# ==========================================================
def main():
    t0 = time.time()

    # -----------------------
    # 1) Start Dask-CUDA
    # -----------------------
    print("🚀 Starting Dask-CUDA cluster...")
    # Prefer TCP for stability; switch to protocol="ucx" if your UCX stack is solid
    cluster = LocalCUDACluster(
        n_workers=4, threads_per_worker=1,
        protocol="tcp",
        rmm_pool_size="20GB",
        device_memory_limit="0.90",
        local_directory=f"{DATA_DIR}/dask-tmp",
        dashboard_address=":8787",
    )
    client = Client(cluster)
    print("✅ Dask dashboard:", client.dashboard_link)

    # -----------------------
    # 2) Init RAPIDS Memory Manager
    # -----------------------
    rmm.reinitialize(managed_memory=True, pool_allocator=True, devices=[0,1,2,3])
    cp.cuda.set_allocator(rmm_cupy_allocator)
    # Also re-init on workers (ensures they use RMM too)
    client.run(lambda: rmm.reinitialize(managed_memory=True, pool_allocator=True))

    gc.collect()

    # -----------------------
    # 3) Load H5AD lazily
    # -----------------------
    print("📂 Loading H5AD in backed mode …")
    adata_b = sc.read_h5ad(H5AD_IN, backed="r")
    print(f"  → backed: {adata_b.n_obs:,} cells × {adata_b.n_vars:,} genes")

    print("🔄 Wrapping X as Dask array …")
    X_data = adata_b.X
    try:
        # Dense HDF5 dataset path
        _ = X_data.ndim
        X_dask = da.from_array(X_data, chunks=(ROW_CHUNK, adata_b.n_vars)).astype(np.float32)

    except AttributeError:
        # Sparse CSR-backed dataset
        print("⚠️ Backed matrix is sparse (_CSRDataset). Streaming into Dask…")
        with h5py.File(H5AD_IN, "r") as h5:
            shape = tuple(h5["X"].attrs["shape"])
        X_dask = da.concatenate([
            da.from_delayed(
                dask.delayed(load_block)(H5AD_IN, i, min(i + ROW_CHUNK, shape[0])),
                shape=(min(ROW_CHUNK, shape[0] - i), shape[1]),
                dtype=np.float32
            )
            for i in range(0, shape[0], ROW_CHUNK)
        ])

    # Metadata to host memory (already pandas under backed mode)
    obs_df = adata_b.obs.copy(deep=True)
    var_df = adata_b.var.copy(deep=True)

    # Build lazy AnnData
    adata = ad.AnnData(X=X_dask, obs=obs_df, var=var_df)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    print(f"✅ AnnData (lazy): {adata.n_obs:,} × {adata.n_vars:,}")

    # -----------------------
    # 4) Move to GPU(s) + QC
    # -----------------------
    print("📦 Moving AnnData to GPU(s)…")
    rsc.get.anndata_to_GPU(adata)
    print("✅ On GPU(s)")

    print("🧪 Running QC metrics …")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="mt",   gene_family_prefix="MT-")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="ribo", gene_family_prefix="RPS")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="hb",   gene_family_prefix="HB")

    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo","hb"])

    # Basic QC filters
    adata = adata[(adata.obs["n_genes_by_counts"] > 200) &
                  (adata.obs["pct_counts_mt"] < 8), :].copy()
    rsc.pp.filter_genes(adata, min_cells=10)
    print(f"✅ After QC: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # -----------------------
    # 5) Scrublet
    # -----------------------
    print("🤖 Running Scrublet (multi-GPU)…")
    rsc.pp.scrublet(
        adata,
        expected_doublet_rate=EXPECTED_RATE,
        sim_doublet_ratio=SIM_RATIO,
        n_prin_comps=N_PCS,
        log_transform=False,
        random_state=0,
    )
    print("✅ Scrublet complete.")
    print(adata.obs[["doublet_score", "predicted_doublet"]].head())

    # Keep only singlets
    adata = adata[adata.obs["predicted_doublet"] == False].copy()
    print(f"✅ Keeping only singlets: {adata.n_obs:,} cells remain")

    # -----------------------
    # 6) Save to disk
    # -----------------------
    print("💾 Moving back to CPU and saving H5AD…")
    rsc.get.anndata_to_CPU(adata)
    gc.collect()

    if isinstance(adata.X, da.Array):
        adata.X = adata.X.compute()
    adata.write_h5ad(H5AD_OUT, compression="gzip")
    print(f"✅ Saved: {H5AD_OUT}")

    # -----------------------
    # 7) Cleanup
    # -----------------------
    client.close()
    cluster.close()
    try:
        adata_b.file.close()
    except Exception:
        pass
    print(f"⏱️ Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
