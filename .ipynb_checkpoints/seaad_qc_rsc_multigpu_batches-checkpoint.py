#!/usr/bin/env python
# seaad_QC_rsc_multigpu_batches.py
# Multi-GPU QC + Scrublet (batched safe version)
# ENV: rapids_singlecell

import os, gc, time
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask, dask.array as da
import numpy as np
import anndata as ad
import h5py
from scipy import sparse
import pandas as pd

# -----------------------
# Config
# -----------------------
DATA_DIR  = "/mnt/data/seaad_dlpfc"
H5AD_IN   = f"{DATA_DIR}/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
H5AD_OUT  = f"{DATA_DIR}/SEAAD_qc_scrublet_batched.h5ad"

ROW_CHUNK = 2000
EXPECTED_RATE = 0.045
SIM_RATIO     = 2.0
N_PCS         = 15
N_TEST        = None    # For testing: 50k subset
BATCH_SIZE    = 10_000    # Process 10k cells per Scrublet batch

# -----------------------
# Dask / RAPIDS tuning
# -----------------------
os.environ.setdefault("DASK_CUDA_MEMORY_POOL", "rmm")
os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__SPILL", "true")
os.environ.setdefault("DASK_DISTRIBUTED__COMM__RETRY__COUNT", "3")
os.environ.setdefault("UCX_TLS", "tcp,cuda_copy,cuda_ipc")
os.environ.setdefault("UCX_SOCKADDR_TLS_PRIORITY", "tcp")

# ==========================================================
# Helper: load block for CSR-backed H5AD
# ==========================================================
def load_block(file_path, start, end, n_rows=None):
    """Load sparse rows [start:end) from CSR-backed h5ad into dense float32."""
    with h5py.File(file_path, "r") as h5:
        X_grp = h5["X"]
        indptr  = X_grp["indptr"]
        indices = X_grp["indices"]
        data    = X_grp["data"]
        full_shape = tuple(X_grp.attrs["shape"])
        n_total = full_shape[0] if n_rows is None else n_rows
        shape = (n_total, full_shape[1])

        row_ptr = indptr[start:end + 1]
        start_ptr, end_ptr = row_ptr[0], row_ptr[-1]
        block_idx  = indices[start_ptr:end_ptr]
        block_data = data[start_ptr:end_ptr]
        block_iptr = row_ptr - start_ptr
        csr = sparse.csr_matrix(
            (block_data, block_idx, block_iptr),
            shape=(end - start, shape[1])
        )
        return csr.toarray().astype(np.float32)

# ==========================================================
# Main
# ==========================================================
def main():
    t0 = time.time()
    print("🚀 Starting Dask-CUDA cluster...")
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

    # Init RMM
    rmm.reinitialize(managed_memory=True, pool_allocator=True, devices=[0,1,2,3])
    cp.cuda.set_allocator(rmm_cupy_allocator)
    client.run(lambda: rmm.reinitialize(managed_memory=True, pool_allocator=True))
    gc.collect()

    # -----------------------
    # Load subset lazily
    # -----------------------
    print("📂 Loading H5AD in backed mode …")
    adata_b = sc.read_h5ad(H5AD_IN, backed="r")
    n_obs = adata_b.n_obs if N_TEST is None else min(N_TEST, adata_b.n_obs)
    print(f"  → backed: {adata_b.n_obs:,} cells × {adata_b.n_vars:,} genes")
    print(f"  ⚙️ Using subset of {n_obs:,} cells for testing.")

    # Wrap expression matrix
    X_data = adata_b.X
    try:
        _ = X_data.ndim
        X_dask = da.from_array(X_data[:n_obs, :], chunks=(ROW_CHUNK, adata_b.n_vars)).astype(np.float32)
    except AttributeError:
        print("⚠️ Backed matrix is sparse (_CSRDataset). Streaming subset into Dask…")
        with h5py.File(H5AD_IN, "r") as h5:
            full_shape = tuple(h5["X"].attrs["shape"])
            shape = (n_obs, full_shape[1])
        X_dask = da.concatenate([
            da.from_delayed(
                dask.delayed(load_block)(H5AD_IN, i, min(i + ROW_CHUNK, n_obs), n_rows=n_obs),
                shape=(min(ROW_CHUNK, n_obs - i), shape[1]),
                dtype=np.float32
            )
            for i in range(0, n_obs, ROW_CHUNK)
        ])

    obs_df = adata_b.obs.iloc[:n_obs].copy(deep=True)
    var_df = adata_b.var.copy(deep=True)

    adata = ad.AnnData(X=X_dask, obs=obs_df, var=var_df)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    print(f"✅ AnnData subset: {adata.n_obs:,} × {adata.n_vars:,}")

    # -----------------------
    # GPU QC
    # -----------------------
    print("📦 Moving AnnData to GPU(s)…")
    rsc.get.anndata_to_GPU(adata)
    print("✅ On GPU(s)")

    print("🧪 Running QC metrics …")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="mt",   gene_family_prefix="MT-")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="ribo", gene_family_prefix="RPS")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="hb",   gene_family_prefix="HB")

    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo","hb"])
    adata = adata[(adata.obs["n_genes_by_counts"] > 200) &
                  (adata.obs["pct_counts_mt"] < 8), :].copy()
    rsc.pp.filter_genes(adata, min_cells=10)
    print(f"✅ After QC: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # -----------------------
    # Scrublet (batched)
    # -----------------------
    print("🤖 Running Scrublet in batches …")
    all_results = []

    for start in range(0, adata.n_obs, BATCH_SIZE):
        end = min(start + BATCH_SIZE, adata.n_obs)
        print(f"➡️  Processing batch {start:,}–{end:,} …")

        # Extract batch into memory
        X_batch = adata.X[start:end, :]
        if isinstance(X_batch, da.Array):
            X_batch = X_batch.compute()
        ad_batch = ad.AnnData(X=X_batch,
                              obs=adata.obs.iloc[start:end].copy(),
                              var=adata.var.copy())

        # Run on GPU
        rsc.get.anndata_to_GPU(ad_batch)
        rsc.pp.scrublet(
            ad_batch,
            expected_doublet_rate=EXPECTED_RATE,
            sim_doublet_ratio=SIM_RATIO,
            n_prin_comps=N_PCS,
            log_transform=False,
            random_state=0
        )

        # Move to CPU and store results
        rsc.get.anndata_to_CPU(ad_batch)
        df = ad_batch.obs[["doublet_score", "predicted_doublet"]].copy()
        all_results.append(df)
        del ad_batch, X_batch
        gc.collect()

    # Merge results
    scrublet_results = pd.concat(all_results, axis=0)
    adata.obs[["doublet_score", "predicted_doublet"]] = scrublet_results
    print("✅ Scrublet complete for all batches.")

    # Keep only singlets
    adata = adata[adata.obs["predicted_doublet"] == False].copy()
    print(f"✅ Keeping only singlets: {adata.n_obs:,} cells remain")

    # -----------------------
    # Save results
    # -----------------------
    print("💾 Moving back to CPU and saving H5AD…")

    # Move back to CPU
    rsc.get.anndata_to_CPU(adata)
    gc.collect()

    # ⚠️ If X is still a Dask array, compute it before saving
    if isinstance(adata.X, da.Array):
        print("⚙️ Converting Dask array to in-memory NumPy before save…")
        adata.X = adata.X.compute()

    # Also, convert sparse matrices to CSR for smaller file size
    from scipy import sparse
    if sparse.issparse(adata.X) and not isinstance(adata.X, sparse.csr_matrix):
        adata.X = adata.X.tocsr()

    print(f"🧠 Final shape before saving: {adata.n_obs:,} × {adata.n_vars:,}")
    adata.write_h5ad(H5AD_OUT, compression="gzip")
    print(f"✅ Saved: {H5AD_OUT}")

    print(f"⏱️ Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()

