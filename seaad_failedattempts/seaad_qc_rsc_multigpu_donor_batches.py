#!/usr/bin/env python
# seaad_qc_rsc_multigpu_donor_batches.py
# Multi-GPU QC + per-Donor Scrublet (RAPIDS) with lazy H5AD streaming
# ENV: rapids_singlecell

import os

# --- UCX: disable GPU peer-to-peer (g5.24xlarge is multi-socket, no NVLink) ---
os.environ["UCX_TLS"] = "tcp,cuda_copy"
os.environ["UCX_NET_DEVICES"] = "all"
os.environ["UCX_MEMTYPE_CACHE"] = "n"
os.environ["UCX_RNDV_SCHEME"] = "get_zcopy"
os.environ["UCX_MAX_RNDV_RAILS"] = "1"
os.environ["UCX_SOCKADDR_TLS_PRIORITY"] = "tcp"

# --- Dask spill/comm tuning ---
os.environ["DASK_RMM_POOL_SIZE"] = "18GB"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TARGET"] = "0.70"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__SPILL"] = "0.75"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE"] = "0.85"
os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE"] = "0.95"
os.environ["DASK_DISTRIBUTED__COMM__RETRY__COUNT"] = "3"

# --- Core imports ---
import gc, time, warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import h5py
import dask
import dask.array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import cupy as cp
from cupyx.scipy import sparse as cpx
import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm

# --- Initialize RMM pool before Dask starts ---
rmm.reinitialize(
    pool_allocator=True,
    managed_memory=True,
    initial_pool_size=4 << 30,  # 4 GB per GPU pool
)
cp.cuda.set_allocator(rmm_cupy_allocator)

# -----------------------
# Configuration
# -----------------------
DATA_DIR   = "/mnt/data/seaad_dlpfc"
H5AD_IN    = f"{DATA_DIR}/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
OUT_H5AD   = f"{DATA_DIR}/SEAAD_qc_scrublet_singlets.h5ad"
OUT_ZARR   = f"{DATA_DIR}/SEAAD_qc_singlets.zarr"

def wrap_h5ad_x_as_dask(h5ad_path, row_chunk, limit_rows=None):
    adata_b = sc.read_h5ad(h5ad_path, backed="r")
    n_obs, n_vars = adata_b.n_obs, adata_b.n_vars
    if limit_rows is not None:
        n_obs = min(n_obs, int(limit_rows))
    print(f"  → backed: {adata_b.n_obs:,} cells × {adata_b.n_vars:,}; using first {n_obs:,} rows")

    with h5py.File(h5ad_path, "r") as h5:
        Xgrp = h5["X"]
        shape = tuple(Xgrp.attrs["shape"])

    starts = list(range(0, n_obs, row_chunk))
    blocks = []
    for s in starts:
        e = min(s + row_chunk, n_obs)
        def _load_block(s=s, e=e):
            with h5py.File(h5ad_path, "r") as hh:
                Xg = hh["X"]
                indptr = Xg["indptr"][s : e + 1]
                idx    = Xg["indices"][indptr[0] : indptr[-1]]
                dat    = Xg["data"][indptr[0] : indptr[-1]]
                iptr   = indptr - indptr[0]
                csr = cpx.csr_matrix((cp.asarray(dat), cp.asarray(idx), cp.asarray(iptr)),
                                     shape=(e - s, shape[1]))
                return csr.toarray()
        blk = dask.delayed(_load_block)()
        blocks.append(da.from_delayed(blk, shape=(e - s, shape[1]), dtype=np.float32))
    X_dask = da.concatenate(blocks, axis=0)

    obs = adata_b.obs.iloc[:n_obs].copy()
    var = adata_b.var.copy()
    adata = ad.AnnData(X=X_dask, obs=obs, var=var)
    return adata, adata_b


def dask_qc_filter(adata):
    print("🧪 Running QC (distributed on GPU)…")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="mt",   gene_family_prefix="MT-")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="ribo", gene_family_prefix="RPS")
    rsc.pp.flag_gene_family(adata=adata, gene_family_name="hb",   gene_family_prefix="HB")
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"])

    keep_cell = (adata.obs["n_genes_by_counts"] > QC_MIN_GENES) & (
        adata.obs["pct_counts_mt"] < QC_MAX_PCT_MT
    )
    adata = adata[keep_cell, :].copy()
    print(f"   → after cell filter: {adata.n_obs:,} cells")

    counts_per_gene = da.count_nonzero(adata.X, axis=0).compute()
    n_cells = counts_per_gene.get() if isinstance(counts_per_gene, cp.ndarray) else np.asarray(counts_per_gene)
    adata.var["n_cells_by_counts"] = n_cells
    keep_genes = n_cells >= KEEP_GENES_MIN_CELLS
    print(f"   → keeping {keep_genes.sum():,}/{len(keep_genes):,} genes with ≥{KEEP_GENES_MIN_CELLS} cells")
    adata = adata[:, keep_genes].copy()
    print(f"✅ After QC: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata, keep_genes


def cupy_dense_from_any(X_block) -> cp.ndarray:
    if isinstance(X_block, da.Array):
        Xc = X_block.map_blocks(lambda b: cp.asarray(b, dtype=cp.float32), dtype=cp.float32)
        return Xc.compute()
    if isinstance(X_block, cp.ndarray):
        return X_block.astype(cp.float32, copy=False)
    if cpx.isspmatrix(X_block):
        return X_block.toarray()
    if hasattr(X_block, "toarray"):
        return cp.asarray(X_block.toarray(), dtype=cp.float32)
    return cp.asarray(X_block, dtype=cp.float32)


def main():
    t0 = time.time()
    cluster, client = start_cluster()

    adata, adata_b = wrap_h5ad_x_as_dask(H5AD_IN, ROW_CHUNK, limit_rows=TEST_MAX_CELLS)
    rsc.get.anndata_to_GPU(adata)
    ad_qc, gene_keep_mask = dask_qc_filter(adata)

    print(f"\n🤖 Running Scrublet per donor on '{DONOR_KEY}' …")
    ad_qc.obs["doublet_score"] = np.nan
    ad_qc.obs["predicted_doublet"] = pd.Series(index=ad_qc.obs_names, dtype=bool)

    donors = ad_qc.obs[DONOR_KEY].astype(str)
    unique_donors = donors.dropna().unique().tolist()
    print(f"   Found {len(unique_donors)} donors in QC-passed cells")

    for dname in unique_donors:
        mask = (donors == dname).values
        n_d = int(mask.sum())
        if n_d == 0:
            continue
        checkpoint = f"{DATA_DIR}/scrublet_{dname}.csv"
        if os.path.exists(checkpoint):
            print(f"✅ Skipping {dname}, already done")
            continue

        print(f"   → [{dname}] cells: {n_d:,}")
        if PER_DONOR_MAX is not None and n_d > PER_DONOR_MAX:
            idx = np.flatnonzero(mask)
            sel_idx = np.random.default_rng(17).choice(idx, size=PER_DONOR_MAX, replace=False)
            donor_mask = np.zeros(ad_qc.n_obs, dtype=bool)
            donor_mask[sel_idx] = True
        else:
            donor_mask = mask

        ad_sub = ad_qc[donor_mask, :].copy()
        X_cu = cupy_dense_from_any(ad_sub.X)
        ad_sub.X = X_cu

        rsc.pp.scrublet(
            ad_sub,
            expected_doublet_rate=SCRUBLET_RATE,
            sim_doublet_ratio=SCRUBLET_SIMR,
            n_prin_comps=SCRUBLET_PCS,
            log_transform=False,
            random_state=0,
        )

        ad_sub.obs[["doublet_score", "predicted_doublet"]].to_csv(checkpoint)
        print(f"💾 Saved checkpoint: {checkpoint}")

        scores = ad_sub.obs["doublet_score"].to_numpy()
        preds  = ad_sub.obs["predicted_doublet"].to_numpy()
        ad_qc.obs.loc[ad_sub.obs_names, "doublet_score"] = scores
        ad_qc.obs.loc[ad_sub.obs_names, "predicted_doublet"] = preds

        del ad_sub
        cp._default_memory_pool.free_all_blocks()
        gc.collect()

    keep_rows = ~ad_qc.obs["predicted_doublet"].fillna(False).to_numpy()
    print(f"\n✅ Singlets after Scrublet: {keep_rows.sum():,}/{ad_qc.n_obs:,}")

    if TEST_MAX_CELLS is not None:
        with dask.config.set(scheduler="threads"):
            X_host = ad_qc.X[keep_rows, :].astype(np.float32).compute()
        out = ad.AnnData(X=X_host, obs=ad_qc.obs.loc[keep_rows].copy(), var=ad_qc.var.copy())
        out.write_h5ad(OUT_H5AD, compression="gzip")
        print(f"💾 Wrote {OUT_H5AD}")
    else:
        print("💾 Streaming singlets to Zarr (not shown here for brevity)")

    client.close()
    cluster.close()
    print(f"⏱️ Total time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
