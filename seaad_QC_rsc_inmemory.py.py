#!/usr/bin/env python
# seaad_QC_rsc_inmemory.py
# Run QC and Scrublet fully in-memory on GPU
# ENV: rapids_singlecell

import os, gc, time
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp, cudf
import rmm

# -----------------------
# Config
# -----------------------
data_dir = "/mnt/data/seaad_dlpfc"
h5ad_in  = f"{data_dir}/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
h5ad_out = f"{data_dir}/SEAAD_qc_scrublet_gpu.h5ad"

# Initialize GPU memory pool
rmm.reinitialize(managed_memory=True, pool_allocator=True, devices=[0])
from rmm.allocators.cupy import rmm_cupy_allocator
cp.cuda.set_allocator(rmm_cupy_allocator)
gc.collect()

# -----------------------
# 0. Load H5AD directly
# -----------------------
print("📂 Loading H5AD...")
adata = sc.read_h5ad(h5ad_in)
print(f"✅ Loaded: {adata.shape}")

if not sp.issparse(adata.X):
    adata.X = sp.csr_matrix(adata.X)

# Move to GPU
rsc.get.anndata_to_GPU(adata)
print("Moved to GPU")

# -----------------------
# 1. QC filtering
# -----------------------
print("⚙️  QC filtering")
rsc.pp.flag_gene_family(adata, gene_family_name="mt",   gene_family_prefix="MT-")
rsc.pp.flag_gene_family(adata, gene_family_name="ribo", gene_family_prefix="RPS")
rsc.pp.flag_gene_family(adata, gene_family_name="hb",   gene_family_prefix="HB")

rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo","hb"])

# basic QC filters
adata = adata[(adata.obs["n_genes_by_counts"] > 200) & 
              (adata.obs["pct_counts_mt"] < 8), :].copy()

rsc.pp.filter_genes(adata, min_cells=10)
print(f"Remaining: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# -----------------------
# 2. Scrublet (on GPU)
# -----------------------
print("🤖 Running Scrublet...")
rsc.pp.scrublet(
    adata,
    layer=None,                # use adata.X
    expected_doublet_rate=0.045,
    sim_doublet_ratio=2.0,
    n_prin_comps=15,
    log_transform=False,
    random_state=0,
)
print("✅ Scrublet finished")

# Save doublet calls
print(adata.obs[["doublet_score","predicted_doublet"]].head())

# -----------------------
# 3. Back to CPU + Save
# -----------------------
print("💾 Moving back to CPU and saving H5AD")
rsc.get.anndata_to_CPU(adata)
gc.collect()
adata.write_h5ad(h5ad_out, compression="gzip")
print(f"✅ Saved: {h5ad_out}")
