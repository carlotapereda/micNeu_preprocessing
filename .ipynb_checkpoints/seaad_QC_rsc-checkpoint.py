import os, time, gc, cupy as cp, cudf, dask.array as da, numpy as np, pandas as pd, zarr
import anndata as ad, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm
import cudf


data_dir = "/mnt/data/seaad_dlpfc"
os.makedirs(data_dir, exist_ok=True)

rmm.reinitialize(managed_memory=False, pool_allocator=True, devices=[0,1,2,3])
cp.cuda.set_allocator(rmm_cupy_allocator)
gc.collect()



print("~~~~~~~~~~~~~~~~~~~ 0 - loading h5ad")
root = zarr.open("/mnt/data/seaad_dlpfc/SEAAD_v2_XisUMIs.zarr", mode="r")

# obs and var are groups of individual columns
obs_cols = list(root["obs"].array_keys())
var_cols = list(root["var"].array_keys())

# build pandas DataFrames column by column (each is a zarr array)
obs = pd.DataFrame({c: np.array(root["obs"][c]) for c in obs_cols})
obs.index = np.array(root["obs_names"])

var = pd.DataFrame({c: np.array(root["var"][c]) for c in var_cols})
var.index = np.array(root["var_names"])

print(obs.shape, var.shape)

X_dask = da.from_zarr(root["X"], chunks=root["X"].chunks)
X_dask = X_dask.rechunk({0: 'auto', 1: -1}) 
adata1 = ad.AnnData(X=X_dask, obs=obs, var=var)

# --- move to GPU (in-place for this RAPIDS version)
rsc.get.anndata_to_GPU(adata1)
adata = adata1
print("Moved to GPU")
del adata1
gc.collect()

# --- Filter APOE directly on GPU ---
print("~~~~~~~~~~~~~~~~~~~ 1 - filtering by APOE (on GPU)")
adata = adata[adata.obs['APOE Genotype'].isin(['3/3', '3/4', '4/4']), :]


print("~~~~~~~~~~~~~~~~~~~~ 2 - checking number of cells per patient")
patientIDcolumnName = 'Donor ID'

# Counts per patient BEFORE ≥1000
counts_before = adata.obs[patientIDcolumnName].value_counts().sort_index()
if isinstance(counts_before, cudf.Series):
    counts_before = counts_before.to_pandas()

# Plot (CPU side)
fig, ax = plt.subplots(figsize=(8, 6))
counts_before.plot(kind='bar', ax=ax)
ax.set_title('Cell counts per patient (APOE-filtered)')
ax.set_xlabel('Donor ID')
ax.set_ylabel('Number of cells')
out_path = f"{data_dir}/cell_counts_apoe_filtered.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print("Saved:", out_path)

# Filter ≥1000 cells/patient
keep_ids = counts_before.index[counts_before >= 1000]
adata = adata[adata.obs[patientIDcolumnName].isin(keep_ids), :]
gc.collect()

# Recompute counts per patient after filter
counts_after = adata.obs[patientIDcolumnName].value_counts().sort_index()
if isinstance(counts_after, cudf.Series):
    counts_after = counts_after.to_pandas()

# Plot histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(counts_before, bins=30, edgecolor='black')
axes[0].set_title("Cells per patient (Before ≥1000)")
axes[0].axvline(x=1000, color='red', linestyle='--', label='Threshold: 1000')
axes[1].hist(counts_after, bins=30, edgecolor='black')
axes[1].set_title("Cells per patient (After ≥1000)")
axes[1].axvline(x=1000, color='red', linestyle='--', label='Threshold: 1000')
out_path = f"{data_dir}/cell_count_hists_before_after.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print("Saved:", out_path)
del counts_before, counts_after; gc.collect()

print("~~~~~~~~~~~~~~~~~~~~ 3 - Low quality cells")

# --- Flag QC gene families ---
rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix="MT-")
rsc.pp.flag_gene_family(adata, gene_family_name="ribo", gene_family_prefix="RPS")
rsc.pp.flag_gene_family(adata, gene_family_name="hb", gene_family_prefix="HB")

# --- GPU QC metrics ---
rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo","hb"])
print("QC metrics computed on GPU")

# --- Filter low-quality cells ---
adata = adata[adata.obs["n_genes_by_counts"] > 200, :]
adata = adata[adata.obs["pct_counts_mt"] < 8, :]
print(f"Number of cells after filtering low-quality: {adata.n_obs}")

print("~~~~~~~~~~~~~~~~~~~~ 4 - Basic gene/cell filters")
rsc.pp.filter_genes(adata, min_cells=10)
print(f"After gene filtering: {adata.shape}")
gc.collect()

# Save the two QC scatter plots
print("Saving QC scatter plots...")

sc.settings.figdir = data_dir
sc.pl.scatter(
    adata, x="total_counts", y="pct_counts_mt",
    save="_total_vs_pctmt.pdf", show=False
)
sc.pl.scatter(
    adata, x="total_counts", y="n_genes_by_counts",
    save="_total_vs_ngenes.pdf", show=False
)
print("QC scatter plots saved to:", data_dir)

print("~~~~~~~~~~~~~~~~~~~~ 5 - Save")
print("Freeing GPU memory before writing...")
cp.get_default_memory_pool().free_all_blocks()

print("Moving to CPU (lazy Dask transfer, safe)...")
rsc.get.anndata_to_CPU(adata)
gc.collect()

print("Writing to Zarr from CPU...")
adata.write_zarr(f"{data_dir}/seaad_qc_cpu.zarr", chunks=(10000, 1000))

out_file = f"{data_dir}/seaad_qc_gpu.zarr"
adata.write_zarr(out_file, chunks=(10000, 1000))
print("Saved:", out_file)

print("~~~~~~~~~~~~~~~~~~~~ ALL DONE")
