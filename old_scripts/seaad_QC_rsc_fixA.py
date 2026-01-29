#!/usr/bin/env python
# seaad_QC_rsc_fixA.py — GPU memory–safe QC + gene filtering + Zarr save
#ENV - rapids_singlecell

import os, time, gc, cupy as cp, cudf, dask.array as da, numpy as np, pandas as pd, zarr
import anndata as ad, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm
import cudf


# ================================
# CONFIG
# ================================
data_dir = "/mnt/data/seaad_dlpfc"
os.makedirs(data_dir, exist_ok=True)

# ✅ Use a single GPU pool (not all 4)
rmm.reinitialize(managed_memory=True, pool_allocator=True, devices=[0])
cp.cuda.set_allocator(rmm_cupy_allocator)
gc.collect()

print("~~~~~~~~~~~~~~~~~~~ 0 - loading h5ad")
root = zarr.open(os.path.join(data_dir, "SEAAD_v2_XisUMIs.zarr"), mode="r")

obs_cols = list(root["obs"].array_keys())
var_cols = list(root["var"].array_keys())

obs = pd.DataFrame({c: np.array(root["obs"][c]) for c in obs_cols})
obs.index = np.array(root["obs_names"])
var = pd.DataFrame({c: np.array(root["var"][c]) for c in var_cols})
var.index = np.array(root["var_names"])
print(obs.shape, var.shape)

X_dask = da.from_zarr(root["X"])
adata1 = ad.AnnData(X=X_dask, obs=obs, var=var)
adata1.obs_names_make_unique()
adata1.var_names_make_unique()

# ---- move to GPU ----
rsc.get.anndata_to_GPU(adata1)
adata = adata1
del adata1; gc.collect()
print("Moved to GPU")

# ================================
# 1. Filter APOE genotypes
# ================================
print("~~~~~~~~~~~~~~~~~~~ 1 - filtering by APOE (on GPU)")
adata = adata[adata.obs['APOE Genotype'].isin(['3/3', '3/4', '4/4']), :].copy()

# ================================
# 2. Filter by donor cell count
# ================================
print("~~~~~~~~~~~~~~~~~~~~ 2 - checking number of cells per patient")
pid = 'Donor ID'

counts_before = adata.obs[pid].value_counts().sort_index()
if isinstance(counts_before, cudf.Series):
    counts_before = counts_before.to_pandas()

fig, ax = plt.subplots(figsize=(8, 6))
counts_before.plot(kind='bar', ax=ax)
ax.set_title('Cell counts per patient (APOE-filtered)')
ax.set_xlabel('Donor ID'); ax.set_ylabel('Number of cells')
fig.savefig(f"{data_dir}/cell_counts_apoe_filtered.pdf", bbox_inches="tight")
plt.close(fig)

keep_ids = counts_before.index[counts_before >= 1000]
adata = adata[adata.obs[pid].isin(keep_ids), :].copy()
gc.collect()

counts_after = adata.obs[pid].value_counts().sort_index()
if isinstance(counts_after, cudf.Series):
    counts_after = counts_after.to_pandas()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(counts_before, bins=30, edgecolor='black')
axes[0].axvline(1000, color='red', linestyle='--')
axes[0].set_title("Before ≥1000")
axes[1].hist(counts_after, bins=30, edgecolor='black')
axes[1].axvline(1000, color='red', linestyle='--')
axes[1].set_title("After ≥1000")
fig.savefig(f"{data_dir}/cell_count_hists_before_after.pdf", bbox_inches="tight")
plt.close(fig)
del counts_before, counts_after; gc.collect()

# ================================
# 3. QC metrics
# ================================
print("~~~~~~~~~~~~~~~~~~~~ 3 - Low quality cells")
if adata.is_view:
    adata = adata.copy()

rsc.pp.flag_gene_family(adata=adata, gene_family_name="mt", gene_family_prefix="MT-")
rsc.pp.flag_gene_family(adata=adata, gene_family_name="ribo", gene_family_prefix="RPS")
rsc.pp.flag_gene_family(adata=adata, gene_family_name="hb", gene_family_prefix="HB")


# moderate row blocks
adata.X = adata.X.rechunk((512, -1))
rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"])
print("QC metrics computed on GPU")

qc_mask = (adata.obs["n_genes_by_counts"] > 200) & (adata.obs["pct_counts_mt"] < 8)
adata = adata[qc_mask, :].copy()
print(f"Number of cells after filtering low-quality: {adata.n_obs}")

# ================================
# 4. Gene filtering (Fix A)
# ================================
print("~~~~~~~~~~~~~~~~~~~~ 4 - Basic gene/cell filters (GPU-safe)")
nvars = adata.n_vars
row_chunk = 2048        # adjust to 1024 if memory tight
col_chunk = min(4096, nvars)
adata.X = adata.X.rechunk((row_chunk, col_chunk))
print("Rechunked for filter_genes:", adata.X.chunks)

# keep RMM allocator (do NOT disable)
try:
    rsc.pp.filter_genes(adata, min_cells=10)
except Exception as e:
    print("⚠️ GPU filter_genes failed, check memory:", e)
print(f"After gene filtering: {adata.n_obs} cells × {adata.n_vars} genes")
cp.get_default_memory_pool().free_all_blocks(); gc.collect()

# ================================
# 5. Save (CPU-only)
# ================================
print("~~~~~~~~~~~~~~~~~~~~ 5 - Save (force CPU)")
rsc.get.anndata_to_CPU(adata)
gc.collect()

def to_numpy_block(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception: pass
    from scipy import sparse
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)

if isinstance(adata.X, da.Array):
    adata.X = adata.X.map_blocks(to_numpy_block, dtype=np.float32,
                                 meta=np.empty((0, 0), dtype=np.float32))
else:
    adata.X = to_numpy_block(adata.X).astype(np.float32, copy=False)

for coll in (adata.layers, adata.obsm, adata.varm):
    for k, v in list(coll.items()):
        if isinstance(v, da.Array):
            coll[k] = v.map_blocks(to_numpy_block,
                                   dtype=np.float32 if v.dtype.kind in "fbiu" else v.dtype,
                                   meta=np.empty((0, 0), dtype=np.float32))
        else:
            try:
                import cupy as cp
                if isinstance(v, cp.ndarray):
                    coll[k] = cp.asnumpy(v)
                    continue
            except Exception: pass
            from scipy import sparse
            if sparse.issparse(v): coll[k] = v.toarray()

gc.collect()

zarr.storage.default_format = 2
out_file = os.path.join(data_dir, "seaad_qc_cpu.zarr")
write_chunks = (2000, 1000)
print(f"Writing safely to {out_file} with chunks={write_chunks} ...")

import dask
with dask.config.set(scheduler="threads"):
    adata.write_zarr(out_file, chunks=write_chunks)

print("✅ Saved:", out_file)
print("~~~~~~~~~~~~~~~~~~~~ ALL DONE")
