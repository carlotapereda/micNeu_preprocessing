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

# build pandas DataFrames column by canndata_to_CPUolumn (each is a zarr array)
obs = pd.DataFrame({c: np.array(root["obs"][c]) for c in obs_cols})
obs.index = np.array(root["obs_names"])

var = pd.DataFrame({c: np.array(root["var"][c]) for c in var_cols})
var.index = np.array(root["var_names"])

print(obs.shape, var.shape)

X_dask = da.from_zarr(root["X"]) 
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
if adata.is_view:
    adata = adata.copy()


# --- Flag QC gene families ---
rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix="MT-")
rsc.pp.flag_gene_family(adata, gene_family_name="ribo", gene_family_prefix="RPS")
rsc.pp.flag_gene_family(adata, gene_family_name="hb", gene_family_prefix="HB")

# Make QC happy: 1 column block, small row blocks to fit GPU memory
# 512 rows × 36,601 genes × 4 bytes ≈ ~71–75 MB per block on GPU
# If you still OOM, try 256. If you have plenty of headroom, 1024 is fine.
target_row_chunk = 512
adata.X = adata.X.rechunk((target_row_chunk, -1))

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
print("~~~~~~~~~~~~~~~~~~~~ 5 - Save (force CPU, no GPU graph, no shuffle)")

# ---- 5.0 Free GPU memory and move all metadata to CPU
import gc, numpy as np, dask.array as da, zarr, dask

try:
    import cupy as cp
    # drop any cached device allocations
    cp.get_default_memory_pool().free_all_blocks()
except Exception:
    pass

# Ensure AnnData is on CPU (metadata + structure)
import rapids_singlecell as rsc
rsc.get.anndata_to_CPU(adata)
gc.collect()

# ---- 5.1 Make sure .X (and layers/obsm/varm) are truly CPU-backed
def to_numpy_block(x):
    """Convert any CuPy/Dask/sparse blocks to dense NumPy arrays."""
    # CuPy -> NumPy
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    # SciPy sparse -> dense
    try:
        from scipy import sparse
        if sparse.issparse(x):
            return x.toarray()
    except Exception:
        pass
    # Dask may pass NumPy already; ensure dtype is float32 for X
    return np.asarray(x)

# Convert X blocks to NumPy and set NumPy "meta" so Dask dispatches CPU kernels
if isinstance(adata.X, da.Array):
    adata.X = adata.X.map_blocks(
        to_numpy_block,
        dtype=np.float32,
        meta=np.empty((0, 0), dtype=np.float32),
    )
else:
    # If X is a CuPy array or sparse or anything else, convert once
    adata.X = to_numpy_block(adata.X).astype(np.float32, copy=False)

# Convert layers / obsm / varm payloads to NumPy too
for coll in (adata.layers, adata.obsm, adata.varm):
    for k, v in list(coll.items()):
        if isinstance(v, da.Array):
            coll[k] = v.map_blocks(
                to_numpy_block,
                dtype=np.float32 if v.dtype.kind in "fbiu" else v.dtype,
                meta=np.empty((0, 0), dtype=np.float32) if v.ndim == 2 else np.empty((0,), dtype=np.float32),
            )
        else:
            # best-effort CPU conversion
            try:
                import cupy as cp
                if isinstance(v, cp.ndarray):
                    coll[k] = cp.asnumpy(v)
                    continue
            except Exception:
                pass
            try:
                from scipy import sparse
                if sparse.issparse(v):
                    coll[k] = v.toarray()
                    continue
            except Exception:
                pass

gc.collect()

# ---- 5.2 Reuse source chunking to avoid a big rechunk/shuffle
# We loaded from a Zarr store earlier as `root`. If it's not in scope, reopen it.
try:
    src_chunks = getattr(root["X"], "chunks", None)
except NameError:
    # fallback: open the source Zarr you read earlier
    src_root = zarr.open("/mnt/data/seaad_dlpfc/SEAAD_v2_XisUMIs.zarr", mode="r")
    src_chunks = getattr(src_root["X"], "chunks", None)

# If X is Dask, align its chunks to the source on-disk chunks (e.g., (5000, 2000))
if isinstance(adata.X, da.Array) and src_chunks is not None:
    adata.X = adata.X.rechunk(src_chunks)

# Verify everything is CPU
def is_cpu_array(x):
    if isinstance(x, da.Array):
        # Dask meta drives dispatch (must be NumPy)
        return x._meta.__class__.__module__.startswith("numpy")
    return isinstance(x, np.ndarray)

print(
    "✅ CPU check before save:",
    "X:", is_cpu_array(adata.X),
    "Layers:", all(is_cpu_array(v) for v in adata.layers.values()) if adata.layers else True,
    "obsm:", all(is_cpu_array(v) for v in adata.obsm.values()) if adata.obsm else True,
    "varm:", all(is_cpu_array(v) for v in adata.varm.values()) if adata.varm else True,
)

# ---- 5.3 Disable CuPy allocator during the write (paranoia guard)
try:
    import cupy as cp
    cp.cuda.set_allocator(None)
except Exception:
    pass

# ---- 5.4 Stick to Zarr v2 for maximum compatibility
zarr.storage.default_format = 2

# Choose output path and chunking; by default, use the current (aligned) chunks of X
out_file = f"{data_dir}/seaad_qc_cpu.zarr"

if isinstance(adata.X, da.Array):
    write_chunks = tuple(c[0] for c in adata.X.chunks)  # (rows_chunk, cols_chunk)
else:
    # Fallback if X is a single NumPy array (AnnData will tile it internally)
    write_chunks = (5000, 2000)

print(f"Writing safely to {out_file} with chunks={write_chunks} ...")

# ---- 5.5 Use CPU threads scheduler so the compute stays on NumPy
with dask.config.set(scheduler="threads"):
    adata.write_zarr(out_file, chunks=write_chunks)

print("✅ Saved:", out_file)

# ------------------------------
# OPTIONAL: streaming fallback
# If you *still* see memory pressure, uncomment the streaming writer below,
# which writes X in row blocks and cannot touch the GPU or do a Dask shuffle.
# ------------------------------
# from numcodecs import Blosc
# from scipy import sparse
# def _write_dataframe_group(df, grp, label):
#     import re
#     def sanitize(s): return re.sub(r"[\/\n\r\t]", "_", str(s))
#     col_order = []
#     for col in df.columns:
#         col_order.append(sanitize(col))
#         vals = df[col]
#         if str(vals.dtype).startswith("category"):
#             arr = vals.astype(str).to_numpy(dtype="U")
#         elif str(vals.dtype).startswith("datetime64"):
#             arr = vals.astype("datetime64[ns]").astype(str).astype("U")
#         elif vals.dtype == object:
#             arr = vals.astype(str).to_numpy(dtype="U")
#         else:
#             arr = vals.to_numpy()
#         grp.create_dataset(sanitize(col), data=arr)
#     grp.attrs["column-order"] = col_order
#
# def save_anndata_streaming_v2(adata, out_path, chunk_rows=5000, chunk_cols=2000, block_rows=5000):
#     zarr.storage.default_format = 2
#     compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
#     n_obs, n_vars = adata.shape
#     root = zarr.open_group(out_path, mode="w")
#     root.create_dataset("obs_names", data=np.array(adata.obs_names, dtype="U"))
#     root.create_dataset("var_names", data=np.array(adata.var_names, dtype="U"))
#     obs_grp = root.create_group("obs"); var_grp = root.create_group("var")
#     root.create_group("uns"); root.create_group("layers")
#     _write_dataframe_group(adata.obs, obs_grp, "obs")
#     _write_dataframe_group(adata.var, var_grp, "var")
#     X_ds = root.create_dataset(
#         "X", shape=(n_obs, n_vars),
#         chunks=(chunk_rows, chunk_cols),
#         dtype="float32", compressor=compressor
#     )
#     for start in range(0, n_obs, block_rows):
#         end = min(start + block_rows, n_obs)
#         block = adata.X[start:end, :]
#         if isinstance(block, da.Array):
#             block = block.compute(scheduler="threads")
#         try:
#             import cupy as cp
#             if isinstance(block, cp.ndarray):
#                 block = cp.asnumpy(block)
#         except Exception:
#             pass
#         if sparse.issparse(block):
#             block = block.toarray()
#         X_ds[start:end, :] = np.asarray(block, dtype=np.float32)
#         del block; gc.collect()
#     root.attrs.update({"shape": (n_obs, n_vars),
#                        "chunks": (chunk_rows, chunk_cols),
#                        "format": "zarr_v2",
#                        "created_by": "save_anndata_streaming_v2"})
#
# # To use fallback instead of AnnData writer:
# # save_anndata_streaming_v2(adata, f"{data_dir}/seaad_qc_cpu_streamed.zarr")
# # print("✅ Saved (streamed):", f"{data_dir}/seaad_qc_cpu_streamed.zarr")

print("~~~~~~~~~~~~~~~~~~~~ ALL DONE")
