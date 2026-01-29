#!/usr/bin/env python
# Zarr v2 ONLY: streamed .h5ad → .zarr conversion
# - Keeps ALL .obs and .var columns (dtype-safe)
# - Streams X in blocks (configurable)
# - Optionally copies all .layers
# Works with zarr==2.x and numcodecs (Blosc)
#env: zarr2-env

import anndata as ad
import numpy as np
import zarr
from numcodecs import Blosc
from scipy import sparse
import os, time, gc, sys, re

# ---------------- CONFIG ----------------
DATA_DIR   = "/mnt/data/seaad_dlpfc/"
H5AD_NAME  = "SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
H5AD_PATH  = os.path.join(DATA_DIR, H5AD_NAME)
ZARR_PATH  = os.path.join(DATA_DIR, "SEAAD_full_v2_streamed.zarr")

# Chunking of Zarr arrays
CHUNK_ROWS, CHUNK_COLS = 5000, 2000
# Row-blocks to stream while copying (memory control)
BLOCK_ROWS             = 5000

# Copy all layers as separate arrays (can be large!)
COPY_LAYERS = True
# ---------------------------------------

def sanitize(name: str) -> str:
    """Make a string safe as a Zarr path component."""
    return re.sub(r"[\/\n\r\t]", "_", str(name))

print("="*90)
print("🚀  STARTING CONVERSION  (.h5ad → Zarr v2, streamed; keep ALL .obs/.var)")
print(f"Input : {H5AD_PATH}")
print(f"Output: {ZARR_PATH}")
print("="*90)

t0 = time.time()

# ✅ Force v2 layout and compressor semantics
zarr.storage.default_format = 2
compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

print(f"[{time.strftime('%H:%M:%S')}] Opening H5AD in backed='r' mode ...")
adata_b = ad.read_h5ad(H5AD_PATH, backed="r")
n_obs, n_vars = adata_b.shape
print(f"[{time.strftime('%H:%M:%S')}] Metadata: {n_obs:,} cells × {n_vars:,} genes")
print(f"[{time.strftime('%H:%M:%S')}] obs columns: {len(adata_b.obs.columns)} | var columns: {len(adata_b.var.columns)}")

# Create root group
print(f"[{time.strftime('%H:%M:%S')}] Creating Zarr v2 store at: {ZARR_PATH}")
root = zarr.open_group(ZARR_PATH, mode="w")

# ---- X (main matrix), chunked + compressed (v2 API) ----
root.create_dataset(
    "X",
    shape=(n_obs, n_vars),
    chunks=(CHUNK_ROWS, CHUNK_COLS),
    dtype="float32",
    compressor=compressor,
)
print(f"✅ Created X with chunks={(CHUNK_ROWS, CHUNK_COLS)}")

# ---- Names (indices) ----
root.create_dataset("obs_names", data=np.array(adata_b.obs_names, dtype="U"))
root.create_dataset("var_names", data=np.array(adata_b.var_names, dtype="U"))
print("✅ Stored obs_names / var_names")

# ---- obs / var groups (write one column at a time, dtype-safe) ----
obs_grp = root.create_group("obs")
var_grp = root.create_group("var")
root.create_group("uns")
layers_grp = root.create_group("layers")
print("✅ Created groups: obs / var / uns / layers")

def write_dataframe_group(df, grp, label: str):
    """
    Write a pandas-like frame (.obs or .var) to a Zarr group
    column-by-column to avoid large memory spikes.
    - For non-numeric/object/categorical: store as Unicode strings.
    - For datetimes: store as ISO strings (portable).
    """
    col_order = []
    total = len(df.columns)
    print(f"[{time.strftime('%H:%M:%S')}] Writing ALL .{label} columns ({total}) ...")
    for i, col in enumerate(df.columns, 1):
        col_sanitized = sanitize(col)
        col_order.append(col_sanitized)
        vals = df[col]

        try:
            if str(vals.dtype).startswith("category"):
                arr = vals.astype(str).to_numpy(dtype="U")
            elif str(vals.dtype).startswith("datetime64"):
                # Portable: ISO strings (AnnData can reparse if needed)
                arr = vals.astype("datetime64[ns]").astype(str).astype("U")
            elif vals.dtype == object:
                arr = vals.astype(str).to_numpy(dtype="U")
            else:
                # numeric/bool/strings already ok
                arr = vals.to_numpy()
            grp.create_dataset(col_sanitized, data=arr)
        except Exception as e:
            print(f"  ⚠️  failed to write {label}['{col}'] ({vals.dtype}): {e} → storing as strings")
            arr = vals.astype(str).to_numpy(dtype="U")
            grp.create_dataset(col_sanitized, data=arr)

        if (i % 5 == 0) or (i == total):
            print(f"   ↳ wrote {label} column {i}/{total}")

    # Minimal hints to help tooling reconstruct ordering (not strictly required)
    grp.attrs["column-order"] = col_order

# Write obs/var column-by-column (no giant DataFrame in RAM)
write_dataframe_group(adata_b.obs, obs_grp, "obs")
write_dataframe_group(adata_b.var, var_grp, "var")

# ---- Stream X in blocks ----
n_batches = int(np.ceil(n_obs / BLOCK_ROWS))
print(f"[{time.strftime('%H:%M:%S')}] Streaming X in {n_batches} batches of {BLOCK_ROWS} rows")
t_stream = time.time()
for i, start in enumerate(range(0, n_obs, BLOCK_ROWS), 1):
    end = min(start + BLOCK_ROWS, n_obs)
    tb = time.time()
    block = adata_b.X[start:end, :]
    if sparse.issparse(block):
        block = block.toarray()
    block = block.astype("float32")
    root["X"][start:end, :] = block
    del block
    gc.collect()
    pct = (end / n_obs) * 100
    print(f"[{time.strftime('%H:%M:%S')}] ✓ X batch {i:>4}/{n_batches}  rows {start:,}-{end:,}  "
          f"{pct:5.2f}%  [{time.time()-tb:5.1f}s]")

print(f"[{time.strftime('%H:%M:%S')}] ✅ X streaming done in {time.time()-t_stream:.1f}s")

# ---- Optionally copy layers (same streaming pattern) ----
if COPY_LAYERS and len(adata_b.layers) > 0:
    print(f"[{time.strftime('%H:%M:%S')}] Copying layers: {list(adata_b.layers.keys())}")
    for lname in adata_b.layers.keys():
        print(f"  • Creating layer '{lname}'")
        layers_grp.create_dataset(
            sanitize(lname),
            shape=(n_obs, n_vars),
            chunks=(CHUNK_ROWS, CHUNK_COLS),
            dtype="float32",
            compressor=compressor,
        )
        t_layer = time.time()
        for i, start in enumerate(range(0, n_obs, BLOCK_ROWS), 1):
            end = min(start + BLOCK_ROWS, n_obs)
            tb = time.time()
            block = adata_b.layers[lname][start:end, :]
            if sparse.issparse(block):
                block = block.toarray()
            block = block.astype("float32")
            layers_grp[sanitize(lname)][start:end, :] = block
            del block
            gc.collect()
            if (i % 20 == 0) or (end == n_obs):
                pct = (end / n_obs) * 100
                print(f"     ↳ layer '{lname}' batch {i}  rows {start:,}-{end:,}  "
                      f"{pct:5.2f}%  [{time.time()-tb:5.1f}s]")
        print(f"  ✓ Finished layer '{lname}' in {time.time()-t_layer:.1f}s")
else:
    print(f"[{time.strftime('%H:%M:%S')}] Skipping layers (COPY_LAYERS={COPY_LAYERS})")

adata_b.file.close()

# ---- Root attrs (informational) ----
root.attrs["shape"]           = (n_obs, n_vars)
root.attrs["chunks"]          = (CHUNK_ROWS, CHUNK_COLS)
root.attrs["format"]          = "zarr_v2"
root.attrs["created_by"]      = "convert_h5ad_to_zarr_v2_streamed.py"
root.attrs["anndata_version"] = str(ad.__version__)

print("="*90)
print(f"✅  Conversion complete in {(time.time()-t0)/60:.1f} min")
print(f"Output: {ZARR_PATH}")
print("="*90)
