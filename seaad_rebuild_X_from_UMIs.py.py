#!/usr/bin/env python
# rebuild_X_from_UMIs.py  (creates a UMI-based Zarr v2)
import anndata as ad, numpy as np, zarr, time, gc
from numcodecs import Blosc
from scipy import sparse
import os

DATA_DIR   = "/mnt/data/seaad_dlpfc/"
H5AD_NAME  = "SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
H5AD_PATH  = os.path.join(DATA_DIR, H5AD_NAME)
ZARR_PATH  = os.path.join(DATA_DIR, "SEAAD_v2_XisUMIs.zarr")

CHUNK_ROWS, CHUNK_COLS = 4000, 2000
BLOCK_ROWS             = 2000

zarr.storage.default_format = 2
compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

print(f"Opening {H5AD_PATH} in backed='r' mode ...")
adata_b = ad.read_h5ad(H5AD_PATH, backed="r")
n_obs, n_vars = adata_b.shape
print(f"{n_obs:,} cells × {n_vars:,} genes")

root = zarr.open_group(ZARR_PATH, mode="w")
root.create_dataset("X", shape=(n_obs, n_vars),
                    chunks=(CHUNK_ROWS, CHUNK_COLS),
                    dtype="float32", compressor=compressor)
root.create_dataset("obs_names", data=np.array(adata_b.obs_names, dtype="U"))
root.create_dataset("var_names", data=np.array(adata_b.var_names, dtype="U"))
obs_grp = root.create_group("obs"); var_grp = root.create_group("var")
root.create_group("uns"); root.create_group("layers")

def write_table(df, grp):
    for col in df.columns:
        s = df[col]
        arr = (s.astype(str).to_numpy(dtype="U")
               if s.dtype == object or str(s.dtype).startswith(("category","datetime64"))
               else s.to_numpy())
        grp.create_dataset(col.replace("/","_"), data=arr)

write_table(adata_b.obs, obs_grp)
write_table(adata_b.var, var_grp)

print("Streaming UMIs into X ...")
for start in range(0, n_obs, BLOCK_ROWS):
    end = min(start + BLOCK_ROWS, n_obs)
    blk = adata_b.layers["UMIs"][start:end, :]
    if sparse.issparse(blk):
        blk = blk.toarray()
    root["X"][start:end, :] = blk.astype("float32")
    del blk; gc.collect()
print("✅ Finished writing UMI-based X")

adata_b.file.close()
root.attrs["format"] = "zarr_v2"
root.attrs["created_by"] = "rebuild_X_from_UMIs.py"
print("Wrote:", ZARR_PATH)
