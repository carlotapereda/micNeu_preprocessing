import anndata as ad
import pandas as pd

adata = ad.read_h5ad("merged_allcells.h5ad", backed="r")
cols = adata.obs.columns
print(f"MERGED: Found {len(cols)} obs columns:\n")
for c in cols:
    print(f"{c}: {adata.obs[c].dtype}")

adata = ad.read_h5ad("../celltypist/mit_celltypist_GPU_counts_only.h5ad", backed="r")
cols = adata.obs.columns
print(f"MIT : Found {len(cols)} obs columns:\n")
for c in cols:
    print(f"{c}: {adata.obs[c].dtype}")

adata = ad.read_h5ad("../celltypist/seaad_celltypist_GPU_counts_only.h5ad", backed="r")
cols = adata.obs.columns
print(f"SEAAD: Found {len(cols)} obs columns:\n")
for c in cols:
    print(f"{c}: {adata.obs[c].dtype}")

adata = ad.read_h5ad("../celltypist/fujita_celltypist_GPU_counts_only.h5ad", backed="r")
cols = adata.obs.columns
print(f"FUJITA : Found {len(cols)} obs columns:\n")
for c in cols:
    print(f"{c}: {adata.obs[c].dtype}")
