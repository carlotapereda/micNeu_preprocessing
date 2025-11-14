import anndata as ad
import pandas as pd

adata = ad.read_h5ad("merged_allcells.h5ad", backed="r")

cols = adata.obs.columns

print(f"Found {len(cols)} obs columns:\n")

for c in cols:
    print(f"{c}: {adata.obs[c].dtype}")
