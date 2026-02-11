import anndata as ad
import pandas as pd

SRC = "../../celltypist/seaad_celltypist_GPU_counts_only.h5ad"
adata = ad.read_h5ad(SRC, backed="r")

print("--- 🔍 SEA-AD Structure Check ---")
print(f"Index head: {adata.obs_names[:5].tolist()}")

# 1. Search for sequence-like columns (ACTG)
barcode_candidates = [
    col for col in adata.obs.columns 
    if adata.obs[col].astype(str).str.contains(r'^[ACGTNacgtn-]+$', regex=True).any()
]

# 2. Search for SEA-AD specific naming conventions
seaad_naming = [col for col in adata.obs.columns if "specimen" in col.lower() or "barcode" in col.lower()]

print(f"Likely barcode columns: {barcode_candidates}")
print(f"Naming-based candidates: {seaad_naming}")

# 3. Check for clinical mapping failures
print("\n--- 🧠 Clinical Value Check ---")
if "Braak" in adata.obs.columns:
    print(f"Unique Braak values: {adata.obs['Braak'].unique()}")
if "CERAD score" in adata.obs.columns:
    print(f"Unique CERAD values: {adata.obs['CERAD score'].unique()}")

# 4. Check for projid structure
if "sample_id" in adata.obs.columns:
    print(f"Sample/Projid Example: {adata.obs['sample_id'].iloc[0]}")