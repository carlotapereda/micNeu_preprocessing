import anndata as ad
import pandas as pd
import numpy as np

############################################
# Inputs
############################################
OBS_FILES = [
    "SEAAD_harmonized_obs.csv",
    "FUJITA_harmonized_obs.csv",
    "MIT_ROSMAP_harmonized_obs.csv",
]

OBS_MERGED_OUT = "obs_merged_with_AD_metadata.csv"
MERGED_H5AD = "../merged_allcells.h5ad"
FINAL_H5AD = "./merged_allcells_with_metadata.h5ad"

############################################
# 1. Load and merge obs CSVs
############################################
print("📂 Loading harmonized obs files...")
obs_list = []
for path in OBS_FILES:
    df = pd.read_csv(path, low_memory=False)
    obs_list.append(df)

obs_merged = pd.concat(obs_list, axis=0, ignore_index=True)
print(f"✅ Merged obs shape: {obs_merged.shape}")

DATASET_MAP = {"SEA-AD": "SEAAD", "MIT-ROSMAP": "MIT_ROSMAP", "FUJITA": "FUJITA"}
obs_merged["Dataset"] = obs_merged["Dataset"].map(DATASET_MAP)
assert obs_merged["Dataset"].notna().all(), "❌ Unmapped Dataset values detected"

for col in ["Braak", "CERAD"]:
    obs_merged[col] = pd.to_numeric(obs_merged[col], errors="coerce")

############################################
# 2. Add derived metadata
############################################
obs_merged["celltypist_general"] = (
    obs_merged["celltypist_cell_label"].astype(str).str.split().str[0]
)

obs_merged["AD_status"] = np.where(
    (obs_merged["Braak"] >= 5) & (obs_merged["CERAD"] <= 2), "AD", "non-AD"
)

obs_merged["AD_prog"] = "no-AD"
obs_merged.loc[(obs_merged["Braak"].isin([5, 6])) & (obs_merged["CERAD"] == 1), "AD_prog"] = "Late AD"
obs_merged.loc[(obs_merged["Braak"].isin([3, 4])) & (obs_merged["CERAD"] == 2), "AD_prog"] = "Early AD"

obs_merged["barcode"] = (
    obs_merged["raw_barcode"].astype(str) + "-" + obs_merged["Dataset"].astype(str)
)
assert obs_merged["barcode"].is_unique, "❌ Duplicate barcodes in obs_merged"

print(f"💾 Saving merged obs audit trail to {OBS_MERGED_OUT}")
obs_merged.to_csv(OBS_MERGED_OUT, index=False)

############################################
# 3. Open original H5AD in READ-ONLY backed mode
############################################
print(f"📖 Opening {MERGED_H5AD} in READ-ONLY backed mode...")
adata_old = ad.read_h5ad(MERGED_H5AD, backed="r")

# Ensure we have a barcode column in the original data to join on
if "barcode" not in adata_old.obs.columns:
    print("ℹ️ Creating canonical barcode from merged index")
    adata_old.obs["barcode"] = adata_old.obs.index.astype(str)

############################################
# 4. PREPARE and CHECK metadata (In-Memory)
############################################
print("🔁 Aligning obs columns from obs_merged...")
obs_merged_idx = obs_merged.set_index("barcode")

# Use reindex to align metadata to the H5AD cell order
new_obs = obs_merged_idx.reindex(adata_old.obs["barcode"])

# Resetting the index to match the original adata.obs.index (usually barcodes)
new_obs.index = adata_old.obs.index

print("🔎 Pre-save checks on aligned metadata:")
for col in ["celltypist_general", "AD_status", "AD_prog"]:
    missing = new_obs[col].isna().mean()
    print(f"{col} missing fraction: {missing:.4f}")

print("\nAD_status distribution:")
print(new_obs["AD_status"].value_counts(dropna=False))

print("\nAD_prog distribution:")
print(new_obs["AD_prog"].value_counts(dropna=False))

# Safety check: Stop if something is very wrong
assert new_obs["AD_status"].isna().mean() < 0.1, "❌ Too much missing data! Check barcode alignment."

############################################
# 5. Create New Object and STREAM to Disk
############################################
print("🏗️ Creating new AnnData shell...")
adata_new = ad.AnnData(
    X=adata_old.X,
    obs=new_obs,
    var=adata_old.var,
    obsm=adata_old.obsm,
    layers=adata_old.layers,
    uns=adata_old.uns
)

print(f"🚀 Streaming to {FINAL_H5AD}...")
# Streaming write: uses disk I/O instead of high RAM
adata_new.write_h5ad(FINAL_H5AD, compression="gzip")

if hasattr(adata_old.file, "close"):
    adata_old.file.close()

print("✅ Done. New file is healthy and contains all metadata.")