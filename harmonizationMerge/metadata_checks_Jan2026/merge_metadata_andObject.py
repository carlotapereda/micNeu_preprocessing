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

############################################
# Normalize Dataset labels to concat_on_disk keys
############################################

DATASET_MAP = {
    "SEA-AD": "SEAAD",
    "MIT-ROSMAP": "MIT_ROSMAP",
    "FUJITA": "FUJITA",
}

obs_merged["Dataset"] = obs_merged["Dataset"].map(DATASET_MAP)
assert obs_merged["Dataset"].notna().all(), "❌ Unmapped Dataset values detected"

############################################
# Coerce Braak / CERAD to numeric
############################################

for col in ["Braak", "CERAD"]:
    obs_merged[col] = pd.to_numeric(obs_merged[col], errors="coerce")

############################################
# 2. Add derived metadata
############################################

# celltypist_general = first word
obs_merged["celltypist_general"] = (
    obs_merged["celltypist_cell_label"]
    .astype(str)
    .str.split()
    .str[0]
)

# AD_status
obs_merged["AD_status"] = np.where(
    (obs_merged["Braak"] >= 5) & (obs_merged["CERAD"] <= 2),
    "AD",
    "non-AD",
)

# AD_prog
obs_merged["AD_prog"] = "no-AD"
obs_merged.loc[
    (obs_merged["Braak"].isin([5, 6])) & (obs_merged["CERAD"] == 1),
    "AD_prog",
] = "Late AD"
obs_merged.loc[
    (obs_merged["Braak"].isin([3, 4])) & (obs_merged["CERAD"] == 2),
    "AD_prog",
] = "Early AD"

############################################
# Canonical barcode: <raw_barcode>-<DATASET>
############################################

obs_merged["barcode"] = (
    obs_merged["raw_barcode"].astype(str)
    + "-"
    + obs_merged["Dataset"].astype(str)
)

assert obs_merged["barcode"].is_unique, "❌ Duplicate barcodes in obs_merged"

############################################
# Save merged obs (audit trail)
############################################

print("💾 Saving merged obs with derived metadata...")
obs_merged.to_csv(OBS_MERGED_OUT, index=False)
print(f"Saved to: {OBS_MERGED_OUT}")

############################################
# 3. Open merged h5ad in READ/WRITE backed mode
############################################

print("📖 Opening merged h5ad in backed r+ mode...")
adata = ad.read_h5ad(MERGED_H5AD, backed="r+")

############################################
# Ensure canonical barcode in adata.obs
############################################

if "barcode" not in adata.obs.columns:
    print("ℹ️ Creating canonical barcode from merged index")
    adata.obs["barcode"] = adata.obs.index.astype(str)

assert adata.obs["barcode"].is_unique, "❌ Non-unique barcodes in merged h5ad"

############################################
# Verify barcode alignment
############################################

barcode_overlap = adata.obs["barcode"].isin(obs_merged["barcode"]).mean()
print("Barcode overlap fraction:", barcode_overlap)
assert barcode_overlap > 0.99, "❌ Barcode mismatch between obs_merged and h5ad"

############################################
# 4. OVERWRITE adata.obs columns from obs_merged
############################################

print("🔁 Replacing / adding obs columns from obs_merged...")

obs_merged_idx = obs_merged.set_index("barcode")

# Columns we want to propagate (everything except barcode)
cols_to_write = [c for c in obs_merged_idx.columns if c != "barcode"]

# Drop overlapping columns first (clean replacement)
overlap_cols = [c for c in cols_to_write if c in adata.obs.columns]
if overlap_cols:
    print(f"🧹 Dropping {len(overlap_cols)} overlapping columns from adata.obs")
    adata.obs.drop(columns=overlap_cols, inplace=True)

# Write columns (aligned by barcode)
for col in cols_to_write:
    adata.obs[col] = obs_merged_idx.loc[adata.obs["barcode"], col].values

############################################
# 5. Post-merge checks
############################################

print("🔎 Post-merge checks:")

for col in ["celltypist_general", "AD_status", "AD_prog"]:
    print(f"{col} missing fraction:",
          adata.obs[col].isna().mean())

print("AD_status distribution:")
print(adata.obs["AD_status"].value_counts(dropna=False))

print("AD_prog distribution:")
print(adata.obs["AD_prog"].value_counts(dropna=False))

############################################
# 6. Flush changes and close file (NO X rewrite)
############################################

print("💾 Closing backed h5ad file (obs already written)...")

if hasattr(adata.file, "close"):
    adata.file.close()


print("✅ Done. adata.obs now reflects obs_merged as source of truth.")
