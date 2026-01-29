import anndata as ad
import pandas as pd
import numpy as np

############################################
# Source
############################################

SRC = "../../celltypist/fujita_celltypist_GPU_counts_only.h5ad"
ATLAS_CSV = "cell-annotation.full-atlas.csv"

print("📖 Reading FUJITA AnnData (backed)...")
adata = ad.read_h5ad(SRC, backed="r")

############################################
# Copy obs (DO NOT edit adata.obs)
############################################

obs = adata.obs.copy()

############################################
# [FUJITA] 1. Remove any columns that are not these
############################################

KEEP_COLS = [
    # QC / counts
    "n_genes_by_counts","log1p_n_genes_by_counts","total_counts","log1p_total_counts",
    "pct_counts_in_top_20_genes","total_counts_mt","log1p_total_counts_mt","pct_counts_mt",
    "total_counts_ribo","log1p_total_counts_ribo","pct_counts_ribo",
    "total_counts_hb","log1p_total_counts_hb","pct_counts_hb",
    "n_genes","doublet_scores","predicted_doublets","doublet_label","outlier","mt_outlier",

    # Metadata
    "individualID","projid","msex","educ","race","apoe_genotype","age_death",
    "cts_mmse30_lv","pmi","braaksc","ceradsc","cogdx",
    "celltypist_cell_label","Study","celltypist_conf_score",
]

obs = obs[[c for c in KEEP_COLS if c in obs.columns]]

############################################
# [FUJITA] 2. Create new column called APOE_Genotype
# keep apoe_genotype column
############################################

APOE_MAP = {
    33: "3/3",
    34: "3/4",
    44: "4/4",
}

obs["APOE_Genotype"] = obs["apoe_genotype"].map(APOE_MAP)

############################################
# [FUJITA] 3. Create new column called Cognitive_Status
# keep cogdx column
############################################

def map_cognitive_status(val):
    if val in [1, 2, 3]:
        return "No dementia"
    if val in [4, 5, 6]:
        return "Dementia"
    return np.nan

obs["Cognitive_Status"] = obs["cogdx"].apply(map_cognitive_status)

############################################
# [FUJITA] 4. Create new column called Sex
# keep msex column
############################################

SEX_MAP = {
    1: "Male",
    0: "Female",
}

obs["Sex"] = obs["msex"].map(SEX_MAP)

############################################
# [FUJITA] 5. Create new column called Age_Death
# keep age_death column
############################################

def normalize_age(val):
    if isinstance(val, str) and val.strip() == "90+":
        return 90
    return val

obs["Age_Death"] = obs["age_death"].apply(normalize_age)

############################################
# [FUJITA] 6. Create new column called Race
# keep race column
############################################

RACE_MAP = {
    1: "White",
    2: "Black and African American",
    3: "American Indian or Alaska Native",
    4: "Native Hawaiian or Other Pacific Islander",
    5: "Asian",
    6: "Other",
    7: "Unknown",
}

obs["Race"] = obs["race"].map(RACE_MAP)

############################################
# [FUJITA] 7. Create Dataset column
############################################

obs["Dataset"] = "FUJITA"

############################################
# [FUJITA] 8. Create barcode columns
############################################
# raw_barcode = original cell barcode (used for atlas matching)
# barcode = Dataset + index (used for uniqueness downstream)

obs.index = obs.index.astype(str)

obs["raw_barcode"] = obs.index
obs["barcode"] = "FUJITA_" + obs["raw_barcode"]


############################################
# [FUJITA] 9. Create column called chemistry
############################################

obs["chemistry"] = "10x 3’ v3"

############################################
# [FUJITA] 10. Create column called batch
############################################
# load cell-annotation.full-atlas.csv
# match barcodes from atlas csv to the raw Fujita barcodes
# add batch values from atlas to Fujita obs

print("📂 Loading cell annotation atlas...")
atlas = pd.read_csv(ATLAS_CSV)

# Rename atlas barcode column
atlas = atlas.rename(columns={"cell": "raw_barcode"})

# Enforce string dtype
atlas["raw_barcode"] = atlas["raw_barcode"].astype(str)
obs["raw_barcode"] = obs["raw_barcode"].astype(str)

# Keep only required columns
atlas = atlas[["raw_barcode", "batch"]].drop_duplicates()

# Merge using raw barcodes
obs = obs.merge(
    atlas,
    on="raw_barcode",
    how="left",
    validate="one_to_one",
)

print("Batch missing fraction:", obs["batch"].isna().mean())

############################################
# [FUJITA] 11. Copy columns under new names (keep both)
############################################

COPY_MAP = {
    "individualID": "Donor_ID",
    "educ": "Years_Education",
    "cts_mmse30_lv": "Last_MMSE",
    "braaksc": "Braak",
    "ceradsc": "CERAD",
}

for src, dst in COPY_MAP.items():
    if src in obs.columns:
        obs[dst] = obs[src]

############################################
# [FUJITA] 12. Save
############################################
# Drop CSV artifacts
obs = obs.drop(columns=["Unnamed: 0", "barcode.1"], errors="ignore")

print("💾 Saving harmonized FUJITA obs...")
obs.to_csv("FUJITA_harmonized_obs.csv", index=False)

############################################
# Print obs names + summaries
############################################

summary_rows = []

for col in obs.columns:
    if pd.api.types.is_numeric_dtype(obs[col]):
        desc = obs[col].describe()
        for stat, val in desc.items():
            summary_rows.append(
                {"column": col, "stat": stat, "value": val}
            )
    else:
        vc = obs[col].value_counts(dropna=False)
        for val, cnt in vc.items():
            summary_rows.append(
                {"column": col, "stat": val, "value": cnt}
            )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("FUJITA_obs_column_summary.csv", index=False)

print("✅ FUJITA harmonization complete.")
print(f"Columns ({len(obs.columns)}):")
print(list(obs.columns))
