import anndata as ad
import pandas as pd
import numpy as np

############################################
# Source
############################################
SRC = "../../celltypist/seaad_celltypist_GPU_counts_only.h5ad"

print("📖 Reading SEA-AD AnnData...")
# Read as backed if the file is massive, but we need to load .obs into memory
adata = ad.read_h5ad(SRC, backed="r")

############################################
# Copy obs (DO NOT edit adata.obs)
############################################
obs = adata.obs.copy()

############################################
# [SEA-AD] Remove any columns that are not these
############################################
KEEP_COLS = [
    "Donor ID", "sample_id", "Sex", "Years of education", 
    "Race (choice=Asian)", "Race (choice=Native Hawaiian or Pacific Islander)",
    "Race (choice=Other)", "Race (choice=Unknown or unreported)",
    "Race (choice=White)", "APOE Genotype", "Age at Death",
    "Last MMSE Score", "PMI", "Braak", "CERAD score",
    "Cognitive Status", "load_name", "Primary Study Name",
    "Secondary Study Name", "celltypist_cell_label", "specify other race",
    "n_genes_by_counts", "log1p_n_genes_by_counts", "total_counts",
    "log1p_total_counts", "pct_counts_in_top_20_genes", "total_counts_mt",
    "log1p_total_counts_mt", "pct_counts_mt", "total_counts_ribo",
    "log1p_total_counts_ribo", "pct_counts_ribo", "total_counts_hb",
    "log1p_total_counts_hb", "pct_counts_hb", "n_genes",
    "doublet_scores", "predicted_doublets", "outlier", "mt_outlier",
    "celltypist_conf_score", "method"
]

obs = obs[[c for c in KEEP_COLS if c in obs.columns]]

############################################
# [SEA-AD] Change SEA-AD column names to match ROSMAP naming
############################################
RENAME_MAP = {
    "Donor ID": "individualID",
    "sample_id": "projid",
    "Sex": "msex",
    "Years of education": "educ",
    "APOE Genotype": "apoe_genotype",
    "Age at Death": "age_death",
    "Last MMSE Score": "cts_mmse30_lv",
    "PMI": "pmi",
    "Braak": "braaksc",
    "CERAD score": "ceradsc",
    "Cognitive Status": "cogdx",
    "load_name": "batch", 
    "method": "chemistry",
}

obs = obs.rename(columns=RENAME_MAP)

# --- CRITICAL FIX: Ensure MMSE is numeric BEFORE copying ---
# This ensures that both cts_mmse30_lv and Last_MMSE are identical floats
if "cts_mmse30_lv" in obs.columns:
    obs["cts_mmse30_lv"] = pd.to_numeric(obs["cts_mmse30_lv"], errors='coerce')

############################################
# [SEA-AD] Create new column called “race”
############################################
def infer_race(row):
    if row.get("Race (choice=Asian)") == "Checked":
        return "Asian"
    if row.get("Race (choice=Native Hawaiian or Pacific Islander)") == "Checked":
        return "Native Hawaiian or Pacific Islander"
    if row.get("Race (choice=Other)") == "Checked":
        return "Other"
    if row.get("Race (choice=Unknown or unreported)") == "Checked":
        return "Unknown"
    if row.get("Race (choice=White)") == "Checked":
        return "White"
    if row.get("specify other race") == "Mixed":
        return "Mixed"
    return np.nan

obs["race"] = obs.apply(infer_race, axis=1)

# Remove old Race columns
obs = obs.drop(
    columns=[c for c in obs.columns if c.startswith("Race") or c == "specify other race"],
    errors="ignore",
)

############################################
# [SEA-AD] Create new column called “Study”
############################################
obs["Study"] = (
    obs["Primary Study Name"].astype(str)
    + " "
    + obs["Secondary Study Name"].astype(str)
)

obs = obs.drop(columns=["Primary Study Name", "Secondary Study Name"], errors="ignore")

############################################
# [SEA-AD] Create new column called Braak (Numeric)
############################################
BRAAK_MAP = {
    "Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3,
    "Braak IV": 4, "Braak V": 5, "Braak VI": 6,
}
# Map to new column while keeping the categorical 'braaksc'
obs["Braak"] = obs["braaksc"].map(BRAAK_MAP)

############################################
# [SEA-AD] Create column called CERAD (Numeric)
############################################
CERAD_MAP = {
    "Absent": 4, "Sparse": 3, "Moderate": 2, "Frequent": 1,
}
# Map to new column while keeping the categorical 'ceradsc'
obs["CERAD"] = obs["ceradsc"].map(CERAD_MAP)

############################################
# [SEA-AD] Copy columns under new names (keep both)
############################################
COPY_MAP = {
    "individualID": "Donor_ID",
    "msex": "Sex",
    "educ": "Years_Education",
    "race": "Race",
    "apoe_genotype": "APOE_Genotype",
    "age_death": "Age_Death",
    "cts_mmse30_lv": "Last_MMSE", # Copies the now-numeric value
    "cogdx": "Cognitive_Status",
}

for src, dst in COPY_MAP.items():
    if src in obs.columns:
        obs[dst] = obs[src]

############################################
# [SEA-AD] Create Dataset column
############################################
obs["Dataset"] = "SEA-AD"

############################################
# [SEA-AD] Create barcode column
############################################
obs.index = obs.index.astype(str)
obs["raw_barcode"] = obs.index
obs["barcode"] = obs["Dataset"] + "_" + obs.index

############################################
# Save
############################################
print("💾 Saving harmonized obs...")
obs.to_csv("SEAAD_harmonized_obs.csv", index=False)

############################################
# Print summary to verify MMSE is numeric and non-zero
############################################
summary_df = obs[["cts_mmse30_lv", "Last_MMSE"]].describe()
print("\n--- Verification of MMSE Harmonization ---")
print(summary_df)

print("\n✅ Done.")