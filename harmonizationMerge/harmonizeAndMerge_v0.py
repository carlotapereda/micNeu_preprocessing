import scanpy as sc
import re
import h5py
import re
import numpy as np
import pandas as pd
import anndata as ad

########################################
# Load files
########################################
files = {
    "fujita": "../celltypist/fujita_celltypist_GPU_counts_only.h5ad",
    "seaad":  "../celltypist/seaad_celltypist_GPU_counts_only.h5ad",
    "mit":    "../celltypist/mit_celltypist_GPU_counts_only.h5ad",
}

# Open in "read/write" backed mode
adata1 = ad.read_h5ad(files["mit"], backed="r")
adata2 = ad.read_h5ad(files["seaad"], backed="r")
adata3 = ad.read_h5ad(files["fujita"], backed="r")


########################################
# print columns
print( " ##################### Obs columns ##################### ")
########################################
print(adata1.obs.columns)
print(list(adata2.obs.columns))
print(adata3.obs.columns)



########################################
# age at death missing values? 
print( " ##################### Age at Death ##################### ")
########################################

# --- adata1 ---
#adata1.obs['age_death'] = pd.to_numeric(adata1.obs['age_death'], errors='coerce')
missing_count = adata1.obs['age_death'].isna().sum()
print(f"Number of missing values in age_death (adata1): {missing_count}")
print("Statistics for adata1:")
print(adata1.obs['age_death'].describe())
print("\n")

# --- adata2 ---
#adata2.obs['Age at Death'] = pd.to_numeric(adata2.obs['Age at Death'], errors='coerce')
missing_count = adata2.obs['Age at Death'].isna().sum()
print(f"Number of missing values in Age at Death (adata2): {missing_count}")
print("Statistics for adata2:")
print(adata2.obs['Age at Death'].describe())
print("\n")

# --- adata3 ---
#adata3.obs['age_death'] = pd.to_numeric(adata3.obs['age_death'], errors='coerce')
missing_count = adata3.obs['age_death'].isna().sum()
print(f"Number of missing values in age_death (adata3): {missing_count}")
print("Statistics for adata3:")
print(adata3.obs['age_death'].describe())
print("\n")

########################################
# Fix age
print( " ##################### Fix Age ##################### ")
########################################

# --- Helper function to clean age columns ---
def clean_age_column(series):
    series = series.astype(str)
    # Replace "90+" etc. with "90"
    series = series.str.replace(r'\+', '', regex=True)
    # Remove anything that isn’t a digit or decimal point (like "years", "yr", etc.)
    series = series.str.replace(r'[^\d\.]', '', regex=True)
    # Convert to numeric (invalid → NaN)
    series = pd.to_numeric(series, errors='coerce')
    return series

# --- Function to clean and summarize each AnnData object ---
def summarize_age(adata, colnames):
    # Find which of the given candidate columns actually exists
    for c in colnames:
        if c in adata.obs.columns:
            col = c
            break
    else:
        print("⚠️ No age column found.")
        return
    
    # Clean and convert to numeric
    adata.obs[col] = clean_age_column(adata.obs[col])
    
    # Drop missing for stats
    ages = adata.obs[col].dropna()
    missing_count = adata.obs[col].isna().sum()
    
    # Print results
    print(f"\n===== {col} ({adata.obs.shape[0]} observations) =====")
    print(f"Missing values: {missing_count}")
    print(f"Range: {ages.min():.1f}–{ages.max():.1f}")
    print(f"Mean: {ages.mean():.2f}")
    print(f"Median: {ages.median():.2f}")
    print("\nFull .describe() output:")
    print(ages.describe())

# --- Run for your three datasets ---
summarize_age(adata1, ["age_death", "Age at Death"])
summarize_age(adata2, ["age_death", "Age at Death"])
summarize_age(adata3, ["age_death", "Age at Death"])

#change name of age at death in SEAAD
adata2.obs["age_death"] = adata2.obs["Age at Death"]
missing_count = adata2.obs['age_death'].isna().sum()
print(f"Number of missing values in age_death (adata2): {missing_count}")
print("Statistics for adata2:")
print(adata2.obs['age_death'].describe())
print("\n")

########################################
# Investigate Sex 
print( " ##################### Investigate Sex ##################### ")
########################################

print(f"adata2 'Sex' unique: {adata2.obs['Sex'].unique()}")
print(f"adata1 sex unique: {adata1.obs['sex'].unique()}")
print(f"adata3 sex unique: {adata3.obs['msex'].unique()}")

# --- adata1 ---
# already has 'sex' column with 'male' / 'female'
adata1.obs['sex'] = (
    adata1.obs['sex']
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({'male': 'Male', 'female': 'Female'})
)
adata1.obs['sex'] = pd.Categorical(adata1.obs['sex'], categories=['Female', 'Male'], ordered=False)
print("adata1 sex unique:", adata1.obs['sex'].unique())


# --- adata2 ---
# has 'Sex' column with 'Male' / 'Female'
adata2.obs.rename(columns={'Sex': 'sex'}, inplace=True)
adata2.obs['sex'] = adata2.obs['sex'].astype(str).str.strip().str.capitalize()
adata2.obs['sex'] = pd.Categorical(adata2.obs['sex'], categories=['Female', 'Male'], ordered=False)
print("adata2 sex unique:", adata2.obs['sex'].unique())


# --- adata3 ---
# has numeric encoding: 1 = Male, 0 = Female
adata3.obs.rename(columns={'msex': 'sex'}, inplace=True) if 'msex' in adata3.obs.columns else None
adata3.obs['sex'] = (
    adata3.obs['sex']
    .replace({1: 'Male', 0: 'Female'})
    .astype(str)
    .str.strip()
)
adata3.obs['sex'] = pd.Categorical(adata3.obs['sex'], categories=['Female', 'Male'], ordered=False)
print("adata3 sex unique:", adata3.obs['sex'].unique())

########################################
# Investigate Sex 
print( " ##################### individualID ##################### ")
########################################


print(f"adata1 individualID unique: {adata1.obs['individualID'].unique()}")
print(f"adata2 sample_id unique: {len(adata2.obs['sample_id'].unique())}")
print(f"adata2 Donor ID unique: {adata2.obs['Donor ID'].unique()}")
print(f"adata3 individualID unique: {adata3.obs['individualID'].unique()}")

########################################
# Batch check
print( " ##################### batch check ##################### ")
########################################

print(f"adata1 batch unique: {adata1.obs['batch'].unique()}")
print(f"adata2 laod_name unique: {adata2.obs['load_name'].unique()}")
print(f"adata3 batch unique: {adata3.obs['batch'].unique()}")
# SEAAD: copy from load_name (best batch proxy)
adata2.obs["batch"] = adata2.obs["load_name"].astype(str)

########################################
print( " ##################### helper functions loading ##################### ")
#######################################

# -----------------------
# Helper functions
# -----------------------

def _first(df, candidates):
    """Return the first existing column among candidates."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(index=df.index, dtype="object")


# ---- APOE ----
_APOE_PAIR_RE = re.compile(r"[eεE]?\s*([234])\s*[/\-\|\s]?\s*[eεE]?\s*([234])")

def apoe_to_std(x):
    """Convert APOE genotype strings/numbers to 'E3/E4' style."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        s = str(int(x))
    else:
        s = str(x)
    s = s.strip().replace("ε", "e").replace("E", "e").replace("-", "/").replace("|", "/")
    m = _APOE_PAIR_RE.search(s)
    if m:
        a, b = m.groups()
        a, b = sorted([a, b])
        return f"E{a}/E{b}"
    digits = [ch for ch in s if ch in "234"]
    if len(digits) >= 2:
        a, b = sorted(digits[:2])
        return f"E{a}/E{b}"
    return np.nan

def apoe_e4_dosage(apoe_std):
    if pd.isna(apoe_std):
        return np.nan
    left, right = apoe_std.replace("E", "").split("/")
    return int(left == "4") + int(right == "4")


# ---- Braak ----
_ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
_INT_TO_ROMAN = {v: k for k, v in _ROMAN_TO_INT.items()}

def parse_braak(x):
    """Convert Braak staging values (roman, numeric, text) to 0..6 scale."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return int(np.clip(int(round(float(x))), 0, 6))
        except:
            return np.nan
    s = str(x).strip().upper().replace("BRAAK", "").strip()
    if s in _ROMAN_TO_INT:
        return _ROMAN_TO_INT[s]
    try:
        return int(np.clip(int(float(s)), 0, 6))
    except:
        for r, v in _ROMAN_TO_INT.items():
            if r in s:
                return v
    return np.nan

def braak_label_from_stage(stage):
    if pd.isna(stage):
        return np.nan
    stage = int(stage)
    return "Braak 0" if stage == 0 else f"Braak {_INT_TO_ROMAN.get(stage, str(stage))}"


# ---- CERAD ----
_CERAD_14_LABEL = {1: "Absent", 2: "Sparse", 3: "Moderate", 4: "Frequent"}
_CERAD_LABEL_14 = {v.upper(): k for k, v in _CERAD_14_LABEL.items()}

def to_cerad_1_4(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            v = int(round(float(x)))
            if v in (1, 2, 3, 4):
                return v
            if v in (0, 1, 2, 3):
                return v + 1
        except:
            return np.nan
    return _CERAD_LABEL_14.get(str(x).strip().upper(), np.nan)

def cerad_14_to_03(v):
    return np.nan if pd.isna(v) else int(v) - 1

def cerad_label_from_14(v):
    return np.nan if pd.isna(v) else _CERAD_14_LABEL.get(int(v), np.nan)


# ---- Race (SEAAD) ----
def derive_race_from_seaad(df: pd.DataFrame) -> pd.Series:
    mapping = {
        "Race (choice=White)": "White",
        "Race (choice=Black_ African American)": "Black or African American",
        "Race (choice=Asian)": "Asian",
        "Race (choice=American Indian_ Alaska Native)": "American Indian/Alaska Native",
        "Race (choice=Native Hawaiian or Pacific Islander)": "Native Hawaiian/Other Pacific Islander",
        "Race (choice=Unknown or unreported)": "Unknown or unreported",
        "Race (choice=Other)": "Other",
    }
    present = [c for c in mapping if c in df.columns]
    if not present:
        return pd.Series(index=df.index, dtype="object")

    def _one(row):
        picks = [mapping[c] for c in present if isinstance(row[c], str) and row[c].strip().lower() == "checked"]
        if not picks and "specify other race" in row and pd.notna(row["specify other race"]):
            return str(row["specify other race"])
        if not picks:
            return np.nan
        return picks[0] if len(picks) == 1 else " / ".join(sorted(set(picks)))

    return df.apply(_one, axis=1)


# -----------------------
# Harmonization
# -----------------------
CANONICAL_QC = [
    "n_genes_by_counts", "log1p_n_genes_by_counts",
    "total_counts", "log1p_total_counts", "pct_counts_in_top_20_genes",
    "total_counts_mt", "log1p_total_counts_mt", "pct_counts_mt",
    "total_counts_ribo", "log1p_total_counts_ribo", "pct_counts_ribo",
    "total_counts_hb", "log1p_total_counts_hb", "pct_counts_hb",
    "n_genes", "doublet_scores", "predicted_doublets", "doublet_label",
    "outlier", "mt_outlier"
]

def harmonize_obs(adata, default_dataset_name=None):
    df = adata.obs.copy()
    std = pd.DataFrame(index=df.index)

    # Dataset / IDs / study
    std["dataset"] = _first(df, ["dataset"])
    if std["dataset"].isna().all() and default_dataset_name:
        std["dataset"] = default_dataset_name
    std["projid"] = _first(df, ["projid"])
    std["study"] = _first(df, ["Study", "Primary Study Name"])
    std["individualID"] = _first(df, ["individualID", "Donor ID", "individualID_y", "individualID_x", "subject"])


    # ✅ FIXED AGE PARSING
    std["pmi"] = pd.to_numeric(_first(df, ["pmi", "PMI"]), errors="coerce")
    std["educ_years"] = pd.to_numeric(_first(df, ["educ", "Years of education"]), errors="coerce")
    std["hispanic_latino"] = _first(df, ["spanish", "Hispanic_Latino"])
    race_from_boxes = derive_race_from_seaad(df)
    std["race"] = race_from_boxes if race_from_boxes.notna().any() else _first(df, ["race"])
    std["species"] = _first(df, ["species", "Organism"])
    std["brain_region"] = _first(df, ["Brain Region"])

    # APOE
    apoe_src = _first(df, ["apoe_genotype", "APOE Genotype"])
    if not apoe_src.empty:
        std["apoe_genotype_std"] = apoe_src.apply(apoe_to_std).astype("category")
        std["apoe_e4_dosage"] = std["apoe_genotype_std"].apply(apoe_e4_dosage).astype("Int64")
        std["apoe_e4_carrier"] = std["apoe_e4_dosage"].fillna(0).astype(int).gt(0)

    # Braak
    braak_src = _first(df, ["braaksc", "Braak"])
    if not braak_src.empty:
        std["braak_stage"] = braak_src.apply(parse_braak).astype("Int64")
        std["braak_label"] = pd.Categorical(
            std["braak_stage"].map(braak_label_from_stage),
            categories=["Braak 0","Braak I","Braak II","Braak III","Braak IV","Braak V","Braak VI"],
            ordered=True
        )

    # CERAD
    cerad_src = _first(df, ["ceradsc", "CERAD score"])
    if not cerad_src.empty:
        cerad_14 = cerad_src.apply(to_cerad_1_4).astype("Int64")
        std["cerad_score_1_4"] = cerad_14
        std["cerad_score_0_3"] = cerad_14.apply(cerad_14_to_03).astype("Int64")
        std["cerad_label"] = pd.Categorical(
            cerad_14.apply(cerad_label_from_14),
            categories=["Absent","Sparse","Moderate","Frequent"],
            ordered=True
        )

    # Cognition & tests
    std["cogdx"] = _first(df, ["cogdx", "Cognitive Status"])
    std["mmse"] = pd.to_numeric(_first(df, ["cts_mmse30_lv", "Last MMSE Score"]), errors="coerce")
    std["moca"] = pd.to_numeric(_first(df, ["Last MOCA Score"]), errors="coerce")
    std["casi"] = pd.to_numeric(_first(df, ["Last CASI Score"]), errors="coerce")

    # Cell-type labels
    std["celltype_major"] = _first(df, ["major_cell_type", "Class", "subset"])
    std["celltype_label"] = _first(df, ["cell_type_high_resolution", "Subclass", "cell.type", "celltypist_cell_label"])
    std["celltype_supertype"] = _first(df, ["Supertype", "celltypist_simplified"])
    std["celltype_conf"] = pd.to_numeric(
        _first(df, ["celltypist_conf_score", "Class confidence", "Subclass confidence", "Supertype confidence"]),
        errors="coerce"
    )

    # QC & doublets
    for c in CANONICAL_QC:
        std[c] = _first(df, [c])
    if "Doublet score" in df.columns and std["doublet_scores"].isna().all():
        std["doublet_scores"] = pd.to_numeric(df["Doublet score"], errors="coerce")

    # Attach back
    for c in std.columns:
        adata.obs[c] = std[c]
    adata.obs["harmonized"] = True
    return adata

########################################
print( " ##################### run harmonization ##################### ")
#######################################

adata1 = harmonize_obs(adata1, default_dataset_name="MIT_ROSMAP")
adata2 = harmonize_obs(adata2, default_dataset_name="SEAAD")
adata3 = harmonize_obs(adata3, default_dataset_name="FUJITA")

########################################
print( " ##################### add chemistry ##################### ")
#######################################
adata2.obs["chemistry"] = adata2.obs["method"]
adata3.obs['chemistry'] = "10x 3' v3" #taken from the paper methods section

print("adata1", adata1.obs['chemistry'].unique())
print("adata2", adata2.obs['chemistry'].unique())
print("adata3", adata3.obs['chemistry'].unique())

########################################
print( " ##################### merge objects ##################### ")
#######################################

# 2. Intersection of genes
common_genes = (
    set(adata1.var_names)
    & set(adata2.var_names)
    & set(adata3.var_names)
)
common_genes = pd.Index(sorted(list(common_genes)))
print(f"Number of common genes: {len(common_genes)}")

adata1 = adata1[:, common_genes].copy()
adata2 = adata2[:, common_genes].copy()
adata3 = adata3[:, common_genes].copy()

# 3. Concatenate
merged = ad.concat(
    {"MIT_ROSMAP": adata1, "SEAAD": adata2, "FUJITA": adata3},
    label="dataset",
    join="inner"
)

print("merged shape:", merged.shape)
print("datasets:", merged.obs["dataset"].value_counts().to_dict())
print("column names", merged.obs.columns)



########################################
print( " ##################### ADD AD STATUS ##################### ")
#######################################
# ADD AD STATUS BASED ON BRAAK AND CERAD SCORING 
# Create AD_status directly in merged.obs
merged.obs.loc[
    (merged.obs['braak_stage'] >= 5) & (merged.obs['cerad_score_0_3'] <= 2),
    'AD_status'
] = 'AD'

# Optional: label everything else as non-AD
merged.obs['AD_status'] = merged.obs['AD_status'].fillna('non-AD')

# Check results
print(merged.obs['AD_status'].value_counts())

########################################
print( " ##################### SAVE ##################### ")
#######################################
merged.obs["projid"] = pd.to_numeric(merged.obs["projid"], errors="coerce").astype("Int64")

for col in merged.obs.columns:
    if merged.obs[col].dtype == "object" or pd.api.types.is_categorical_dtype(merged.obs[col].dtype):
        merged.obs[col] = merged.obs[col].astype(str)

# Now write safely
merged.write_h5ad("merged_allcells.h5ad", , compression="gzip")
