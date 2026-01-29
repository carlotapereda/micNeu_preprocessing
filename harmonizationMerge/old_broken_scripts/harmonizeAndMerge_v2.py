# ==========
# SETUP
# ==========
import re
import numpy as np
import pandas as pd
import anndata as ad
import h5py

files = {
    "MIT_ROSMAP": "../celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEAAD":      "../celltypist/seaad_celltypist_GPU_counts_only.h5ad",
    "FUJITA":     "../celltypist/fujita_celltypist_GPU_counts_only.h5ad",
}

# ==========
# HELPERS (your originals, lightly modularized)
# ==========

def _first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(index=df.index, dtype="object")

# ---- Age helpers ----
def clean_age_column(series):
    s = series.astype(str)
    s = s.str.replace(r'\+', '', regex=True)
    s = s.str.replace(r'[^\d\.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

def unify_age_death_inplace(df):
    # Create a single numeric age_death column from multiple possibilities
    col = None
    for c in ["age_death", "Age at Death"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        df["age_death"] = pd.Series(np.nan, index=df.index, dtype="float")
    else:
        df["age_death"] = clean_age_column(df[col])

# ---- Sex helpers ----
def standardize_sex_inplace(df):
    s = None
    # pick first present column among common variants
    for c in ["sex", "Sex", "msex"]:
        if c in df.columns:
            s = df[c].copy()
            break
    if s is None:
        df["sex"] = pd.Series(pd.Categorical([], categories=["Female", "Male"]), index=df.index)
        return
    s = s.replace({1: "Male", 0: "Female", "1": "Male", "0": "Female"})
    s = s.astype(str).str.strip().str.lower()
    s = s.replace({"f": "female", "m": "male"})
    s = s.map({"female": "Female", "male": "Male"})
    df["sex"] = pd.Categorical(s, categories=["Female", "Male"], ordered=False)

# ---- APOE ----
_APOE_PAIR_RE = re.compile(r"[eεE]?\s*([234])\s*[/\-\|\s]?\s*[eεE]?\s*([234])")
def apoe_to_std(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        s = str(int(x))
    else:
        s = str(x)
    s = s.strip().replace("ε", "e").replace("E", "e").replace("-", "/").replace("|", "/")
    m = _APOE_PAIR_RE.search(s)
    if m:
        a, b = sorted(m.groups())
        return f"E{a}/E{b}"
    digits = [ch for ch in s if ch in "234"]
    if len(digits) >= 2:
        a, b = sorted(digits[:2])
        return f"E{a}/E{b}"
    return np.nan

def apoe_e4_dosage(apoe_std):
    if pd.isna(apoe_std): return np.nan
    left, right = apoe_std.replace("E", "").split("/")
    return int(left == "4") + int(right == "4")

# ---- Braak ----
_ROMAN_TO_INT = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6}
_INT_TO_ROMAN = {v:k for k,v in _ROMAN_TO_INT.items()}
def parse_braak(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try: return int(np.clip(int(round(float(x))), 0, 6))
        except: return np.nan
    s = str(x).strip().upper().replace("BRAAK", "").strip()
    if s in _ROMAN_TO_INT: return _ROMAN_TO_INT[s]
    try: return int(np.clip(int(float(s)), 0, 6))
    except:
        for r,v in _ROMAN_TO_INT.items():
            if r in s: return v
    return np.nan

def braak_label_from_stage(stage):
    if pd.isna(stage): return np.nan
    stage = int(stage)
    return "Braak 0" if stage == 0 else f"Braak {_INT_TO_ROMAN.get(stage, str(stage))}"

# ---- CERAD ----
_CERAD_14_LABEL = {1:"Absent",2:"Sparse",3:"Moderate",4:"Frequent"}
_CERAD_LABEL_14 = {v.upper():k for k,v in _CERAD_14_LABEL.items()}

def to_cerad_1_4(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            v = int(round(float(x)))
            if v in (1,2,3,4): return v
            if v in (0,1,2,3): return v + 1
        except: return np.nan
    return _CERAD_LABEL_14.get(str(x).strip().upper(), np.nan)

def cerad_14_to_03(v):
    return np.nan if pd.isna(v) else int(v) - 1

def cerad_label_from_14(v):
    return np.nan if pd.isna(v) else _CERAD_14_LABEL.get(int(v), np.nan)

# ---- Race (SEAAD checkbox columns) ----
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

# ---- QC columns to copy if present ----
CANONICAL_QC = [
    "n_genes_by_counts", "log1p_n_genes_by_counts",
    "total_counts", "log1p_total_counts", "pct_counts_in_top_20_genes",
    "total_counts_mt", "log1p_total_counts_mt", "pct_counts_mt",
    "total_counts_ribo", "log1p_total_counts_ribo", "pct_counts_ribo",
    "total_counts_hb", "log1p_total_counts_hb", "pct_counts_hb",
    "n_genes", "doublet_scores", "predicted_doublets", "doublet_label",
    "outlier", "mt_outlier"
]

def harmonize_obs_inplace(adata):
    """Write harmonized columns into adata.obs (works fine when adata is in backed mode; changes live in memory)."""
    df = adata.obs.copy()
    std = pd.DataFrame(index=df.index)

    # Dataset / IDs / study
    std["dataset"]      = _first(df, ["dataset"])  # will be present after concat(label="dataset")
    std["projid"]       = _first(df, ["projid"])
    std["study"]        = _first(df, ["Study", "Primary Study Name"])
    std["individualID"] = _first(df, ["individualID", "Donor ID", "individualID_y", "individualID_x", "subject"])

    # Age, PMI, education, race, species/region
    unify_age_death_inplace(df)
    std["age_death"]     = df["age_death"]
    std["pmi"]           = pd.to_numeric(_first(df, ["pmi", "PMI"]), errors="coerce")
    std["educ_years"]    = pd.to_numeric(_first(df, ["educ", "Years of education"]), errors="coerce")
    std["hispanic_latino"] = _first(df, ["spanish", "Hispanic_Latino"])
    race_from_boxes      = derive_race_from_seaad(df)
    std["race"]          = race_from_boxes if race_from_boxes.notna().any() else _first(df, ["race"])
    std["species"]       = _first(df, ["species", "Organism"])
    std["brain_region"]  = _first(df, ["Brain Region"])

    # Sex
    tmp = df.copy()
    standardize_sex_inplace(tmp)
    std["sex"] = tmp["sex"]

    # APOE
    apoe_src = _first(df, ["apoe_genotype", "APOE Genotype"])
    if not apoe_src.empty:
        std["apoe_genotype_std"] = apoe_src.apply(apoe_to_std).astype("category")
        std["apoe_e4_dosage"]    = std["apoe_genotype_std"].apply(apoe_e4_dosage).astype("Int64")
        std["apoe_e4_carrier"]   = std["apoe_e4_dosage"].fillna(0).astype(int).gt(0)

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
        std["cerad_label"]     = pd.Categorical(
            cerad_14.apply(cerad_label_from_14),
            categories=["Absent","Sparse","Moderate","Frequent"],
            ordered=True
        )

    # Cognition/tests
    std["cogdx"] = _first(df, ["cogdx", "Cognitive Status"])
    std["mmse"]  = pd.to_numeric(_first(df, ["cts_mmse30_lv", "Last MMSE Score"]), errors="coerce")
    std["moca"]  = pd.to_numeric(_first(df, ["Last MOCA Score"]), errors="coerce")
    std["casi"]  = pd.to_numeric(_first(df, ["Last CASI Score"]), errors="coerce")

    # Cell-type labels
    std["celltype_major"]    = _first(df, ["major_cell_type", "Class", "subset"])
    std["celltype_label"]    = _first(df, ["cell_type_high_resolution", "Subclass", "cell.type", "celltypist_cell_label"])
    std["celltype_supertype"]= _first(df, ["Supertype", "celltypist_simplified"])
    std["celltype_conf"]     = pd.to_numeric(
        _first(df, ["celltypist_conf_score", "Class confidence", "Subclass confidence", "Supertype confidence"]),
        errors="coerce"
    )

    # QC
    for c in CANONICAL_QC:
        std[c] = _first(df, [c])

    # Chemistry: SEAAD from 'method', FUJITA fixed string, otherwise Unknown
    chem = pd.Series("Unknown", index=df.index, dtype="object")
    if "method" in df.columns:
        chem = chem.mask(df.get("dataset","") == "SEAAD", df["method"].astype(str))
    chem = chem.mask(df.get("dataset","") == "FUJITA", "10x 3' v3")
    std["chemistry"] = chem

    # Attach back to the backed AnnData (in-memory view of obs)
    for c in std.columns:
        adata.obs[c] = std[c]
    adata.obs["harmonized"] = True

def compute_ad_status_inplace(df):
    # AD if Braak >=5 and CERAD (0–3 scale) <=2, else non-AD
    braak = pd.to_numeric(df.get("braak_stage"), errors="coerce")
    cerad = pd.to_numeric(df.get("cerad_score_0_3"), errors="coerce")
    status = pd.Series("non-AD", index=df.index, dtype="object")
    status.loc[(braak >= 5) & (cerad <= 2)] = "AD"
    df["AD_status"] = status

def persist_obs_h5ad(h5ad_path, obs_df):
    # Write only .obs back to the H5AD, leaving .X on disk untouched
    # (works even if the AnnData itself was opened in backed mode elsewhere)
    with h5py.File(h5ad_path, "r+") as f:
        # Convert plain object columns to strings for HDF5 friendliness
        for col in obs_df.columns:
            if pd.api.types.is_object_dtype(obs_df[col]):
                obs_df[col] = obs_df[col].astype(str)
        ad.io.write_elem(f, "obs", obs_df)

# ==========
# 1) MERGE DIRECTLY ON DISK (inner-join on genes)
# ==========
ad.experimental.concat_on_disk(
    files,                        # mapping {dataset_key: path}
    out_file="merged_allcells.h5ad",
    join="inner",                 # intersection of genes
    label="dataset",              # writes dataset column using keys
    index_unique="_"              # avoid duplicate cell IDs across inputs
)

# ==========
# 2) ENRICH .obs WITHOUT LOADING THE MATRIX
# ==========
merged_b = ad.read_h5ad("merged_allcells.h5ad", backed="r")
harmonize_obs_inplace(merged_b)          # compute harmonized columns in memory
obs = merged_b.obs.copy()
# extra cleanups
obs["projid"] = pd.to_numeric(obs.get("projid"), errors="coerce").astype("Int64")
compute_ad_status_inplace(obs)           # add AD_status

# ==========
# 3) PERSIST ONLY THE OBS TABLE BACK INTO THE FILE
# ==========
persist_obs_h5ad("merged_allcells.h5ad", obs)

print("Done. Wrote merged_allcells.h5ad with harmonized metadata and AD_status (X never loaded).")
