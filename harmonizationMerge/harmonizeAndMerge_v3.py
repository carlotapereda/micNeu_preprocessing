#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from datetime import datetime
import re
import h5py
import numpy as np
import pandas as pd
import anndata as ad

# ==========
# Logging helpers
# ==========
START = time.perf_counter()

def _now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def memline():
        rss = _PROC.memory_info().rss / (1024**3)
        return f" | RSS {rss:.2f} GB"
except Exception:
    def memline():
        return ""

def log(msg):
    elapsed = time.perf_counter() - START
    print(f"[{_now_ts()}] (+{elapsed:7.2f}s) {msg}{memline()}", flush=True)

# ==========
# File map
# ==========
files = {
    "MIT_ROSMAP": "../celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEAAD":      "../celltypist/seaad_celltypist_GPU_counts_only.h5ad",
    "FUJITA":     "../celltypist/fujita_celltypist_GPU_counts_only.h5ad",
}

# ==========
# Small utilities
# ==========
def clean_age_column(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(index=pd.RangeIndex(0), dtype="float64")
    s = series.astype(str)
    s = s.str.replace(r'\+', '', regex=True)
    s = s.str.replace(r'[^\d\.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

def _first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(index=df.index, dtype="object")

# ---- Sex standardization (bug-fixed) ----
def standardize_sex_inplace(df: pd.DataFrame) -> None:
    src = None
    for c in ["sex", "Sex", "msex"]:
        if c in df.columns:
            src = c
            break

    if src is None:
        # Create full-length NA categorical
        df["sex"] = pd.Series(
            pd.Categorical.from_codes(
                np.full(len(df), -1, dtype=int),
                categories=["Female", "Male"]
            ),
            index=df.index
        )
        log("standardize_sex_inplace: no source column found → created NA categorical 'sex'")
        return

    s = df[src]
    s = s.replace({1: "Male", 0: "Female", "1": "Male", "0": "Female"})
    s = s.astype(str).str.strip().str.lower()
    s = s.replace({"f": "female", "m": "male"})
    s = s.map({"female": "Female", "male": "Male"})  # others -> NaN
    df["sex"] = pd.Categorical(s, categories=["Female", "Male"], ordered=False)

    # Quick counts
    vc = pd.Series(df["sex"]).value_counts(dropna=False)
    log(f"standardize_sex_inplace: 'sex' value counts: {vc.to_dict()}")

# ---- APOE ----
_APOE_PAIR_RE = re.compile(r"[eεE]?\s*([234])\s*[/\-\|\s]?\s*[eεE]?\s*([234])")
def apoe_to_std(x):
    if pd.isna(x): return np.nan
    s = str(int(x)) if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
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
        for r, v in _ROMAN_TO_INT.items():
            if r in s: return v
    return np.nan

def braak_label_from_stage(stage):
    if pd.isna(stage): return np.nan
    stage = int(stage)
    return "Braak 0" if stage == 0 else f"Braak {_INT_TO_ROMAN.get(stage, str(stage))}"

# ---- CERAD ----
_CERAD_14_LABEL = {1:"Absent", 2:"Sparse", 3:"Moderate", 4:"Frequent"}
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

def cerad_14_to_03(v): return np.nan if pd.isna(v) else int(v) - 1
def cerad_label_from_14(v): return np.nan if pd.isna(v) else _CERAD_14_LABEL.get(int(v), np.nan)

# ---- SEAAD race boxes → run only on SEAAD rows
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
        log("derive_race_from_seaad: no SEAAD race checkbox columns present")
        return pd.Series(index=df.index, dtype="object")

    out = pd.Series(np.nan, index=df.index, dtype="object")
    mask = df["dataset"].eq("SEAAD") if "dataset" in df.columns else pd.Series(False, index=df.index)
    if not mask.any():
        log("derive_race_from_seaad: dataset!=SEAAD for all rows → skipping")
        return out

    log(f"derive_race_from_seaad: computing for {mask.sum():,} SEAAD rows across {len(present)} checkbox cols")
    # Row-wise (only SEAAD rows) to preserve multi-pick join behavior
    def _one(row):
        picks = [mapping[c] for c in present
                 if isinstance(row[c], str) and row[c].strip().lower() == "checked"]
        if not picks and "specify other race" in row and pd.notna(row["specify other race"]):
            return str(row["specify other race"])
        if not picks: return np.nan
        return picks[0] if len(picks) == 1 else " / ".join(sorted(set(picks)))
    out.loc[mask] = df.loc[mask, present + (["specify other race"] if "specify other race" in df.columns else [])].apply(_one, axis=1)
    vc = out.loc[mask].value_counts(dropna=False).head(10).to_dict()
    log(f"derive_race_from_seaad: top labels (SEAAD) → {vc}")
    return out

# Canonical QC columns
CANONICAL_QC = [
    "n_genes_by_counts","log1p_n_genes_by_counts",
    "total_counts","log1p_total_counts","pct_counts_in_top_20_genes",
    "total_counts_mt","log1p_total_counts_mt","pct_counts_mt",
    "total_counts_ribo","log1p_total_counts_ribo","pct_counts_ribo",
    "total_counts_hb","log1p_total_counts_hb","pct_counts_hb",
    "n_genes","doublet_scores","predicted_doublets","doublet_label",
    "outlier","mt_outlier"
]

def harmonize_obs_inplace(adata):
    log("harmonize_obs_inplace: begin")
    df = adata.obs.copy()
    std = pd.DataFrame(index=df.index)

    # Dataset / IDs / study
    std["dataset"]      = _first(df, ["dataset"])
    std["projid"]       = _first(df, ["projid"])
    std["study"]        = _first(df, ["Study", "Primary Study Name"])
    std["individualID"] = _first(df, ["individualID", "Donor ID", "individualID_y", "individualID_x", "subject"])

    # ------- Age / PMI / education / race / region -------
    col = next((c for c in ["age_death", "Age at Death"] if c in df.columns), None)
    if col is not None:
        std["age_death"] = clean_age_column(df[col])
        log(f"harmonize: age_death non-null: {std['age_death'].notna().sum():,}")
    std["pmi"]           = pd.to_numeric(_first(df, ["pmi", "PMI"]), errors="coerce")
    std["educ_years"]    = pd.to_numeric(_first(df, ["educ", "Years of education"]), errors="coerce")
    std["hispanic_latino"]= _first(df, ["spanish", "Hispanic_Latino"])

    race_from_boxes = derive_race_from_seaad(df)
    std["race"] = race_from_boxes if race_from_boxes.notna().any() else _first(df, ["race"])

    std["species"]      = _first(df, ["species", "Organism"])
    std["brain_region"] = _first(df, ["Brain Region"])

    # ------- Sex -------
    tmp = df.copy()
    standardize_sex_inplace(tmp)
    std["sex"] = tmp["sex"]

    # ------- APOE -------
    apoe_src = _first(df, ["apoe_genotype", "APOE Genotype"])
    if not apoe_src.empty:
        std["apoe_genotype_std"] = apoe_src.apply(apoe_to_std).astype("category")
        std["apoe_e4_dosage"]    = std["apoe_genotype_std"].apply(apoe_e4_dosage).astype("Int64")
        std["apoe_e4_carrier"]   = std["apoe_e4_dosage"].fillna(0).astype(int).gt(0)
        log("harmonize: APOE fields added")

    # ------- Braak (→ stage + label) -------
    braak_src = _first(df, ["braaksc", "Braak"])
    if not braak_src.empty:
        std["braak_stage"] = braak_src.apply(parse_braak).astype("Int64")
        std["braak_label"] = pd.Categorical(
            std["braak_stage"].map(braak_label_from_stage),
            categories=["Braak 0","Braak I","Braak II","Braak III","Braak IV","Braak V","Braak VI"],
            ordered=True
        )
        log(f"harmonize: braak_stage non-null: {std['braak_stage'].notna().sum():,}")

    # ------- CERAD (→ 1–4, 0–3, label) -------
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
        log(f"harmonize: cerad_score_0_3 non-null: {std['cerad_score_0_3'].notna().sum():,}")

    # ------- AD_status (built-in here) -------
    # Only compute if we have both braak_stage and cerad_score_0_3
    if "braak_stage" in std.columns and "cerad_score_0_3" in std.columns:
        b = pd.to_numeric(std["braak_stage"], errors="coerce")
        c = pd.to_numeric(std["cerad_score_0_3"], errors="coerce")
        ad_status = pd.Series("non-AD", index=std.index, dtype="object")
        ad_status.loc[(b >= 5) & (c <= 2)] = "AD"
        std["AD_status"] = ad_status
        log(f"harmonize: AD_status counts → {std['AD_status'].value_counts(dropna=False).to_dict()}")
    else:
        std["AD_status"] = pd.Series("non-AD", index=std.index, dtype="object")
        log("harmonize: AD_status defaulted (missing Braak or CERAD in input)")

    # ------- Cognition/tests -------
    std["cogdx"] = _first(df, ["cogdx", "Cognitive Status"])
    std["mmse"]  = pd.to_numeric(_first(df, ["cts_mmse30_lv", "Last MMSE Score"]), errors="coerce")
    std["moca"]  = pd.to_numeric(_first(df, ["Last MOCA Score"]), errors="coerce")
    std["casi"]  = pd.to_numeric(_first(df, ["Last CASI Score"]), errors="coerce")

    # ------- Cell-type labels -------
    std["celltype_major"]      = _first(df, ["major_cell_type", "Class", "subset"])
    std["celltype_label"]      = _first(df, ["cell_type_high_resolution", "Subclass", "cell.type", "celltypist_cell_label"])
    std["celltype_supertype"]  = _first(df, ["Supertype", "celltypist_simplified"])
    std["celltype_conf"]       = pd.to_numeric(
        _first(df, ["celltypist_conf_score", "Class confidence", "Subclass confidence", "Supertype confidence"]),
        errors="coerce"
    )

    # ------- QC -------
    for c in CANONICAL_QC:
        std[c] = _first(df, [c])

    # ------- Chemistry -------
    chem = pd.Series("Unknown", index=df.index, dtype="object")
    if "method" in df.columns and "dataset" in df.columns:
        chem = chem.mask(df["dataset"].eq("SEAAD"), df["method"].astype(str))
    if "dataset" in df.columns:
        chem = chem.mask(df["dataset"].eq("FUJITA"), "10x 3' v3")
    std["chemistry"] = chem

    # Attach back
    for c in std.columns:
        adata.obs[c] = std[c]
    adata.obs["harmonized"] = True
    log(f"harmonize_obs_inplace: done; columns added: {list(std.columns)}")

def compute_ad_status_inplace(df: pd.DataFrame):
    braak = pd.to_numeric(df.get("braak_stage"), errors="coerce")
    cerad = pd.to_numeric(df.get("cerad_score_0_3"), errors="coerce")
    status = pd.Series("non-AD", index=df.index, dtype="object")
    status.loc[(braak >= 5) & (cerad <= 2)] = "AD"
    df["AD_status"] = status
    vc = df["AD_status"].value_counts(dropna=False).to_dict()
    log(f"compute_ad_status_inplace: AD_status counts → {vc}")

def persist_obs_h5ad(h5ad_path, obs_df):
    log("persist_obs_h5ad: begin (pre-convert)")
    # Convert plain objects to strings for HDF5 friendliness
    for col in obs_df.columns:
        if pd.api.types.is_object_dtype(obs_df[col]):
            obs_df[col] = obs_df[col].astype(str)
    log(f"persist_obs_h5ad: writing obs with {obs_df.shape[0]:,} rows and {obs_df.shape[1]} columns")

    # anndata version compatibility for write_elem
    try:
        from anndata.io import write_elem
    except Exception:
        from anndata._io.utils import write_elem  # fallback for older versions

    with h5py.File(h5ad_path, "r+") as f:
        write_elem(f, "obs", obs_df)
    log("persist_obs_h5ad: done")

# ==========
# Main
# ==========
def main():
    log(f"Python {sys.version.split()[0]} | anndata {ad.__version__} | pandas {pd.__version__}")

    # Input file checks
    for k, p in files.items():
        if os.path.exists(p):
            size_gb = os.path.getsize(p) / (1024**3)
            log(f"Input {k}: {p} ({size_gb:.2f} GB)")
        else:
            log(f"Input {k}: {p} (MISSING)"); raise FileNotFoundError(p)

    # 1) Out-of-core concat over inner gene set
    log("Step 1/3: concat_on_disk (join='inner') starting…")
    ad.experimental.concat_on_disk(
        in_files=files,
        out_file="merged_allcells.h5ad",
        axis=0,
        join="inner",
        label="dataset",
        index_unique="-"   # ensure unique cell IDs
    )
    if os.path.exists("merged_allcells.h5ad"):
        size_gb = os.path.getsize("merged_allcells.h5ad") / (1024**3)
        log(f"concat_on_disk: wrote merged_allcells.h5ad ({size_gb:.2f} GB)")
    else:
        raise RuntimeError("concat_on_disk did not produce merged_allcells.h5ad")

    # 2) Open merged in backed mode and harmonize .obs
    log("Step 2/3: opening merged_allcells.h5ad (backed='r')")
    merged_b = ad.read_h5ad("merged_allcells.h5ad", backed="r")
    try:
        n_obs, n_vars = merged_b.n_obs, merged_b.n_vars
    except Exception:
        n_obs, n_vars = merged_b.shape
    log(f"Merged shape: {n_obs:,} cells × {n_vars:,} genes")

    log("Harmonizing metadata (.obs only)…")
    harmonize_obs_inplace(merged_b)

    # 2a) Copy .obs to memory (needed for persistence)
    log("Copying .obs to memory (DataFrame)…")
    obs = merged_b.obs.copy()
    obs_mem_gb = obs.memory_usage(index=True, deep=True).sum() / (1024**3)
    log(f".obs loaded: approx {obs_mem_gb:.2f} GB in-memory")

    # 2b) SEAAD: batch/chemistry; FUJITA chemistry
    if "dataset" in obs.columns:
        is_seaad = obs["dataset"].eq("SEAAD")
        if "load_name" in obs.columns:
            obs.loc[is_seaad, "batch"] = obs.loc[is_seaad, "load_name"].astype(str)
        if "method" in obs.columns:
            obs.loc[is_seaad, "chemistry"] = obs.loc[is_seaad, "method"].astype(str)
        if "dataset" in obs.columns:
            obs.loc[obs["dataset"].eq("FUJITA"), "chemistry"] = "10x 3' v3"
        vc_ds = obs["dataset"].value_counts().to_dict()
        log(f"Dataset counts (obs): {vc_ds}")

    # 2c) AD status
    if "projid" in obs.columns:
        obs["projid"] = pd.to_numeric(obs["projid"], errors="coerce").astype("Int64")
    compute_ad_status_inplace(obs)


    # --- release the read-only backed handle before opening with h5py r+ ---
    log("Releasing backed file handle so we can write to H5AD …")
    try:
        if hasattr(merged_b, "file") and hasattr(merged_b.file, "close"):
            merged_b.file.close()          # closes the HDF5 handle held by AnnData
            log("Closed backed handle via merged_b.file.close()")
    except Exception as e:
        log(f"(warn) merged_b.file.close() failed: {e}")
    
    # make sure nothing still references the file
    del merged_b
    import gc; gc.collect()

    # 3) Persist only the .obs table back into the H5AD
    log("Step 3/3: persisting updated .obs back into merged_allcells.h5ad…")
    persist_obs_h5ad("merged_allcells.h5ad", obs)
    log("All done ✅")

if __name__ == "__main__":
    main()
