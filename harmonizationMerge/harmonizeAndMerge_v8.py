#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, re, h5py
from datetime import datetime
import numpy as np
import pandas as pd
import anndata as ad

# ==========
# Config
# ==========
MERGED_OUT   = "merged_allcells.h5ad"
INDEX_UNIQUE = "-"   # must match concat_on_disk(index_unique)
FILES = {
    "MIT_ROSMAP": "../celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEAAD":      "../celltypist/seaad_celltypist_GPU_counts_only.h5ad",
    "FUJITA":     "../celltypist/fujita_celltypist_GPU_counts_only.h5ad",
}

# ==========
# Logging
# ==========
START = time.perf_counter()
def _now_ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def memline(): return f" | RSS {_PROC.memory_info().rss/(1024**3):.2f} GB"
except Exception:
    def memline(): return ""
def log(msg):
    print(f"[{_now_ts()}] (+{time.perf_counter()-START:7.2f}s) {msg}{memline()}", flush=True)

# ==========
# Source obs columns we may need
# ==========
NEEDED_OBS_COLS = [
    # demographics / pathology
    "Age at Death","age_death","braaksc","Braak","ceradsc","CERAD score",
    "Sex","sex","msex","Gender","pmi","PMI","educ","Years of education","race",
    "spanish","Hispanic_Latino","species","Organism","Brain Region",
    # SEAAD checkbox race
    "Race (choice=White)","Race (choice=Black_ African American)","Race (choice=Asian)",
    "Race (choice=American Indian_ Alaska Native)","Race (choice=Native Hawaiian or Pacific Islander)",
    "Race (choice=Unknown or unreported)","Race (choice=Other)","specify other race",
    # identities / study
    "projid","Study","Primary Study Name","Donor ID","sample_id",
    "individualID","individualID_x","individualID_y","subject",
    # CellTypist (will be passed through unmodified)
    "celltypist_cell_label","celltypist_simplified","celltypist_conf_score",
    "Class","Subclass","Supertype","Class confidence","Subclass confidence","Supertype confidence",
    # QC
    "n_genes_by_counts","log1p_n_genes_by_counts","total_counts","log1p_total_counts",
    "pct_counts_in_top_20_genes","total_counts_mt","log1p_total_counts_mt","pct_counts_mt",
    "total_counts_ribo","log1p_total_counts_ribo","pct_counts_ribo",
    "total_counts_hb","log1p_total_counts_hb","pct_counts_hb",
    "n_genes",
    "Doublet score","doublet_scores","predicted_doublets","doublet_label","outlier","mt_outlier",
    # batch / chemistry / APOE
    "load_name","method","apoe_genotype","APOE Genotype",
    # cognition
    "cogdx","Cognitive Status","cts_mmse30_lv","Last MMSE Score","Last MOCA Score","Last CASI Score",
]

# ==========
# Utils
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

# ---- Sex normalization (dataset-aware) ----
def normalize_sex_values(series: pd.Series) -> pd.Categorical:
    s = series.copy()
    s = s.replace({1: "Male", 0: "Female", "1": "Male", "0": "Female"})
    s = (
        s.astype("string")
         .str.strip()
         .str.lower()
         .replace({
             "female": "female", "male": "male",
             "f": "female", "m": "male",
             "woman": "female", "man": "male",
             "female (f)": "female", "male (m)": "male",
             "unknown": "na", "unk": "na", "u": "na",
             "n/a": "na", "na": "na", "none": "na", "other": "na",
         })
    )
    s = s.where(s.isin(["female", "male", "na"]), s.str[:1].map({"f": "female", "m": "male"}))
    out = s.replace({"female": "Female", "male": "Male", "na": pd.NA})
    return pd.Categorical(out, categories=["Female", "Male"])

def build_sex_raw(df: pd.DataFrame) -> pd.Series:
    """
    Use the columns discovered in your audit:
      - MIT_ROSMAP: prefer 'msex', else 'sex'
      - SEAAD     : prefer 'Gender', else 'Sex'
      - FUJITA    : 'msex' if present
    """
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    if "dataset" not in df.columns:
        return out
    ds = df["dataset"].astype(str)

    mask = ds.eq("MIT_ROSMAP")
    if "msex" in df.columns:
        out.loc[mask] = df.loc[mask, "msex"]
    elif "sex" in df.columns:
        out.loc[mask] = df.loc[mask, "sex"]

    mask = ds.eq("SEAAD")
    if "Gender" in df.columns:
        out.loc[mask] = df.loc[mask, "Gender"]
    elif "Sex" in df.columns:
        out.loc[mask] = df.loc[mask, "Sex"]

    mask = ds.eq("FUJITA")
    if "msex" in df.columns:
        out.loc[mask] = df.loc[mask, "msex"]

    return out

# ---- APOE ----
_APOE_PAIR_RE = re.compile(r"[eεE]?\s*([234])\s*[/\-\|\s]?\s*[eεE]?\s*([234])")

def apoe_to_std(x):
    if pd.isna(x):
        return np.nan
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
    if pd.isna(apoe_std):
        return np.nan
    l, r = apoe_std.replace("E", "").split("/")
    return int(l == "4") + int(r == "4")

# ---- Braak/CERAD ----
_ROMAN_TO_INT = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6}
_INT_TO_ROMAN = {v:k for k,v in _ROMAN_TO_INT.items()}

def parse_braak(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return int(np.clip(int(round(float(x))), 0, 6))
        except Exception:
            return np.nan
    s = str(x).strip().upper().replace("BRAAK", "").strip()
    if s in _ROMAN_TO_INT:
        return _ROMAN_TO_INT[s]
    try:
        return int(np.clip(int(float(s)), 0, 6))
    except Exception:
        for r, v in _ROMAN_TO_INT.items():
            if r in s:
                return v
    return np.nan

def braak_label_from_stage(stage):
    if pd.isna(stage):
        return np.nan
    stage = int(stage)
    return "Braak 0" if stage == 0 else f"Braak {_INT_TO_ROMAN.get(stage, str(stage))}"

_CERAD_14_LABEL = {1:"Absent",2:"Sparse",3:"Moderate",4:"Frequent"}
_CERAD_LABEL_14 = {v.upper():k for k,v in _CERAD_14_LABEL.items()}

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
        except Exception:
            return np.nan
    return _CERAD_LABEL_14.get(str(x).strip().upper(), np.nan)

def cerad_14_to_03(v):
    return np.nan if pd.isna(v) else int(v) - 1

def cerad_label_from_14(v):
    return np.nan if pd.isna(v) else _CERAD_14_LABEL.get(int(v), np.nan)

# ---- SEAAD race from checkbox columns ----
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
    out = pd.Series(np.nan, index=df.index, dtype="object")
    if not present or "dataset" not in df.columns:
        log("derive_race_from_seaad: no checkbox columns or dataset missing")
        return out
    mask = df["dataset"].astype(str).eq("SEAAD")
    if not mask.any():
        log("derive_race_from_seaad: no SEAAD rows")
        return out

    def _one(row):
        picks = [
            mapping[c]
            for c in present
            if isinstance(row.get(c), str) and row[c].strip().lower() == "checked"
        ]
        if not picks and "specify other race" in row and pd.notna(row["specify other race"]):
            return str(row["specify other race"])
        if not picks:
            return np.nan
        return picks[0] if len(picks) == 1 else " / ".join(sorted(set(picks)))

    out.loc[mask] = df.loc[
        mask,
        present + (["specify other race"] if "specify other race" in df.columns else []),
    ].apply(_one, axis=1)
    vc = out.loc[mask].value_counts(dropna=False).head(10).to_dict()
    log(f"derive_race_from_seaad: top labels {vc}")
    return out

# ---- QC columns to copy if present ----
CANONICAL_QC = [
    "n_genes_by_counts","log1p_n_genes_by_counts","total_counts","log1p_total_counts",
    "pct_counts_in_top_20_genes","total_counts_mt","log1p_total_counts_mt","pct_counts_mt",
    "total_counts_ribo","log1p_total_counts_ribo","pct_counts_ribo",
    "total_counts_hb","log1p_total_counts_hb","pct_counts_hb",
    "n_genes","doublet_scores","predicted_doublets","doublet_label","outlier","mt_outlier"
]

# ==========
# Build base .obs from sources aligned to merged index
# ==========
def build_obs_union_from_sources(files_map, merged_index, merged_dataset, index_unique="-"):
    parts = []
    merged_index = pd.Index(merged_index)
    sex_re = re.compile(r"\b(msex|sex|gender|sex[_\s]*at[_\s]*birth)\b", flags=re.I)
    braak_aliases = {"braak", "braak stage", "braak_stage"}
    cerad_aliases = {"cerad", "cerad score", "cerad_score"}

    for ds_key, path in files_map.items():
        sel = (merged_dataset == ds_key)
        n_sel = int(sel.sum())
        if not n_sel:
            log(f"{ds_key}: no rows in merged; skip")
            continue

        suffix = index_unique + ds_key
        merged_ids = merged_index[sel]
        orig_ids   = merged_ids.str.rsplit(suffix, n=1).str[0]

        log(f"{ds_key}: loading source obs for {n_sel:,} rows …")
        src = ad.read_h5ad(path, backed="r")
        src_cols = list(src.obs.columns)

        # discover dynamic columns we might need (sex variants; alias names; APOE)
        dynamic_sex   = [c for c in src_cols if sex_re.search(c)]
        dynamic_braak = [c for c in src_cols if c.lower() in braak_aliases]
        dynamic_cerad = [c for c in src_cols if c.lower() in cerad_aliases]
        dynamic_apoe  = [c for c in src_cols if re.search(r"apoe", c, flags=re.I)]

        keep = [c for c in NEEDED_OBS_COLS if c in src_cols]
        keep = sorted(set(keep + dynamic_sex + dynamic_braak + dynamic_cerad + dynamic_apoe))

        so = src.obs[keep].reindex(orig_ids)
        if hasattr(src, "file") and hasattr(src.file, "close"):
            src.file.close()
        del src

        so.index = merged_ids
        so["dataset"] = ds_key
        parts.append(so)

        # small sanity (silence FutureWarning with dtype="object")
        bnn = int(so.get("braaksc",     pd.Series(index=so.index, dtype="object")).notna().sum()) \
            + int(so.get("Braak",       pd.Series(index=so.index, dtype="object")).notna().sum())
        cnn = int(so.get("ceradsc",     pd.Series(index=so.index, dtype="object")).notna().sum()) \
            + int(so.get("CERAD score", pd.Series(index=so.index, dtype="object")).notna().sum())
        sx  = dynamic_sex if dynamic_sex else "none"
        log(f"{ds_key}: base_obs non-null — Braak {bnn:,}, CERAD {cnn:,}; sex-like cols: {sx}")

    if not parts:
        return pd.DataFrame(index=merged_index)
    out = pd.concat(parts, axis=0).reindex(merged_index)
    return out

# ==========
# Harmonize on a plain DataFrame and return standardized columns
# ==========
def harmonize_obs_from_df(df: pd.DataFrame) -> pd.DataFrame:
    log("harmonize_obs_from_df: begin")
    std = pd.DataFrame(index=df.index)

    # Helper: SQL-style COALESCE across columns, applied row-by-row
    def coalesce_cols(candidates):
        out = pd.Series(pd.NA, index=df.index, dtype="object")
        for c in candidates:
            if c in df.columns:
                mask = out.isna() & df[c].notna()
                out.loc[mask] = df.loc[mask, c]
        return out

    # ----- Dataset / IDs / study -----
    std["dataset"]      = _first(df, ["dataset"])
    std["projid"]       = _first(df, ["projid"])

    # SEAAD: if projid is missing, use sample_id as a surrogate ID
    if "dataset" in df.columns and "sample_id" in df.columns:
        ds = df["dataset"].astype(str)
        mask = ds.eq("SEAAD") & std["projid"].isna() & df["sample_id"].notna()
        std.loc[mask, "projid"] = df.loc[mask, "sample_id"]

    std["study"]        = coalesce_cols(["Study", "Primary Study Name"])
    std["individualID"] = coalesce_cols(
        ["individualID", "Donor ID", "individualID_y", "individualID_x", "subject"]
    )

    # ----- Age / PMI / education / hispanic / race / species / region -----
    age_raw = coalesce_cols(["age_death", "Age at Death"])
    if age_raw.notna().any():
        std["age_death"] = clean_age_column(age_raw)
        log(f"harmonize: age_death non-null {std['age_death'].notna().sum():,}")
    else:
        std["age_death"] = pd.Series(np.nan, index=df.index)

    std["pmi"]        = pd.to_numeric(coalesce_cols(["pmi", "PMI"]), errors="coerce")
    std["educ_years"] = pd.to_numeric(coalesce_cols(["educ", "Years of education"]), errors="coerce")
    std["hispanic_latino"] = coalesce_cols(["spanish", "Hispanic_Latino"])

    # Impute missing PMI with median PMI per dataset
    if "dataset" in std.columns:
        ds_series = std["dataset"].astype(str)
        for ds_name in ds_series.unique():
            mask_ds = ds_series.eq(ds_name)
            median_val = std.loc[mask_ds, "pmi"].median()
            if pd.notna(median_val):
                missing_mask = mask_ds & std["pmi"].isna()
                if missing_mask.any():
                    std.loc[missing_mask, "pmi"] = median_val
        log(f"harmonize: pmi missing after imputation {std['pmi'].isna().sum():,}")

    df2 = df.copy()
    if "dataset" not in df2.columns:
        df2["dataset"] = std["dataset"]
    race_from_boxes = derive_race_from_seaad(df2)

    race = race_from_boxes
    if race.isna().all() and "race" in df.columns:
        race = df["race"]
    elif "race" in df.columns:
        mask = race.isna()
        race.loc[mask] = df.loc[mask, "race"]
    std["race"] = race

    std["species"]      = coalesce_cols(["species", "Organism"])
    std["brain_region"] = coalesce_cols(["Brain Region"])

    # ----- Sex -----
    std["sex_raw"] = build_sex_raw(df)
    std["sex"]     = normalize_sex_values(std["sex_raw"])
    log(f"harmonize: sex non-null {std['sex'].notna().sum():,}")

    # ----- APOE -----
    apoe_cols = [c for c in df.columns
                 if re.search(r"apoe\s*[_\- ]*\s*genotype", c, flags=re.I)]
    if not apoe_cols:
        apoe_cols = [c for c in df.columns if "apoe" in c.lower()]

    if apoe_cols:
        apoe_src = coalesce_cols(apoe_cols)
        if apoe_src.notna().any():
            std["apoe_genotype_std"] = apoe_src.apply(apoe_to_std).astype("category")
            std["apoe_e4_dosage"]    = std["apoe_genotype_std"].apply(apoe_e4_dosage).astype("Int64")
            std["apoe_e4_carrier"]   = std["apoe_e4_dosage"].fillna(0).astype(int).gt(0)
            log("harmonize: APOE added")

    # ----- Braak -----
    braak_cols = [c for c in ["braaksc", "Braak"] if c in df.columns]
    if braak_cols:
        braak_src = coalesce_cols(braak_cols)
        if braak_src.notna().any():
            tmp = braak_src.apply(parse_braak)
            std["braak_stage"] = tmp.astype("Int64")
            std["braak_label"] = pd.Categorical(
                std["braak_stage"].map(braak_label_from_stage),
                categories=["Braak 0","Braak I","Braak II",
                            "Braak III","Braak IV","Braak V","Braak VI"],
                ordered=True
            )
            log(f"harmonize: braak_stage non-null {std['braak_stage'].notna().sum():,}")

    # ----- CERAD -----
    cerad_cols = [c for c in ["ceradsc", "CERAD score"] if c in df.columns]
    if cerad_cols:
        cerad_src = coalesce_cols(cerad_cols)
        if cerad_src.notna().any():
            cerad_14 = cerad_src.apply(to_cerad_1_4).astype("Int64")
            std["cerad_score_1_4"] = cerad_14
            std["cerad_score_0_3"] = cerad_14.apply(cerad_14_to_03).astype("Int64")
            std["cerad_label"]     = pd.Categorical(
                cerad_14.apply(cerad_label_from_14),
                categories=["Absent","Sparse","Moderate","Frequent"],
                ordered=True
            )
            log(f"harmonize: cerad_score_0_3 non-null {std['cerad_score_0_3'].notna().sum():,}")

    # ----- AD status -----
    if "braak_stage" in std.columns and "cerad_score_0_3" in std.columns:
        b = pd.to_numeric(std["braak_stage"], errors="coerce")
        c = pd.to_numeric(std["cerad_score_0_3"], errors="coerce")
        ad_status = pd.Series("non-AD", index=std.index, dtype="object")
        ad_status.loc[(b >= 5) & (c <= 2)] = "AD"
        std["AD_status"] = ad_status
        log(f"harmonize: AD_status {std['AD_status'].value_counts(dropna=False).to_dict()}")
    else:
        std["AD_status"] = pd.Series("non-AD", index=std.index, dtype="object")
        log("harmonize: AD_status defaulted (missing Braak/CERAD)")

    # ----- Cognition -----
    std["cogdx"] = coalesce_cols(["cogdx", "Cognitive Status"])
    std["mmse"]  = pd.to_numeric(coalesce_cols(["cts_mmse30_lv", "Last MMSE Score"]), errors="coerce")
    std["moca"]  = pd.to_numeric(_first(df, ["Last MOCA Score"]), errors="coerce")
    std["casi"]  = pd.to_numeric(_first(df, ["Last CASI Score"]), errors="coerce")

    # ----- QC & doublets -----
    for c in CANONICAL_QC:
        if c in df.columns:
            std[c] = df[c]
        elif c == "doublet_scores" and "Doublet score" in df.columns:
            # Fallback if some dataset has only "Doublet score"
            std[c] = pd.to_numeric(df["Doublet score"], errors="coerce")
        else:
            std[c] = pd.Series(np.nan, index=df.index)

    # ----- Chemistry & batch -----
    chem = pd.Series("Unknown", index=df.index, dtype="object")
    if "method" in df.columns:
        chem = chem.mask(std["dataset"].astype(str).eq("SEAAD"),
                         df["method"].astype(str))
    chem = chem.mask(std["dataset"].astype(str).eq("FUJITA"),
                     "10x 3' v3")
    std["chemistry"] = chem

    if "load_name" in df.columns:
        std["batch"] = np.where(
            std["dataset"].astype(str).eq("SEAAD"),
            df["load_name"].astype(str),
            _first(df, ["batch"])
        )

    # NOTE: we now keep projid as string-like, so SEAAD can use sample_id there
    std["harmonized"] = True
    log("harmonize_obs_from_df: done")
    return std

# ==========
# Persist only /obs
# ==========
def persist_obs_h5ad(h5ad_path, obs_df):
    log("persist_obs_h5ad: begin (pre-convert)")
    for col in obs_df.columns:
        if pd.api.types.is_object_dtype(obs_df[col]):
            obs_df[col] = obs_df[col].astype(str)
    log(f"persist_obs_h5ad: writing obs {obs_df.shape[0]:,} rows × {obs_df.shape[1]} cols")
    try:
        from anndata.io import write_elem
    except Exception:
        from anndata._io.utils import write_elem
    with h5py.File(h5ad_path, "r+") as f:
        write_elem(f, "obs", obs_df)
    log("persist_obs_h5ad: done")

# ==========
# Main
# ==========
def main():
    log(f"Python {sys.version.split()[0]} | anndata {ad.__version__} | pandas {pd.__version__}")
    for k,p in FILES.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{k} missing: {p}")
        log(f"Input {k}: {p} ({os.path.getsize(p)/(1024**3):.2f} GB)")

    # 1) On-disk concat (inner gene set)
    log("Step 1/3: concat_on_disk (join='inner') …")
    ad.experimental.concat_on_disk(
        in_files=FILES,
        out_file=MERGED_OUT,
        axis=0,
        join="inner",
        label="dataset",
        index_unique=INDEX_UNIQUE
    )
    if not os.path.exists(MERGED_OUT):
        raise RuntimeError("concat_on_disk did not produce output")
    log(f"concat_on_disk: wrote {MERGED_OUT} ({os.path.getsize(MERGED_OUT)/(1024**3):.2f} GB)")

    # 2) Open merged (backed) and assemble base_obs from sources
    log("Step 2/3: open merged (backed='r')")
    m = ad.read_h5ad(MERGED_OUT, backed="r")
    try:
        n_obs, n_vars = m.n_obs, m.n_vars
    except Exception:
        n_obs, n_vars = m.shape
    log(f"Merged shape: {n_obs:,} cells × {n_vars:,} genes")

    if "dataset" not in m.obs.columns:
        raise RuntimeError("Merged .obs lacks 'dataset' — need label='dataset' in concat_on_disk")

    log("Assembling base_obs from sources (no matrix load) …")
    base_obs = build_obs_union_from_sources(FILES, m.obs_names, m.obs["dataset"], INDEX_UNIQUE)
    base_gb = base_obs.memory_usage(index=True, deep=True).sum()/(1024**3)
    log(f"base_obs built: {base_obs.shape} (~{base_gb:.2f} GB)")

    log("Harmonizing metadata from base_obs …")
    harm = harmonize_obs_from_df(base_obs)
    harm_gb = harm.memory_usage(index=True, deep=True).sum()/(1024**3)
    log(f"harmonized obs size: ~{harm_gb:.2f} GB")

    # === Pass-through: KEEP CellTypist columns EXACTLY as in sources ===
    passthrough_cols = [
        c for c in base_obs.columns
        if c.startswith("celltypist_")
        or c in ["Class","Subclass","Supertype","Class confidence",
                 "Subclass confidence","Supertype confidence"]
    ]
    celltypist_df = base_obs[passthrough_cols].copy() if passthrough_cols else pd.DataFrame(index=base_obs.index)
    if passthrough_cols:
        log(f"CellTypist pass-through columns: {passthrough_cols}")

    # Scaffold from merged (keeps dataset/index), join harmonized (drop any potential overlap with pass-through),
    # then join CellTypist columns unmodified.
    log("Joining harmonized fields + CellTypist pass-through onto merged scaffold …")
    harm_no_ct = harm.drop(columns=[c for c in harm.columns if c in passthrough_cols], errors="ignore")
    scaffold   = m.obs[["dataset"]].copy()
    obs        = scaffold.join(harm_no_ct.drop(columns=["dataset"], errors="ignore"), how="left")
    if not celltypist_df.empty:
        obs = obs.join(celltypist_df, how="left")

    # Optional: quick summaries
    log(f"Dataset counts: {obs['dataset'].value_counts().to_dict()}")
    if "sex" in obs.columns:
        try:
            sex_counts = obs.groupby("dataset")["sex"].value_counts(dropna=False).to_dict()
            log(f"sex by dataset → {sex_counts}")
        except Exception as e:
            log(f"(warn) sex-by-dataset summary failed: {e}")
    log(
        f"Non-null braak_stage={obs.get('braak_stage', pd.Series(index=obs.index)).notna().sum():,}, "
        f"cerad_0_3={obs.get('cerad_score_0_3', pd.Series(index=obs.index)).notna().sum():,}"
    )
    if "AD_status" in obs.columns:
        log(f"AD_status: {obs['AD_status'].value_counts(dropna=False).to_dict()}")

    # Close backed handle before writing
    log("Releasing backed file handle before writing …")
    try:
        if hasattr(m, "file") and hasattr(m.file, "close"):
            m.file.close()
            log("Closed backed handle")
    except Exception as e:
        log(f"(warn) close handle failed: {e}")
    del m
    import gc; gc.collect()

    # 3) Persist only /obs back into the merged file
    log("Step 3/3: persist updated .obs into H5AD …")
    persist_obs_h5ad(MERGED_OUT, obs)
    log("All done ✅")

if __name__ == "__main__":
    main()
