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
# Config
# ==========
MERGED_OUT = "merged_allcells.h5ad"
INDEX_UNIQUE = "-"   # must match concat_on_disk(index_unique)
FILES = {
    "MIT_ROSMAP": "../celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEAAD":      "../celltypist/seaad_celltypist_GPU_counts_only.h5ad",
    "FUJITA":     "../celltypist/fujita_celltypist_GPU_counts_only.h5ad",
}

# ==========
# Logging helpers
# ==========
START = time.perf_counter()
def _now_ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def memline():
        return f" | RSS {_PROC.memory_info().rss/(1024**3):.2f} GB"
except Exception:
    def memline(): return ""
def log(msg):
    print(f"[{_now_ts()}] (+{time.perf_counter()-START:7.2f}s) {msg}{memline()}", flush=True)

# ==========
# Column candidates we may need from sources
# ==========
NEEDED_OBS_COLS = [
    # demographics / pathology
    "Age at Death","age_death","braaksc","Braak","ceradsc","CERAD score",
    "Sex","sex","msex","pmi","PMI","educ","Years of education","race",
    "spanish","Hispanic_Latino","species","Organism","Brain Region",
    # SEAAD checkbox race
    "Race (choice=White)","Race (choice=Black_ African American)","Race (choice=Asian)",
    "Race (choice=American Indian_ Alaska Native)","Race (choice=Native Hawaiian or Pacific Islander)",
    "Race (choice=Unknown or unreported)","Race (choice=Other)","specify other race",
    # identities / study
    "projid","Study","Primary Study Name","Donor ID","individualID","individualID_x","individualID_y","subject",
    # cell types
    "major_cell_type","Class","subset","cell_type_high_resolution","Subclass","cell.type",
    "celltypist_cell_label","Supertype","celltypist_simplified",
    "celltypist_conf_score","Class confidence","Subclass confidence","Supertype confidence",
    # QC
    "n_genes_by_counts","log1p_n_genes_by_counts","total_counts","log1p_total_counts",
    "pct_counts_in_top_20_genes","total_counts_mt","log1p_total_counts_mt","pct_counts_mt",
    "total_counts_ribo","log1p_total_counts_ribo","pct_counts_ribo",
    "total_counts_hb","log1p_total_counts_hb","pct_counts_hb",
    "n_genes","Doublet score","predicted_doublets","doublet_label","outlier","mt_outlier",
    # batch / chemistry / APOE
    "load_name","method","apoe_genotype","APOE Genotype",
    # cognition
    "cogdx","Cognitive Status","cts_mmse30_lv","Last MMSE Score","Last MOCA Score","Last CASI Score",
]

# ==========
# Utilities
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

# ---- Sex standardization (robust) ----
def standardize_sex_inplace(df: pd.DataFrame) -> None:
    src = None
    for c in ["sex", "Sex", "msex"]:
        if c in df.columns:
            src = c; break
    if src is None:
        df["sex"] = pd.Series(
            pd.Categorical.from_codes(np.full(len(df), -1, dtype=int), categories=["Female","Male"]),
            index=df.index
        )
        log("standardize_sex_inplace: no source column → NA categorical 'sex'")
        return
    s = df[src].replace({1:"Male",0:"Female","1":"Male","0":"Female"})
    s = s.astype(str).str.strip().str.lower().replace({"f":"female","m":"male"})
    s = s.map({"female":"Female","male":"Male"})
    df["sex"] = pd.Categorical(s, categories=["Female","Male"], ordered=False)
    vc = pd.Series(df["sex"]).value_counts(dropna=False)
    log(f"standardize_sex_inplace: counts {vc.to_dict()}")

# ---- APOE ----
_APOE_PAIR_RE = re.compile(r"[eεE]?\s*([234])\s*[/\-\|\s]?\s*[eεE]?\s*([234])")
def apoe_to_std(x):
    if pd.isna(x): return np.nan
    s = str(int(x)) if isinstance(x,(int,float,np.integer,np.floating)) else str(x)
    s = s.strip().replace("ε","e").replace("E","e").replace("-","/").replace("|","/")
    m = _APOE_PAIR_RE.search(s)
    if m:
        a,b = sorted(m.groups())
        return f"E{a}/E{b}"
    digits = [ch for ch in s if ch in "234"]
    if len(digits) >= 2:
        a,b = sorted(digits[:2]); return f"E{a}/E{b}"
    return np.nan
def apoe_e4_dosage(apoe_std):
    if pd.isna(apoe_std): return np.nan
    l,r = apoe_std.replace("E","").split("/")
    return int(l=="4") + int(r=="4")

# ---- Braak ----
_ROMAN_TO_INT = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6}
_INT_TO_ROMAN = {v:k for k,v in _ROMAN_TO_INT.items()}
def parse_braak(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.integer,np.floating)):
        try: return int(np.clip(int(round(float(x))),0,6))
        except: return np.nan
    s = str(x).strip().upper().replace("BRAAK","").strip()
    if s in _ROMAN_TO_INT: return _ROMAN_TO_INT[s]
    try: return int(np.clip(int(float(s)),0,6))
    except:
        for r,v in _ROMAN_TO_INT.items():
            if r in s: return v
    return np.nan
def braak_label_from_stage(stage):
    if pd.isna(stage): return np.nan
    stage = int(stage)
    return "Braak 0" if stage==0 else f"Braak {_INT_TO_ROMAN.get(stage,str(stage))}"

# ---- CERAD ----
_CERAD_14_LABEL = {1:"Absent",2:"Sparse",3:"Moderate",4:"Frequent"}
_CERAD_LABEL_14 = {v.upper():k for k,v in _CERAD_14_LABEL.items()}
def to_cerad_1_4(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.integer,np.floating)):
        try:
            v = int(round(float(x)))
            if v in (1,2,3,4): return v
            if v in (0,1,2,3): return v+1
        except: return np.nan
    return _CERAD_LABEL_14.get(str(x).strip().upper(), np.nan)
def cerad_14_to_03(v): return np.nan if pd.isna(v) else int(v)-1
def cerad_label_from_14(v): return np.nan if pd.isna(v) else _CERAD_14_LABEL.get(int(v), np.nan)

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
        picks = [mapping[c] for c in present if isinstance(row.get(c),str) and row[c].strip().lower()=="checked"]
        if not picks and "specify other race" in row and pd.notna(row["specify other race"]):
            return str(row["specify other race"])
        if not picks: return np.nan
        return picks[0] if len(picks)==1 else " / ".join(sorted(set(picks)))
    out.loc[mask] = df.loc[mask, present + (["specify other race"] if "specify other race" in df.columns else [])].apply(_one, axis=1)
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
    for ds_key, path in files_map.items():
        sel = (merged_dataset == ds_key)
        n_sel = int(sel.sum())
        if not n_sel:
            log(f"{ds_key}: no rows in merged; skip")
            continue
        suffix = index_unique + ds_key
        merged_ids = merged_index[sel]
        # strip suffix once from the right
        orig_ids = merged_ids.str.rsplit(suffix, n=1).str[0]

        log(f"{ds_key}: loading source obs for {n_sel:,} rows …")
        src = ad.read_h5ad(path, backed="r")
        keep = [c for c in NEEDED_OBS_COLS if c in src.obs.columns]
        so = (src.obs[keep]).reindex(orig_ids)
        if hasattr(src, "file") and hasattr(src.file, "close"):
            src.file.close()
        del src

        so.index = merged_ids
        so["dataset"] = ds_key
        parts.append(so)

        # small sanity
        bnn = int(so.get("braaksc", pd.Series(index=so.index)).notna().sum()) + int(so.get("Braak", pd.Series(index=so.index)).notna().sum())
        cnn = int(so.get("ceradsc", pd.Series(index=so.index)).notna().sum()) + int(so.get("CERAD score", pd.Series(index=so.index)).notna().sum())
        log(f"{ds_key}: base_obs non-null — Braak {bnn:,}, CERAD {cnn:,}")

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

    # Dataset / IDs / study
    std["dataset"]      = _first(df, ["dataset"])
    std["projid"]       = _first(df, ["projid"])
    std["study"]        = _first(df, ["Study","Primary Study Name"])
    std["individualID"] = _first(df, ["individualID","Donor ID","individualID_y","individualID_x","subject"])

    # Age / PMI / educ / race / species / region
    col = next((c for c in ["age_death","Age at Death"] if c in df.columns), None)
    if col is not None:
        std["age_death"] = clean_age_column(df[col]); log(f"harmonize: age_death non-null {std['age_death'].notna().sum():,}")
    std["pmi"]            = pd.to_numeric(_first(df, ["pmi","PMI"]), errors="coerce")
    std["educ_years"]     = pd.to_numeric(_first(df, ["educ","Years of education"]), errors="coerce")
    std["hispanic_latino"]= _first(df, ["spanish","Hispanic_Latino"])

    # Race (SEAAD boxes preferred)
    df2 = df.copy()
    if "dataset" not in df2.columns:
        df2["dataset"] = std["dataset"]
    race_from_boxes = derive_race_from_seaad(df2)
    std["race"] = race_from_boxes if race_from_boxes.notna().any() else _first(df, ["race"])

    std["species"]      = _first(df, ["species","Organism"])
    std["brain_region"] = _first(df, ["Brain Region"])

    # Sex
    tmp = df.copy()
    standardize_sex_inplace(tmp)
    std["sex"] = tmp["sex"]

    # APOE
    apoe_src = _first(df, ["apoe_genotype","APOE Genotype"])
    if not apoe_src.empty:
        std["apoe_genotype_std"] = apoe_src.apply(apoe_to_std).astype("category")
        std["apoe_e4_dosage"]    = std["apoe_genotype_std"].apply(apoe_e4_dosage).astype("Int64")
        std["apoe_e4_carrier"]   = std["apoe_e4_dosage"].fillna(0).astype(int).gt(0)
        log("harmonize: APOE added")

    # Braak
    braak_src = _first(df, ["braaksc","Braak"])
    if not braak_src.empty:
        std["braak_stage"] = braak_src.apply(parse_braak).astype("Int64")
        std["braak_label"] = pd.Categorical(
            std["braak_stage"].map(braak_label_from_stage),
            categories=["Braak 0","Braak I","Braak II","Braak III","Braak IV","Braak V","Braak VI"],
            ordered=True
        )
        log(f"harmonize: braak_stage non-null {std['braak_stage'].notna().sum():,}")

    # CERAD
    cerad_src = _first(df, ["ceradsc","CERAD score"])
    if not cerad_src.empty:
        cerad_14 = cerad_src.apply(to_cerad_1_4).astype("Int64")
        std["cerad_score_1_4"] = cerad_14
        std["cerad_score_0_3"] = cerad_14.apply(cerad_14_to_03).astype("Int64")
        std["cerad_label"]     = pd.Categorical(
            cerad_14.apply(cerad_label_from_14),
            categories=["Absent","Sparse","Moderate","Frequent"],
            ordered=True
        )
        log(f"harmonize: cerad_score_0_3 non-null {std['cerad_score_0_3'].notna().sum():,}")

    # AD_status
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

    # Cognition
    std["cogdx"] = _first(df, ["cogdx","Cognitive Status"])
    std["mmse"]  = pd.to_numeric(_first(df, ["cts_mmse30_lv","Last MMSE Score"]), errors="coerce")
    std["moca"]  = pd.to_numeric(_first(df, ["Last MOCA Score"]), errors="coerce")
    std["casi"]  = pd.to_numeric(_first(df, ["Last CASI Score"]), errors="coerce")

    # Cell-type labels
    std["celltype_major"]     = _first(df, ["major_cell_type","Class","subset"])
    std["celltype_label"]     = _first(df, ["cell_type_high_resolution","Subclass","cell.type"])
    std["celltype_supertype"] = _first(df, ["Supertype","celltypist_simplified"])
    std["celltype_conf"]      = pd.to_numeric(
        _first(df, ["celltypist_conf_score","Class confidence","Subclass confidence","Supertype confidence"]),
        errors="coerce"
    )

    # QC
    for c in CANONICAL_QC:
        std[c] = _first(df, [c])

    # Chemistry & batch
    chem = pd.Series("Unknown", index=df.index, dtype="object")
    if "method" in df.columns:
        chem = chem.mask(std["dataset"].astype(str).eq("SEAAD"), df["method"].astype(str))
    chem = chem.mask(std["dataset"].astype(str).eq("FUJITA"), "10x 3' v3")
    std["chemistry"] = chem
    if "load_name" in df.columns:
        std["batch"] = np.where(std["dataset"].astype(str).eq("SEAAD"), df["load_name"].astype(str), _first(df, ["batch"]))

    # types
    std["projid"] = pd.to_numeric(std.get("projid"), errors="coerce").astype("Int64")

    # final marker
    std["harmonized"] = True
    log("harmonize_obs_from_df: done")
    return std

# ==========
# Persist only /obs back to H5AD
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

    # Scaffold from merged (keeps only dataset/index) and join harmonized
    log("Joining harmonized fields onto merged scaffold …")
    scaffold = m.obs[["dataset"]].copy()
    obs = scaffold.join(harm.drop(columns=["dataset"], errors="ignore"), how="left")

    # Optional: quick summaries
    log(f"Dataset counts: {obs['dataset'].value_counts().to_dict()}")
    log(f"Non-null braak_stage={obs['braak_stage'].notna().sum():,}, cerad_0_3={obs['cerad_score_0_3'].notna().sum():,}")
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
