#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Imports
# ============================================================
# Standard library
import os, sys, time, re, h5py
from datetime import datetime

# Scientific / data stack
import numpy as np
import pandas as pd
import anndata as ad

# ============================================================
# Config
# ============================================================
# Output merged H5AD file
MERGED_OUT   = "merged_allcells.h5ad"

# Separator used by concat_on_disk to make obs_names unique
# (must match index_unique below)
INDEX_UNIQUE = "-"

# Input datasets (CellTypist-annotated, counts-only)
FILES = {
    "MIT_ROSMAP": "../celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEAAD":      "../celltypist/seaad_celltypist_GPU_counts_only.h5ad",
    "FUJITA":     "../celltypist/fujita_celltypist_GPU_counts_only.h5ad",
}

# ============================================================
# Logging utilities
# ============================================================
# Wall-clock timer start
START = time.perf_counter()

# Human-readable timestamp
def _now_ts(): 
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Try to enable memory usage reporting (RSS)
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def memline(): 
        return f" | RSS {_PROC.memory_info().rss/(1024**3):.2f} GB"
except Exception:
    # Fallback if psutil is unavailable
    def memline(): 
        return ""

# Unified logger with elapsed time + optional memory
def log(msg):
    print(
        f"[{_now_ts()}] (+{time.perf_counter()-START:7.2f}s) "
        f"{msg}{memline()}",
        flush=True
    )

# ============================================================
# Source .obs columns we may want to preserve
# (union across datasets; harmonization happens later)
# ============================================================
NEEDED_OBS_COLS = [
    # Demographics / pathology
    "Age at Death","age_death","braaksc","Braak","ceradsc","CERAD score",
    "Sex","sex","msex","Gender","pmi","PMI","educ","Years of education","race",
    "spanish","Hispanic_Latino","species","Organism","Brain Region",

    # SEAAD race checkboxes
    "Race (choice=White)","Race (choice=Black_ African American)","Race (choice=Asian)",
    "Race (choice=American Indian_ Alaska Native)",
    "Race (choice=Native Hawaiian or Pacific Islander)",
    "Race (choice=Unknown or unreported)","Race (choice=Other)",
    "specify other race",

    # Study / donor identifiers
    "projid","Study","Primary Study Name","Donor ID","sample_id",
    "individualID","individualID_x","individualID_y","subject",

    # CellTypist outputs (passed through verbatim later)
    "celltypist_cell_label","celltypist_simplified","celltypist_conf_score",
    "Class","Subclass","Supertype",
    "Class confidence","Subclass confidence","Supertype confidence",

    # QC metrics
    "n_genes_by_counts","log1p_n_genes_by_counts","total_counts","log1p_total_counts",
    "pct_counts_in_top_20_genes","total_counts_mt","log1p_total_counts_mt","pct_counts_mt",
    "total_counts_ribo","log1p_total_counts_ribo","pct_counts_ribo",
    "total_counts_hb","log1p_total_counts_hb","pct_counts_hb",
    "n_genes",
    "Doublet score","doublet_scores","predicted_doublets",
    "doublet_label","outlier","mt_outlier",

    # Batch / chemistry / APOE
    "load_name","method","apoe_genotype","APOE Genotype",

    # Cognition
    "cogdx","Cognitive Status","cts_mmse30_lv",
    "Last MMSE Score","Last MOCA Score","Last CASI Score",
]

# ============================================================
# Utility helpers
# ============================================================

def clean_age_column(series: pd.Series) -> pd.Series:
    """
    Normalize age strings like '85+', '90 yrs', etc.
    Keeps digits and decimal point, coerces invalid values to NaN.
    """
    if series is None:
        return pd.Series(index=pd.RangeIndex(0), dtype="float64")
    s = series.astype(str)
    s = s.str.replace(r'\+', '', regex=True)
    s = s.str.replace(r'[^\d\.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

def _first(df, candidates):
    """
    Return the first column (by priority order) that exists in df.
    """
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(index=df.index, dtype="object")

# ============================================================
# Sex normalization (dataset-aware)
# ============================================================

def normalize_sex_values(series: pd.Series) -> pd.Categorical:
    """
    Normalize sex labels across datasets into:
      - Female
      - Male
      - NA
    """
    s = series.copy()

    # Numeric encodings
    s = s.replace({1: "Male", 0: "Female", "1": "Male", "0": "Female"})

    # Normalize strings
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

    # Fallback: infer from first letter if possible
    s = s.where(
        s.isin(["female", "male", "na"]),
        s.str[:1].map({"f": "female", "m": "male"})
    )

    out = s.replace({"female": "Female", "male": "Male", "na": pd.NA})
    return pd.Categorical(out, categories=["Female", "Male"])

def build_sex_raw(df: pd.DataFrame) -> pd.Series:
    """
    Dataset-specific priority rules for sex column selection.
    """
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    if "dataset" not in df.columns:
        return out

    ds = df["dataset"].astype(str)

    # MIT-ROSMAP
    mask = ds.eq("MIT_ROSMAP")
    if "msex" in df.columns:
        out.loc[mask] = df.loc[mask, "msex"]
    elif "sex" in df.columns:
        out.loc[mask] = df.loc[mask, "sex"]

    # SEAAD
    mask = ds.eq("SEAAD")
    if "Gender" in df.columns:
        out.loc[mask] = df.loc[mask, "Gender"]
    elif "Sex" in df.columns:
        out.loc[mask] = df.loc[mask, "Sex"]

    # FUJITA
    mask = ds.eq("FUJITA")
    if "msex" in df.columns:
        out.loc[mask] = df.loc[mask, "msex"]

    return out

# ============================================================
# APOE parsing helpers
# ============================================================

# Regex for APOE allele pairs (e.g., E3/E4, ε4-ε4)
_APOE_PAIR_RE = re.compile(r"[eεE]?\s*([234])\s*[/\-\|\s]?\s*[eεE]?\s*([234])")

def apoe_to_std(x):
    """
    Convert heterogeneous APOE genotype encodings into a canonical form:
      - 'E2/E3', 'E3/E4', 'E4/E4', etc.

    Handles inputs like:
      - 'E3/E4', 'ε3/ε4', '3/4', '3-4', '3|4'
      - integers or floats (e.g. 34, 44)
      - mixed or poorly formatted strings
    """

    # ------------------------------------------------------------
    # 1. Missing values → missing output
    # ------------------------------------------------------------
    # If the value is NaN / None / pandas NA, propagate as NaN
    if pd.isna(x):
        return np.nan

    # ------------------------------------------------------------
    # 2. Normalize input into a clean string
    # ------------------------------------------------------------
    # If the input is numeric (int, float, numpy scalar),
    # convert to int first to avoid '34.0' → '34.0'
    # Example: 34.0 → '34'
    s = (
        str(int(x)) if isinstance(x, (int, float, np.integer, np.floating))
        else str(x)
    )

    # ------------------------------------------------------------
    # 3. Canonicalize common APOE formatting variants
    # ------------------------------------------------------------
    # - Strip whitespace
    # - Replace Greek epsilon (ε) with 'e'
    # - Normalize case ('E' → 'e')
    # - Normalize separators ('-', '|', whitespace → '/')
    #
    # Examples:
    #   'ε3-ε4' → 'e3/e4'
    #   'E4|E4' → 'e4/e4'
    #   ' 3 4 ' → '3/4'
    s = (
        s.strip()
         .replace("ε", "e")
         .replace("E", "e")
         .replace("-", "/")
         .replace("|", "/")
    )

    # ------------------------------------------------------------
    # 4. Regex-based extraction of allele pairs
    # ------------------------------------------------------------
    # _APOE_PAIR_RE matches two APOE alleles (2, 3, or 4),
    # optionally prefixed with 'e' and separated by '/', '-', '|', or space.
    #
    # Example matches:
    #   'e3/e4' → ('3', '4')
    #   '3/4'   → ('3', '4')
    #   'e4e4'  → ('4', '4')
    m = _APOE_PAIR_RE.search(s)
    if m:
        # Sort alleles so representation is order-invariant:
        #   'E4/E3' → 'E3/E4'
        a, b = sorted(m.groups())
        return f"E{a}/E{b}"

    # ------------------------------------------------------------
    # 5. Fallback: extract allele digits manually
    # ------------------------------------------------------------
    # If regex fails, scan the string for any APOE allele digits.
    # This handles cases like:
    #   'APOE34'
    #   'genotype: 3,4'
    #   'E3E4'
    digits = [ch for ch in s if ch in "234"]

    # If at least two allele digits are found, use the first two
    # and enforce canonical ordering.
    if len(digits) >= 2:
        a, b = sorted(digits[:2])
        return f"E{a}/E{b}"

    # ------------------------------------------------------------
    # 6. If all parsing attempts fail, return missing
    # ------------------------------------------------------------
    return np.nan

def apoe_e4_dosage(apoe_std):
    """
    Count number of E4 alleles (0, 1, or 2).
    """
    if pd.isna(apoe_std):
        return np.nan
    l, r = apoe_std.replace("E", "").split("/")
    return int(l == "4") + int(r == "4")

# ============================================================
# Braak / CERAD normalization
# ============================================================

# ============================================================
# Braak stage helpers
# ============================================================

# Mapping from Roman numerals → integer Braak stage
# Used when datasets encode Braak as 'I', 'II', ..., 'VI'
_ROMAN_TO_INT = {
    "I":   1,
    "II":  2,
    "III": 3,
    "IV":  4,
    "V":   5,
    "VI":  6
}

# Inverse mapping from integer → Roman numeral
# Built programmatically to guarantee consistency with _ROMAN_TO_INT
_INT_TO_ROMAN = {v: k for k, v in _ROMAN_TO_INT.items()}


def parse_braak(x):
    """
    Parse Braak stage into a normalized integer in [0, 6].

    Handles:
      - Numeric values (0–6, floats, strings of numbers)
      - Roman numerals ('I'–'VI')
      - Strings like 'Braak III', 'BRAAK IV', 'stage V'
      - Mixed or partially corrupted inputs

    Returns:
      - Integer Braak stage (0–6)
      - np.nan if parsing fails
    """

    # ------------------------------------------------------------
    # 1. Missing input → missing output
    # ------------------------------------------------------------
    # Propagate NaN / None / pandas NA
    if pd.isna(x):
        return np.nan

    # ------------------------------------------------------------
    # 2. Fast path: numeric input
    # ------------------------------------------------------------
    # Covers:
    #   - ints (3)
    #   - floats (3.0, 2.7)
    #   - numpy scalar types
    #
    # Strategy:
    #   - Convert to float → round → int
    #   - Clip into valid Braak range [0, 6]
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return int(
                np.clip(
                    int(round(float(x))),
                    0,
                    6
                )
            )
        except Exception:
            # Any weird numeric conversion failure → missing
            return np.nan

    # ------------------------------------------------------------
    # 3. Normalize string representation
    # ------------------------------------------------------------
    # Convert to string, strip whitespace, uppercase for consistency,
    # and remove the literal word 'BRAAK' if present.
    #
    # Examples:
    #   'Braak III' → 'III'
    #   'braak iv'  → 'IV'
    #   ' Stage V ' → 'STAGE V'
    s = (
        str(x)
        .strip()
        .upper()
        .replace("BRAAK", "")
        .strip()
    )

    # ------------------------------------------------------------
    # 4. Exact Roman numeral match
    # ------------------------------------------------------------
    # If the cleaned string is exactly 'I'–'VI', use direct lookup.
    if s in _ROMAN_TO_INT:
        return _ROMAN_TO_INT[s]

    # ------------------------------------------------------------
    # 5. Attempt numeric parsing from string
    # ------------------------------------------------------------
    # Handles cases like:
    #   '3'
    #   '3.0'
    #   ' 4 '
    try:
        return int(
            np.clip(
                int(float(s)),
                0,
                6
            )
        )

    # ------------------------------------------------------------
    # 6. Fallback: Roman numeral embedded in longer string
    # ------------------------------------------------------------
    # Handles cases like:
    #   'STAGE IV'
    #   'BRAAK V (AD)'
    #   'III/IV'
    except Exception:
        for r, v in _ROMAN_TO_INT.items():
            if r in s:
                return v

    # ------------------------------------------------------------
    # 7. If all parsing strategies fail → missing
    # ------------------------------------------------------------
    return np.nan


def braak_label_from_stage(stage):
    """
    Convert a numeric Braak stage into a human-readable label.

    Examples:
      0 → 'Braak 0'
      1 → 'Braak I'
      6 → 'Braak VI'
    """

    # Missing stage → missing label
    if pd.isna(stage):
        return np.nan

    # Ensure integer (handles pandas nullable ints)
    stage = int(stage)

    # Special-case stage 0 (no Roman numeral)
    if stage == 0:
        return "Braak 0"

    # Use Roman numeral if known, else fall back to raw number
    return f"Braak {_INT_TO_ROMAN.get(stage, str(stage))}"


# ============================================================
# CERAD helpers
# ============================================================

# Canonical CERAD 1–4 scale → descriptive labels
_CERAD_14_LABEL = {
    1: "Absent",
    2: "Sparse",
    3: "Moderate",
    4: "Frequent"
}

# Inverse mapping for string → numeric lookup
# (uppercased for case-insensitive matching)
_CERAD_LABEL_14 = {
    v.upper(): k for k, v in _CERAD_14_LABEL.items()
}


def to_cerad_1_4(x):
    """
    Normalize CERAD scores into the canonical 1–4 scale.

    Handles:
      - Numeric encodings in either 0–3 or 1–4
      - String labels ('Absent', 'Sparse', etc.)
      - Mixed or loosely formatted inputs

    Returns:
      - Integer CERAD score in [1, 4]
      - np.nan if parsing fails
    """

    # ------------------------------------------------------------
    # 1. Missing input → missing output
    # ------------------------------------------------------------
    if pd.isna(x):
        return np.nan

    # ------------------------------------------------------------
    # 2. Numeric handling
    # ------------------------------------------------------------
    # Covers:
    #   - Proper 1–4 scale (return as-is)
    #   - Legacy 0–3 scale (shift by +1)
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            v = int(round(float(x)))

            # Already 1–4
            if v in (1, 2, 3, 4):
                return v

            # Legacy 0–3 → convert to 1–4
            if v in (0, 1, 2, 3):
                return v + 1

        except Exception:
            return np.nan

    # ------------------------------------------------------------
    # 3. String label handling
    # ------------------------------------------------------------
    # Matches:
    #   'Absent', 'SPARSE', 'moderate', etc.
    return _CERAD_LABEL_14.get(
        str(x).strip().upper(),
        np.nan
    )


def cerad_14_to_03(v):
    """
    Convert CERAD score from 1–4 scale back to 0–3 scale.
    """
    return np.nan if pd.isna(v) else int(v) - 1


def cerad_label_from_14(v):
    """
    Convert CERAD 1–4 score to human-readable label.

    Examples:
      1 → 'Absent'
      4 → 'Frequent'
    """
    return (
        np.nan
        if pd.isna(v)
        else _CERAD_14_LABEL.get(int(v), np.nan)
    )

# ============================================================
# SEAAD race checkbox parsing
# ============================================================

def derive_race_from_seaad(df: pd.DataFrame) -> pd.Series:
    """
    Decode SEAAD race from checkbox-style columns.
    """
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

# ============================================================
# Canonical QC columns to carry forward
# ============================================================
CANONICAL_QC = [
    "n_genes_by_counts","log1p_n_genes_by_counts","total_counts","log1p_total_counts",
    "pct_counts_in_top_20_genes","total_counts_mt","log1p_total_counts_mt","pct_counts_mt",
    "total_counts_ribo","log1p_total_counts_ribo","pct_counts_ribo",
    "total_counts_hb","log1p_total_counts_hb","pct_counts_hb",
    "n_genes","doublet_scores","predicted_doublets",
    "doublet_label","outlier","mt_outlier"
]

# ============================================================
# Build base .obs aligned to merged index
# (no expression matrix loaded)
# ============================================================
def build_obs_union_from_sources(files_map, merged_index, merged_dataset, index_unique="-"):
    """
    For each dataset:
      - Read only .obs from the source H5AD (backed mode)
      - Map original obs_names → merged obs_names
      - Extract relevant metadata columns
    """
    parts = []
    merged_index = pd.Index(merged_index)

    # Regex helpers to discover variant column names
    sex_re = re.compile(r"\b(msex|sex|gender|sex[_\s]*at[_\s]*birth)\b", flags=re.I)
    braak_aliases = {"braak", "braak stage", "braak_stage"}
    cerad_aliases = {"cerad", "cerad score", "cerad_score"}

    for ds_key, path in files_map.items():
        sel = (merged_dataset == ds_key)
        n_sel = int(sel.sum())
        if not n_sel:
            log(f"{ds_key}: no rows in merged; skip")
            continue

        # Recover original obs_names by stripping concat suffix
        suffix = index_unique + ds_key
        merged_ids = merged_index[sel]
        orig_ids   = merged_ids.str.rsplit(suffix, n=1).str[0]

        log(f"{ds_key}: loading source obs for {n_sel:,} rows …")

        src = ad.read_h5ad(path, backed="r")
        src_cols = list(src.obs.columns)

        # Dynamically discover columns we may need
        dynamic_sex   = [c for c in src_cols if sex_re.search(c)]
        dynamic_braak = [c for c in src_cols if c.lower() in braak_aliases]
        dynamic_cerad = [c for c in src_cols if c.lower() in cerad_aliases]
        dynamic_apoe  = [c for c in src_cols if re.search(r"apoe", c, flags=re.I)]

        # Final column set to extract
        keep = [c for c in NEEDED_OBS_COLS if c in src_cols]
        keep = sorted(set(keep + dynamic_sex + dynamic_braak + dynamic_cerad + dynamic_apoe))

        # Reindex onto merged obs_names
        so = src.obs[keep].reindex(orig_ids)

        # Close file handle early
        if hasattr(src, "file") and hasattr(src.file, "close"):
            src.file.close()
        del src

        so.index = merged_ids
        so["dataset"] = ds_key
        parts.append(so)

        # Quick sanity logging
        bnn = int(so.get("braaksc", pd.Series(index=so.index, dtype="object")).notna().sum()) \
            + int(so.get("Braak",   pd.Series(index=so.index, dtype="object")).notna().sum())
        cnn = int(so.get("ceradsc", pd.Series(index=so.index, dtype="object")).notna().sum()) \
            + int(so.get("CERAD score", pd.Series(index=so.index, dtype="object")).notna().sum())
        sx  = dynamic_sex if dynamic_sex else "none"

        log(f"{ds_key}: base_obs non-null — Braak {bnn:,}, CERAD {cnn:,}; sex-like cols: {sx}")

    if not parts:
        return pd
