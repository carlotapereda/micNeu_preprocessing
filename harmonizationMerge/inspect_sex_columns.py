#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, sys, os, time
from datetime import datetime
import numpy as np
import pandas as pd
import anndata as ad

START = time.perf_counter()
def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{ts()}] (+{time.perf_counter()-START:7.2f}s) {msg}", flush=True)

# Heuristic for mapping raw values -> Female/Male/NA
def normalize_sex(series: pd.Series) -> pd.Series:
    s = series.copy()
    # numeric encodings
    s = s.replace({1: "Male", 0: "Female", "1": "Male", "0": "Female"})
    s = s.astype("string").str.strip().str.lower()
    s = s.replace({
        "female":"female", "male":"male",
        "f":"female", "m":"male",
        "woman":"female", "man":"male",
        "female (f)":"female","male (m)":"male",
        "unknown":"na","unk":"na","u":"na","n/a":"na","na":"na", "none":"na", "other":"na"
    })
    # first-letter fallback if still unexpected
    s = s.where(s.isin(["female","male","na"]), s.str[:1].map({"f":"female","m":"male"}))
    # final normalize to title case / NA
    out = s.replace({"female": "Female", "male": "Male", "na": pd.NA})
    return pd.Categorical(out, categories=["Female", "Male"])

def inspect_file(label: str, path: str, pattern: str, outdir: str, topn: int = 20):
    log(f"{label}: opening {path} (backed='r')")
    a = ad.read_h5ad(path, backed="r")
    cols = list(a.obs.columns)

    # find matches (explicit names + regex) — includes 'msex'
    rx = re.compile(pattern, flags=re.I)
    explicit = {"msex", "sex", "gender", "sex at birth", "sex_at_birth"}
    matches = sorted({
        c for c in cols
        if (c.lower() in explicit) or rx.search(c)
    })
    if not matches:
        log(f"{label}: no columns match /{pattern}/ — try loosening the regex")
        a.file.close(); return

    log(f"{label}: found {len(matches)} sex/gender-like columns → {matches}")

    rows = []
    for c in matches:
        s = a.obs[c]
        nonnull = int(s.notna().sum())
        nulls = int(s.isna().sum())
        uniq = int(s.nunique(dropna=True))
        # top raw values
        vc = s.astype("string").str.strip().str.lower().value_counts(dropna=False).head(topn)
        top_raw = "; ".join([f"{k if k is not pd.NA else 'NA'}={int(v):,}" for k, v in vc.items()])
        # normalized proposal
        s_norm = normalize_sex(s)
        vc_norm = pd.Series(s_norm).value_counts(dropna=False)
        top_norm = "; ".join([f"{str(k)}={int(v):,}" for k, v in vc_norm.items()])

        rows.append({
            "dataset": label,
            "column": c,
            "non_null": nonnull,
            "null": nulls,
            "unique_raw": uniq,
            "top_raw_values": top_raw,
            "normalized_counts": top_norm,
        })

        # pretty print to console
        log(f"{label} :: {c}: non-null={nonnull:,}, null={nulls:,}, unique={uniq}")
        print("  raw top:", top_raw)
        print("  norm   :", top_norm)

    # write TSV summary
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_tsv = os.path.join(outdir, f"{label}_sex_columns.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)
    log(f"{label}: wrote summary → {out_tsv}")

    a.file.close()

def main():
    ap = argparse.ArgumentParser(description="Inspect sex/gender columns in per-dataset h5ad files (no matrix load).")
    ap.add_argument("--mit",   default="../celltypist/mit_celltypist_GPU_counts_only.h5ad")
    ap.add_argument("--seaad", default="../celltypist/seaad_celltypist_GPU_counts_only.h5ad")
    ap.add_argument("--fujita",default="../celltypist/fujita_celltypist_GPU_counts_only.h5ad")
    # default regex now includes 'msex'
    ap.add_argument("--regex", default=r"\b(msex|sex|gender|sex[_\s]*at[_\s]*birth)\b",
                    help="Regex used to detect candidate columns (case-insensitive). Also checks explicit names.")
    ap.add_argument("--outdir", default="sex_column_audit")
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    files = {
        "MIT_ROSMAP": args.mit,
        "SEAAD": args.seaad,
        "FUJITA": args.fujita,
    }

    log(f"pattern=/{args.regex}/")
    for label, path in files.items():
        if not os.path.exists(path):
            log(f"{label}: MISSING → {path}")
            continue
        inspect_file(label, path, args.regex, args.outdir, topn=args.topn)

    log("done ✅")

if __name__ == "__main__":
    main()
