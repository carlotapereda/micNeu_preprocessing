#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, sys, os, time
from datetime import datetime
import numpy as np
import pandas as pd
import anndata as ad

# ============================================================
# Logging helpers
# ============================================================
START = time.perf_counter()
def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{ts()}] (+{time.perf_counter()-START:7.2f}s) {msg}", flush=True)

# ============================================================
# APOE normalization helpers — you already use these in merge
# ============================================================
def apoe_to_std(x):
    """
    Normalize APOE genotypes into canonical format: E2/E2, E2/E3, E3/E4, E4/E4, etc.
    Accepts many messy formats like '34', '3/4', 'E3E4', 'e3/e4', 'APOE 3/4', etc.
    Returns pd.NA if parsing fails.
    """
    if pd.isna(x): return pd.NA
    s = str(x).strip().upper().replace(" ", "")
    s = s.replace("APOE", "").replace("-", "").replace("_", "")
    s = s.replace("E", "")

    # extract digits
    nums = re.findall(r"[2-4]", s)
    if len(nums) != 2:
        return pd.NA
    return f"E{nums[0]}/E{nums[1]}"

def apoe_e4_dosage(std):
    """Return number of E4 alleles from normalized APOE genotype."""
    if pd.isna(std): return pd.NA
    return std.count("4")

# ============================================================
# Inspect APOE genotype columns
# ============================================================
def inspect_file(label: str, path: str, pattern: str, outdir: str, topn: int = 20):
    log(f"{label}: opening {path} (backed='r')")
    a = ad.read_h5ad(path, backed="r")
    cols = list(a.obs.columns)

    rx = re.compile(pattern, flags=re.I)

    explicit = {
        "apoe", "apoe4", "apoe_status",
        "apoe genotype", "apoe_genotype",
        "apoe-genotype", "apoe-alleles",
        "apoe_geno"
    }

    matches = sorted({
        c for c in cols
        if (c.lower() in explicit) or rx.search(c)
    })

    if not matches:
        log(f"{label}: no APOE columns match /{pattern}/")
        a.file.close()
        return

    log(f"{label}: found {len(matches)} APOE-like columns → {matches}")

    rows = []
    for c in matches:
        s = a.obs[c]

        nonnull = int(s.notna().sum())
        nulls   = int(s.isna().sum())
        uniq    = int(s.nunique(dropna=True))

        # raw values
        vc = (
            s.astype("string")
             .str.strip()
             .str.upper()
             .value_counts(dropna=False)
             .head(topn)
        )
        top_raw = "; ".join([f"{k if k is not pd.NA else 'NA'}={int(v):,}" for k,v in vc.items()])

        # normalized canonical APOE genotype
        std = s.apply(apoe_to_std)
        vc_std = std.value_counts(dropna=False)
        top_std = "; ".join([f"{str(k)}={int(v):,}" for k,v in vc_std.items()])

        # E4 dosage
        e4 = std.apply(apoe_e4_dosage)
        vc_e4 = e4.value_counts(dropna=False)
        top_e4 = "; ".join([f"{str(k)}={int(v):,}" for k,v in vc_e4.items()])

        rows.append({
            "dataset": label,
            "column": c,
            "non_null": nonnull,
            "null": nulls,
            "unique": uniq,
            "top_raw_values": top_raw,
            "normalized_apoe": top_std,
            "e4_dosage_counts": top_e4,
        })

        log(f"{label} :: {c}: non-null={nonnull:,}, null={nulls:,}, unique={uniq}")
        print("  raw:", top_raw)
        print("  std:", top_std)
        print("  e4 :", top_e4)

    # save TSV
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_tsv = os.path.join(outdir, f"{label}_apoe_columns.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)
    log(f"{label}: wrote summary → {out_tsv}")

    a.file.close()

# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Inspect APOE genotype columns in per-dataset h5ad files (no matrix load).")

    ap.add_argument("--mit",    default="../celltypist/mit_celltypist_GPU_counts_only.h5ad")
    ap.add_argument("--seaad",  default="../celltypist/seaad_celltypist_GPU_counts_only.h5ad")
    ap.add_argument("--fujita", default="../celltypist/fujita_celltypist_GPU_counts_only.h5ad")

    # REGEX now searches for APOE variants
    ap.add_argument(
        "--regex",
        default=r"\b(apoe|apoe[_\s-]*genotype|apoe[_\s-]*status|apoe4)\b",
        help="Regex used to detect APOE genotype columns."
    )
    ap.add_argument("--outdir", default="apoe_column_audit")
    ap.add_argument("--topn", type=int, default=20)

    args = ap.parse_args()

    files = {
        "MIT_ROSMAP": args.mit,
        "SEAAD":      args.seaad,
        "FUJITA":     args.fujita,
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
