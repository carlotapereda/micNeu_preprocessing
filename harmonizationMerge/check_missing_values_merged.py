#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check missing values per column and per dataset in an AnnData object.
Usage:
    python check_missing_values_merged.py --h5ad merged_allcells.h5ad
"""

import argparse
import pandas as pd
import numpy as np
import anndata as ad

def is_missing(x):
    """Custom missing-value logic: NaN, None, empty string."""
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

def main():
    p = argparse.ArgumentParser(description="Check missing values per dataset in adata.obs.")
    p.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    args = p.parse_args()

    print(f"\nLoading AnnData: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad, backed="r")

    if "dataset" not in adata.obs.columns:
        raise ValueError("❌ The obs column 'dataset' is required for grouping but not found.")

    df = adata.obs.copy()

    # Convert all columns to object/dtype where missing checks work properly
    df = df.replace({None: np.nan})  # unify None → NaN

    # Track columns with missing values
    missing_summary = {}

    print("\n================= Checking columns for missing values =================\n")

    for col in df.columns:
        # vectorized missing check
        missing_mask = df[col].apply(is_missing).astype(bool)
        total_missing = missing_mask.sum()


        if total_missing == 0:
            continue  # skip non-missing columns

        # Missing values per dataset
        by_dataset = (
            df.groupby("dataset")[col]
            .apply(lambda x: x.apply(is_missing).astype(bool).sum())
            .rename("missing_count")
        )

        missing_summary[col] = by_dataset

        print(f"\n--- Column: {col} ---")
        print(f"Total missing: {total_missing:,}")
        print(by_dataset[by_dataset > 0].to_string())
        print("--------------------------------------")

    print("\n================= FINAL SUMMARY =================\n")

    if not missing_summary:
        print("🎉 No missing values in any columns!")
    else:
        print("Columns with missing values:")
        for col, summary in missing_summary.items():
            print(f"• {col}: {summary.sum():,} missing across all datasets")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
