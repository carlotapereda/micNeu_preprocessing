#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Example:
# python plot_boxplot_from_merged.py \
#   --h5ad merged_allcells.h5ad \
#   --num-col pct_counts_mt \
#   --cat-col celltypist_simplified \
#   --outdir plots_boxplots

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
from pathlib import Path


def boxplot_numeric_by_category(adata, num_col, cat_col, out_png):
    # Safety checks
    if num_col not in adata.obs.columns:
        raise RuntimeError(f"Numerical column '{num_col}' not found in obs.")
    if cat_col not in adata.obs.columns:
        raise RuntimeError(f"Categorical column '{cat_col}' not found in obs.")

    print(f"📊 Loading columns: {num_col} (numeric), {cat_col} (category)")

    df = adata.obs[[num_col, cat_col]].copy()

    # Ensure proper types
    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    df[cat_col] = df[cat_col].astype("string").fillna("NA")

    # Drop NaN rows for numeric
    df = df.dropna(subset=[num_col])

    # Sort categories by median value for nice plotting
    medians = df.groupby(cat_col)[num_col].median().sort_values()
    order = medians.index.tolist()

    plt.figure(figsize=(max(10, len(order) * 0.6), 6))
    df.boxplot(column=num_col, by=cat_col, positions=range(len(order)), vert=True)

    plt.xticks(range(len(order)), order, rotation=45, ha="right")
    plt.ylabel(num_col)
    plt.title(f"{num_col} by {cat_col}")
    plt.suptitle("")  # remove "Boxplot grouped by ..." default
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"✔ saved {out_png}")


def main():
    p = argparse.ArgumentParser(description="Boxplot of a numeric obs column stratified by a categorical obs column.")
    p.add_argument("--h5ad", default="merged_allcells.h5ad")
    p.add_argument("--num-col", required=True, help="Name of the numerical column in .obs")
    p.add_argument("--cat-col", required=True, help="Name of the categorical column in .obs")
    p.add_argument("--outdir", default="plots_boxplots")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"📂 Opening {args.h5ad} (backed='r')")
    adata = ad.read_h5ad(args.h5ad, backed="r")

    outfile = Path(args.outdir) / f"boxplot_{args.num_col}_by_{args.cat_col}.png"

    boxplot_numeric_by_category(
        adata,
        args.num_col,
        args.cat_col,
        outfile.as_posix()
    )

    if hasattr(adata, "file") and hasattr(adata.file, "close"):
        adata.file.close()


if __name__ == "__main__":
    main()
