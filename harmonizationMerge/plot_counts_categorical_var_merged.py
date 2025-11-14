#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Example:
# python plot_counts_categorical_var_merged.py \
#   --h5ad merged_allcells.h5ad \
#   --column celltypist_cell_label \
#   --outdir plots_counts

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
from pathlib import Path


def thousand(n):
    try:
        return f"{int(n):,}"
    except:
        return str(n)


def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.annotate(
            thousand(h),
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 2),
            textcoords="offset points",
        )


def plot_counts_by_column(adata, column, out_png):
    if column not in adata.obs.columns:
        raise RuntimeError(
            f"Column '{column}' not found in adata.obs. "
            f"Available columns:\n{list(adata.obs.columns)}"
        )

    print(f"📊 Counting values for column: {column}")

    values = adata.obs[column].astype("string").fillna("NA")

    vc = values.value_counts().sort_values(ascending=False)
    labels = vc.index.tolist()
    vals = vc.to_numpy()

    plt.figure(figsize=(max(10, len(labels) * 0.5), 6))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Cell count")
    plt.title(f"Counts per '{column}'")
    annotate_bars(plt.gca())
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✔ saved {out_png}")


def main():
    p = argparse.ArgumentParser(description="Bar plot of counts for any obs column in merged h5ad")
    p.add_argument("--h5ad", default="merged_allcells.h5ad")
    p.add_argument("--column", required=True, help="obs column name to count")
    p.add_argument("--outdir", default="plots_counts")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"📂 Opening {args.h5ad} (backed='r')")
    adata = ad.read_h5ad(args.h5ad, backed="r")

    outfile = Path(args.outdir) / f"counts_by_{args.column}.png"

    plot_counts_by_column(adata, args.column, outfile.as_posix())

    if hasattr(adata, "file") and hasattr(adata.file, "close"):
        adata.file.close()


if __name__ == "__main__":
    main()
