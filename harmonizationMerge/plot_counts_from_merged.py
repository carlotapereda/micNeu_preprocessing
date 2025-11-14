#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python plot_counts_from_merged.py --h5ad merged_allcells.h5ad --outdir plots_counts --top-n-labels 40


import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
from pathlib import Path

def thousand(n):
    try: return f"{int(n):,}"
    except: return str(n)

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h): continue
        ax.annotate(thousand(h),
                    (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")

def counts_apoe_by_sex(adata, out_png):
    needed = {"apoe_genotype_std", "sex"}
    missing = [c for c in needed if c not in adata.obs.columns]
    if missing:
        raise RuntimeError(f"Missing columns in merged .obs: {missing}. Run the harmonization script first.")

    obs = adata.obs[list(needed)].copy()
    obs["apoe_genotype_std"] = obs["apoe_genotype_std"].astype("string").fillna("NA").astype("category")
    obs["sex"] = obs["sex"].astype("string").fillna("NA").astype("category")

    preferred = ["E2/E2","E2/E3","E3/E3","E2/E4","E3/E4","E4/E4","NA"]
    apoe = obs["apoe_genotype_std"].cat.categories.tolist()
    apoe = [c for c in preferred if c in apoe] + [c for c in apoe if c not in preferred]
    obs["apoe_genotype_std"] = obs["apoe_genotype_std"].cat.set_categories(apoe, ordered=True)

    counts = obs.groupby(["apoe_genotype_std","sex"]).size().unstack(fill_value=0).sort_index()
    sexes = list(counts.columns)
    x = np.arange(len(counts))
    width = 0.8 / max(len(sexes), 1)

    plt.figure(figsize=(max(8, len(counts)*0.9), 6))
    for i, s in enumerate(sexes):
        vals = counts[s].to_numpy()
        plt.bar(x + i*width - (len(sexes)-1)*width/2, vals, width=width, label=f"{s} ({thousand(vals.sum())})")
    plt.xticks(x, counts.index.tolist(), rotation=45, ha="right")
    plt.ylabel("Cell count")
    plt.title("Cells by APOE genotype × Sex")
    plt.legend(frameon=False)
    annotate_bars(plt.gca())
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✔ saved {out_png}")

def counts_by_celltypist_label(adata, out_png, top_n=None):
    col = "celltypist_cell_label"
    if col not in adata.obs.columns:
        raise RuntimeError(f"Missing column '{col}' in merged .obs. Run the harmonization step that writes it.")

    lab = adata.obs[col].astype("string").fillna("NA")
    vc = lab.value_counts().sort_values(ascending=False)
    if top_n is not None:
        vc = vc.head(int(top_n))

    labels = vc.index.tolist()
    vals = vc.to_numpy()

    plt.figure(figsize=(max(10, len(labels)*0.5), 6))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Cell count")
    plt.title("Cells by Celltypist label")
    annotate_bars(plt.gca())
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✔ saved {out_png}")

def main():
    p = argparse.ArgumentParser(description="Bar plots from merged h5ad.")
    p.add_argument("--h5ad", default="merged_allcells.h5ad")
    p.add_argument("--outdir", default="plots_counts")
    p.add_argument("--top-n-labels", type=int, default=None, help="Limit #bars for celltypist labels")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.h5ad, backed="r")

    # counts by APOE x sex
    counts_apoe_by_sex(adata, (Path(args.outdir) / "cells_by_apoe_sex.png").as_posix())

    # counts by celltypist label
    counts_by_celltypist_label(adata, (Path(args.outdir) / "cells_by_celltypist_label.png").as_posix(),
                               top_n=args.top_n_labels)

    if hasattr(adata, "file") and hasattr(adata.file, "close"):
        adata.file.close()

if __name__ == "__main__":
    main()
