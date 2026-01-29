#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python plot_umap_per_dataset.py --h5ad merged_allcells.h5ad --outdir plots_umap --sample-n 500000

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
from pathlib import Path

def ensure_umap(adata):
    if "X_umap" not in adata.obsm_keys():
        raise RuntimeError("No `.obsm['X_umap']` found. Compute UMAP and save it in the merged file first.")
    return adata.obsm["X_umap"]

def thousand(n):
    try: return f"{int(n):,}"
    except: return str(n)

def main():
    p = argparse.ArgumentParser(description="Plot UMAP per dataset from merged h5ad.")
    p.add_argument("--h5ad", default="merged_allcells.h5ad")
    p.add_argument("--outdir", default="plots_umap")
    p.add_argument("--sample-n", type=int, default=500_000, help="Max points per dataset (None=all)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--point-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    if "dataset" not in adata.obs.columns:
        raise RuntimeError("merged .obs is missing 'dataset'. Re-run concat_on_disk with label='dataset'.")

    X_umap = ensure_umap(adata)
    n = X_umap.shape[0]
    datasets = adata.obs["dataset"].astype("category")

    # stable color map across datasets (even though we plot one dataset at a time)
    cats = list(datasets.cat.categories)
    palette = plt.cm.tab10.colors
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

    rng = np.random.default_rng(args.seed)

    for ds in cats:
        mask = (datasets == ds).to_numpy()
        idx_all = np.nonzero(mask)[0]
        if idx_all.size == 0:
            continue

        if args.sample_n is not None and args.sample_n < idx_all.size:
            idx = rng.choice(idx_all, size=args.sample_n, replace=False)
        else:
            idx = idx_all

        umap = np.asarray(X_umap[idx, :])
        c = color_map[ds]

        plt.figure(figsize=(9, 9))
        plt.scatter(umap[:, 0], umap[:, 1],
                    s=args.point_size, c=[c], alpha=args.alpha,
                    linewidths=0, rasterized=True)
        plt.title(f"UMAP — {ds}  (n={thousand(idx.size)} of {thousand(idx_all.size)})")
        plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
        plt.tight_layout()
        out = Path(args.outdir) / f"umap_{ds}.png"
        plt.savefig(out.as_posix(), dpi=300)
        plt.close()
        print(f"✔ saved {out}")

    # close handle
    if hasattr(adata, "file") and hasattr(adata.file, "close"):
        adata.file.close()

if __name__ == "__main__":
    main()
