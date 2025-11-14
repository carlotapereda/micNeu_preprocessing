#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute UMAP + Leiden with RAPIDS-singlecell on a merged h5ad,
then save UMAP plots.

Usage:
    python compute_umap_rapids.py \
        --h5ad merged_allcells.h5ad \
        --outdir umap_plots

Requires:
    - rapids_singlecell
    - scanpy
    - cupy, rmm
"""

#(scvi-env-clean) python compute_umap_rapids.py \
#  --h5ad merged_allcells.h5ad \
#  --outdir umap_plots


import os
import time
import warnings
import argparse

import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

import scanpy as sc
import anndata as ad
import numpy as np
import rapids_singlecell as rsc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def log(msg: str):
    """Tiny timestamped logger."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def setup_gpus():
    """Configure visible GPUs + RAPIDS memory pool."""
    import scipy
    log(f"SciPy version: {scipy.__version__}")

    # Make all 4 GPUs visible (adjust if you want fewer)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    log(f"GPUs visible: {os.environ['CUDA_VISIBLE_DEVICES']}")
    log(f"Detected GPUs: {cp.cuda.runtime.getDeviceCount()}")

    # RAPIDS memory: pooled + managed for oversubscription safety
    rmm.reinitialize(
        managed_memory=True,     # safer — can oversubscribe and spill to host
        pool_allocator=True,     # enable fast reuse
        initial_pool_size=None,  # let RAPIDS decide
        devices=[0, 1, 2, 3],    # explicitly register these GPUs
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    log("RAPIDS RMM memory pool initialized.")


def run_rapids_pipeline(adata: ad.AnnData, n_pcs: int = 50, n_neighbors: int = 15,
                        resolution: float = 0.5):
    """
    Run PCA → neighbors → UMAP → Leiden on GPU using rapids_singlecell.
    Does NOT assume HVGs (uses all genes).
    """
    log("Sending AnnData to GPU…")
    rsc.get.anndata_to_GPU(adata)
    log("✅ AnnData now backed by GPU arrays.")

    # PCA
    log(f"Running PCA on GPU (n_comps={n_pcs}, use_highly_variable=False)…")
    rsc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=False)
    log("✅ Done PCA.")

    # Neighbors
    log(f"Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs})…")
    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    log("✅ Done neighbors.")

    # UMAP
    log("Running UMAP on GPU…")
    rsc.tl.umap(adata)
    log("✅ Done UMAP.")

    # Leiden
    log(f"Clustering with Leiden (resolution={resolution})…")
    rsc.tl.leiden(adata, resolution=resolution)
    log("✅ Done Leiden.")

    log("🎉 RAPIDS GPU steps complete.")


def make_basic_umap_plots(adata: ad.AnnData, outdir: str):
    """
    Make basic UMAP plots using scanpy:
      - colored by dataset
      - colored by celltypist_simplified (if available)
      - colored by leiden
    """
    os.makedirs(outdir, exist_ok=True)

    # Ensure UMAP is there
    if "X_umap" not in adata.obsm_keys():
        raise RuntimeError("No UMAP embedding found in .obsm['X_umap'].")

    # Global settings
    sc.settings.figdir = outdir
    sc.settings.autoshow = False

    # 1) UMAP by dataset
    if "dataset" in adata.obs.columns:
        log("Plotting UMAP colored by 'dataset'…")
        sc.pl.umap(
            adata,
            color="dataset",
            save="_dataset.png",
            show=False,
        )
    else:
        log("⚠️ 'dataset' column not found in adata.obs. Skipping dataset plot.")


    # 3) UMAP by Leiden
    if "leiden" in adata.obs.columns:
        log("Plotting UMAP colored by 'leiden'…")
        sc.pl.umap(
            adata,
            color="leiden",
            save="_leiden.png",
            show=False,
        )
    else:
        log("⚠️ 'leiden' clusters not found. Skipping Leiden plot.")

    log(f"✅ UMAP plots saved in: {outdir}")


def main():
    warnings.filterwarnings("ignore")

    p = argparse.ArgumentParser(
        description="Compute UMAP+Leiden on GPU with RAPIDS-singlecell and plot UMAP."
    )
    p.add_argument("--h5ad", default="merged_allcells.h5ad",
                   help="Input merged AnnData file (.h5ad).")
    p.add_argument("--outdir", default="umap_plots",
                   help="Output directory for plots and updated h5ad.")
    p.add_argument("--no-save-h5ad", action="store_true",
                   help="Do not write updated h5ad with UMAP+Leiden.")
    p.add_argument("--npcs", type=int, default=50,
                   help="Number of PCs for PCA and neighbors (default: 50).")
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="Number of neighbors for knn graph (default: 15).")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Leiden resolution (default: 0.5).")

    args = p.parse_args()

    start = time.time()

    log("========== GPU UMAP + LEIDEN PIPELINE ==========")
    setup_gpus()

    # ---------- Load AnnData ----------
    log(f"Loading AnnData from: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    log(f"Loaded AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # You said you don't want HVGs, so we keep the full object.
    # If later you want HVGs, you can filter here.

    # ---------- RAPIDS pipeline ----------
    run_rapids_pipeline(
        adata,
        n_pcs=args.npcs,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
    )

    # ---------- Save updated h5ad ----------
    os.makedirs(args.outdir, exist_ok=True)
    if not args.no_save_h5ad:
        out_h5ad = os.path.join(
            args.outdir,
            os.path.basename(args.h5ad).replace(".h5ad", "_with_umap_leiden.h5ad"),
        )
        log(f"Writing updated AnnData with UMAP+Leiden → {out_h5ad}")
        adata.write(out_h5ad)
        log("✅ Saved updated h5ad.")

    # ---------- UMAP plots ----------
    make_basic_umap_plots(adata, args.outdir)

    elapsed = time.time() - start
    log(f"🎯 Done. Total runtime: {elapsed/60:.1f} min")
    log("===============================================")


if __name__ == "__main__":
    main()
