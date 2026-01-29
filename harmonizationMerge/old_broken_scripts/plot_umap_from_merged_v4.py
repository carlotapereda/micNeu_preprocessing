#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute HVGs on CPU, run PCA on CPU, then run RAPIDS neighbors/UMAP/Leiden
on HVG-only AnnData, copy embeddings/clusters back to the full object,
save, and plot.

Usage:
    python plot_umap_from_merged_v4.py \
        --h5ad merged_allcells.h5ad \
        --outdir umap_plots \
        --n-top-genes 2000
"""

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


# =========================
# Small logger
# =========================
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =========================
# GPU + RMM setup
# =========================
def setup_gpus():
    import scipy
    log(f"SciPy version: {scipy.__version__}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    log(f"GPUs visible: {os.environ['CUDA_VISIBLE_DEVICES']}")
    log(f"Detected GPUs: {cp.cuda.runtime.getDeviceCount()}")

    # RAPIDS memory pool (managed)
    rmm.reinitialize(
        managed_memory=True,
        pool_allocator=True,
        initial_pool_size=None,
        devices=[0, 1, 2, 3],
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    log("RAPIDS RMM memory pool initialized.")


# =========================
# RAPIDS pipeline on HVG AnnData
# =========================
def run_rapids_pipeline(
    adata_hvg: ad.AnnData,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    resolution: float = 0.5,
):
    """
    Run PCA on CPU, then neighbors → UMAP → Leiden on GPU.
    UMAP parameters are explicitly set to match Scanpy defaults.
    """

    # ---- PCA (CPU) ----
    log(f"Running PCA on CPU with Scanpy (n_comps={n_pcs})…")
    sc.pp.pca(
        adata_hvg,
        n_comps=n_pcs,
        use_highly_variable=False,
        svd_solver="arpack",
    )
    log("✅ Done CPU PCA.")

    # ---- Move to GPU ----
    log("Sending HVG AnnData to GPU…")
    rsc.get.anndata_to_GPU(adata_hvg)
    log("✅ HVG AnnData now on GPU.")

    # ---- Neighbors ----
    log(f"Computing neighbors on GPU (n_neighbors={n_neighbors}, n_pcs={n_pcs})…")
    rsc.pp.neighbors(
        adata_hvg,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
        metric="euclidean",  # Scanpy default
    )
    log("✅ Done neighbors.")

    # ---- UMAP (Scanpy-like) ----
    log("Running UMAP on GPU (Scanpy-like parameters)…")
    rsc.tl.umap(
        adata_hvg,
        n_components=2,
        min_dist=0.5,               # Scanpy default
        spread=1.0,                 # Scanpy default
        n_epochs=200,               # Scanpy default (typically ~200)
        learning_rate=1.0,          # Scanpy default
        init="spectral",            # Scanpy default
        metric="euclidean",
        rep="X_pca",
        negative_sample_rate=5,     # Scanpy default
        random_state=0,             # deterministic
    )
    log("✅ Done UMAP (Scanpy-like).")

    # ---- Leiden ----
    log(f"Clustering with Leiden (resolution={resolution})…")
    rsc.tl.leiden(
        adata_hvg,
        resolution=resolution,
        random_state=0,             # deterministic
    )
    log("✅ Done Leiden.")

    log("🎉 RAPIDS GPU steps complete (Scanpy-style UMAP).")


# =========================
# UMAP plotting on full AnnData
# =========================
def make_basic_umap_plots(adata: ad.AnnData, outdir: str):

    os.makedirs(outdir, exist_ok=True)

    if "X_umap" not in adata.obsm_keys():
        raise RuntimeError("No UMAP embedding found in .obsm['X_umap'].")

    sc.settings.figdir = outdir
    sc.settings.autoshow = False

    # Dataset plot
    if "dataset" in adata.obs.columns:
        log("Plotting UMAP colored by 'dataset'…")
        sc.pl.umap(adata, color="dataset", save="_dataset.png", show=False)
    else:
        log("⚠️ No dataset column.")

    # Celltypist plot
    if "celltypist_cell_label" in adata.obs.columns:
        log("Plotting celltypist labels…")
        sc.pl.umap(
            adata,
            color="celltypist_cell_label",
            save="_celltypist_cell_label.png",
            show=False,
        )

    # Leiden plot
    if "leiden" in adata.obs.columns:
        log("Plotting UMAP colored by 'leiden'…")
        sc.pl.umap(adata, color="leiden", save="_leiden.png", show=False)

    log(f"✅ UMAP plots saved in: {outdir}")


# =========================
# Main
# =========================
def main():
    warnings.filterwarnings("ignore")

    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default="merged_allcells.h5ad")
    p.add_argument("--outdir", default="umap_plots")
    p.add_argument("--no-save-h5ad", action="store_true")
    p.add_argument("--npcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--n-top-genes", type=int, default=2000)

    args = p.parse_args()
    start = time.time()

    log("========== GPU HVG UMAP + LEIDEN PIPELINE (CPU PCA) ==========")

    setup_gpus()

    # ---- Load ----
    log(f"Loading AnnData from: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    log(f"Loaded AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # ---- HVGs ----
    log(f"Computing HVGs (n_top_genes={args.n_top_genes})…")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_genes,
        flavor="seurat_v3",
        subset=False,
        layer=None,
        inplace=True,
    )

    n_hvg = int(adata.var["highly_variable"].sum())
    log(f"Selected {n_hvg} HVGs.")

    # ---- Subset ----
    log("Creating HVG-only AnnData…")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    log(f"HVG AnnData: {adata_hvg.shape[0]:,} cells × {adata_hvg.shape[1]:,} genes")

    # ---- GPU pipeline ----
    run_rapids_pipeline(
        adata_hvg,
        n_pcs=args.npcs,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
    )

    # ---- Copy back to full AnnData ----
    log("Copying embeddings back to full AnnData…")

    if not adata.obs_names.equals(adata_hvg.obs_names):
        raise RuntimeError("Mismatch in obs indices.")

    # UMAP
    X_umap = adata_hvg.obsm["X_umap"]
    if "cupy" in type(X_umap).__module__:
        X_umap = cp.asnumpy(X_umap)
    adata.obsm["X_umap"] = X_umap

    # PCA
    if "X_pca" in adata_hvg.obsm_keys():
        X_pca = adata_hvg.obsm["X_pca"]
        if "cupy" in type(X_pca).__module__:
            X_pca = cp.asnumpy(X_pca)
        adata.obsm["X_pca"] = X_pca

    # Leiden
    if "leiden" in adata_hvg.obs.columns:
        adata.obs["leiden"] = adata_hvg.obs["leiden"].astype("category")

    # Free memory
    del adata_hvg
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

    # ---- Save ----
    os.makedirs(args.outdir, exist_ok=True)
    if not args.no_save_h5ad:
        out_h5ad = os.path.join(
            args.outdir,
            os.path.basename(args.h5ad).replace(".h5ad", "_with_umap_leiden.h5ad"),
        )
        log(f"Saving updated AnnData → {out_h5ad}")
        adata.write(out_h5ad)

    # ---- Plot ----
    make_basic_umap_plots(adata, args.outdir)

    elapsed = (time.time() - start) / 60
    log(f"🎯 Done. Total runtime: {elapsed:.1f} min")
    log("===============================================")


if __name__ == "__main__":
    main()
