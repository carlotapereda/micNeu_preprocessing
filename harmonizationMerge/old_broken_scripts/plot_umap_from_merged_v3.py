#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute HVGs on CPU, run PCA on CPU, then run RAPIDS neighbors/UMAP/Leiden
on HVG-only AnnData, copy embeddings/clusters back to the full object,
save, and plot.

Usage:
    python plot_umap_from_merged_v3.py \
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

    # Make all 4 GPUs visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    log(f"GPUs visible: {os.environ['CUDA_VISIBLE_DEVICES']}")
    log(f"Detected GPUs: {cp.cuda.runtime.getDeviceCount()}")

    # RAPIDS memory pool (managed for safety)
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
def run_rapids_pipeline(adata_hvg: ad.AnnData,
                        n_pcs: int = 50,
                        n_neighbors: int = 15,
                        resolution: float = 0.5):
    """
    Run PCA on CPU, then neighbors → UMAP → Leiden on GPU using rapids_singlecell,
    but force UMAP parameters to match Scanpy defaults as closely as possible.
    """

    # 1) PCA on CPU
    log(f"Running PCA on CPU with Scanpy (n_comps={n_pcs})…")
    sc.pp.pca(
        adata_hvg,
        n_comps=n_pcs,
        use_highly_variable=False,
        svd_solver="arpack",
    )
    log("✅ Done CPU PCA. 'X_pca' is now in adata_hvg.obsm.")

    # 2) AnnData to GPU
    log("Sending HVG AnnData (with X_pca) to GPU…")
    rsc.get.anndata_to_GPU(adata_hvg)
    log("✅ HVG AnnData now backed by GPU arrays.")

    # 3) Neighbors on GPU
    log(f"Computing neighbors on GPU (n_neighbors={n_neighbors}, n_pcs={n_pcs})…")
    rsc.pp.neighbors(
        adata_hvg,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
        metric="euclidean",   # Scanpy default
        method="rapids",
    )
    log("✅ Done neighbors.")

    # 4) UMAP — ***Scanpy-matching parameters***
    log("Running UMAP on GPU (Scanpy-like parameters)…")
    rsc.tl.umap(
        adata_hvg,
        n_components=2,
        min_dist=0.5,               # Scanpy default
        spread=1.0,                 # Scanpy default
        n_epochs=200,               # Scanpy default
        learning_rate=1.0,          # Scanpy default
        init="spectral",            # Scanpy default
        rep="X_pca",
        metric="euclidean",
        negative_sample_rate=5,     # Scanpy default
        random_state=0,             # Force reproducibility
    )
    log("✅ Done UMAP (Scanpy-like).")

    # 5) Leiden
    log(f"Clustering with Leiden (resolution={resolution}) on GPU…")
    rsc.tl.leiden(
        adata_hvg,
        resolution=resolution,
        random_state=0,             # make deterministic
    )
    log("✅ Done Leiden.")

    log("🎉 RAPIDS GPU steps complete (Scanpy-style UMAP).")


# =========================
# UMAP plotting on full AnnData
# =========================
def make_basic_umap_plots(adata: ad.AnnData, outdir: str):
    """
    Make basic UMAP plots using scanpy on the FULL AnnData:
      - colored by dataset
      - colored by celltypist_simplified (or cell_type)
      - colored by leiden
    """
    os.makedirs(outdir, exist_ok=True)

    if "X_umap" not in adata.obsm_keys():
        raise RuntimeError("No UMAP embedding found in .obsm['X_umap'].")

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
        log("⚠️ 'dataset' column not found. Skipping dataset plot.")

    # 2) UMAP by celltypist_simplified (or fallback)
    if "celltypist_cell_label" in adata.obs.columns:
        log("Plotting UMAP colored by 'celltypist_cell_label'…")
        sc.pl.umap(
            adata,
            color="celltypist_cell_label",
            save="celltypist_cell_label.png",
            show=False,
        )
    elif "cell_type" in adata.obs.columns:
        log("Plotting UMAP colored by 'cell_type' (fallback)…")
        sc.pl.umap(
            adata,
            color="cell_type",
            save="_cell_type.png",
            show=False,
        )
    else:
        log("⚠️ No 'celltypist_simplified' or 'cell_type'. Skipping that plot.")

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
        log("⚠️ 'leiden' not found. Skipping Leiden plot.")

    log(f"✅ UMAP plots saved in: {outdir}")


# =========================
# Main
# =========================
def main():
    warnings.filterwarnings("ignore")

    p = argparse.ArgumentParser(
        description="Compute HVGs on CPU, run CPU PCA + RAPIDS neighbors/UMAP/Leiden "
                    "on HVGs, copy back to full AnnData, and plot."
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
    p.add_argument("--n-top-genes", type=int, default=2000,
                   help="Number of HVGs to use for PCA/UMAP (default: 2000).")

    args = p.parse_args()

    start = time.time()
    log("========== GPU HVG UMAP + LEIDEN PIPELINE (CPU PCA) ==========")

    setup_gpus()

    # ---------- Load full AnnData ----------
    log(f"Loading AnnData from: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    log(f"Loaded AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # ---------- HVG selection on CPU ----------
    log(f"Computing highly variable genes on CPU with Scanpy "
        f"(n_top_genes={args.n_top_genes})…")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_genes,
        flavor="seurat_v3",
        subset=False,
        layer=None,
        inplace=True,
    )

    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("HVG computation failed: 'highly_variable' not in adata.var.")

    n_hvg = int(adata.var["highly_variable"].sum())
    log(f"Selected {n_hvg} highly variable genes.")

    # Subset to HVG AnnData (cells unchanged, genes reduced)
    log("Creating HVG-only AnnData for RAPIDS pipeline…")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    log(f"HVG AnnData shape: {adata_hvg.shape[0]:,} cells × {adata_hvg.shape[1]:,} genes")

    # ---------- Run PCA (CPU) + neighbors/UMAP/Leiden (GPU) ----------
    run_rapids_pipeline(
        adata_hvg,
        n_pcs=args.npcs,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
    )

    # ---------- Copy embeddings/clusters back to FULL AnnData ----------
    log("Copying UMAP, PCA, and Leiden labels back to full AnnData…")

    # Sanity check: obs indices must match
    if not adata.obs_names.equals(adata_hvg.obs_names):
        raise RuntimeError("Obs indices between full AnnData and HVG AnnData do not match!")

    # UMAP
    X_umap = adata_hvg.obsm.get("X_umap", None)
    if X_umap is None:
        raise RuntimeError("HVG object has no .obsm['X_umap'] after RAPIDS UMAP.")

    # Convert to CPU numpy if cupy
    if "cupy" in type(X_umap).__module__:
        X_umap = cp.asnumpy(X_umap)
    adata.obsm["X_umap"] = X_umap

    # PCA (optional but nice to have)
    if "X_pca" in adata_hvg.obsm_keys():
        X_pca = adata_hvg.obsm["X_pca"]
        if "cupy" in type(X_pca).__module__:
            X_pca = cp.asnumpy(X_pca)
        adata.obsm["X_pca"] = X_pca

    # Leiden
    if "leiden" in adata_hvg.obs.columns:
        adata.obs["leiden"] = adata_hvg.obs["leiden"].astype("category")
    else:
        log("⚠️ 'leiden' not found in HVG AnnData.obs – cannot copy to full AnnData.")

    # Free HVG object (and some GPU memory)
    del adata_hvg
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

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

    # ---------- UMAP plots on FULL AnnData ----------
    make_basic_umap_plots(adata, args.outdir)

    elapsed = time.time() - start
    log(f"🎯 Done. Total runtime: {elapsed/60:.1f} min")
    log("===============================================")


if __name__ == "__main__":
    main()
