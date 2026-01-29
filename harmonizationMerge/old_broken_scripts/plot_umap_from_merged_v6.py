#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU neighbors + GPU Leiden + CPU Scanpy UMAP
Correct sparse → CPU conversion
Fully deterministic
Handles >4M cells
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
import scipy.sparse as sp
import rapids_singlecell as rsc
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#########################
# Logging
#########################
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


#########################
# GPU setup
#########################
def setup_gpus():
    import scipy
    log(f"SciPy version: {scipy.__version__}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    log(f"GPUs visible: {os.environ['CUDA_VISIBLE_DEVICES']}")

    n_gpus = cp.cuda.runtime.getDeviceCount()
    log(f"Detected GPUs: {n_gpus}")

    rmm.reinitialize(
        managed_memory=True,
        pool_allocator=True,
        initial_pool_size=None,
        devices=list(range(n_gpus)),
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    log("RAPIDS RMM memory pool initialized.")


#########################
# CPU converters
#########################
def to_cpu_dense(x):
    """Convert CuPy dense → NumPy dense."""
    if "cupy" in type(x).__module__:
        return cp.asnumpy(x)
    return x


def to_cpu_sparse(x):
    """Convert CuPy sparse → SciPy CSR sparse."""
    if "cupyx" in type(x).__module__ or "cupy" in type(x).__module__:
        x = x.get()
    if not sp.isspmatrix_csr(x):
        x = x.tocsr()
    return x


#########################
# GPU neighbors + Leiden + CPU UMAP
#########################
def run_pipeline(adata_hvg, n_pcs=10, n_neighbors=5, resolution=0.5):

    ########## PCA (CPU) ##########
    log(f"Running PCA on CPU ({n_pcs} PCs)…")
    sc.pp.pca(
        adata_hvg,
        n_comps=n_pcs,
        use_highly_variable=False,
        svd_solver="arpack",
    )
    log("✓ PCA complete.")

    ########## Move HVG AnnData to GPU ##########
    log("Sending HVG AnnData to GPU…")
    rsc.get.anndata_to_GPU(adata_hvg)
    log("✓ AnnData now on GPU.")

    ########## Neighbors (GPU) ##########
    log("Running neighbors on GPU…")
    rsc.pp.neighbors(
        adata_hvg,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
        random_state=0,
        metric="euclidean",
    )
    log("✓ Neighbors complete.")

    ########## Leiden (GPU) ##########
    log("Running Leiden on GPU…")
    rsc.tl.leiden(
        adata_hvg,
        resolution=resolution,
        random_state=0,
    )
    log("✓ Leiden complete.")

    ########## Convert everything back to CPU ##########
    log("Converting matrices + graphs back to CPU…")

    # X matrix
    if "cupyx" in type(adata_hvg.X).__module__:
        X = adata_hvg.X.get()
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()
        adata_hvg.X = X.copy()  # ensure writable
    else:
        adata_hvg.X = to_cpu_dense(adata_hvg.X)

    # X_pca
    if "X_pca" in adata_hvg.obsm:
        if "cupy" in type(adata_hvg.obsm["X_pca"]).__module__:
            adata_hvg.obsm["X_pca"] = cp.asnumpy(adata_hvg.obsm["X_pca"])

    # neighbor graphs
    if "distances" in adata_hvg.obsp:
        adata_hvg.obsp["distances"] = to_cpu_sparse(adata_hvg.obsp["distances"])
    if "connectivities" in adata_hvg.obsp:
        adata_hvg.obsp["connectivities"] = to_cpu_sparse(adata_hvg.obsp["connectivities"])

    # free GPU memory
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

    log("✓ All matrices converted to CPU.")

    ########## UMAP (CPU, EXACT SCANPY) ##########
    log("Running UMAP on CPU with Scanpy…")
    sc.tl.umap(
        adata_hvg,
        min_dist=0.5,
        spread=1.0,
        random_state=0,
        neighbors_key="neighbors",
    )
    log("✓ Scanpy UMAP complete.")

    return adata_hvg


#########################
# Plotting
#########################
def make_plots(adata, outdir):
    os.makedirs(outdir, exist_ok=True)

    sc.settings.figdir = outdir
    sc.settings.autoshow = False

    if "dataset" in adata.obs:
        sc.pl.umap(adata, color="dataset", save="_dataset", show=False)
    if "celltypist_cell_label" in adata.obs:
        sc.pl.umap(adata, color="celltypist_cell_label", save="_celltypist", show=False)
    if "leiden" in adata.obs:
        sc.pl.umap(adata, color="leiden", save="_leiden", show=False)


#########################
# Main
#########################
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
    log("========== GPU neighbors + GPU Leiden + CPU Scanpy UMAP ==========")

    setup_gpus()

    ########## Load AnnData ##########
    log(f"Loading: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    log(f"{adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    ########## HVGs ##########
    log(f"Selecting {args.n_top_genes} HVGs…")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_genes,
        flavor="seurat_v3",
        inplace=True,
    )

    ########## HVG AnnData subset ##########
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    log(f"HVG AnnData = {adata_hvg.shape[0]:,} × {adata_hvg.shape[1]:,}")

    ########## Check index consistency early ##########
    if not adata.obs_names.equals(adata_hvg.obs_names):
        raise RuntimeError("ERROR: obs_names mismatch after HVG subset.")

    ########## Run RAPIDS neighbors + Leiden + CPU UMAP ##########
    adata_hvg = run_pipeline(
        adata_hvg,
        n_pcs=args.npcs,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
    )

    ########## Copy embeddings back ##########
    log("Copying embeddings back to full AnnData…")
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.obs["leiden"] = adata_hvg.obs["leiden"].astype("category")

    ########## Save ##########
    if not args.no_save_h5ad:
        out = os.path.join(
            args.outdir,
            os.path.basename(args.h5ad).replace(".h5ad", "_with_umap_leiden.h5ad"),
        )
        log(f"Saving: {out}")
        adata.write(out)

    ########## Plot ##########
    make_plots(adata, args.outdir)

    elapsed = (time.time() - start) / 60
    log(f"✓ Done. Runtime: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
