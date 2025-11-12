# ============================================================
# CellTypist counts-only QC + RAPIDS UMAP (multi-dataset)
# ============================================================

########################################
# 0) Imports & GPU memory setup
########################################
import os
import time
import warnings
import re
import cupy as cp
import rmm
import random
from rmm.allocators.cupy import rmm_cupy_allocator

import scanpy as sc
import anndata as ad
import numpy as np
import rapids_singlecell as rsc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

# Quick check of SciPy / GPUs
import scipy
print("scipy:", scipy.__version__)

# Set visible GPUs (adjust if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"GPUs visible: {os.environ['CUDA_VISIBLE_DEVICES']}")
print("Detected GPUs:", cp.cuda.runtime.getDeviceCount())

# RAPIDS memory: pooled + managed for oversubscription safety
rmm.reinitialize(
    managed_memory=True,       # allows paging to host if needed
    pool_allocator=True,       # fast reuse
    initial_pool_size=None,    # let RAPIDS pick
    devices=[0, 1, 2, 3],
)
cp.cuda.set_allocator(rmm_cupy_allocator)

# Global seeds
np.random.seed(42)
cp.random.seed(42)
random.seed(42)

# Scanpy randomness (affects Leiden)
sc.settings.set_figure_params(dpi=100)
sc.settings.verbosity = 3

########################################
# 1) Files to process
########################################
files = {
    "fujita": "fujita_celltypist_GPU_counts_only.h5ad",
    "seaad":  "seaad_celltypist_GPU_counts_only.h5ad",
    "mit":    "mit_celltypist_GPU_counts_only.h5ad",
}

# Output name helpers
def out_png(name): return f"{name}_umap_celltypist_simplified.png"
def out_h5ad(name): return f"{name}_umap_labeled.h5ad"

########################################
# 2) Helper: barcode validation
########################################
def validate_barcodes(adata, tag):
    n_obs = adata.n_obs
    barcodes = adata.obs_names

    missing = barcodes.isna().sum()
    empty = (barcodes.astype(str).str.strip() == "").sum()
    unique = barcodes.is_unique

    print(f"── {tag.upper()} ──")
    print(f"Total rows in obs: {n_obs}")
    print(f"Missing barcodes: {missing}")
    print(f"Empty barcodes:   {empty}")
    print(f"Unique barcodes:  {unique}")
    if missing == 0 and empty == 0 and unique:
        print("✅ Barcode check passed.\n")
    else:
        print("❗ Barcode check FAILED — please fix before proceeding.\n")

########################################
# 3) Helper: gene name inspection
########################################
def inspect_gene_ids(adata, tag, n_preview=8):
    genes = adata.var_names[:n_preview].tolist()
    print(f"{tag}: first {n_preview} var_names:", genes)
    if len(genes) == 0:
        print("   (no genes?)")
        return

    ens_pct = np.mean([g.startswith("ENSG") for g in genes])
    sym_alnum_pct = np.mean([bool(re.match(r"^[A-Za-z0-9._-]+$", g)) for g in genes])

    if ens_pct > 0.8:
        print("→ Likely Ensembl IDs\n")
    elif sym_alnum_pct > 0.8:
        print("→ Likely gene symbols\n")
    else:
        print("→ Mixed or custom identifiers\n")

########################################
# 4) Helper: ensure counts layer exists
########################################
def ensure_counts_layer(adata):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

########################################
# 5) Helper: build celltypist_simplified
########################################
def add_celltypist_simplified(adata, tag):
    col = "celltypist_cell_label"
    if col not in adata.obs:
        print(f"{tag}: '{col}' not found — skipping simplification.")
        return False
    # first token before the first space; handle NaNs safely
    simplified = (
        adata.obs[col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.split(" ")
        .str[0]
    )
    # Keep empty/na rows as 'NA' to avoid plotting errors
    simplified = simplified.replace({"": "NA", "nan": "NA", "None": "NA"})
    adata.obs["celltypist_simplified"] = simplified
    print(f"{tag}: created 'celltypist_simplified' "
          f"({adata.obs['celltypist_simplified'].nunique()} unique).")
    return True

########################################
# 6) Helper: CPU preprocessing → GPU UMAP
########################################
def run_gpu_umap(adata, tag, n_hvg=5000, n_comps=50, n_neighbors=15, leiden_res=0.5):
    start = time.time()
    print(f"{tag}: starting CPU preprocessing…")

    # Normalize/log1p/HVG on CPU using counts
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    
    # Identify HVGs on the same normalized data
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="seurat_v3"
    )
    
    # Subset to HVGs
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    print(f"{tag}: CPU preprocessing done (HVGs: {adata_hvg.n_vars})")
    
    # Move to GPU
    rsc.get.anndata_to_GPU(adata_hvg)
    
    # Scale (optional but helps)
    rsc.pp.scale(adata_hvg)
    
    # RAPIDS workflow
    rsc.pp.pca(adata_hvg, n_comps=n_comps, use_highly_variable=True)
    print(f"{tag}: done PCA")
    rsc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_comps)
    print(f"{tag}: done neighbors")
    rsc.tl.umap(adata_hvg, min_dist=0.4, random_state=42)
    print(f"{tag}: done UMAP")
    rsc.tl.leiden(adata_hvg, resolution=leiden_res,random_state=42)
    print(f"{tag}: done Leiden")
    
    print(f"{tag}: ✅ RAPIDS GPU steps complete in {time.time()-start:.1f}s")

    return adata_hvg

########################################
# 7) Helper: plot UMAP by simplified label
########################################
def plot_umap(adata_hvg, tag, png_path, color_key="celltypist_simplified"):
    # Move back to CPU for plotting to ensure stability
    rsc.get.anndata_to_CPU(adata_hvg)

    if color_key not in adata_hvg.obs:
        print(f"{tag}: '{color_key}' not in obs; skipping plot.")
        return

    sc.pl.umap(
        adata_hvg,
        color=color_key,
        frameon=False,
        sort_order=False,
        wspace=0.4,
        show=False,
        title=f"{tag} — UMAP by {color_key}"
    )
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"{tag}: 💾 saved {png_path}")

########################################
# 8) Main loop
########################################
for name, path in files.items():
    print("=" * 70)
    print(f"Loading {name}: {path}")
    adata = sc.read_h5ad(path)
    print(f"{name}: loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"{name}: matrix dtype: {adata.X.dtype}; layers: {list(adata.layers.keys())}")

    # (1) Barcode checks
    validate_barcodes(adata, name)

    # (2) Gene ID inspection
    inspect_gene_ids(adata, name)

    # (3) Simplified label
    has_simplified = add_celltypist_simplified(adata, name)

    # (4) GPU UMAP (works with your example call sequence)
    adata_hvg = run_gpu_umap(
        adata, name,
        n_hvg=5000, n_comps=50, n_neighbors=15, leiden_res=0.5
    )

    # Carry the simplified label onto the HVG object for plotting
    if has_simplified:
        adata_hvg.obs["celltypist_simplified"] = (
            adata.obs["celltypist_simplified"].reindex(adata_hvg.obs_names).values
        )

    # (5) Plot & Save
    png = out_png(name)
    plot_umap(adata_hvg, name, png_path=png, color_key="celltypist_simplified")

    # Save an updated h5ad that includes UMAP + Leiden + simplified label
    # Note: Save the HVG object (analysis view) to keep file sizes small.
    out_path = out_h5ad(name)
    adata_hvg.write(out_path, compression="gzip")
    print(f"{name}: 💾 saved {out_path}")

    # Free GPU pool blocks between datasets
    try:
        import gc
        del adata_hvg
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        rmm.reinitialize(managed_memory=True, pool_allocator=True)
    except Exception as e:
        print(f"{name}: (non-fatal) GPU memory cleanup issue: {e}")

print("=" * 70)
print("✅ All datasets processed.")
