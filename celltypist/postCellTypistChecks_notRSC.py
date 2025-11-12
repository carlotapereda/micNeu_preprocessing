# ============================================================
# CellTypist counts-only QC + Scanpy UMAP (Light Save Version)
# ============================================================

import os
import time
import gc
import random
import warnings
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------
# Global setup
# --------------------------------------------
np.random.seed(42)
random.seed(42)
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100)

# --------------------------------------------
# Input files
# --------------------------------------------
files = {
    "fujita": "fujita_celltypist_GPU_counts_only.h5ad",
    "seaad":  "seaad_celltypist_GPU_counts_only.h5ad",
    "mit":    "mit_celltypist_GPU_counts_only.h5ad",
}

def out_png(name): return f"{name}_umap_celltypist_simplified.png"
def out_h5ad(name): return f"{name}_umap_light.h5ad"

# --------------------------------------------
# Helper: barcode validation
# --------------------------------------------
def validate_barcodes(adata, tag):
    barcodes = adata.obs_names
    missing = barcodes.isna().sum()
    empty = (barcodes.astype(str).str.strip() == "").sum()
    unique = barcodes.is_unique
    print(f"\n── {tag.upper()} ──")
    print(f"Total cells: {adata.n_obs:,}")
    print(f"Missing barcodes: {missing}")
    print(f"Empty barcodes:   {empty}")
    print(f"Unique barcodes:  {unique}")
    if missing == 0 and empty == 0 and unique:
        print("✅ Barcode check passed.\n")
    else:
        print("❗ Barcode check FAILED — please fix before proceeding.\n")

# --------------------------------------------
# Helper: simplified CellTypist label
# --------------------------------------------
def add_celltypist_simplified(adata, tag):
    col = "celltypist_cell_label"
    if col not in adata.obs:
        print(f"{tag}: '{col}' not found — skipping simplification.")
        return False
    simplified = (
        adata.obs[col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.split(" ")
        .str[0]
        .replace({"": "NA", "nan": "NA", "None": "NA"})
    )
    adata.obs["celltypist_simplified"] = simplified
    print(f"{tag}: created 'celltypist_simplified' "
          f"({adata.obs['celltypist_simplified'].nunique()} unique).")
    return True

# --------------------------------------------
# Helper: Scanpy preprocessing + UMAP
# --------------------------------------------
def run_scanpy_umap(adata, tag, n_hvg=5000, n_comps=50, n_neighbors=15, leiden_res=0.5):
    start = time.time()
    print(f"{tag}: starting Scanpy preprocessing…")

    # Normalize/log1p
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)

    # HVG selection
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"{tag}: HVGs selected ({adata.n_vars:,} genes)")

    # Scale and PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    print(f"{tag}: PCA complete")

    # Neighbors, UMAP, Leiden
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.umap(adata, min_dist=0.4, random_state=42)
    sc.tl.leiden(adata, resolution=leiden_res, random_state=42)

    print(f"{tag}: ✅ Scanpy pipeline complete in {time.time()-start:.1f}s")
    return adata

# --------------------------------------------
# Helper: plot UMAP by simplified label
# --------------------------------------------
def plot_umap(adata, tag, png_path, color_key="celltypist_simplified"):
    if color_key not in adata.obs:
        print(f"{tag}: '{color_key}' not in obs; skipping plot.")
        return
    sc.pl.umap(
        adata,
        color=color_key,
        frameon=False,
        sort_order=False,
        wspace=0.4,
        show=False,
        title=f"{tag} — UMAP by {color_key}",
    )
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"{tag}: 💾 saved {png_path}")

# --------------------------------------------
# Helper: make lightweight version
# --------------------------------------------
def make_light_adata(adata):
    """Return a stripped-down AnnData with only obs + obsm + uns."""
    light = sc.AnnData(obs=adata.obs.copy())
    if "X_umap" in adata.obsm:
        light.obsm["X_umap"] = adata.obsm["X_umap"].copy()
    if "leiden" in adata.obs:
        light.obs["leiden"] = adata.obs["leiden"].copy()
    if "celltypist_simplified" in adata.obs:
        light.obs["celltypist_simplified"] = adata.obs["celltypist_simplified"].copy()
    if hasattr(adata, "uns") and "umap" in adata.uns:
        light.uns["umap"] = adata.uns["umap"]
    return light

# --------------------------------------------
# Main loop
# --------------------------------------------
for name, path in files.items():
    print("=" * 70)
    print(f"Loading {name}: {path}")
    adata = sc.read_h5ad(path, backed=None)
    print(f"{name}: loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # 1) Barcode checks
    validate_barcodes(adata, name)

    # 2) Simplified labels
    has_simplified = add_celltypist_simplified(adata, name)

    # 3) Run Scanpy pipeline
    adata_hvg = run_scanpy_umap(
        adata, name,
        n_hvg=5000, n_comps=50, n_neighbors=15, leiden_res=0.5
    )

    # 4) Attach labels
    if has_simplified:
        adata_hvg.obs["celltypist_simplified"] = (
            adata.obs["celltypist_simplified"].reindex(adata_hvg.obs_names).values
        )

    # 5) Plot
    png = out_png(name)
    plot_umap(adata_hvg, name, png_path=png)

    # 6) Create light version (drop matrix + layers)
    light = make_light_adata(adata_hvg)
    out_path = out_h5ad(name)
    light.write(out_path, compression="gzip")
    print(f"{name}: 💾 saved light version → {out_path}")

    # 7) Memory cleanup
    del adata_hvg, light
    adata.X = None
    del adata
    gc.collect()
    print(f"{name}: 🧹 memory cleared.\n")

print("=" * 70)
print("✅ All datasets processed.")
