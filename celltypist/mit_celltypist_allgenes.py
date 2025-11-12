#Celltypist tries with rapids-singlecell

########################################
# 0. Imports and GPU memory setup
########################################
import os
import time, wget
import warnings
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

from cuml.decomposition import PCA

import scanpy as sc
import anndata as ad
import numpy as np
import rapids_singlecell as rsc
from celltypist import models, annotate

warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

########################################
# CHECKS
########################################

import scipy
print(scipy.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"GPUs visible: {os.environ['CUDA_VISIBLE_DEVICES']}")
print("Detected GPUs:", cp.cuda.runtime.getDeviceCount())

########################################
# MEMORY ALLOCATION
########################################

# Enable memory pool and multi-GPU allocation
rmm.reinitialize(
    managed_memory=True,       # safer — pages to CPU when oversubscribed
    pool_allocator=True,       # enables fast reuse
    initial_pool_size=None,    # let RAPIDS handle automatically
    devices=[0,1,2,3],         # explicitly register all 4 GPUs
)
cp.cuda.set_allocator(rmm_cupy_allocator)

########################################
# Load and move data to GPU
########################################
input_file = "../PFC_filtered_apoe_singlets.h5ad"
adata_full = sc.read_h5ad(input_file)
print(f"{adata_full.n_obs:,} cells × {adata_full.n_vars:,} genes loaded")
print(f"Matrix dtype: {adata_full.X.dtype}, layer keys: {list(adata_full.layers.keys())}")

# move AnnData structure to GPU
#rsc.get.anndata_to_GPU(adata)

if "counts" not in adata_full.layers:
    adata_full.layers["counts"] = adata_full.X.copy()


########################################
# SET PARAMETERS
########################################

umap_celltypist_path = "mit_celltypist_gpu.png"
final_path_h5ad = "mit_celltypist_GPU_counts_only.h5ad"

start = time.time()

########################################
# Normalize, log-transform, HVG, regress, scale
########################################
sc.pp.normalize_total(adata_full, target_sum=1e4)
print("done norm")
sc.pp.log1p(adata_full) # .X is now log1p-normalized (all genes)
print("done log")
sc.pp.highly_variable_genes(adata_full, n_top_genes=5000, flavor="seurat_v3", layer="counts")
adata_hvg = adata_full[:, adata_full.var["highly_variable"].values].copy()
print("done hvgs")
print("✅ Preprocessing complete on CPU")

########################################
# PCA / UMAP
########################################
rsc.get.anndata_to_GPU(adata_hvg)
rsc.pp.pca(adata_hvg, n_comps=50, use_highly_variable=False) 
print("done PCA")
rsc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)
print("done neighbors") 
rsc.tl.umap(adata_hvg)
print("done UMAP")
rsc.tl.leiden(adata_hvg, resolution=0.5)
print("done leiden")
print("✅ RAPIDS GPU steps complete")

########################################
# 6. Move data back to CPU for CellTypist
########################################
print("Move data back to CPU for CellTypist...")
rsc.get.anndata_to_CPU(adata_hvg)
#adata_celltypist = adata[:, :].copy()
#adata_celltypist.layers.clear()  # drop layers not needed for CellTypist

########################################
# 7. GPU-accelerated CellTypist annotation
########################################
print("starting celltypist...")
models.download_models(model="Adult_Human_PrefrontalCortex.pkl")
model = models.Model.load(model="Adult_Human_PrefrontalCortex.pkl")

pred = annotate(
    adata_full,
    model=model,
    use_GPU=True,
    majority_voting=True,
    over_clustering=adata_hvg.obs["leiden"]
)
adata_pred = pred.to_adata()
cols = adata_pred.obs.columns
label_col = "majority_voting" if "majority_voting" in cols else (
    "predicted_labels" if "predicted_labels" in cols else "predicted_label")

adata_full.obs["celltypist_cell_label"] = adata_pred.obs.loc[adata_full.obs_names, label_col]
if "conf_score" in cols:
    adata_full.obs["celltypist_conf_score"] = adata_pred.obs.loc[adata_full.obs_names, "conf_score"]

# put labels on the HVG object for plotting
for col in ["celltypist_cell_label", "celltypist_conf_score"]:
    adata_hvg.obs[col] = adata_full.obs[col].reindex(adata_hvg.obs_names).values

# 🔥 Free memory
del adata_pred
import gc; gc.collect()
cp.get_default_memory_pool().free_all_blocks()
rmm.reinitialize(managed_memory=True, pool_allocator=True)

print("✅ CellTypist annotation done (GPU)")

########################################
# 8. Visualization
########################################
import matplotlib.pyplot as plt
sc.pl.umap(adata_hvg, color=["celltypist_cell_label", "celltypist_conf_score"],
           frameon=False, sort_order=False, wspace=1, show=False)
plt.savefig(umap_celltypist_path, dpi=200, bbox_inches="tight")

########################################
# 9. Save annotated dataset
########################################
#adata.write("mit_celltypist_GPU_preproc.h5ad")
#print("💾 Saved final annotated dataset.")

########################################
# 10. Prepare final AnnData for saving
########################################

# Drop large intermediate arrays
adata_full.obsm.clear()
adata_full.varm.clear()
adata_full.uns.clear()

# Create a lightweight copy with only counts and annotations
adata_final = ad.AnnData(
    X=adata_full.layers["counts"],          # raw counts
    obs=adata_full.obs.copy(),                         # keep only relevant annotation columns
    var=adata_full.var.copy()               # keep all genes (not subset to HVGs)
)
# Save
adata_final.write(final_path_h5ad, compression="gzip")
print("💾 Saved final counts-only annotated dataset.")