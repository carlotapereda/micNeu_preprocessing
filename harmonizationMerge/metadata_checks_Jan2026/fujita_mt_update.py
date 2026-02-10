import anndata as ad
import pandas as pd
import numpy as np
import gc
from scipy.stats import median_abs_deviation

############################################
# Config / Inputs
############################################
SRC_H5AD = "../../celltypist/fujita_celltypist_GPU_counts_only.h5ad"
OUTPUT_PATH = "fujita_final_QC_filtered_symbols.h5ad"

############################################
# 1. Open Source (Backed)
############################################
print(f"📖 Opening {SRC_H5AD} in backed mode...")
adata = ad.read_h5ad(SRC_H5AD, backed="r")

############################################
# 2. Fix Metadata in Memory
############################################
print("🧬 Recalculating MT/Ribo/HB metrics on Symbols...")

mt_genes = adata.var_names[adata.var_names.str.startswith("MT-")].tolist()
ribo_genes = adata.var_names[adata.var_names.str.startswith(("RPS", "RPL"))].tolist()
hb_genes = adata.var_names[adata.var_names.str.contains("^HB[^(P)]")].tolist()

obs_fixed = adata.obs.copy()

# Subset to memory is RAM-safe for just these columns
print(f"   - Summing {len(mt_genes)} MT, {len(ribo_genes)} Ribo, {len(hb_genes)} HB genes...")
obs_fixed["total_counts_mt"] = np.ravel(adata[:, mt_genes].to_memory().X.sum(axis=1))
obs_fixed["total_counts_ribo"] = np.ravel(adata[:, ribo_genes].to_memory().X.sum(axis=1))
obs_fixed["total_counts_hb"] = np.ravel(adata[:, hb_genes].to_memory().X.sum(axis=1))

print("   - Updating percentages...")
obs_fixed["pct_counts_mt"] = 100 * obs_fixed["total_counts_mt"] / obs_fixed["total_counts"]
obs_fixed["pct_counts_ribo"] = 100 * obs_fixed["total_counts_ribo"] / obs_fixed["total_counts"]
obs_fixed["pct_counts_hb"] = 100 * obs_fixed["total_counts_hb"] / obs_fixed["total_counts"]
obs_fixed[["pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"]] = obs_fixed[["pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"]].fillna(0)

############################################
# 3. Outlier Logic
############################################
def is_outlier(data, nmads):
    m = np.median(data)
    s = median_abs_deviation(data)
    return (data < m - nmads * s) | (data > m + nmads * s)

mt_mad_outlier = is_outlier(obs_fixed["pct_counts_mt"], 3)
mt_hard_outlier = obs_fixed["pct_counts_mt"] > 8.0
obs_fixed["mt_outlier"] = mt_mad_outlier | mt_hard_outlier
keep_mask = ~obs_fixed["mt_outlier"]

print(f"✂️  Filtering: {obs_fixed['mt_outlier'].sum():,} outliers removed. {keep_mask.sum():,} cells remaining.")

############################################
# 4. Stream to Disk (Memory Optimized)
############################################
# Slice metadata
obs_cleaned = obs_fixed[keep_mask].copy()

# Clear the heavy dataframe and run Garbage Collection
del obs_fixed
gc.collect() 

# Create the View
adata_view = adata[keep_mask, :]
adata_view.obs = obs_cleaned

# OPTIONAL: If you have extra layers you don't need, clear them to save huge disk/RAM
# if "X_pca" in adata_view.layers: del adata_view.layers["X_pca"]

print(f"🚀 Streaming filtered data to {OUTPUT_PATH}...")
# compression='gzip' is essential for EFS storage
adata_view.write_h5ad(OUTPUT_PATH, compression="gzip")

# Final Cleanup
if hasattr(adata.file, "close"):
    adata.file.close()

print(f"✅ Done! {OUTPUT_PATH} is ready for analysis.")