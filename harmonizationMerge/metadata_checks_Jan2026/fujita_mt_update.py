import anndata as ad
import pandas as pd
import numpy as np
import gc
from scipy.stats import median_abs_deviation

############################################
# Config
############################################
SRC_H5AD = "../../celltypist/fujita_celltypist_GPU_counts_only.h5ad"
OUTPUT_PATH = "/mnt/data/fujita_final_QC_filtered.h5ad"

# With 121GB RAM, 500k cells per chunk is very safe
CHUNK_SIZE = 500000 

############################################
# 1. Calculate Filter Mask (Backed)
############################################
print(f"📖 Opening {SRC_H5AD} to calculate metrics...")
adata_backed = ad.read_h5ad(SRC_H5AD, backed="r")

mt_genes = adata_backed.var_names[adata_backed.var_names.str.startswith("MT-")].tolist()

# Calculate MT % without loading full matrix
print("🧬 Calculating MT metrics...")
obs_calc = adata_backed.obs.copy()
obs_calc["total_counts_mt"] = np.ravel(adata_backed[:, mt_genes].to_memory().X.sum(axis=1))
obs_calc["pct_counts_mt"] = (100 * obs_calc["total_counts_mt"] / obs_calc["total_counts"]).fillna(0)

def is_outlier(data, nmads):
    m = np.median(data)
    s = median_abs_deviation(data)
    return (data < m - nmads * s) | (data > m + nmads * s)

keep_mask_global = ~(is_outlier(obs_calc["pct_counts_mt"], 3) | (obs_calc["pct_counts_mt"] > 8.0))
print(f"✂️  Global filtering logic: {keep_mask_global.sum():,} cells to keep.")

############################################
# 2. Chunked Loading (Bypassing Backed-Subsetting)
############################################
print(f"🚀 Loading data in chunks of {CHUNK_SIZE} to avoid OOM Killer...")

chunks = []
total_cells = adata_backed.n_obs

for i in range(0, total_cells, CHUNK_SIZE):
    end = min(i + CHUNK_SIZE, total_cells)
    print(f"   → Processing block: {i:,} to {end:,}")
    
    # Load a raw block into memory
    adata_chunk = adata_backed[i:end, :].to_memory()
    
    # Apply the pre-calculated mask to this chunk
    chunk_mask = keep_mask_global[i:end]
    adata_chunk = adata_chunk[chunk_mask, :].copy()
    
    # Update metadata for this subset
    adata_chunk.obs["pct_counts_mt"] = obs_calc["pct_counts_mt"].iloc[i:end][chunk_mask].values
    adata_chunk.obs["mt_outlier"] = True
    
    chunks.append(adata_chunk)
    gc.collect()

# Close the backed handle immediately
adata_backed.file.close()
del adata_backed
gc.collect()

############################################
############################################
# 3. Final Merge and Save
############################################
print("🔗 Merging chunks into final object...")
# 'inner' join is appropriate here since all chunks have the same genes
adata_final = ad.concat(chunks, join="inner")

# Clean up chunk list to free RAM for the write operation
del chunks
gc.collect()

print(f"💾 Saving to {OUTPUT_PATH} (with compression)...")
# With 1.1M cells, this might take a few minutes
adata_final.write_h5ad(OUTPUT_PATH, compression="gzip")

print(f"✅ Success! Process complete. File: {OUTPUT_PATH}")