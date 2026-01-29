import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import scanpy as sc
import matplotlib.pyplot as plt
import os

print("🚀 Starting Fujita APOE subset pipeline...")

# -------------------------------
# STEP 1: Load metadata only (backed mode)
# -------------------------------
print("📂 Opening AnnData in backed (read-only) mode...")
adata = sc.read_h5ad('dejag_combined.h5ad', backed='r')
adata.obs_names_make_unique()
print(f"✅ Opened AnnData in backed mode with {adata.n_obs:,} cells")

# -------------------------------
# STEP 2: Load metadata tables
# -------------------------------
print("🧬 Loading metadata files...")
cell_annotation = pd.read_csv("cell-annotation.full-atlas.csv").set_index("cell")
ROSMAP = pd.read_csv("ROSMAP_clinical.csv").set_index("individualID")
print(f"✅ cell_annotation: {cell_annotation.shape}, ROSMAP: {ROSMAP.shape}")

cell_annotation_clean = cell_annotation.drop(columns=["batch", "state"], errors="ignore")

print("🔗 Merging metadata to determine APOE genotype subset...")
obs_meta = adata.obs.join(cell_annotation_clean, how="left", rsuffix="_anno")
obs_meta = obs_meta.join(ROSMAP, on="individualID", how="left")

# -------------------------------
# STEP 3: Determine APOE subset
# -------------------------------
print("🧩 Filtering barcodes with APOE 33/34/44...")
keep_barcodes = obs_meta.loc[
    obs_meta['apoe_genotype'].isin([33, 34, 44])
].index.tolist()
print(f"✅ Found {len(keep_barcodes):,} barcodes to keep")

# -------------------------------
# STEP 4: Load only that subset into memory
# -------------------------------
print("📥 Loading only APOE subset into memory...")
adata_subset = sc.read_h5ad('dejag_combined.h5ad', backed=None)[keep_barcodes, :].copy()
print(f"✅ Loaded APOE subset: {adata_subset.shape}")

# -------------------------------
# STEP 5: Add merged metadata
# -------------------------------
adata_subset.obs = obs_meta.loc[keep_barcodes].copy()
print(f"✅ Added merged metadata. obs shape: {adata_subset.obs.shape}")

# -------------------------------
# STEP 6: Filter patients with <1000 cells
# -------------------------------
print("📊 Filtering patients with <1000 cells...")
adata_subset.obs['projid'] = adata_subset.obs['projid'].astype(str)
cluster_counts = adata_subset.obs['projid'].value_counts()
keep_patients = cluster_counts.index[cluster_counts >= 1000]
filtered_adata = adata_subset[adata_subset.obs['projid'].isin(keep_patients)].copy()
print(f"✅ Filtered down to {len(keep_patients)} patients, shape: {filtered_adata.shape}")

# -------------------------------
# STEP 7: QC METRICS
# -------------------------------
print("🧪 Calculating QC metrics...")
filtered_adata.var["mt"] = filtered_adata.var_names.str.startswith("MT-")
filtered_adata.var["ribo"] = filtered_adata.var_names.str.startswith(("RPS", "RPL"))
filtered_adata.var["hb"] = filtered_adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    filtered_adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
)
print("✅ QC metrics calculated.")

# -------------------------------
# STEP 8: MAD-based outlier filtering
# -------------------------------
print("🚧 Detecting QC outliers using MAD thresholds...")

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

filtered_adata.obs["outlier"] = (
    is_outlier(filtered_adata, "log1p_total_counts", 5)
    | is_outlier(filtered_adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(filtered_adata, "pct_counts_in_top_20_genes", 5)
)
print("General QC outliers:")
print(filtered_adata.obs.outlier.value_counts())

filtered_adata.obs["mt_outlier"] = is_outlier(filtered_adata, "pct_counts_mt", 3) | (
    filtered_adata.obs["pct_counts_mt"] > 8
)
print("Mitochondrial QC outliers:")
print(filtered_adata.obs.mt_outlier.value_counts())

print(f"Total cells before outlier filtering: {filtered_adata.n_obs}")
filtered_adata = filtered_adata[(~filtered_adata.obs.outlier) & (~filtered_adata.obs.mt_outlier)].copy()
print(f"✅ Remaining cells after filtering low-quality cells: {filtered_adata.n_obs}")

sc.pl.scatter(
    filtered_adata,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    save="_qc_scatter_postmtfilter.pdf"
)
print("✅ Saved QC scatter plot (_qc_scatter_postmtfilter.pdf)")

# -------------------------------
# STEP 9: Cell/gene filters
# -------------------------------
print("🔬 Filtering low-quality cells and genes (post-outlier)...")
before_cells = filtered_adata.n_obs
before_genes = filtered_adata.n_vars
sc.pp.filter_cells(filtered_adata, min_genes=200)
sc.pp.filter_genes(filtered_adata, min_cells=10)
print(f"✅ Removed {before_cells - filtered_adata.n_obs:,} cells, "
      f"{before_genes - filtered_adata.n_vars:,} genes.")
print(f"✅ Post-QC shape: {filtered_adata.shape}")

# -------------------------------
# STEP 10: SAVE
# -------------------------------
print("💾 Saving filtered AnnData object...")
filtered_adata.write_h5ad('fujita_filtered.apoe.h5ad')
print("✅ Saved: fujita_filtered.apoe.h5ad")

print("🎉 Done! Only APOE 33/34/44 subset loaded and quality-filtered.")
