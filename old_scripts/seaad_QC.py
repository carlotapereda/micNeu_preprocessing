import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import scanpy as sc
import matplotlib.pyplot as plt
import os

print("🚀 Starting SEAAD APOE subset pipeline...")

# -------------------------------
# STEP 1: Load metadata only (backed mode)
# -------------------------------
print("📂 Opening AnnData in backed (read-only) mode...")
adata = sc.read_h5ad('adata_with_UMIs_as_X.h5ad', backed='r')
print(f"✅ Opened AnnData in backed mode with {adata.n_obs:,} cells")

# -------------------------------
# STEP 3: Determine APOE subset
# -------------------------------
print("🧩 Filtering barcodes with APOE 33/34/44...")
obs_meta = adata.obs.copy()  # this materializes the DataFrame
adata.file.close()
keep_barcodes = obs_meta.loc[
    obs_meta['APOE Genotype'].isin(['4/4', '3/3', '3/4'])
].index.tolist()
print(f"✅ Found {len(keep_barcodes):,} barcodes to keep")


# -------------------------------
# STEP 4: Load only that subset into memory
# -------------------------------
print("📥 Loading only APOE subset into memory...")
adata_subset = sc.read_h5ad('adata_with_UMIs_as_X.h5ad', backed=None)[keep_barcodes, :].copy()
adata_subset.obs_names_make_unique()

print(f"✅ Loaded APOE subset: {adata_subset.shape}")

# -------------------------------
# STEP 6: Filter patients with <1000 cells
# -------------------------------
print("📊 Filtering patients with <1000 cells...")
adata_subset.obs['Donor ID'] = adata_subset.obs['Donor ID'].astype(str)
cluster_counts = adata_subset.obs['Donor ID'].value_counts()
keep_patients = cluster_counts.index[cluster_counts >= 1000]
filtered_adata = adata_subset[adata_subset.obs['Donor ID'].isin(keep_patients)].copy()
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
    save="seaad_qc_scatter_postmtfilter.pdf"
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
filtered_adata.write_h5ad('seaad_filtered_apoe.h5ad')
print("✅ Saved: seaad_filtered_apoe")

print("🎉 Done! Only APOE 33/34/44 subset loaded and quality-filtered.")
