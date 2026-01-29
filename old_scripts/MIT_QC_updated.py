#MIT_ROSMAP 
# 1) Filter patients 
# 2) Add metadata
# 3) QC



import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import median_abs_deviation  


##################################
# LOAD DATA
##################################

adata = sc.read_h5ad(
    '/mnt/data/mit_pfc_mathysCell2023/PFC427_raw_data.h5ad',
    backed='r'  # lazy load (doesn't put full matrix in RAM)
)

adata_obs = adata.obs.copy()
adata_obs.rename(columns={'individual_ID': 'individualID'}, inplace=True)
adata_obs['barcode'] = adata_obs.index

##################################
# ADD METADATA
##################################
output_dir = "/mnt/data/mit_pfc_mathysCell2023/metadata_outputs"

# Load the latest cleaned indiv_bc and indiv_clinical CSVs
indiv_bc_path = sorted([f for f in os.listdir(output_dir) if "indiv_bc_cleaned" in f])[-1]
indiv_clinical_path = sorted([f for f in os.listdir(output_dir) if "indiv_clinical_merged" in f])[-1]

indiv_bc = pd.read_csv(os.path.join(output_dir, indiv_bc_path))
indiv_clinical = pd.read_csv(os.path.join(output_dir, indiv_clinical_path))

print(f"Loaded indiv_bc: {indiv_bc.shape} → {indiv_bc_path}")
print(f"Loaded indiv_clinical: {indiv_clinical.shape} → {indiv_clinical_path}")


adata.obs.rename(columns={'individual_ID': 'individualID'}, inplace=True)
adata.obs['barcode'] = adata.obs.index #create a barcode column based on index

#Merge obs with barcode metadata
adataobs = adata.obs.copy()
print(f"Original adata.obs shape: {adataobs.shape}")

# Merge on barcode
adataobs_meta = adataobs.merge(indiv_bc, on="barcode", how="left")
print(f"After merging with indiv_bc: {adataobs_meta.shape}")

# Merge on projid
adataobs_meta2 = adataobs_meta.merge(indiv_clinical, on="projid", how="left")
print(f"After merging with indiv_clinical: {adataobs_meta2.shape}")

##################################
# SUBSET METADATA BY APOE GENOTYPE
##################################

keep_barcodes = adataobs_meta2.loc[
    adataobs_meta2['apoe_genotype'].isin([33, 34, 44]), 'barcode'
].tolist()

print(f"✅ Found {len(keep_barcodes):,} barcodes matching APOE 33/34/44")


##################################
# LOAD ONLY APOE SUBSET INTO MEMORY
##################################
adata_subset = sc.read_h5ad(
    '/mnt/data/mit_pfc_mathysCell2023/PFC427_raw_data.h5ad',
    backed=None  # full load into memory now
)[keep_barcodes, :]

print(f"✅ Subset loaded into memory: {adata_subset.shape}")

# Add merged metadata to subset
adata_subset.obs = adataobs_meta2.set_index('barcode').loc[keep_barcodes]
print(f"✅ Updated subset obs with metadata (shape: {adata_subset.obs.shape})")


##################################
# REMOVE PATIENTS WITH LESS THAN 1000 CELLS
##################################
# Calculate cell counts per patient for the original dataset (before filtering)
counts_before = adata_subset.obs['individualID'].value_counts().sort_index()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot before filtering
counts_before.plot(kind='bar', ax=ax)
ax.set_title('Cell counts per patient (Before Filtering)')
ax.set_xlabel('projid')
ax.set_ylabel('Number of cells')

plt.tight_layout()
plt.show()

# Recalculate the counts with projid as string
adata_subset = adata_subset.copy()
adata_subset.obs['projid'] = adata_subset.obs['projid'].astype(str)

cluster_counts = adata_subset.obs['projid'].value_counts()
keep = cluster_counts.index[cluster_counts >= 1000] 

# Now subset the AnnData object
filtered_adata = adata_subset[adata_subset.obs['projid'].isin(keep)]


# Compute cell counts per patient
counts_before = adata_subset.obs['projid'].value_counts()
counts_after = filtered_adata.obs['projid'].value_counts()

# Create side-by-side histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram before filtering
axes[0].hist(counts_before, bins=30, edgecolor='black')
axes[0].set_title("Distribution of Cells per Patient (Before Filtering)")
axes[0].set_xlabel("Number of Cells")
axes[0].set_ylabel("Frequency")
# Draw a vertical dotted line at 1000 cells
axes[0].axvline(x=1000, color='red', linestyle='--', label='Threshold: 1000')
axes[0].legend()

# Histogram after filtering
axes[1].hist(counts_after, bins=30, edgecolor='black')
axes[1].set_title("Distribution of Cells per Patient (After Filtering)")
axes[1].set_xlabel("Number of Cells")
axes[1].set_ylabel("Frequency")
# Draw a vertical dotted line at 1000 cells
axes[1].axvline(x=1000, color='red', linestyle='--', label='Threshold: 1000')
axes[1].legend()

plt.tight_layout()
plt.show()


print("🧪 STEP 7 – Calculate QC metrics for MIT_ROSMAP subset...")

adata_subset.var["mt"]   = adata_subset.var_names.str.startswith("MT-")
adata_subset.var["ribo"] = adata_subset.var_names.str.startswith(("RPS", "RPL"))
adata_subset.var["hb"]   = adata_subset.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata_subset,
    qc_vars=["mt", "ribo", "hb"],
    inplace=True,
    percent_top=[20],
    log1p=True
)
print("✅ QC metrics added to adata_subset.obs")

# -------------------------------
# STEP 8 – MAD-based outlier filtering
# -------------------------------
print("🚧 STEP 8 – Filtering QC outliers (using MAD)...")

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

adata_subset.obs["outlier"] = (
    is_outlier(adata_subset, "log1p_total_counts", 5)
    | is_outlier(adata_subset, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata_subset, "pct_counts_in_top_20_genes", 5)
)
print("General QC outliers:")
print(adata_subset.obs.outlier.value_counts())

adata_subset.obs["mt_outlier"] = is_outlier(adata_subset, "pct_counts_mt", 3) | (
    adata_subset.obs["pct_counts_mt"] > 8
)
print("Mitochondrial QC outliers:")
print(adata_subset.obs.mt_outlier.value_counts())

print(f"Total cells before outlier filtering: {adata_subset.n_obs:,}")
adata_subset = adata_subset[(~adata_subset.obs.outlier) & (~adata_subset.obs.mt_outlier)].copy()
print(f"✅ Cells remaining after outlier filtering: {adata_subset.n_obs:,}")

sc.pl.scatter(
    adata_subset,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    save="_MITROSMAP_qc_scatter_postmtfilter.pdf"
)
print("✅ Saved QC scatter plot (_MITROSMAP_qc_scatter_postmtfilter.pdf)")

# -------------------------------
# STEP 9 – Cell and gene filters
# -------------------------------
print("🔬 STEP 9 – Applying min gene / min cell thresholds...")

before_cells = adata_subset.n_obs
before_genes = adata_subset.n_vars
sc.pp.filter_cells(adata_subset, min_genes=200)
sc.pp.filter_genes(adata_subset, min_cells=10)
print(f"✅ Removed {before_cells - adata_subset.n_obs:,} cells and "
      f"{before_genes - adata_subset.n_vars:,} genes.")
print(f"✅ Final QC-filtered shape: {adata_subset.shape}")


##################################
# SAVE OBJECT
##################################
print("💾 STEP 10 – Saving final filtered MIT_ROSMAP AnnData...")
adata_subset.write_h5ad("PFC_filtered_apoe_QC.h5ad")
print("🎉 Saved PFC_filtered_apoe_QC.h5ad with QC and outlier filtering applied.")


