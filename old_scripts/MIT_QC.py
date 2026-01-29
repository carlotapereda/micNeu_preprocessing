#MIT_ROSMAP 
# 1) Filter patients 
# 2) Add metadata
# 3) QC



import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os

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

##################################
# SAVE OBJECT
##################################

#Save object
adata_subset = filtered_adata.copy()
adata_subset.write_h5ad('PFC_filtered_apoe.h5ad')



