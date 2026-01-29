#merge MIT ROSMAP metadata

import os
import sys
from datetime import datetime
import pandas as pd
import scanpy as sc

############################
# SETUP LOGGING
############################
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"/mnt/data/mit_pfc_mathysCell2023/merge_rosmap_metadata_{timestamp}.log"
output_dir = "/mnt/data/mit_pfc_mathysCell2023/metadata_outputs"
os.makedirs(output_dir, exist_ok=True)

# Redirect all prints to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

tee = Tee(sys.stdout, open(log_file, "w"))
sys.stdout = tee
sys.stderr = tee

print(f"Logging started at {datetime.now()}")
print(f"Log file: {log_file}")
print(f"Output directory: {output_dir}")

############################
# LOAD INPUT FILES
############################

rosmap_clinical = pd.read_csv('/mnt/data/mit_pfc_mathysCell2023/metadata_files/ROSMAP_clinical.csv', low_memory = False)  
indiv_bc = pd.read_csv('/mnt/data/mit_pfc_mathysCell2023/metadata_files/Individual_Cells_Across_Studies_individualID.csv', low_memory = False)
indiv_df = pd.read_csv("/mnt/data/mit_pfc_mathysCell2023/metadata_files/MIT_ROSMAP_Multiomics_individual_metadata.csv", low_memory = False)

print("Files loaded successfully.")
print(f"ROSMAP clinical: {rosmap_clinical.shape}")
print(f"Individual BC: {indiv_bc.shape}")
print(f"Individual DF: {indiv_df.shape}")

###########################
# INDIV_BC
###########################
indiv_bc_raw = indiv_bc.copy()
indiv_bc = indiv_bc[['Mathys_Cell_2023', 'batch', 'chemistry', 'individualID']]
indiv_bc.rename(columns={'Mathys_Cell_2023': 'barcode'}, inplace=True)

# Group by 'individualID' and count unique batches
unique_batches = indiv_bc.groupby('individualID')['batch'].nunique()
all_consistent = (unique_batches == 1).all()
print("All individualIDs have a consistent batch:", all_consistent)

# Optionally print inconsistent IDs
inconsistent_ids = unique_batches[unique_batches != 1]
if not inconsistent_ids.empty:
    print("Inconsistent batch assignments found for:", inconsistent_ids.index.tolist())

# Save cleaned indiv_bc
indiv_bc_path = os.path.join(output_dir, f"indiv_bc_cleaned_{timestamp}.csv")
indiv_bc.to_csv(indiv_bc_path, index=False)
print(f"Saved cleaned indiv_bc to {indiv_bc_path}")

###########################
# BARCODE PRESENCE CHECKS
###########################

###########################
# LOAD ADATA (optional check)
###########################
adata_path = "/mnt/data/mit_pfc_mathysCell2023/PFC427_raw_data.h5ad"
if os.path.exists(adata_path):
    print(f"🔹 Loading AnnData object from {adata_path} in backed mode...")
    adata = sc.read_h5ad(adata_path, backed="r")
    print(f"Loaded backed AnnData with shape: {adata.shape}")
else:
    print(f"⚠️ AnnData file not found at {adata_path}, skipping barcode checks.")
    adata = None  # ensure defined

###########################
# BARCODE PRESENCE CHECKS
###########################
if adata is not None:
    adata_barcodes = adata.obs_names

    are_all_present = indiv_bc_raw['Mathys_Cell_2023'].isin(adata_barcodes).all()
    print("All indiv_bc_raw['Mathys_Cell_2023'] present in adata.obs_names:", are_all_present)

    present = indiv_bc_raw['Mathys_Cell_2023'].isin(adata_barcodes)
    num_present, num_not_present = present.sum(), (~present).sum()
    print("Barcodes present in adata.obs_names:", num_present)
    print("Barcodes not present:", num_not_present)

    are_all_present = pd.Index(adata_barcodes).isin(indiv_bc_raw['Mathys_Cell_2023']).all()
    print("All adata.obs_names present in indiv_bc_raw['Mathys_Cell_2023']:", are_all_present)

    present = pd.Index(adata_barcodes).isin(indiv_bc_raw['Mathys_Cell_2023'])
    num_present, num_not_present = present.sum(), (~present).sum()
    print("Barcodes present in indiv_bc_raw['Mathys_Cell_2023']:", num_present)
    print("Barcodes not present:", num_not_present)
else:
    print("⚠️ 'adata' not found in environment. Skipping barcode presence checks.")

###########################
# INDIV_DF
###########################
#Edit barcodes
indiv_df = indiv_df.dropna(axis=1, how='all')
indiv_df = indiv_df.drop_duplicates()
print("Dropped all-NA columns and duplicates from indiv_df.")
print(f"Cleaned indiv_df shape: {indiv_df.shape}")

indiv_df_path = os.path.join(output_dir, f"indiv_df_cleaned_{timestamp}.csv")
indiv_df.to_csv(indiv_df_path, index=False)
print(f"Saved cleaned indiv_df to {indiv_df_path}")

###########################
# MERGE DF AND CLINICAL
###########################

#Merge ROSMAP_clinical and indiv_df
indiv_clinical = indiv_df.merge(rosmap_clinical, on='individualID', how='left')
print(f"Merged indiv_df and rosmap_clinical. Result shape: {indiv_clinical.shape}")

indiv_clinical_path = os.path.join(output_dir, f"indiv_clinical_merged_{timestamp}.csv")
indiv_clinical.to_csv(indiv_clinical_path, index=False)
print(f"Saved merged indiv_clinical to {indiv_clinical_path}")

print("✅ Merge completed successfully.")
print(f"All outputs and logs saved in: {output_dir}")
