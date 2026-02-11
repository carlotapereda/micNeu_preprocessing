import anndata as ad
import pandas as pd
import numpy as np
import os
import sys

# Source H5ADs (Mix of EBS and relative paths)
FILES = {
    "FUJITA": "/mnt/data/fujita_final_QC_filtered.h5ad",
    "MIT-ROSMAP": "/mnt/efs/home/ubuntu/human_PFC/micNeu_preprocessing/celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEA-AD": "/mnt/efs/home/ubuntu/human_PFC/micNeu_preprocessing/celltypist/seaad_celltypist_GPU_counts_only.h5ad",
}

# Source CSVs (Now on /mnt/data)
OBS_FILES = [
    "/mnt/data/mergehuPFC/SEAAD_harmonized_obs.csv",
    "/mnt/data/mergehuPFC/FUJITA_harmonized_obs.csv",
    "/mnt/data/mergehuPFC/MIT_ROSMAP_harmonized_obs.csv",
]

# Output to fast EBS storage
MERGED_TEMP = "/mnt/data/mergehuPFC/merged_expression_only.h5ad"
FINAL_H5AD = "/mnt/data/mergehuPFC/merged_allcells_with_metadata.h5ad"

def log(msg):
    print(f"--- {msg}")

def main():
    log(f"Initializing Merge on EBS Storage (Anndata {ad.__version__})")

    # 1. Load CSVs
    log("Loading and combining harmonized CSVs...")
    obs_list = []
    for path in OBS_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSV: {path}")
        obs_list.append(pd.read_csv(path, low_memory=False))
    
    obs_all = pd.concat(obs_list, axis=0)

    # 2. Derived Metadata
    log("Calculating AD metrics...")
    for col in ["Braak", "CERAD"]:
        obs_all[col] = pd.to_numeric(obs_all[col], errors="coerce")

    obs_all["celltypist_general"] = obs_all["celltypist_cell_label"].astype(str).str.split().str[0]
    obs_all["AD_status"] = np.where((obs_all["Braak"] >= 5) & (obs_all["CERAD"] <= 2), "AD", "non-AD")
    
    obs_all["AD_prog"] = "no-AD"
    obs_all.loc[(obs_all["Braak"].isin([5, 6])) & (obs_all["CERAD"] == 1), "AD_prog"] = "Late AD"
    obs_all.loc[(obs_all["Braak"].isin([3, 4])) & (obs_all["CERAD"] == 2), "AD_prog"] = "Early AD"

    obs_all.set_index("barcode", inplace=True)

    # 3. Concatenate Expression Matrix
    log("Executing concat_on_disk (Reading from EFS, Writing to EBS)...")
    ad.experimental.concat_on_disk(
        in_files=FILES,
        out_file=MERGED_TEMP,
        axis=0,
        join="inner",
        label="dataset_origin",
        index_unique="_" 
    )

    # 4. Final Assembly
    log("Opening merged file (backed) and aligning indices...")
    m = ad.read_h5ad(MERGED_TEMP, backed="r")

    # Flip indices to match Dataset_Barcode format
    new_names = []
    for name in m.obs_names:
        barcode_part, dataset_part = name.rsplit("_", 1)
        new_names.append(f"{dataset_part}_{barcode_part}")
    
    final_obs = obs_all.reindex(new_names)
    final_obs.index = new_names 

    log("Sanitizing metadata types for HDF5...")
    for col in final_obs.columns:
        if final_obs[col].dtype == 'object' or col in ["projid", "individualID", "Donor_ID"]:
            final_obs[col] = final_obs[col].astype(str).replace("nan", "Unknown")

    adata_final = ad.AnnData(X=m.X, obs=final_obs, var=m.var, obsm=m.obsm, layers=m.layers, uns=m.uns)

    log(f"Streaming to {FINAL_H5AD}...")
    # No gzip for maximum speed on the first successful run
    adata_final.write_h5ad(FINAL_H5AD)
    
    m.file.close()
    if os.path.exists(MERGED_TEMP):
        os.remove(MERGED_TEMP)
        
    log("✅ DONE! Verify with: ad.read_h5ad(FINAL_H5AD, backed='r')")

if __name__ == "__main__":
    main()