import anndata as ad
import pandas as pd
import os

FILES = {
    "FUJITA": "/mnt/data/fujita_final_QC_filtered.h5ad",
    "MIT-ROSMAP": "/mnt/efs/home/ubuntu/human_PFC/micNeu_preprocessing/celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEA-AD": "/mnt/efs/home/ubuntu/human_PFC/micNeu_preprocessing/celltypist/seaad_celltypist_GPU_counts_only.h5ad",
}

METADATA_CSV = "/mnt/data/mergehuPFC/final_merged_metadata.csv"
MERGED_TEMP = "/mnt/data/mergehuPFC/merged_expression_only.h5ad"
FINAL_H5AD = "/mnt/data/mergehuPFC/merged_allcells_with_metadata.h5ad"

def main():
    # 1. Concatenate Expression Matrices on Disk
    print(f"--- Executing concat_on_disk (Anndata {ad.__version__})...")
    ad.experimental.concat_on_disk(
        in_files=FILES,
        out_file=MERGED_TEMP,
        axis=0,
        join="inner",
        label="dataset_origin",
        index_unique="_" 
    )

    # 2. Load prepared metadata
    print("--- Loading prepared metadata...")
    obs_all = pd.read_csv(METADATA_CSV, low_memory=False)
    obs_all.set_index("barcode", inplace=True)

    # 3. Open merged file (backed) for alignment
    m = ad.read_h5ad(MERGED_TEMP, backed="r")

    # Align indices: Convert "barcode_DATASET" -> "DATASET_barcode"
    print("--- Reindexing and aligning metadata...")
    new_names = []
    for name in m.obs_names:
        barcode_part, dataset_part = name.rsplit("_", 1)
        new_names.append(f"{dataset_part}_{barcode_part}")
    
    final_obs = obs_all.reindex(new_names)
    final_obs.index = new_names 

    # 4. Final Assembly
    adata_final = ad.AnnData(
        X=m.X, 
        obs=final_obs, 
        var=m.var, 
        obsm=m.obsm, 
        layers=m.layers, 
        uns=m.uns
    )

    print(f"--- Streaming final object to {FINAL_H5AD}...")
    adata_final.write_h5ad(FINAL_H5AD)
    
    # Cleanup
    m.file.close()
    if os.path.exists(MERGED_TEMP):
        os.remove(MERGED_TEMP)
        
    print("✅ DONE! Verify with: ad.read_h5ad(FINAL_H5AD, backed='r')")

if __name__ == "__main__":
    main()