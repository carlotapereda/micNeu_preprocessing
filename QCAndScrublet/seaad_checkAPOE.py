import scanpy as sc
adata = sc.read_h5ad("adata_with_UMIs_as_X.h5ad", backed='r')
print("adata_with_UMIs_as_X.h5ad")
print(adata.obs["APOE Genotype"].unique())

import scanpy as sc
adata = sc.read_h5ad("/mnt/data/seaad_dlpfc/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad", backed='r')
print("/mnt/data/seaad_dlpfc/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad")
print(adata.obs["APOE Genotype"].unique())
