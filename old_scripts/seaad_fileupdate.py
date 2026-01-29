import scanpy as sc
print("reading")
adata = sc.read_h5ad("adata_with_UMIs_as_X.h5ad", backed=None)
print("done reading")
adata.write_h5ad("adata_with_UMIs_as_X_fixed.h5ad")
print("done")