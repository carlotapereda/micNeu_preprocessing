import anndata as ad

print("loading file...")

src = "/mnt/data/seaad_dlpfc/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
dst = "/mnt/data/seaad_dlpfc/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.zarr"

adata_lazy = ad.experimental.read_lazy(src)  # lazy, no big arrays loaded
print("done loading lazy file. ")
print("writing file...")


adata_lazy.write_zarr(
    dst,
    chunks=(2000, 2000),
    compression="zstd",
    compression_opts=6,
)

print("done writing file.")

adata.file.close()
print("done closing adata.")
