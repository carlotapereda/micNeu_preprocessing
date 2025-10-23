import scanpy as sc
import anndata as ad
from pathlib import Path
import gzip
import shutil
import zarr
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# Paths
# ----------------------------
input_dir = Path("/mnt/data/dejag_fujitaNatGen2025")
output_zarr = Path("dejag_combined.zarr")

# Remove existing Zarr if re-running
if output_zarr.exists():
    print("🧹 Removing existing Zarr store...")
    shutil.rmtree(output_zarr)

# ----------------------------
# Files
# ----------------------------
mtx_files = sorted(input_dir.glob("*.matrix.mtx.gz"))
print(f"📂 Found {len(mtx_files)} MTX files")

# ----------------------------
# Utility: Read one sample
# ----------------------------
def read_one(mtx_file):
    prefix = mtx_file.name.replace(".matrix.mtx.gz", "")
    barcodes = input_dir / f"{prefix}.barcodes.tsv.gz"
    features = input_dir / f"{prefix}.features.tsv.gz"

    print(f"🔄 Reading {prefix}")
    adata = sc.read_mtx(mtx_file).T  # transpose to cells × genes

    # Read compressed barcodes/features efficiently
    with gzip.open(features, "rt") as f:
        adata.var_names = [line.strip().split("\t")[0] for line in f]
    with gzip.open(barcodes, "rt") as f:
        adata.obs_names = [line.strip() for line in f]

    adata.obs["sample"] = prefix
    print(f"✅ Done {prefix} | shape={adata.shape}")
    return adata

# ----------------------------
# Step 1–3: Process in batches
# ----------------------------
batch_size = 25
compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=2)

for i in range(0, len(mtx_files), batch_size):
    subset = mtx_files[i:i + batch_size]
    print(f"\n🧩 Processing batch {i // batch_size + 1} ({len(subset)} files)")

    # Parallel read (I/O bound)
    with ThreadPoolExecutor(max_workers=8) as ex:
        adatas = list(ex.map(read_one, subset))

    # Concatenate in memory (fits easily on r5.16xlarge)
    adata_batch = ad.concat(adatas, label="sample", merge="same")

    # Write batch to Zarr (append mode)
    print(f"💾 Writing batch {i // batch_size + 1} to {output_zarr}")
    adata_batch.write_zarr(output_zarr, mode="a", compressor=compressor)

    # Free memory
    del adatas, adata_batch

print("✅ All batches written to Zarr store")

# ----------------------------
# Step 4: Lazy merge on-disk
# ----------------------------
print("🔗 Combining all subgroups lazily...")
root = zarr.open_group(output_zarr, mode="r")
adatas = [ad.read_zarr(f"{output_zarr}/{g}", backed='r') for g in root.group_keys()]

adata_combined = ad.concat(adatas, label="sample", merge="same")
adata_combined.write_zarr(output_zarr, mode="w", compressor=compressor)

print("🎉 Combined Zarr successfully saved at", output_zarr)
