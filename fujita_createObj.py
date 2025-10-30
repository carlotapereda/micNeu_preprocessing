import scanpy as sc
import anndata as ad
from pathlib import Path
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# Paths
# ----------------------------
input_dir = Path("/mnt/data/dejag_fujitaNatGen2025")
output_h5ad = Path("dejag_combined.h5ad")

# Remove existing file if re-running
if output_h5ad.exists():
    print("🧹 Removing existing H5AD file...")
    output_h5ad.unlink()

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

    with gzip.open(features, "rt") as f:
        adata.var_names = [line.strip().split("\t")[0] for line in f]
    with gzip.open(barcodes, "rt") as f:
        barcodes_list = [line.strip() for line in f]
        # prepend sample name to make them unique
        adata.obs_names = [f"{prefix}_{bc}" for bc in barcodes_list]

    adata.obs["sample"] = prefix
    print(f"✅ Done {prefix} | shape={adata.shape}")
    return adata
    
def read_one(mtx_file):
    prefix = mtx_file.name.replace(".matrix.mtx.gz", "")
    barcodes = input_dir / f"{prefix}.barcodes.tsv.gz"
    features = input_dir / f"{prefix}.features.tsv.gz"

    print(f"🔄 Reading {prefix}")
    adata = sc.read_mtx(mtx_file).T  # transpose to cells × genes

    with gzip.open(features, "rt") as f:
        adata.var_names = [line.strip().split("\t")[0] for line in f]
    with gzip.open(barcodes, "rt") as f:
        barcodes_list = [line.strip() for line in f]
        # prepend sample name to make them unique
        adata.obs_names = [f"{prefix}_{bc}" for bc in barcodes_list]

    adata.obs["sample"] = prefix
    print(f"✅ Done {prefix} | shape={adata.shape}")
    return adata

# ----------------------------
# Step 1–3: Process in batches
# ----------------------------
batch_size = 25
all_batches = []

for i in range(0, len(mtx_files), batch_size):
    subset = mtx_files[i:i + batch_size]
    print(f"\n🧩 Processing batch {i // batch_size + 1} ({len(subset)} files)")

    # Parallel read (I/O bound)
    with ThreadPoolExecutor(max_workers=8) as ex:
        adatas = list(ex.map(read_one, subset))

    # Concatenate in memory
    adata_batch = ad.concat(adatas, label="sample", merge="same")
    all_batches.append(adata_batch)

    # Free memory
    del adatas

# ----------------------------
# Step 4: Combine all batches
# ----------------------------
print("🔗 Combining all batches...")
adata_combined = ad.concat(all_batches, label="sample", merge="same")

# ----------------------------
# Step 5: Save compressed H5AD
# ----------------------------
print(f"💾 Saving combined dataset to {output_h5ad}")
adata_combined.write_h5ad(output_h5ad, compression="gzip")

print(f"🎉 Combined H5AD successfully saved at {output_h5ad}")
