import h5py

src = "/mnt/data/seaad_dlpfc/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"

def print_structure(name, obj):
    """Print structure with indentation and metadata."""
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 {name}/")
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        print(f"{indent}📄 {name}  shape={shape} dtype={dtype}")

print(f"🔍 Inspecting HDF5 structure: {src}")
with h5py.File(src, "r") as f:
    f.visititems(print_structure)
print("✅ Done listing file structure.")
