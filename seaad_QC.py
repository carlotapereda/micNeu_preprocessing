# QC of SEAAD (memory-safe)


import pandas as pd
import scanpy as sc
import numpy as np
from scipy.stats import median_abs_deviation
import os, time, gc
import matplotlib.pyplot as plt
from scipy import sparse

gc.collect()

sc.settings._vector_friendly = True
sc.settings.autosave = False
sc.settings.autoshow = False
sc.settings.verbosity = 2


print("~~~~~~~~~~~~~~~~~~~ 0 - loading h5ad")
data_dir = "/mnt/data/seaad_dlpfc/"
filename = "SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
file_path = os.path.join(data_dir, filename)

t0 = time.time()
adata = sc.read_h5ad(file_path, backed='r')
print(f"Loading took {time.time() - t0:.2f} seconds")


print("~~~~~~~~~~~~~~~~~~~ 1 - filtering by APOE")
keep = pd.Index(adata.obs_names)[adata.obs['APOE Genotype'].isin(['3/3', '3/4', '4/4'])]

print("~~~~~~~~~~~~~~~~~~~ 1.5 - load to memory and free memory")
adata = adata[keep, :].to_memory()  # Only load filtered subset into RAM
# Optional: shrink matrix dtype / format
if sparse.issparse(adata.X):
    adata.X = adata.X.tocsr()
    if adata.X.dtype != np.float32:
        adata.X.data = adata.X.data.astype(np.float32, copy=False)
else:
    adata.X = np.asarray(adata.X, dtype=np.float32)

if adata.raw is not None:
    adata.raw = None
if "raw" in adata.uns: 
    del adata.uns["raw"]
adata.uns.clear()

# --- Preserve only the UMI layer (if exists) ---
if "UMIs" in adata.layers:
    print(f"Keeping UMI layer, removing others: {list(adata.layers.keys())}")
    umi_layer = adata.layers["UMIs"]
    adata.layers.clear()
    adata.layers["UMIs"] = umi_layer
    # Ensure .X contains the raw UMI counts (optional but recommended)
    adata.X = umi_layer
else:
    print("No UMI layer found — .X left unchanged")

gc.collect()
print("Layers after cleanup:", list(adata.layers.keys()))
print("X dtype:", adata.X.dtype)
print("Shape:", adata.shape)



print("~~~~~~~~~~~~~~~~~~~~ 2 - checking number of cells per patient")
patientIDcolumnName = 'Donor ID'

# Counts per patient BEFORE ≥1000 filter (on the APOE-filtered data)
counts_before = adata.obs[patientIDcolumnName].value_counts().sort_index()

# Quick bar chart
fig, ax = plt.subplots(figsize=(8, 6))
counts_before.plot(kind='bar', ax=ax)
ax.set_title('Cell counts per patient (APOE-filtered)')
ax.set_xlabel('Donor ID')
ax.set_ylabel('Number of cells')
out_path = "/mnt/data/seaad_dlpfc/cell_counts_apoe_filtered.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print("Saved to:", out_path)

# Apply ≥1000 cells/patient and COMMIT the subset to adata
keep_ids = counts_before.index[counts_before >= 1000]
adata = adata[adata.obs[patientIDcolumnName].isin(keep_ids), :]
gc.collect()

# Recompute counts per patient after the ≥1000 filter
counts_after = adata.obs[patientIDcolumnName].value_counts()

# One set of histograms (no duplication)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(counts_before, bins=30, edgecolor='black')
axes[0].set_title("Cells per patient (Before ≥1000)")
axes[0].set_xlabel("Number of cells"); axes[0].set_ylabel("Frequency")
axes[0].axvline(x=1000, color='red', linestyle='--', label='Threshold: 1000'); axes[0].legend()
axes[1].hist(counts_after, bins=30, edgecolor='black')
axes[1].set_title("Cells per patient (After ≥1000)")
axes[1].set_xlabel("Number of cells"); axes[1].set_ylabel("Frequency")
axes[1].axvline(x=1000, color='red', linestyle='--', label='Threshold: 1000'); axes[1].legend()
out_path = "/mnt/data/seaad_dlpfc/cell_count_hists_before_after.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print("Saved to:", out_path)

# Free temporary arrays
del counts_before, counts_after, keep_ids; gc.collect()
gc.collect()

print("~~~~~~~~~~~~~~~~~~~~ 3 - Low quality cells")
if adata.is_view:
    adata = adata.copy()

# Annotate feature categories
adata.var["mt"]   = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
adata.var["hb"]   = adata.var_names.str.contains("^HB(?!P)", regex=True)

# Cheaper QC: remove percent_top to avoid costly ranking over 1M cells
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True, percent_top=None
)

# Outlier helpers (MADS on obs metrics only—doesn't touch X)
def is_outlier(M: np.ndarray, nmads: float):
    med = np.median(M)
    mad = median_abs_deviation(M, scale=1)  # raw MAD; no 1.4826 scaling so nmads≈robust SDs
    return (M < med - nmads * mad) | (M > med + nmads * mad)

# Build masks without duplicating big arrays
m_total   = adata.obs["log1p_total_counts"].to_numpy()
m_genes   = adata.obs["log1p_n_genes_by_counts"].to_numpy()
m_top20   = adata.obs.get("pct_counts_in_top_20_genes", pd.Series(np.zeros(adata.n_obs))).to_numpy()  # may not exist now
m_mt      = adata.obs["pct_counts_mt"].to_numpy()

outlier_mask = (
    is_outlier(m_total, 5) |
    is_outlier(m_genes, 5) |
    is_outlier(m_top20, 5)
)
mt_mask = (is_outlier(m_mt, 3) | (m_mt > 8))

adata.obs["outlier"]    = outlier_mask
adata.obs["mt_outlier"] = mt_mask

print(f"Total number of cells (pre-LQ filter): {adata.n_obs}")

# Apply filter (single copy)
keep_mask = (~outlier_mask) & (~mt_mask)
adata = adata[keep_mask, :].copy()
print(f"Number of cells after filtering low-quality: {adata.n_obs}")

# Clean up temporary arrays to free RAM
del m_total, m_genes, m_top20, m_mt, outlier_mask, mt_mask, keep_mask; gc.collect()

# QC scatter (post-filter)
# Note: sc.pl.scatter writes to sc.settings.figdir if save=... use suffix only.
sc.settings.figdir = "/mnt/data/seaad_dlpfc"
sc.pl.scatter(
    adata, x="total_counts", y="n_genes_by_counts", color="pct_counts_mt",
    save="_qc_scatter_postmtfilter.pdf"
)

print("~~~~~~~~~~~~~~~~~~~~ 4 - Basic gene/cell filters")
# These operate on X; keep them minimal
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
gc.collect()


print("~~~~~~~~~~~~~~~~~~~~ 5 - Save")
# Clean obs names for writing
adata.obs.columns = adata.obs.columns.str.replace("/", "_")

# Save checkpoint (compressed)
adata.write_zarr("/mnt/data/seaad_dlpfc/seaad_qc.zarr", chunks=(10000, 1000))
print("Saved:", out_file)

print("~~~~~~~~~~~~~~~~~~~~ ALL DONE")
