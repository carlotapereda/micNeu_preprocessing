import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

# ------------------------------------------------------------------
# Input / output paths
# ------------------------------------------------------------------
# Source: original AnnData file where X is empty but UMIs live in layers
src = "/mnt/data/seaad_dlpfc/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"

# Destination: cleaned AnnData with UMIs restored as X
dst = "adata_with_UMIs_as_X.h5ad"

print("🔧 Opening source file...")
with h5py.File(src, "r") as f:
    print("📖 Reading metadata...")

    # ==============================================================
    # 1️⃣ Build obs (cell metadata)
    #    - Handle broken columns
    #    - Restore categorical columns manually
    # ==============================================================
    obs_cols = [k for k in f["obs"].keys() if not k.startswith("__")]
    obs_dict = {}
    n_obs_expected = None  # enforce consistent length across obs columns

    for col in obs_cols:
        try:
            vals = np.array(f["obs"][col])

            # Skip malformed obs columns (e.g. 2D arrays)
            if vals.ndim != 1:
                print(f"⚠️ Skipping {col}: not 1D ({vals.shape})")
                continue

            # Set expected number of cells using the first valid column
            if n_obs_expected is None:
                n_obs_expected = len(vals)
                print(f"📏 Expected n_obs = {n_obs_expected}")

            # Enforce same length across all obs columns
            if len(vals) != n_obs_expected:
                print(f"⚠️ Skipping {col}: length {len(vals)} != {n_obs_expected}")
                continue

            # Convert byte strings / objects → Python strings
            if vals.dtype.kind in {"S", "O"}:
                vals = vals.astype(str)

            obs_dict[col] = vals

        except Exception as e:
            print(f"⚠️ Skipping obs column {col}: {e}")

    # --------------------------------------------------------------
    # Restore categorical columns stored in obs/__categories
    # AnnData stores categories separately as:
    #   - obs/__categories/<col>  → category labels
    #   - obs/<col>               → integer codes
    # --------------------------------------------------------------
    if "obs/__categories" in f:
        cat_group = f["obs/__categories"]
        print(f"📂 Restoring categories for {len(cat_group.keys())} columns...")

        for cat_col in cat_group.keys():
            try:
                # Category labels
                cats = np.array(cat_group[cat_col])

                # Integer category codes per cell
                codes = np.array(f["obs"][cat_col])

                # Map integer codes → category labels
                labels = np.array([
                    cats[c] if (0 <= c < len(cats)) else "NaN"
                    for c in codes
                ])

                obs_dict[cat_col] = labels.astype(str)
                print(f"✅ Restored categorical column: {cat_col}")

            except Exception as e:
                print(f"⚠️ Failed restoring {cat_col}: {e}")

    # Final obs DataFrame
    obs = pd.DataFrame(obs_dict)

    # Use string indices (AnnData convention)
    obs.index = np.arange(len(obs)).astype(str)

    print(f"✅ obs shape: {obs.shape}")

    # ==============================================================
    # 2️⃣ Build var (gene metadata)
    # ==============================================================
    var_cols = list(f["var"].keys())
    var_dict = {}

    for col in var_cols:
        try:
            vals = np.array(f["var"][col])

            # Skip malformed gene columns
            if vals.ndim != 1:
                print(f"⚠️ Skipping var column {col}: not 1D ({vals.shape})")
                continue

            # Convert byte strings / objects → Python strings
            if vals.dtype.kind in {"S", "O"}:
                vals = vals.astype(str)

            var_dict[col] = vals

        except Exception as e:
            print(f"⚠️ Skipping var column {col}: {e}")

    var = pd.DataFrame(var_dict)

    # Choose a reasonable gene index
    if "_index" in var:
        var.index = var["_index"].astype(str)
        var = var.drop(columns=["_index"])
    elif "gene_ids" in var:
        var.index = var["gene_ids"].astype(str)
    else:
        var.index = np.arange(var.shape[0]).astype(str)

    print(f"✅ var shape: {var.shape}")

    # ==============================================================
    # 3️⃣ Rebuild X from layers/UMIs
    #    - UMIs are stored as a CSR sparse matrix
    #    - We reconstruct it explicitly
    # ==============================================================
    print("🧬 Rebuilding X from layers/UMIs...")
    umi_grp = f["layers/UMIs"]

    # CSR components
    data = umi_grp["data"][:]
    indices = umi_grp["indices"][:]
    indptr = umi_grp["indptr"][:]

    # Matrix shape inferred from obs × var
    n_obs = len(obs)
    n_var = len(var)
    shape = (n_obs, n_var)

    # Reconstruct sparse count matrix
    X = csr_matrix((data, indices, indptr), shape=shape)

    print(f"✅ X shape: {shape}, nnz={X.nnz:,}")

# ==============================================================
# 4️⃣ Write clean AnnData object
# ==============================================================
print("💾 Writing clean AnnData object...")
adata = ad.AnnData(X=X, obs=obs, var=var)
adata.write_h5ad(dst)

print(f"🎉 Done! Clean file saved to {dst}")
