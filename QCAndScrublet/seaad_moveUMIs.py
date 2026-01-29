import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

src = "/mnt/data/seaad_dlpfc/SEAAD_A9_RNAseq_all-nuclei.2024-02-13.h5ad"
dst = "adata_with_UMIs_as_X.h5ad"

print("🔧 Opening source file...")
with h5py.File(src, "r") as f:
    print("📖 Reading metadata...")

    # ------------------------------------
    # 1️⃣ Build obs (handle categories + invalid shapes)
    # ------------------------------------
    obs_cols = [k for k in f["obs"].keys() if not k.startswith("__")]
    obs_dict = {}
    n_obs_expected = None

    for col in obs_cols:
        try:
            vals = np.array(f["obs"][col])
            if vals.ndim != 1:
                print(f"⚠️ Skipping {col}: not 1D ({vals.shape})")
                continue
            if n_obs_expected is None:
                n_obs_expected = len(vals)
                print(f"📏 Expected n_obs = {n_obs_expected}")
            if len(vals) != n_obs_expected:
                print(f"⚠️ Skipping {col}: length {len(vals)} != {n_obs_expected}")
                continue
            # Cast to string if needed
            if vals.dtype.kind in {"S", "O"}:
                vals = vals.astype(str)
            obs_dict[col] = vals
        except Exception as e:
            print(f"⚠️ Skipping obs column {col}: {e}")

    # --- Restore all categorical columns ---
    if "obs/__categories" in f:
        cat_group = f["obs/__categories"]
        print(f"📂 Restoring categories for {len(cat_group.keys())} columns...")
        for cat_col in cat_group.keys():
            try:
                # category labels
                cats = np.array(cat_group[cat_col])
                # numeric codes
                codes = np.array(f["obs"][cat_col])
                # map codes to labels
                labels = np.array([
                    cats[c] if (0 <= c < len(cats)) else "NaN"
                    for c in codes
                ])
                obs_dict[cat_col] = labels.astype(str)
                print(f"✅ Restored categorical column: {cat_col}")
            except Exception as e:
                print(f"⚠️ Failed restoring {cat_col}: {e}")

    # --- Finalize obs dataframe ---
    obs = pd.DataFrame(obs_dict)
    obs.index = np.arange(len(obs)).astype(str)
    print(f"✅ obs shape: {obs.shape}")

    # ------------------------------------
    # 2️⃣ Build var
    # ------------------------------------
    var_cols = list(f["var"].keys())
    var_dict = {}
    for col in var_cols:
        try:
            vals = np.array(f["var"][col])
            if vals.ndim != 1:
                print(f"⚠️ Skipping var column {col}: not 1D ({vals.shape})")
                continue
            if vals.dtype.kind in {"S", "O"}:
                vals = vals.astype(str)
            var_dict[col] = vals
        except Exception as e:
            print(f"⚠️ Skipping var column {col}: {e}")

    var = pd.DataFrame(var_dict)

    if "_index" in var:
        var.index = var["_index"].astype(str)
        var = var.drop(columns=["_index"])
    elif "gene_ids" in var:
        var.index = var["gene_ids"].astype(str)
    else:
        var.index = np.arange(var.shape[0]).astype(str)

    print(f"✅ var shape: {var.shape}")

    # ------------------------------------
    # 3️⃣ Build sparse X from layers/UMIs
    # ------------------------------------
    print("🧬 Rebuilding X from layers/UMIs...")
    umi_grp = f["layers/UMIs"]

    data = umi_grp["data"][:]
    indices = umi_grp["indices"][:]
    indptr = umi_grp["indptr"][:]

    n_obs = len(obs)
    n_var = len(var)
    shape = (n_obs, n_var)
    X = csr_matrix((data, indices, indptr), shape=shape)
    print(f"✅ X shape: {shape}, nnz={X.nnz:,}")

print("💾 Writing clean AnnData object...")
adata = ad.AnnData(X=X, obs=obs, var=var)
adata.write_h5ad(dst)
print(f"🎉 Done! Clean file saved to {dst}")
