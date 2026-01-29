import anndata as ad
import pandas as pd

out_path = "obs_columns_names.txt"

with open(out_path, "w") as f:

    def write_obs_summary(label, path):
        adata = ad.read_h5ad(path, backed="r")
        cols = adata.obs.columns

        f.write(f"{label}: Found {len(cols)} obs columns:\n\n")
        for c in cols:
            f.write(f"{c}: {adata.obs[c].dtype}\n")
        f.write("\n" + "-" * 60 + "\n\n")

    write_obs_summary(
        "MERGED",
        "../merged_allcells.h5ad"
    )

    write_obs_summary(
        "MIT",
        "../../celltypist/mit_celltypist_GPU_counts_only.h5ad"
    )

    write_obs_summary(
        "SEAAD",
        "../../celltypist/seaad_celltypist_GPU_counts_only.h5ad"
    )

    write_obs_summary(
        "FUJITA",
        "../../celltypist/fujita_celltypist_GPU_counts_only.h5ad"
    )

print(f"✅ Saved obs column summaries to {out_path}")
