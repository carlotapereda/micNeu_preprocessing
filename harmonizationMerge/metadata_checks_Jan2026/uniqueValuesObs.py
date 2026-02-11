import anndata as ad
import pandas as pd
import numpy as np

OUT_CSV = "obs_selected_columns_unique_summary.csv"

# -----------------------------
# Column definitions per dataset
# -----------------------------
COLUMNS = {
    "FUJITA": [
        "individualID", "projid", "msex", "educ", "race",
        "apoe_genotype", "age_death", "cts_mmse30_lv", "pmi",
        "braaksc", "ceradsc", "cogdx", "pct_counts_mt",
        "total_counts", "celltypist_cell_label",
        "predicted_doublets", "doublet_scores", "Study", "age_first_ad_dx"
    ],
    "MIT": [
        "individualID", "projid", "msex", "educ", "race",
        "apoe_genotype", "age_death", "cts_mmse30_lv", "pmi",
        "braaksc", "ceradsc", "cogdx", "pct_counts_mt",
        "total_counts", "celltypist_cell_label",
        "predicted_doublets", "doublet_scores",
        "batch", "Study", "age_first_ad_dx"
    ],
    "SEAAD": [
        "APOE Genotype", "Age at Death", "Braak", "CERAD score",
        "Cognitive Status", "Donor ID", "sample_id",
        "Last MMSE Score", "PMI", "Sex", "Years of education",
        "Primary Study Name", "Secondary Study Name",
        "Race (choice=Asian)",
        "Race (choice=Native Hawaiian or Pacific Islander)",
        "Race (choice=Other)",
        "Race (choice=Unknown or unreported)",
        "Race (choice=White)",
        "celltypist_cell_label", "predicted_doublets",
        "doublet_scores", "load_name", "pct_counts_mt",
        "total_counts", "Age of dementia diagnosis", "specify other race", "method"
    ],
}

FILES = {
    "FUJITA": "/mnt/data/fujita_final_QC_filtered.h5ad",
    "MIT": "../../celltypist/mit_celltypist_GPU_counts_only.h5ad",
    "SEAAD": "../../celltypist/seaad_celltypist_GPU_counts_only.h5ad",
}

# -----------------------------
# Helpers
# -----------------------------
def summarize_column(series, max_uniques=25):
    dtype = str(series.dtype)
    n_missing = int(series.isna().sum())

    if pd.api.types.is_numeric_dtype(series):
        return {
            "dtype": dtype,
            "kind": "numeric",
            "n_unique": int(series.nunique(dropna=True)),
            "unique_values": None,
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "n_missing": n_missing,
        }
    else:
        uniques = series.dropna().unique()
        uniques = uniques[:max_uniques]

        return {
            "dtype": dtype,
            "kind": "categorical",
            "n_unique": int(series.nunique(dropna=True)),
            "unique_values": "; ".join(map(str, uniques)),
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "n_missing": n_missing,
        }

# -----------------------------
# Main
# -----------------------------
rows = []

for dataset, path in FILES.items():
    adata = ad.read_h5ad(path, backed="r")

    for col in COLUMNS[dataset]:
        if col not in adata.obs:
            rows.append({
                "dataset": dataset,
                "column": col,
                "dtype": "MISSING",
                "kind": "missing",
                "n_unique": None,
                "unique_values": None,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "n_missing": None,
            })
            continue

        summary = summarize_column(adata.obs[col])

        rows.append({
            "dataset": dataset,
            "column": col,
            **summary
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"✅ Saved structured obs summary to {OUT_CSV}")
