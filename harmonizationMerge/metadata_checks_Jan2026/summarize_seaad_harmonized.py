import pandas as pd
import os

# -----------------------------
# Configuration
# -----------------------------
INPUT_CSV = "SEAAD_harmonized_obs.csv"
OUT_CSV = "seaad_harmonized_column_summary_report.csv"
MAX_UNIQUES_DISPLAY = 25

# -----------------------------
# Helpers
# -----------------------------
def summarize_column(series, max_uniques=MAX_UNIQUES_DISPLAY):
    dtype = str(series.dtype)
    n_missing = int(series.isna().sum())

    if pd.api.types.is_numeric_dtype(series):
        return {
            "dtype": dtype,
            "kind": "numeric",
            "n_unique": int(series.nunique(dropna=True)),
            "unique_values": None,
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
            "mean": float(series.mean()) if not series.empty else None,
            "median": float(series.median()) if not series.empty else None,
            "n_missing": n_missing,
        }
    else:
        # For large strings/categoricals, we dropna before unique for speed
        uniques = series.dropna().unique()
        display_uniques = uniques[:max_uniques]

        return {
            "dtype": dtype,
            "kind": "categorical",
            "n_unique": len(uniques),
            "unique_values": "; ".join(map(str, display_uniques)),
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "n_missing": n_missing,
        }

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
    else:
        # 1. Get headers only first to know what columns to loop through
        headers = pd.read_csv(INPUT_CSV, nrows=0).columns.tolist()
        print(f"Found {len(headers)} columns. Starting low-memory processing...")

        rows = []

        # 2. Iterate through columns one by one
        for col in headers:
            print(f"Processing: {col}...", end="\r")
            
            # Read only this specific column
            col_data = pd.read_csv(INPUT_CSV, usecols=[col])[col]
            
            summary = summarize_column(col_data)
            
            rows.append({
                "source_file": INPUT_CSV,
                "column": col,
                **summary
            })

        # 3. Save report
        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(OUT_CSV, index=False)

        print(f"\n✅ Finished! Summary saved to {OUT_CSV}")