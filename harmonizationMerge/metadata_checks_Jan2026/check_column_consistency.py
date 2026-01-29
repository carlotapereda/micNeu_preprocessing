import pandas as pd

############################################
# Inputs
############################################

FILES = {
    "SEA-AD": "SEAAD_harmonized_obs.csv",
    "FUJITA": "FUJITA_harmonized_obs.csv",
    "MIT-ROSMAP": "MIT_ROSMAP_harmonized_obs.csv",
}

############################################
# Load columns only
############################################

cols = {}
for name, path in FILES.items():
    df = pd.read_csv(path, nrows=1)
    cols[name] = list(df.columns)
    print(f"{name}: {len(cols[name])} columns")

############################################
# Compute shared and unique columns
############################################

sets = {k: set(v) for k, v in cols.items()}

common_cols = set.intersection(*sets.values())
all_cols = set.union(*sets.values())

############################################
# Report
############################################

print("\n==============================")
print("Columns present in ALL datasets")
print("==============================")
for c in sorted(common_cols):
    print(c)

print("\n==============================")
print("Dataset-specific differences")
print("==============================")

rows = []
for name, s in sets.items():
    missing = sorted(all_cols - s)
    extra = sorted(s - common_cols)

    print(f"\n{name}")
    print(f"  Missing ({len(missing)}): {missing}")
    print(f"  Extra   ({len(extra)}): {extra}")

    for c in missing:
        rows.append({
            "dataset": name,
            "column": c,
            "status": "missing",
        })
    for c in extra:
        rows.append({
            "dataset": name,
            "column": c,
            "status": "extra",
        })

############################################
# Optional: check column order consistency
############################################

print("\n==============================")
print("Column order check")
print("==============================")

ref_name = list(cols.keys())[0]
ref_cols = cols[ref_name]

for name, col_list in cols.items():
    if col_list == ref_cols:
        print(f"{name}: column order MATCHES {ref_name}")
    else:
        print(f"{name}: column order DOES NOT MATCH {ref_name}")

############################################
# Save report
############################################

diff_df = pd.DataFrame(rows)
diff_df.to_csv("obs_column_differences.csv", index=False)

print("\n📄 Saved column difference report to:")
print("  obs_column_differences.csv")
