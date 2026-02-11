import pandas as pd
import numpy as np
import os

OBS_FILES = [
    "/mnt/data/mergehuPFC/SEAAD_harmonized_obs.csv",
    "/mnt/data/mergehuPFC/FUJITA_harmonized_obs.csv",
    "/mnt/data/mergehuPFC/MIT_ROSMAP_harmonized_obs.csv",
]
OUTPUT_METADATA = "/mnt/data/mergehuPFC/final_merged_metadata.csv"

def main():
    print("--- Loading and combining harmonized CSVs...")
    obs_list = []
    for path in OBS_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSV: {path}")
        obs_list.append(pd.read_csv(path, low_memory=False))
    
    obs_all = pd.concat(obs_list, axis=0)

    print("--- Calculating AD metrics and derived columns...")
    for col in ["Braak", "CERAD"]:
        obs_all[col] = pd.to_numeric(obs_all[col], errors="coerce")

    # Business Logic
    obs_all["celltypist_general"] = obs_all["celltypist_cell_label"].astype(str).str.split().str[0]
    obs_all["AD_status"] = np.where((obs_all["Braak"] >= 5) & (obs_all["CERAD"] <= 2), "AD", "non-AD")
    
    obs_all["AD_prog"] = "no-AD"
    obs_all.loc[(obs_all["Braak"].isin([5, 6])) & (obs_all["CERAD"] == 1), "AD_prog"] = "Late AD"
    obs_all.loc[(obs_all["Braak"].isin([3, 4])) & (obs_all["CERAD"] == 2), "AD_prog"] = "Early AD"

    # Sanitize for HDF5 compatibility
    print("--- Sanitizing metadata types...")
    for col in obs_all.columns:
        if obs_all[col].dtype == 'object' or col in ["projid", "individualID", "Donor_ID"]:
            obs_all[col] = obs_all[col].astype(str).replace("nan", "Unknown")

    obs_all.to_csv(OUTPUT_METADATA, index=False)
    print(f"✅ Metadata prepared and saved to {OUTPUT_METADATA}")

if __name__ == "__main__":
    main()