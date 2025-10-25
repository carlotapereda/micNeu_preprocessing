import os, time, gc, pandas as pd, cupy as cp, rmm
import rapids_singlecell as rsc
import scanpy as sc
from rmm.allocators.cupy import rmm_cupy_allocator
import cudf
import numpy as np
import math

# --------------------------
# CONFIG
# --------------------------
data_dir = "/mnt/data/seaad_dlpfc"
IN_ZARR   = f"{data_dir}/seaad_qc_gpu.zarr"
OUT_ZARR  = f"{data_dir}/seaad_scrublet_gpu.zarr"
CHECKPOINT = f"{data_dir}/scrublet_results_checkpoint.csv"

COUNTS_LAYER = "counts"     # or None if you didn’t store counts in layers
PATIENT_COL  = "Donor ID"
EXPECTED_DOUBLET_RATE = 0.045
SIM_DOUBLET_RATIO     = 2.0
N_PCS                 = 15

# optional memory-chunk size if you have huge donors
CHUNK_SIZE = 100_000  

# --------------------------
# GPU + RMM SETUP
# --------------------------
rmm.reinitialize(managed_memory=False, pool_allocator=True, devices=[0,1,2,3])
cp.cuda.set_allocator(rmm_cupy_allocator)

print("📂 Loading QC-filtered Zarr…")
adata = sc.read_zarr(IN_ZARR)
print(f"✅ Loaded: {adata.shape}")

# --------------------------
# Helper: Scrublet per donor
# --------------------------
def run_scrublet_gpu(sub_adata):
    rsc.pp.scrublet(
        sub_adata,
        layer=COUNTS_LAYER,
        sim_doublet_ratio=SIM_DOUBLET_RATIO,
        expected_doublet_rate=EXPECTED_DOUBLET_RATE,
        n_prin_comps=N_PCS,
        log_transform=False,
        verbose=False,
        copy=False,
        random_state=0,
    )
    return sub_adata.obs[["doublet_score", "predicted_doublet"]].copy()

# --------------------------
# Main loop (checkpoint-safe)
# --------------------------
t0 = time.time()

donors = adata.obs[PATIENT_COL].astype("category").cat.categories.tolist()
print(f"Found {len(donors)} donors.")

if os.path.exists(CHECKPOINT):
    ckpt = pd.read_csv(CHECKPOINT)
    done = set(ckpt["barcode"].tolist())
    print(f"Resuming: {len(done):,} barcodes already processed.")
else:
    done = set()
    ckpt = pd.DataFrame()

buffer = []
total_cells = adata.n_obs
processed = len(done)

for d_i, donor in enumerate(donors, 1):
    mask = adata.obs[PATIENT_COL].values == donor
    idx = np.where(mask)[0]
    donor_n = len(idx)
    if donor_n == 0:
        continue

    print(f"\n[{d_i}/{len(donors)}] Donor {donor} → {donor_n:,} cells")
    n_chunks = math.ceil(donor_n / CHUNK_SIZE)

    for c in range(n_chunks):
        start, end = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, donor_n)
        sel = idx[start:end]
        barcodes = adata.obs_names[sel]

        # skip if all barcodes done
        if done:
            keep = ~np.isin(barcodes, list(done))
            if not np.any(keep):
                print(f"  Chunk {c+1}/{n_chunks}: already processed.")
                continue
            sel = sel[keep]
            barcodes = barcodes[keep]

        chunk_n = len(barcodes)
        print(f"  Chunk {c+1}/{n_chunks}: {chunk_n:,} cells")

        sub = adata[sel, :].copy()
        t_scr = time.time()
        obs_res = run_scrublet_gpu(sub)
        print(f"    ✅ Done in {time.time() - t_scr:.1f}s")

        obs_res["barcode"] = barcodes
        obs_res[PATIENT_COL] = donor
        buffer.append(obs_res)
        processed += chunk_n
        print(f"    Progress: {processed:,}/{total_cells:,}")

        # periodic flush
        if processed % 200_000 < CHUNK_SIZE:
            df = pd.concat(buffer, ignore_index=True)
            mode = "a" if os.path.exists(CHECKPOINT) else "w"
            header = not os.path.exists(CHECKPOINT)
            df.to_csv(CHECKPOINT, index=False, mode=mode, header=header)
            buffer = []
            print(f"    💾 Checkpoint flushed → {CHECKPOINT}")

        del sub, obs_res
        gc.collect()

# final flush
if buffer:
    df = pd.concat(buffer, ignore_index=True)
    mode = "a" if os.path.exists(CHECKPOINT) else "w"
    header = not os.path.exists(CHECKPOINT)
    df.to_csv(CHECKPOINT, index=False, mode=mode, header=header)
    print(f"💾 Final checkpoint flushed → {CHECKPOINT}")

# merge results back into adata.obs
final = pd.read_csv(CHECKPOINT)
adata.obs = adata.obs.join(
    final.set_index("barcode")[["doublet_score", "predicted_doublet"]],
    how="left"
)

# save final Zarr only
adata.write_zarr(OUT_ZARR, chunks=(10000, 1000))
print(f"✅ Results written to: {OUT_ZARR}")
print(f"⏱️ Done in {(time.time() - t0)/60:.2f} min")
