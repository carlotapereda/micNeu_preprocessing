#!/usr/bin/env python
# seaad_scrublet_gpu.py — robust GPU Scrublet with per-donor chunking + safe checkpoints

import os, time, gc, math, tempfile
import numpy as np
import pandas as pd
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
import rapids_singlecell as rsc
import scanpy as sc

# --------------------------
# CONFIG
# --------------------------
data_dir = "/mnt/data/seaad_dlpfc"
IN_ZARR   = f"{data_dir}/seaad_qc_gpu.zarr"          # must be QC-filtered and counts-backed
OUT_ZARR  = f"{data_dir}/seaad_scrublet_gpu.zarr"
CHECKPOINT = f"{data_dir}/scrublet_results_checkpoint.csv"

PATIENT_COL  = "Donor ID"
COUNTS_LAYER = "counts"   # set to None if you kept raw counts in X
EXPECTED_DOUBLET_RATE = 0.045
SIM_DOUBLET_RATIO     = 2.0
N_PCS                 = 15

# Tuning
ROW_CHUNK = 512           # smaller -> lower GPU footprint; try 256 if you see memory pressure
CHUNK_SIZE = 100_000      # max cells per donor sub-chunk
FLUSH_EVERY = 1           # flush every chunk to keep checkpoints tight

# --------------------------
# GPU + RMM SETUP
# --------------------------
rmm.reinitialize(managed_memory=True, pool_allocator=True, devices=[0])
cp.cuda.set_allocator(rmm_cupy_allocator)

def gpu_free():
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()

print("📂 Loading QC-filtered Zarr…")
adata = sc.read_zarr(IN_ZARR)
print(f"✅ Loaded: {adata.shape}")

# Ensure counts layer choice is valid
if COUNTS_LAYER and COUNTS_LAYER not in adata.layers:
    print(f"⚠️  layers['{COUNTS_LAYER}'] not found; using X as counts")
    COUNTS_LAYER = None

# Ensure GPU layout and favorable chunking
rsc.get.anndata_to_GPU(adata)
# one gene block, modest row blocks
adata.X = adata.X.rechunk((ROW_CHUNK, -1))
print("Chunks:", adata.X.chunks)

# --------------------------
# Helper: Scrublet per donor chunk (in-place on sub-AnnData)
# --------------------------
def run_scrublet_gpu(sub_adata):
    rsc.pp.scrublet(
        sub_adata,
        layer=COUNTS_LAYER,                # None -> use X
        sim_doublet_ratio=SIM_DOUBLET_RATIO,
        expected_doublet_rate=EXPECTED_DOUBLET_RATE,
        n_prin_comps=N_PCS,
        log_transform=False,
        verbose=False,
        copy=False,
        random_state=0,
    )
    # return a (barcode, doublet_score, predicted_doublet) frame
    out = sub_adata.obs[["doublet_score", "predicted_doublet"]].copy()
    out["barcode"] = sub_adata.obs_names.to_numpy()
    return out

# --------------------------
# Checkpoint helpers
# --------------------------
def append_checkpoint(df):
    # atomic append to avoid partial writes
    if df.empty:
        return
    mode = "a" if os.path.exists(CHECKPOINT) else "w"
    header = not os.path.exists(CHECKPOINT)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=data_dir, suffix=".tmp") as tmp:
        df.to_csv(tmp.name, index=False, header=header)
        tmp.flush()
        os.fsync(tmp.fileno())
        # append atomically
        with open(CHECKPOINT, mode) as out:
            with open(tmp.name, "r") as inp:
                out.write(inp.read())
    os.remove(tmp.name)

def load_done_barcodes():
    if not os.path.exists(CHECKPOINT):
        return set()
    ckpt = pd.read_csv(CHECKPOINT, usecols=["barcode"])
    return set(ckpt["barcode"].astype(str).tolist())

# --------------------------
# Main
# --------------------------
t0 = time.time()

# donor list (as Python strings)
donors = pd.Categorical(adata.obs[PATIENT_COL].to_pandas(copy=False)).categories.tolist()
print(f"Found {len(donors)} donors.")

done = load_done_barcodes()
if done:
    print(f"Resuming: {len(done):,} barcodes already processed.")

total_cells = adata.n_obs
processed = len(done)

for d_i, donor in enumerate(donors, 1):
    # Mask on GPU without pulling entire column to host
    donor_mask = (adata.obs[PATIENT_COL].values == donor)
    # Convert mask to host indices cheaply
    try:
        # cudf/cupy boolean -> host indices
        import cudf
        if isinstance(donor_mask, cudf.Series):
            donor_mask = donor_mask.to_pandas().values
    except Exception:
        pass
    donor_idx = np.where(np.asarray(donor_mask))[0]
    donor_n = len(donor_idx)
    if donor_n == 0:
        continue

    print(f"\n[{d_i}/{len(donors)}] Donor {donor} → {donor_n:,} cells")
    n_chunks = math.ceil(donor_n / CHUNK_SIZE)

    for c in range(n_chunks):
        start, end = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, donor_n)
        sel = donor_idx[start:end]
        barcodes = adata.obs_names[sel].astype(str)

        # Skip already processed
        if done:
            keep = ~np.fromiter((b in done for b in barcodes), dtype=bool)
            if not keep.any():
                print(f"  Chunk {c+1}/{n_chunks}: already processed.")
                continue
            sel = sel[keep]
            barcodes = barcodes[keep]

        chunk_n = len(barcodes)
        if chunk_n == 0:
            continue

        print(f"  Chunk {c+1}/{n_chunks}: {chunk_n:,} cells")
        sub = adata[sel, :].copy()  # materializes just the metadata; X stays lazy/Dask

        # (Optional) push sub.X to preferred chunking
        try:
            sub.X = sub.X.rechunk((min(ROW_CHUNK, sub.n_obs), -1))
        except Exception:
            pass

        t_scr = time.time()
        try:
            obs_res = run_scrublet_gpu(sub)
        finally:
            # release sub's device buffers early
            del sub
            gpu_free()

        print(f"    ✅ Scrublet in {time.time() - t_scr:.1f}s")

        obs_res[PATIENT_COL] = donor
        # Enforce the same order as barcodes to be safe
        obs_res = obs_res.set_index("barcode").loc[barcodes].reset_index()

        append_checkpoint(obs_res)
        processed += chunk_n
        print(f"    Progress: {processed:,}/{total_cells:,}")

        # Free buffers each loop
        del obs_res
        gpu_free()

# --------------------------
# Merge results to adata.obs and write output
# --------------------------
print("\n🔗 Merging Scrublet results into obs …")
final = pd.read_csv(CHECKPOINT)
# Ensure 'barcode' is string
final["barcode"] = final["barcode"].astype(str)
# Align on obs_names (string)
obs_index = pd.Index(adata.obs_names.astype(str), name="barcode")
final = final.set_index("barcode").reindex(obs_index)
# Attach
adata.obs["doublet_score"] = final["doublet_score"].values
adata.obs["predicted_doublet"] = final["predicted_doublet"].values

print("💾 Writing Scrublet-annotated Zarr …")
# retain a practical chunking for downstream (rows × genes)
# keep one gene block to be friendly with RAPIDS downstream
adata.X = adata.X.rechunk((ROW_CHUNK, -1))
adata.write_zarr(OUT_ZARR, chunks=(min(10_000, adata.n_obs), min(2_000, adata.n_vars)))

print(f"✅ Results written to: {OUT_ZARR}")
print(f"⏱️ Done in {(time.time() - t0)/60:.2f} min")
