# scrublet_huge.py
import os, time, math, gc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scrublet as scr

# --------------------------
# CONFIG
# --------------------------
PATH = "fujita_filtered_apoe.h5ad"

PATIENT_COL = "individualID"       # <- exact column name in .obs

# If your raw counts are stored in a layer (recommended), put its name here:
COUNTS_LAYER = "counts"        # set to None to use .X

# Chunking: max cells to load in memory at once (tune to your RAM; start with 50k)
CHUNK_SIZE = 50_000

# Scrublet knobs (be conservative with PCs for speed/memory)
SCR_EXPECTED_DOUBLET_RATE = 0.045
SCR_SIM_DOUBLET_RATIO     = 2
SCR_N_PCS                 = 15

# Checkpoint output (so you can resume/review without rewriting the big .h5ad)
CHECKPOINT_CSV = "Fujita_scrublet_results_checkpoint.csv"

# --------------------------
# Helpers
# --------------------------
def get_matrix(adata_view):
    """
    Return a CSR sparse raw-counts matrix for the given view.
    Will:
      - use adata.layers[COUNTS_LAYER] if available,
      - else use adata.X,
      - ensure CSR sparse,
      - drop all-zero genes (saves memory/time).
    """
    print("      Extracting raw count matrix...")
    if COUNTS_LAYER and COUNTS_LAYER in adata_view.layers:
        print(f"      Using layer '{COUNTS_LAYER}'")
        M = adata_view.layers[COUNTS_LAYER]
    else:
        print("      Using adata.X (no layer found)")
        M = adata_view.X

    # Convert to CSR sparse
    if sp.issparse(M):
        print("      Matrix already sparse (CSR or CSC)")
        M = M.tocsr()
    else:
        print("      Converting dense to CSR sparse matrix")
        M = sp.csr_matrix(M)

    # Drop all-zero genes for this chunk to reduce dimensionality
    gene_nnz = np.diff(M.tocsc().indptr)  # non-zeros per gene
    keep_genes = gene_nnz > 0
    dropped = np.sum(~keep_genes)
    if dropped > 0:
        print(f"      Dropping {dropped:,} all-zero genes for this chunk")
        M = M[:, keep_genes]
    else:
        print("      No all-zero genes to drop")

    return M

def run_scrublet_on_matrix(M):
    """
    Run Scrublet on a CSR counts matrix and return (scores, preds).
    Guarantees preds is a boolean ndarray the same length as scores,
    even when Scrublet can't auto-pick a threshold.
    """
    print("      Initializing Scrublet...")
    scrub = scr.Scrublet(
        M,
        expected_doublet_rate=SCR_EXPECTED_DOUBLET_RATE,
        sim_doublet_ratio=SCR_SIM_DOUBLET_RATIO
    )

    print("      Running scrub_doublets()...")
    scores, preds = scrub.scrub_doublets(n_prin_comps=SCR_N_PCS)

    if preds is None:
        print("      Auto-threshold failed — selecting fallback threshold...")
        s = np.asarray(scores, dtype=float)
        s = np.nan_to_num(s, nan=0.0,
                          posinf=np.nanmax(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0,
                          neginf=0.0)

        sim = getattr(scrub, "doublet_scores_sim_", None)
        if sim is not None and np.size(sim) > 0 and np.isfinite(sim).any():
            sim = np.asarray(sim, dtype=float)
            sim = sim[np.isfinite(sim)]
            thresh = float(np.quantile(sim, 0.90))
            print(f"      Using simulated doublet 90th percentile threshold = {thresh:.4f}")
        else:
            q = max(0.50, min(0.995, 1.0 - SCR_EXPECTED_DOUBLET_RATE))
            thresh = float(np.quantile(s, q))
            print(f"      Using observed-score percentile fallback threshold = {thresh:.4f}")

        try:
            scrub.call_doublets(threshold=thresh)
            preds = scrub.predicted_doublets_
            print("      Threshold applied successfully via Scrublet")
        except Exception:
            preds = s >= thresh
            print("      Manual boolean thresholding fallback applied")

    print(f"      Scrublet finished. Doublets predicted: {np.sum(preds):,}/{len(preds):,}")
    preds = np.asarray(preds, dtype=bool)
    return scores, preds


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    t0 = time.time()
    print("Opening AnnData in backed read–write mode…")
    adata = sc.read_h5ad(PATH, backed='r+')

    adata.obs_names_make_unique()
    assert adata.obs_names.is_unique, "obs_names (barcodes) must be unique for .loc assignment"
    print("Verified unique obs_names (barcodes).")

    if 'doublet_scores' not in adata.obs.columns:
        adata.obs['doublet_scores'] = np.nan
        print("Created column 'doublet_scores' in .obs")
    if 'predicted_doublets' not in adata.obs.columns:
        adata.obs['predicted_doublets'] = False
        print("Created column 'predicted_doublets' in .obs")

    donors = adata.obs[PATIENT_COL].astype('category').cat.categories.tolist()
    print(f"Found {len(donors)} donors → {donors}")

    ckpt = None
    if os.path.exists(CHECKPOINT_CSV):
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        done_barcodes = set(ckpt['barcode'].tolist())
        print(f"Checkpoint found with {len(done_barcodes):,} cells already processed.")
    else:
        done_barcodes = set()
        print("No checkpoint found, starting fresh.")

    results_buffer = []
    total_cells = adata.n_obs
    processed = 0
    print(f"Total cells in dataset: {total_cells:,}")

    for d_i, donor in enumerate(donors, 1):
        donor_mask = (adata.obs[PATIENT_COL].values == donor)
        donor_indices = np.where(donor_mask)[0]
        donor_n = donor_indices.size
        if donor_n == 0:
            continue

        print(f"\n[{d_i}/{len(donors)}] Processing donor {donor!r} → {donor_n:,} cells")
        n_chunks = math.ceil(donor_n / CHUNK_SIZE)

        for c in range(n_chunks):
            start = c * CHUNK_SIZE
            end   = min((c + 1) * CHUNK_SIZE, donor_n)
            idx   = donor_indices[start:end]

            barcodes = adata.obs_names[idx]
            if done_barcodes:
                keep_mask = ~np.isin(barcodes, list(done_barcodes))
                if not np.any(keep_mask):
                    print(f"  Chunk {c+1}/{n_chunks}: already processed, skipping.")
                    continue
                idx = idx[keep_mask]
                barcodes = barcodes[keep_mask]

            chunk_n = idx.size
            print(f"  Chunk {c+1}/{n_chunks}: {chunk_n:,} cells  (rows {start}–{end-1})")

            view = adata[idx, :]
            M = get_matrix(view)

            print("    Running Scrublet on current chunk...")
            t_scr = time.time()
            scores, preds = run_scrublet_on_matrix(M)
            print(f"    Scrublet done in {time.time() - t_scr:.1f}s")

            if len(scores) != len(barcodes) or len(preds) != len(barcodes):
                print("    Warning: score/pred length mismatch for chunk; skipping write for this chunk.")
                continue

            adata.obs.loc[barcodes, 'doublet_scores'] = scores
            adata.obs.loc[barcodes, 'predicted_doublets'] = preds.astype(bool)
            print(f"    Wrote results for {len(barcodes):,} cells to adata.obs")

            results_buffer.append(pd.DataFrame({
                'barcode': barcodes,
                'doublet_scores': scores,
                'predicted_doublets': preds.astype(bool),
                PATIENT_COL: donor
            }))

            processed += chunk_n
            print(f"    Progress: {processed:,}/{total_cells:,} cells")

            del view, M, scores, preds
            gc.collect()
            print("    Freed memory for current chunk")

            if processed % 200_000 < CHUNK_SIZE:
                if results_buffer:
                    print("    Flushing checkpoint buffer to CSV...")
                    df_res = pd.concat(results_buffer, ignore_index=True)
                    mode = 'a' if os.path.exists(CHECKPOINT_CSV) else 'w'
                    header = not os.path.exists(CHECKPOINT_CSV)
                    df_res.to_csv(CHECKPOINT_CSV, index=False, mode=mode, header=header)
                    results_buffer = []
                    print(f"    Checkpoint flushed → {CHECKPOINT_CSV}")

    if results_buffer:
        print("Final flush of checkpoint buffer...")
        df_res = pd.concat(results_buffer, ignore_index=True)
        mode = 'a' if os.path.exists(CHECKPOINT_CSV) else 'w'
        header = not os.path.exists(CHECKPOINT_CSV)
        df_res.to_csv(CHECKPOINT_CSV, index=False, mode=mode, header=header)
        print(f"Final checkpoint flushed → {CHECKPOINT_CSV}")

    out_obs_csv = "fujita_doublets_obs.csv"
    print("Saving final obs summary CSV...")
    adata.obs[['doublet_scores', 'predicted_doublets']].to_csv(out_obs_csv)
    print(f"Obs summary saved → {out_obs_csv}")

    print("Creating singlets-only AnnData and saving...")
    adata = adata.to_memory()
    adata_singlets = adata[adata.obs["predicted_doublets"] == False, :].copy()
    adata_singlets.write_h5ad("Fujita_filtered_apoe_singlets.h5ad")
    print("Saved singlets AnnData → Fujita_filtered_apoe_singlets.h5ad")

    print(f"All done in {(time.time() - t0)/60:.1f} min.")
