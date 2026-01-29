import anndata as ad
import pandas as pd
import mygene

# --- Load metadata only (backed mode avoids loading matrix into RAM)
adata = ad.read_h5ad("Fujita_filtered_apoe_singlets.h5ad", backed="r")
ensembl_ids = [g.split('.')[0] for g in adata.var_names]   # strip version numbers

# --- Query mygene for human symbols
mg = mygene.MyGeneInfo()
print("Querying mygene.info — this takes 1–2 min ...")
res = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')

# --- Make mapping
mapping = {r['query']: r.get('symbol', r['query']) for r in res}

# --- Load full AnnData (only now) and apply mapping
adata = ad.read_h5ad("Fujita_filtered_apoe_singlets.h5ad")
adata.var_names = [mapping.get(g.split('.')[0], g) for g in adata.var_names]
adata.var_names_make_unique()

# --- Save with gene symbols in name for clarity
out_path = "fujita_filtered_apoe_singlets_symbol.h5ad"
adata.write(out_path)
print(f"✅ Saved {out_path} with gene symbols instead of Ensembl IDs.")
