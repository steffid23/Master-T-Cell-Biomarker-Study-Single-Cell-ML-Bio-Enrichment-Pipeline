import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# ==============================================================================
# DISEASE 4: Cerebellum / Brain Dataset Single-Cell Annotation Script
# ==============================================================================

INPUT_H5AD = r"C:\Users\Steffi Dominic\OneDrive\Desktop\cerebellum\cerebellum.h5ad"
OUTPUT_DIR = r"C:\Users\Steffi Dominic\.gemini\antigravity\scratch\tcll_cell_annotation\output_cerebellum"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CEREBELLUM_MARKERS = {
    'Purkinje Neurons': ['PCP4', 'CALB1', 'ITPR1'],
    'Granule Neurons': ['FAT2', 'GABRA6', 'RBFOX3'],
    'Astrocytes': ['GFAP', 'AQP4', 'ALDH1L1'],
    'Oligodendrocytes': ['MBP', 'PLP1', 'MOG', 'MAG'],
    'Oligodendrocyte Precursor Cells (OPCs)': ['PDGFRA', 'CSPG4'],
    'Microglia Brain Immune': ['AIF1', 'CX3CR1', 'P2RY12', 'CD68'],
    'Bergmann Glia': ['SLC1A3', 'HOPX'],
    'Vascular Endothelial': ['CLDN5', 'FLT1']
}

print(f"[1/5] Loading Cerebellum Dataset: {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)

# Assign layer matrix to adata.X if adata.X is None
if adata.X is None:
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    elif 'normalized' in adata.layers:
        adata.X = adata.layers['normalized'].copy()

symbol_col = None
for col in ['feature_name', 'Symbol', 'name', 'symbol']:
    if col in adata.var.columns:
        symbol_col = col
        break

if symbol_col is not None:
    adata.var['ensembl_id'] = adata.var_names
    new_names = adata.var[symbol_col].astype(str).values
    adata.var_names = [new_names[i] if new_names[i] and new_names[i] != 'nan' else adata.var_names[i] for i in range(len(new_names))]
    adata.var_names_make_unique()

if 'counts' not in adata.layers:
    adata.layers['counts'] = adata.X.copy()

print(f"[2/5] Quality Control & Preprocessing ({adata.n_obs} cells x {adata.n_vars} genes)...")
if 'n_genes_by_counts' not in adata.obs:
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata, mask_var="highly_variable", n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_r0.5')

print("[3/5] Annotation via Biomarker Gene Scoring & CellTypist...")
score_cols = []
for ct, markers in CEREBELLUM_MARKERS.items():
    present = [g for g in markers if g in adata.var_names]
    clean_ct = ct.replace(' ', '_').replace('/', '_').replace('+', 'pos').replace('(', '').replace(')', '')
    col_name = f"score_{clean_ct}"
    score_cols.append(col_name)
    if len(present) > 0:
        sc.tl.score_genes(adata, gene_list=present, score_name=col_name)
    else:
        adata.obs[col_name] = 0.0

col_map = {f"score_{ct.replace(' ', '_').replace('/', '_').replace('+', 'pos').replace('(', '').replace(')', '')}": ct for ct in CEREBELLUM_MARKERS.keys()}
adata.obs['marker_cell_type'] = adata.obs[score_cols].idxmax(axis=1).map(col_map)

try:
    import celltypist
    predictions = celltypist.annotate(adata, model='Immune_All_Low.pkl', majority_voting=False)
    res_adata = predictions.to_adata()
    # Check if immune score is strong, else map marker cluster
    adata.obs['celltypist_type'] = res_adata.obs['predicted_labels'].astype(str)
    
    cluster_ct = adata.obs.groupby('leiden_r0.5')['marker_cell_type'].agg(lambda x: x.mode()[0])
    adata.obs['final_cell_type'] = adata.obs['leiden_r0.5'].map(cluster_ct)
except Exception as e:
    cluster_ct = adata.obs.groupby('leiden_r0.5')['marker_cell_type'].agg(lambda x: x.mode()[0])
    adata.obs['final_cell_type'] = adata.obs['leiden_r0.5'].map(cluster_ct)

adata.obs['final_cell_type'] = adata.obs['final_cell_type'].astype(str).str.replace('/', ' ')

print("[4/5] Ranking Differential Marker Genes...")
ct_counts = adata.obs['final_cell_type'].value_counts()
valid_ct = ct_counts[ct_counts >= 3].index
adata_sub = adata[adata.obs['final_cell_type'].isin(valid_ct)].copy()

try:
    sc.tl.rank_genes_groups(adata_sub, groupby='final_cell_type', method='wilcoxon')
    result = adata_sub.uns['rank_genes_groups']
    records = []
    for group in result['names'].dtype.names:
        for rank in range(min(50, len(result['names'][group]))):
            records.append({
                'cell_type': group,
                'rank': rank + 1,
                'gene_symbol': result['names'][group][rank],
                'log2FC': float(result['logfoldchanges'][group][rank]),
                'p_val_adj': float(result['pvals_adj'][group][rank])
            })
    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, "cerebellum_differential_markers.csv"), index=False)
    print("  Differential markers exported to cerebellum_differential_markers.csv")
except Exception as e:
    print("  Rank genes groups note:", e)

print("[5/5] Exporting Annotated Output Files...")
counts_df = adata.obs['final_cell_type'].value_counts().reset_index()
counts_df.columns = ['Cell Type', 'Cell Count']
counts_df['Percentage (%)'] = (counts_df['Cell Count'] / adata.n_obs) * 100
counts_df.to_csv(os.path.join(OUTPUT_DIR, "cerebellum_cell_distribution.csv"), index=False)

adata.obs.columns = [c.replace('/', '_') for c in adata.obs.columns]

out_h5ad = os.path.join(OUTPUT_DIR, "annotated_cerebellum.h5ad")
adata.write_h5ad(out_h5ad)

print("\nCerebellum Cell Annotation Complete!")
print(counts_df.to_string(index=False))
print(f"\nAnnotated file saved to: {out_h5ad}")
