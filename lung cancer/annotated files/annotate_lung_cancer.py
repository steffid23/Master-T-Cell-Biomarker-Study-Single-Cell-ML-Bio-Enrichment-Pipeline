import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# ==============================================================================
# DISEASE 2: Lung Cancer (NSCLC) Single-Cell Annotation Script
# ==============================================================================

INPUT_H5AD = r"C:\Users\Steffi Dominic\OneDrive\Desktop\T-cell biomarker discovery project\lungcancer.h5ad"
OUTPUT_DIR = r"C:\Users\Steffi Dominic\.gemini\antigravity\scratch\tcll_cell_annotation\output_lung_cancer"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LUNG_MARKERS = {
    'Lung Tumor-Infiltrating CD8+ T-cell': ['CD3D', 'CD3E', 'CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1'],
    'Lung CD4+ Helper T-cell': ['CD3D', 'CD3E', 'CD4', 'IL7R', 'LEF1', 'TCF7'],
    'Exhausted T-cell (Tex)': ['CD8A', 'PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'CTLA4'],
    'Regulatory T-cell (Treg)': ['CD3D', 'CD4', 'FOXP3', 'IL2RA', 'IKZF2'],
    'Alveolar Macrophage or Monocyte': ['CD14', 'FCGR3A', 'MARCO', 'MSR1'],
    'Malignant or Epithelial Tumor Cell': ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'MKI67']
}

print(f"[1/5] Loading Lung Cancer Dataset: {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)

symbol_col = None
for col in ['Symbol', 'feature_name', 'name', 'symbol']:
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
adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[(adata.obs['n_genes_by_counts'] >= 200) & (adata.obs['pct_counts_mt'] <= 25.0)].copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata, mask_var="highly_variable", n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_r0.5')

print("[3/5] Annotation via CellTypist & Biomarker Gene Scoring...")
score_cols = []
for ct, markers in LUNG_MARKERS.items():
    present = [g for g in markers if g in adata.var_names]
    clean_ct = ct.replace(' ', '_').replace('/', '_').replace('+', 'pos')
    col_name = f"score_{clean_ct}"
    score_cols.append(col_name)
    if len(present) > 0:
        sc.tl.score_genes(adata, gene_list=present, score_name=col_name)
    else:
        adata.obs[col_name] = 0.0

col_map = {f"score_{ct.replace(' ', '_').replace('/', '_').replace('+', 'pos')}": ct for ct in LUNG_MARKERS.keys()}
adata.obs['marker_cell_type'] = adata.obs[score_cols].idxmax(axis=1).map(col_map)

try:
    import celltypist
    predictions = celltypist.annotate(adata, model='Immune_All_Low.pkl', majority_voting=False)
    res_adata = predictions.to_adata()
    adata.obs['final_cell_type'] = res_adata.obs['predicted_labels'].astype(str)
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
    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, "lung_cancer_differential_markers.csv"), index=False)
    print("  Differential markers exported to lung_cancer_differential_markers.csv")
except Exception as e:
    print("  Rank genes groups note:", e)

print("[5/5] Exporting Annotated Output Files...")
counts_df = adata.obs['final_cell_type'].value_counts().reset_index()
counts_df.columns = ['Cell Type', 'Cell Count']
counts_df['Percentage (%)'] = (counts_df['Cell Count'] / adata.n_obs) * 100
counts_df.to_csv(os.path.join(OUTPUT_DIR, "lung_cancer_cell_distribution.csv"), index=False)

# Clean all column names in adata.obs to prevent HDF5 forward-slash errors
adata.obs.columns = [c.replace('/', '_') for c in adata.obs.columns]

out_h5ad = os.path.join(OUTPUT_DIR, "annotated_lung_cancer.h5ad")
adata.write_h5ad(out_h5ad)

print("\nLung Cancer Cell Annotation Complete!")
print(counts_df.head(12).to_string(index=False))
print(f"\nAnnotated file saved to: {out_h5ad}")
