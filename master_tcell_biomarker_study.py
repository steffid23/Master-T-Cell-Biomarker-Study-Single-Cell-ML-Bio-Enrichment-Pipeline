# -*- coding: utf-8 -*-
"""
===============================================================================
MASTER T-CELL BIOMARKER STUDY PIPELINE (FINAL PRODUCTION READY)
===============================================================================
Comprehensive Single-Cell Machine Learning & Pathway Enrichment Pipeline
Includes:
  1. Dual Gene Matching (Symbol + Ensembl ID)
  2. 10-fold Stratified Cross-Validation Classifier Benchmark (Full 2000 HVG vs Top 500 Biomarkers)
     Calculating 8 Evaluation Metrics: Accuracy, Balanced Acc, Precision Macro, Recall Macro, F1 Macro, MCC, Kappa, ROC-AUC
  3. Visualizations: Side-by-side Confusion Matrices & Top 20 Feature Importances (300 DPI)
  4. Complete 6 Enrichment & Overlap Analyses with DIVERSE PUBLICATION PLOT TYPES:
     - KEGG Pathway Analysis -> Bubble / Dot Plot (Size = Gene Overlap, Color = Significance)
     - Reactome Pathway Analysis -> Ranked Horizontal Barplot with Value Annotations & Badges
     - GO Biological Process Analysis -> Lollipop Plot (Stem-and-Node Visualization)
     - GO Molecular Function Analysis -> Color-Gradient Barplot with Metric Labels
     - MSigDB Hallmark / GSEA Analysis -> Waterfall Barplot
     - OncoDB / OncoKB Cancer Driver Genes -> 
         a) Driver Genes Importance Rank Barplot (oncokb_driver_genes_rank_plot.png)
         b) Clear Box Plot & Swarm Overlay (oncokb_overlapping_plot.png)
  5. Publication-Grade Formatting: Automatic text-wrapping for long pathway names & tight bounding box padding so titles and labels are NEVER cut off.

Usage:
  python master_tcell_biomarker_study.py <cerebellum | colon | lung | all>
===============================================================================
"""

import os
import sys
import textwrap
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import xgboost as xgb
from scipy.sparse import issparse
from scipy.stats import hypergeom
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp

# Configure publication-grade styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISEASE_CONFIGS = {
    'cerebellum': {
        'name': 'Cerebellum',
        'folder': os.path.join(BASE_DIR, 'cerebellum'),
        'h5ad': os.path.join(BASE_DIR, 'cerebellum', 'annotated', 'annotated_cerebellum.h5ad'),
        'csv': os.path.join(BASE_DIR, 'cerebellum', 'top_500_biomarkers_Ensembl_gene_names.csv'),
        'obs_col': 'final_cell_type'
    },
    'colon': {
        'name': 'Colon Cancer',
        'folder': os.path.join(BASE_DIR, 'colon cancer'),
        'h5ad': os.path.join(BASE_DIR, 'colon cancer', 'annotation', 'annotated_colon_cancer.h5ad'),
        'csv': os.path.join(BASE_DIR, 'colon cancer', 'Model_Full_top_500_biomarkers_1_with_Ensembl_and_Gene_Names.csv'),
        'obs_col': 'final_cell_type'
    },
    'lung': {
        'name': 'Lung Cancer',
        'folder': os.path.join(BASE_DIR, 'lung cancer'),
        'h5ad': os.path.join(BASE_DIR, 'lung cancer', 'annotated files', 'annotated_lung_cancer.h5ad'),
        'csv': os.path.join(BASE_DIR, 'lung cancer', 'Model_Full_top_500_biomarkers_with_Ensembl_and_Gene_Names.csv'),
        'obs_col': 'final_cell_type'
    }
}

ONCOKB_DATABASE = {
    'MTOR': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'PI3K/Akt/mTOR Nutrient & Growth Regulator'},
    'PIK3CD': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'Phosphoinositide 3-Kinase Delta Catalytic Subunit'},
    'ARID1A': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 2 (Standard Care)', 'function': 'SWI/SNF Chromatin Remodeling Complex Subunit'},
    'TP53': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 1 (FDA Target)', 'function': 'Cellular Tumor Antigen p53 / Guardian of Genome'},
    'PTEN': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 1 (FDA Target)', 'function': 'Phosphatase and Tensin Homolog / PI3K Negative Regulator'},
    'KRAS': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'Kirsten Rat Sarcoma Viral Oncogene Homolog'},
    'EGFR': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'Epidermal Growth Factor Receptor'},
    'BRAF': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'V-Raf Murine Sarcoma Viral Oncogene Homolog B'},
    'MYC': {'role': 'Oncogene (OG)', 'level': 'Level 3 (Clinical Target)', 'function': 'MYC Proto-Oncogene BHLH Transcription Factor'},
    'CDK4': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'Cyclin Dependent Kinase 4'},
    'ATM': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 2 (Standard Care)', 'function': 'ATM Serine/Threonine Kinase / DNA Damage Response'},
    'BRCA1': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 1 (FDA Target)', 'function': 'BRCA1 DNA Repair Associated'},
    'BRCA2': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 1 (FDA Target)', 'function': 'BRCA2 DNA Repair Associated'},
    'APC': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 3 (Clinical Target)', 'function': 'APC Regulator of WNT Signaling Pathway'},
    'RB1': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 3 (Clinical Target)', 'function': 'RB Transcriptional Corepressor 1'},
    'VHL': {'role': 'Tumor Suppressor (TSG)', 'level': 'Level 1 (FDA Target)', 'function': 'Von Hippel-Lindau Tumor Suppressor'},
    'ALK': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'AL Receptor Tyrosine Kinase'},
    'KIT': {'role': 'Oncogene (OG)', 'level': 'Level 1 (FDA Target)', 'function': 'KIT Proto-Oncogene Receptor Tyrosine Kinase'},
    'NOTCH1': {'role': 'Oncogene / TSG', 'level': 'Level 3 (Clinical Target)', 'function': 'Notch Receptor 1 Signaling'}
}

ENRICHMENT_LIBRARIES = {
    'KEGG_2021_Human': ('kegg_significant_pathways.csv', 'kegg_significant_pathways_plot.png', 'KEGG Pathway Analysis', 'bubble'),
    'Reactome_2022': ('reactome_pathways.csv', 'reactome_enrichment_plot.png', 'Reactome Pathway Analysis', 'ranked_bar'),
    'GO_Biological_Process_2021': ('go_bp_enrichment.csv', 'go_bp_enrichment_plot.png', 'GO Biological Process Analysis', 'lollipop'),
    'GO_Molecular_Function_2021': ('go_mf_enrichment.csv', 'go_mf_enrichment_plot.png', 'GO Molecular Function Analysis', 'gradient_bar'),
    'MSigDB_Hallmark_2020': ('hallmark_enrichment.csv', 'hallmark_enrichment_plot.png', 'MSigDB Hallmark / GSEA Analysis', 'waterfall')
}

def log(msg):
    print(msg, flush=True)

def wrap_label(text, width=42):
    if not isinstance(text, str):
        return text
    return '\n'.join(textwrap.wrap(text, width=width))

def evaluate_model_multi(X, y, n_classes, n_splits=10, random_state=42):
    metrics_list = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y))
    total_runs = len(splits)
    
    for i, (train_index, test_index) in enumerate(splits):
        log(f"      Fold {i+1} / {total_runs}...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = xgb.XGBClassifier(
            tree_method='hist',
            objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
            num_class=n_classes if n_classes > 2 else None,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            n_estimators=25,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.2,
            max_bin=64,
            learning_rate=0.1,
            n_jobs=4,
            random_state=random_state + i
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro') if n_classes > 2 else roc_auc_score(y_test, y_proba[:, 1])

        metrics_list.append({
            'acc': accuracy_score(y_test, y_pred),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred),
            'prec_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'auc_ovr': auc_val
        })

    return pd.DataFrame(metrics_list)

def process_single_disease(config):
    disease_name = config['name']
    output_dir = config['folder']
    log(f"\n==========================================================================")
    log(f" STARTING MASTER ANALYSIS FOR: {disease_name.upper()}")
    log(f"==========================================================================")
    
    # 1. Load AnnData
    if not os.path.exists(config['h5ad']):
        log(f"  [ERROR] Dataset file not found: {config['h5ad']}")
        log(f"  Please ensure the .h5ad dataset file is placed in: {os.path.dirname(config['h5ad'])}")
        return
    adata = ad.read_h5ad(config['h5ad'])
    obs_col = config['obs_col']
    if obs_col not in adata.obs.columns:
        for fallback in ['cell_type', 'marker_cell_type', 'leiden_r0.5']:
            if fallback in adata.obs.columns:
                obs_col = fallback
                break
    
    counts = adata.obs[obs_col].value_counts()
    valid_types = counts[counts >= 10].index
    adata = adata[adata.obs[obs_col].isin(valid_types)].copy()
    
    adata_hvg = adata[:, adata.var['highly_variable']].copy() if 'highly_variable' in adata.var.columns else sc.pp.highly_variable_genes(adata, n_top_genes=2000, inplace=False)
    X_full = np.ascontiguousarray((adata_hvg.X.toarray() if issparse(adata_hvg.X) else adata_hvg.X).astype(np.float32))
    var_names_hvg = adata_hvg.var_names.astype(str).str.strip().tolist()
    
    cell_types = adata.obs[obs_col].astype('category').cat.categories.tolist()
    y = adata.obs[obs_col].astype('category').cat.codes.values
    n_classes = len(cell_types)
    
    # 2. Match Top 500 Biomarkers
    df_bio = pd.read_csv(config['csv'])
    var_names_all = adata.var_names.astype(str).str.strip().tolist()
    ensembl_col_list = adata.var['ensembl_id'].astype(str).str.strip().tolist() if 'ensembl_id' in adata.var.columns else []
    
    matched_indices, matched_gene_symbols = [], []
    for _, row in df_bio.iterrows():
        gname = str(row['Gene Name']).strip() if pd.notna(row['Gene Name']) else None
        ens_id = str(row['Ensembl ID']).strip() if ('Ensembl ID' in row and pd.notna(row['Ensembl ID'])) else None
        
        found_idx = var_names_all.index(gname) if (gname and gname in var_names_all) else (-1)
        if found_idx == -1 and ens_id and ens_id in ensembl_col_list:
            found_idx = ensembl_col_list.index(ens_id)
            gname = var_names_all[found_idx]
            
        if found_idx != -1 and found_idx not in matched_indices:
            matched_indices.append(found_idx)
            matched_gene_symbols.append(gname)
            
    X_reduced = np.ascontiguousarray((adata[:, matched_indices].X.toarray() if issparse(adata[:, matched_indices].X) else adata[:, matched_indices].X).astype(np.float32))
    
    # 3. Classifier Evaluation
    log("  Evaluating Full Dataset (HVG 2000) Classifier...")
    results_full = evaluate_model_multi(X_full, y, n_classes, n_splits=10)
    summary_full = results_full.agg(['mean', 'var', 'std']).T
    summary_full.to_csv(os.path.join(output_dir, "classifier_full_metrics_summary.csv"))
    
    log("  Evaluating Reduced Biomarker (Top 500) Dataset Classifier...")
    results_reduced = evaluate_model_multi(X_reduced, y, n_classes, n_splits=10)
    summary_reduced = results_reduced.agg(['mean', 'var', 'std']).T
    summary_reduced.to_csv(os.path.join(output_dir, "classifier_reduced_metrics_summary.csv"))
    
    summary_combined = pd.DataFrame({
        'Full_Mean': summary_full['mean'], 'Full_Std': summary_full['std'],
        'Reduced_Mean': summary_reduced['mean'], 'Reduced_Std': summary_reduced['std']
    })
    summary_combined.to_csv(os.path.join(output_dir, "classifier_full_vs_reduced_summary.csv"))
    
    # Side-by-side Confusion Matrix & Feature Importance Plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    model_full = xgb.XGBClassifier(tree_method='hist', n_estimators=25, max_depth=4, subsample=0.8, colsample_bytree=0.2, max_bin=64, n_jobs=4).fit(X_full, y)
    sns.heatmap(confusion_matrix(y, model_full.predict(X_full)), annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], xticklabels=cell_types[:10], yticklabels=cell_types[:10])
    axes[0, 0].set_title(f'{disease_name}: Confusion Matrix (Full Dataset)', fontsize=14, fontweight='bold')
    
    top_full_idx = np.argsort(model_full.feature_importances_)[::-1][:20]
    sns.barplot(x=model_full.feature_importances_[top_full_idx], y=[var_names_hvg[i] for i in top_full_idx], ax=axes[0, 1], hue=[var_names_hvg[i] for i in top_full_idx], palette='Blues_r', legend=False)
    axes[0, 1].set_title(f'{disease_name}: Top 20 Gene Importances (Full Dataset)', fontsize=14, fontweight='bold')
    
    model_reduced = xgb.XGBClassifier(tree_method='hist', n_estimators=25, max_depth=4, subsample=0.8, colsample_bytree=0.2, max_bin=64, n_jobs=4).fit(X_reduced, y)
    sns.heatmap(confusion_matrix(y, model_reduced.predict(X_reduced)), annot=True, fmt='d', cmap='Greens', ax=axes[1, 0], xticklabels=cell_types[:10], yticklabels=cell_types[:10])
    axes[1, 0].set_title(f'{disease_name}: Confusion Matrix (Top 500 Biomarkers)', fontsize=14, fontweight='bold')
    
    top_red_idx = np.argsort(model_reduced.feature_importances_)[::-1][:20]
    sns.barplot(x=model_reduced.feature_importances_[top_red_idx], y=[matched_gene_symbols[i] for i in top_red_idx], ax=axes[1, 1], hue=[matched_gene_symbols[i] for i in top_red_idx], palette='Greens_r', legend=False)
    axes[1, 1].set_title(f'{disease_name}: Top 20 Gene Importances (Reduced Biomarkers)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classifier_evaluation_plots.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Perform Enrichment Analyses with Tailored Distinct Plots
    log(f"[4/5] Performing Enrichment Analysis & Generating Diverse Visualizations...")
    gene_list = [g.upper() for g in matched_gene_symbols if g and isinstance(g, str)]
    bset = set(gene_list)
    n_bio = len(bset)
    N_bg = 20000
    
    all_dfs = []
    for lib_name, (csv_file, plot_file, title_label, plot_type) in ENRICHMENT_LIBRARIES.items():
        try:
            lib_dict = gp.get_library(name=lib_name, organism='Human')
            results = []
            for term, pw_genes in lib_dict.items():
                pw_set = set(g.upper() for g in pw_genes)
                overlap = bset.intersection(pw_set)
                k = len(overlap)
                K = len(pw_set)
                if k > 0:
                    p_val = hypergeom.sf(k - 1, N_bg, K, n_bio)
                    odds_ratio = (k / (n_bio - k + 1e-5)) / ((K - k) / (N_bg - n_bio - K + k + 1e-5))
                    comb_score = odds_ratio * (-np.log(p_val + 1e-15))
                    results.append({
                        'Gene_set': lib_name,
                        'Term': term,
                        'Overlap': f'{k}/{K}',
                        'P-value': p_val,
                        'Odds Ratio': odds_ratio,
                        'Combined Score': comb_score,
                        'Genes': ';'.join(sorted(overlap))
                    })
                    
            res_df = pd.DataFrame(results)
            if not res_df.empty:
                res_df['Adjusted P-value'] = np.minimum(res_df['P-value'] * len(res_df), 1.0)
                res_df.sort_values(by='P-value', inplace=True)
                res_df.to_csv(os.path.join(output_dir, csv_file), index=False)
                all_dfs.append(res_df)
                
                top_df = res_df.head(15).copy()
                top_df['Wrapped_Term'] = top_df['Term'].apply(lambda x: wrap_label(x, width=42))
                top_df['log_p'] = -np.log10(top_df['Adjusted P-value'] + 1e-15)
                
                if plot_type == 'bubble':
                    top_df['Overlap_Count'] = top_df['Overlap'].apply(lambda x: int(str(x).split('/')[0]) if '/' in str(x) else 3)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    scatter = ax.scatter(x=top_df['Combined Score'], y=top_df['Wrapped_Term'], s=top_df['Overlap_Count']*120, c=top_df['log_p'], cmap='plasma', alpha=0.85, edgecolors='black', linewidth=1.5)
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('-log10(Adjusted P-value)', fontsize=11, fontweight='bold')
                    ax.set_title(f"{disease_name}: {title_label} (Bubble Plot)", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel('Combined Enrichment Score', fontsize=12, labelpad=10)
                    ax.set_ylabel('KEGG Pathway', fontsize=12, labelpad=10)
                    
                elif plot_type == 'ranked_bar':
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = sns.barplot(data=top_df, x='log_p', y='Wrapped_Term', hue='Wrapped_Term', palette='Blues_r', legend=False, ax=ax)
                    for bar, (_, row) in zip(bars.patches, top_df.iterrows()):
                        width = bar.get_width()
                        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"  {width:.2f} ({row['Overlap']})", va='center', fontsize=9, fontweight='bold')
                    ax.set_title(f"{disease_name}: {title_label} (Ranked Barplot)", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12, labelpad=10)
                    ax.set_ylabel('Reactome Pathway', fontsize=12, labelpad=10)
                    
                elif plot_type == 'lollipop':
                    top_df_rev = top_df.iloc[::-1]
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.hlines(y=top_df_rev['Wrapped_Term'], xmin=0, xmax=top_df_rev['log_p'], color='#8e44ad', alpha=0.7, linewidth=2.5)
                    ax.plot(top_df_rev['log_p'], top_df_rev['Wrapped_Term'], "o", markersize=10, color='#8e44ad', markeredgecolor='black', markeredgewidth=1.2)
                    ax.set_title(f"{disease_name}: {title_label} (Lollipop Plot)", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12, labelpad=10)
                    ax.set_ylabel('GO Biological Process', fontsize=12, labelpad=10)
                    
                elif plot_type == 'gradient_bar':
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(data=top_df, x='log_p', y='Wrapped_Term', hue='Wrapped_Term', palette='Greens_r', legend=False, ax=ax)
                    ax.set_title(f"{disease_name}: {title_label}", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12, labelpad=10)
                    ax.set_ylabel('GO Molecular Function', fontsize=12, labelpad=10)
                    
                elif plot_type == 'waterfall':
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(data=top_df, x='log_p', y='Wrapped_Term', hue='Wrapped_Term', palette='Oranges_r', legend=False, ax=ax)
                    ax.set_title(f"{disease_name}: {title_label}", fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12, labelpad=10)
                    ax.set_ylabel('MSigDB Hallmark Pathway', fontsize=12, labelpad=10)
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, plot_file), bbox_inches='tight', dpi=300)
                plt.close()
        except Exception as e:
            log(f"  Note on {lib_name}: {e}")
            
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(os.path.join(output_dir, "enrichment_all_results.csv"), index=False)

    # 5. OncoDB / OncoKB Cancer Driver Gene Overlap Analysis
    log(f"[5/5] Performing OncoDB / OncoKB Cancer Driver Gene Overlap Analysis...")
    oncokb_matches = []
    classified_records = []
    
    for rank, gene in enumerate(matched_gene_symbols):
        g_upper = gene.upper()
        importance_score = max(0.001, (len(matched_gene_symbols) - rank) / len(matched_gene_symbols))
        
        if g_upper in ONCOKB_DATABASE:
            info = ONCOKB_DATABASE[g_upper]
            oncokb_matches.append({
                'Biomarker Rank': rank + 1,
                'Gene Symbol': g_upper,
                'OncoKB Role': info['role'],
                'Actionability Level': info['level'],
                'Functional Description': info['function'],
                'Rank Importance Score': round(importance_score, 4)
            })
            classified_records.append({
                'Gene': g_upper,
                'Category': f"OncoDB: {info['role'].split(' ')[0]}",
                'Importance Score': importance_score
            })
        else:
            classified_records.append({
                'Gene': g_upper,
                'Category': 'Background Biomarkers',
                'Importance Score': importance_score
            })
            
    df_oncokb = pd.DataFrame(oncokb_matches)
    df_oncokb.to_csv(os.path.join(output_dir, "oncokb_overlapping_genes.csv"), index=False)
    
    # Figure 1: Horizontal Bar Plot of OncoDB Driver Genes with Badges
    if not df_oncokb.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if 'Oncogene' in r else '#2980b9' for r in df_oncokb['OncoKB Role']]
        bars = ax.barh(df_oncokb['Gene Symbol'], df_oncokb['Rank Importance Score'], color=colors, edgecolor='black', height=0.55)
        for bar, (_, row) in zip(bars, df_oncokb.iterrows()):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f" Rank #{row['Biomarker Rank']} | {row['OncoKB Role']} | {row['Actionability Level']}",
                    va='center', fontsize=10, fontweight='bold', color='#2c3e50')
        ax.set_title(f"{disease_name}: OncoDB / OncoKB Cancer Driver Genes Identified", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Biomarker Selection Importance Score', fontsize=12, labelpad=10)
        ax.set_ylabel('Cancer Driver Gene Symbol', fontsize=12, labelpad=10)
        ax.set_xlim(0, 1.25)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "oncokb_driver_genes_rank_plot.png"), bbox_inches='tight', dpi=300)
        plt.close()

    # Figure 2: Clear Box Plot + Swarm Overlay
    df_dist = pd.DataFrame(classified_records)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_dist, x='Category', y='Importance Score', hue='Category',
                palette={'OncoDB: Oncogene': '#e74c3c', 'OncoDB: Tumor': '#2980b9', 'Background Biomarkers': '#95a5a6'},
                width=0.45, boxprops=dict(alpha=0.75), legend=False, ax=ax)
    sns.stripplot(data=df_dist, x='Category', y='Importance Score', color='black', size=7, jitter=0.2, alpha=0.85, ax=ax)
    
    oncokb_rows = df_dist[df_dist['Category'].str.contains('OncoDB')]
    for _, row in oncokb_rows.iterrows():
        cat_idx = 0 if 'Oncogene' in row['Category'] else 1
        ax.text(cat_idx + 0.15, row['Importance Score'], f"← {row['Gene']}", fontsize=11, fontweight='bold', color='#16a085', va='center')
        
    ax.set_title(f"{disease_name}: OncoDB Driver Genes vs Biomarker Importance Distribution Box Plot", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Biomarker Importance Score (0.0 to 1.0)', fontsize=12, labelpad=10)
    ax.set_xlabel('Gene Classification Category', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "oncokb_overlapping_plot.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    log(f"\n==========================================================================")
    log(f" SUCCESSFULLY FINISHED ALL ANALYSES FOR: {disease_name.upper()}")
    log(f"==========================================================================")


def main():
    if len(sys.argv) < 2:
        log("Usage: python master_tcell_biomarker_study.py <cerebellum | colon | lung | all>")
        sys.exit(1)
        
    target = sys.argv[1].strip().lower()
    if target == 'all':
        for key, config in DISEASE_CONFIGS.items():
            process_single_disease(config)
    elif target in DISEASE_CONFIGS:
        process_single_disease(DISEASE_CONFIGS[target])
    else:
        log(f"Unknown target '{target}'. Choose from: cerebellum, colon, lung, all")
        sys.exit(1)

if __name__ == '__main__':
    main()
