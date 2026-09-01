# Master T-Cell Biomarker Study: Complete Project & Code Explanation

> **PROMPT FOR CHATGPT / LLM TUTOR**:
> *"I am preparing for an interview in Bioinformatics, Machine Learning, and Data Science. Below is the complete technical documentation, mathematical models, and code breakdown for my **Master T-Cell Biomarker Study Pipeline**. Please read this document carefully. Afterwards, test me with mock interview questions, explain any concepts I ask about in detail, and help me refine my explanations."*

---

## 1. 📌 Executive Project Summary

* **Project Title**: Master T-Cell Biomarker Study Pipeline (Single-Cell Machine Learning & Pathway Enrichment)
* **Target Disease Cohorts**: Cerebellum, Colon Cancer, and Lung Cancer scRNA-seq datasets.
* **Core Goal**: Identify, benchmark, and functionally annotate T-cell biomarkers across single-cell expression datasets, comparing a **Full 2,000 Highly Variable Gene (HVG)** dataset against a **Reduced Top 500 Biomarker** panel to validate model parsimony and classification accuracy.
* **Primary Stack**: Python, `scanpy`, `anndata`, `xgboost`, `scikit-learn`, `scipy.stats` (hypergeometric tests), `gseapy`, `matplotlib`, `seaborn`.

---

## 2. 🏗️ Complete Workflow Architecture

```
                                 [ DATA INPUT LAYER ]
                  Single-Cell AnnData (.h5ad) + Biomarker Metadata (.csv)
                  Cohorts: Cerebellum | Colon Cancer | Lung Cancer
                                           │
                                           ▼
                            [ DUAL GENE MATCHING ENGINE ]
                  Robust Mapping via HGNC Symbols + Ensembl Identifiers
                                           │
                                           ▼
                      [ MACHINE LEARNING BENCHMARK ENGINE ]
                    XGBoost 10-Fold Stratified Cross-Validation
                Comparison: Full 2000 HVGs vs. Reduced Top 500 Biomarkers
                  8 Metrics: Acc, Bal Acc, Prec, Rec, F1, MCC, Kappa, AUC
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    ▼                                              ▼
    [ PATHWAY ENRICHMENT ENGINE ]                   [ ONCOKB DRIVER MATCHING ]
  Hypergeometric Test (N=20,000)                  Cross-reference OncoDB Target DB
  • KEGG (Bubble Plot)                            • Importance Rank Barplot
  • Reactome (Ranked Barplot)                     • Boxplot + Swarm Distribution
  • GO BP (Lollipop Plot)                         • Actionability Level Badges
  • GO MF (Gradient Barplot)                      (FDA Targetable, Standard of Care)
  • MSigDB Hallmark (Waterfall)
```

---

## 3. 🔬 Comprehensive Technical Component Breakdown

### Step 1: Data Ingestion & AnnData Processing
- Reads single-cell AnnData (`.h5ad`) files containing single-cell transcriptomic expression matrices.
- Filters out rare cell types with fewer than 10 cells (`adata.obs['final_cell_type'].value_counts() >= 10`) to eliminate cross-validation fold instantiation crashes.
- Converts SciPy sparse CSR matrices (`csr_matrix`) to continuous C-aligned `float32` NumPy arrays for optimal XGBoost memory alignment.
- Extracts the top 2,000 Highly Variable Genes (HVGs) using `scanpy.pp.highly_variable_genes`.

### Step 2: Dual Gene Identifier Matching Engine
- Addresses gene nomenclature mismatches between HGNC Gene Symbols (e.g. `CD3E`) and Ensembl IDs (e.g. `ENSG00000198851`).
- Matching logic:
  1. Searches for exact HGNC Symbol match in `adata.var_names`.
  2. If unmapped, searches by Ensembl ID in `adata.var['ensembl_id']`.
- Constructs the reduced expression sub-matrix ($X_{\text{reduced}}$) containing ~500 matched biomarkers.

### Step 3: XGBoost 10-Fold Stratified Cross-Validation Benchmark
- **Validation Scheme**: `StratifiedKFold(n_splits=10, shuffle=True)` ensures identical cell-type class proportions in every train/validation split, preventing data leakage.
- **Model Architecture**: XGBoost Classifier (`xgb.XGBClassifier`) with:
  - `tree_method='hist'`: Fast histogram-based binning on continuous expression values.
  - `colsample_bytree=0.2`: Subsamples 20% of features per tree to prevent dominant genes from masking weaker secondary signals.
  - `objective='multi:softprob'`: Predicts multi-class probability vectors.
- **8 Quantitative Evaluation Metrics Calculated per Fold**:
  1. **Accuracy**: Total correct predictions over total samples.
  2. **Balanced Accuracy**: Macro-average recall across all cell types.
  3. **Precision (Macro)**: Unweighted mean of precision per class.
  4. **Recall (Macro)**: Unweighted mean of sensitivity per class.
  5. **F1-Score (Macro)**: Harmonic mean of precision and recall.
  6. **Matthews Correlation Coefficient (MCC)**: Symmetrical confusion matrix evaluation ranging from -1 to +1 (best metric for imbalanced scRNA-seq classes).
  7. **Cohen's Kappa**: Inter-rater agreement adjusting for chance agreement.
  8. **ROC-AUC (One-vs-Rest Macro)**: Area under ROC curve integrated across multi-class predictions.

### Step 4: Hypergeometric Bio-Enrichment Engine
- Evaluates statistical over-representation using the survival function of the Hypergeometric Distribution (`scipy.stats.hypergeom.sf`):
  $$P(X \ge k) = \sum_{x=k}^{K} \frac{\binom{K}{x} \binom{N - K}{n - x}}{\binom{N}{n}}$$
  Where $N=20,000$ (background human genome size), $n = \text{biomarker panel size}$, $K = \text{pathway size}$, $k = \text{observed overlap}$.
- Calculates **Combined Score**: $\text{Odds Ratio} \times (-\ln P_{\text{adj}})$.
- Generates 5 custom 300 DPI publication plots:
  - **KEGG Pathways**: Bubble / Dot Plot (Node size = Overlap, Color = $-\log_{10} P$).
  - **Reactome Pathways**: Ranked Horizontal Barplot with metric badges.
  - **GO Biological Process**: Stem-and-node Lollipop Plot.
  - **GO Molecular Function**: Color-Gradient Barplot.
  - **MSigDB Hallmark**: Waterfall Barplot.

### Step 5: OncoDB / OncoKB Translational Oncology Mapping
- Cross-references biomarkers against curated OncoKB FDA actionability target databases (*MTOR*, *PIK3CD*, *ARID1A*, *TP53*, *PTEN*, *KRAS*, *EGFR*, *BRAF*, *MYC*, *CDK4*, *ATM*, *BRCA1/2*, *APC*, *RB1*, *VHL*, *ALK*, *KIT*, *NOTCH1*).
- Calculates normalized Rank Importance Score:
  $$\text{Score} = \frac{N_{\text{biomarkers}} - \text{Rank}}{N_{\text{biomarkers}}}$$
- Generates OncoKB Importance Rank Barplots with clinical badges (FDA Level 1 Targetable, Level 2 Standard of Care) and Boxplot + Swarm Plot Overlay comparing Oncogenes vs. Tumor Suppressor Genes vs. Background Biomarkers.

---

## 4. 🐍 Code Implementation Walkthrough (`master_tcell_biomarker_study.py`)

```python
# Key imports
import os, sys, textwrap
import numpy as np, pandas as pd
import anndata as ad, scanpy as sc, xgboost as xgb
from scipy.stats import hypergeom
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns, gseapy as gp

# Dynamic relative path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 10-Fold Stratified Evaluation Function
def evaluate_model_multi(X, y, n_classes, n_splits=10, random_state=42):
    metrics_list = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = xgb.XGBClassifier(
            tree_method='hist',
            objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
            n_estimators=25, max_depth=4, subsample=0.8, colsample_bytree=0.2, max_bin=64, learning_rate=0.1, n_jobs=4
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
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
```

---

## 5. ❓ Expected Interview Technical Q&A

### Q1: Why compare 2000 HVGs vs 500 Biomarkers?
**Answer**: To test model parsimony. In clinical settings, a 2,000-gene assay is cost-prohibitive. Proving that a 500-gene subset achieves virtually identical F1, MCC, and ROC-AUC scores demonstrates that feature reduction preserves discriminating biological signal without loss of predictive power.

### Q2: Why use MCC alongside F1-Score?
**Answer**: Single-cell cell types are highly imbalanced. F1-Score can be driven up by majority cell classes. Matthews Correlation Coefficient (MCC) evaluates all four quadrants of the confusion matrix (TP, TN, FP, FN) symmetrically, making it the most reliable metric for imbalanced genomics classification.

### Q3: How was data leakage prevented?
**Answer**: Cross-validation was performed using `StratifiedKFold`. Model fitting, hyperparameter evaluation, and predictions were strictly isolated within each of the 10 training/validation split loops.

### Q4: Why use XGBoost over Deep Learning or Random Forest?
**Answer**: scRNA-seq expression matrices are high-dimensional and non-linear. XGBoost handles sparsity natively without dense imputation, trains significantly faster using histogram tree binning (`tree_method='hist'`), and outputs feature importances for downstream biological pathway enrichment.
