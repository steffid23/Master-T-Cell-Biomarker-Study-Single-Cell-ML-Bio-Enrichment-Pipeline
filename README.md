# Master T-Cell Biomarker Study Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Single-Cell Machine Learning](https://img.shields.io/badge/Single--Cell-Machine%20Learning-brightgreen.svg)](https://scanpy.readthedocs.io/)
[![XGBoost Benchmark](https://img.shields.io/badge/ML-XGBoost%2010--Fold%20CV-orange.svg)](https://xgboost.readthedocs.io/)

A production-ready single-cell machine learning and bio-enrichment pipeline designed for robust T-cell biomarker discovery and functional validation across diverse disease cohorts (**Cerebellum**, **Colon Cancer**, and **Lung Cancer**).

---

## Overview

This study integrates high-dimensional single-cell RNA-sequencing (scRNA-seq) datasets with gradient boosted machine learning classifiers and hyper-geometric pathway enrichment to identify, benchmark, and functionally annotate key cell-type biomarkers.

The pipeline compares machine learning performance between **Full HVG Datasets (2,000 Highly Variable Genes)** and **Reduced Biomarker Datasets (Top 500 Candidate Biomarkers)** to validate feature selection fidelity and model parsimony.

```
                    ┌──────────────────────────────────────────────┐
                    │    Single-Cell AnnData Datasets (.h5ad)      │
                    │   (Cerebellum | Colon Cancer | Lung Cancer)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │    Dual Gene Identifier Matching Engine     │
                    │       (HGNC Symbol + Ensembl ID Mapping)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │    10-Fold Stratified CV Benchmark           │
                    │   (XGBoost Full 2000 HVG vs Top 500 Bio)     │
                    │   Evaluates 8 Publication-Grade Metrics      │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    ▼                                              ▼
┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
│     Functional Pathway Enrichment       │    │     OncoDB / OncoKB Driver Overlap      │
│  • KEGG Pathway (Bubble Plot)           │    │  • Importance Rank Barplot              │
│  • Reactome Pathway (Ranked Barplot)    │    │  • Boxplot & Swarm Distribution         │
│  • GO Biological Process (Lollipop)     │    │  • Actionability & Functional Badges    │
│  • GO Molecular Function (Gradient Bar) │    └─────────────────────────────────────────┘
│  • MSigDB Hallmark (Waterfall Barplot)  │
└─────────────────────────────────────────┘
```

---

## Key Features

1. **Dual Gene Matching Engine**: Resolves discrepancies between HGNC gene symbols and Ensembl IDs across single-cell AnnData objects and candidate biomarker tables.
2. **10-Fold Stratified Cross-Validation Benchmark**:
   - Rigorously evaluates multi-class cell-type classification using XGBoost (`hist` tree method).
   - Computes **8 quantitative metrics**: Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1-Score, Matthews Correlation Coefficient (MCC), Cohen's Kappa, and Macro ROC-AUC.
3. **Publication-Grade Visualizations (300 DPI)**:
   - Side-by-side Confusion Matrices and Top-20 Feature Importances.
   - Dynamic, publication-formatted figures (Bubble plots, Ranked Barplots, Lollipop diagrams, Gradient Barplots, and Waterfall plots).
   - Automatic multi-line label wrapping and tight bounding box padding preventing text clipping.
4. **OncoDB & OncoKB Driver Gene Mapping**:
   - Cross-references biomarker profiles against FDA-targetable oncogenes and tumor suppressors (e.g., *MTOR*, *PIK3CD*, *ARID1A*, *TP53*, *PTEN*, *KRAS*, *EGFR*, *BRAF*, *MYC*, *CDK4*, *ATM*, *BRCA1/2*, *APC*, *RB1*, *VHL*, *ALK*, *KIT*, *NOTCH1*).

---

## Repository Structure

```
t-cell biomarker study/
├── master_tcell_biomarker_study.py   # Main pipeline orchestrator script
├── requirements.txt                  # Python dependency specifications
├── .gitignore                        # Git exclusion rules for raw .h5ad datasets & caches
├── LICENSE                           # MIT Open Source License
├── README.md                         # Project documentation
│
├── cerebellum/                       # Cerebellum cohort outputs & metadata
│   ├── annotated/                    # Annotated single-cell datasets
│   ├── classifier/                   # ML evaluation matrices & plots
│   ├── enrichment/                   # KEGG, Reactome, GO, Hallmark CSVs & PNGs
│   └── top_500_biomarkers_Ensembl_gene_names.csv
│
├── colon cancer/                     # Colon Cancer cohort outputs & metadata
│   ├── annotation/                   # Annotated single-cell datasets
│   ├── classifier/                   # ML evaluation matrices & plots
│   ├── enrichment/                   # Pathway & Driver gene results
│   └── Model_Full_top_500_biomarkers_1_with_Ensembl_and_Gene_Names.csv
│
└── lung cancer/                      # Lung Cancer cohort outputs & metadata
    ├── annotated files/              # Annotated single-cell datasets
    ├── classifier/                   # ML evaluation matrices & plots
    ├── enrichment/                   # Pathway & Driver gene results
    └── Model_Full_top_500_biomarkers_with_Ensembl_and_Gene_Names.csv
```

---

## Installation & Requirements

### Prerequisites
- **Python 3.9+**
- Recommended virtual environment setup:

```bash
# Clone the repository
git clone https://github.com/<your-username>/t-cell-biomarker-study.git
cd t-cell-biomarker-study

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## Usage

The master pipeline can be executed for specific disease cohorts or run globally across all cohorts:

```bash
# Run analysis for Cerebellum cohort
python master_tcell_biomarker_study.py cerebellum

# Run analysis for Colon Cancer cohort
python master_tcell_biomarker_study.py colon

# Run analysis for Lung Cancer cohort
python master_tcell_biomarker_study.py lung

# Run comprehensive master pipeline across ALL cohorts
python master_tcell_biomarker_study.py all
```

---

## Pipeline Outputs

Running the pipeline populates each cohort directory with standardized CSV tables and high-resolution figures:

| Output Category | Filename / Artifact | Description |
| :--- | :--- | :--- |
| **Classifier Evaluation** | `classifier_full_vs_reduced_summary.csv` | Summary of 10-fold CV performance comparing 2000 HVGs vs 500 Biomarkers. |
| **Model Plots** | `classifier_evaluation_plots.png` | 4-panel figure with side-by-side Confusion Matrices & Top 20 Feature Importances. |
| **Enrichment CSVs** | `enrichment_all_results.csv` | Combined pathways from KEGG, Reactome, GO BP, GO MF, and MSigDB Hallmark. |
| **KEGG Pathway** | `kegg_significant_pathways_plot.png` | Bubble plot (Bubble size = Overlap count, Color = Significance). |
| **Reactome Pathway** | `reactome_enrichment_plot.png` | Ranked horizontal barplot with metric badges. |
| **GO Biological Process** | `go_bp_enrichment_plot.png` | Stem-and-node Lollipop plot visualization. |
| **GO Molecular Function**| `go_mf_enrichment_plot.png` | Color-gradient barplot with metric labels. |
| **OncoKB Driver Genes** | `oncokb_driver_genes_rank_plot.png` | Importance rank barplot for actionability levels (FDA Targetable, Standard Care). |
| **Driver Distribution** | `oncokb_overlapping_plot.png` | Boxplot with swarm plot overlay comparing Oncogenes, Tumor Suppressors & Background. |

---

## Methodological Details

### 1. Stratified 10-Fold Cross-Validation
To eliminate data leakage and sample bias:
- Datasets are partitioned into 10 stratified folds matching exact cell-type proportions.
- XGBoost classifier is configured with histogram tree building (`tree_method='hist'`) and multi-class softprob objective.

### 2. Hypergeometric Enrichment Test
Enrichment significance is calculated via the survival function of the hypergeometric distribution:
$$P = \sum_{x=k}^{K} \frac{\binom{K}{x} \binom{N - K}{n - x}}{\binom{N}{n}}$$
where $N$ is background genome size (20,000), $n$ is biomarker list length, $K$ is pathway gene set size, and $k$ is observed gene overlap.

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
