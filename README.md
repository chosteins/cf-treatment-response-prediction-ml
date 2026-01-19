# Machine Learning for Predicting CFTR Modulator Response from Clinical and Microbiome Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete code and synthetic data for reproducing the machine learning pipeline described in:

> **Hosteins, C., et al.** (2025). Machine Learning for Predicting Continuous and Binary CFTR Modulator Response from Clinical and Microbiome Data. *Proceedings of CIBB 2025*

## Overview

This project implements a comprehensive machine learning pipeline to predict nutritional response to CFTR modulator therapy in pediatric cystic fibrosis patients using:
- **Clinical features**: BMI, age, lung function, bacterial colonization
- **Lung microbiome data**: 16S rRNA sequencing with 62 bacterial genera
- **Multiple ML approaches**: LASSO, Random Forest, FLORAL, Graph Neural Networks

### Key Features
- Multiple compositional data transformations (ALR, CLR, rCLR, ILR, PA, arcsine)
- Five distinct modeling approaches for comparison
- Phylogeny-informed GNN using taxonomic hierarchies
- 20-fold cross-validation
- Comprehensive evaluation metrics (AUROC, AUPRC, RMSE, MAE)

## Repository Structure

```
.
├── data/                          # Simulated patient data
│   ├── simulated_microbiome.csv   # Main dataset (73 patients × 2 timepoints)
│   ├── taxa_table_grouped_renamed.csv  # Taxonomic hierarchy
│   └── transformed/               # Pre-computed transformations (generated)
├── src/                           # Analysis scripts
│   ├── 00_transformation_preprocessing.R  # Compositional transformations
│   ├── 01_run_logistic.R         # Logistic regression baseline
│   ├── 02_run_lasso.R            # LASSO with multiple transformations
│   ├── 03_run_rf.R               # Random Forest on CLR data
│   ├── 04_run_floral.R           # FLORAL (log-ratio LASSO)
│   ├── 05_run_gnn.py             # Graph Neural Network
│   ├── splits/                    # Cross-validation fold assignments
│   └── results/                   # Model outputs (generated)
├── utils/                         # Helper functions
│   ├── data_loading.R            # Data loading utilities
│   ├── evaluation.R              # Performance metrics
│   └── gnn_utils.py              # GNN-specific functions
├── run_all.sh                     # Bash pipeline automation
├── run_all.py                     # Python pipeline automation
├── Makefile                       # Make-based automation
├── requirements.txt               # R and Python dependencies
└── README.md                      # This file
```

## Quick Start

### Prerequisites

**Required software:**
- R ≥ 4.0.0
- Python ≥ 3.8
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cftr-microbiome-ml.git
   cd cftr-microbiome-ml
   ```

2. **Install R packages**
   ```r
   # Open R and run:
   install.packages(c("dplyr", "tidyr", "glmnet", "ranger", "pROC", 
                      "PRROC", "caret", "compositions", "zCompositions", 
                      "matrixStats"))
   
   # Install FLORAL from GitHub
   remotes::install_github("tinglab/FLORAL")
   ```

3. **Install Python packages**
   ```bash
   pip install torch torch-geometric pandas numpy scikit-learn ete3
   ```
   
   > **Note:** PyTorch Geometric may require additional dependencies. See [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

4. **Verify installation**
   ```bash
   make check
   # or
   python run_all.py --help
   ```

### Running the Pipeline

**Option 1: Complete pipeline (recommended for first run)**
```bash
bash run_all.sh
# or
python run_all.py
# or
make all
```

**Option 2: Run specific models**
```bash
# Run only LASSO and Random Forest
bash run_all.sh --models lasso,rf

# Skip preprocessing (if already done)
bash run_all.sh --skip-preprocessing

# Using Make
make lasso rf
```

**Option 3: Manual execution**
```bash
cd src

# Step 1: Data transformations
Rscript 00_transformation_preprocessing.R

# Step 2: Run models
Rscript 01_run_logistic.R
Rscript 02_run_lasso.R
Rscript 03_run_rf.R
Rscript 04_run_floral.R
python3 05_run_gnn.py
```

### Expected Runtime
- Preprocessing: ~1 minute
- Logistic regression: ~2 minutes
- LASSO: ~10 minutes
- Random Forest: ~15 minutes
- FLORAL: ~20 minutes
- GNN: ~30 minutes

**Total pipeline: ~1.5 hours** (with 20-fold CV)

## Results

Results are saved in `src/results/` with the following structure:

```
src/results/
├── logistic/
│   ├── logistic_results_summary.csv
│   └── logistic_zscore_*.rds
├── lasso/
│   ├── lasso_results_summary.csv
│   ├── lasso_selected_variables.csv
│   └── lasso_*_transformed.rds
├── rf/
│   ├── rf_results_summary.csv
│   ├── rf_selected_variables.csv
│   └── rf_CLR_*.rds
├── floral/
│   ├── floral_results_summary.csv
│   ├── floral_selected_variables.csv
│   └── floral_zscore_*.rds
└── gnn/
    ├── gnn_results_summary.csv
    ├── gnn_zscore_label_results.csv
    └── gnn_model.pt
```

### Key Output Files

- `*_results_summary.csv`: Performance metrics (AUROC, AUPRC, RMSE, MAE) for each outcome
- `*_selected_variables.csv`: Variables selected by feature selection methods
- `*.rds` / `*.pt`: Trained model objects

### Example Results Inspection

```r
# Load LASSO results
results <- read.csv("src/results/lasso/lasso_results_summary.csv")
print(results)

# View selected features
features <- read.csv("src/results/lasso/lasso_selected_variables.csv")
print(features)
```

## Methods

### Data Preprocessing

**Compositional Transformations:**
- **Relative Abundance (RA)**: Direct proportions
- **Presence-Absence (PA)**: Binary encoding
- **Additive Log-Ratio (ALR)**: Reference taxon: *Granulicatella*
- **Centered Log-Ratio (CLR)**: Geometric mean reference
- **Robust CLR (rCLR)**: Median-based for zero-robustness
- **Isometric Log-Ratio (ILR)**: Orthonormal basis
- **Arcsine Square Root**: Variance stabilization

**Zero Handling:** Bayesian-multiplicative replacement (zCompositions)

### Models

1. **Logistic Regression Baseline**: Clinical covariates only (no microbiome)
2. **LASSO**: L1-regularized regression across all transformations
3. **Random Forest**: Non-parametric ensemble on CLR data
4. **FLORAL**: Log-ratio LASSO with compositional constraints
5. **Graph Neural Network (GNN)**: Phylogeny-informed with Graph Isomorphism Network (GIN)

### Evaluation

- **Binary outcome** (nutritional status): AUROC, AUPRC
- **Continuous outcome** (BMI change): RMSE, MAE
- **Cross-validation**: 20-fold stratified

## Data Description

### Simulated Dataset

The `simulated_microbiome.csv` contains synthetic data for **73 pediatric CF patients** (146 rows with M0 and M12 timepoints):

**Clinical variables:**
- Demographics: Age, Sex
- Anthropometry: BMI z-score, nutritional status
- Lung function: ppFEV1
- Colonization: *P. aeruginosa*, *A. fumigatus*

**Microbiome features:**
- 62 bacterial genera (prefix: `b_*`)
- Alpha diversity: Chao1, Shannon, Simpson, Pielou

**Outcomes:**
- `zscore_binary`: Nutritional status (0=Thin, 1=Normal/Overweight)
- `delta`: Change in BMI z-score (continuous)

> **Important:** This dataset is **simulated** to preserve statistical properties while ensuring patient privacy. Real data are not publicly available due to ethical constraints.

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{hosteins2025cftr,
  title={Machine Learning for Predicting Continuous and Binary CFTR Modulator Response from Clinical and Microbiome Data},
  author={Hosteins, C{\'e}line and [Other Authors]},
  booktitle={Proceedings of the 20th International Conference on Computational Intelligence Methods for Bioinformatics and Biostatistics (CIBB 2025)},
  year={2025},
  organization={Springer}
}
```

## Contact

- **Céline Hosteins**: celine.hosteins@u-bordeaux.fr
- **Issues**: [GitHub Issues](https://github.com/chosteins/cf-treatment-response-prediction-ml/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Study conducted at Bordeaux Population Health (BPH), Bordeaux, France
- Funded by BPH

## Troubleshooting

### Common Issues


## Version History

- **v1.0.0** (2025-01): Initial release for CIBB 2025

---

**Last updated:** January 19th, 2025
