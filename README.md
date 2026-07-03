# CreditFusion-XAI

**CreditFusion-XAI: A Leakage-Aware Multi-Dataset Framework for Explainable Credit Default Prediction**

This repository contains the complete reproducible Python implementation for the ITAF 2026 / Springer proceedings manuscript.

## Repository Name

`CreditFusion-XAI`

## Description

Leakage-aware multi-dataset framework for explainable credit default prediction using machine learning, training-only SMOTE, cross-dataset validation, and permutation feature importance.

## Datasets

This project uses three public Kaggle datasets:

1. Bank Loan Data  
   https://www.kaggle.com/datasets/udaymalviya/bank-loan-data

2. Credit Risk Dataset  
   https://www.kaggle.com/datasets/laotse/credit-risk-dataset

3. Credit Risk Analysis  
   https://www.kaggle.com/datasets/nanditapore/credit-risk-analysis

## Main Features

- Multi-dataset credit-risk integration
- Schema harmonization
- Duplicate removal
- Leakage-aware preprocessing
- Source-dataset exclusion from features
- Training-only SMOTE
- Stratified-random validation
- Leave-one-dataset-out validation
- Six ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - HistGradientBoosting
  - XGBoost
  - KNN
- Accuracy, balanced accuracy, precision, recall, F1, MCC, FPR, FNR, ROC-AUC, and PR-AUC
- Permutation feature importance
- Publication-ready CSV tables and figures

## Repository Structure

```text
CreditFusion-XAI/
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
├── src/
│   └── creditfusion_xai.py
├── data/
│   └── README.md
├── results/
│   ├── creditfusion_model_results.csv
│   ├── creditfusion_best_by_split.csv
│   ├── creditfusion_feature_importance.csv
│   └── dataset_summary.csv
├── figures/
└── notebooks/
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/CreditFusion-XAI.git
cd CreditFusion-XAI
pip install -r requirements.txt
```

## Kaggle Access

The script uses `kagglehub` first. If Kaggle CLI is needed, create a Kaggle API token:

Kaggle → Account → Create New Token

Then place `kaggle.json` in:

```bash
~/.kaggle/kaggle.json
```

Set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Run the Main Experiment

Fast reproducible run, matching the manuscript setting:

```bash
python src/creditfusion_xai.py \
  --output-dir outputs \
  --max-rows-per-dataset 30000 \
  --seed 42 \
  --test-size 0.20 \
  --n-repeats-importance 5 \
  --importance-sample-size 5000
```

Full-data run:

```bash
python src/creditfusion_xai.py \
  --output-dir outputs_full \
  --max-rows-per-dataset None \
  --seed 42
```

## Outputs

The script creates:

```text
outputs/
├── dataset_summary.csv
├── merged_standardized_credit_data.csv
├── creditfusion_model_results.csv
├── creditfusion_best_by_split.csv
├── creditfusion_feature_importance.csv
└── figures/
    ├── Fig2_Stratified_Model_Comparison.png
    ├── Fig3_Cross_Dataset_Robustness.png
    ├── Fig4_Permutation_Feature_Importance.png
    └── Fig5_Correlation_Heatmap.png
```

## Validation Protocol

The manuscript reports two validation protocols:

1. **Stratified random split**  
   Evaluates within-distribution performance.

2. **Leave-one-dataset-out validation**  
   Evaluates cross-dataset robustness by training on two datasets and testing on the third dataset.

SMOTE is applied only to the training data inside the pipeline to prevent test-set leakage.

## Representative Manuscript Results

Using 30,000 records from each dataset:

| Setting | Best Model | Accuracy | F1 | MCC | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Stratified random | Random Forest | 0.9316 | 0.8393 | 0.7967 | 0.9714 |
| Held-out Bank Loan Data | Random Forest | 0.8722 | 0.6499 | 0.5973 | 0.8841 |
| Held-out Credit Risk Analysis | Random Forest | 0.9339 | 0.8323 | 0.7991 | 0.9682 |
| Held-out Credit Risk Dataset | Random Forest | 0.9841 | 0.9630 | 0.9532 | 0.9992 |

## Data Availability Statement

The datasets are publicly available from Kaggle:

- https://www.kaggle.com/datasets/udaymalviya/bank-loan-data
- https://www.kaggle.com/datasets/laotse/credit-risk-dataset
- https://www.kaggle.com/datasets/nanditapore/credit-risk-analysis

## Code Availability Statement

The complete Python implementation used for preprocessing, model training, validation, and result generation is available in this repository.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{abdelsalam2026creditfusion,
  title={CreditFusion-XAI: A Leakage-Aware Multi-Dataset Framework for Explainable Credit Default Prediction},
  author={Abd-Elsalam, Shimaa M. and Sayed, Hosam Eldin Fawzan},
  booktitle={Proceedings of ITAF 2026},
  year={2026}
}
```

## License

This repository is released under the MIT License.
