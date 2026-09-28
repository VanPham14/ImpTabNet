# Uncertainty-Aware TabNet for Credit Scoring

An enhanced TabNet for credit scoring that combines periodic (PLR) numerical embeddings with a parameter-sharing BatchEnsemble, giving instance-level feature selection, competitive accuracy, and built-in predictive uncertainty in a single end-to-end model.

## Highlights

- **PLR embedding** for richer representation of continuous features.
- **BatchEnsemble** applied across the full TabNet encoder: k submodels share weights but produce diverse predictions in a single forward pass, yielding an uncertainty estimate at no extra inference cost.
- Evaluated on two real-world datasets (Home Credit, Taiwan) against 7 baselines (XGBoost, LightGBM, CatBoost, TabM, TabNet, MLP, Logistic Regression), with multi-seed runs, an ablation study, and DeLong significance testing.
- **Test AUC:** 0.787 (Home Credit) / 0.781 (Taiwan), competitive with strong GBDTs and consistently ahead of the original TabNet.
- Uncertainty is positively correlated with prediction error (Spearman ρ ≈ 0.27–0.29) and supports selective prediction (deferring uncertain cases to human review).

## Repository Structure

```
├── Parameters/                                  # Tuned hyperparameters (Optuna, 30 trials per model)
│   ├── HomeCredit_dts/
│   │   ├── PLR_Ensemble_gridsearch.xlsx         # Sweep over ensemble size k and embedding dim d
│   │   ├── best_params_baselines_NN.json        # Neural baselines
│   │   ├── best_params_baselines_Proposed.json  # Proposed model
│   │   └── best_params_baselines_T.json         # Tree-based models + Logistic Regression + TabM
│   └── Taiwan/
│       ├── PLR_Ensemble_gridsearch.xlsx         # Sweep over ensemble size k and embedding dim d
│       ├── best_params_baselines_NN.json        # Neural baselines
│       ├── best_params_baselines_Proposed.json  # Proposed model
│       ├── best_params_baselines_T.json         # Tree-based models + Logistic Regression
│       └── best_params_baselines_TabM.json      # TabM (stored separately on Taiwan)
├── SourceCode/
│   ├── HomeCredit_code.ipynb                    # Original pipeline for Home Credit (submitted version)
│   └── Taiwan_code.ipynb                        # Original pipeline for Taiwan (submitted version)
│   ├── update_HomeCredit_code.ipynb             # Updated pipeline for Home Credit (camera-ready: DeLong CIs, per-seed tests)
│   ├── update_Taiwan_code.ipynb                 # Updated pipeline for Taiwan (camera-ready: DeLong CIs, per-seed tests)
│   ├── HomeCredit_ablation_only.ipynb           # Ablation study for Home Credit (3 seeds, Table III)
│   └── Taiwan_ablation_only.ipynb               # Ablation study for Taiwan (3 seeds, Table III)
└── README.md
```

## Data

Both datasets are public and must be downloaded separately:

| Dataset | Source | Samples | Features | Default rate |
|---|---|---|---|---|
| Home Credit | [Kaggle: Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) | 307,511 | 148 (after feature engineering) | 0.081 |
| Taiwan | [UCI: Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) | 30,000 | 23 | 0.221 |

**Feature engineering.** All feature engineering was developed for this study. For Home Credit, the main application table is combined with six auxiliary tables (bureau, bureau balance, previous applications, POS cash balance, installment payments, credit card balance), aggregated at the applicant level, followed by derived financial ratios and external-score features, giving 148 features. Taiwan uses its original 23 variables. The same feature set is used for every model.

**Split.** Stratified 60% / 20% / 20% (train / validation / test). All preprocessing statistics (imputation, WOE, scaling) are fitted on the training split only.

## Hyperparameters

- Every model is tuned independently on each dataset with Optuna (TPE sampler, seed 42, 30 trials), selecting on validation AUC. Hyperparameters are not transferred between datasets.
- The selected values for all models are stored in `Parameters/<dataset>/`. Each JSON file maps a model name to its best parameters and best validation AUC.
- TabM parameters are stored in `best_params_baselines_T.json` for Home Credit, but in a separate file `best_params_baselines_TabM.json` for Taiwan.
- Tree-based models use early stopping on the validation set to set the number of estimators.
- Final ensemble size and embedding dimension of the proposed model: **k = 8, d = 8** (Home Credit) and **k = 4, d = 16** (Taiwan), chosen on the validation set (see `PLR_Ensemble_gridsearch.xlsx`).

## Usage

1. Clone the repo and install dependencies: PyTorch, XGBoost, LightGBM, CatBoost, Optuna, `pytorch-tabnet`, `tabm`, `entmax`.
2. Download the datasets (see [Data](#data)).
3. Open the notebook for the dataset in `SourceCode/` and run it from top to bottom. Each notebook is self-contained and includes the model implementation (PLR embedding, BatchEnsemble TabNet) and the full pipeline:
   - **A** Setup and helper functions
   - **B** Data loading, feature engineering, split, preprocessing, baseline models
   - **C** Proposed model training
   - **D** Multi-seed evaluation
   - **E** Ablation study
   - **F** DeLong test
   - **G** Feature importance from attention masks
   - **H** Uncertainty analysis and selective prediction
4. Optuna tuning cells are commented out by default; the notebooks load the saved parameters from `Parameters/<dataset>/`. Uncomment these cells to re-run tuning.

Results are averaged over three seeds (42, 1, 7).

## Citation

If you use this code, please cite:

```
V. T. T. Pham, "Uncertainty-Aware TabNet for Credit Scoring: Periodic Embeddings and Batch Ensembling," FAIR'2026.
```
