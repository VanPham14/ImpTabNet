Uncertainty-Aware TabNet for Credit Scoring
An enhanced TabNet for credit scoring that combines periodic (PLR) numerical embeddings with a parameter-sharing BatchEnsemble, giving instance-level feature selection, competitive accuracy, and built-in predictive uncertainty — all in a single end-to-end model.

Highlights
PLR embedding for richer representation of continuous features.
BatchEnsemble applied across the full TabNet encoder: k submodels share weights but produce diverse predictions in a single forward pass, yielding an uncertainty estimate at no extra inference cost.
Evaluated on two real-world datasets (Home Credit, Taiwan) against 7 baselines (XGBoost, LightGBM, CatBoost, TabM, TabNet, MLP, Logistic Regression), with multi-seed runs, ablation study, and DeLong significance testing.
Test AUC: 0.787 (Home Credit) / 0.781 (Taiwan) — competitive with strong GBDTs, consistently ahead of the original TabNet.
Uncertainty is positively correlated with prediction error (Spearman ρ ≈ 0.27–0.29) and supports selective prediction (deferring uncertain cases to human review).
Repository Structure
├── notebooks/          # Data prep, training, evaluation, ablation, uncertainty analysis
├── src/                 # Model implementation (PLR embedding, BatchEnsemble TabNet)
└── README.md
Usage
Clone the repo and install dependencies (PyTorch, XGBoost, LightGBM, CatBoost, Optuna).
Run the notebooks in order: preprocessing → baseline tuning → proposed model training → evaluation.
Citation
If you use this code, please cite:

V. T. T. Pham, "Uncertainty-Aware TabNet for Credit Scoring: Periodic Embeddings and Batch Ensembling," FAIR'2026.
