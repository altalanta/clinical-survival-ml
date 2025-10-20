# Methods Summary

## Cross-validation & Leakage Prevention
- All preprocessing (imputation, scaling, encoding) is wrapped inside an sklearn `Pipeline` and refit within each cross-validation fold.
- Nested CV uses stratified outer/inner folds; transformers fitted on a fold train split are *not* reused elsewhere.
- Out-of-fold (OOF) predictions are aggregated across folds to produce unbiased estimates.

## Metric Computation
- **Concordance (C-index):** Harrell's estimator computed from OOF risk scores.
- **Integrated Brier Score / Brier@τ:** Uses `sksurv.metrics.brier_score` with IPCW weights to correct for censoring.
- **Time-dependent AUC:** `sksurv.metrics.cumulative_dynamic_auc` evaluated on OOF predictions.
- **Uncertainty:** Metrics are bootstrapped (default 200 replicates) to report 95% percentile intervals.

## Calibration (IPCW)
- For each horizon τ, survival predictions are binned. Observed survival is estimated via inverse probability of censoring weighting (IPCW):
  - Censoring distribution \(\hat G(t)\) is estimated using Kaplan–Meier on \((T_i, 1-\delta_i)\).
  - Weights: \(w_i(τ) = \frac{\mathbb{1}[T_i > τ]}{\hat G(τ)} + \frac{\mathbb{1}[T_i \le τ, \delta_i=1]}{\hat G(T_i)}\).
  - Observed survival in each bin is \(\sum_i w_i(τ) \mathbb{1}[T_i > τ] / \sum_i w_i(τ)\).
- Normal-approximation bands are added per bin.

## Decision-Curve Analysis
- Net benefit at horizon τ and threshold \(p_t\):
  - Weighted true positives \(TP = \sum_i w_i(τ) \mathbb{1}[T_i \le τ, \delta_i=1, \hat r_i \ge p_t]\).
  - Weighted false positives \(FP = \sum_i w_i(τ) \mathbb{1}[T_i > τ, \hat r_i \ge p_t]\).
  - Net benefit \(NB = TP/n - FP/n \cdot p_t/(1-p_t)\).
- Treat-all and treat-none strategies are plotted alongside the model curve.

## XGBoost AFT Survival
- Survival is derived from the fitted AFT distribution: \(S(t) = 1 - F((\log t - \mu)/\sigma)\).
- Supported loss distributions: `normal` and `logistic` with scale parameter `aft_loss_distribution_scale`.
- Survival matrices are clipped to \([0,1]\) and enforced to be monotone non-increasing in time.
