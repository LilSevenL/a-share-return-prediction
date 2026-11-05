# V1: historical research code

This directory collects the V1 source files retained in this repository: data acquisition and preparation, the original 50-factor library, five-day targets, rolling model training and tuning, backtesting, final retraining and the already-inspected historical test.

These are the retained historical V1 implementations. The directory reorganization only adapts repository/data paths; it does not change factor formulas, fitting methods or trading rules. Fitted models and vendor data are not bundled.

## Code map

| Stage | Main files |
|---|---|
| Data acquisition | `main.py`, `download_all_stocks.py`, `download_extra.py`, `download_minutes.py` |
| Data preparation | `preprocess_data.py`, `build_parquet.py` |
| Factors, targets and splits | `build_features.py`, `build_targets.py`, `build_model_dataset.py` |
| OLS and Elastic Net | `train_ols.py`, `tune_ols.py`, `tune_lasso.py`, `tune_elastic_net.py`, `refine_elastic_net.py` |
| Other linear experiments | `tune_factor_subsets.py`, `tune_daily_weights.py`, `tune_cross_sectional.py`, `tune_relative_target.py`, `tune_target_winsor.py`, `tune_weight_exponent.py` |
| LightGBM and MLP | `train_lightgbm.py`, `train_mlp.py`, corresponding `tune_lightgbm_*.py` and `tune_mlp_training.py` |
| Shared training and runner | `training_common.py`, `train_models.py` |
| Validation and execution | `generate_validation_signals.py`, `backtest_validation.py`, `stress_test_validation.py`, `test_market_impact.py` |
| Final historical artifacts | `retrain_final_models.py`, `generate_permanent_test_signals.py`, `evaluate_permanent_test.py`, `analyze_test_results.py` |

## Reports and instructions

- [Selected V1 reports](../reports/v1)
- [Original pipeline documentation](docs/LEGACY_PIPELINE.md)
- [Environment and reproduction guide](../v2/docs/REPRODUCING.md)
- [Data requirements](../data/README.md)
- [V1/V2 model comparison](../v2/docs/V1_V2_MODEL_COMPARISON.md)

## Execution and compatibility

Run from the repository root, for example:

```sh
python v1/train_ols.py --help
python v1/train_models.py --help
```

Data defaults continue to point to `<repository>/quant_data/`. V2 imports the shared V1 modules through its path bootstrap. There are no duplicate root-level Python scripts. Source files are retained historical implementations with path adaptations for the new layout; original frozen trained artifacts are not modified.

V1 final models and the already-inspected 2025–2026 test are historical, read-only research artifacts. Do not tune V2 against that test or describe it as an untouched independent test.
