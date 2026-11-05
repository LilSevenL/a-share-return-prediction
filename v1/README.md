# V1: historical research code

This directory collects the V1 source files retained in this repository: data acquisition and preparation, the original 50-factor library, five-day targets, rolling model training and tuning, backtesting, final retraining and the already-inspected historical test.

These files are byte-for-byte archival copies of the corresponding root-level Python scripts at the time this directory was added. They are not reconstructed snapshots of every earlier experiment, and do not include fitted models or vendor data.

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

- [Selected V1 reports](../reports/v1/)
- [Original pipeline documentation](../docs/LEGACY_PIPELINE.md)
- [Environment and reproduction guide](../docs/REPRODUCING.md)
- [Data requirements](../data/README.md)
- [V1/V2 model comparison](../v2/V1_V2_MODEL_COMPARISON.md)

## Execution and compatibility

This directory provides a convenient V1 browsing archive. **Use the root-level scripts for execution**, for example `python train_ols.py --help` from the repository root. The historical scripts resolve data paths relative to their own location; directly executing a copy under `v1/` would point several defaults at `v1/quant_data/`. No vendor data or intermediate artifacts are bundled there.

The root-level scripts remain the canonical executable copies because V2 imports shared modules from them. V2 code and experiment notes live under `../v2/`. This organizational change does not alter model fitting, data, trading rules or saved results.

V1 final models and the already-inspected 2025–2026 test are historical, read-only research artifacts. Do not tune V2 against that test or describe it as an untouched independent test.
