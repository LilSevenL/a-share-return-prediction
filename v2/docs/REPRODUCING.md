# Reproducing the research

## Environment and data-free checks

Use Python 3.12. `requirements.txt` records the core package versions from the research environment; it is not a platform-independent lockfile. Optional historical downloaders, plots, LightGBM and PyTorch models use `requirements-optional.txt`. A clean dependency installation has not been independently verified.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s v2/tests -p test_streaming_ols.py
python -m unittest discover -s v2/tests -p test_linear_losses.py
python -m unittest discover -s v2/tests -p test_optimized_losses.py
python -m unittest discover -s v2/tests -p test_execution_hardening.py
```

Run commands from the repository root. V2 executable scripts are invoked as `python v2/experiments/<script>.py` (or the appropriate features/models/backtest directory); the V2 bootstrap locates sibling modules and shared V1 code automatically.

## Historical workflow

1. Prepare separately licensed local market data, labels and legacy factors; see [data requirements](../../data/README.md) and [legacy pipeline](../../v1/docs/LEGACY_PIPELINE.md). Downloaders retain their historical proxy endpoint configuration; inspect `v1/main.py` and `v2/features/repair_historical_universe.py` before making network requests. Credentials are read from the local environment, never supplied in this repository.
2. Follow the factor design, data audit and numerical repair stages in the [research index](RESEARCH_DOCUMENT_INDEX.md). `build_local_features.py` and `repair_local_numerics.py` produce the local feature inputs used by the stable baseline. Builders have prerequisites; consult their argument parsers and the stage plans before running them.
3. Fit the stable OLS baseline with `python v2/experiments/run_local_ols.py --folds 2017 2018 2019 2020 2021 2022 2023`. This requires the local feature tables, targets, calendar and source provenance manifests.
4. Follow stages 12–25 for matched loss/weight experiments, factor ablation, expanded 500,000-row samples, continuous accounts and execution stress. Saved model/preprocessing and sample artifacts feed later stages.
5. Follow stages 26–34 for Elastic Net, training-sample audits, paired gates and repeated seeds. `run_elastic_compact_gate_seeds.py` requires the previously saved 333-factor preprocessing and paired 500,000-row sample-key files; it cannot run directly from a fresh clone without those prerequisites.
6. Stages 35–38 attribute costs, stock/board contributions and run the fixed initial board-budget diagnostic. These require saved predictions and account outputs.

The repository publishes source code and historical reports, not a one-command reconstruction of all vendor data and intermediate artifacts. File hashes in original provenance records refer to the original local artifacts. No training or backtesting was rerun to publish this snapshot.

## Development/test separation

Use 2017–2023 only as the already-inspected development period. V1 final retraining and its 2025–2026 test are historical, read-only results, not a new holdout for V2. Freeze candidate selection rules before acquiring new independent evaluation data. Do not interpret repeated seeds as independent market validation.

## Directory migration

Only code organization, import paths and source-file provenance paths changed. Data paths still resolve to the repository-level `quant_data/`; the migration did not rerun or modify historical training results. Existing original manifests may contain old source paths/hashes and are not expected to match reorganized source bytes without a fresh, separately recorded run.

## LightGBM update, November 20, 2026

The completed optimization, frozen configuration, confirmation results and model limits are described in [LightGBM V2 release](LGBM_V2_RELEASE.md). Earlier tables on this page retain their original linear-experiment scope.
