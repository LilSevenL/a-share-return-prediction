# LightGBM V2 research release

Published November 20, 2026, retaining the existing V1/V2 directory structure. Historical notes retain their recorded execution decisions and dates.

## Results and reading order

1. Read [the final summary](LGBM_V2_FINAL_SUMMARY.md).
2. Read [labels and losses](LGBM_STAGE6A_RESULTS.md), [input experiments](LGBM_INPUT_RESULTS.md) and [final selection](LGBM_FINAL_SELECTION.md).
3. Read [confirmation results](LGBM_CONFIRMATION_RESULTS.md) and [post-hoc diagnosis](LGBM_POSTHOC_DIAGNOSIS.md).
4. The [plan and revisions](LIGHTGBM_V2_PLAN.md), [context](LGBM_V2_CONTEXT.md) and [worklog](LGBM_V2_WORKLOG.md) record sequential development decisions.
5. [Published aggregate tables](../../reports/v2/lightgbm/) include input, seed, full-sample refit, confirmation and post-hoc summaries. Original settings are archived in [frozen_config.json](../../configs/v2/lightgbm/frozen_config.json).

Stage 1 explores structure. Stage 2 explores L1/L2 and smoothing. Stage 6a compares labels and losses. Stage 3 checks a matched rank-label linear model, permutation diagnostics and sampling. Stage 4 checks learning rate and stopping rules. Input experiments compare factor sets, availability gates and feature ranks. Final stages repeat seeds, refit complete samples and freeze configuration before confirmation.

## Source locations

- Shared data, metrics, stopping and caches live in v2/models/lgbm_v2_common.py and lgbm_v2_cache.py.
- Stage runners and diagnostics live in v2/experiments/.
- Synthetic checks live in v2/tests/test_lgbm_v2_common.py.
- Extension builders and overlap checks live in v2/features/. The 2024–2026 feature extension remains paused and has produced no extension dataset.

## Environment and checks

The research used Python 3.12 and LightGBM 4.7.0. Core dependencies remain in the root requirements.txt and optional models in configs/requirements-optional.txt.

Run from the repository root.

~~~sh
python -m unittest discover -s v2/tests -p test_lgbm_v2_common.py
python v2/experiments/run_lgbm_v2_pilot.py --help
python v2/experiments/run_lgbm_v2_stage1.py --help
~~~

Full fits require local market data, targets, metadata, fold-specific factor lists and caches. Consult [data requirements](../../data/README.md) and each runner's argument parser. No training or data downloading was performed for this publication.

## Provenance

Source paths were adapted to the existing grouped layout. [publication_sources.json](../../configs/v2/lightgbm/publication_sources.json) maps original and published hashes.

The archived frozen_config.json preserves the original research hashes. Its strict checks require matching original code and artifacts. A reproduction using the published layout must record a separate configuration and provenance snapshot. Hash mismatch in an old artifact does not authorize disabling verification.

Vendor datasets, fitted models, per-stock predictions/orders and large caches remain local. CSVs here contain aggregate research results. Notes naming local disks and quant_data paths describe the original environment.

## Publication completeness check

The GitHub snapshot was cloned into a separate checkout and checked without local market datasets. All 78 existing unit tests and all 14 argparse-based LightGBM/extension command-line help entry points passed using the local research Python environment. Repository Markdown links had no missing targets, and import inspection found no unresolved modules in that environment.

The audit found and corrected two extension-builder provenance paths left over from the flat directory layout. The original build and repair diffs were added. All mapped source/config paths, static source references and 23 published source hashes were then verified, and the 78 tests passed again.

Full model training and a clean dependency installation were outside this publication check. Market datasets, intermediate caches and fitted models are required to rerun the historical fitting workflow.

## Model interpretation

The 0.0827 to 0.1296 label comparison uses the penalized selection score. Mean Rank IC is 0.0873 to 0.1375.

The medium-depth final model accepts approximately 0.001 selection-score loss versus the deep model. Failure to reject a difference at t=2 does not establish equivalence. Sequential searches and revised rules make development t statistics descriptive.

A single placebo initially failed its preset threshold. Expanded multi-seed permutation checks were then specified and run on development folds. They support signal above a random-label baseline and cover only part of the potential leakage mechanisms.

The 2022–2023 results are frozen algorithm-specific confirmation on already-inspected historical years. V1 also evaluated later history. Future reuse must disclose that history. Current-universe survivorship bias and numerical assumptions remain documented limitations.

The original narrow extension changes are archived in [build diff](feature_extension_build.diff) and [repair diff](feature_extension_repair.diff). These describe the flat-layout research sources. Published import/path adaptations are tracked separately in publication_sources.json.
