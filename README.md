# A-share Return Prediction Research

Research code for predicting five-trading-day A-share returns, from an OLS baseline to Elastic Net and robust linear losses, with factor audits and transaction-cost-aware portfolio simulation.

**Status: historical development research, updated November 20, 2026.** LightGBM V2 optimization and frozen confirmation are complete. The 2017–2023 period has been inspected repeatedly, and the local universe has survivorship bias due to incomplete delisted-stock coverage.

## Start here

- [V1 code and instructions](v1/README.md)
- [Research overview and latest results (Chinese)](v2/docs/V2_PROGRESS_SUMMARY.md)
- [V1/V2 OLS and Elastic Net comparison (Chinese)](v2/docs/V1_V2_MODEL_COMPARISON.md)
- [Full research document index: 38 stages](v2/docs/RESEARCH_DOCUMENT_INDEX.md)
- [Reproduction guide and dependencies](v2/docs/REPRODUCING.md)
- [Data requirements](data/README.md)
- [Completed LightGBM V2 study and reproduction notes](v2/docs/LGBM_V2_RELEASE.md)
- [LightGBM final summary (Chinese)](v2/docs/LGBM_V2_FINAL_SUMMARY.md)

## LightGBM V2 findings

The frozen model uses 448 raw factors, daily rank-return targets, up to 127 leaves, depth 8, at least 2,000 observations per leaf on the 2-million-row refit, L2 regularization of 1,000 and three-seed averaging. Fold-specific L1 is approximately 39–42.

| Label experiment, same 2017–2021 development folds | Mean daily Rank IC | Selection score |
|---|---:|---:|
| L2, raw return target | 0.0873 | 0.0827 |
| Huber, raw return target | 0.0919 | 0.0881 |
| L2, daily demeaned return | 0.1239 | 0.1164 |
| L2, daily rank-return target | 0.1375 | 0.1296 |

The selection score subtracts 0.25 times the standard deviation of annual Rank ICs. The rank-label procedure improved mean Rank IC by about 57.5%, with accompanying regularization scaling and reselected stopping rounds.

| Frozen confirmation year | LightGBM Rank IC | Matched rank-label Elastic Net Rank IC |
|---|---:|---:|
| 2022 | 0.1069 | 0.0900 |
| 2023 | 0.0828 | 0.0707 |

The pooled daily difference is +0.0145 with a Newey–West t statistic of 2.63 at lag 5. The model retains full-cross-section ranking gains, while its top-50 selection advantage weakens in recent years.

The original continuous-account simulation returned −22.1% for LightGBM and −12.6% for the matched Elastic Net over 2022–2023 under base impact. These segments inherit different capital and positions from 2017–2021. Deployment and inclusion in the existing long-only strategy have not been supported by the confirmation evidence. The existing linear candidate and its fusion used these years during earlier development.

See the [confirmation report](v2/docs/LGBM_CONFIRMATION_RESULTS.md), [post-hoc diagnosis](v2/docs/LGBM_POSTHOC_DIAGNOSIS.md) and [aggregate tables](reports/v2/lightgbm/).

## Selected findings

The factor design expanded from 50 to 500 candidates. After availability checks, 449 were computable and 448 entered each baseline fold. Removing the momentum/reversal and conditional-filtering groups produced a 333-factor research candidate.

| Comparison, 2017–2023 | Average daily Rank IC |
|---|---:|
| V2 OLS, all eligible training rows | 0.05465 |
| OLS, matched 200,000 rows per fold | 0.05098 |
| Huber, same 200,000 rows per fold | 0.06673 |
| MAE, same 200,000 rows per fold | 0.06966 |
| Reduced Huber, 500,000 rows per fold | 0.07305 |
| Reduced MAE, 500,000 rows per fold | 0.07704 |

Huber and MAE change the loss function and are separate linear candidates, not OLS fits. Comparisons across training sizes also change the fitted preprocessing/scale estimates.

For the 333-factor Elastic Net, a training-only availability filter required at least 30 valid factors from the legacy 50-factor library. Paired comparisons held the evaluation universe, regularization rules and preprocessing fixed within each seed.

| Three-seed range | Without training filter | With training filter |
|---|---:|---:|
| Average daily Rank IC | 0.06903–0.07012 | 0.07280–0.07344 |
| Seven-year cumulative net return, doubled impact | −14.43% to −8.57% | 60.56%–72.68% |
| Maximum drawdown, doubled impact | −41.85% to −39.99% | −37.03% to −35.91% |

The simulation uses RMB 100 million initial capital, 20 bp round-trip base costs plus impact, strict daily-bar execution constraints, top-50 entry and top-100 retention under the existing five-day review cycle. These are **seven-year cumulative returns, not annual returns**. See the [paired experiment](reports/v2/elastic_compact_gate_seeds/REPORT.md) and [separate board-budget diagnostic](reports/v2/elastic_board_accounts/REPORT.md). The three seeds reuse the same historical period and are not independent market samples.

## Repository layout

```text
README.md             Research question, results, limits and entry points
requirements.txt      Core research dependencies
v1/                   Historical V1 code and pipeline documentation
v2/
  features/           Factor definitions, construction and data audits
  models/             Linear models, robust losses and preprocessing
  backtest/           Portfolio accounting and execution engines
  experiments/        Training runs, comparisons, attribution and verification
  tests/              Synthetic numerical and accounting checks
  docs/               Research plans, progress and reproduction guide
configs/              Factor catalogues, experiment plan and optional dependencies
reports/v1/           Selected historical V1 reports
reports/v2/           Historical V2 experiment reports
data/README.md        Data sourcing and expected local layout
```

All Python source files live under `v1/` or `v2/`. V2 uses the shared V1 implementations without duplicate root-level scripts. Data remain local under `<repository>/quant_data/`, which is ignored by Git. Raw/vendor data, fitted models, predictions, orders, caches and credentials are not distributed. Reproducing historical results requires separately prepared data and prerequisite artifacts; some report links refer to those unbundled artifacts.

## Quick check without market data

Use Python 3.12 and install `requirements.txt` in a clean virtual environment. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s v2/tests -p test_streaming_ols.py
python -m unittest discover -s v2/tests -p test_linear_losses.py
```

These checks use synthetic numerical fixtures. They do not download data, rerun training, or establish trading profitability. See [REPRODUCING.md](v2/docs/REPRODUCING.md) for the experiment sequence.

## Research boundaries

- Incomplete delisted-stock history introduces survivorship bias. Financial inputs remain gated where point-in-time provenance is inadequate.
- Daily bars do not reproduce intraday queues or realized fills. Impact costs have not been calibrated against actual executions; corporate actions and historical ST coverage remain limitations.
- Training branches using future tradability (`target_valid`) are historical diagnostics. The feature-availability candidate uses a training-only feature gate instead.
- V1 final models and its already-inspected 2025–2026 test are historical artifacts. They must not be reused for V2 model selection or represented as an untouched test.
- LightGBM V2 was frozen before its 2022–2023 algorithm-specific confirmation. These years were inspected in earlier research. Forward evaluation requires later-acquired data after configuration freeze.

Original experiment reports are mostly in Chinese. No open-source license has been selected for this initial publication.
