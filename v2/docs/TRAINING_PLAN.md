# V2 training plan

Status: user requested a broader factor design, replacing the concentration in conditional filters. The CURRENT design is `research_design/FACTOR_CATALOG_500.md`: 500 defined candidates across ten research directions. It is a design specification, not yet a numerical implementation. The prior 50+450-filter builder and its 8-stock check are superseded design artifacts; do not use them as the new 500-factor production pipeline. Feature design and implementation come before further model training.

## Scope

- Keep V1 scripts, trained models, preprocessing, reports and permanent-Test artifacts intact.
- Reuse existing raw data and unaffected processing outputs.
- Initialize V2 model parameters from scratch by default; any warm-start experiment must be explicitly labeled.
- Store new training artifacts under `quant_data/processed/v2/`, with separate directories for each experiment.
- Do not run legacy training entry points with their default output paths for V2.

## Evaluation

- Compare the V1 configuration and V2 candidates using matching time folds, training data boundaries, preprocessing fit boundaries and evaluation metrics.
- Use historical rolling validation for development. The existing 2024 confirmation period has already been inspected and is not a new untouched holdout.
- Never evaluate a model trained through 2024 as out-of-sample on 2024.
- The already inspected 2025-2026 permanent Test cannot be used as a fresh independent test or to select V2 settings.
- Freeze the selected V2 configuration before collecting a genuine forward-test result.

## Mentor directions and staged experiments

1. Expand the original 50 features toward roughly 500 economically motivated features. Prioritize conditional historical returns: within the past 20 sessions, rank days by volume or large-order flows and aggregate returns over the top half. Compare high/low activity and their differences, with explicit missing-data and tie handling. Feature count is a research scope, not a guarantee of improved returns.
2. Separate overnight and open-to-close returns using daily prices. True intraday path dependence (high before low versus low before high) requires intraday observations; daily OHLC cannot establish that order. Do not invent minute data or infer the intraday sequence from OHLC.
3. Compare squared, absolute and Huber residual losses. MAE and Huber models are linear regressions but are not OLS. Huber reduces the influence of large residuals relative to MSE. Robust losses and volatility weighting must be separately tested because they change emphasis in different ways.
4. Add stock-level weights using information available at the signal time, beginning with past 20-session volatility. Do not weight on realized future returns.
5. Report ordinary Rank IC and the SAME weighted Rank IC definition for all candidates. Cross-model comparison on a common metric is valid, but neither metric alone establishes trading profitability. Ultimately compare identical portfolio rules, costs and execution constraints.

## Stage 1: implemented and launched

- Entry point: `v2/models/train_linear_baselines.py`.
- Output: `quant_data/processed/v2/linear_baselines_50/`.
- Full-data expanding validation folds 2017-2023; only the physical `final_split=training` partition is opened. No model training subsample is used in the main run.
- Keep the existing 50 factors and target fixed. Training-only preprocessing uses the existing seeded sample of up to 500,000 rows.
- Candidates: ordinary OLS, WLS with existing date exponent 0.75, WLS with date weights multiplied by relative historical-volatility weights.
- Stock multiplier: historical volatility_20d divided by same-day median positive volatility, clipped to [0.5, 2.0] BEFORE normalization, then normalized to daily mean 1. Missing volatility receives a neutral raw multiplier. This preserves each day's aggregate weight. Bounds are a first experimental choice, not a previously validated optimum.
- Weighted Rank IC is daily weighted Pearson correlation of ordinary average ranks, using the same volatility rule for every candidate. Days are averaged equally. Other weighted-rank conventions are possible; this one is explicit and fixed.
- Coefficients and intercept are jointly fitted through centered weighted normal equations, with pseudoinverse cutoff 1e-10. This corrects the original pure-OLS script's approximate intercept handling, so the new pure-OLS result is a new matched control, not claimed as exact V1 reproduction.
- The final V1 19-factor Elastic Net remains a separate reference; its previously selected subset is not an independent fresh selection on the same development folds.
- Three numerical tests passed: weighted coefficients/intercept versus augmented least squares, daily weight preservation and past-date invariance, and equal-weight Rank IC equivalence with ties. A 20,000-training/10,000-validation-row pipeline smoke run completed; its metrics are debug-only and must not be used as research results.
- Configuration, source hashes, input file inventory, preprocessing, predictions and per-fold metrics are saved. Resume requires the same recorded configuration and sources.

## Next stages after the user's sequencing correction

1. Implement the CURRENT balanced 500-feature design under `research_design/`, then generate and audit the feature table. The earlier 50+450-filter implementation is superseded and must not be substituted for the balanced design.
2. Compare regularized linear models on original versus expanded feature sets. Highly correlated expansions make pure OLS a control rather than an assumed preferred model.
3. MSE/MAE/Huber objectives and a feature-count-independent linear fitting API are implemented in `linear_losses.py`. The balanced pipeline and common simulator have now completed an 80-stock integration run, including an unweighted OLS control plus nine loss/weight configurations. See `IMPLEMENTATION_GUIDE.md` for authoritative current status, explicit financial gates, actual outputs and full-market memory limits. This is not completed full-market research.
4. Run common execution-aware portfolio simulations before selecting a candidate for deployment or making P&L claims.
5. Freeze V2 and establish a genuine forward evaluation period.

## Initial inspection

- Existing final model files and `final_training_manifest.json` are present in `quant_data/processed/final_models_2011_2024/`.
- Existing training utilities support explicit dataset, output and fold arguments.
- The documented interpreter `E:/Anaconda/envs/quant_ml/python.exe` successfully imported NumPy, SciPy, DuckDB and PyArrow. Available memory at inspection was about 18.8 GB and free disk about 55.7 GB; training uses four compute threads and an 8 GB DuckDB memory limit.
