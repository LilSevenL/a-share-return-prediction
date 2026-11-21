"""V2 LightGBM pilot: small-sample end-to-end validation and full-size timing run.

Modes (see v2/docs/LIGHTGBM_V2_PLAN.md section 9):
  smoke   20,000-row sample, one outer month, 200 rounds; checks every step of the pipeline.
  timing  2,000,000-row sample (or --rows), full outer year, medium tree, leaf threshold 0.25%.

Both use outer year 2021 by default, the 448-factor set, no training gate, and an
Elastic Net control fitted on the same inner fit set. No 2024+ data are read.

Examples (project root, Windows):
  & E:\\Anaconda\\envs\\quant_ml\\python.exe v2\\run_lgbm_v2_pilot.py --mode smoke
  & E:\\Anaconda\\envs\\quant_ml\\python.exe v2\\run_lgbm_v2_pilot.py --mode timing
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lgbm_v2_common as common
from lgbm_v2_common import DATA, KEY, Timer

MODES = {
    'smoke': dict(rows=20_000, early_rows=5_000, cap=200, every=10, patience=50,
                  outer_months=1, label_rows=200, leaf_trees=3),
    'timing': dict(rows=2_000_000, early_rows=300_000, cap=3000, every=10, patience=200,
                   outer_months=None, label_rows=1000, leaf_trees=10),
}
MEDIUM = dict(num_leaves=127, max_depth=8, leaf_fraction=0.0025)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--year', type=int, default=2021)
    parser.add_argument('--rows', type=int, help='override the training sample size')
    parser.add_argument('--feature-set', choices=['full', 'compact'], default='full')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-limit', default='6GB', help='DuckDB memory limit')
    args = parser.parse_args()
    if not 2017 <= args.year <= 2021:
        raise ValueError('Pilot is restricted to the 2017-2021 search period')
    return args


def lgbm_params(n_fit, threads):
    return dict(objective='regression', metric='None', boosting='gbdt',
                learning_rate=0.02, num_leaves=MEDIUM['num_leaves'], max_depth=MEDIUM['max_depth'],
                min_data_in_leaf=int(math.ceil(MEDIUM['leaf_fraction'] * n_fit)),
                feature_fraction=0.3, bagging_fraction=1.0, bagging_freq=0, max_bin=255,
                lambda_l1=0.0, lambda_l2=0.0, extra_trees=False, min_gain_to_split=0.0,
                feature_pre_filter=False, force_col_wise=True, deterministic=True,
                seed=42, num_threads=threads, verbosity=-1)


def fit_elastic(x_fit, y_fit, names):
    """V2 Elastic Net control (L1 ratio 0.5, alpha 0.1 x critical), fitted on the inner fit set only."""
    from research_controls import fit_preprocessing, transform_features
    from tune_elastic_net import fit_elastic_net_from_gram
    # fit_preprocessing casts its whole input to float64 before drawing 50k rows; pass exactly
    # the rows it would draw (same seed and call; only row order differs, which the stats ignore)
    # so the preprocessing is unchanged without a ~6 GB float64 copy.
    picks = np.random.default_rng(42).choice(len(x_fit), min(len(x_fit), 50_000), replace=False)
    stats = fit_preprocessing(x_fit[np.sort(picks)], names, seed=42, max_rows=50_000)
    width = len(stats['active_names'])
    xx, xy, sx = np.zeros((width, width)), np.zeros(width), np.zeros(width)
    sy, n = 0.0, 0
    for start in range(0, len(y_fit), 100_000):
        z = transform_features(x_fit[start:start + 100_000], stats).astype(np.float64)
        target = y_fit[start:start + 100_000]
        xx += z.T @ z
        xy += target @ z
        sx += z.sum(axis=0)
        sy += float(target.sum())
        n += len(z)
    mx, my = sx / n, sy / n
    gram = xx / n - np.outer(mx, mx)
    rhs = xy / n - mx * my
    alpha = 0.1 * float(np.max(np.abs(rhs))) / 0.5
    fit = fit_elastic_net_from_gram(dict(gram=gram, rhs=rhs, mean_x=mx, mean_y=my), alpha, 0.5,
                                    np.zeros(width), 30_000, 1e-10)
    if not fit['converged']:
        raise RuntimeError('Elastic Net control did not converge')
    return dict(stats=stats, coefficient=np.asarray(fit['coefficient']), intercept=float(fit['intercept']),
                alpha=alpha, nonzero=int(fit['nonzero_factors']), iterations=int(fit['iterations']),
                active_count=width)


def predict_elastic(model, x):
    from research_controls import transform_features
    return transform_features(x, model['stats']).astype(np.float64) @ model['coefficient'] + model['intercept']


def main():
    import lightgbm as lgb
    args = parse_args()
    spec = dict(MODES[args.mode])
    if args.rows:
        spec['rows'] = args.rows
    output = DATA / 'v2' / f'lightgbm_pilot_{args.mode}_{args.year}_{args.feature_set}_{spec["rows"]}'
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(common.__file__), _source_path('v2/models/research_controls.py'),
               _source_path('tune_elastic_net.py')]
    config = dict(mode=args.mode, year=args.year, feature_set=args.feature_set, spec=spec,
                  structure=MEDIUM, threads=args.threads, sample_seed=common.SAMPLE_SEED,
                  sample_order='hash(ts_code, CAST(signal_date AS DATE), 20261025), ts_code, signal_date',
                  training_gate=False, test_used=False,
                  sources={p.name: common.sha256(p) for p in sources})
    config_path = output / 'config.json'
    if config_path.exists() and common.read_json(config_path) != json.loads(json.dumps(config, default=str)):
        raise RuntimeError('Existing output belongs to a different configuration; remove it or change the run')
    common.save_json(config_path, config)

    timer = Timer()
    started = time.perf_counter()
    names = common.feature_names(args.year, args.feature_set)
    con = common.open_connection(output, args.threads, args.memory_limit)
    with timer('views_and_pool'):
        common.create_views(con, args.year)
        pool_rows = common.create_pool(con, args.year)
    with timer('sampling'):
        common.create_samples(con, args.year, spec['rows'], spec['early_rows'])
        checks = dict(pool_rows=pool_rows, boundaries=common.boundary_checks(con, args.year))
        keys_path = str(output / 'sample_keys.parquet').replace('\\', '/')
        con.execute(f"COPY (SELECT ts_code, signal_date FROM full_sample ORDER BY priority, ts_code, signal_date) "
                    f"TO {common.quote(keys_path)} (FORMAT PARQUET)")
    with timer('label_check'):
        checks['labels'] = common.label_check(con, 'full_sample', spec['label_rows'])
    with timer('subset_check'):
        sample_keys = con.execute('SELECT ts_code, signal_date FROM full_sample').fetch_df()
        checks['subset'] = common.subset_check(sample_keys, args.year)
        del sample_keys
    common.save_json(output / 'checks.json', checks)
    print(f'[checks] pool={pool_rows:,} fit={checks["boundaries"]["inner_fit"]["rows"]:,} '
          f'early={checks["boundaries"]["early_stop"]["rows"]:,} labels/boundaries passed', flush=True)

    with timer('fetch_inner_fit'):
        x_fit, y_fit, fit_keys = common.fetch_matrix(con, 'inner_fit', names)
    with timer('fetch_early_stop'):
        x_es, y_es, es_keys = common.fetch_matrix(con, 'early_stop', names)
    fit_dates = fit_keys['signal_date'].to_numpy()
    print(f'[data] inner fit {x_fit.shape}, early stop {x_es.shape}, rss peak {common.peak_rss_gb():.2f} GB', flush=True)

    params = lgbm_params(len(y_fit), args.threads)
    monitor = common.RankICEarlyStopping(y_es, es_keys['signal_date'].to_numpy(),
                                         every=spec['every'], patience=spec['patience'])
    with timer('dataset_construct'):
        # Dataset gets the full training params so lgb.train does not try to change
        # dataset-level settings (feature_pre_filter, max_bin) on an already-built handle.
        train_set = lgb.Dataset(x_fit, label=y_fit, feature_name=names, free_raw_data=False, params=params)
        valid_set = lgb.Dataset(x_es, label=y_es, reference=train_set, params=params)
        train_set.construct()
        valid_set.construct()
    with timer('train'):
        monitor.started = time.perf_counter()  # per-100-round laps exclude Dataset construction
        booster = lgb.train(params, train_set, num_boost_round=spec['cap'], valid_sets=[valid_set],
                            valid_names=['early_stop'], feval=monitor.feval, callbacks=monitor.callbacks())
    best = monitor.best_iteration
    if best <= 0:
        raise RuntimeError('No valid early-stopping score was recorded')
    # lgb.train truncates the returned booster to best_iteration after an early stop,
    # so current_iteration() would report the best round; use the last round actually run.
    trained_rounds = monitor.iteration
    peak_after_train = common.peak_rss_gb()
    print(f'[train] rounds={trained_rounds} best={best} early-stop RankIC={monitor.best_score:.5f} '
          f'time={timer.records["train"]:.0f}s rss peak {common.peak_rss_gb():.2f} GB', flush=True)

    with timer('diagnostics'):
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(y_fit), min(300_000, len(y_fit)), replace=False))
        train_ic = common.DailyRankIC(y_fit[idx], fit_dates[idx])(booster.predict(x_fit[idx], num_iteration=best))
        diagnostics = dict(trees=common.tree_diagnostics(booster, best),
                           leaf_coverage=common.leaf_date_coverage(booster, x_fit, fit_dates, best,
                                                                   trees=spec['leaf_trees']),
                           inner_fit_rank_ic=train_ic, early_stop_rank_ic=monitor.best_score,
                           fit_minus_early_stop_rank_ic=train_ic - monitor.best_score)
    with timer('elastic_control'):
        elastic = fit_elastic(x_fit, y_fit, names)
    del train_set, valid_set, x_es
    gc.collect()

    frames = []
    with timer('outer_predict'):
        for month in common.outer_months(con, args.year, spec['outer_months']):
            x_out, frame = common.fetch_outer_month(con, args.year, month, names)
            frame['lgbm'] = booster.predict(x_out, num_iteration=best)
            frame['elastic'] = predict_elastic(elastic, x_out)
            frames.append(frame)
            del x_out
    outer = pd.concat(frames, ignore_index=True)
    con.close()
    if not np.isfinite(outer[['lgbm', 'elastic']].to_numpy()).all():
        raise RuntimeError('Non-finite outer predictions')

    with timer('outer_metrics'):
        metrics, daily = {}, {}
        for column in ('lgbm', 'elastic'):
            metrics[column], daily[column] = common.evaluate_outer(outer, column)
            metrics[column]['timing_split'] = common.timing_selection_split(outer, column)
        paired = daily['lgbm'][['date', 'rank_ic']].merge(daily['elastic'][['date', 'rank_ic']],
                                                          on='date', suffixes=('_lgbm', '_elastic'))
        difference = paired['rank_ic_lgbm'] - paired['rank_ic_elastic']
        comparison = dict(days=len(paired), mean_rank_ic_difference=float(difference.mean()),
                          newey_west_t_lag5=common.newey_west_tstat(difference.to_numpy(), lag=5))

    booster_path = output / 'lightgbm_model.txt'
    booster_path.write_text(booster.model_to_string(num_iteration=best), encoding='utf-8')
    outer.to_parquet(output / 'outer_predictions.parquet', index=False)
    pd.DataFrame(monitor.curve, columns=['iteration', 'early_stop_rank_ic', 'seconds']).to_parquet(
        output / 'early_stop_curve.parquet', index=False)
    pd.concat([daily['lgbm'].assign(model='lgbm'), daily['elastic'].assign(model='elastic')]).to_parquet(
        output / 'daily_metrics.parquet', index=False)

    laps = monitor.lap_times
    per_100 = [round(b[1] - a[1], 2) for a, b in zip([(0, 0.0)] + laps[:-1], laps)]
    summary = dict(
        status='completed', mode=args.mode, year=args.year, feature_set=args.feature_set,
        rows=dict(training_pool=pool_rows, full_sample=checks['boundaries']['full_sample']['rows'],
                  inner_fit=len(y_fit), early_stop=len(y_es), outer=len(outer),
                  outer_scored=int(np.isfinite(outer['y']).sum()),
                  full_to_inner_ratio=checks['boundaries']['full_sample']['rows'] / len(y_fit)),
        lightgbm=dict(params=params, rounds_trained=trained_rounds, best_iteration=best,
                      hit_round_limit=monitor.hit_round_limit(spec['cap']),
                      early_stop_best_rank_ic=monitor.best_score),
        elastic_control={k: v for k, v in elastic.items() if k not in ('stats', 'coefficient')},
        outer_metrics=metrics, lgbm_vs_elastic=comparison, diagnostics=diagnostics,
        timing_seconds=dict(**timer.records, total=round(time.perf_counter() - started, 1),
                            seconds_per_100_rounds=per_100,
                            mean_seconds_per_100_rounds=float(np.mean(per_100)) if per_100 else None),
        peak_memory_gb=common.peak_rss_gb(), peak_memory_gb_after_lgbm_train=peak_after_train,
        machine=dict(platform=platform.platform(), processor=platform.processor(), threads=args.threads,
                     lightgbm=lgb.__version__),
        test_used=False)
    common.save_json(output / 'summary.json', summary)

    lg, en = metrics['lgbm'], metrics['elastic']
    print('\n=== Pilot summary ===')
    print(f'output: {output}')
    print(f'LightGBM  RankIC={lg["mean_rank_ic"]:.5f}  top50-pool={lg["top50_minus_pool_5d_simple_return"]}  '
          f'leaves={diagnostics["trees"]["mean_leaves"]:.1f} depth={diagnostics["trees"]["mean_depth"]:.1f}')
    print(f'Elastic   RankIC={en["mean_rank_ic"]:.5f}  top50-pool={en["top50_minus_pool_5d_simple_return"]}')
    print(f'diff={comparison["mean_rank_ic_difference"]:.5f}  NW t={comparison["newey_west_t_lag5"]:.2f}')
    print(f'train {timer.records["train"]:.0f}s for {trained_rounds} rounds; '
          f'construct {timer.records["dataset_construct"]:.0f}s; total {summary["timing_seconds"]["total"]:.0f}s; '
          f'peak memory {summary["peak_memory_gb"]:.2f} GB')


if __name__ == '__main__':
    main()
