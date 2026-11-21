"""V2 LightGBM stage 1: tree structure screen (plan section 7, stage 1).

3 structures x 3 leaf thresholds x outer years 2017/2019/2021 = 27 fits, 448 factors, no training
gate, screening flow (inner fit -> early stop on the inner early-stopping set -> predict outer year).
All other settings are fixed at the plan values. Data come from the per-fold caches built by
lgbm_v2_cache.py; the Elastic Net control is the one cached per fold.

  python v2/experiments/run_lgbm_v2_stage1.py --verify      cache equivalence check against the timing pilot
  python v2/experiments/run_lgbm_v2_stage1.py               run all 27 fits (resumable; one subprocess per fit)
  python v2/experiments/run_lgbm_v2_stage1.py --summarize   summary table and v2/docs/LGBM_STAGE1_RESULTS.md inputs
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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from lgbm_v2_common import DATA, Timer

OUT = DATA / 'v2' / 'lgbm_stage1'
VERIFY_OUT = DATA / 'v2' / 'lgbm_cache_equivalence_2021_full'
PILOT = DATA / 'v2' / 'lightgbm_pilot_timing_2021_full_2000000'
YEARS = (2017, 2019, 2021)
STRUCTURES = {'shallow': dict(num_leaves=31, max_depth=6),
              'medium': dict(num_leaves=127, max_depth=8),
              'deep': dict(num_leaves=511, max_depth=-1)}
LEAF_FRACTIONS = {'0.1pct': 0.001, '0.25pct': 0.0025, '1pct': 0.01}
FIXED = dict(learning_rate=0.02, cap=3000, every=10, patience=200, feature_set='full', leaf_trees=10)


def lgbm_params(structure, leaf_fraction, n_fit, threads=8):
    """Same parameter set as the pilot's lgbm_params, with structure and leaf threshold varied."""
    return dict(objective='regression', metric='None', boosting='gbdt',
                learning_rate=FIXED['learning_rate'], num_leaves=structure['num_leaves'],
                max_depth=structure['max_depth'],
                min_data_in_leaf=int(math.ceil(leaf_fraction * n_fit)),
                feature_fraction=0.3, bagging_fraction=1.0, bagging_freq=0, max_bin=255,
                lambda_l1=0.0, lambda_l2=0.0, extra_trees=False, min_gain_to_split=0.0,
                feature_pre_filter=False, force_col_wise=True, deterministic=True,
                seed=42, num_threads=threads, verbosity=-1)


def config_name(year, structure, leaf):
    return f'{year}_{structure}_{leaf}'


def fit_one(year, structure_key, leaf_key, output, threads=8):
    import lightgbm as lgb
    structure, leaf_fraction = STRUCTURES[structure_key], LEAF_FRACTIONS[leaf_key]
    output.mkdir(parents=True, exist_ok=True)
    timer, started = Timer(), time.perf_counter()
    with timer('load_cache'):
        data = cache.load(year, FIXED['feature_set'])
    names, manifest = data['names'], data['manifest']
    sources = [Path(__file__), Path(common.__file__), Path(cache.__file__)]
    config = dict(stage=1, year=year, structure=structure_key, leaf=leaf_key, leaf_fraction=leaf_fraction,
                  fixed=FIXED, threads=threads, cache_hashes=manifest['hashes'],
                  cache_params=manifest['params'], test_used=False,
                  sources={p.name: common.sha256(p) for p in sources})
    config_path = output / 'config.json'
    if config_path.exists() and common.read_json(config_path) != json.loads(json.dumps(config, default=str)):
        raise RuntimeError(f'{output}: existing output belongs to a different configuration or code version')
    common.save_json(config_path, config)

    x_fit, y_fit, fit_dates = data['fit_x'], data['fit_y'], data['fit_keys']['signal_date'].to_numpy()
    x_es, y_es = data['es_x'], data['es_y']
    params = lgbm_params(structure, leaf_fraction, len(y_fit), threads)
    monitor = common.RankICEarlyStopping(y_es, data['es_keys']['signal_date'].to_numpy(),
                                         every=FIXED['every'], patience=FIXED['patience'])
    with timer('dataset_construct'):
        train_set = lgb.Dataset(x_fit, label=y_fit, feature_name=names, free_raw_data=False, params=params)
        valid_set = lgb.Dataset(x_es, label=y_es, reference=train_set, params=params)
        train_set.construct()
        valid_set.construct()
    with timer('train'):
        monitor.started = time.perf_counter()
        booster = lgb.train(params, train_set, num_boost_round=FIXED['cap'], valid_sets=[valid_set],
                            valid_names=['early_stop'], feval=monitor.feval, callbacks=monitor.callbacks())
    best = monitor.best_iteration
    if best <= 0:
        raise RuntimeError('No valid early-stopping score was recorded')
    trained_rounds = monitor.iteration  # booster is truncated to best_iteration after an early stop
    peak_after_train = common.peak_rss_gb()
    print(f'[{year} {structure_key} {leaf_key}] rounds={trained_rounds} best={best} '
          f'es RankIC={monitor.best_score:.5f} train={timer.records["train"]:.0f}s', flush=True)

    with timer('diagnostics'):
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(y_fit), min(300_000, len(y_fit)), replace=False))
        train_ic = common.DailyRankIC(y_fit[idx], fit_dates[idx])(booster.predict(x_fit[idx], num_iteration=best))
        diagnostics = dict(trees=common.tree_diagnostics(booster, best),
                           leaf_coverage=common.leaf_date_coverage(booster, x_fit, fit_dates, best,
                                                                   trees=FIXED['leaf_trees']),
                           inner_fit_rank_ic=train_ic, early_stop_rank_ic=monitor.best_score,
                           fit_minus_early_stop_rank_ic=train_ic - monitor.best_score)
    del train_set, valid_set
    gc.collect()

    with timer('outer_predict'):
        outer_x = data['outer_x']
        pred = np.concatenate([booster.predict(np.asarray(outer_x[s:s + 200_000]), num_iteration=best)
                               for s in range(0, len(outer_x), 200_000)])
    outer = data['outer'].assign(lgbm=pred)
    if not np.isfinite(pred).all():
        raise RuntimeError('Non-finite outer predictions')
    with timer('outer_metrics'):
        metrics, daily = common.evaluate_outer(outer, 'lgbm')
        metrics['timing_split'] = common.timing_selection_split(outer, 'lgbm')
        # Date strings differ in resolution between the cached Elastic table and this run; pair on dates.
        left = daily[['date', 'rank_ic']].assign(date=lambda d: pd.to_datetime(d['date']).dt.normalize())
        right = data['elastic_daily'][['date', 'rank_ic']].assign(
            date=lambda d: pd.to_datetime(d['date']).dt.normalize())
        paired = left.merge(right, on='date', suffixes=('_lgbm', '_elastic'), validate='one_to_one')
        if len(paired) != len(left) or len(paired) != len(right):
            raise RuntimeError(f'LightGBM/Elastic daily pairing incomplete: {len(paired)} of {len(left)}/{len(right)}')
        difference = paired['rank_ic_lgbm'] - paired['rank_ic_elastic']
        comparison = dict(days=len(paired), elastic_mean_rank_ic=data['elastic_metrics']['mean_rank_ic'],
                          mean_rank_ic_difference=float(difference.mean()),
                          newey_west_t_lag5=common.newey_west_tstat(difference.to_numpy(), lag=5))

    (output / 'lightgbm_model.txt').write_text(booster.model_to_string(num_iteration=best), encoding='utf-8')
    outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(output / 'outer_predictions.parquet', index=False)
    pd.DataFrame(monitor.curve, columns=['iteration', 'early_stop_rank_ic', 'seconds']).to_parquet(
        output / 'early_stop_curve.parquet', index=False)
    daily.to_parquet(output / 'daily_metrics.parquet', index=False)

    laps = monitor.lap_times
    per_100 = [round(b[1] - a[1], 2) for a, b in zip([(0, 0.0)] + laps[:-1], laps)]
    rows = manifest['rows']
    summary = dict(
        status='completed', stage=1, year=year, structure=structure_key, leaf=leaf_key,
        rows=dict(training_pool=rows['training_pool'], full_sample=rows['full_sample'],
                  full_sample_below_target=rows['full_sample'] < cache.ROWS,
                  inner_fit=len(y_fit), early_stop=len(y_es), outer=len(outer),
                  outer_scored=int(np.isfinite(outer['y']).sum()),
                  full_to_inner_ratio=rows['full_sample'] / len(y_fit)),
        lightgbm=dict(params=params, rounds_trained=trained_rounds, best_iteration=best,
                      hit_round_limit=monitor.hit_round_limit(FIXED['cap']),
                      early_stop_best_rank_ic=monitor.best_score),
        outer_metrics=metrics, lgbm_vs_elastic=comparison, diagnostics=diagnostics,
        timing_seconds=dict(**timer.records, total=round(time.perf_counter() - started, 1),
                            seconds_per_100_rounds=per_100,
                            mean_seconds_per_100_rounds=float(np.mean(per_100)) if per_100 else None),
        peak_memory_gb=common.peak_rss_gb(), peak_memory_gb_after_lgbm_train=peak_after_train,
        machine=dict(platform=platform.platform(), processor=platform.processor(), threads=threads,
                     lightgbm=lgb.__version__),
        test_used=False)
    common.save_json(output / 'summary.json', summary)  # written last: marks a completed fit
    print(f'[{year} {structure_key} {leaf_key}] outer RankIC={metrics["mean_rank_ic"]:.5f} '
          f'vs elastic {comparison["elastic_mean_rank_ic"]:.5f} NW t={comparison["newey_west_t_lag5"]:.2f} '
          f'leaves={diagnostics["trees"]["mean_leaves"]:.1f} depth={diagnostics["trees"]["mean_depth"]:.2f} '
          f'total={summary["timing_seconds"]["total"]:.0f}s peak={summary["peak_memory_gb"]:.2f}GB', flush=True)
    return summary


def completed(output):
    path = output / 'summary.json'
    return path.exists() and common.read_json(path).get('status') == 'completed'


def verify():
    """Re-run the pilot's timing configuration from the cache and compare with the pilot summary."""
    summary = fit_one(2021, 'medium', '0.25pct', VERIFY_OUT) if not completed(VERIFY_OUT) \
        else common.read_json(VERIFY_OUT / 'summary.json')
    pilot = common.read_json(PILOT / 'summary.json')
    data = cache.load(2021, 'full')
    rows = [
        ('inner_fit_rows', pilot['rows']['inner_fit'], summary['rows']['inner_fit'], 0),
        ('early_stop_rows', pilot['rows']['early_stop'], summary['rows']['early_stop'], 0),
        ('outer_rows', pilot['rows']['outer'], summary['rows']['outer'], 0),
        ('outer_scored_rows', pilot['rows']['outer_scored'], summary['rows']['outer_scored'], 0),
        ('min_data_in_leaf', pilot['lightgbm']['params']['min_data_in_leaf'],
         summary['lightgbm']['params']['min_data_in_leaf'], 0),
        ('best_iteration', pilot['lightgbm']['best_iteration'], summary['lightgbm']['best_iteration'], 0),
        ('rounds_trained', pilot['lightgbm']['rounds_trained'], summary['lightgbm']['rounds_trained'], 0),
        ('early_stop_rank_ic', pilot['lightgbm']['early_stop_best_rank_ic'],
         summary['lightgbm']['early_stop_best_rank_ic'], 1e-6),
        ('outer_lgbm_mean_rank_ic', pilot['outer_metrics']['lgbm']['mean_rank_ic'],
         summary['outer_metrics']['mean_rank_ic'], 1e-6),
        ('outer_lgbm_top50_minus_pool', pilot['outer_metrics']['lgbm']['top50_minus_pool_5d_simple_return'],
         summary['outer_metrics']['top50_minus_pool_5d_simple_return'], 1e-9),
        ('outer_elastic_mean_rank_ic', pilot['outer_metrics']['elastic']['mean_rank_ic'],
         data['elastic_metrics']['mean_rank_ic'], 1e-6),
        ('lgbm_vs_elastic_paired_days', pilot['lgbm_vs_elastic']['days'], summary['lgbm_vs_elastic']['days'], 0),
        ('lgbm_vs_elastic_nw_t', pilot['lgbm_vs_elastic']['newey_west_t_lag5'],
         summary['lgbm_vs_elastic']['newey_west_t_lag5'], 1e-4),
        ('mean_leaves', pilot['diagnostics']['trees']['mean_leaves'],
         summary['diagnostics']['trees']['mean_leaves'], 0),
    ]
    # Row-level: compare the cached outer frame and predictions with the pilot's saved outer predictions.
    pilot_outer = pd.read_parquet(PILOT / 'outer_predictions.parquet')
    ours = pd.read_parquet(VERIFY_OUT / 'outer_predictions.parquet')
    same_keys = bool((pilot_outer['ts_code'].to_numpy() == ours['ts_code'].to_numpy()).all()
                     and (pd.to_datetime(pilot_outer['signal_date']).to_numpy()
                          == pd.to_datetime(ours['signal_date']).to_numpy()).all()) \
        if len(pilot_outer) == len(ours) else False
    lgbm_diff = float(np.max(np.abs(pilot_outer['lgbm'].to_numpy() - ours['lgbm'].to_numpy()))) if same_keys else None
    elastic_diff = float(np.max(np.abs(pilot_outer['elastic'].to_numpy() - data['elastic_pred']))) if same_keys else None
    y_same = bool(np.array_equal(pilot_outer['y'].to_numpy(), data['outer']['y'].to_numpy(), equal_nan=True)
                  and np.array_equal(pilot_outer['eligible'].to_numpy(), data['outer']['eligible'].to_numpy())
                  and np.array_equal(pilot_outer['vol20'].to_numpy(), data['outer']['vol20'].to_numpy(),
                                     equal_nan=True)) if same_keys else False
    report = dict(items=[dict(name=n, pilot=a, cache=b, abs_diff=None if isinstance(a, bool) else abs(a - b),
                              tolerance=tol, passed=abs(a - b) <= tol) for n, a, b, tol in rows],
                  outer_rows_same_keys_and_order=same_keys,
                  outer_y_eligible_vol20_identical=y_same,
                  outer_lgbm_prediction_max_abs_diff=lgbm_diff,
                  outer_elastic_prediction_max_abs_diff=elastic_diff)
    report['passed'] = bool(all(i['passed'] for i in report['items']) and same_keys and y_same
                            and lgbm_diff is not None and lgbm_diff <= 1e-12
                            and elastic_diff is not None and elastic_diff <= 1e-9)
    common.save_json(VERIFY_OUT / 'equivalence.json', report)
    for item in report['items']:
        print(f"{'OK ' if item['passed'] else 'BAD'} {item['name']}: pilot={item['pilot']} cache={item['cache']}")
    print(f'rows same keys/order={same_keys} labels identical={y_same} '
          f'max|dpred| lgbm={lgbm_diff} elastic={elastic_diff}')
    print('EQUIVALENCE', 'PASSED' if report['passed'] else 'FAILED', flush=True)
    if not report['passed']:
        raise SystemExit(1)


def run_all(threads):
    started = time.perf_counter()
    for year in YEARS:
        for structure in STRUCTURES:
            for leaf in LEAF_FRACTIONS:
                output = OUT / config_name(year, structure, leaf)
                if completed(output):
                    print(f'[skip] {output.name} already completed', flush=True)
                    continue
                print(f'[start] {output.name} at {time.strftime("%H:%M:%S")}', flush=True)
                code = subprocess.call([sys.executable, '-u', __file__, '--one', str(year), structure, leaf,
                                        '--threads', str(threads)])
                if code != 0:
                    raise SystemExit(f'{output.name} failed with exit code {code}; stopping')
    print(f'[done] all 27 fits complete; this invocation took {time.perf_counter() - started:.0f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--one', nargs=3, metavar=('YEAR', 'STRUCTURE', 'LEAF'))
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.one:
        year, structure, leaf = int(args.one[0]), args.one[1], args.one[2]
        if year not in YEARS or structure not in STRUCTURES or leaf not in LEAF_FRACTIONS:
            raise ValueError('Not a stage 1 configuration')
        fit_one(year, structure, leaf, OUT / config_name(year, structure, leaf), args.threads)
    elif args.summarize:
        from summarize_lgbm_v2_stage1 import summarize
        summarize()
    else:
        run_all(args.threads)


if __name__ == '__main__':
    main()
