"""V2 LightGBM stage 8 (review flow on the complete 2M-row sample), final selection and freeze (plan section 17).

  python v2/experiments/run_lgbm_v2_review.py --build-full YEAR     complete 2M matrix (fit + es cache + missing keys fetched)
  python v2/experiments/run_lgbm_v2_review.py --one YEAR CONFIG SEED
  python v2/experiments/run_lgbm_v2_review.py --elastic YEAR
  python v2/experiments/run_lgbm_v2_review.py --all                 build -> 45 refits -> 5 Elastic(D) -> summary -> select -> freeze
  python v2/experiments/run_lgbm_v2_review.py --summarize
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import gc
import gzip
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
from lgbm_v2_common import DATA, KEY, ROOT, Timer
from run_lgbm_v2_stage2 import PENALTY, YEARS, eligible_metrics, paired_t
import run_lgbm_v2_input as inp
import run_lgbm_v2_final as fin
import run_lgbm_v2_stage6a as s6

FULL = ROOT / 'lgbm_full_cache'          # F: project directory
OUT = fin.OUT / 'review'
FINAL = fin.OUT
FULL_VERSION = 1
SIMPLICITY = ['F3', 'F1', 'F2']           # medium < deep; same structure: larger lambda2 = simpler
MIN_GAIN, MIN_T = 0.001, 2.0


def norm(frame):
    f = frame[KEY].copy()
    f['signal_date'] = pd.to_datetime(f['signal_date']).dt.normalize()
    return f


# ---------------------------------------------------------------- complete 2M-row matrix
def full_dir(year):
    return FULL / f'{year}'


def full_params(year):
    return dict(full_version=FULL_VERSION, year=int(year), feature_set='full', rows=cache.ROWS,
                sample_seed=common.SAMPLE_SEED, training_gate=False,
                row_order='full_sample_keys.parquet order (priority, ts_code, signal_date)',
                sources='fit_x / es_x from lgbm_cache/{year}_full; keys in neither fetched with common.fetch_matrix',
                d_label='daily rank of raw label among all stocks with a finite raw label that day (same as 6a)',
                dtype_x='float32')


def build_full(year, threads=8):
    out = full_dir(year)
    if (out / 'manifest.json').exists():
        load_full(year)
        print(f'[full {year}] exists and verifies', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = cache.load(year, 'full')
    names = base['names']
    full_keys = norm(pd.read_parquet(base['dir'] / 'full_sample_keys.parquet'))
    assert cache.key_hash(full_keys) == base['manifest']['hashes']['full_sample_keys'], 'full_sample_keys hash differs'
    assert len(full_keys) == cache.ROWS
    fit_keys = norm(base['fit_keys']).assign(i_fit=np.arange(len(base['fit_y'])))
    es_keys = norm(base['es_keys']).assign(i_es=np.arange(len(base['es_y'])))
    m = full_keys.assign(pos=np.arange(len(full_keys))).merge(fit_keys, on=KEY, how='left', validate='one_to_one') \
        .merge(es_keys, on=KEY, how='left', validate='one_to_one').sort_values('pos')
    assert not (m['i_fit'].notna() & m['i_es'].notna()).any(), 'a key is in both fit and es'
    in_fit, in_es = m['i_fit'].notna().to_numpy(), m['i_es'].notna().to_numpy()
    # every inner-fit key must be part of the full sample (inner fit is a subset of the full sample)
    assert in_fit.sum() == len(fit_keys), 'some inner-fit keys are not in the full sample'
    missing = m.loc[~in_fit & ~in_es, KEY]
    sy = m['signal_date'].dt.year
    missing_by_year = sy[~in_fit & ~in_es].value_counts().sort_index()
    # fetch the full sample (keys, y, exit) from the database exactly as the cache builder defines it
    con = common.open_connection(out, threads, '6GB')
    common.create_views(con, year)
    pool_rows = common.create_pool(con, year)
    assert pool_rows == base['manifest']['rows']['training_pool'], 'training pool differs from the base cache'
    common.create_samples(con, year, cache.ROWS, cache.EARLY_ROWS)
    checks = dict(boundaries=common.boundary_checks(con, year))
    db = con.execute('SELECT ts_code, signal_date, CAST(exit_date AS DATE) exit_date, y FROM full_sample '
                     'ORDER BY priority, ts_code, signal_date').fetch_df()
    db_keys = norm(db)
    assert (db_keys['ts_code'].to_numpy() == full_keys['ts_code'].to_numpy()).all() and \
        (db_keys['signal_date'].to_numpy() == full_keys['signal_date'].to_numpy()).all(), 'DB full sample order differs'
    exit_max = pd.to_datetime(db['exit_date']).max()
    assert exit_max < pd.Timestamp(f'{year}-01-01'), f'{year}: a full-sample row exits in the outer year'
    checks['max_exit_date'] = str(exit_max.date())
    checks['labels_full_sample'] = common.label_check(con, 'full_sample', cache.LABEL_ROWS)
    # fetch only the missing keys, with the cache builder's code
    con.register('missing_keys_df', missing.assign(signal_date=missing['signal_date'].dt.date))
    con.execute('CREATE OR REPLACE TEMP TABLE missing AS SELECT s.* FROM full_sample s '
                'JOIN missing_keys_df k ON k.ts_code = s.ts_code AND k.signal_date = s.signal_date')
    n_missing = con.execute('SELECT count(*) FROM missing').fetchone()[0]
    assert n_missing == len(missing), f'missing keys not all found in full_sample ({n_missing} of {len(missing)})'
    checks['labels_missing_rows'] = common.label_check(con, 'missing', min(cache.LABEL_ROWS, n_missing))
    t = time.perf_counter()
    mx, my, mkeys = common.fetch_matrix(con, 'missing', names)
    fetch_seconds = round(time.perf_counter() - t, 1)
    con.close()
    mkeys = norm(mkeys).assign(i_miss=np.arange(len(my)))
    m = m.merge(mkeys, on=KEY, how='left', validate='one_to_one').sort_values('pos')
    src = np.where(in_fit, 0, np.where(in_es, 1, 2))
    assert ((src == 2) == m['i_miss'].notna().to_numpy()).all(), 'missing-key mapping inconsistent'
    # assemble in full_sample_keys order
    x = np.lib.format.open_memmap(out / 'full_x.npy', mode='w+', dtype=np.float32, shape=(cache.ROWS, len(names)))
    y = np.empty(cache.ROWS)
    for code, arr_x, arr_y, col in ((0, base['fit_x'], base['fit_y'], 'i_fit'), (1, base['es_x'], base['es_y'], 'i_es'),
                                    (2, mx, my, 'i_miss')):
        rows = np.flatnonzero(src == code)
        idx = m[col].to_numpy()[rows].astype(np.int64)
        for s in range(0, len(rows), 200_000):
            x[rows[s:s + 200_000]] = arr_x[idx[s:s + 200_000]]
        y[rows] = arr_y[idx]
    x.flush()
    # assertions: values equal to fit / es caches on shared keys; labels equal to the database full sample
    for code, arr_x, arr_y, col, name in ((0, base['fit_x'], base['fit_y'], 'i_fit', 'fit'),
                                          (1, base['es_x'], base['es_y'], 'i_es', 'es')):
        rows = np.flatnonzero(src == code)
        idx = m[col].to_numpy()[rows].astype(np.int64)
        for s in range(0, len(rows), 200_000):
            assert np.array_equal(np.asarray(x[rows[s:s + 200_000]]), arr_x[idx[s:s + 200_000]], equal_nan=True), \
                f'{name}: assembled values differ from the cache'
        assert np.array_equal(y[rows], arr_y[idx]), f'{name}: labels differ from the cache'
        checks[f'equal_to_{name}_cache_rows'] = int(len(rows))
    assert np.array_equal(y, db['y'].to_numpy()), 'assembled labels differ from the database full sample'
    assert np.isfinite(y).all()
    checks['labels_equal_to_database_full_sample'] = True
    del x
    # D label on the complete sample
    d, dinfo = inp.d_labels_for(year, full_keys, y)
    base_d = np.load(s6.LABELS / f'{year}_D.npy')
    fit_rows = np.flatnonzero(src == 0)
    assert np.array_equal(d[fit_rows], base_d[m['i_fit'].to_numpy()[fit_rows].astype(np.int64)]), \
        'D labels on inner-fit keys differ from stage 6a'
    checks['d_equal_to_6a_on_fit_keys'] = int(len(fit_rows))
    np.save(out / 'full_y.npy', y)
    np.save(out / 'full_d.npy', d)
    cache._save_keys(out / 'full_keys.parquet', full_keys)
    common.save_json(out / 'checks.json', checks)
    counts = dict(from_fit=int((src == 0).sum()), from_es=int((src == 1).sum()), fetched_missing=int((src == 2).sum()),
                  fetched_by_signal_year={str(k): int(v) for k, v in missing_by_year.items()},
                  es_rows_not_in_full_sample=int(len(es_keys) - (src == 1).sum()))
    manifest = dict(params=full_params(year), feature_names=names, rows=counts,
                    shapes=dict(full_x=[cache.ROWS, len(names)], full_y=[cache.ROWS], full_d=[cache.ROWS]),
                    hashes=dict(full_keys=cache.key_hash(full_keys), full_keys_and_y=cache.key_hash(full_keys, y),
                                full_d=cache.key_hash(full_keys, d), base_cache=base['manifest']['hashes']),
                    file_bytes={p.name: p.stat().st_size for p in sorted(out.glob('*.npy'))},
                    d_label=dinfo, checks=checks,
                    source_sha256_at_build={p.name: common.sha256(p) for p in (Path(__file__), Path(common.__file__),
                                                                                Path(cache.__file__), Path(inp.__file__))},
                    timing_seconds=dict(fetch_missing=fetch_seconds, total=round(time.perf_counter() - started, 1)),
                    test_used=False)
    common.save_json(out / 'manifest.json', manifest)
    load_full(year)
    print(f'[full {year}] fit {counts["from_fit"]:,} + es {counts["from_es"]:,} + fetched {counts["fetched_missing"]:,} '
          f'{counts["fetched_by_signal_year"]}; max exit {checks["max_exit_date"]}; ratio D {dinfo["ratio_D"]:.3f}; '
          f'{manifest["timing_seconds"]["total"]}s', flush=True)


def load_full(year):
    out = full_dir(year)
    manifest = common.read_json(out / 'manifest.json')
    if manifest['params'] != full_params(year) or manifest['feature_names'] != common.feature_names(year, 'full'):
        raise RuntimeError(f'{out}: params or feature list differ')
    for name, size in manifest['file_bytes'].items():
        if (out / name).stat().st_size != size:
            raise RuntimeError(f'{out}: {name} size changed')
    data = dict(manifest=manifest, x=np.load(out / 'full_x.npy'), y=np.load(out / 'full_y.npy'), d=np.load(out / 'full_d.npy'))
    keys = pd.read_parquet(out / 'full_keys.parquet')
    if cache.key_hash(keys, data['y']) != manifest['hashes']['full_keys_and_y'] or \
            cache.key_hash(keys, data['d']) != manifest['hashes']['full_d']:
        raise RuntimeError(f'{out}: key/label hash mismatch')
    assert list(data['x'].shape) == manifest['shapes']['full_x'] and data['x'].dtype == np.float32
    data['keys'] = keys
    data['dates'] = keys['signal_date'].to_numpy().astype('datetime64[D]')
    return data


# ---------------------------------------------------------------- one review-flow refit
def review_dir(year, config, seed):
    return OUT / f'{year}_{config}_s{seed}'


def completed(output):
    return fin.completed(output)


def fit_review(year, config, seed, threads=8):
    import lightgbm as lgb
    output = review_dir(year, config, seed)
    output.mkdir(parents=True, exist_ok=True)
    timer, started = Timer(), time.perf_counter()
    sel = common.read_json(fin.seed_dir(year, config, seed) / 'summary.json')
    rounds = int(sel['lightgbm']['best_iteration'])
    l1 = float(sel['lightgbm']['params']['lambda_l1'])
    with timer('load'):
        data = load_full(year)
        base = cache.load(year, 'full')
    names = common.feature_names(year, 'full')
    params = fin.params_for(config, s6.build_labels(year)['ratio_D'], cache.ROWS, seed, threads)
    assert params['lambda_l1'] == l1, 'lambda1 differs from the selection flow'
    assert params['min_data_in_leaf'] == math.ceil(0.001 * cache.ROWS)
    sel_params = dict(sel['lightgbm']['params'])
    for k in params:
        if k != 'min_data_in_leaf':
            assert params[k] == sel_params[k], f'{k}: {params[k]} vs selection flow {sel_params[k]}'
    sources = [Path(__file__), Path(fin.__file__), Path(inp.__file__), Path(common.__file__), Path(cache.__file__),
               _source_path('v2/experiments/run_lgbm_v2_stage1.py'), _source_path('v2/experiments/run_lgbm_v2_stage2.py'), _source_path('v2/experiments/run_lgbm_v2_stage6a.py')]
    config_json = dict(stage=8, year=year, config=config, seed=seed, rounds=rounds, params=params,
                       full_hashes=data['manifest']['hashes'], test_used=False,
                       sources={s.name: common.sha256(s) for s in sources})
    path = output / 'config.json'
    if path.exists() and common.read_json(path) != json.loads(json.dumps(config_json, default=str)):
        raise RuntimeError(f'{output}: existing output belongs to a different configuration or code version')
    common.save_json(path, config_json)
    with timer('dataset_construct'):
        train_set = lgb.Dataset(data['x'], label=data['d'], feature_name=names, free_raw_data=False, params=params)
        train_set.construct()
    trees_per_round = []

    def record_trees(env):
        trees_per_round.append(env.model.num_trees())
    record_trees.order = 20
    with timer('train'):
        booster = lgb.train(params, train_set, num_boost_round=rounds, callbacks=[record_trees])
    trees_built = booster.num_trees()
    assert len(trees_per_round) == rounds and trees_per_round[-1] == trees_built
    skipped = rounds - trees_built
    with timer('save_model'):
        text = booster.model_to_string(num_iteration=-1)
        seeds_used = fin.derived_seeds(text)
        with gzip.open(output / 'lightgbm_model.txt.gz', 'wt', encoding='utf-8', compresslevel=6) as h:
            h.write(text)
        del text
        np.save(output / 'trees_per_round.npy', np.asarray(trees_per_round, dtype=np.int32))
    with timer('diagnostics'):
        trees = common.tree_diagnostics(booster, trees_built)
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(cache.ROWS, 300_000, replace=False))
        idx = idx[np.argsort(data['dates'][idx], kind='stable')]
        train_ic = common.DailyRankIC(data['y'][idx], data['dates'][idx])(booster.predict(data['x'][idx]))
    del train_set
    data['x'] = None
    gc.collect()
    with timer('outer_predict'):
        pred = np.concatenate([booster.predict(a) for a in inp.outer_chunks(base['outer_x'], None)])
    assert np.isfinite(pred).all()
    outer = base['outer'].assign(lgbm=pred)
    metrics, daily = common.evaluate_outer(outer, 'lgbm')
    eligible, eligible_daily = eligible_metrics(outer, 'lgbm')
    outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(output / 'outer_predictions.parquet', index=False)
    daily.to_parquet(output / 'daily_metrics.parquet', index=False)
    eligible_daily.to_parquet(output / 'eligible_daily_metrics.parquet', index=False)
    summary = dict(status='completed', stage=8, year=year, config=config, seed=seed, derived_seeds=seeds_used,
                   rows=dict(fit=cache.ROWS, outer=len(outer), outer_eligible_scored=eligible['rows']),
                   lightgbm=dict(params=params, rounds=rounds, trees_built=trees_built, skipped_rounds=skipped,
                                 selection_flow_best_iteration=rounds,
                                 selection_flow_eligible=sel['eligible_metrics']['mean_rank_ic']),
                   eligible_metrics=eligible, outer_metrics=metrics,
                   diagnostics=dict(trees=trees, full_sample_rank_ic=train_ic),
                   timing_seconds=dict(**timer.records, total=round(time.perf_counter() - started, 1)),
                   peak_memory_gb=common.peak_rss_gb(),
                   machine=dict(platform=platform.platform(), threads=threads, lightgbm=lgb.__version__), test_used=False)
    common.save_json(output / 'summary.json', summary)
    print(f'[review {year} {config} s{seed}] rounds={rounds} trees={trees_built} skipped={skipped} '
          f'eligible={eligible["mean_rank_ic"]:.5f} (selection {sel["eligible_metrics"]["mean_rank_ic"]:.5f}) '
          f'all={metrics["mean_rank_ic"]:.5f} leaves={trees["mean_leaves"]:.1f} depth={trees["mean_depth"]:.1f} '
          f'train={timer.records["train"]:.0f}s', flush=True)


# ---------------------------------------------------------------- Elastic(D) on the complete sample
def elastic_dir(year):
    return OUT / 'elastic' / f'{year}'


def fit_elastic_full(year):
    from run_lgbm_v2_pilot import fit_elastic, predict_elastic
    out = elastic_dir(year)
    if (out / 'metrics.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = load_full(year)
    base = cache.load(year, 'full')
    names = base['names']
    model = fit_elastic(data['x'], data['d'], names)
    data['x'] = None
    pred = np.concatenate([predict_elastic(model, a) for a in inp.outer_chunks(base['outer_x'], None)])
    frame = base['outer'].assign(elastic=pred)
    elig, elig_daily = eligible_metrics(frame, 'elastic')
    allm, all_daily = common.evaluate_outer(frame, 'elastic')
    np.save(out / 'outer_pred.npy', pred)
    elig_daily.to_parquet(out / 'eligible_daily_metrics.parquet', index=False)
    all_daily.to_parquet(out / 'daily_metrics.parquet', index=False)
    common.save_json(out / 'model.json', dict(coefficient=np.asarray(model['coefficient']).tolist(),
                                              intercept=model['intercept'], alpha=model['alpha'], nonzero=model['nonzero'],
                                              iterations=model['iterations'], active_names=model['stats']['active_names'],
                                              label='D', sample='complete 2M rows'))
    metrics = dict(year=year, eligible=elig, all_sample_mean_rank_ic=allm['mean_rank_ic'], alpha=model['alpha'],
                   nonzero=model['nonzero'], rows=cache.ROWS, seconds=round(time.perf_counter() - started, 1))
    common.save_json(out / 'metrics.json', metrics)
    print(f'[elastic D full {year}] eligible {elig["mean_rank_ic"]:.5f} ({model["nonzero"]} nonzero, '
          f'{metrics["seconds"]:.0f}s)', flush=True)


# ---------------------------------------------------------------- summary, selection, freeze
def score_array(values):
    v = np.asarray(values, float)
    return float(v.mean() - PENALTY * v.std(ddof=0))


def daily_from(dirs):
    parts = []
    for y, d in zip(YEARS, dirs):
        f = pd.read_parquet(d / 'eligible_daily_metrics.parquet')
        f['date'] = pd.to_datetime(f['date']).dt.normalize()
        parts.append(f.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def averaged(config):
    """Three-seed average of review-flow predictions per fold; writes eligible daily tables."""
    elig, allm, out_dirs, top = [], [], [], []
    for y in YEARS:
        frames = [pd.read_parquet(review_dir(y, config, s) / 'outer_predictions.parquet') for s in fin.SEEDS]
        for f in frames[1:]:
            assert (f['ts_code'].to_numpy() == frames[0]['ts_code'].to_numpy()).all()
        outer = cache.load(y, 'full')['outer']
        assert (outer['ts_code'].to_numpy() == frames[0]['ts_code'].to_numpy()).all()
        outer = outer.assign(lgbm=np.mean([f['lgbm'].to_numpy() for f in frames], axis=0))
        e, ed = eligible_metrics(outer, 'lgbm')
        m, dd = common.evaluate_outer(outer, 'lgbm')
        d = OUT / 'average' / f'{y}_{config}'
        d.mkdir(parents=True, exist_ok=True)
        outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(d / 'outer_predictions.parquet', index=False)
        ed.to_parquet(d / 'eligible_daily_metrics.parquet', index=False)
        elig.append(e['mean_rank_ic'])
        allm.append(m['mean_rank_ic'])
        top.append((m['top50_minus_pool_5d_simple_return'], m['consecutive_day_top50_overlap']))
        out_dirs.append(d)
    return elig, allm, out_dirs, top


def nw(a, b):
    j = a.merge(b, on=['year', 'date'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(j) != len(a) or len(j) != len(b):
        raise RuntimeError('Incomplete daily pairing')
    diff = (j['rank_ic_a'] - j['rank_ic_b']).to_numpy()
    return float(diff.mean()), common.newey_west_tstat(diff, lag=5)


def summarize():
    out = OUT / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    rows, per_seed, fits, daily = [], [], [], {}
    e_dirs = [elastic_dir(y) for y in YEARS]
    e_daily = daily_from(e_dirs)
    e_elig = [common.read_json(d / 'metrics.json')['eligible']['mean_rank_ic'] for d in e_dirs]
    e_all = [common.read_json(d / 'metrics.json')['all_sample_mean_rank_ic'] for d in e_dirs]
    sel_table = pd.read_csv(fin.OUT / 'stage7' / 'seed_scores.csv')
    for config in fin.CONFIGS:
        elig, allm, dirs, top = averaged(config)
        daily[config] = daily_from(dirs)
        per_fold_t = []
        for y, d in zip(YEARS, dirs):
            a = pd.read_parquet(d / 'eligible_daily_metrics.parquet')
            b = pd.read_parquet(elastic_dir(y) / 'eligible_daily_metrics.parquet')
            for f in (a, b):
                f['date'] = pd.to_datetime(f['date']).dt.normalize()
            per_fold_t.append(paired_t(a, b)['newey_west_t_lag5'])
        ediff, et = nw(daily[config], e_daily)
        ss = [common.read_json(review_dir(y, config, s) / 'summary.json') for y in YEARS for s in fin.SEEDS]
        sel_avg = float(sel_table[(sel_table['config'] == config) & (sel_table['seed'] == 'avg3')]['eligible_score'].iloc[0])
        sel_42 = float(sel_table[(sel_table['config'] == config) & (sel_table['seed'] == '42')]['eligible_score'].iloc[0])
        rows.append(dict(config=config, **{f'eligible_{y}': v for y, v in zip(YEARS, elig)},
                         review_score=score_array(elig), all_score=score_array(allm),
                         negative_fold=bool(min(elig) < 0),
                         selection_avg3_score=sel_avg, change_vs_selection_avg3=score_array(elig) - sel_avg,
                         selection_seed42_score=sel_42, change_vs_selection_seed42=score_array(elig) - sel_42,
                         elastic_concat_diff=ediff, elastic_concat_t=et,
                         elastic_fold_t=' / '.join(f'{t:.2f}' for t in per_fold_t),
                         mean_leaves=float(np.mean([s['diagnostics']['trees']['mean_leaves'] for s in ss])),
                         mean_depth=float(np.mean([s['diagnostics']['trees']['mean_depth'] for s in ss])),
                         max_depth=int(np.max([s['diagnostics']['trees']['max_depth'] for s in ss])),
                         top50_minus_pool=float(np.mean([t[0] for t in top])),
                         top50_overlap=float(np.mean([t[1] for t in top])),
                         train_seconds_mean=float(np.mean([s['timing_seconds']['train'] for s in ss])),
                         train_seconds_total=float(np.sum([s['timing_seconds']['total'] for s in ss]))))
        for s in fin.SEEDS:
            sv = [common.read_json(review_dir(y, config, s) / 'summary.json') for y in YEARS]
            ev = [x['eligible_metrics']['mean_rank_ic'] for x in sv]
            per_seed.append(dict(config=config, seed=s, **{f'eligible_{y}': v for y, v in zip(YEARS, ev)},
                                 review_score=score_array(ev),
                                 all_score=score_array([x['outer_metrics']['mean_rank_ic'] for x in sv])))
            for y, x in zip(YEARS, sv):
                lg = x['lightgbm']
                fits.append(dict(config=config, seed=s, year=y, rounds=lg['rounds'], trees_built=lg['trees_built'],
                                 skipped_rounds=lg['skipped_rounds'], min_data_in_leaf=lg['params']['min_data_in_leaf'],
                                 lambda_l1=lg['params']['lambda_l1'], eligible=x['eligible_metrics']['mean_rank_ic'],
                                 selection_eligible=lg['selection_flow_eligible'],
                                 mean_leaves=x['diagnostics']['trees']['mean_leaves'],
                                 mean_depth=x['diagnostics']['trees']['mean_depth'],
                                 train_seconds=x['timing_seconds']['train'],
                                 data_random_seed=x['derived_seeds']['data_random_seed'],
                                 feature_fraction_seed=x['derived_seeds']['feature_fraction_seed']))
    table = pd.DataFrame(rows)
    per_seed = pd.DataFrame(per_seed)
    per_seed['avg3_minus_seed'] = per_seed.apply(
        lambda r: float(table.loc[table['config'] == r['config'], 'review_score'].iloc[0]) - r['review_score'], axis=1)
    pairs = []
    for a, b in (('F1', 'F2'), ('F1', 'F3'), ('F2', 'F3')):
        diff, t = nw(daily[a], daily[b])
        sa = float(table.loc[table['config'] == a, 'review_score'].iloc[0])
        sb = float(table.loc[table['config'] == b, 'review_score'].iloc[0])
        pairs.append(dict(pair=f'{a}-{b}', score_diff=sa - sb, mean_daily_diff=diff, nw_t=t))
    elastic = dict(eligible_by_fold=e_elig, score=score_array(e_elig), all_score=score_array(e_all),
                   selection_flow_elastic=[common.read_json(inp.elastic_dir(y, 'full_none_raw') / 'metrics.json')
                                           ['eligible']['mean_rank_ic'] for y in YEARS])
    elastic['selection_flow_score'] = score_array(elastic['selection_flow_elastic'])
    table.to_csv(out / 'review_table.csv', index=False)
    per_seed.to_csv(out / 'review_per_seed.csv', index=False)
    pd.DataFrame(fits).to_csv(out / 'review_fits.csv', index=False)
    pd.DataFrame(pairs).to_csv(out / 'review_pairs.csv', index=False)
    common.save_json(out / 'elastic_full.json', elastic)
    selection = select(table, daily)
    common.save_json(FINAL / 'selection.json', selection)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(table.round(5).to_string(index=False))
        print(per_seed.round(5).to_string(index=False))
        print(pd.DataFrame(pairs).round(5).to_string(index=False))
    print(f'[elastic full] {elastic}', flush=True)
    print(f'[FINAL] {selection["final"]}  {selection["reason"]}', flush=True)
    return selection


def select(table, daily):
    ok = table[~table['negative_fold']]
    removed = table.loc[table['negative_fold'], 'config'].tolist()
    top = ok.sort_values('review_score', ascending=False).iloc[0]
    T = top['config']
    members, gaps = [], {}
    for _, r in ok.iterrows():
        if r['config'] == T:
            members.append(T)
            continue
        diff, t = nw(daily[T], daily[r['config']])
        gain = float(top['review_score'] - r['review_score'])
        clear = gain >= MIN_GAIN and t >= MIN_T
        gaps[r['config']] = dict(score_gap=gain, mean_daily_diff=diff, nw_t=t, beaten_by_T=bool(clear))
        if not clear:
            members.append(r['config'])
    final = min(members, key=SIMPLICITY.index)
    reason = (f'T = {T} (review score {top["review_score"]:.5f}); plateau = {sorted(members, key=SIMPLICITY.index)}; '
              f'simplest in plateau (medium < deep; same structure larger lambda2) = {final}')
    return dict(removed_negative_fold=removed, T=T, gaps_to_T=gaps, plateau=members, simplicity_order=SIMPLICITY,
                final=final, reason=reason,
                rule='plan section 8 / 17 item 5: drop negative folds; T = highest review score; plateau = configs whose '
                     'gap to T is not (score gap >= 0.001 and NW t >= 2); pick the simplest in the plateau')


def freeze():
    sel = common.read_json(FINAL / 'selection.json')
    config = sel['final']
    c = fin.CONFIGS[config]
    by_fold = []
    for y in YEARS:
        base = common.read_json(cache.cache_dir(y, 'full') / 'manifest.json')
        fm = common.read_json(full_dir(y) / 'manifest.json')
        seeds = []
        for s in fin.SEEDS:
            sm = common.read_json(review_dir(y, config, s) / 'summary.json')
            seeds.append(dict(seed=s, rounds=sm['lightgbm']['rounds'], trees_built=sm['lightgbm']['trees_built'],
                              derived_seeds=sm['derived_seeds']))
        p = common.read_json(review_dir(y, config, 42) / 'summary.json')['lightgbm']['params']
        by_fold.append(dict(year=y, lambda_l1=p['lambda_l1'], min_data_in_leaf_review=p['min_data_in_leaf'],
                            sample_key_hashes=base['hashes'], complete_sample_hashes=fm['hashes'],
                            complete_sample_rows=fm['rows'], seeds=seeds))
    params = {k: v for k, v in common.read_json(review_dir(YEARS[0], config, 42) / 'summary.json')['lightgbm']['params'].items()
              if k not in ('lambda_l1', 'min_data_in_leaf', 'seed')}
    code = [Path(__file__), Path(fin.__file__), Path(inp.__file__), Path(common.__file__), Path(cache.__file__),
            _source_path('v2/experiments/run_lgbm_v2_stage1.py'), _source_path('v2/experiments/run_lgbm_v2_stage2.py'), _source_path('v2/experiments/run_lgbm_v2_stage6a.py'),
            _source_path('v2/experiments/run_lgbm_v2_pilot.py'), _source_path('v2/models/research_controls.py'), _source_path('tune_elastic_net.py')]
    frozen = dict(
        final=config, definition=dict(structure=c['structure'], lambda_l2=c['lambda_l2'],
                                      lambda_l1_rule=f'sig2({c["l1_factor"]} x ratio_D) (2 significant digits), '
                                                     'ratio_D = std(D label) / std(raw label) on the inner fit set',
                                      min_data_in_leaf_rule='ceil(0.001 x actual fit rows)'),
        lightgbm_params=params,
        training_procedure=dict(
            sample='seed 20261025, priority = hash(ts_code, CAST(signal_date AS DATE), 20261025), ordered by priority, '
                   'ts_code, signal_date; training pool = finite raw label, signal and exit < Y-01-01, from 2011-01-01; '
                   'first 2,000,000 rows; no gate; 448 raw factors (feature_names(year, "full"))',
            rounds='inner fit (full-sample rows with signal and exit < Y-1-01-01) + early-stopping set (first 300,000 '
                   'Y-1 pool rows by priority, exit < Y-01-01): rule R1 = eligible... see early_stopping',
            early_stopping=dict(metric='all-sample daily Rank IC of the raw label on the early-stopping set, every 10 rounds',
                                patience=fin.PATIENCE, cap=fin.CAP, rounds_used='best_iteration (rounds, not scaled)'),
            refit='complete 2,000,000-row sample, D label rebuilt on it, num_boost_round = best_iteration of the same '
                  'seed and fold, no early stopping; lambda1 from the selection flow of the fold; '
                  'min_data_in_leaf = ceil(0.001 x 2,000,000) = 2000',
            seeds=fin.SEEDS, prediction='raw predictions of the three seeds averaged'),
        d_label='per signal date, (average rank of the raw 5-day log return - 0.5) / number of stocks with a finite raw '
                'label that day (all stocks, not only sampled rows); exit dates of rows used < Y-01-01',
        scoring=dict(eligible_pool='daily, turnover >= threshold and listed >= 60 days, finite label, >= 20 stocks; '
                                   'equal-weight Spearman per day', score='mean - 0.25 x std (ddof=0) of the five fold '
                                                                          'eligible Rank ICs',
                     significance='Newey-West t, lag 5, on concatenated daily differences'),
        folds=by_fold, selection=sel,
        code_sha256={p.name: common.sha256(p) for p in code},
        frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), test_used=False)
    frozen['training_procedure']['rounds'] = ('inner fit (full-sample rows with signal and exit < Y-1-01-01) + '
                                              'early-stopping set (first 300,000 Y-1 pool rows by priority, exit < Y-01-01); '
                                              'R1: patience 200, cap 6000, evaluated every 10 rounds')
    path = FINAL / 'frozen_config.json'
    assert not path.exists(), f'{path} already exists; frozen configs are not overwritten'
    common.save_json(path, frozen)
    print(f'[freeze] {path}', flush=True)


# ---------------------------------------------------------------- driver
def run_sub(args, label):
    print(f'[start] {label} at {time.strftime("%m-%d %H:%M:%S")}', flush=True)
    code = subprocess.call([sys.executable, '-u', __file__] + args)
    if code != 0:
        raise SystemExit(f'{label} failed with exit code {code}; stopping')


def run_all():
    for y in YEARS:
        if not (full_dir(y) / 'manifest.json').exists():
            run_sub(['--build-full', str(y)], f'build full {y}')
    for y in YEARS:
        if not (elastic_dir(y) / 'metrics.json').exists():
            run_sub(['--elastic', str(y)], f'Elastic(D) full {y}')
    for config in fin.CONFIGS:
        for seed in fin.SEEDS:
            for y in YEARS:
                if not completed(review_dir(y, config, seed)):
                    run_sub(['--one', str(y), config, str(seed)], f'review {y} {config} s{seed}')
    summarize()
    if not (FINAL / 'frozen_config.json').exists():
        freeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--build-full', type=int)
    parser.add_argument('--one', nargs=3)
    parser.add_argument('--elastic', type=int)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.build_full:
        build_full(args.build_full, args.threads)
    elif args.one:
        fit_review(int(args.one[0]), args.one[1], int(args.one[2]), args.threads)
    elif args.elastic:
        fit_elastic_full(args.elastic)
    elif args.summarize:
        summarize()
    elif args.freeze:
        freeze()
    elif args.all:
        started = time.perf_counter()
        run_all()
        print(f'[done] stage 8 complete in {(time.perf_counter() - started) / 3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
