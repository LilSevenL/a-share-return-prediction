"""V2 LightGBM stage 9: confirmation years 2022 / 2023, four prediction sets, evaluation, backtests (plan sections 7, 18).

Only calls the frozen code (sha256 listed in lgbm_final/frozen_config.json, checked at start of every command):
  cache rows       lgbm_v2_common primitives in the same order as lgbm_v2_cache.build (whose 2017-2021 guard is
                   frozen); the replica is verified to reproduce the 2021 cache hashes before 2022 is touched
  D label / ratio  run_lgbm_v2_stage6a.build_labels
  selection flow   run_lgbm_v2_final.fit_seed (F3, output at run_lgbm_v2_final.seed_dir)
  complete sample  run_lgbm_v2_review.build_full / fit_review / fit_elastic_full
  scoring          run_lgbm_v2_stage2.eligible_metrics / paired_t, lgbm_v2_common.evaluate_outer
Backtest: the existing V2 linear-candidate backtest (run_elastic_compact_gate_seeds.backtest calls), unchanged.

  python v2/experiments/run_lgbm_v2_confirm.py --fits          verify replica -> per year: cache, labels, selection x3, full, Elastic, review x3, average
  python v2/experiments/run_lgbm_v2_confirm.py --predictions   four prediction sets 2017-2023
  python v2/experiments/run_lgbm_v2_confirm.py --evaluate      yearly prediction metrics and 2022/2023 paired tests
  python v2/experiments/run_lgbm_v2_confirm.py --backtests     wrapper check + 8 backtests (resumable)
  python v2/experiments/run_lgbm_v2_confirm.py --report        tables for LGBM_CONFIRMATION_RESULTS.md
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from lgbm_v2_common import DATA, KEY, ROOT
from run_lgbm_v2_stage2 import eligible_metrics, paired_t
import run_lgbm_v2_input as inp
import run_lgbm_v2_final as fin
import run_lgbm_v2_review as rv
import run_lgbm_v2_stage6a as s6

CONFIRM_YEARS = [2022, 2023]
ALL_YEARS = list(range(2017, 2024))
CONFIG = 'F3'
OUT = DATA / 'v2' / 'lgbm_confirm'
FROZEN = DATA / 'v2' / 'lgbm_final' / 'frozen_config.json'
CANDIDATE = DATA / 'v2' / 'elastic_compact_gate_seeds'
CANDIDATE_COLUMNS = [f's{s}_features_only' for s in (20261025, 20261026, 20261027)]
GROUPS = ['G1_lgbm_final', 'G2_elastic_D', 'G3_linear_candidate', 'G4_fusion_50_50']
IMPACTS = [0.5, 1.0]


# ---------------------------------------------------------------- frozen-code check
def check_frozen():
    frozen = common.read_json(FROZEN)
    assert frozen['final'] == CONFIG
    bad = {}
    for name, h in frozen['code_sha256'].items():
        p = _source_path('v2/'+name) if (_source_path('v2/'+name)).exists() else ROOT / name
        if common.sha256(p) != h:
            bad[name] = 'changed'
    if bad:
        raise SystemExit(f'Frozen code changed: {bad}; stopping')
    return frozen


def log_event(name):
    path = OUT / 'events.json'
    events = common.read_json(path) if path.exists() else {}
    if name not in events:
        events[name] = time.strftime('%Y-%m-%d %H:%M:%S')
        common.save_json(path, events)
    return events[name]


# ---------------------------------------------------------------- cache rows (replica of lgbm_v2_cache.build, no year guard)
def build_cache(year, out, threads=8, memory='6GB'):
    """Same calls, same order and same files as lgbm_v2_cache.build; only the 2017-2021 guard and the output path differ."""
    from run_lgbm_v2_pilot import fit_elastic, predict_elastic
    feature_set = 'full'
    if (out / 'manifest.json').exists():
        raise RuntimeError(f'{out} already has a manifest')
    out.mkdir(parents=True, exist_ok=True)
    timing, started = {}, time.perf_counter()
    names = common.feature_names(year, feature_set)
    con = common.open_connection(out, threads, memory)
    t = time.perf_counter()
    common.create_views(con, year)
    pool_rows = common.create_pool(con, year)
    common.create_samples(con, year, cache.ROWS, cache.EARLY_ROWS)
    checks = dict(pool_rows=pool_rows, boundaries=common.boundary_checks(con, year))
    full_keys = con.execute('SELECT ts_code, signal_date FROM full_sample '
                            'ORDER BY priority, ts_code, signal_date').fetch_df()
    cache._save_keys(out / 'full_sample_keys.parquet', full_keys)
    checks['labels'] = common.label_check(con, 'full_sample', cache.LABEL_ROWS)
    checks['subset'] = common.subset_check(full_keys, year)
    common.save_json(out / 'checks.json', checks)
    timing['pool_sampling_checks'] = round(time.perf_counter() - t, 1)
    print(f'[{year}] pool={pool_rows:,} full={len(full_keys):,} '
          f'fit={checks["boundaries"]["inner_fit"]["rows"]:,} early={checks["boundaries"]["early_stop"]["rows"]:,} '
          f'checks passed ({timing["pool_sampling_checks"]}s)', flush=True)
    hashes = dict(full_sample_keys=cache.key_hash(full_keys))
    shapes = {}
    for prefix, table in (('fit', 'inner_fit'), ('es', 'early_stop')):
        t = time.perf_counter()
        x, y, keys = common.fetch_matrix(con, table, names)
        np.save(out / f'{prefix}_x.npy', x)
        np.save(out / f'{prefix}_y.npy', y)
        cache._save_keys(out / f'{prefix}_keys.parquet', keys)
        hashes[f'{prefix}_keys_and_y'] = cache.key_hash(keys, y)
        shapes[f'{prefix}_x'], shapes[f'{prefix}_y'] = list(x.shape), list(y.shape)
        timing[f'fetch_{table}'] = round(time.perf_counter() - t, 1)
        print(f'[{year}] {table} {x.shape} {timing[f"fetch_{table}"]}s', flush=True)
        del x, y, keys
    t = time.perf_counter()
    xs, frames = [], []
    for month in common.outer_months(con, year):
        x_out, frame = common.fetch_outer_month(con, year, month, names)
        xs.append(x_out)
        frames.append(frame)
    con.close()
    outer_x = np.concatenate(xs)
    del xs
    meta = pd.concat(frames, ignore_index=True)
    if meta.duplicated(KEY).any() or not meta['signal_date'].is_monotonic_increasing:
        raise RuntimeError('Outer rows must be unique and sorted by date')
    if (pd.to_datetime(meta['signal_date']).dt.year != year).any():
        raise RuntimeError('Outer cache contains rows outside the outer year')
    np.save(out / 'outer_x.npy', outer_x)
    meta['signal_date'] = pd.to_datetime(meta['signal_date']).astype('datetime64[ns]')
    meta.to_parquet(out / 'outer_meta.parquet', index=False)
    hashes['outer_keys_and_y'] = cache.key_hash(meta, meta['y'].to_numpy())
    shapes['outer_x'] = list(outer_x.shape)
    timing['fetch_outer'] = round(time.perf_counter() - t, 1)
    print(f'[{year}] outer {outer_x.shape} {timing["fetch_outer"]}s', flush=True)
    del outer_x
    t = time.perf_counter()
    elastic_dir = out / 'elastic'
    elastic_dir.mkdir(exist_ok=True)
    x_fit, y_fit = np.load(out / 'fit_x.npy'), np.load(out / 'fit_y.npy')
    model = fit_elastic(x_fit, y_fit, names)
    del x_fit
    outer_x = np.load(out / 'outer_x.npy', mmap_mode='r')
    pred = np.concatenate([predict_elastic(model, np.asarray(outer_x[s:s + 200_000]))
                           for s in range(0, len(outer_x), 200_000)])
    np.save(elastic_dir / 'outer_pred.npy', pred)
    frame = meta.assign(elastic=pred)
    metrics, daily = common.evaluate_outer(frame, 'elastic')
    metrics['timing_split'] = common.timing_selection_split(frame, 'elastic')
    daily.to_parquet(elastic_dir / 'daily_metrics.parquet', index=False)
    stats = model['stats']
    common.save_json(elastic_dir / 'model.json', dict(
        coefficient=np.asarray(model['coefficient']).tolist(), intercept=model['intercept'],
        alpha=model['alpha'], nonzero=model['nonzero'], iterations=model['iterations'],
        active_count=model['active_count'], active_names=stats['active_names'], preprocessing=stats))
    common.save_json(elastic_dir / 'metrics.json', metrics)
    timing['elastic'] = round(time.perf_counter() - t, 1)
    print(f'[{year}] elastic RankIC={metrics["mean_rank_ic"]:.5f} {timing["elastic"]}s', flush=True)
    timing['total'] = round(time.perf_counter() - started, 1)
    # same source list as lgbm_v2_cache.build (that file is the definition; this replica only drives it)
    sources = [Path(common.__file__), Path(cache.__file__), _source_path('v2/experiments/run_lgbm_v2_pilot.py'),
               _source_path('v2/models/research_controls.py'), _source_path('tune_elastic_net.py')]
    manifest = dict(params=cache.generation_params(year, feature_set), feature_names=names,
                    rows=dict(training_pool=pool_rows, full_sample=len(full_keys),
                              inner_fit=shapes['fit_y'][0], early_stop=shapes['es_y'][0],
                              outer=shapes['outer_x'][0], outer_scored=int(np.isfinite(meta['y']).sum())),
                    shapes=shapes, hashes=hashes,
                    file_bytes={p.name: p.stat().st_size for p in sorted(out.glob('*.npy'))},
                    source_sha256_at_build={p.name: common.sha256(p) for p in sources},
                    elastic_outer_mean_rank_ic=metrics['mean_rank_ic'], timing_seconds=timing,
                    built_by='run_lgbm_v2_confirm.build_cache (replica of lgbm_v2_cache.build)',
                    test_used=False)
    common.save_json(out / 'manifest.json', manifest)
    print(f'[{year}] cache complete in {timing["total"]}s -> {out}', flush=True)
    return manifest


def verify_replica():
    """Rebuild 2021 with the replica into a scratch folder and require identical hashes, shapes, sizes and Elastic predictions."""
    path = OUT / 'replica_check.json'
    if path.exists() and common.read_json(path).get('passed'):
        return
    tmp = ROOT / '_confirm_replica_2021'
    if tmp.exists():
        shutil.rmtree(tmp)
    ref_dir = cache.cache_dir(2021, 'full')
    ref = common.read_json(ref_dir / 'manifest.json')
    new = build_cache(2021, tmp)
    result = dict(params=new['params'] == ref['params'], feature_names=new['feature_names'] == ref['feature_names'],
                  rows=new['rows'] == ref['rows'], shapes=new['shapes'] == ref['shapes'],
                  hashes=new['hashes'] == ref['hashes'], file_bytes=new['file_bytes'] == ref['file_bytes'],
                  elastic_pred_equal=bool(np.array_equal(np.load(tmp / 'elastic' / 'outer_pred.npy'),
                                                         np.load(ref_dir / 'elastic' / 'outer_pred.npy'))))
    for name in ('fit_x', 'es_x'):
        a, b = np.load(tmp / f'{name}.npy', mmap_mode='r'), np.load(ref_dir / f'{name}.npy', mmap_mode='r')
        result[f'{name}_equal'] = bool(all(np.array_equal(a[s:s + 200_000], b[s:s + 200_000], equal_nan=True)
                                           for s in range(0, len(a), 200_000)))
        del a, b
    import gc
    gc.collect()
    result['passed'] = all(result.values())
    common.save_json(path, result)
    print(f'[replica check 2021] {result}', flush=True)
    if not result['passed']:
        raise SystemExit('Replica cache builder does not reproduce 2021; stopping')
    shutil.rmtree(tmp)


# ---------------------------------------------------------------- per-year steps (each in its own process)
def step_cache(year):
    out = cache.cache_dir(year, 'full')
    if not (out / 'manifest.json').exists():
        if out.exists():
            shutil.rmtree(out)   # incomplete build; no manifest means nothing was finished
        log_event(f'first_read_{year}')
        build_cache(year, out)
    cache.load(year, 'full')
    print(f'[{year}] cache reload and manifest verification passed', flush=True)


def step_labels(year):
    info = s6.build_labels(year)
    m = inp.elastic_cell(year, 'full_none_raw')   # selection-flow Elastic(D) used by fit_seed's comparison
    print(f'[{year}] ratio D {info["ratio_D"]:.4f}; max exit used {info["max_exit_date_used"]}; '
          f'selection-flow Elastic(D) eligible {m["eligible"]["mean_rank_ic"]:.5f}', flush=True)


def average(year):
    d = OUT / 'average' / f'{year}_{CONFIG}'
    frames = [pd.read_parquet(rv.review_dir(year, CONFIG, s) / 'outer_predictions.parquet') for s in fin.SEEDS]
    for f in frames[1:]:
        assert (f['ts_code'].to_numpy() == frames[0]['ts_code'].to_numpy()).all()
    outer = cache.load(year, 'full')['outer']
    assert (outer['ts_code'].to_numpy() == frames[0]['ts_code'].to_numpy()).all()
    outer = outer.assign(lgbm=np.mean([f['lgbm'].to_numpy() for f in frames], axis=0))
    e, ed = eligible_metrics(outer, 'lgbm')
    m, _ = common.evaluate_outer(outer, 'lgbm')
    d.mkdir(parents=True, exist_ok=True)
    outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(d / 'outer_predictions.parquet', index=False)
    ed.to_parquet(d / 'eligible_daily_metrics.parquet', index=False)
    common.save_json(d / 'metrics.json', dict(year=year, eligible=e, all_sample=m, seeds=fin.SEEDS))
    print(f'[{year} FINAL avg3] eligible {e["mean_rank_ic"]:.5f} all {m["mean_rank_ic"]:.5f}', flush=True)


def run_sub(args, label):
    print(f'[start] {label} at {time.strftime("%m-%d %H:%M:%S")}', flush=True)
    code = subprocess.call([sys.executable, '-u', __file__] + args)
    if code != 0:
        raise SystemExit(f'{label} failed with exit code {code}; stopping')


def run_fits():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'replica_check.json').exists() or not common.read_json(OUT / 'replica_check.json').get('passed'):
        run_sub(['--verify-replica'], 'replica check 2021')
    for y in CONFIRM_YEARS:
        if not (cache.cache_dir(y, 'full') / 'manifest.json').exists():
            run_sub(['--cache', str(y)], f'cache {y}')
        if not (s6.LABELS / f'{y}.json').exists() or not (inp.elastic_dir(y, 'full_none_raw') / 'metrics.json').exists():
            run_sub(['--labels', str(y)], f'labels and selection-flow Elastic(D) {y}')
        for s in fin.SEEDS:
            if not fin.completed(fin.seed_dir(y, CONFIG, s)):
                run_sub(['--select', str(y), str(s)], f'selection {y} {CONFIG} s{s}')
        if not (rv.full_dir(y) / 'manifest.json').exists():
            run_sub(['--full', str(y)], f'complete sample {y}')
        if not (rv.elastic_dir(y) / 'metrics.json').exists():
            run_sub(['--elastic-full', str(y)], f'Elastic(D) complete sample {y}')
        for s in fin.SEEDS:
            if not rv.completed(rv.review_dir(y, CONFIG, s)):
                run_sub(['--review', str(y), str(s)], f'review {y} {CONFIG} s{s}')
        average(y)


# ---------------------------------------------------------------- step 2: four prediction sets
def outer_meta(year):
    m = pd.read_parquet(cache.cache_dir(year, 'full') / 'outer_meta.parquet')
    m['signal_date'] = pd.to_datetime(m['signal_date']).dt.normalize()
    return m


def daily_rank(frame, column):
    return frame.groupby('signal_date')[column].rank(method='average') / frame.groupby('signal_date')[column].transform('count')


def build_predictions():
    out = OUT / 'predictions'
    out.mkdir(parents=True, exist_ok=True)
    coverage = []
    for y in ALL_YEARS:
        meta = outer_meta(y)
        g1_dir = (rv.OUT / 'average' / f'{y}_{CONFIG}') if y <= 2021 else (OUT / 'average' / f'{y}_{CONFIG}')
        g1 = pd.read_parquet(g1_dir / 'outer_predictions.parquet')
        g1['signal_date'] = pd.to_datetime(g1['signal_date']).dt.normalize()
        assert (g1['ts_code'].to_numpy() == meta['ts_code'].to_numpy()).all() and \
            (g1['signal_date'].to_numpy() == meta['signal_date'].to_numpy()).all(), f'{y}: G1 keys differ from outer'
        g2 = np.load(rv.elastic_dir(y) / 'outer_pred.npy')
        assert len(g2) == len(meta)
        cand_path = CANDIDATE / str(y) / 'predictions.parquet'
        if not cand_path.exists():
            raise SystemExit(f'{y}: linear candidate predictions missing ({cand_path}); stopping')
        cand = pd.read_parquet(cand_path, columns=KEY + CANDIDATE_COLUMNS + ['volatility_20d'])
        cand['signal_date'] = pd.to_datetime(cand['signal_date']).dt.normalize()
        cand['G3_linear_candidate'] = cand[CANDIDATE_COLUMNS].mean(axis=1)
        frame = meta.assign(G1_lgbm_final=g1['lgbm'].to_numpy(), G2_elastic_D=g2)
        frame = frame.merge(cand[KEY + ['G3_linear_candidate', 'volatility_20d']], on=KEY, how='outer',
                            validate='one_to_one', indicator=True)
        cov = dict(year=y, outer_rows=len(meta), candidate_rows=len(cand),
                   only_outer=int((frame['_merge'] == 'left_only').sum()),
                   only_candidate=int((frame['_merge'] == 'right_only').sum()))
        both = frame['_merge'] == 'both'
        cov['volatility_equal'] = bool(np.allclose(frame.loc[both, 'vol20'], frame.loc[both, 'volatility_20d'], equal_nan=True))
        frame = frame.drop(columns='_merge')
        for g in GROUPS[:3]:
            cov[f'{g}_finite'] = int(np.isfinite(frame[g]).sum())
        # 50/50 fusion: daily cross-sectional ranks on stocks where both G1 and G3 have predictions
        ok = np.isfinite(frame['G1_lgbm_final']) & np.isfinite(frame['G3_linear_candidate'])
        sub = frame.loc[ok, KEY + ['G1_lgbm_final', 'G3_linear_candidate']]
        fused = 0.5 * daily_rank(sub, 'G1_lgbm_final') + 0.5 * daily_rank(sub, 'G3_linear_candidate')
        frame['G4_fusion_50_50'] = np.nan
        frame.loc[ok, 'G4_fusion_50_50'] = fused.to_numpy()
        cov['G4_fusion_50_50_finite'] = int(ok.sum())
        days = frame.groupby('signal_date')[GROUPS].count()
        cov['days'] = len(days)
        cov['days_with_unequal_coverage'] = int((days.nunique(axis=1) > 1).sum())
        cov['same_stock_set_every_day'] = bool(cov['days_with_unequal_coverage'] == 0 and cov['only_outer'] == 0
                                               and cov['only_candidate'] == 0)
        coverage.append(cov)
        frame = frame.sort_values(['signal_date', 'ts_code']).reset_index(drop=True)
        frame[KEY + ['y', 'vol20', 'eligible', 'volatility_20d'] + GROUPS].to_parquet(out / f'{y}.parquet', index=False)
        print(f'[predictions {y}] {cov}', flush=True)
    pd.DataFrame(coverage).to_csv(OUT / 'coverage.csv', index=False)
    common.save_json(OUT / 'predictions_manifest.json', dict(
        groups=GROUPS, coverage=coverage,
        sources=dict(G1='2017-2021 lgbm_final/review/average/{y}_F3 (stage-8 three-seed average); 2022-2023 lgbm_confirm/average/{y}_F3',
                     G2='lgbm_final/review/elastic/{y}/outer_pred.npy (complete 2M-row Elastic(D), 2022-2023 fitted in stage 9)',
                     G3=f'elastic_compact_gate_seeds/{{y}}/predictions.parquet, columns {CANDIDATE_COLUMNS}, equal-weight mean of the raw predictions',
                     G4='0.5 x daily rank(G1)/n + 0.5 x daily rank(G3)/n on stocks with both predictions'),
        candidate_files_sha256={str(y): common.sha256(CANDIDATE / str(y) / 'predictions.parquet') for y in ALL_YEARS}))


def load_predictions():
    frames = [pd.read_parquet(OUT / 'predictions' / f'{y}.parquet') for y in ALL_YEARS]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- step 3a: prediction metrics
def evaluate():
    out = OUT / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    rows, pairs, daily = [], [], {}
    for y in ALL_YEARS:
        frame = pd.read_parquet(OUT / 'predictions' / f'{y}.parquet')
        for g in GROUPS:
            f = frame.loc[np.isfinite(frame[g])]
            e, ed = eligible_metrics(f, g)
            m, _ = common.evaluate_outer(f, g)
            daily[(y, g)] = ed
            ed.assign(year=y, group=g).to_parquet(out / f'eligible_daily_{y}_{g}.parquet', index=False)
            rows.append(dict(year=y, group=g, eligible_rank_ic=e['mean_rank_ic'], eligible_days=e['days'],
                             all_sample_rank_ic=m['mean_rank_ic'], top50_minus_pool=m['top50_minus_pool_5d_simple_return'],
                             top50_overlap=m['consecutive_day_top50_overlap'], rows=len(f)))
            print(f'[eval {y} {g}] eligible {e["mean_rank_ic"]:.5f} all {m["mean_rank_ic"]:.5f}', flush=True)
        for a, b in (('G1_lgbm_final', 'G2_elastic_D'), ('G4_fusion_50_50', 'G3_linear_candidate')):
            r = paired_t(daily[(y, a)], daily[(y, b)])
            pairs.append(dict(year=y, pair=f'{a} - {b}', days=r['days'], mean_daily_diff=r['mean_rank_ic_difference'],
                              nw_t=r['newey_west_t_lag5']))
    for label, years in (('2017-2021', ALL_YEARS[:5]), ('2022-2023', CONFIRM_YEARS)):
        for a, b in (('G1_lgbm_final', 'G2_elastic_D'), ('G4_fusion_50_50', 'G3_linear_candidate')):
            la = pd.concat([daily[(y, a)] for y in years]); lb = pd.concat([daily[(y, b)] for y in years])
            r = paired_t(la, lb)
            pairs.append(dict(year=label, pair=f'{a} - {b}', days=r['days'], mean_daily_diff=r['mean_rank_ic_difference'],
                              nw_t=r['newey_west_t_lag5']))
    pd.DataFrame(rows).to_csv(out / 'yearly_metrics.csv', index=False)
    pd.DataFrame(pairs).to_csv(out / 'paired_tests.csv', index=False)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


# ---------------------------------------------------------------- step 3b: backtests (existing V2 linear-candidate code, unchanged)
def backtest_frame(frame, column):
    """The signal frame the existing backtest expects: KEY, volatility_20d, prediction (rows with a prediction)."""
    f = frame.loc[np.isfinite(frame[column]), KEY + ['volatility_20d', column]].rename(columns={column: 'prediction'})
    f['signal_date'] = pd.to_datetime(f['signal_date']).astype('datetime64[us]')
    return f.reset_index(drop=True)


def run_backtests():
    from run_elastic_19_control import prepare, buffer_panel, annotate_orders, simulate_released_cash, audit_ledger
    out = OUT / 'backtests'
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / 'progress.json'
    results = common.read_json(progress_path) if progress_path.exists() else {}
    frame = load_predictions()
    assert frame['signal_date'].dt.year.min() == 2017 and frame['signal_date'].dt.year.max() == 2023
    # same calls as run_elastic_compact_gate_seeds.backtest: one market template, then per signal set
    template, _, panel, missing, evidence = prepare(backtest_frame(frame, GROUPS[0]))
    del evidence
    template = template.drop(columns='prediction')
    template['entry_tradable'] = template.entry_tradable.fillna(False) & template.strict_buy
    jobs = [('wrapper_check_s20261025_features_only', 0.5)] + [(g, i) for g in GROUPS for i in IMPACTS]
    for group, impact in jobs:
        key = f'{group}|{impact}'
        if key in results:
            continue
        started = time.perf_counter()
        if group.startswith('wrapper_check'):
            cand = pd.concat([pd.read_parquet(CANDIDATE / str(y) / 'predictions.parquet',
                                              columns=KEY + ['s20261025_features_only']) for y in ALL_YEARS])
            cand['signal_date'] = pd.to_datetime(cand['signal_date']).astype('datetime64[us]')
            signals = cand.rename(columns={'s20261025_features_only': 'prediction'})
        else:
            signals = backtest_frame(frame, group)[KEY + ['prediction']]
        orders = template.merge(signals, on=KEY, validate='one_to_one')
        assert len(orders) == len(template), f'{group}: prediction missing for {len(template) - len(orders)} tradable rows'
        assert np.isfinite(orders['prediction']).all()
        selected, decision = buffer_panel(orders, panel, 100)
        metrics, nav, trades = simulate_released_cash(orders, selected, aum=1e8, cost_bps=20,
                                                      impact_coefficient=impact, round_lots=True)
        trades = annotate_orders(trades, decision, panel, orders)
        audit = audit_ledger(trades, nav, orders, panel, missing)
        assert audit['filled_entry_missing_limits'] == audit['sold_exit_missing_limits'] == 0
        np.testing.assert_allclose(nav.nav, np.cumprod(1 + nav.daily_return))
        tag = f'{group}_{impact:g}'
        nav.to_parquet(out / f'{tag}_nav.parquet', index=False)
        trades.to_parquet(out / f'{tag}_orders.parquet', index=False)
        results[key] = dict(**metrics, audit=audit, seconds=round(time.perf_counter() - started, 1))
        if group.startswith('wrapper_check'):
            ref = common.read_json(CANDIDATE / 'summary.json')['results']['s20261025_features_only']['0.5']['net_return']
            results[key]['reference_net_return'] = ref
            results[key]['reproduces_reference'] = bool(abs(metrics['net_return'] - ref) < 1e-9)
            if not results[key]['reproduces_reference']:
                common.save_json(progress_path, results)
                raise SystemExit(f'Backtest wrapper does not reproduce the stored candidate result '
                                 f'({metrics["net_return"]} vs {ref}); stopping')
        common.save_json(progress_path, results)
        print(f'[backtest {group} impact {impact}] net {metrics["net_return"]:.4f} maxDD {metrics["max_drawdown"]:.4f} '
              f'({results[key]["seconds"]:.0f}s)', flush=True)


# ---------------------------------------------------------------- report tables
def backtest_tables():
    out = OUT / 'backtests'
    results = common.read_json(out / 'progress.json')
    yearly, periods = [], []
    for g in GROUPS:
        for impact in IMPACTS:
            r = results[f'{g}|{impact}']
            nav = pd.read_parquet(out / f'{g}_{impact:g}_nav.parquet')
            date_col = 'date' if 'date' in nav.columns else nav.columns[0]
            nav['year'] = pd.to_datetime(nav[date_col]).dt.year
            for b in r['audit']['annual_boundaries']:
                d = nav.loc[nav['year'] == b['year'], 'daily_return']
                yearly.append(dict(group=g, impact=impact, year=b['year'], net_return=b['net_return'],
                                   within_year_drawdown=b['within_year_drawdown'],
                                   sharpe=float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else np.nan))
            for label, years in (('2017-2023', ALL_YEARS), ('2017-2021', ALL_YEARS[:5]), ('2022-2023', CONFIRM_YEARS)):
                sub = nav.loc[nav['year'].isin(years)]
                d = sub['daily_return']
                growth = float(np.prod(1 + d))
                peak = np.maximum.accumulate(np.cumprod(1 + d.to_numpy()))
                periods.append(dict(group=g, impact=impact, period=label, net_return=growth - 1,
                                    annualized=growth ** (252 / len(d)) - 1,
                                    max_drawdown=float(np.min(np.cumprod(1 + d.to_numpy()) / peak - 1)),
                                    sharpe=float(d.mean() / d.std() * np.sqrt(252)),
                                    annualized_two_way_turnover=r['annualized_two_way_turnover'] if label == '2017-2023' else np.nan,
                                    fill_failure_rate=1 - r['filled_orders'] / r['orders'] if label == '2017-2023' else np.nan,
                                    total_cost_amount=r['total_cost_amount'] if label == '2017-2023' else np.nan))
    pd.DataFrame(yearly).to_csv(OUT / 'backtest_yearly.csv', index=False)
    pd.DataFrame(periods).to_csv(OUT / 'backtest_periods.csv', index=False)
    print(pd.DataFrame(periods).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fits', action='store_true')
    parser.add_argument('--verify-replica', action='store_true')
    parser.add_argument('--cache', type=int)
    parser.add_argument('--labels', type=int)
    parser.add_argument('--select', nargs=2, type=int)
    parser.add_argument('--full', type=int)
    parser.add_argument('--elastic-full', type=int)
    parser.add_argument('--review', nargs=2, type=int)
    parser.add_argument('--predictions', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--backtests', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    check_frozen()
    for y in [args.cache, args.labels, args.full, args.elastic_full] + [v[0] for v in (args.select, args.review) if v]:
        if y is not None and y not in CONFIRM_YEARS:
            raise SystemExit('stage 9 runs only 2022 and 2023')
    if args.fits:
        started = time.perf_counter()
        run_fits()
        print(f'[done] confirmation fits in {(time.perf_counter() - started) / 3600:.2f} h', flush=True)
    elif args.verify_replica:
        verify_replica()
    elif args.cache:
        step_cache(args.cache)
    elif args.labels:
        step_labels(args.labels)
    elif args.select:
        y, s = args.select
        fin.fit_seed(y, CONFIG, s, fin.seed_dir(y, CONFIG, s))
    elif args.full:
        rv.build_full(args.full)
    elif args.elastic_full:
        rv.fit_elastic_full(args.elastic_full)
    elif args.review:
        rv.fit_review(args.review[0], CONFIG, args.review[1])
    elif args.predictions:
        build_predictions()
    elif args.evaluate:
        evaluate()
    elif args.backtests:
        run_backtests()
    elif args.report:
        backtest_tables()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
