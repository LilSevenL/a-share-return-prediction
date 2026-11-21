"""V2 LightGBM stages 7 (seeds, selection flow) and 8 (review flow), plan section 17.

Shortlist (all: 448 raw factors, no gate, D label, lr 0.02, cap 6000, patience 200, R1, ff 0.3, bagging 1.0, max_bin 255):
  F1 deep 0.1%,   lambda1 = sig2(50 x ratio_D), lambda2 10000   (seed 42 = stage-3 baseline = P4)
  F2 deep 0.1%,   lambda1 = sig2(50 x ratio_D), lambda2 1000    (seed 42 = stage 6a D x1)
  F3 medium 0.1%, lambda1 = sig2(10 x ratio_D), lambda2 1000    (seed 42 = stage-3 mediumD)

  python v2/experiments/run_lgbm_v2_final.py --seeds         reuse check -> gate refits -> 30 seed fits -> stage-7 summary
  python v2/experiments/run_lgbm_v2_final.py --one YEAR CONFIG SEED [--check]
  python v2/experiments/run_lgbm_v2_final.py --summarize-seeds
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
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from lgbm_v2_common import DATA, Timer
from run_lgbm_v2_stage1 import LEAF_FRACTIONS, STRUCTURES, lgbm_params
from run_lgbm_v2_stage2 import PENALTY, YEARS, eligible_metrics, paired_t
import run_lgbm_v2_input as inp

OUT = DATA / 'v2' / 'lgbm_final'
SEEDS = [42, 7, 2026]
NEW_SEEDS = [7, 2026]
CAP, PATIENCE, EVERY, TRUNCATION = 6000, 200, 10, 0.95
MIN_GAIN, MIN_T, PLATEAU_DIFF = 0.001, 2.0, 0.001
CONFIGS = {
    'F1': dict(structure='deep', l1_factor=50, lambda_l2=10000, seed42=DATA / 'v2' / 'lgbm_stage3', seed42_name='deepD_x1_l2-10000'),
    'F2': dict(structure='deep', l1_factor=50, lambda_l2=1000, seed42=DATA / 'v2' / 'lgbm_stage6a', seed42_name='D_x1'),
    'F3': dict(structure='medium', l1_factor=10, lambda_l2=1000, seed42=DATA / 'v2' / 'lgbm_stage3', seed42_name='mediumD'),
}
GATE_REFITS = [(2017, 'F2'), (2017, 'F3')]   # F1 2017 seed 42 was already reproduced exactly by the input stage gate
DERIVED = ('seed', 'data_random_seed', 'feature_fraction_seed', 'bagging_seed', 'extra_seed', 'drop_seed', 'objective_seed')


def seed_dir(year, config, seed):
    if seed == 42:
        c = CONFIGS[config]
        return c['seed42'] / f'{year}_{c["seed42_name"]}'
    return OUT / 'seeds' / f'{year}_{config}_s{seed}'


def completed(output):
    path = output / 'summary.json'
    return path.exists() and common.read_json(path).get('status') == 'completed'


def params_for(config, ratio, n_fit, seed, threads=8):
    c = CONFIGS[config]
    params = lgbm_params(STRUCTURES[c['structure']], LEAF_FRACTIONS['0.1pct'], n_fit, threads)
    params.update(lambda_l1=float(inp.sig2(c['l1_factor'] * ratio)), lambda_l2=float(c['lambda_l2']), seed=int(seed))
    return params


def derived_seeds(model_text):
    out = {}
    for name in DERIVED:
        m = re.search(rf'^\[{name}: (-?\d+)\]$', model_text, re.M)
        out[name] = int(m.group(1)) if m else None
    return out


def ratio_d(year):
    import run_lgbm_v2_stage6a as s6
    return s6.build_labels(year)['ratio_D']


# ---------------------------------------------------------------- seed-42 reuse check
def check_reuse():
    rows = []
    for config in CONFIGS:
        for year in YEARS:
            d = seed_dir(year, config, 42)
            s = common.read_json(d / 'summary.json')
            assert s['status'] == 'completed', f'{d}: not completed'
            expected = params_for(config, ratio_d(year), s['rows']['inner_fit'], 42)
            got = s['lightgbm']['params']
            if got != expected:
                diff = {k: (got.get(k), expected.get(k)) for k in set(got) | set(expected) if got.get(k) != expected.get(k)}
                raise SystemExit(f'{d}: seed-42 configuration differs from {config}: {diff}; stopping')
            assert s['extra_params'] == {}, f'{d}: extra params {s["extra_params"]}'
            cfg = common.read_json(d / 'config.json')
            assert cfg['cap'] == CAP and cfg['patience'] == PATIENCE and cfg['every'] == EVERY, f'{d}: cap/patience differ'
            assert cfg['cache_params']['feature_set'] == 'full' and cfg['cache_params']['training_gate'] is False
            assert cfg['training_label']['scheme'] == 'D'
            base = common.read_json(cache.cache_dir(year, 'full') / 'manifest.json')['hashes']
            assert cfg['cache_hashes'] == base, f'{d}: cache hashes differ'
            with gzip.open(d / 'lightgbm_model_all_rounds.txt.gz', 'rt', encoding='utf-8') as h:
                tail = h.read()[-20000:]
            rows.append(dict(config=config, year=year, dir=str(d), lambda_l1=got['lambda_l1'], lambda_l2=got['lambda_l2'],
                             min_data_in_leaf=got['min_data_in_leaf'], **derived_seeds(tail)))
    common.save_json(OUT / 'seed42_reuse_check.json', dict(passed=True, rows=rows,
                                                           rule='all LightGBM params equal the shortlist definition except seed; '
                                                                'cap/patience/every, cache hashes, feature set, no gate, D label'))
    print(f'[reuse] seed-42 directories match F1-F3 for all folds ({len(rows)} checked)', flush=True)


# ---------------------------------------------------------------- one selection-flow fit
def fit_seed(year, config, seed, output, threads=8):
    import lightgbm as lgb
    output.mkdir(parents=True, exist_ok=True)
    timer, started = Timer(), time.perf_counter()
    with timer('load_cache'):
        data = inp.load_cell(year, 'full', 'none', 'raw')
    names, ratio = data['names'], data['ratio_D']
    x_fit, y_fit, fit_dates, label = data['fit_x'], data['fit_y'], data['fit_dates'], data['fit_d']
    x_es, y_es = data['es_x'], data['es_y']
    params = params_for(config, ratio, len(y_fit), seed, threads)
    sources = [Path(__file__), Path(inp.__file__), Path(common.__file__), Path(cache.__file__),
               _source_path('v2/experiments/run_lgbm_v2_stage1.py'), _source_path('v2/experiments/run_lgbm_v2_stage2.py'), _source_path('v2/experiments/run_lgbm_v2_stage6a.py')]
    config_json = dict(stage=7, year=year, config=config, seed=seed, params=params, ratio_D=ratio, cap=CAP,
                       patience=PATIENCE, every=EVERY, cache_hashes=data['base_manifest']['hashes'], test_used=False,
                       sources={s.name: common.sha256(s) for s in sources})
    config_path = output / 'config.json'
    if config_path.exists() and common.read_json(config_path) != json.loads(json.dumps(config_json, default=str)):
        raise RuntimeError(f'{output}: existing output belongs to a different configuration or code version')
    common.save_json(config_path, config_json)
    monitor = common.RankICEarlyStopping(y_es, data['es_dates'], every=EVERY, patience=PATIENCE)
    with timer('dataset_construct'):
        train_set = lgb.Dataset(x_fit, label=label, feature_name=names, free_raw_data=False, params=params)
        valid_set = lgb.Dataset(x_es, label=y_es, reference=train_set, params=params)
        train_set.construct()
        valid_set.construct()
    trees_per_round = []

    def record_trees(env):
        trees_per_round.append(env.model.num_trees())
    record_trees.order = 20
    with timer('train'):
        monitor.started = time.perf_counter()
        booster = lgb.train(params, train_set, num_boost_round=CAP, valid_sets=[valid_set], valid_names=['early_stop'],
                            feval=monitor.feval, callbacks=monitor.callbacks() + [record_trees], keep_training_booster=True)
    best, trained = monitor.best_iteration, monitor.iteration
    if best <= 0:
        raise RuntimeError('No valid early-stopping score was recorded')
    trees_built = booster.num_trees()
    monotone = all(b >= a for a, b in zip(trees_per_round, trees_per_round[1:]))
    if len(trees_per_round) != trained or trees_per_round[-1] != trees_built or not monotone:
        raise RuntimeError('Per-round tree counts are inconsistent with the trained booster')
    skipped, best_trees = trained - trees_built, trees_per_round[best - 1]
    print(f'[{year} {config} s{seed}] lambda1 {params["lambda_l1"]:g} lambda2 {params["lambda_l2"]:g} rounds={trained} '
          f'best={best} trees@best={best_trees} skipped={skipped} es RankIC={monitor.best_score:.5f} '
          f'train={timer.records["train"]:.0f}s', flush=True)
    with timer('save_full_model'):
        text = booster.model_to_string(num_iteration=-1)
        seeds_used = derived_seeds(text)
        assert seeds_used['seed'] == seed, f'model text records seed {seeds_used["seed"]}'
        with gzip.open(output / 'lightgbm_model_all_rounds.txt.gz', 'wt', encoding='utf-8', compresslevel=6) as h:
            h.write(text)
        del text
        np.save(output / 'trees_per_round.npy', np.asarray(trees_per_round, dtype=np.int32))
        common.save_json(output / 'best_iteration.json', dict(best_iteration=best, best_model_trees=best_trees,
                                                              rounds_trained=trained, trees_built=trees_built,
                                                              skipped_rounds=skipped))
    with timer('diagnostics'):
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(y_fit), min(300_000, len(y_fit)), replace=False))
        train_ic = common.DailyRankIC(y_fit[idx], fit_dates[idx])(booster.predict(x_fit[idx], num_iteration=best_trees))
        trees = common.tree_diagnostics(booster, best_trees)
        diagnostics = dict(trees=trees, inner_fit_rank_ic=train_ic, early_stop_rank_ic=monitor.best_score,
                           fit_minus_early_stop_rank_ic=train_ic - monitor.best_score)
    del train_set, valid_set
    gc.collect()
    with timer('outer_predict'):
        pred = np.concatenate([booster.predict(a, num_iteration=best_trees) for a in inp.outer_chunks(data['outer_x'], None)])
    if not np.isfinite(pred).all():
        raise RuntimeError('Non-finite outer predictions')
    outer = data['outer'].assign(lgbm=pred)
    with timer('outer_metrics'):
        metrics, daily = common.evaluate_outer(outer, 'lgbm')
        eligible, eligible_daily = eligible_metrics(outer, 'lgbm')
        e_dir = inp.elastic_dir(year, 'full_none_raw')
        e_daily = pd.read_parquet(e_dir / 'eligible_daily_metrics.parquet')
        e_daily['date'] = pd.to_datetime(e_daily['date']).dt.normalize()
        comparison = dict(eligible=paired_t(eligible_daily, e_daily),
                          elastic_eligible_mean_rank_ic=common.read_json(e_dir / 'metrics.json')['eligible']['mean_rank_ic'])
    outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(output / 'outer_predictions.parquet', index=False)
    pd.DataFrame(monitor.curve, columns=['iteration', 'early_stop_rank_ic', 'seconds']).to_parquet(
        output / 'early_stop_curve.parquet', index=False)
    daily.to_parquet(output / 'daily_metrics.parquet', index=False)
    eligible_daily.to_parquet(output / 'eligible_daily_metrics.parquet', index=False)
    summary = dict(
        status='completed', stage=7, year=year, config=config, seed=seed, derived_seeds=seeds_used,
        rows=dict(inner_fit=len(y_fit), early_stop=len(y_es), outer=len(outer), outer_eligible_scored=eligible['rows']),
        lightgbm=dict(params=params, cap=CAP, rounds_trained=trained, trees_built=trees_built, skipped_rounds=skipped,
                      best_iteration=best, best_model_trees=best_trees, truncated=best >= TRUNCATION * CAP,
                      early_stop_best_rank_ic=monitor.best_score),
        eligible_metrics=eligible, outer_metrics=metrics, lgbm_vs_elastic=comparison, diagnostics=diagnostics,
        timing_seconds=dict(**timer.records, total=round(time.perf_counter() - started, 1)),
        peak_memory_gb=common.peak_rss_gb(),
        machine=dict(platform=platform.platform(), threads=threads, lightgbm=lgb.__version__), test_used=False)
    common.save_json(output / 'summary.json', summary)
    print(f'[{year} {config} s{seed}] eligible RankIC={eligible["mean_rank_ic"]:.5f} (Elastic(D) '
          f'{comparison["elastic_eligible_mean_rank_ic"]:.5f}, t={comparison["eligible"]["newey_west_t_lag5"]:.2f}) '
          f'all={metrics["mean_rank_ic"]:.5f} leaves={trees["mean_leaves"]:.1f} total={summary["timing_seconds"]["total"]:.0f}s',
          flush=True)
    return summary


# ---------------------------------------------------------------- gate refits: new code reproduces seed-42 F2 / F3
def gate_refits():
    path = OUT / 'gate_check.json'
    if path.exists() and common.read_json(path).get('passed'):
        return
    results = []
    for year, config in GATE_REFITS:
        out = OUT / 'seeds' / f'_check_{year}_{config}_s42'
        if not completed(out):
            run_sub(['--one', str(year), config, '42', '--check'], f'check {year} {config} s42')
        ref = seed_dir(year, config, 42)
        a, b = common.read_json(out / 'summary.json'), common.read_json(ref / 'summary.json')
        ca, cb = pd.read_parquet(out / 'early_stop_curve.parquet'), pd.read_parquet(ref / 'early_stop_curve.parquet')
        pa, pb = pd.read_parquet(out / 'outer_predictions.parquet'), pd.read_parquet(ref / 'outer_predictions.parquet')
        r = dict(year=year, config=config, params_equal=a['lightgbm']['params'] == b['lightgbm']['params'],
                 best_iteration=[a['lightgbm']['best_iteration'], b['lightgbm']['best_iteration']],
                 trees_built=[a['lightgbm']['trees_built'], b['lightgbm']['trees_built']],
                 curve_points=[len(ca), len(cb)],
                 curve_max_abs_diff=float(np.max(np.abs(ca['early_stop_rank_ic'].to_numpy() - cb['early_stop_rank_ic'].to_numpy())))
                 if len(ca) == len(cb) else None,
                 prediction_max_abs_diff=float(np.max(np.abs(pa['lgbm'].to_numpy() - pb['lgbm'].to_numpy()))),
                 keys_equal=bool((pa['ts_code'].to_numpy() == pb['ts_code'].to_numpy()).all()),
                 eligible=[a['eligible_metrics']['mean_rank_ic'], b['eligible_metrics']['mean_rank_ic']])
        r['passed'] = bool(r['params_equal'] and r['best_iteration'][0] == r['best_iteration'][1]
                           and r['trees_built'][0] == r['trees_built'][1] and len(ca) == len(cb)
                           and r['curve_max_abs_diff'] == 0.0 and r['prediction_max_abs_diff'] == 0.0 and r['keys_equal'])
        print(f'[gate check] {r}', flush=True)
        results.append(r)
        if not r['passed']:
            common.save_json(path, dict(passed=False, results=results))
            raise SystemExit(f'New fit code does not reproduce {config} {year} seed 42; stopping')
    common.save_json(path, dict(passed=True, results=results))


# ---------------------------------------------------------------- driver
def run_sub(args, label):
    print(f'[start] {label} at {time.strftime("%m-%d %H:%M:%S")}', flush=True)
    code = subprocess.call([sys.executable, '-u', __file__] + args)
    if code != 0:
        raise SystemExit(f'{label} failed with exit code {code}; stopping')


def run_seeds():
    OUT.mkdir(parents=True, exist_ok=True)
    check_reuse()
    gate_refits()
    for seed in NEW_SEEDS:
        for config in CONFIGS:
            for year in YEARS:
                if not completed(seed_dir(year, config, seed)):
                    run_sub(['--one', str(year), config, str(seed)], f'{year}_{config}_s{seed}')
    summarize_seeds()


# ---------------------------------------------------------------- stage-7 summary
def score_array(values):
    v = np.asarray(values, float)
    return float(v.mean() - PENALTY * v.std(ddof=0))


def eligible_daily_of(config, seed):
    parts = []
    for y in YEARS:
        d = pd.read_parquet(seed_dir(y, config, seed) / 'eligible_daily_metrics.parquet')
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        parts.append(d.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def nw_pair(a, b):
    joined = a.merge(b, on=['year', 'date'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(joined) != len(a) or len(joined) != len(b):
        raise RuntimeError('Incomplete daily pairing')
    diff = (joined['rank_ic_a'] - joined['rank_ic_b']).to_numpy()
    return float(diff.mean()), common.newey_west_tstat(diff, lag=5)


def average_predictions(year, config):
    """Three-seed raw prediction average for one fold; keys must be identical across seeds."""
    frames = [pd.read_parquet(seed_dir(year, config, s) / 'outer_predictions.parquet') for s in SEEDS]
    base = frames[0]
    for f in frames[1:]:
        assert (f['ts_code'].to_numpy() == base['ts_code'].to_numpy()).all()
        assert (pd.to_datetime(f['signal_date']).to_numpy() == pd.to_datetime(base['signal_date']).to_numpy()).all()
    outer = cache.load(year, 'full')['outer']
    assert (outer['ts_code'].to_numpy() == base['ts_code'].to_numpy()).all()
    return outer.assign(lgbm=np.mean([f['lgbm'].to_numpy() for f in frames], axis=0))


def summarize_seeds():
    rows, fold_rows, best_rows = [], [], []
    for config in CONFIGS:
        for seed in SEEDS:
            ss = [common.read_json(seed_dir(y, config, seed) / 'summary.json') for y in YEARS]
            elig = [s['eligible_metrics']['mean_rank_ic'] for s in ss]
            allm = [s['outer_metrics']['mean_rank_ic'] for s in ss]
            best = [s['lightgbm']['best_iteration'] for s in ss]
            rows.append(dict(config=config, seed=seed, **{f'eligible_{y}': v for y, v in zip(YEARS, elig)},
                             eligible_score=score_array(elig), all_score=score_array(allm),
                             negative_fold=bool(min(elig) < 0), truncated=any(b >= TRUNCATION * CAP for b in best),
                             mean_leaves=float(np.mean([s['diagnostics']['trees']['mean_leaves'] for s in ss])),
                             train_seconds=float(np.mean([s['timing_seconds']['train'] for s in ss]))))
            for y, s in zip(YEARS, ss):
                lg = s['lightgbm']
                best_rows.append(dict(config=config, seed=seed, year=y, best_iteration=lg['best_iteration'],
                                      best_model_trees=lg.get('best_model_trees', lg['best_iteration']),
                                      rounds_trained=lg['rounds_trained'], trees_built=lg['trees_built'],
                                      skipped_rounds=lg.get('skipped_rounds', lg['rounds_trained'] - lg['trees_built']),
                                      lambda_l1=lg['params']['lambda_l1'], early_stop_rank_ic=lg['early_stop_best_rank_ic'],
                                      eligible=s['eligible_metrics']['mean_rank_ic'],
                                      derived_seeds=json.dumps(s.get('derived_seeds')) if s.get('derived_seeds') else None))
        # three-seed average predictions
        avg_elig, avg_all, avg_daily = [], [], []
        for y in YEARS:
            outer = average_predictions(y, config)
            e, ed = eligible_metrics(outer, 'lgbm')
            m, _ = common.evaluate_outer(outer, 'lgbm')
            avg_elig.append(e['mean_rank_ic'])
            avg_all.append(m['mean_rank_ic'])
            ed['date'] = pd.to_datetime(ed['date']).dt.normalize()
            avg_daily.append(ed.assign(year=y))
            out = OUT / 'seed_average' / f'{y}_{config}'
            out.mkdir(parents=True, exist_ok=True)
            outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(out / 'outer_predictions.parquet', index=False)
            ed.to_parquet(out / 'eligible_daily_metrics.parquet', index=False)
        rows.append(dict(config=config, seed='avg3', **{f'eligible_{y}': v for y, v in zip(YEARS, avg_elig)},
                         eligible_score=score_array(avg_elig), all_score=score_array(avg_all),
                         negative_fold=bool(min(avg_elig) < 0), truncated=None))
    table = pd.DataFrame(rows)
    # cross-seed spread
    spread = []
    for config in CONFIGS:
        t = table[(table['config'] == config) & (table['seed'] != 'avg3')]
        spread.append(dict(config=config, **{f'std_{y}': float(t[f'eligible_{y}'].std(ddof=0)) for y in YEARS},
                           std_score=float(t['eligible_score'].std(ddof=0)),
                           range_score=float(t['eligible_score'].max() - t['eligible_score'].min()),
                           mean_single_seed_score=float(t['eligible_score'].mean()),
                           avg3_score=float(table[(table['config'] == config) & (table['seed'] == 'avg3')]['eligible_score'].iloc[0])))
    spread = pd.DataFrame(spread)
    # ranking and within-seed paired t
    ranking, pairs = [], []
    for seed in SEEDS + ['avg3']:
        t = table[table['seed'] == seed].sort_values('eligible_score', ascending=False)
        ranking.append(dict(seed=seed, order=' > '.join(t['config']), scores=' / '.join(f'{v:.5f}' for v in t['eligible_score'])))
    daily = {(c, s): eligible_daily_of(c, s) for c in CONFIGS for s in SEEDS}
    for c in CONFIGS:
        daily[(c, 'avg3')] = pd.concat([pd.read_parquet(OUT / 'seed_average' / f'{y}_{c}' / 'eligible_daily_metrics.parquet')
                                        .assign(year=y) for y in YEARS], ignore_index=True)
        daily[(c, 'avg3')]['date'] = pd.to_datetime(daily[(c, 'avg3')]['date']).dt.normalize()
    for seed in SEEDS + ['avg3']:
        for a, b in (('F1', 'F2'), ('F1', 'F3'), ('F2', 'F3')):
            diff, t = nw_pair(daily[(a, seed)], daily[(b, seed)])
            sa = float(table[(table['config'] == a) & (table['seed'] == seed)]['eligible_score'].iloc[0])
            sb = float(table[(table['config'] == b) & (table['seed'] == seed)]['eligible_score'].iloc[0])
            pairs.append(dict(seed=seed, pair=f'{a}-{b}', score_diff=sa - sb, mean_daily_diff=diff, nw_t=t))
    for config in CONFIGS:   # average vs each single seed
        for seed in SEEDS:
            diff, t = nw_pair(daily[(config, 'avg3')], daily[(config, seed)])
            pairs.append(dict(seed=f'avg3 vs {seed}', pair=config, score_diff=float(
                table[(table['config'] == config) & (table['seed'] == 'avg3')]['eligible_score'].iloc[0]
                - table[(table['config'] == config) & (table['seed'] == seed)]['eligible_score'].iloc[0]),
                mean_daily_diff=diff, nw_t=t))
    out = OUT / 'stage7'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'seed_scores.csv', index=False)
    spread.to_csv(out / 'seed_spread.csv', index=False)
    pd.DataFrame(ranking).to_csv(out / 'seed_ranking.csv', index=False)
    pd.DataFrame(pairs).to_csv(out / 'seed_pairs.csv', index=False)
    pd.DataFrame(best_rows).to_csv(out / 'best_iterations.csv', index=False)
    print('[stage 7] summary written', flush=True)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(table.round(5).to_string(index=False))
        print(spread.round(5).to_string(index=False))
        print(pd.DataFrame(ranking).to_string(index=False))
        print(pd.DataFrame(pairs).round(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', action='store_true')
    parser.add_argument('--one', nargs=3, metavar=('YEAR', 'CONFIG', 'SEED'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--summarize-seeds', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.one:
        year, config, seed = int(args.one[0]), args.one[1], int(args.one[2])
        assert config in CONFIGS and year in YEARS
        if args.check:
            assert seed == 42
            out = OUT / 'seeds' / f'_check_{year}_{config}_s42'
        else:
            assert seed in NEW_SEEDS
            out = seed_dir(year, config, seed)
        fit_seed(year, config, seed, out, args.threads)
    elif args.seeds:
        started = time.perf_counter()
        run_seeds()
        print(f'[done] stage 7 complete in {(time.perf_counter() - started) / 3600:.2f} h', flush=True)
    elif args.summarize_seeds:
        summarize_seeds()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
