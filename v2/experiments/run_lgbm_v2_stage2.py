"""V2 LightGBM stage 2: choose M, then the L1/L2 small grid with boundary extension.

Plan section 7 stage 2 steps 1-3, as revised in plan section 12 (2026-11-15):
  * selection metric = eligible-pool daily Rank IC (all-sample Rank IC reported alongside)
  * round cap 6000 (lr 0.02, patience 200 unchanged); best_iteration >= 0.95 x cap in any fold
    marks the combination as truncated: it cannot be the winner
  * the model with ALL trained rounds is saved (gzip text) together with best_iteration
  * early-stopping metric unchanged (all-sample daily Rank IC on the inner early-stopping set)

Phases (each fit runs in its own subprocess; completed fits are skipped):
  python v2/experiments/run_lgbm_v2_stage2.py --phase m      medium x 3 leaf thresholds x 5 folds at lambda=0 -> M
  python v2/experiments/run_lgbm_v2_stage2.py --phase grid   M and C x lambda1 {0,10,50} x lambda2 {0,1000,10000}
                                                 x 5 folds, then boundary extension (at most two)
  python v2/experiments/run_lgbm_v2_stage2.py --one YEAR STRUCTURE LEAF L1 L2 [--fresh]

Reuse rule: a lambda=(0,0) cell whose stage-1 fit stopped early (rounds_trained < 3000) is
deterministic and identical under a 6000 cap, so its stage-1 outputs are reused (eligible metrics
are recomputed from the saved outer predictions). --fresh forces a new fit instead.
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
import shutil
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
from run_lgbm_v2_stage1 import OUT as STAGE1

OUT = DATA / 'v2' / 'lgbm_stage2'
YEARS = (2017, 2018, 2019, 2020, 2021)
CAP, PATIENCE, EVERY, STAGE1_CAP = 6000, 200, 10, 3000
TRUNCATION = 0.95
L1_GRID, L2_GRID = (0, 10, 50), (0, 1000, 10000)
EXTENSION = {'l1': (50, 200), 'l2': (10000, 30000)}  # boundary value -> one added value (plan step 3)
C = ('deep', '0.1pct')
PENALTY = 0.25


# ---------------------------------------------------------------- eligible-pool evaluation
def eligible_metrics(frame, column):
    """Eligible-pool daily Rank IC: eligible rows with finite labels, >=20 stocks per day.

    Same daily computation as evaluate_common (equal-weight Spearman per day), restricted to the
    eligible pool used for the Top50 spread.
    """
    from research_controls import evaluate_common
    scored = frame.loc[frame['eligible'].to_numpy(bool) & np.isfinite(frame['y'].to_numpy(float))]
    metrics, daily = evaluate_common(scored[column].to_numpy(float), scored['y'].to_numpy(float),
                                     scored['signal_date'].to_numpy(), np.ones(len(scored)))
    daily = pd.DataFrame(daily)
    daily['date'] = pd.to_datetime(daily['date']).dt.normalize()
    ric = daily['rank_ic'].to_numpy(float)
    out = dict(rows=metrics['rows'], days=metrics['days'], mean_rank_ic=metrics['mean_rank_ic'],
               mean_ic=metrics['mean_ic'], rank_ic_valid_days=metrics['rank_ic_valid_days'],
               rank_ic_positive_share=float(np.mean(ric[np.isfinite(ric)] > 0)) if len(ric) else None)
    return out, daily[['date', 'n', 'rank_ic', 'ic']]


def paired_t(left, right):
    """Pair two daily tables on date; NW t (lag 5) of the mean difference. Raises if pairing is partial."""
    joined = left[['date', 'rank_ic']].merge(right[['date', 'rank_ic']], on='date', suffixes=('_a', '_b'),
                                             validate='one_to_one')
    if len(joined) != len(left) or len(joined) != len(right):
        raise RuntimeError(f'Daily pairing incomplete: {len(joined)} of {len(left)}/{len(right)}')
    diff = (joined['rank_ic_a'] - joined['rank_ic_b']).to_numpy()
    return dict(days=len(joined), mean_rank_ic_difference=float(diff.mean()),
                newey_west_t_lag5=common.newey_west_tstat(diff, lag=5))


def all_sample_daily(daily):
    daily = daily[['date', 'rank_ic']].copy()
    daily['date'] = pd.to_datetime(daily['date']).dt.normalize()
    return daily


def elastic_eligible(year, outer=None, elastic_pred=None):
    """Eligible-pool metrics of the cached same-key Elastic Net control (computed, never stored in the cache)."""
    if outer is None:
        directory = cache.cache_dir(year, 'full')
        outer = pd.read_parquet(directory / 'outer_meta.parquet')
        elastic_pred = np.load(directory / 'elastic' / 'outer_pred.npy')
    return eligible_metrics(outer.assign(elastic=elastic_pred), 'elastic')


# ---------------------------------------------------------------- naming
def combo_label(structure, leaf, l1, l2, ps=0):
    return f'{structure}_{leaf}_l1-{l1:g}_l2-{l2:g}' + (f'_ps-{ps:g}' if ps else '')


def fit_dir(year, structure, leaf, l1, l2, ps=0):
    return OUT / f'{year}_{combo_label(structure, leaf, l1, l2, ps)}'


def completed(output):
    path = output / 'summary.json'
    return path.exists() and common.read_json(path).get('status') == 'completed'


def truncated(best_iteration, cap=CAP):
    return best_iteration >= TRUNCATION * cap


# ---------------------------------------------------------------- one fit
def fit_one(year, structure_key, leaf_key, l1, l2, output, threads=8, extra=None, train_label=None,
            label_info=None, stage=2, tag=None):
    """One fit. `extra` adds LightGBM params (path_smooth, objective, alpha); `train_label` replaces the
    inner-fit training label only (early stopping, diagnostics and all scoring stay on the raw label)."""
    import lightgbm as lgb
    structure, leaf_fraction = STRUCTURES[structure_key], LEAF_FRACTIONS[leaf_key]
    output.mkdir(parents=True, exist_ok=True)
    timer, started = Timer(), time.perf_counter()
    with timer('load_cache'):
        data = cache.load(year, 'full')
    names, manifest = data['names'], data['manifest']
    sources = [Path(__file__), Path(common.__file__), Path(cache.__file__),
               _source_path('v2/experiments/run_lgbm_v2_stage1.py')]
    config = dict(stage=stage, year=year, structure=structure_key, leaf=leaf_key, leaf_fraction=leaf_fraction,
                  lambda_l1=l1, lambda_l2=l2, cap=CAP, patience=PATIENCE, every=EVERY,
                  truncation_rule=f'best_iteration >= {TRUNCATION} x cap', threads=threads,
                  cache_hashes=manifest['hashes'], cache_params=manifest['params'], test_used=False,
                  sources={p.name: common.sha256(p) for p in sources})
    if extra:
        config['extra_params'] = extra
    if label_info is not None:
        config['training_label'] = label_info
    config_path = output / 'config.json'
    if config_path.exists() and common.read_json(config_path) != json.loads(json.dumps(config, default=str)):
        raise RuntimeError(f'{output}: existing output belongs to a different configuration or code version')
    common.save_json(config_path, config)

    x_fit, y_fit, fit_dates = data['fit_x'], data['fit_y'], data['fit_keys']['signal_date'].to_numpy()
    x_es, y_es = data['es_x'], data['es_y']
    params = lgbm_params(structure, leaf_fraction, len(y_fit), threads)
    params.update(lambda_l1=float(l1), lambda_l2=float(l2), **(extra or {}))
    fit_label = y_fit if train_label is None else np.asarray(train_label, dtype=np.float64)
    if len(fit_label) != len(y_fit) or not np.isfinite(fit_label).all():
        raise RuntimeError('Training label must be finite and aligned with the inner fit set')
    monitor = common.RankICEarlyStopping(y_es, data['es_keys']['signal_date'].to_numpy(),
                                         every=EVERY, patience=PATIENCE)
    with timer('dataset_construct'):
        train_set = lgb.Dataset(x_fit, label=fit_label, feature_name=names, free_raw_data=False, params=params)
        valid_set = lgb.Dataset(x_es, label=y_es, reference=train_set, params=params)
        train_set.construct()
        valid_set.construct()
    # Trees present after each round. With feature_fraction < 1 and a strong lambda_l1, some rounds find no split
    # with positive gain in their sampled features; LightGBM then adds no tree for that round (and lgb.train keeps
    # iterating), so the round number and the tree count diverge. The model at the early-stopping best round is
    # the first trees_per_round[best - 1] trees.
    trees_per_round = []

    def record_trees(env):
        trees_per_round.append(env.model.num_trees())
    record_trees.order = 20
    with timer('train'):
        monitor.started = time.perf_counter()
        # keep_training_booster keeps every trained tree (lgb.train would otherwise truncate to best).
        booster = lgb.train(params, train_set, num_boost_round=CAP, valid_sets=[valid_set],
                            valid_names=['early_stop'], feval=monitor.feval,
                            callbacks=monitor.callbacks() + [record_trees], keep_training_booster=True)
    best, trained_rounds = monitor.best_iteration, monitor.iteration
    if best <= 0:
        raise RuntimeError('No valid early-stopping score was recorded')
    trees_built = booster.num_trees()
    monotone = all(b >= a for a, b in zip(trees_per_round, trees_per_round[1:]))
    if len(trees_per_round) != trained_rounds or trees_per_round[-1] != trees_built or not monotone:
        raise RuntimeError('Per-round tree counts are inconsistent with the trained booster')
    skipped_rounds = trained_rounds - trees_built
    growth_stopped = skipped_rounds > 0
    best_trees = trees_per_round[best - 1]
    if growth_stopped:
        print(f'[note] {skipped_rounds} of {trained_rounds} rounds added no tree (no split with positive gain in the '
              f'sampled features); best round {best} = {best_trees} trees', flush=True)
    peak_after_train = common.peak_rss_gb()
    name = tag or combo_label(structure_key, leaf_key, l1, l2, (extra or {}).get('path_smooth', 0))
    print(f'[{year} {name}] rounds={trained_rounds} best={best} '
          f'es RankIC={monitor.best_score:.5f} train={timer.records["train"]:.0f}s', flush=True)

    with timer('save_full_model'):
        # num_iteration=-1 saves all trained rounds (None would mean best_iteration in LightGBM).
        text = booster.model_to_string(num_iteration=-1)
        with gzip.open(output / 'lightgbm_model_all_rounds.txt.gz', 'wt', encoding='utf-8', compresslevel=6) as h:
            h.write(text)
        model_bytes = len(text)
        del text
        common.save_json(output / 'best_iteration.json',
                         dict(best_iteration=best, best_model_trees=best_trees, rounds_trained=trained_rounds,
                              trees_built=trees_built, skipped_rounds=skipped_rounds,
                              note='predict with num_iteration=best_model_trees for the early-stopped model'))

    with timer('diagnostics'):
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(y_fit), min(300_000, len(y_fit)), replace=False))
        train_ic = common.DailyRankIC(y_fit[idx], fit_dates[idx])(booster.predict(x_fit[idx], num_iteration=best_trees))
        diagnostics = dict(trees=common.tree_diagnostics(booster, best_trees),
                           leaf_coverage=common.leaf_date_coverage(booster, x_fit, fit_dates, best_trees, trees=10),
                           inner_fit_rank_ic=train_ic, early_stop_rank_ic=monitor.best_score,
                           fit_minus_early_stop_rank_ic=train_ic - monitor.best_score)
    del train_set, valid_set
    gc.collect()

    with timer('outer_predict'):
        outer_x = data['outer_x']
        pred = np.concatenate([booster.predict(np.asarray(outer_x[s:s + 200_000]), num_iteration=best_trees)
                               for s in range(0, len(outer_x), 200_000)])
    if not np.isfinite(pred).all():
        raise RuntimeError('Non-finite outer predictions')
    outer = data['outer'].assign(lgbm=pred)
    with timer('outer_metrics'):
        metrics, daily = common.evaluate_outer(outer, 'lgbm')
        metrics['timing_split'] = common.timing_selection_split(outer, 'lgbm')
        eligible, eligible_daily = eligible_metrics(outer, 'lgbm')
        elastic_elig, elastic_elig_daily = elastic_eligible(year, data['outer'], data['elastic_pred'])
        comparison = dict(all_sample=paired_t(all_sample_daily(daily), all_sample_daily(data['elastic_daily'])),
                          eligible=paired_t(eligible_daily, elastic_elig_daily),
                          elastic_all_sample_mean_rank_ic=data['elastic_metrics']['mean_rank_ic'],
                          elastic_eligible_mean_rank_ic=elastic_elig['mean_rank_ic'])

    outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(output / 'outer_predictions.parquet', index=False)
    pd.DataFrame(monitor.curve, columns=['iteration', 'early_stop_rank_ic', 'seconds']).to_parquet(
        output / 'early_stop_curve.parquet', index=False)
    daily.to_parquet(output / 'daily_metrics.parquet', index=False)
    eligible_daily.to_parquet(output / 'eligible_daily_metrics.parquet', index=False)

    laps = monitor.lap_times
    per_100 = [round(b[1] - a[1], 2) for a, b in zip([(0, 0.0)] + laps[:-1], laps)]
    rows = manifest['rows']
    summary = dict(
        status='completed', stage=stage, source='fresh', year=year, structure=structure_key, leaf=leaf_key,
        lambda_l1=l1, lambda_l2=l2, extra_params=extra or {}, training_label=label_info,
        rows=dict(training_pool=rows['training_pool'], full_sample=rows['full_sample'], inner_fit=len(y_fit),
                  early_stop=len(y_es), outer=len(outer), outer_scored=int(np.isfinite(outer['y']).sum()),
                  outer_eligible_scored=eligible['rows']),
        lightgbm=dict(params=params, cap=CAP, rounds_trained=trained_rounds, trees_built=trees_built,
                      tree_growth_stopped=growth_stopped, skipped_rounds=skipped_rounds, best_iteration=best,
                      best_model_trees=best_trees,
                      truncated=truncated(best), early_stop_best_rank_ic=monitor.best_score,
                      full_model_text_bytes=model_bytes),
        eligible_metrics=eligible, outer_metrics=metrics, lgbm_vs_elastic=comparison, diagnostics=diagnostics,
        timing_seconds=dict(**timer.records, total=round(time.perf_counter() - started, 1),
                            seconds_per_100_rounds=per_100,
                            mean_seconds_per_100_rounds=float(np.mean(per_100)) if per_100 else None),
        peak_memory_gb=common.peak_rss_gb(), peak_memory_gb_after_lgbm_train=peak_after_train,
        machine=dict(platform=platform.platform(), processor=platform.processor(), threads=threads,
                     lightgbm=lgb.__version__),
        test_used=False)
    common.save_json(output / 'summary.json', summary)  # written last: marks a completed fit
    print(f'[{year} {name}] eligible RankIC={eligible["mean_rank_ic"]:.5f} '
          f'(elastic {elastic_elig["mean_rank_ic"]:.5f}, t={comparison["eligible"]["newey_west_t_lag5"]:.2f}) '
          f'all={metrics["mean_rank_ic"]:.5f} trunc={truncated(best)} leaves={diagnostics["trees"]["mean_leaves"]:.1f} '
          f'total={summary["timing_seconds"]["total"]:.0f}s peak={summary["peak_memory_gb"]:.2f}GB', flush=True)
    return summary


# ---------------------------------------------------------------- reuse of stage-1 lambda=0 fits
def stage1_reusable(year, structure, leaf):
    path = STAGE1 / f'{year}_{structure}_{leaf}' / 'summary.json'
    if not path.exists():
        return None
    s1 = common.read_json(path)
    return s1 if s1['lightgbm']['rounds_trained'] < STAGE1_CAP else None


def reuse_stage1(year, structure, leaf, output):
    """Stage-1 fit stopped early at a 3000 cap -> identical at 6000 (deterministic). Copy and add eligible metrics."""
    s1 = stage1_reusable(year, structure, leaf)
    if s1 is None:
        raise RuntimeError(f'{year} {structure} {leaf}: stage-1 fit is not reusable')
    source = STAGE1 / f'{year}_{structure}_{leaf}'
    output.mkdir(parents=True, exist_ok=True)
    directory = cache.cache_dir(year, 'full')
    outer = pd.read_parquet(directory / 'outer_meta.parquet')
    pred = pd.read_parquet(source / 'outer_predictions.parquet')
    if not (pred['ts_code'].to_numpy() == outer['ts_code'].to_numpy()).all() or \
            not (pd.to_datetime(pred['signal_date']).to_numpy() == pd.to_datetime(outer['signal_date']).to_numpy()).all():
        raise RuntimeError(f'{source}: outer predictions do not align with the cache')
    outer['lgbm'] = pred['lgbm'].to_numpy()
    eligible, eligible_daily = eligible_metrics(outer, 'lgbm')
    elastic_elig, elastic_elig_daily = elastic_eligible(year)
    daily = pd.read_parquet(source / 'daily_metrics.parquet')
    elastic_daily = pd.read_parquet(directory / 'elastic' / 'daily_metrics.parquet')
    for name in ('daily_metrics.parquet', 'early_stop_curve.parquet', 'outer_predictions.parquet'):
        shutil.copy2(source / name, output / name)
    eligible_daily.to_parquet(output / 'eligible_daily_metrics.parquet', index=False)
    summary = dict(s1)
    summary.update(
        stage=2, source='reused_stage1', reused_from=str(source), lambda_l1=0, lambda_l2=0,
        reuse_basis=(f'stage-1 fit stopped by early stopping at round {s1["lightgbm"]["rounds_trained"]} < 3000; '
                     'deterministic training is identical under a 6000 cap'),
        eligible_metrics=eligible,
        lgbm_vs_elastic=dict(all_sample=paired_t(all_sample_daily(daily), all_sample_daily(elastic_daily)),
                             eligible=paired_t(eligible_daily, elastic_elig_daily),
                             elastic_all_sample_mean_rank_ic=s1['lgbm_vs_elastic']['elastic_mean_rank_ic'],
                             elastic_eligible_mean_rank_ic=elastic_elig['mean_rank_ic']))
    summary['lightgbm'] = dict(s1['lightgbm'], cap=CAP, truncated=truncated(s1['lightgbm']['best_iteration']),
                               full_model_saved=False)
    summary['rows'] = dict(s1['rows'], outer_eligible_scored=eligible['rows'])
    common.save_json(output / 'summary.json', summary)
    print(f'[{year} {combo_label(structure, leaf, 0, 0)}] reused stage 1 (rounds {s1["lightgbm"]["rounds_trained"]}, '
          f'best {s1["lightgbm"]["best_iteration"]}) eligible RankIC={eligible["mean_rank_ic"]:.5f}', flush=True)
    return summary


# ---------------------------------------------------------------- selection helpers
def load_summary(year, structure, leaf, l1, l2, ps=0):
    return common.read_json(fit_dir(year, structure, leaf, l1, l2, ps) / 'summary.json')


def combo_scores(structure, leaf, l1, l2, years=YEARS, ps=0):
    rows = [load_summary(y, structure, leaf, l1, l2, ps) for y in years]
    elig = np.array([r['eligible_metrics']['mean_rank_ic'] for r in rows])
    full = np.array([r['outer_metrics']['mean_rank_ic'] for r in rows])
    best = [r['lightgbm']['best_iteration'] for r in rows]
    return dict(structure=structure, leaf=leaf, l1=l1, l2=l2, ps=ps,
                mean_leaves=float(np.mean([r['diagnostics']['trees']['mean_leaves'] for r in rows])),
                **{f'eligible_{y}': v for y, v in zip(years, elig)},
                eligible_mean=float(elig.mean()), eligible_std=float(elig.std(ddof=0)),
                eligible_score=float(elig.mean() - PENALTY * elig.std(ddof=0)),
                **{f'all_{y}': v for y, v in zip(years, full)},
                all_mean=float(full.mean()), all_score=float(full.mean() - PENALTY * full.std(ddof=0)),
                negative_fold=bool((elig < 0).any()), negative_fold_all_sample=bool((full < 0).any()),
                best_iterations=best, truncated=any(truncated(b) for b in best))


def winner(table):
    ok = table.loc[~table['negative_fold'] & ~table['truncated']]
    if not len(ok):
        raise RuntimeError('No admissible combination (all truncated or with a negative fold)')
    return ok.sort_values('eligible_score', ascending=False).iloc[0]


# ---------------------------------------------------------------- driver
def run_job(year, structure, leaf, l1, l2, threads, fresh=False, ps=0):
    output = fit_dir(year, structure, leaf, l1, l2, ps)
    if completed(output):
        return
    reusable = l1 == 0 and l2 == 0 and not fresh and not ps and stage1_reusable(year, structure, leaf) is not None
    print(f'[start] {output.name} ({"reuse" if reusable else "fit"}) at {time.strftime("%H:%M:%S")}', flush=True)
    args = [sys.executable, '-u', __file__, '--one', str(year), structure, leaf, str(l1), str(l2),
            '--threads', str(threads)] + (['--fresh'] if fresh else []) + (['--path-smooth', str(ps)] if ps else [])
    code = subprocess.call(args)
    if code != 0:
        raise SystemExit(f'{output.name} failed with exit code {code}; stopping')


# Cells re-fitted fresh although reusable: the per-100-round diagnostic needs all-round models for
# 2021 at lambda=0 (C and every medium candidate, since M is not known before phase m). Comparing
# them with stage 1 also checks the reuse basis empirically.
FRESH_CELLS = {(2021, 'deep', '0.1pct', 0, 0)} | {(2021, 'medium', leaf, 0, 0) for leaf in LEAF_FRACTIONS}


def phase_m(threads):
    for leaf in LEAF_FRACTIONS:
        for year in YEARS:
            run_job(year, 'medium', leaf, 0, 0, threads, fresh=(year, 'medium', leaf, 0, 0) in FRESH_CELLS)
    table = pd.DataFrame([combo_scores('medium', leaf, 0, 0) for leaf in LEAF_FRACTIONS])
    eligible_pool = table.loc[~table['negative_fold']].sort_values('eligible_score', ascending=False)
    if not len(eligible_pool):
        raise RuntimeError('All medium configurations have a negative fold')
    chosen = eligible_pool.iloc[0]
    by_all = table.loc[~table['negative_fold_all_sample']].sort_values('all_score', ascending=False)
    decision = dict(rule='M = medium configuration with the highest five-fold eligible selection score; '
                         'configurations with a negative eligible fold are excluded (plan section 8 rule 2)',
                    m_leaf=chosen['leaf'], m_truncated=bool(chosen['truncated']),
                    m_by_all_sample_score=by_all.iloc[0]['leaf'] if len(by_all) else None,
                    table=json.loads(table.to_json(orient='records')))
    common.save_json(OUT / 'm_selection.json', decision)
    print(f'[M] medium_{chosen["leaf"]} (eligible score {chosen["eligible_score"]:.5f}, truncated={bool(chosen["truncated"])}); '
          f'all-sample choice would be medium_{decision["m_by_all_sample_score"]}', flush=True)
    return chosen['leaf']


def grid_values(state, structure):
    return state['l1'][structure], state['l2'][structure]


def phase_grid(threads):
    m_leaf = common.read_json(OUT / 'm_selection.json')['m_leaf']
    structures = [('medium', m_leaf), C]
    state_path = OUT / 'grid_state.json'
    state = common.read_json(state_path) if state_path.exists() else dict(
        l1={s: list(L1_GRID) for s, _ in structures}, l2={s: list(L2_GRID) for s, _ in structures},
        extensions=[])

    def run_grid():
        for structure, leaf in structures:
            l1s, l2s = grid_values(state, structure)
            for l1 in l1s:
                for l2 in l2s:
                    for year in YEARS:
                        run_job(year, structure, leaf, l1, l2, threads,
                                fresh=(year, structure, leaf, l1, l2) in FRESH_CELLS)

    def score_table():
        return pd.DataFrame([combo_scores(s, leaf, a, b) for s, leaf in structures
                             for a in grid_values(state, s)[0] for b in grid_values(state, s)[1]])

    run_grid()
    for attempt in range(3):  # initial grid + at most two extensions
        table = score_table()
        best = winner(table)
        print(f'[winner] {combo_label(best["structure"], best["leaf"], best["l1"], best["l2"])} '
              f'eligible score {best["eligible_score"]:.5f}', flush=True)
        structure = best['structure']
        added = []
        for axis in ('l1', 'l2'):
            boundary, new = EXTENSION[axis]
            if best[axis] == boundary and new not in state[axis][structure]:
                state[axis][structure].append(new)
                added.append(f'{axis}={new}')
        at_new_boundary = [axis for axis in ('l1', 'l2') if best[axis] == EXTENSION[axis][1]]
        if not added:
            state['final'] = dict(winner=combo_label(best['structure'], best['leaf'], best['l1'], best['l2']),
                                  at_extended_boundary=at_new_boundary,
                                  extensions_used=len(state['extensions']))
            common.save_json(state_path, state)
            if at_new_boundary:
                print(f'[note] winner is at an already-extended boundary {at_new_boundary}; the plan defines no '
                      f'further values, so no further extension', flush=True)
            break
        if len(state['extensions']) >= 2:
            state['final'] = dict(winner=combo_label(best['structure'], best['leaf'], best['l1'], best['l2']),
                                  note='extension limit (2) reached', pending=added)
            common.save_json(state_path, state)
            print('[note] extension limit reached', flush=True)
            break
        state['extensions'].append(dict(after_winner=combo_label(best['structure'], best['leaf'], best['l1'], best['l2']),
                                        structure=structure, added=added))
        common.save_json(state_path, state)
        print(f'[extend] {structure}: {added}', flush=True)
        run_grid()
    print('[done] stage 2 grid and extensions complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=['m', 'grid'])
    parser.add_argument('--one', nargs=5, metavar=('YEAR', 'STRUCTURE', 'LEAF', 'L1', 'L2'))
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--path-smooth', type=float, default=0.0)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.one:
        year, structure, leaf = int(args.one[0]), args.one[1], args.one[2]
        l1, l2 = float(args.one[3]), float(args.one[4])
        l1, l2 = (int(l1) if l1.is_integer() else l1), (int(l2) if l2.is_integer() else l2)
        if year not in YEARS or structure not in STRUCTURES or leaf not in LEAF_FRACTIONS:
            raise ValueError('Not a stage 2 configuration')
        ps = int(args.path_smooth) if float(args.path_smooth).is_integer() else args.path_smooth
        output = fit_dir(year, structure, leaf, l1, l2, ps)
        if ps:
            fit_one(year, structure, leaf, l1, l2, output, args.threads, extra=dict(path_smooth=float(ps)))
        elif l1 == 0 and l2 == 0 and not args.fresh and stage1_reusable(year, structure, leaf) is not None:
            reuse_stage1(year, structure, leaf, output)
        else:
            fit_one(year, structure, leaf, l1, l2, output, args.threads)
    elif args.phase == 'm':
        phase_m(args.threads)
    elif args.phase == 'grid':
        phase_grid(args.threads)
    else:
        parser.error('choose --phase or --one')


if __name__ == '__main__':
    main()
