"""V2 LightGBM stage 4 (plan section 15): learning rate and early stopping, D label, configuration = P3.

P3 = deep 0.1%, lambda1 x1 (200-210 by fold), lambda2 10000, feature_fraction 0.3, bagging 1.0, max_bin 255,
no extra_trees, path_smooth 0. Only the learning rate and the round cap change:
    lr 0.05 / 0.03 / 0.02 / 0.01  ->  cap 4000 / 6000 / 8000 / 16000
Every fit trains to the cap (no early stopping) and records
  * every 10 rounds, two early-stopping-set curves on the raw label: all-sample daily Rank IC and eligible-pool
    daily Rank IC (eligibility recomputed for the early-stopping keys with the outer definition);
  * trees after every round (skipped rounds), the full model;
  * the outer eligible-pool (and all-sample) daily Rank IC every 100 trees, by accumulating raw scores of
    100-tree segments; the last point is asserted equal to a one-shot prediction with all trees.
Early-stopping rules R1-R6 are then compared offline on the saved curves (no new fits).

  python v2/experiments/run_lgbm_v2_stage4.py --all             eligibility, fits (resumable), analysis, summary
  python v2/experiments/run_lgbm_v2_stage4.py --one YEAR LR [--l2 L2]   one fit (subprocess)
  python v2/experiments/run_lgbm_v2_stage4.py --analyze         steps 2-4 from saved outputs
Outputs: quant_data/processed/v2/lgbm_stage4/ . Log: v2/lgbm_stage4_log.txt.
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
from run_lgbm_v2_stage2 import PENALTY, YEARS, completed, eligible_metrics
import run_lgbm_v2_stage3 as s3

OUT = DATA / 'v2' / 'lgbm_stage4'
LRS = {0.05: 4000, 0.03: 6000, 0.02: 8000, 0.01: 16000}
LR_ORDER = (0.05, 0.03, 0.02, 0.01)  # larger first: a smaller lr must beat the current choice
BASE_LR, BASE_RULE, BASE_PATIENCE = 0.02, 'R1', 200
P3_L2, RECHECK_L2 = 10000, 1000
EVERY, OUTER_STEP, TRUNCATION = 10, 100, 0.95
MIN_GAIN, MIN_T = 0.001, 2.0  # practical threshold (plan section 8 item 8 / section 15 item 2)
SMOOTH_WINDOW, SMOOTH_DELTA = 3, 0.0001
STAGE3_BASE = 'deepD_x1_l2-10000'
VERIFY_YEAR, VERIFY_ROUNDS = 2017, 1000
RULES = ('R1', 'R2', 'R3', 'R4', 'R5', 'R6')
RULE_TEXT = {'R1': '当前规则（早停集全样本，耐心 200×0.02/lr）', 'R2': '耐心减半', 'R3': '耐心加倍',
             'R4': '平滑（3 次评估滚动均值，提升 < 0.0001 不算）', 'R5': '资格池早停（早停集资格池指标）',
             'R6': '固定轮数（留一折外层平均曲线峰值）'}
CHUNK = 200_000


def lr_tag(lr):
    return f'lr{lr:g}'


def fit_dir(year, lr, l2=P3_L2):
    return OUT / (f'{year}_{lr_tag(lr)}' + ('' if l2 == P3_L2 else f'_l2-{l2:g}'))


# ---------------------------------------------------------------- eligibility of the early-stopping keys
def eligibility(keys, partition_year):
    """Outer eligible-pool definition (fetch_outer_month) for arbitrary keys within one daily partition year."""
    import duckdb
    keys = keys[['ts_code', 'signal_date']].copy()
    keys['signal_date'] = pd.to_datetime(keys['signal_date']).dt.normalize()
    if not (keys['signal_date'].dt.year == partition_year).all():
        raise RuntimeError(f'Keys outside {partition_year}')
    if partition_year > 2021:
        raise ValueError('Search folds only: 2022+ data are not read')
    con = duckdb.connect()
    con.execute('SET threads=8')
    files = [str(p) for p in common.parquet_files(common.PARQUET / 'daily', 'year', partition_year, partition_year)]
    con.read_parquet(files, hive_partitioning=False).create_view('daily')
    basic = pd.read_csv(common.STOCK_BASIC, dtype={'list_date': str})[['ts_code', 'list_date']]
    basic['list_date'] = pd.to_datetime(basic['list_date'], format='%Y%m%d', errors='coerce')
    con.register('basic', basic)
    con.register('k', keys.assign(row=np.arange(len(keys))))
    frame = con.execute("""
        SELECT k.row,
               coalesce(d.amount >= 50000 AND b.list_date IS NOT NULL
                        AND date_diff('day', b.list_date, CAST(k.signal_date AS DATE)) >= 60, false) eligible
        FROM k
        LEFT JOIN daily d ON d.ts_code = k.ts_code AND CAST(d.trade_date AS DATE) = CAST(k.signal_date AS DATE)
        LEFT JOIN basic b ON b.ts_code = k.ts_code
        ORDER BY k.row""").fetch_df()
    con.close()
    if len(frame) != len(keys) or not np.array_equal(frame['row'].to_numpy(), np.arange(len(keys))):
        raise RuntimeError('Eligibility join changed the row set (duplicate daily rows?)')
    return frame['eligible'].to_numpy(bool)


def es_eligible(year):
    """Eligible flags of the cached early-stopping keys (dates all in year - 1), cached under OUT/es_eligible."""
    directory = OUT / 'es_eligible'
    directory.mkdir(parents=True, exist_ok=True)
    keys = pd.read_parquet(cache.cache_dir(year, 'full') / 'es_keys.parquet')
    key_hash = cache.key_hash(keys)
    path, meta_path = directory / f'{year}.npy', directory / f'{year}.json'
    if path.exists() and meta_path.exists():
        meta = common.read_json(meta_path)
        flags = np.load(path)
        if meta['es_keys_hash'] != key_hash or len(flags) != len(keys):
            raise RuntimeError(f'{path}: cached eligibility does not match the early-stopping keys')
        return flags
    dates = pd.to_datetime(keys['signal_date'])
    if not (dates.dt.year == year - 1).all():
        raise RuntimeError(f'{year}: early-stopping dates are not all in {year - 1}')
    flags = eligibility(keys, year - 1)
    # The same function must reproduce the cached outer eligibility exactly (definition check).
    outer = pd.read_parquet(cache.cache_dir(year, 'full') / 'outer_meta.parquet', columns=['ts_code', 'signal_date',
                                                                                        'eligible'])
    outer_flags = eligibility(outer, year)
    if not np.array_equal(outer_flags, outer['eligible'].to_numpy(bool)):
        raise RuntimeError(f'{year}: eligibility recomputation differs from the cached outer eligibility')
    per_day = pd.Series(flags).groupby(dates.dt.normalize().to_numpy()).sum()
    meta = dict(year=year, es_year=year - 1, es_keys_hash=key_hash, rows=len(flags), eligible=int(flags.sum()),
                eligible_share=float(flags.mean()), days=int(len(per_day)),
                days_with_at_least_20_eligible=int((per_day >= 20).sum()),
                min_eligible_per_day=int(per_day.min()), outer_definition_check='identical to cached outer flags',
                definition='daily.amount >= 50000 (thousand yuan) and listed >= 60 days on the signal date')
    np.save(path, flags)
    common.save_json(meta_path, meta)
    print(f'[es-eligible {year}] {meta["eligible"]:,} of {len(flags):,} ({meta["eligible_share"]:.3f}); '
          f'days {meta["days"]}, >=20 eligible {meta["days_with_at_least_20_eligible"]}; outer check passed', flush=True)
    return flags


# ---------------------------------------------------------------- curves recorded during training
class Curves:
    """Early-stopping-set curves every `every` rounds; never stops training."""

    def __init__(self, y, dates, eligible, every):
        self.all = common.DailyRankIC(y, dates)
        self.idx = np.flatnonzero(eligible)
        self.elig = common.DailyRankIC(np.asarray(y)[self.idx], np.asarray(dates)[self.idx])
        self.every, self.iteration, self.total = every, 0, None
        self.rows, self.last = [], float('nan')
        self.started = time.perf_counter()
        self.lap_times = []

    def before(self, env):
        self.iteration = env.iteration + 1
        self.total = env.end_iteration
        if self.iteration > 1 and (self.iteration - 1) % 100 == 0:
            self.lap_times.append((self.iteration - 1, time.perf_counter() - self.started))

    def feval(self, preds, _data):
        it = self.iteration
        if it % self.every == 0 or it == self.total:
            preds = np.asarray(preds)
            self.last = self.all(preds)
            self.rows.append((it, self.last, self.elig(preds[self.idx]), time.perf_counter() - self.started))
        return 'daily_rank_ic', self.last, True


def predict_chunks(booster, x, **kwargs):
    return np.concatenate([booster.predict(np.asarray(x[s:s + CHUNK]), **kwargs) for s in range(0, len(x), CHUNK)])


def day_index(dates, mask):
    starts, ends = common.day_slices(np.asarray(dates)[mask])
    d = np.asarray(dates)[mask]
    return [pd.Timestamp(d[s]).normalize() for s, e in zip(starts, ends) if e - s >= 20]


# ---------------------------------------------------------------- one fit
def fit_one(year, lr, output, threads=8, l2=P3_L2):
    import lightgbm as lgb
    cap = LRS[lr]
    output.mkdir(parents=True, exist_ok=True)
    timer, started = Timer(), time.perf_counter()
    with timer('load_cache'):
        data = cache.load(year, 'full')
        es_flags = es_eligible(year)
        label = s3.d_label(year)
    names, manifest = data['names'], data['manifest']
    l1 = s3.num(s3.deep_l1(year, 1.0))
    sources = [Path(__file__), Path(common.__file__), Path(cache.__file__), _source_path('v2/experiments/run_lgbm_v2_stage1.py'),
               _source_path('v2/experiments/run_lgbm_v2_stage2.py'), _source_path('v2/experiments/run_lgbm_v2_stage3.py'),
               _source_path('v2/experiments/run_lgbm_v2_stage6a.py')]
    config = dict(stage=4, year=year, structure='deep', leaf='0.1pct', lambda_l1=l1, lambda_l2=l2,
                  learning_rate=lr, cap=cap, every=EVERY, outer_step=OUTER_STEP, early_stopping='none (train to cap)',
                  training_label=dict(scheme='D', label_file=f'labels/{year}_D.npy', std_ratio=s3.ratio_d(year)),
                  threads=threads, cache_hashes=manifest['hashes'], cache_params=manifest['params'],
                  es_eligible=common.read_json(OUT / 'es_eligible' / f'{year}.json'), test_used=False,
                  sources={p.name: common.sha256(p) for p in sources})
    config_path = output / 'config.json'
    if config_path.exists() and common.read_json(config_path) != json.loads(json.dumps(config, default=str)):
        raise RuntimeError(f'{output}: existing output belongs to a different configuration or code version')
    common.save_json(config_path, config)

    x_fit, y_fit = data['fit_x'], data['fit_y']
    x_es, y_es = data['es_x'], data['es_y']
    if len(label) != len(y_fit) or not np.isfinite(label).all():
        raise RuntimeError('D label must be finite and aligned with the inner fit set')
    if len(es_flags) != len(y_es):
        raise RuntimeError('Early-stopping eligibility is not aligned with the early-stopping set')
    params = lgbm_params(STRUCTURES['deep'], LEAF_FRACTIONS['0.1pct'], len(y_fit), threads)
    params.update(lambda_l1=float(l1), lambda_l2=float(l2), learning_rate=float(lr))
    curves = Curves(y_es, data['es_keys']['signal_date'].to_numpy(), es_flags, EVERY)
    with timer('dataset_construct'):
        train_set = lgb.Dataset(x_fit, label=np.asarray(label, dtype=np.float64), feature_name=names,
                                free_raw_data=False, params=params)
        valid_set = lgb.Dataset(x_es, label=y_es, reference=train_set, params=params)
        train_set.construct()
        valid_set.construct()
    trees_per_round = []

    def record_trees(env):
        trees_per_round.append(env.model.num_trees())
    record_trees.order = 20

    def before(env):
        curves.before(env)
    before.before_iteration = True
    before.order = 10
    with timer('train'):
        curves.started = time.perf_counter()
        booster = lgb.train(params, train_set, num_boost_round=cap, valid_sets=[valid_set],
                            valid_names=['early_stop'], feval=curves.feval, callbacks=[before, record_trees],
                            keep_training_booster=True)
    trees_built = booster.num_trees()
    monotone = all(b >= a for a, b in zip(trees_per_round, trees_per_round[1:]))
    if len(trees_per_round) != cap or trees_per_round[-1] != trees_built or not monotone:
        raise RuntimeError(f'Rounds/trees inconsistent: {len(trees_per_round)} rounds (cap {cap}), '
                           f'{trees_built} trees, monotone={monotone}')
    tpr = np.asarray(trees_per_round, dtype=np.int64)
    curve = pd.DataFrame(curves.rows, columns=['iteration', 'es_all', 'es_eligible', 'seconds'])
    if not np.array_equal(curve['iteration'].to_numpy(), np.arange(EVERY, cap + 1, EVERY)):
        raise RuntimeError('Early-stopping curve is not on the 10-round grid')
    curve['trees'] = tpr[curve['iteration'].to_numpy() - 1]
    skipped = int(cap - trees_built)
    peak_after_train = common.peak_rss_gb()
    print(f'[{year} {lr_tag(lr)} l2={l2:g}] rounds={cap} trees={trees_built} skipped={skipped} '
          f'train={timer.records["train"]:.0f}s es_all max={curve["es_all"].max():.5f} '
          f'es_elig max={curve["es_eligible"].max():.5f}', flush=True)

    with timer('save_full_model'):
        text = booster.model_to_string(num_iteration=-1)
        with gzip.open(output / 'lightgbm_model_all_rounds.txt.gz', 'wt', encoding='utf-8', compresslevel=6) as h:
            h.write(text)
        model_bytes = len(text)
        del text
        np.save(output / 'trees_per_round.npy', tpr)
        curve.to_parquet(output / 'es_curve.parquet', index=False)
    del train_set, valid_set, x_fit
    data.pop('fit_x', None)
    gc.collect()

    with timer('outer_curve'):
        outer = data['outer']
        x_out = np.asarray(data['outer_x'])
        y = outer['y'].to_numpy(float)
        dates = outer['signal_date'].to_numpy()
        finite = np.isfinite(y)
        elig = finite & outer['eligible'].to_numpy(bool)
        all_ic, elig_ic = common.DailyRankIC(y[finite], dates[finite]), common.DailyRankIC(y[elig], dates[elig])
        all_days, elig_days = day_index(dates, finite), day_index(dates, elig)
        points = list(range(OUTER_STEP, trees_built + 1, OUTER_STEP))
        if trees_built % OUTER_STEP:
            points.append(trees_built)
        cum, previous = np.zeros(len(x_out)), 0
        daily_elig, daily_all = {}, {}
        for k in points:
            cum += predict_chunks(booster, x_out, start_iteration=previous, num_iteration=k - previous, raw_score=True)
            previous = k
            daily_elig[k] = elig_ic.daily(cum[elig])
            daily_all[k] = all_ic.daily(cum[finite])
    with timer('outer_full_check'):
        full = predict_chunks(booster, x_out, num_iteration=trees_built, raw_score=True)
        max_abs = float(np.max(np.abs(full - cum)))
        if not np.allclose(cum, full, rtol=1e-9, atol=1e-12):
            raise RuntimeError(f'Segment-accumulated outer scores differ from the one-shot prediction ({max_abs})')
        last = points[-1]
        one_shot_elig = elig_ic(full[elig])
        if abs(one_shot_elig - float(np.nanmean(daily_elig[last]))) > 1e-9:
            raise RuntimeError('Last outer curve point differs from the one-shot eligible Rank IC')
        reference, _ = eligible_metrics(outer.assign(lgbm=full), 'lgbm')
        reference_all, _ = eligible_metrics(outer.assign(lgbm=full, eligible=True), 'lgbm')
        if abs(reference['mean_rank_ic'] - one_shot_elig) > 1e-9 or \
                abs(reference_all['mean_rank_ic'] - float(np.nanmean(daily_all[last]))) > 1e-9:
            raise RuntimeError('Curve metric differs from the pipeline eligible / all-sample Rank IC')
    stage3 = None
    if lr == BASE_LR and l2 == P3_L2:
        with timer('stage3_consistency'):
            stage3 = stage3_consistency(year, curve, tpr, booster, x_out)
    del x_out, cum, full
    gc.collect()

    frame_elig = pd.DataFrame(daily_elig, index=pd.DatetimeIndex(elig_days, name='date'))
    frame_all = pd.DataFrame(daily_all, index=pd.DatetimeIndex(all_days, name='date'))
    frame_elig.columns = [str(c) for c in frame_elig.columns]
    frame_all.columns = [str(c) for c in frame_all.columns]
    frame_elig.to_parquet(output / 'outer_daily_eligible.parquet')
    frame_all.to_parquet(output / 'outer_daily_all.parquet')
    outer_curve = pd.DataFrame(dict(trees=points, eligible=[float(np.nanmean(daily_elig[k])) for k in points],
                                    all_sample=[float(np.nanmean(daily_all[k])) for k in points]))
    outer_curve.to_parquet(output / 'outer_curve.parquet', index=False)
    peak = outer_curve.loc[outer_curve['eligible'].idxmax()]

    laps = curves.lap_times
    per_100 = [round(b[1] - a[1], 2) for a, b in zip([(0, 0.0)] + laps[:-1], laps)]
    summary = dict(
        status='completed', stage=4, year=year, learning_rate=lr, cap=cap, lambda_l1=l1, lambda_l2=l2,
        rows=dict(inner_fit=len(y_fit), early_stop=len(y_es), early_stop_eligible=int(es_flags.sum()),
                  outer=len(outer), outer_scored=int(finite.sum()), outer_eligible_scored=int(elig.sum())),
        lightgbm=dict(params=params, rounds_trained=cap, trees_built=trees_built, skipped_rounds=skipped,
                      full_model_text_bytes=model_bytes),
        es_curve=dict(points=len(curve), all_max=float(curve['es_all'].max()),
                      all_argmax_round=int(curve.loc[curve['es_all'].idxmax(), 'iteration']),
                      eligible_max=float(curve['es_eligible'].max()),
                      eligible_argmax_round=int(curve.loc[curve['es_eligible'].idxmax(), 'iteration'])),
        outer_curve=dict(points=len(points), step_trees=OUTER_STEP, peak_trees=int(peak['trees']),
                         peak_eligible=float(peak['eligible']), last_trees=int(last),
                         last_eligible=one_shot_elig, last_all_sample=reference_all['mean_rank_ic'],
                         segment_vs_one_shot_max_abs=max_abs, eligible_days=len(elig_days)),
        stage3_consistency=stage3,
        timing_seconds=dict(**timer.records, total=round(time.perf_counter() - started, 1),
                            seconds_per_100_rounds=per_100,
                            mean_seconds_per_100_rounds=float(np.mean(per_100)) if per_100 else None),
        peak_memory_gb=common.peak_rss_gb(), peak_memory_gb_after_lgbm_train=peak_after_train,
        machine=dict(platform=platform.platform(), processor=platform.processor(), threads=threads,
                     lightgbm=lgb.__version__),
        test_used=False)
    common.save_json(output / 'summary.json', summary)  # written last: marks a completed fit
    print(f'[{year} {lr_tag(lr)} l2={l2:g}] outer peak {peak["eligible"]:.5f} at {int(peak["trees"])} trees; '
          f'last {one_shot_elig:.5f} at {last}; seg-vs-oneshot {max_abs:.2e}; total={summary["timing_seconds"]["total"]:.0f}s '
          f'peak={summary["peak_memory_gb"]:.2f}GB', flush=True)
    return summary


def stage3_consistency(year, curve, tpr, booster, x_out):
    """lr 0.02 is the stage-3 baseline trained further: its curve and trees must match the stage-3 fit."""
    ref_dir = s3.OUT / f'{year}_{STAGE3_BASE}'
    ref = pd.read_parquet(ref_dir / 'early_stop_curve.parquet')
    ref_summary = common.read_json(ref_dir / 'summary.json')['lightgbm']
    joined = ref.merge(curve, on='iteration', how='inner', validate='one_to_one')
    if len(joined) != len(ref):
        raise RuntimeError('Stage-3 curve iterations are not all present in the stage-4 curve')
    equal = joined['early_stop_rank_ic'].to_numpy() == joined['es_all'].to_numpy()
    first = joined['iteration'] <= VERIFY_ROUNDS
    result = dict(stage3_rounds=int(ref_summary['rounds_trained']), compared_points=len(joined),
                  identical_points=int(equal.sum()),
                  first_1000_rounds_identical=bool(equal[first.to_numpy()].all()), first_1000_points=int(first.sum()),
                  max_abs_difference=float(np.max(np.abs(joined['early_stop_rank_ic'] - joined['es_all']))),
                  stage3_trees_built=int(ref_summary['trees_built']),
                  stage4_trees_at_stage3_rounds=int(tpr[ref_summary['rounds_trained'] - 1]),
                  stage3_best_model_trees=int(ref_summary['best_model_trees']),
                  stage4_trees_at_stage3_best=int(tpr[ref_summary['best_iteration'] - 1]))
    best_trees = ref_summary['best_model_trees']
    pred = predict_chunks(booster, x_out, num_iteration=best_trees)
    ref_pred = pd.read_parquet(ref_dir / 'outer_predictions.parquet')['lgbm'].to_numpy()
    result['outer_prediction_max_abs_difference_at_stage3_best'] = float(np.max(np.abs(pred - ref_pred)))
    ok = (result['identical_points'] == len(joined) and result['stage3_trees_built'] == result['stage4_trees_at_stage3_rounds']
          and result['stage3_best_model_trees'] == result['stage4_trees_at_stage3_best']
          and result['outer_prediction_max_abs_difference_at_stage3_best'] == 0.0)
    result['all_identical'] = bool(ok)
    print(f'[stage3 check {year}] {result}', flush=True)
    if not result['first_1000_rounds_identical'] or not ok:
        raise RuntimeError(f'lr 0.02 does not reproduce the stage-3 baseline for {year}: {result}')
    return result


def run_job(year, lr, threads, l2=P3_L2):
    output = fit_dir(year, lr, l2)
    if completed(output):
        return
    print(f'[start] {output.name} at {time.strftime("%m-%d %H:%M:%S")}', flush=True)
    code = subprocess.call([sys.executable, '-u', __file__, '--one', str(year), f'{lr:g}', '--l2', f'{l2:g}',
                            '--threads', str(threads)])
    if code != 0:
        raise SystemExit(f'{output.name} failed with exit code {code}; stopping')


# ---------------------------------------------------------------- offline rules (step 2)
class Fit:
    def __init__(self, year, lr, l2=P3_L2):
        d = fit_dir(year, lr, l2)
        self.year, self.lr, self.cap, self.l2 = year, lr, LRS[lr], l2
        self.summary = common.read_json(d / 'summary.json')
        self.es = pd.read_parquet(d / 'es_curve.parquet')
        self.tpr = np.load(d / 'trees_per_round.npy')
        self.daily_elig = pd.read_parquet(d / 'outer_daily_eligible.parquet')
        self.daily_all = pd.read_parquet(d / 'outer_daily_all.parquet')
        self.points = np.array([int(c) for c in self.daily_elig.columns])
        self.elig_curve = self.daily_elig.mean(skipna=True).to_numpy()
        self.all_curve = self.daily_all.mean(skipna=True).to_numpy()

    def nearest_point(self, trees):
        i = int(np.argmin(np.abs(self.points - trees)))
        return int(self.points[i]), int(abs(self.points[i] - trees))

    def round_for_trees(self, trees):
        return int(np.searchsorted(self.tpr, trees) + 1)  # first round with >= trees trees

    def daily_at(self, point):
        e = self.daily_elig[str(point)].rename('rank_ic').reset_index()
        a = self.daily_all[str(point)].rename('rank_ic').reset_index()
        return e.assign(year=self.year), a.assign(year=self.year)


def patience(lr, factor=1.0):
    return BASE_PATIENCE * BASE_LR / lr * factor


def stop_rule(iterations, values, pat, delta=0.0):
    """Same logic as RankICEarlyStopping.after: improvement > best + delta resets; stop when it - best >= patience."""
    best, best_it = -np.inf, 0
    for it, v in zip(iterations, values):
        if np.isfinite(v) and v > best + delta:
            best, best_it = v, it
        elif it - best_it >= pat:
            return int(best_it), True
    return int(best_it), False


def select_round(fit, rule):
    it = fit.es['iteration'].to_numpy()
    if rule == 'R1':
        return stop_rule(it, fit.es['es_all'].to_numpy(), patience(fit.lr))
    if rule == 'R2':
        return stop_rule(it, fit.es['es_all'].to_numpy(), patience(fit.lr, 0.5))
    if rule == 'R3':
        return stop_rule(it, fit.es['es_all'].to_numpy(), patience(fit.lr, 2.0))
    if rule == 'R4':
        smooth = fit.es['es_all'].rolling(SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean().to_numpy()
        return stop_rule(it, smooth, patience(fit.lr), SMOOTH_DELTA)
    if rule == 'R5':
        return stop_rule(it, fit.es['es_eligible'].to_numpy(), patience(fit.lr))
    raise ValueError(rule)


def common_grid(fits):
    """100-tree points present in every fold (partial last points excluded)."""
    grid = set(int(p) for p in fits[0].points if p % OUTER_STEP == 0)
    for f in fits[1:]:
        grid &= set(int(p) for p in f.points if p % OUTER_STEP == 0)
    return np.array(sorted(grid))


def mean_curve(fits, grid):
    return np.mean([[f.elig_curve[np.flatnonzero(f.points == g)[0]] for g in grid] for f in fits], axis=0)


def evaluate_rule(fits, rule):
    """Per fold: selected round, trees, outer point; then fold scores and concatenated daily series."""
    rows, elig_parts, all_parts = [], [], []
    grid = common_grid(fits) if rule == 'R6' else None
    for f in fits:
        if rule == 'R6':
            others = [g for g in fits if g.year != f.year]
            curve = mean_curve(others, grid)
            point = int(grid[int(np.argmax(curve))])
            trees, rounding = point, 0
            rnd, stopped = f.round_for_trees(point), None
        else:
            rnd, stopped = select_round(f, rule)
            if rnd <= 0:
                raise RuntimeError(f'{f.year} lr {f.lr} {rule}: no round selected')
            trees = int(f.tpr[rnd - 1])
            point, rounding = f.nearest_point(trees)
        e, a = f.daily_at(point)
        elig_parts.append(e)
        all_parts.append(a)
        rows.append(dict(year=f.year, round=rnd, trees=trees, point=point, rounding_trees=rounding,
                         stopped_before_cap=stopped, truncated=bool(rnd >= TRUNCATION * f.cap),
                         eligible=float(e['rank_ic'].mean()), all_sample=float(a['rank_ic'].mean())))
    elig = np.array([r['eligible'] for r in rows])
    full = np.array([r['all_sample'] for r in rows])
    return dict(rule=rule, folds=rows, eligible_score=float(elig.mean() - PENALTY * elig.std(ddof=0)),
                all_score=float(full.mean() - PENALTY * full.std(ddof=0)),
                negative_fold=bool((elig < 0).any()), truncated=any(r['truncated'] for r in rows),
                max_rounding_trees=max(r['rounding_trees'] for r in rows),
                daily=pd.concat(elig_parts, ignore_index=True), daily_all=pd.concat(all_parts, ignore_index=True))


def unified_rounds(fits):
    grid = common_grid(fits)
    curve = mean_curve(fits, grid)
    point = int(grid[int(np.argmax(curve))])
    return dict(trees=point, mean_eligible=float(curve.max()),
                rounds_by_fold={f.year: f.round_for_trees(point) for f in fits},
                truncated=any(f.round_for_trees(point) >= TRUNCATION * f.cap for f in fits))


def compare(new, ref):
    """Selection-score difference and NW t of concatenated daily eligible Rank IC differences (new - ref)."""
    diff, t = s3.nw(new['daily'], ref['daily'])
    gain = new['eligible_score'] - ref['eligible_score']
    return dict(score_diff=float(gain), daily_mean_diff=diff, t=t, passes=bool(gain >= MIN_GAIN and t >= MIN_T))


def admissible(result):
    return not result['truncated'] and not result['negative_fold']


def analyze(threads=8, run_recheck=True):
    fits = {lr: [Fit(y, lr) for y in YEARS] for lr in LRS}
    results = {(lr, rule): evaluate_rule(fits[lr], rule) for lr in LRS for rule in RULES}
    unified = {lr: unified_rounds(fits[lr]) for lr in LRS}
    comparisons = []

    # step 3.3: learning rate under R1 (larger lr kept unless a smaller one passes the threshold)
    current = None
    for lr in LR_ORDER:
        r = results[(lr, 'R1')]
        if not admissible(r):
            comparisons.append(dict(step='lr (R1)', candidate=f'{lr:g}+R1', reference=None, admissible=False,
                                    note='truncated or negative fold: excluded'))
            continue
        if current is None:
            current = lr
            comparisons.append(dict(step='lr (R1)', candidate=f'{lr:g}+R1', reference=None, admissible=True,
                                    note='largest admissible lr: starting choice'))
            continue
        c = compare(r, results[(current, 'R1')])
        comparisons.append(dict(step='lr (R1)', candidate=f'{lr:g}+R1', reference=f'{current:g}+R1', admissible=True,
                                **c, note='adopted' if c['passes'] else 'not adopted'))
        if c['passes']:
            current = lr
    if current is None:
        raise SystemExit('No admissible learning rate under R1')
    chosen_lr = current

    # step 3.4: early-stopping rule at the chosen lr
    ref = results[(chosen_lr, 'R1')]
    qualifying = []
    for rule in RULES[1:]:
        r = results[(chosen_lr, rule)]
        if not admissible(r):
            comparisons.append(dict(step=f'rule @ lr {chosen_lr:g}', candidate=f'{chosen_lr:g}+{rule}',
                                    reference=f'{chosen_lr:g}+R1', admissible=False,
                                    note='truncated or negative fold: excluded'))
            continue
        c = compare(r, ref)
        comparisons.append(dict(step=f'rule @ lr {chosen_lr:g}', candidate=f'{chosen_lr:g}+{rule}',
                                reference=f'{chosen_lr:g}+R1', admissible=True, **c,
                                note='qualifies' if c['passes'] else 'not adopted'))
        if c['passes']:
            qualifying.append(rule)
    chosen_rule = max(qualifying, key=lambda q: results[(chosen_lr, q)]['eligible_score']) if qualifying else 'R1'
    p4 = (chosen_lr, chosen_rule)
    base = results[(BASE_LR, BASE_RULE)]
    p4_vs_base = compare(results[p4], base) if p4 != (BASE_LR, BASE_RULE) else None

    # step 3.5: conditional recheck of (lambda1 x1, lambda2 1000) at the P4 learning rate
    recheck = None
    if chosen_lr != BASE_LR:
        if run_recheck:
            for year in YEARS:
                run_job(year, chosen_lr, threads, RECHECK_L2)
        alt = evaluate_rule([Fit(y, chosen_lr, RECHECK_L2) for y in YEARS], chosen_rule)
        recheck = dict(config=f'lambda1 x1, lambda2 {RECHECK_L2} @ lr {chosen_lr:g} + {chosen_rule}', result=alt,
                       vs_p4=compare(alt, results[p4]))

    decision = dict(p4=dict(learning_rate=chosen_lr, rule=chosen_rule, rule_text=RULE_TEXT[chosen_rule],
                            cap=LRS[chosen_lr], eligible_score=results[p4]['eligible_score'],
                            rounds_by_fold={r['year']: r['round'] for r in results[p4]['folds']},
                            unified_rounds_r6=unified[chosen_lr] if chosen_rule == 'R6' else None),
                    baseline=dict(learning_rate=BASE_LR, rule=BASE_RULE, eligible_score=base['eligible_score']),
                    p4_vs_baseline=p4_vs_base, qualifying_rules=qualifying)
    return dict(fits=fits, results=results, unified=unified, comparisons=comparisons, decision=decision,
                recheck=recheck)


# ---------------------------------------------------------------- step 4: summary
def f4(x, d=4):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def strip(result):
    return {k: v for k, v in result.items() if k not in ('daily', 'daily_all')}


def summarize(state):
    fits, results, unified, decision = state['fits'], state['results'], state['unified'], state['decision']
    OUT.mkdir(parents=True, exist_ok=True)
    # lr-level costs
    lr_rows = []
    for lr in LRS:
        s = [f.summary for f in fits[lr]]
        lr_rows.append(dict(learning_rate=lr, cap=LRS[lr],
                            train_seconds=[round(x['timing_seconds']['train']) for x in s],
                            train_seconds_total=round(sum(x['timing_seconds']['train'] for x in s)),
                            total_seconds=round(sum(x['timing_seconds']['total'] for x in s)),
                            mean_seconds_per_100_rounds=float(np.mean([x['timing_seconds']['mean_seconds_per_100_rounds']
                                                                       for x in s])),
                            peak_memory_gb=max(x['peak_memory_gb'] for x in s),
                            skipped_rounds=[x['lightgbm']['skipped_rounds'] for x in s],
                            trees_built=[x['lightgbm']['trees_built'] for x in s],
                            model_mb=[round(x['lightgbm']['full_model_text_bytes'] / 1e6) for x in s]))
    lr_table = pd.DataFrame(lr_rows)
    lr_table.to_csv(OUT / 'lr_costs.csv', index=False, encoding='utf-8')

    rule_rows, fold_rows = [], []
    for (lr, rule), r in results.items():
        rule_rows.append(dict(learning_rate=lr, rule=rule, eligible_score=r['eligible_score'], all_score=r['all_score'],
                              **{f'eligible_{x["year"]}': x['eligible'] for x in r['folds']},
                              rounds=[x['round'] for x in r['folds']], trees=[x['trees'] for x in r['folds']],
                              truncated=r['truncated'], negative_fold=r['negative_fold'],
                              max_rounding_trees=r['max_rounding_trees']))
        for x in r['folds']:
            fold_rows.append(dict(learning_rate=lr, rule=rule, **x))
    rule_table = pd.DataFrame(rule_rows)
    rule_table.to_csv(OUT / 'rules_table.csv', index=False, encoding='utf-8')
    pd.DataFrame(fold_rows).to_csv(OUT / 'rules_by_fold.csv', index=False, encoding='utf-8')
    comp = pd.DataFrame(state['comparisons'])
    comp.to_csv(OUT / 'comparisons.csv', index=False, encoding='utf-8')

    curve_rows = []
    for lr in LRS:
        r1 = {x['year']: x for x in results[(lr, 'R1')]['folds']}
        for f in fits[lr]:
            i = int(np.argmax(f.elig_curve))
            curve_rows.append(dict(learning_rate=lr, year=f.year, peak_trees=int(f.points[i]),
                                   peak_round=f.round_for_trees(int(f.points[i])), peak_eligible=float(f.elig_curve[i]),
                                   r1_round=r1[f.year]['round'], r1_trees=r1[f.year]['trees'],
                                   r1_eligible=r1[f.year]['eligible'], peak_minus_r1=float(f.elig_curve[i]) - r1[f.year]['eligible'],
                                   last_trees=int(f.points[-1]), last_eligible=float(f.elig_curve[-1])))
    curve_table = pd.DataFrame(curve_rows)
    curve_table.to_csv(OUT / 'outer_curve_summary.csv', index=False, encoding='utf-8')
    # outer curves for all fits in one long file (plots / later checks)
    long = []
    for lr in LRS:
        for f in fits[lr]:
            long.append(pd.DataFrame(dict(learning_rate=lr, year=f.year, trees=f.points, eligible=f.elig_curve,
                                          all_sample=f.all_curve)))
    pd.concat(long).to_csv(OUT / 'outer_curves_long.csv', index=False, encoding='utf-8')

    payload = dict(decision=decision, unified_r6={f'{k:g}': v for k, v in unified.items()},
                   comparisons=state['comparisons'],
                   recheck=None if state['recheck'] is None else dict(
                       config=state['recheck']['config'], result=strip(state['recheck']['result']),
                       vs_p4=state['recheck']['vs_p4']))
    common.save_json(OUT / 'p4.json', json.loads(json.dumps(payload, default=lambda o: o.item() if hasattr(o, 'item') else str(o))))

    # markdown tables (the narrative doc is written separately from these)
    lines = ['## 各学习率的耗时、内存、跳过轮次', '',
             '| lr | 上限 | 训练秒（各折） | 训练合计 | 每 100 轮秒 | 峰值内存 GB | 跳过轮次 | 树数 | 模型 MB |',
             '|---:|---:|---|---:|---:|---:|---|---|---|']
    for r in lr_rows:
        lines.append(f'| {r["learning_rate"]:g} | {r["cap"]} | {r["train_seconds"]} | {r["train_seconds_total"]} | '
                     f'{r["mean_seconds_per_100_rounds"]:.1f} | {r["peak_memory_gb"]:.2f} | {r["skipped_rounds"]} | '
                     f'{r["trees_built"]} | {r["model_mb"]} |')
    lines += ['', '## lr × 规则（资格池选择分数；截断 = 选中轮数在任一折 ≥ 0.95 × 上限）', '',
              '| lr | 规则 | ' + ' | '.join(str(y) for y in YEARS) + ' | 资格池分数 | 全样本分数 | 每折选中轮数 | 每折树数 | 截断 | 负折 | 取整误差最大（树） |',
              '|---:|---|' + '---:|' * len(YEARS) + '---:|---:|---|---|---|---|---:|']
    for r in rule_rows:
        lines.append(f'| {r["learning_rate"]:g} | {r["rule"]} | ' + ' | '.join(f4(r[f"eligible_{y}"]) for y in YEARS) +
                     f' | **{f4(r["eligible_score"])}** | {f4(r["all_score"])} | {r["rounds"]} | {r["trees"]} | '
                     f'{"是" if r["truncated"] else "否"} | {"是" if r["negative_fold"] else "否"} | {r["max_rounding_trees"]} |')
    lines += ['', '## R6 五折统一轮数（全部五折外层平均曲线峰值，供冻结时使用）', '',
              '| lr | 树数 | 五折平均资格池 Rank IC | 各折对应轮数 | 截断 |', '|---:|---:|---:|---|---|']
    for lr, u in unified.items():
        lines.append(f'| {lr:g} | {u["trees"]} | {f4(u["mean_eligible"])} | {list(u["rounds_by_fold"].values())} | '
                     f'{"是" if u["truncated"] else "否"} |')
    lines += ['', '## 选择过程的两两比较（分数差 = 资格池选择分数差；门槛：≥ 0.001 且 NW t ≥ 2）', '',
              '| 步骤 | 候选 | 对照 | 可入选 | 分数差 | 日度均值差 | NW t | 结论 |', '|---|---|---|---|---:|---:|---:|---|']
    for c in state['comparisons']:
        lines.append(f'| {c["step"]} | {c["candidate"]} | {c.get("reference") or ""} | {"是" if c["admissible"] else "否"} | '
                     f'{f4(c.get("score_diff"), 5)} | {f4(c.get("daily_mean_diff"), 5)} | {f4(c.get("t"), 2)} | {c["note"]} |')
    lines += ['', '## 外层曲线摘要（资格池；峰值在每 100 棵树的网格上）', '',
              '| lr | 折 | 峰值树数 | 峰值轮数 | 峰值 | R1 轮数 | R1 树数 | R1 处的值 | 峰值 − R1 | 最后一点树数 | 最后一点值 |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in curve_rows:
        lines.append(f'| {r["learning_rate"]:g} | {r["year"]} | {r["peak_trees"]} | {r["peak_round"]} | {f4(r["peak_eligible"])} | '
                     f'{r["r1_round"]} | {r["r1_trees"]} | {f4(r["r1_eligible"])} | {f4(r["peak_minus_r1"])} | '
                     f'{r["last_trees"]} | {f4(r["last_eligible"])} |')
    (OUT / 'stage4_tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    d = decision
    print(f'[P4] lr {d["p4"]["learning_rate"]:g} + {d["p4"]["rule"]} score {d["p4"]["eligible_score"]:.5f} '
          f'(baseline lr 0.02 + R1 {d["baseline"]["eligible_score"]:.5f}); vs baseline {d["p4_vs_baseline"]}', flush=True)
    if state['recheck']:
        rc = state['recheck']
        print(f'[recheck] {rc["config"]}: score {rc["result"]["eligible_score"]:.5f} trunc={rc["result"]["truncated"]} '
              f'vs P4 {rc["vs_p4"]}', flush=True)


# ---------------------------------------------------------------- pipeline
def run_all(threads):
    for year in YEARS:
        es_eligible(year)
    run_job(VERIFY_YEAR, BASE_LR, threads)  # determinism gate first (fit_one asserts against stage 3)
    check = common.read_json(fit_dir(VERIFY_YEAR, BASE_LR) / 'summary.json')['stage3_consistency']
    if not check or not check['first_1000_rounds_identical']:
        raise SystemExit('Determinism check against stage 3 failed; stopping')
    print(f'[gate] lr 0.02 {VERIFY_YEAR} reproduces stage 3: {check}', flush=True)
    for lr in (BASE_LR, 0.05, 0.03, 0.01):
        for year in YEARS:
            run_job(year, lr, threads)
    state = analyze(threads)
    summarize(state)
    print('[done] stage 4 round complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--one', nargs=2, metavar=('YEAR', 'LR'))
    parser.add_argument('--l2', type=float, default=P3_L2)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--es-eligible', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.one:
        year, lr = int(args.one[0]), float(args.one[1])
        if year not in YEARS or lr not in LRS:
            raise ValueError('Not a stage-4 fit')
        fit_one(year, lr, fit_dir(year, lr, s3.num(args.l2)), args.threads, s3.num(args.l2))
        return
    if args.es_eligible:
        for year in YEARS:
            es_eligible(year)
    if args.analyze:
        summarize(analyze(args.threads))
    if args.all:
        run_all(args.threads)


if __name__ == '__main__':
    main()
