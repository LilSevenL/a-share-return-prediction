"""V2 LightGBM stage 6a: training label and loss (plan section 7 stage 6 A-D, revised in section 13).

On P* (stage-2 wrap-up result, including the path_smooth decision), raw factor inputs, five folds:
  A  L2 + raw return                  (P* stage-2 fits, reused)
  B  Huber + raw return               alpha = 1.35 x MAD of the fold's inner-fit raw labels
  C  L2 + daily cross-sectional demeaned return
  D  L2 + daily cross-sectional rank return, (average rank - 0.5) / valid stocks that day, in (0, 1)
C and D daily statistics use every stock with a finite raw label on that signal date (targets table, not
only sampled rows), signal-date cross-section only; every label used has exit_date < outer-year 01-01
(asserted). Early stopping and every score stay on the raw label (eligible daily Rank IC for selection).
lambda2 unchanged. lambda1 start: B = A; C, D = lambda1 x std(transformed)/std(raw) on the fold's inner fit
set, rounded to 2 significant digits, per fold.
Round 1: B, C, D at the start lambda1. lambda1 review: the best non-A scheme plus non-A schemes with
|NW t| < 2 against it (at most 2 in total) also run lambda1 x 0.5 and x 2. Each scheme keeps its best
lambda1 (no negative fold, not truncated). Recommendation: adopt the best non-A scheme only if its NW t
against A is >= 2; otherwise keep A. The recommendation is reported, never acted on.

  python v2/experiments/run_lgbm_v2_stage6a.py                  labels -> round 1 -> lambda1 review -> summary
  python v2/experiments/run_lgbm_v2_stage6a.py --one YEAR SCHEME MULT
  python v2/experiments/run_lgbm_v2_stage6a.py --summarize
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import math
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from lgbm_v2_common import DATA
from run_lgbm_v2_stage2 import (OUT as STAGE2, PENALTY, YEARS, combo_label, completed, eligible_metrics,
                                fit_dir as stage2_dir, fit_one, truncated)

OUT = DATA / 'v2' / 'lgbm_stage6a'
LABELS = OUT / 'labels'
SCHEMES = ('A', 'B', 'C', 'D')
NAMES = {'A': 'L2 + 原始收益', 'B': 'Huber + 原始收益', 'C': 'L2 + 每日截面去均值收益', 'D': 'L2 + 每日截面排名收益'}
REVIEW_MULTS = (0.5, 2.0)


def p_star():
    p = common.read_json(STAGE2 / 'p_star.json')['p_star']
    num = lambda v: (lambda f: int(f) if f.is_integer() else f)(float(v))  # numpy ints were saved as strings
    return p['structure'], p['leaf'], num(p['l1']), num(p['l2']), num(p['path_smooth'])


def sig2(x):
    if x == 0:
        return 0.0
    return float(round(x, 1 - int(math.floor(math.log10(abs(x))))))


# ---------------------------------------------------------------- labels
def build_labels(year):
    """Demeaned (C) and rank (D) training labels for the fold's inner fit set, plus the Huber alpha (B)."""
    path = LABELS / f'{year}.json'
    if path.exists():
        return common.read_json(path)
    LABELS.mkdir(parents=True, exist_ok=True)
    directory = cache.cache_dir(year, 'full')
    keys = pd.read_parquet(directory / 'fit_keys.parquet')
    keys['signal_date'] = pd.to_datetime(keys['signal_date']).dt.normalize()
    y_fit = np.load(directory / 'fit_y.npy')
    first, last = keys['signal_date'].min(), keys['signal_date'].max()
    con = common.open_connection(OUT, 8, '6GB')
    common.create_views(con, year)
    targets = con.execute(f"""
        SELECT ts_code, CAST(signal_date AS DATE) signal_date, CAST(exit_date AS DATE) exit_date,
               target_logret_5d_raw y
        FROM targets
        WHERE isfinite(target_logret_5d_raw)
          AND CAST(signal_date AS DATE) >= DATE '{first.date()}' AND CAST(signal_date AS DATE) <= DATE '{last.date()}'
    """).fetch_df()
    con.close()
    targets['signal_date'] = pd.to_datetime(targets['signal_date']).dt.normalize()
    exit_dates = pd.to_datetime(targets['exit_date'])
    outer = pd.Timestamp(f'{year}-01-01')
    assert exit_dates.max() < outer, f'{year}: a label used for daily statistics exits on/after {outer.date()}'
    if targets.duplicated(['ts_code', 'signal_date']).any():
        raise RuntimeError('Duplicate target keys')
    grouped = targets.groupby('signal_date')['y']
    targets['n_day'] = grouped.transform('size')
    targets['demeaned'] = targets['y'] - grouped.transform('mean')
    targets['rank'] = (grouped.rank(method='average') - 0.5) / targets['n_day']
    joined = keys.merge(targets, on=['ts_code', 'signal_date'], how='left', validate='one_to_one')
    if joined['y'].isna().any():
        raise RuntimeError(f'{year}: {int(joined["y"].isna().sum())} inner-fit keys missing from targets')
    if not np.array_equal(joined['y'].to_numpy(), y_fit):
        raise RuntimeError(f'{year}: raw labels from targets differ from the cached inner-fit labels')
    demeaned = joined['demeaned'].to_numpy(np.float64)
    rank = joined['rank'].to_numpy(np.float64)
    assert np.isfinite(demeaned).all() and np.isfinite(rank).all()
    assert rank.min() > 0 and rank.max() < 1
    np.save(LABELS / f'{year}_C.npy', demeaned)
    np.save(LABELS / f'{year}_D.npy', rank)
    raw_std = float(y_fit.std())
    mad = float(np.median(np.abs(y_fit - np.median(y_fit))))
    info = dict(year=year, inner_fit_rows=len(y_fit), daily_stat_rows=len(targets),
                daily_stat_days=int(targets['signal_date'].nunique()),
                daily_stat_signal_range=[str(first.date()), str(last.date())],
                max_exit_date_used=str(exit_dates.max().date()),
                rows_with_exit_in_early_stop_year=int((exit_dates >= pd.Timestamp(f'{year - 1}-01-01')).sum()),
                median_stocks_per_day=float(targets.groupby('signal_date').size().median()),
                sampled_share_of_daily_rows=len(y_fit) / len(targets),
                raw_std=raw_std, demeaned_std=float(demeaned.std()), rank_std=float(rank.std()),
                ratio_C=float(demeaned.std()) / raw_std, ratio_D=float(rank.std()) / raw_std,
                raw_median=float(np.median(y_fit)), raw_mad=mad, huber_alpha=1.35 * mad,
                fit_keys_hash=cache.key_hash(keys, y_fit))
    common.save_json(path, info)
    print(f'[labels {year}] rows {len(targets):,} on {info["daily_stat_days"]} days; ratio C {info["ratio_C"]:.3f} '
          f'D {info["ratio_D"]:.3f}; huber alpha {info["huber_alpha"]:.5f}; max exit {info["max_exit_date_used"]}', flush=True)
    return info


def scheme_setup(year, scheme, mult):
    structure, leaf, l1, l2, ps = p_star()
    if l1 == 0:
        raise RuntimeError('P* has lambda1 = 0: the plan switches to {0, 2, 10} in transformed units; not implemented, stop')
    info = build_labels(year)
    extra = dict(path_smooth=float(ps)) if ps else {}
    label, label_info = None, dict(scheme=scheme, name=NAMES[scheme])
    if scheme == 'B':
        start = float(l1)
        extra.update(objective='huber', alpha=info['huber_alpha'])
        label_info.update(huber_alpha=info['huber_alpha'], raw_mad=info['raw_mad'])
    elif scheme in ('C', 'D'):
        ratio = info[f'ratio_{scheme}']
        start = sig2(l1 * ratio)
        label = np.load(LABELS / f'{year}_{scheme}.npy')
        label_info.update(std_ratio=ratio, label_file=f'labels/{year}_{scheme}.npy')
    else:
        raise ValueError(scheme)
    lam1 = start * mult
    label_info.update(lambda1_start=start, lambda1_multiplier=mult, lambda1=lam1, lambda2=l2,
                      p_star=combo_label(structure, leaf, l1, l2, ps))
    return structure, leaf, lam1, l2, extra, label, label_info


def fit_dir(year, scheme, mult):
    return OUT / f'{year}_{scheme}_x{mult:g}'


def a_dir(year):
    structure, leaf, l1, l2, ps = p_star()
    return stage2_dir(year, structure, leaf, l1, l2, ps)


def run_one(year, scheme, mult, threads):
    structure, leaf, lam1, l2, extra, label, label_info = scheme_setup(year, scheme, mult)
    lam1 = int(lam1) if float(lam1).is_integer() else lam1
    fit_one(year, structure, leaf, lam1, l2, fit_dir(year, scheme, mult), threads, extra=extra or None,
            train_label=label, label_info=label_info, stage='6a', tag=f'6a-{scheme}-x{mult:g} (lambda1 {lam1:g})')


def run_job(year, scheme, mult, threads):
    output = fit_dir(year, scheme, mult)
    if completed(output):
        return
    print(f'[start] {output.name} at {time.strftime("%H:%M:%S")}', flush=True)
    code = subprocess.call([sys.executable, '-u', __file__, '--one', str(year), scheme, str(mult),
                            '--threads', str(threads)])
    if code != 0:
        raise SystemExit(f'{output.name} failed with exit code {code}; stopping')


# ---------------------------------------------------------------- scoring
def directory(year, scheme, mult):
    return a_dir(year) if scheme == 'A' else fit_dir(year, scheme, mult)


def scores(scheme, mult):
    rows = [common.read_json(directory(y, scheme, mult) / 'summary.json') for y in YEARS]
    elig = np.array([r['eligible_metrics']['mean_rank_ic'] for r in rows])
    full = np.array([r['outer_metrics']['mean_rank_ic'] for r in rows])
    best = [r['lightgbm']['best_iteration'] for r in rows]
    lam1 = [r['lightgbm']['params']['lambda_l1'] for r in rows]
    return dict(scheme=scheme, mult=mult, label=f'{scheme} x{mult:g}',
                **{f'eligible_{y}': v for y, v in zip(YEARS, elig)},
                eligible_score=float(elig.mean() - PENALTY * elig.std(ddof=0)), eligible_mean=float(elig.mean()),
                **{f'all_{y}': v for y, v in zip(YEARS, full)},
                all_score=float(full.mean() - PENALTY * full.std(ddof=0)),
                negative_fold=bool((elig < 0).any()), truncated=any(truncated(b) for b in best),
                best_iterations=best, lambda1_by_year=lam1)


def daily(scheme, mult):
    parts = []
    for y in YEARS:
        d = pd.read_parquet(directory(y, scheme, mult) / 'eligible_daily_metrics.parquet')
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        parts.append(d.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def nw(a, b):
    joined = a.merge(b, on=['year', 'date'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(joined) != len(a) or len(joined) != len(b):
        raise RuntimeError('Incomplete daily pairing')
    diff = (joined['rank_ic_a'] - joined['rank_ic_b']).to_numpy()
    return float(diff.mean()), common.newey_west_tstat(diff, lag=5)


def admissible(s):
    return not s['negative_fold'] and not s['truncated']


# ---------------------------------------------------------------- driver
def run_all(threads):
    structure, leaf, l1, l2, ps = p_star()
    print(f'[P*] {combo_label(structure, leaf, l1, l2, ps)}', flush=True)
    for year in YEARS:
        build_labels(year)
    for scheme in ('B', 'C', 'D'):
        for year in YEARS:
            run_job(year, scheme, 1.0, threads)
    round1 = {s: scores(s, 1.0) for s in ('B', 'C', 'D')}
    ranked = sorted(round1.values(), key=lambda s: (admissible(s), s['eligible_score']), reverse=True)
    top = ranked[0]
    review = [top['scheme']]
    for s in ranked[1:]:
        if len(review) >= 2:
            break
        _, t = nw(daily(top['scheme'], 1.0), daily(s['scheme'], 1.0))
        if abs(t) < 2:
            review.append(s['scheme'])
    print(f'[review] best non-A round 1: {top["scheme"]}; lambda1 review for {review}', flush=True)
    common.save_json(OUT / 'review_decision.json', dict(round1_best=top['scheme'], review=review,
                                                        round1={k: v['eligible_score'] for k, v in round1.items()}))
    for scheme in review:
        for mult in REVIEW_MULTS:
            for year in YEARS:
                run_job(year, scheme, mult, threads)
    summarize()
    print('[done] stage 6a complete', flush=True)


# ---------------------------------------------------------------- summary
def fit_rows(scheme, mult):
    out = []
    for y in YEARS:
        s = common.read_json(directory(y, scheme, mult) / 'summary.json')
        m, d, t = s['outer_metrics'], s['diagnostics'], s['diagnostics']['trees']
        out.append(dict(scheme=scheme, mult=mult, label=f'{scheme} x{mult:g}', year=y,
                        lambda1=s['lightgbm']['params']['lambda_l1'], lambda2=s['lightgbm']['params']['lambda_l2'],
                        objective=s['lightgbm']['params'].get('objective'), alpha=s['lightgbm']['params'].get('alpha'),
                        eligible_rank_ic=s['eligible_metrics']['mean_rank_ic'], all_rank_ic=m['mean_rank_ic'],
                        best_iteration=s['lightgbm']['best_iteration'], truncated=s['lightgbm']['truncated'],
                        mean_leaves=t['mean_leaves'], mean_depth=t['mean_depth'],
                        fit_minus_es=d['fit_minus_early_stop_rank_ic'],
                        top50_minus_pool=m['top50_minus_pool_5d_simple_return'],
                        top50_overlap=m['consecutive_day_top50_overlap'],
                        timing_variance_share=m['timing_split']['timing_variance_share'],
                        timing_correlation=m['timing_split']['timing_correlation'],
                        t_vs_elastic_eligible=s['lgbm_vs_elastic']['eligible']['newey_west_t_lag5'],
                        train_seconds=s['timing_seconds']['train'], total_seconds=s['timing_seconds']['total']))
    return out


def summarize():
    review = common.read_json(OUT / 'review_decision.json')
    variants = [('A', 1.0)] + [(s, 1.0) for s in ('B', 'C', 'D')] + \
               [(s, m) for s in review['review'] for m in REVIEW_MULTS]
    table = pd.DataFrame([scores(s, m) for s, m in variants])
    fits = pd.DataFrame([r for s, m in variants for r in fit_rows(s, m)])
    fits.to_csv(OUT / 'fits_table.csv', index=False, encoding='utf-8')
    agg = fits.groupby('label').agg(mean_leaves=('mean_leaves', 'mean'), mean_depth=('mean_depth', 'mean'),
                                    fit_minus_es=('fit_minus_es', 'mean'), top50_minus_pool=('top50_minus_pool', 'mean'),
                                    top50_overlap=('top50_overlap', 'mean'),
                                    timing_variance_share=('timing_variance_share', 'mean'),
                                    timing_correlation=('timing_correlation', 'mean'),
                                    total_seconds=('total_seconds', 'sum'))
    table = table.merge(agg, left_on='label', right_index=True)
    table.to_csv(OUT / 'variant_table.csv', index=False, encoding='utf-8')

    best = {}
    for scheme in SCHEMES:
        rows = table.loc[table['scheme'] == scheme]
        ok = rows.loc[~rows['negative_fold'] & ~rows['truncated']]
        pick = (ok if len(ok) else rows).sort_values('eligible_score', ascending=False).iloc[0]
        best[scheme] = (pick['mult'], bool(len(ok) > 0))
    pair = []
    for i, a in enumerate(SCHEMES):
        for b in SCHEMES[i + 1:]:
            d, t = nw(daily(a, best[a][0]), daily(b, best[b][0]))
            pair.append(dict(a=f'{a} x{best[a][0]:g}', b=f'{b} x{best[b][0]:g}', mean_diff_a_minus_b=d, newey_west_t_lag5=t))
    pair = pd.DataFrame(pair)
    pair.to_csv(OUT / 'pairwise_nw.csv', index=False, encoding='utf-8')
    non_a = [(s, best[s][0]) for s in ('B', 'C', 'D') if best[s][1]]
    best_non_a = max(non_a, key=lambda sm: float(table.loc[(table['scheme'] == sm[0]) & (table['mult'] == sm[1]), 'eligible_score'].iat[0])) \
        if non_a else None
    if best_non_a:
        d, t = nw(daily(*best_non_a), daily('A', 1.0))
        recommendation = dict(best_non_a=f'{best_non_a[0]} x{best_non_a[1]:g}', mean_diff_vs_A=d, t_vs_A=t,
                              adopt=bool(t >= 2), recommendation=(f'采用方案 {best_non_a[0]}' if t >= 2 else '保留方案 A'))
    else:
        recommendation = dict(best_non_a=None, recommendation='保留方案 A（无可入选的非 A 方案）', adopt=False)
    labels = {y: common.read_json(LABELS / f'{y}.json') for y in YEARS}
    common.save_json(OUT / 'stage6a_summary.json', dict(p_star=combo_label(*p_star()[:4], p_star()[4]),
                                                        best_lambda1_multiplier={k: v[0] for k, v in best.items()},
                                                        recommendation=recommendation, review=review))

    f = lambda x, d=4: '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'
    L = [f"P* = **{combo_label(*p_star()[:4], p_star()[4])}**；早停与评分始终用原始收益的日度 Rank IC；选择口径为资格池。", '',
         '## A. 各方案的五折资格池 Rank IC、选择分数与全样本分数', '',
         '| 方案 | λ1 倍数 | 逐折 λ1 | 2017 | 2018 | 2019 | 2020 | 2021 | 资格池分数 | 全样本分数 | 负折 | 截断 |',
         '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for _, r in table.iterrows():
        L.append(f"| {r['scheme']} {NAMES[r['scheme']]} | {r['mult']:g} | {'/'.join(f'{v:g}' for v in r['lambda1_by_year'])} | "
                 + ' | '.join(f(r[f'eligible_{y}']) for y in YEARS) +
                 f" | **{f(r['eligible_score'])}** | {f(r['all_score'])} | {'是' if r['negative_fold'] else '否'} | "
                 f"{'是' if r['truncated'] else '否'} |")
    L += ['', '## B. 方案两两比较（各方案取最佳 λ1；资格池日度 Rank IC 差，五折拼接，NW t lag 5）', '',
          '| 方案 a | 方案 b | 平均差（a − b） | NW t |', '|---|---|---:|---:|']
    for _, r in pair.iterrows():
        L.append(f"| {r['a']} | {r['b']} | {f(r['mean_diff_a_minus_b'], 5)} | {f(r['newey_west_t_lag5'], 2)} |")
    L += ['', '## C. 各方案对同折 Elastic Net 的配对 t（资格池，逐折）', '',
          '| 方案 | 2017 | 2018 | 2019 | 2020 | 2021 |', '|---|---:|---:|---:|---:|---:|']
    for label, g in fits.groupby('label', sort=False):
        L.append(f"| {label} | " + ' | '.join(f(v, 2) for v in g.set_index('year').reindex(YEARS)['t_vs_elastic_eligible']) + ' |')
    L += ['', '## D. 轮数、复杂度、头部与择时诊断（五折平均；best_iteration 按 2017–2021）', '',
          '| 方案 | best_iteration | 截断 | 平均叶 | 平均深 | 内部−早停 IC | Top50−池 | Top50 重合 | 择时方差占比 | 日均预测与日均收益相关 | 合计秒 |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in table.iterrows():
        L.append(f"| {r['label']} | {'/'.join(map(str, r['best_iterations']))} | {'是' if r['truncated'] else '否'} | "
                 f"{r['mean_leaves']:.1f} | {r['mean_depth']:.2f} | {f(r['fit_minus_es'])} | {f(r['top50_minus_pool'], 5)} | "
                 f"{f(r['top50_overlap'], 3)} | {f(r['timing_variance_share'], 3)} | {f(r['timing_correlation'], 3)} | "
                 f"{r['total_seconds']:.0f} |")
    L += ['', '## E. λ1 复核与各折换算比例', '',
          f"第一轮（倍数 1）资格池分数：" + '、'.join(f"{k} {f(v)}" for k, v in review['round1'].items()) +
          f"；最佳非 A 方案 {review['round1_best']}；做 λ1 × {{0.5, 2}} 复核的方案：{'、'.join(review['review'])}。",
          f"各方案取的 λ1 倍数（去掉负折与截断后分数最高）：" + '、'.join(f"{k} ×{v[0]:g}" for k, v in best.items()), '',
          '| 折 | 原始标签 std | 去均值 std | 排名 std | 比例 C | 比例 D | Huber alpha (1.35×MAD) | 日度统计行数 | 抽样占比 | 最大退出日 | 退出日落在早停年份的行数 |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for y, i in labels.items():
        L.append(f"| {y} | {f(i['raw_std'], 5)} | {f(i['demeaned_std'], 5)} | {f(i['rank_std'], 5)} | {f(i['ratio_C'], 3)} | "
                 f"{f(i['ratio_D'], 3)} | {f(i['huber_alpha'], 5)} | {i['daily_stat_rows']:,} | {f(i['sampled_share_of_daily_rows'], 3)} | "
                 f"{i['max_exit_date_used']} | {i['rows_with_exit_in_early_stop_year']:,} |")
    L += ['', '## F. 按采用规则的建议（只报告，由评审确认）', '']
    if recommendation.get('best_non_a'):
        L.append(f"- 最佳非 A 方案：{recommendation['best_non_a']}；对 A 的资格池日度差 {f(recommendation['mean_diff_vs_A'], 5)}，"
                 f"NW t = {f(recommendation['t_vs_A'], 2)}。")
    L.append(f"- 规则：最佳非 A 方案对 A 的 NW t ≥ 2 才采用。**建议：{recommendation['recommendation']}**。")
    text = '\n'.join(L) + '\n'
    (OUT / 'stage6a_tables.md').write_text(text, encoding='utf-8')
    print(text)
    return table, recommendation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--one', nargs=3, metavar=('YEAR', 'SCHEME', 'MULT'))
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--labels-only', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.one:
        year, scheme, mult = int(args.one[0]), args.one[1], float(args.one[2])
        if year not in YEARS or scheme not in ('B', 'C', 'D'):
            raise ValueError('Not a stage 6a fit')
        run_one(year, scheme, mult, args.threads)
    elif args.labels_only:
        for year in YEARS:
            build_labels(year)
    elif args.summarize:
        summarize()
    else:
        run_all(args.threads)


if __name__ == '__main__':
    main()
