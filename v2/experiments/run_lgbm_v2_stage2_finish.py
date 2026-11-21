"""Stage 2 wrap-up (plan section 13 items 1 and 4): compressed fine search, mechanical choice of P*, path_smooth.

  python v2/experiments/run_lgbm_v2_stage2_finish.py            fine search -> P* (with the (200, 1000) rule) -> path_smooth

Same fit settings as stage 2 (cap 6000, patience 200, all-round models), via run_lgbm_v2_stage2 --one.
P* rule (deep 0.1% only):
  1. candidates = every lambda combination run for deep 0.1% (path_smooth 0), minus negative-fold or truncated ones
  2. T = highest eligible selection score
  3. plateau = candidates with |NW t vs T| < 2 (T included), eligible daily Rank IC, five folds concatenated
  4. pick the plateau member with the fewest five-fold mean leaves; members within 5% of that leaf count are
     tied and the one with the larger lambda2 wins (then fewer leaves)
  5. neighbourhood: its adjacent run combinations on the grid (nearest lambda1 at the same lambda2, nearest
     lambda2 at the same lambda1) must all be in the plateau; otherwise fall back to the next-fewest leaves
  6. if the pick has lambda1 = 100, run (200, 1000) once and redo 1-5; no further extension
path_smooth: compare 0 and 1000 on P*; adopt 1000 only if NW t (1000 minus 0) >= 2.
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import time

import numpy as np
import pandas as pd

import lgbm_v2_common as common
from run_lgbm_v2_stage2 import OUT, YEARS, combo_label, combo_scores, completed, fit_dir, run_job

STRUCTURE, LEAF = 'deep', '0.1pct'
FINE = [(100, 1000), (50, 300), (50, 3000)]
EXTRA = (200, 1000)
PATH_SMOOTH = 1000


def daily(l1, l2, ps=0):
    parts = []
    for y in YEARS:
        d = pd.read_parquet(fit_dir(y, STRUCTURE, LEAF, l1, l2, ps) / 'eligible_daily_metrics.parquet')
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        parts.append(d.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def nw(a, b):
    joined = a.merge(b, on=['year', 'date'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(joined) != len(a) or len(joined) != len(b):
        raise RuntimeError('Incomplete daily pairing')
    diff = (joined['rank_ic_a'] - joined['rank_ic_b']).to_numpy()
    return float(diff.mean()), common.newey_west_tstat(diff, lag=5), len(joined)


def run_combo(l1, l2, threads, ps=0):
    for year in YEARS:
        run_job(year, STRUCTURE, LEAF, l1, l2, threads, ps=ps)


def run_combos():
    combos = []
    for y_dir in sorted(OUT.glob(f'2017_{STRUCTURE}_{LEAF}_l1-*_l2-*')):
        if '_ps-' in y_dir.name:
            continue
        l1 = float(y_dir.name.split('_l1-')[1].split('_l2-')[0])
        l2 = float(y_dir.name.split('_l2-')[1])
        l1, l2 = (int(l1) if l1.is_integer() else l1), (int(l2) if l2.is_integer() else l2)
        if all(completed(fit_dir(y, STRUCTURE, LEAF, l1, l2)) for y in YEARS):
            combos.append((l1, l2))
    return sorted(combos)


def neighbours(choice, combos):
    l1, l2 = choice
    out = []
    same_l2 = sorted(a for a, b in combos if b == l2)
    same_l1 = sorted(b for a, b in combos if a == l1)
    i, j = same_l2.index(l1), same_l1.index(l2)
    if i > 0: out.append((same_l2[i - 1], l2))
    if i < len(same_l2) - 1: out.append((same_l2[i + 1], l2))
    if j > 0: out.append((l1, same_l1[j - 1]))
    if j < len(same_l1) - 1: out.append((l1, same_l1[j + 1]))
    return out


def choose():
    combos = run_combos()
    table = pd.DataFrame([combo_scores(STRUCTURE, LEAF, a, b) for a, b in combos])
    table['label'] = [combo_label(STRUCTURE, LEAF, a, b) for a, b in combos]
    cand = table.loc[~table['negative_fold'] & ~table['truncated']].copy()
    if not len(cand):
        raise RuntimeError('No admissible deep 0.1% combination')
    top = cand.sort_values('eligible_score', ascending=False).iloc[0]
    base = daily(top['l1'], top['l2'])
    stats = []
    for _, r in table.iterrows():
        if (r['l1'], r['l2']) == (top['l1'], top['l2']):
            stats.append((0.0, 0.0))
        else:
            d, t, _ = nw(base, daily(r['l1'], r['l2']))
            stats.append((d, t))
    table['mean_diff_T_minus'] = [s[0] for s in stats]
    table['t_vs_T'] = [s[1] for s in stats]
    table['candidate'] = ~table['negative_fold'] & ~table['truncated']
    table['in_plateau'] = table['candidate'] & (table['t_vs_T'].abs() < 2)
    plateau = table.loc[table['in_plateau']].copy()
    plateau_set = {(r['l1'], r['l2']) for _, r in plateau.iterrows()}

    # rule 4 ordering: fewest leaves; members within 5% of the minimum are tied -> larger lambda2 first
    order, remaining = [], plateau.copy()
    while len(remaining):
        least = remaining['mean_leaves'].min()
        tied = remaining.loc[remaining['mean_leaves'] <= least * 1.05]
        pick = tied.sort_values(['l2', 'mean_leaves'], ascending=[False, True]).iloc[0]
        order.append(pick)
        remaining = remaining.loc[~((remaining['l1'] == pick['l1']) & (remaining['l2'] == pick['l2']))]
    checks, chosen = [], None
    for pick in order:
        nb = neighbours((pick['l1'], pick['l2']), combos)
        missing = [n for n in nb if n not in plateau_set]
        checks.append(dict(combo=pick['label'], mean_leaves=pick['mean_leaves'], neighbours=nb,
                           neighbours_outside_plateau=missing))
        if not missing:
            chosen = pick
            break
    if chosen is None:
        raise RuntimeError('No plateau member passes the neighbourhood check')
    return table, top, plateau, chosen, checks


def path_smooth_test(l1, l2, threads):
    run_combo(l1, l2, threads, ps=PATH_SMOOTH)
    diff, t, days = nw(daily(l1, l2, PATH_SMOOTH), daily(l1, l2))
    with_ps = combo_scores(STRUCTURE, LEAF, l1, l2, ps=PATH_SMOOTH)
    without = combo_scores(STRUCTURE, LEAF, l1, l2)
    return dict(days=days, mean_diff_ps_minus_0=diff, newey_west_t_lag5=t, adopt=bool(t >= 2),
                eligible_score_ps=with_ps['eligible_score'], eligible_score_0=without['eligible_score'],
                all_score_ps=with_ps['all_score'], all_score_0=without['all_score'],
                truncated_ps=with_ps['truncated'], negative_fold_ps=with_ps['negative_fold'],
                eligible_by_year_ps={y: with_ps[f'eligible_{y}'] for y in YEARS},
                best_iterations_ps=with_ps['best_iterations'], mean_leaves_ps=with_ps['mean_leaves'])


def report(table, top, plateau, chosen, checks, rounds, ps):
    f = lambda x, d=4: f'{x:.{d}f}'
    p_star = combo_label(STRUCTURE, LEAF, chosen['l1'], chosen['l2'])
    L = ['### I1. 深 0.1% 全部 λ 组合（第二阶段网格 + 压缩精搜）与 P* 机械规则', '',
         f"T（候选中资格池分数最高）= **{combo_label(STRUCTURE, LEAF, top['l1'], top['l2'])}**；"
         f"平台 = 与 T 的 |NW t| < 2（含 T）；**P* = {p_star}**。", '',
         '| 组合 | 2017 | 2018 | 2019 | 2020 | 2021 | 资格池分数 | 全样本分数 | 五折平均叶子 | 负折 | 截断 | 对 T 平均差 | 对 T 的 t | 平台 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|']
    for _, r in table.sort_values('eligible_score', ascending=False).iterrows():
        L.append(f"| {r['label']} | " + ' | '.join(f(r[f'eligible_{y}']) for y in YEARS) +
                 f" | **{f(r['eligible_score'])}** | {f(r['all_score'])} | {r['mean_leaves']:.1f} | "
                 f"{'是' if r['negative_fold'] else '否'} | {'是' if r['truncated'] else '否'} | "
                 f"{f(r['mean_diff_T_minus'], 5)} | {f(r['t_vs_T'], 2)} | {'是' if r['in_plateau'] else ''} |")
    L += ['', '规则 4–5 的检查顺序（叶子最少 → 5% 内并列取 λ2 更大 → 邻域检查）：', '']
    for c in checks:
        L.append(f"- {c['combo']}：平均叶子 {c['mean_leaves']:.1f}；网格邻居 {c['neighbours']}；"
                 f"不在平台的邻居 {c['neighbours_outside_plateau'] or '无'}"
                 + (' → **选中**' if c['combo'] == p_star else ' → 退到下一个'))
    L += ['', f"精搜轮次记录：{rounds}", '',
          f"### I2. path_smooth：P* 上 1000 对 0（五折）", '',
          f"- 资格池分数：path_smooth = 1000 为 {f(ps['eligible_score_ps'])}，0 为 {f(ps['eligible_score_0'])}；"
          f"全样本分数 {f(ps['all_score_ps'])} 对 {f(ps['all_score_0'])}",
          f"- 逐折资格池 Rank IC（1000）：" + '、'.join(f"{y} {f(v)}" for y, v in ps['eligible_by_year_ps'].items()),
          f"- best_iteration（1000）：{'/'.join(map(str, ps['best_iterations_ps']))}；截断 {'是' if ps['truncated_ps'] else '否'}；"
          f"平均叶子 {ps['mean_leaves_ps']:.1f}",
          f"- 日度差（1000 − 0）均值 {f(ps['mean_diff_ps_minus_0'], 5)}，NW t = {f(ps['newey_west_t_lag5'], 2)}（{ps['days']} 天）",
          f"- 结论：**{'采用 path_smooth = 1000' if ps['adopt'] else '不采用，保持 path_smooth = 0'}**（规则：t ≥ 2 才采用）"]
    text = '\n'.join(L) + '\n'
    (OUT / 'p_star.md').write_text(text, encoding='utf-8')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    rounds = []
    for l1, l2 in FINE:
        run_combo(l1, l2, args.threads)
    table, top, plateau, chosen, checks = choose()
    rounds.append(dict(pick=combo_label(STRUCTURE, LEAF, chosen['l1'], chosen['l2'])))
    if chosen['l1'] == 100:
        print('[extend] P* has lambda1 = 100 -> running (200, 1000) once', flush=True)
        run_combo(*EXTRA, args.threads)
        table, top, plateau, chosen, checks = choose()
        rounds.append(dict(after_extra_200=combo_label(STRUCTURE, LEAF, chosen['l1'], chosen['l2'])))
    print(f"[P*] {combo_label(STRUCTURE, LEAF, chosen['l1'], chosen['l2'])} (T = {combo_label(STRUCTURE, LEAF, top['l1'], top['l2'])}, "
          f"plateau {len(plateau)})", flush=True)
    ps = path_smooth_test(chosen['l1'], chosen['l2'], args.threads)
    print(f"[path_smooth] t = {ps['newey_west_t_lag5']:.2f} adopt = {ps['adopt']}", flush=True)
    table.to_csv(OUT / 'p_star_table.csv', index=False, encoding='utf-8')
    common.save_json(OUT / 'p_star.json', dict(
        p_star=dict(structure=STRUCTURE, leaf=LEAF, l1=chosen['l1'], l2=chosen['l2'],
                    path_smooth=PATH_SMOOTH if ps['adopt'] else 0),
        T=combo_label(STRUCTURE, LEAF, top['l1'], top['l2']),
        plateau=[combo_label(STRUCTURE, LEAF, r['l1'], r['l2']) for _, r in plateau.iterrows()],
        neighbourhood_checks=checks, rounds=rounds, path_smooth=ps,
        decided_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    report(table, top, plateau, chosen, checks, rounds, ps)
    print('[done] stage 2 wrap-up complete', flush=True)


if __name__ == '__main__':
    main()
