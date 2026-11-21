"""Summarize V2 LightGBM stage 2 (M choice, lambda small grid, extensions) and run the round-curve diagnostic.

Outputs in quant_data/processed/v2/lgbm_stage2/:
  fits_table.csv            one row per fit (combination x fold)
  m_table.csv               medium candidates, five folds, eligible and all-sample scores
  grid_table.csv            every (structure, lambda) combination: per-fold eligible Rank IC and scores
  winner_pairwise_nw.csv    winner vs every other combination, eligible daily Rank IC, NW t lag 5
  round_curves.csv          per-100-round eligible Rank IC on the outer year (diagnostic only)
  stage2_tables.md          markdown tables for v2/docs/LGBM_STAGE2_RESULTS.md
Selection uses the eligible-pool score (plan section 12); nothing here changes a selection.
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import gzip

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from run_lgbm_v2_stage1 import LEAF_FRACTIONS
from run_lgbm_v2_stage1 import OUT as STAGE1
from run_lgbm_v2_stage2 import (C, FRESH_CELLS, OUT, YEARS, combo_label, combo_scores, eligible_metrics,
                                fit_dir, paired_t, winner)

f = lambda x, d=4: '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def fit_row(year, structure, leaf, l1, l2):
    s = common.read_json(fit_dir(year, structure, leaf, l1, l2) / 'summary.json')
    m, d, t = s['outer_metrics'], s['diagnostics'], s['diagnostics']['trees']
    return dict(combo=combo_label(structure, leaf, l1, l2), structure=structure, leaf=leaf, l1=l1, l2=l2, year=year,
                source=s['source'], eligible_rank_ic=s['eligible_metrics']['mean_rank_ic'],
                eligible_ic=s['eligible_metrics']['mean_ic'], all_rank_ic=m['mean_rank_ic'],
                best_iteration=s['lightgbm']['best_iteration'], rounds_trained=s['lightgbm']['rounds_trained'],
                truncated=s['lightgbm']['truncated'], mean_leaves=t['mean_leaves'], max_leaves=t['max_leaves'],
                mean_depth=t['mean_depth'], fit_minus_es=d['fit_minus_early_stop_rank_ic'],
                early_stop_rank_ic=s['lightgbm']['early_stop_best_rank_ic'],
                top50_minus_pool=m['top50_minus_pool_5d_simple_return'], top50_overlap=m['consecutive_day_top50_overlap'],
                t_vs_elastic_eligible=s['lgbm_vs_elastic']['eligible']['newey_west_t_lag5'],
                diff_vs_elastic_eligible=s['lgbm_vs_elastic']['eligible']['mean_rank_ic_difference'],
                t_vs_elastic_all=s['lgbm_vs_elastic']['all_sample']['newey_west_t_lag5'],
                train_seconds=s['timing_seconds']['train'], total_seconds=s['timing_seconds']['total'],
                peak_memory_gb=s['peak_memory_gb'])


def eligible_daily(year, structure, leaf, l1, l2):
    d = pd.read_parquet(fit_dir(year, structure, leaf, l1, l2) / 'eligible_daily_metrics.parquet')
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    return d.assign(year=year)


def round_curve(year, structure, leaf, l1, l2, step=100):
    """Eligible-pool outer Rank IC after every `step` rounds, from the all-round model."""
    import lightgbm as lgb
    directory = fit_dir(year, structure, leaf, l1, l2)
    path = directory / 'lightgbm_model_all_rounds.txt.gz'
    if not path.exists():
        return None
    with gzip.open(path, 'rt', encoding='utf-8') as h:
        booster = lgb.Booster(model_str=h.read())
    info = common.read_json(directory / 'best_iteration.json')
    best = info.get('best_model_trees', info['best_iteration'])  # curve is indexed by trees; equal unless rounds were skipped
    data_dir = cache.cache_dir(year, 'full')
    outer_x = np.load(data_dir / 'outer_x.npy', mmap_mode='r')
    outer = pd.read_parquet(data_dir / 'outer_meta.parquet')
    total = booster.num_trees()
    raw = np.zeros(len(outer))
    rows = []
    for start in range(0, total, step):
        n = min(step, total - start)
        for s in range(0, len(outer_x), 200_000):
            raw[s:s + 200_000] += booster.predict(np.asarray(outer_x[s:s + 200_000]), start_iteration=start,
                                                  num_iteration=n)
        m, _ = eligible_metrics(outer.assign(p=raw), 'p')
        rows.append(dict(combo=combo_label(structure, leaf, l1, l2), year=year, rounds=start + n,
                         eligible_rank_ic=m['mean_rank_ic'], best_iteration=best, rounds_trained=total))
    curve = pd.DataFrame(rows)
    # sanity: the curve at best_iteration must match the stored early-stopped metric (when best is on the grid)
    stored = common.read_json(directory / 'summary.json')['eligible_metrics']['mean_rank_ic']
    at_best = curve.loc[curve['rounds'] == best, 'eligible_rank_ic']
    if len(at_best) and abs(float(at_best.iat[0]) - stored) > 1e-9:
        raise RuntimeError(f'{directory.name}: curve at best_iteration {at_best.iat[0]} != stored {stored}')
    return curve


def summarize(curves=True):
    state = common.read_json(OUT / 'grid_state.json')
    m_leaf = common.read_json(OUT / 'm_selection.json')['m_leaf']
    structures = [('medium', m_leaf), C]
    combos = [(s, leaf, a, b) for s, leaf in structures for a in state['l1'][s] for b in state['l2'][s]]

    # medium candidates (lambda = 0) and the grid
    m_table = pd.DataFrame([combo_scores('medium', leaf, 0, 0) for leaf in LEAF_FRACTIONS]).sort_values(
        'eligible_score', ascending=False)
    m_table.to_csv(OUT / 'm_table.csv', index=False, encoding='utf-8')
    grid = pd.DataFrame([combo_scores(*c) for c in combos])
    grid['combo'] = [combo_label(*c) for c in combos]
    grid = grid.sort_values('eligible_score', ascending=False).reset_index(drop=True)
    grid['rank'] = np.arange(1, len(grid) + 1)
    grid['all_rank'] = grid['all_score'].rank(ascending=False, method='min').astype(int)

    fits = pd.DataFrame([fit_row(y, *c) for c in combos for y in YEARS]
                        + [fit_row(y, 'medium', leaf, 0, 0) for leaf in LEAF_FRACTIONS if leaf != m_leaf
                           for y in YEARS])
    fits.to_csv(OUT / 'fits_table.csv', index=False, encoding='utf-8')
    agg = fits.groupby('combo').agg(mean_leaves=('mean_leaves', 'mean'), max_leaves=('max_leaves', 'max'),
                                    mean_depth=('mean_depth', 'mean'), fit_minus_es=('fit_minus_es', 'mean'),
                                    top50_minus_pool=('top50_minus_pool', 'mean'),
                                    top50_overlap=('top50_overlap', 'mean'),
                                    mean_t_vs_elastic=('t_vs_elastic_eligible', 'mean'),
                                    min_t_vs_elastic=('t_vs_elastic_eligible', 'min'),
                                    train_seconds=('train_seconds', 'mean'), total_seconds=('total_seconds', 'sum'))
    grid = grid.merge(agg, left_on='combo', right_index=True, how='left')
    grid.to_csv(OUT / 'grid_table.csv', index=False, encoding='utf-8')

    win = winner(grid)
    win_label = combo_label(win['structure'], win['leaf'], win['l1'], win['l2'])
    base = pd.concat([eligible_daily(y, win['structure'], win['leaf'], win['l1'], win['l2']) for y in YEARS])
    pairs = []
    for c in combos:
        if combo_label(*c) == win_label:
            continue
        other = pd.concat([eligible_daily(y, *c) for y in YEARS])
        joined = base.merge(other, on=['year', 'date'], suffixes=('_w', '_o'), validate='one_to_one')
        if len(joined) != len(base):
            raise RuntimeError('Incomplete daily pairing in winner comparison')
        diff = (joined['rank_ic_w'] - joined['rank_ic_o']).to_numpy()
        pairs.append(dict(winner=win_label, other=combo_label(*c), days=len(joined), mean_diff=float(diff.mean()),
                          newey_west_t_lag5=common.newey_west_tstat(diff, lag=5)))
    pairs = pd.DataFrame(pairs).sort_values('newey_west_t_lag5')
    pairs.to_csv(OUT / 'winner_pairwise_nw.csv', index=False, encoding='utf-8')

    # complexity diagnostic (plan stage 2 item 4): C vs M at lambda = 0, five folds
    def lam0(structure, leaf):
        return fits.loc[(fits['structure'] == structure) & (fits['leaf'] == leaf) & (fits['l1'] == 0) & (fits['l2'] == 0)]
    cm, mm = lam0(*C), lam0('medium', m_leaf)
    complexity = dict(c_mean_leaves=float(cm['mean_leaves'].mean()), m_mean_leaves=float(mm['mean_leaves'].mean()),
                      ratio=float(cm['mean_leaves'].mean() / mm['mean_leaves'].mean()),
                      c_mean_depth=float(cm['mean_depth'].mean()), m_mean_depth=float(mm['mean_depth'].mean()),
                      per_year_ratio={int(y): float(cm.set_index('year').loc[y, 'mean_leaves'] /
                                                    mm.set_index('year').loc[y, 'mean_leaves']) for y in YEARS})
    complexity['passed'] = complexity['ratio'] >= 3 and complexity['c_mean_depth'] > complexity['m_mean_depth']

    # reuse basis check: cells refitted fresh at a 6000 cap vs their reusable stage-1 fits
    reuse_check = []
    for (year, structure, leaf, l1, l2) in sorted(FRESH_CELLS):
        s2 = fit_dir(year, structure, leaf, l1, l2) / 'summary.json'
        s1 = STAGE1 / f'{year}_{structure}_{leaf}' / 'summary.json'
        if s2.exists() and s1.exists():
            a, b = common.read_json(s2), common.read_json(s1)
            reuse_check.append(dict(cell=f'{year}_{structure}_{leaf}', stage1_rounds=b['lightgbm']['rounds_trained'],
                                    stage2_rounds=a['lightgbm']['rounds_trained'],
                                    stage1_best=b['lightgbm']['best_iteration'], stage2_best=a['lightgbm']['best_iteration'],
                                    rank_ic_abs_diff=abs(a['outer_metrics']['mean_rank_ic'] - b['outer_metrics']['mean_rank_ic'])))
    reuse_check = pd.DataFrame(reuse_check)

    # round-curve diagnostic: 2020 and 2021, C and M at lambda = 0 and at the winner's lambda
    curve_frames = []
    if curves:
        lams = [(0, 0)] + ([(win['l1'], win['l2'])] if (win['l1'], win['l2']) != (0, 0) else [])
        for year in (2020, 2021):
            for structure, leaf in structures:
                for a, b in lams:
                    if a in state['l1'][structure] and b in state['l2'][structure]:
                        c = round_curve(year, structure, leaf, a, b)
                        if c is not None:
                            curve_frames.append(c)
                            print(f'[curve] {year} {combo_label(structure, leaf, a, b)} done', flush=True)
        if curve_frames:
            pd.concat(curve_frames).to_csv(OUT / 'round_curves.csv', index=False, encoding='utf-8')

    # ---------------- markdown
    L = ['## A. M 的五折选择表（λ = 0，上限 6000）', '',
         '| 中结构配置 | 2017 | 2018 | 2019 | 2020 | 2021 | 资格池分数 | 全样本分数 | 有负折 | 截断 | best_iteration |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    for _, r in m_table.iterrows():
        L.append(f"| medium_{r['leaf']} | " + ' | '.join(f(r[f'eligible_{y}']) for y in YEARS) +
                 f" | **{f(r['eligible_score'])}** | {f(r['all_score'])} | {'是' if r['negative_fold'] else '否'} | "
                 f"{'是' if r['truncated'] else '否'} | {'/'.join(map(str, r['best_iterations']))} |")
    L += ['', '全样本 Rank IC（同一批拟合）：', '', '| 中结构配置 | ' + ' | '.join(map(str, YEARS)) + ' | 全样本分数 |',
          '|---|' + '---:|' * (len(YEARS) + 1)]
    for _, r in m_table.iterrows():
        L.append(f"| medium_{r['leaf']} | " + ' | '.join(f(r[f'all_{y}']) for y in YEARS) + f" | {f(r['all_score'])} |")
    L += ['', f"## B. 复杂度诊断：C = deep_0.1pct vs M = medium_{m_leaf}（λ = 0，五折）", '',
          f"- C 实际平均叶子 {complexity['c_mean_leaves']:.1f}，M {complexity['m_mean_leaves']:.1f}，比值 "
          f"**{complexity['ratio']:.2f}**（逐折 " + '、'.join(f"{y}: {v:.2f}" for y, v in complexity['per_year_ratio'].items()) + '）',
          f"- 平均深度 C {complexity['c_mean_depth']:.2f}，M {complexity['m_mean_depth']:.2f}",
          f"- “≥ 3 倍且更深”：{'满足' if complexity['passed'] else '**不满足**'}", '']
    L += [f'## C. λ 网格（含外延）五折资格池 Rank IC 与分数；胜出组合：**{win_label}**', '',
          '| 排名 | 组合 | 2017 | 2018 | 2019 | 2020 | 2021 | 资格池分数 | 全样本分数 | 全样本排名 | 负折 | 截断 |',
          '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for _, r in grid.iterrows():
        L.append(f"| {r['rank']} | {r['combo']} | " + ' | '.join(f(r[f'eligible_{y}']) for y in YEARS) +
                 f" | **{f(r['eligible_score'])}** | {f(r['all_score'])} | {r['all_rank']} | "
                 f"{'是' if r['negative_fold'] else '否'} | {'是' if r['truncated'] else '否'} |")
    L += ['', f'## D. 胜出组合（{win_label}）对其余组合：资格池日度 Rank IC 差（五折拼接）NW t（lag 5）', '',
          '| 对比组合 | 天数 | 平均差 | NW t |', '|---|---:|---:|---:|']
    for _, r in pairs.iterrows():
        L.append(f"| {r['other']} | {r['days']} | {f(r['mean_diff'], 5)} | {f(r['newey_west_t_lag5'], 2)} |")
    L += ['', '## E. 每个组合的轮数、复杂度、头部指标与耗时（五折平均；best_iteration 按 2017–2021）', '',
          '| 组合 | best_iteration | 截断 | 平均叶 | 最大叶 | 平均深 | 内部−早停 IC | Top50−池 | Top50 重合 | 对 Elastic t（资格池，均值/最小） | 平均训练秒 | 合计秒 |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for _, r in grid.iterrows():
        L.append(f"| {r['combo']} | {'/'.join(map(str, r['best_iterations']))} | {'是' if r['truncated'] else '否'} | "
                 f"{r['mean_leaves']:.1f} | {r['max_leaves']} | {r['mean_depth']:.2f} | {f(r['fit_minus_es'])} | "
                 f"{f(r['top50_minus_pool'], 5)} | {f(r['top50_overlap'], 3)} | {f(r['mean_t_vs_elastic'], 2)} / "
                 f"{f(r['min_t_vs_elastic'], 2)} | {r['train_seconds']:.0f} | {r['total_seconds']:.0f} |")
    if len(reuse_check):
        L += ['', '## F. 复用依据的实证核对（6000 上限重新拟合 vs 第一阶段 3000 上限）', '',
              '| 格子 | 第一阶段轮数/best | 第二阶段轮数/best | 全样本 Rank IC 绝对差 |', '|---|---|---|---:|']
        for _, r in reuse_check.iterrows():
            L.append(f"| {r['cell']} | {r['stage1_rounds']}/{r['stage1_best']} | {r['stage2_rounds']}/{r['stage2_best']} | "
                     f"{r['rank_ic_abs_diff']:.2e} |")
    if curve_frames:
        curves_df = pd.concat(curve_frames)
        L += ['', '## G. 诊断：外层年份每 100 轮资格池 Rank IC（全轮数模型；只诊断，不用于选择）', '',
              '| 折 | 组合 | 早停 best | 训练轮数 | best 处 | 曲线最高（轮数） | 100 | 300 | 500 | 1000 | 2000 | 3000 | 末轮 |',
              '|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for (combo, year), c in curves_df.groupby(['combo', 'year']):
            c = c.set_index('rounds')['eligible_rank_ic']
            best = int(curves_df.loc[(curves_df['combo'] == combo) & (curves_df['year'] == year), 'best_iteration'].iat[0])
            at = lambda k: f(float(c.loc[k])) if k in c.index else ''
            nearest = c.index[np.argmin(np.abs(c.index - best))]
            L.append(f"| {year} | {combo} | {best} | {c.index.max()} | {f(float(c.loc[nearest]))} (≈{nearest}) | "
                     f"{f(float(c.max()))} ({int(c.idxmax())}) | {at(100)} | {at(300)} | {at(500)} | {at(1000)} | "
                     f"{at(2000)} | {at(3000)} | {f(float(c.iloc[-1]))} |")
    text = '\n'.join(L) + '\n'
    (OUT / 'stage2_tables.md').write_text(text, encoding='utf-8')
    common.save_json(OUT / 'stage2_summary.json', dict(winner=win_label, m_leaf=m_leaf, complexity=complexity,
                                                       grid_state=state))
    print(text)
    return grid, win_label, complexity


if __name__ == '__main__':
    import sys
    summarize(curves='--no-curves' not in sys.argv)
