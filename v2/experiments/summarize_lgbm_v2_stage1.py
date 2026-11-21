"""Summarize V2 LightGBM stage 1 (27 fits). Reports only; selects nothing for stage 2.

Outputs in quant_data/processed/v2/lgbm_stage1/:
  summary_table.csv   one row per fit (config x outer year)
  config_table.csv    one row per configuration: per-year Rank IC, selection score (plan section 8)
  pairwise_nw.csv     top-score configuration vs every other: daily Rank IC difference, NW t (lag 5)
  stage1_tables.md    markdown tables used in v2/docs/LGBM_STAGE1_RESULTS.md
Selection score = mean of yearly mean daily Rank IC - 0.25 x std of yearly Rank IC, std with ddof=0
(same convention as tune_elastic_net.py / tune_lightgbm_structure.py); the ddof=1 score is also listed.
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import numpy as np
import pandas as pd

import lgbm_v2_common as common
from run_lgbm_v2_stage1 import LEAF_FRACTIONS, OUT, STRUCTURES, YEARS, config_name

PENALTY = 0.25


def load_rows():
    rows = []
    for year in YEARS:
        for structure in STRUCTURES:
            for leaf in LEAF_FRACTIONS:
                s = common.read_json(OUT / config_name(year, structure, leaf) / 'summary.json')
                m, d, t = s['outer_metrics'], s['diagnostics'], s['diagnostics']['trees']
                rows.append(dict(
                    config=f'{structure}_{leaf}', structure=structure, leaf=leaf, year=year,
                    num_leaves=s['lightgbm']['params']['num_leaves'], max_depth=s['lightgbm']['params']['max_depth'],
                    min_data_in_leaf=s['lightgbm']['params']['min_data_in_leaf'],
                    full_sample_rows=s['rows']['full_sample'], inner_fit_rows=s['rows']['inner_fit'],
                    early_stop_rows=s['rows']['early_stop'], outer_scored_rows=s['rows']['outer_scored'],
                    rank_ic=m['mean_rank_ic'], weighted_rank_ic=m['mean_weighted_rank_ic'], ic=m['mean_ic'],
                    rank_ic_positive_share=m['rank_ic_positive_share'], days=m['days'],
                    best_iteration=s['lightgbm']['best_iteration'], rounds_trained=s['lightgbm']['rounds_trained'],
                    hit_round_limit=s['lightgbm']['hit_round_limit'],
                    early_stop_rank_ic=s['lightgbm']['early_stop_best_rank_ic'],
                    mean_leaves=t['mean_leaves'], max_leaves=t['max_leaves'], mean_depth=t['mean_depth'],
                    max_depth_seen=t['max_depth'], inner_fit_rank_ic=d['inner_fit_rank_ic'],
                    fit_minus_early_stop_rank_ic=d['fit_minus_early_stop_rank_ic'],
                    median_days_per_leaf=float(np.median(d['leaf_coverage']['median_days_per_leaf'])),
                    top50_minus_pool=m['top50_minus_pool_5d_simple_return'],
                    top50_overlap=m['consecutive_day_top50_overlap'],
                    consecutive_day_rank_corr=m['consecutive_day_rank_corr'],
                    timing_variance_share=m['timing_split']['timing_variance_share'],
                    elastic_rank_ic=s['lgbm_vs_elastic']['elastic_mean_rank_ic'],
                    diff_vs_elastic=s['lgbm_vs_elastic']['mean_rank_ic_difference'],
                    nw_t_vs_elastic=s['lgbm_vs_elastic']['newey_west_t_lag5'],
                    train_seconds=s['timing_seconds']['train'], total_seconds=s['timing_seconds']['total'],
                    sec_per_100_rounds=s['timing_seconds']['mean_seconds_per_100_rounds'],
                    peak_memory_gb=s['peak_memory_gb'],
                    top20_features=';'.join(n for n, _ in t['top20_gain'])))
    return pd.DataFrame(rows)


def daily_series(config):
    structure, leaf = config.split('_', 1)
    parts = []
    for year in YEARS:
        d = pd.read_parquet(OUT / config_name(year, structure, leaf) / 'daily_metrics.parquet', columns=['date', 'rank_ic'])
        parts.append(d.assign(year=year, date=pd.to_datetime(d['date']).dt.normalize()))
    return pd.concat(parts, ignore_index=True)


def fmt(x, digits=4):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{digits}f}'


def summarize():
    table = load_rows()
    if len(table) != 27:
        raise RuntimeError(f'Expected 27 completed fits, found {len(table)}')
    table.to_csv(OUT / 'summary_table.csv', index=False, encoding='utf-8')

    configs = []
    for config, group in table.groupby('config', sort=False):
        ric = group.set_index('year')['rank_ic'].reindex(YEARS).to_numpy()
        configs.append(dict(
            config=config, structure=group['structure'].iat[0], leaf=group['leaf'].iat[0],
            **{f'rank_ic_{y}': v for y, v in zip(YEARS, ric)},
            mean_rank_ic=float(ric.mean()), std_rank_ic_ddof0=float(ric.std(ddof=0)),
            score=float(ric.mean() - PENALTY * ric.std(ddof=0)),
            score_ddof1=float(ric.mean() - PENALTY * ric.std(ddof=1)),
            negative_fold=bool((ric < 0).any()),
            mean_weighted_rank_ic=float(group['weighted_rank_ic'].mean()),
            mean_top50_minus_pool=float(group['top50_minus_pool'].mean()),
            mean_top50_overlap=float(group['top50_overlap'].mean()),
            mean_rank_corr=float(group['consecutive_day_rank_corr'].mean()),
            mean_leaves=float(group['mean_leaves'].mean()),
            **{f'leaves_{y}': v for y, v in zip(YEARS, group.set_index('year')['mean_leaves'].reindex(YEARS))},
            mean_depth=float(group['mean_depth'].mean()), max_leaves=int(group['max_leaves'].max()),
            best_iterations='/'.join(str(v) for v in group.set_index('year')['best_iteration'].reindex(YEARS)),
            any_hit_cap=bool(group['hit_round_limit'].any()),
            mean_gap_fit_minus_es=float(group['fit_minus_early_stop_rank_ic'].mean()),
            mean_diff_vs_elastic=float(group['diff_vs_elastic'].mean()),
            mean_train_seconds=float(group['train_seconds'].mean())))
    configs = pd.DataFrame(configs).sort_values('score', ascending=False).reset_index(drop=True)
    configs['rank'] = np.arange(1, len(configs) + 1)
    configs.to_csv(OUT / 'config_table.csv', index=False, encoding='utf-8')

    # Pairwise NW t: top-score eligible configuration (no negative fold, plan section 8 rule 2) vs others.
    eligible = configs.loc[~configs['negative_fold']]
    top = (eligible if len(eligible) else configs)['config'].iat[0]
    base = daily_series(top)
    pairwise = []
    for config in configs['config']:
        if config == top:
            continue
        other = daily_series(config)
        joined = base.merge(other, on=['year', 'date'], suffixes=('_top', '_other'))
        diff = (joined['rank_ic_top'] - joined['rank_ic_other']).to_numpy()
        pairwise.append(dict(top=top, other=config, days=len(joined), mean_diff=float(diff.mean()),
                             newey_west_t_lag5=common.newey_west_tstat(diff, lag=5)))
    pairwise = pd.DataFrame(pairwise)
    pairwise.to_csv(OUT / 'pairwise_nw.csv', index=False, encoding='utf-8')

    # Leaf-count ratio for the stage 2 complexity diagnostic (deep 0.1% vs each medium configuration).
    leaves = table.pivot_table(index='year', columns='config', values='mean_leaves')
    depth = table.pivot_table(index='year', columns='config', values='mean_depth')
    ratio_rows = []
    for leaf in LEAF_FRACTIONS:
        medium = f'medium_{leaf}'
        r = leaves['deep_0.1pct'] / leaves[medium]
        ratio_rows.append(dict(medium=medium, **{f'ratio_{y}': float(r[y]) for y in YEARS},
                               ratio_mean_of_leaves=float(leaves['deep_0.1pct'].mean() / leaves[medium].mean()),
                               deep_depth_mean=float(depth['deep_0.1pct'].mean()),
                               medium_depth_mean=float(depth[medium].mean())))
    ratios = pd.DataFrame(ratio_rows)

    lines = ['## A. 每个配置 × 折的外层日度 Rank IC（按选择分数排序）', '',
             '| 排名 | 配置 | 2017 | 2019 | 2021 | 均值 | 标准差(ddof=0) | 选择分数 | 分数(ddof=1) | 有负折 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for _, r in configs.iterrows():
        lines.append(f"| {r['rank']} | {r['config']} | {fmt(r['rank_ic_2017'])} | {fmt(r['rank_ic_2019'])} | "
                     f"{fmt(r['rank_ic_2021'])} | {fmt(r['mean_rank_ic'])} | {fmt(r['std_rank_ic_ddof0'])} | "
                     f"**{fmt(r['score'])}** | {fmt(r['score_ddof1'])} | {'是' if r['negative_fold'] else '否'} |")
    lines += ['', f'## B. 分数第一（{top}）对其余配置：日度 Rank IC 差（三折拼接）的 Newey-West t（lag 5）', '',
              '| 对比配置 | 天数 | 平均差 | NW t |', '|---|---:|---:|---:|']
    for _, r in pairwise.iterrows():
        lines.append(f"| {r['other']} | {r['days']} | {fmt(r['mean_diff'], 5)} | {fmt(r['newey_west_t_lag5'], 2)} |")
    lines += ['', '## C. 复杂度、轮数与头部指标（三折平均）', '',
              '| 配置 | 平均叶子 2017/2019/2021 | 平均深度 | 最大叶子 | best_iteration 2017/2019/2021 | 撞上限 | '
              '内部拟合−早停 IC 差 | 加权 Rank IC | Top50−池 | 相邻日 Top50 重合 | 对 Elastic 平均差 | 训练秒 |',
              '|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for _, r in configs.iterrows():
        lines.append(f"| {r['config']} | {r['leaves_2017']:.1f} / {r['leaves_2019']:.1f} / {r['leaves_2021']:.1f} | "
                     f"{r['mean_depth']:.2f} | {r['max_leaves']} | {r['best_iterations']} | "
                     f"{'是' if r['any_hit_cap'] else '否'} | {fmt(r['mean_gap_fit_minus_es'])} | "
                     f"{fmt(r['mean_weighted_rank_ic'])} | {fmt(r['mean_top50_minus_pool'], 5)} | "
                     f"{fmt(r['mean_top50_overlap'], 3)} | {fmt(r['mean_diff_vs_elastic'], 5)} | "
                     f"{r['mean_train_seconds']:.0f} |")
    lines += ['', '## D. 第二阶段复杂度诊断的输入：深结构 0.1% 的实际平均叶子数 ÷ 中结构各组', '',
              '| 中结构配置 | 2017 | 2019 | 2021 | 三折叶子均值之比 | 深 0.1% 平均深度 | 中结构平均深度 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for _, r in ratios.iterrows():
        lines.append(f"| {r['medium']} | {r['ratio_2017']:.2f} | {r['ratio_2019']:.2f} | {r['ratio_2021']:.2f} | "
                     f"{r['ratio_mean_of_leaves']:.2f} | {r['deep_depth_mean']:.2f} | {r['medium_depth_mean']:.2f} |")
    lines += ['', '## E. 每折明细（含对同折 Elastic Net 的配对 NW t）', '',
              '| 折 | 配置 | min_data_in_leaf | Rank IC | 加权 | IC | 正天数占比 | best | 撞上限 | 平均叶 | 最大叶 | 平均深 | '
              '内部−早停 | Top50−池 | Top50 重合 | 择时方差占比 | Elastic | NW t | 训练秒 | 峰值 GB |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in table.sort_values(['year', 'structure', 'leaf']).iterrows():
        lines.append(f"| {r['year']} | {r['config']} | {r['min_data_in_leaf']} | {fmt(r['rank_ic'])} | "
                     f"{fmt(r['weighted_rank_ic'])} | {fmt(r['ic'])} | {fmt(r['rank_ic_positive_share'], 3)} | "
                     f"{r['best_iteration']} | {'是' if r['hit_round_limit'] else '否'} | {r['mean_leaves']:.1f} | "
                     f"{r['max_leaves']} | {r['mean_depth']:.2f} | {fmt(r['fit_minus_early_stop_rank_ic'])} | "
                     f"{fmt(r['top50_minus_pool'], 5)} | {fmt(r['top50_overlap'], 3)} | "
                     f"{fmt(r['timing_variance_share'], 3)} | {fmt(r['elastic_rank_ic'])} | "
                     f"{fmt(r['nw_t_vs_elastic'], 2)} | {r['train_seconds']:.0f} | {r['peak_memory_gb']:.2f} |")
    (OUT / 'stage1_tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return table, configs, pairwise, ratios


if __name__ == '__main__':
    summarize()
