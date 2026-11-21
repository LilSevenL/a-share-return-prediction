"""Stage 1 recomputed on the eligible pool (plan section 12): no new fits, report only.

Uses the saved stage-1 outer predictions and the cached outer metadata (eligible, y). Outputs in
quant_data/processed/v2/lgbm_stage1/: eligible_summary_table.csv, eligible_config_table.csv,
eligible_pairwise_nw.csv, eligible_tables.md.
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
from run_lgbm_v2_stage1 import LEAF_FRACTIONS, OUT, STRUCTURES, YEARS, config_name
from run_lgbm_v2_stage2 import PENALTY, eligible_metrics, elastic_eligible, paired_t
import lgbm_v2_common as common


def main():
    rows, daily = [], {}
    elastic = {}
    for year in YEARS:
        directory = cache.cache_dir(year, 'full')
        outer = pd.read_parquet(directory / 'outer_meta.parquet')
        elastic[year] = elastic_eligible(year)
        for structure in STRUCTURES:
            for leaf in LEAF_FRACTIONS:
                source = OUT / config_name(year, structure, leaf)
                pred = pd.read_parquet(source / 'outer_predictions.parquet')
                assert (pred['ts_code'].to_numpy() == outer['ts_code'].to_numpy()).all()
                assert (pd.to_datetime(pred['signal_date']).to_numpy() == pd.to_datetime(outer['signal_date']).to_numpy()).all()
                frame = outer.assign(lgbm=pred['lgbm'].to_numpy())
                m, d = eligible_metrics(frame, 'lgbm')
                s1 = common.read_json(source / 'summary.json')
                vs = paired_t(d, elastic[year][1])
                config = f'{structure}_{leaf}'
                daily[(config, year)] = d
                rows.append(dict(config=config, year=year, eligible_rank_ic=m['mean_rank_ic'],
                                 eligible_ic=m['mean_ic'], eligible_rows=m['rows'],
                                 eligible_positive_share=m['rank_ic_positive_share'],
                                 all_rank_ic=s1['outer_metrics']['mean_rank_ic'],
                                 elastic_eligible_rank_ic=elastic[year][0]['mean_rank_ic'],
                                 diff_vs_elastic_eligible=vs['mean_rank_ic_difference'],
                                 nw_t_vs_elastic_eligible=vs['newey_west_t_lag5'],
                                 best_iteration=s1['lightgbm']['best_iteration'],
                                 rounds_trained=s1['lightgbm']['rounds_trained']))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'eligible_summary_table.csv', index=False, encoding='utf-8')

    configs = []
    for config, g in table.groupby('config', sort=False):
        e = g.set_index('year')['eligible_rank_ic'].reindex(YEARS).to_numpy()
        a = g.set_index('year')['all_rank_ic'].reindex(YEARS).to_numpy()
        configs.append(dict(config=config, **{f'eligible_{y}': v for y, v in zip(YEARS, e)},
                            eligible_mean=e.mean(), eligible_std=e.std(ddof=0),
                            eligible_score=e.mean() - PENALTY * e.std(ddof=0),
                            all_score=a.mean() - PENALTY * a.std(ddof=0), negative_fold=bool((e < 0).any()),
                            mean_t_vs_elastic=g['nw_t_vs_elastic_eligible'].mean()))
    configs = pd.DataFrame(configs)
    configs['all_rank'] = configs['all_score'].rank(ascending=False).astype(int)
    configs = configs.sort_values('eligible_score', ascending=False).reset_index(drop=True)
    configs['eligible_rank'] = np.arange(1, len(configs) + 1)
    configs.to_csv(OUT / 'eligible_config_table.csv', index=False, encoding='utf-8')

    top = configs.loc[~configs['negative_fold'], 'config'].iat[0]
    pair = []
    for config in configs['config']:
        if config == top:
            continue
        a = pd.concat([daily[(top, y)] for y in YEARS], ignore_index=True)
        b = pd.concat([daily[(config, y)] for y in YEARS], ignore_index=True)
        pair.append(dict(top=top, other=config, **paired_t(a, b)))
    pair = pd.DataFrame(pair)
    pair.to_csv(OUT / 'eligible_pairwise_nw.csv', index=False, encoding='utf-8')

    f = lambda x, d=4: f'{x:.{d}f}'
    lines = ['### G1. 资格池口径排名（与全样本口径对照）', '',
             '| 资格池排名 | 配置 | 2017 | 2019 | 2021 | 均值 | 标准差 | 资格池分数 | 全样本分数 | 全样本排名 | 有负折 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for _, r in configs.iterrows():
        lines.append(f"| {r['eligible_rank']} | {r['config']} | {f(r['eligible_2017'])} | {f(r['eligible_2019'])} | "
                     f"{f(r['eligible_2021'])} | {f(r['eligible_mean'])} | {f(r['eligible_std'])} | "
                     f"**{f(r['eligible_score'])}** | {f(r['all_score'])} | {r['all_rank']} | "
                     f"{'是' if r['negative_fold'] else '否'} |")
    lines += ['', f'### G2. 资格池分数第一（{top}）对其余配置：资格池日度 Rank IC 差（三折拼接）的 NW t（lag 5）', '',
              '| 对比配置 | 天数 | 平均差 | NW t |', '|---|---:|---:|---:|']
    for _, r in pair.iterrows():
        lines.append(f"| {r['other']} | {r['days']} | {f(r['mean_rank_ic_difference'], 5)} | {f(r['newey_west_t_lag5'], 2)} |")
    lines += ['', '### G3. 每折资格池明细与对同折 Elastic（资格池评价）的配对 NW t', '',
              '| 折 | 配置 | 资格池行数 | 资格池 Rank IC | 全样本 Rank IC | 资格池 IC | 正天数占比 | Elastic 资格池 | 差 | NW t |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in table.sort_values(['year', 'config']).iterrows():
        lines.append(f"| {r['year']} | {r['config']} | {r['eligible_rows']:,} | {f(r['eligible_rank_ic'])} | "
                     f"{f(r['all_rank_ic'])} | {f(r['eligible_ic'])} | {f(r['eligible_positive_share'], 3)} | "
                     f"{f(r['elastic_eligible_rank_ic'])} | {f(r['diff_vs_elastic_eligible'], 5)} | "
                     f"{f(r['nw_t_vs_elastic_eligible'], 2)} |")
    lines += ['', '### G4. 同折 Elastic Net：全样本与资格池口径', '', '| 折 | 全样本 Rank IC | 资格池 Rank IC | 资格池行数 |',
              '|---:|---:|---:|---:|']
    for y in YEARS:
        allm = common.read_json(cache.cache_dir(y, 'full') / 'elastic' / 'metrics.json')['mean_rank_ic']
        lines.append(f"| {y} | {f(allm)} | {f(elastic[y][0]['mean_rank_ic'])} | {elastic[y][0]['rows']:,} |")
    text = '\n'.join(lines) + '\n'
    (OUT / 'eligible_tables.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
