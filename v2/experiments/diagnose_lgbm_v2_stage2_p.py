"""Stage 2 offline diagnostic for the provisional main configuration P (plan section 13). No new fits.

P = deep 0.1%, lambda1 = 50, lambda2 = 1000; backup = medium 0.1%, lambda1 = 10, lambda2 = 1000.
  * NW t (lag 5) of P vs the other 17 stage-2 combinations, eligible daily Rank IC, five folds concatenated
  * per-100-round eligible outer Rank IC curves of P and the backup on all five folds (all-round models):
    curve peak round, early-stopping best, and the Rank IC gap between them
Outputs in quant_data/processed/v2/lgbm_stage2/: p_pairwise_nw.csv, p_round_curves.csv, p_round_summary.csv,
p_diagnostic.md. Diagnostic only; nothing here is used for selection.
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
from run_lgbm_v2_stage2 import C, OUT, YEARS, combo_label
from summarize_lgbm_v2_stage2 import eligible_daily, round_curve

P = ('deep', '0.1pct', 50, 1000)
BACKUP = ('medium', '0.1pct', 10, 1000)


def main():
    state = common.read_json(OUT / 'grid_state.json')
    m_leaf = common.read_json(OUT / 'm_selection.json')['m_leaf']
    combos = [(s, leaf, a, b) for s, leaf in [('medium', m_leaf), C]
              for a in state['l1'][s] for b in state['l2'][s]]
    base = pd.concat([eligible_daily(y, *P) for y in YEARS])
    pairs = []
    for c in combos:
        if c == P:
            continue
        other = pd.concat([eligible_daily(y, *c) for y in YEARS])
        joined = base.merge(other, on=['year', 'date'], suffixes=('_p', '_o'), validate='one_to_one')
        if len(joined) != len(base):
            raise RuntimeError('Incomplete daily pairing')
        diff = (joined['rank_ic_p'] - joined['rank_ic_o']).to_numpy()
        pairs.append(dict(p=combo_label(*P), other=combo_label(*c), days=len(joined), mean_diff=float(diff.mean()),
                          newey_west_t_lag5=common.newey_west_tstat(diff, lag=5)))
    pairs = pd.DataFrame(pairs).sort_values('newey_west_t_lag5')
    pairs.to_csv(OUT / 'p_pairwise_nw.csv', index=False, encoding='utf-8')

    curves = []
    for combo in (P, BACKUP):
        for year in YEARS:
            c = round_curve(year, *combo)
            if c is None:
                raise RuntimeError(f'{year} {combo_label(*combo)}: all-round model missing')
            curves.append(c)
            print(f'[curve] {year} {combo_label(*combo)}', flush=True)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(OUT / 'p_round_curves.csv', index=False, encoding='utf-8')
    rows = []
    for (combo, year), c in curves.groupby(['combo', 'year'], sort=False):
        best = int(c['best_iteration'].iat[0])
        stored = common.read_json(OUT / f'{year}_{combo}' / 'summary.json')['eligible_metrics']['mean_rank_ic']
        peak = c.loc[c['eligible_rank_ic'].idxmax()]
        rows.append(dict(combo=combo, year=year, best_iteration=best, rounds_trained=int(c['rounds_trained'].iat[0]),
                         eligible_at_best=stored, peak_round=int(peak['rounds']), peak_eligible=float(peak['eligible_rank_ic']),
                         peak_minus_best=float(peak['eligible_rank_ic']) - stored,
                         at_100=float(c.loc[c['rounds'] == 100, 'eligible_rank_ic'].iat[0])))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'p_round_summary.csv', index=False, encoding='utf-8')

    f = lambda x, d=4: f'{x:.{d}f}'
    L = [f'### H1. P（{combo_label(*P)}）对第二阶段其余 17 个组合：资格池日度 Rank IC 差（五折拼接）NW t（lag 5）', '',
         '| 对比组合 | 天数 | 平均差（P − 对比） | NW t |', '|---|---:|---:|---:|']
    for _, r in pairs.iterrows():
        L.append(f"| {r['other']} | {r['days']} | {f(r['mean_diff'], 5)} | {f(r['newey_west_t_lag5'], 2)} |")
    L += ['', f'### H2. P 与备选（{combo_label(*BACKUP)}）五折外层每 100 轮资格池 Rank IC：峰值轮数 vs 早停 best', '',
          '| 组合 | 折 | 早停 best | 训练轮数 | best 处资格池 Rank IC | 曲线峰值轮数 | 峰值 Rank IC | 峰值 − best 处 | 100 轮处 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in summary.iterrows():
        L.append(f"| {r['combo']} | {r['year']} | {r['best_iteration']} | {r['rounds_trained']} | {f(r['eligible_at_best'])} | "
                 f"{r['peak_round']} | {f(r['peak_eligible'])} | {f(r['peak_minus_best'])} | {f(r['at_100'])} |")
    for combo, g in summary.groupby('combo', sort=False):
        L.append(f"| {combo} | 五折均值 | | | {f(g['eligible_at_best'].mean())} | | {f(g['peak_eligible'].mean())} | "
                 f"{f(g['peak_minus_best'].mean())} | {f(g['at_100'].mean())} |")
    text = '\n'.join(L) + '\n'
    (OUT / 'p_diagnostic.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
