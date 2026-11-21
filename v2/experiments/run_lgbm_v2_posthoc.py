"""事后诊断 (post-hoc diagnosis) of the V2 LightGBM confirmation results. Changes no frozen result or conclusion.

  python v2/experiments/run_lgbm_v2_posthoc.py --structure     diagnostics 1, 2, 4 (read existing predictions only)
  python v2/experiments/run_lgbm_v2_posthoc.py --trees Y SEED  diagnostic 3: one fixed-rounds refit (FINAL config, complete 2M sample)
  python v2/experiments/run_lgbm_v2_posthoc.py --fixed         diagnostic 3 driver: refits -> metrics -> 2022-2023 backtests
  python v2/experiments/run_lgbm_v2_posthoc.py --curves        diagnostic 3: saved early-stopping curves

Diagnostic 3 trains each (year, seed) once to the largest round count and predicts with num_iteration = 500 / 1500 /
3000 / 4500. Without early stopping a booster truncated to k iterations equals a booster trained for k rounds (same
params, data and seed; deterministic=True); this is asserted against the frozen review model at its own round count.
Reads no 2024+ data. Frozen code is imported, never edited.
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from lgbm_v2_common import DATA, KEY, ROOT
from run_lgbm_v2_stage2 import eligible_metrics
import run_lgbm_v2_final as fin
import run_lgbm_v2_review as rv
import run_lgbm_v2_stage6a as s6
import run_lgbm_v2_confirm as conf

OUT = DATA / 'v2' / 'lgbm_posthoc'
PRED = conf.OUT / 'predictions'
YEARS = conf.ALL_YEARS
GROUPS = conf.GROUPS
STYLE = ['log_circ_mv', 'volatility_20d', 'logret_60d', 'log_amount', 'turnover_rate_f']
FEATURES = common.FEATURES
TREE_PLAN = {2023: [500, 1500, 3000, 4500], 2022: [4500]}


# ---------------------------------------------------------------- helpers
def spearman_by_day(pred, y, starts, ends):
    out = []
    for s, e in zip(starts, ends):
        if e - s < 20:
            out.append(np.nan)
            continue
        a = pd.Series(pred[s:e]).rank().to_numpy()
        b = pd.Series(y[s:e]).rank().to_numpy()
        out.append(np.corrcoef(a, b)[0, 1])
    return np.array(out)


def eligible_frame(year):
    f = pd.read_parquet(PRED / f'{year}.parquet')
    f = f.loc[f['eligible'].astype(bool) & np.isfinite(f['y'])].copy()
    f['ret'] = np.expm1(f['y'])
    f['ex'] = f['ret'] - f.groupby('signal_date')['ret'].transform('mean')
    return f.sort_values(['signal_date', 'ts_code']).reset_index(drop=True)


def board_of(codes):
    return np.select([codes.str.match(r'^68[89]'), codes.str.match(r'^30[01]'), codes.str.endswith('.BJ'),
                      codes.str.match(r'^(60|00)')], ['STAR', 'ChiNext', 'Beijing', 'Main'], 'Other')


def style_frame(year, keys):
    parts = []
    for path in sorted((FEATURES / f'feature_year={year}').glob('month_*.parquet')):
        parts.append(pd.read_parquet(path, columns=KEY + STYLE))
    s = pd.concat(parts, ignore_index=True)
    s['signal_date'] = pd.to_datetime(s['signal_date']).dt.normalize()
    m = keys[KEY].merge(s, on=KEY, how='left', validate='one_to_one')
    assert len(m) == len(keys)
    return m


# ---------------------------------------------------------------- diagnostics 1, 2, 4
def structure():
    OUT.mkdir(parents=True, exist_ok=True)
    basic = pd.read_csv(common.STOCK_BASIC, dtype=str)[['ts_code', 'industry']]
    q_rows, top_rows, ic_rows, style_rows, board_rows, ind_rows, overlap_rows, daily_top = [], [], [], [], [], [], [], []
    nav = pd.read_parquet(conf.OUT / 'backtests' / 'G1_lgbm_final_0.5_nav.parquet')
    nav['date'] = pd.to_datetime(nav['date'])
    n22 = nav.loc[nav['date'].dt.year == 2022].reset_index(drop=True)
    peak_i = int(np.argmax(n22['nav'].to_numpy()[:int(np.argmin(n22['nav'].to_numpy() / np.maximum.accumulate(n22['nav'].to_numpy()))) + 1]))
    trough_i = int(np.argmin(n22['nav'].to_numpy() / np.maximum.accumulate(n22['nav'].to_numpy())))
    window = (n22['date'][peak_i], n22['date'][trough_i])
    for year in YEARS:
        f = eligible_frame(year)
        dates = f['signal_date'].to_numpy()
        bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
        starts, ends = bounds[:-1], bounds[1:]
        day_index = np.repeat(np.arange(len(starts)), ends - starts)
        n_day = (ends - starts)[day_index]
        styles = style_frame(year, f)
        for c in STYLE:
            v = styles[c].where(np.isfinite(styles[c]))
            f[f'pct_{c}'] = v.groupby(f['signal_date']).rank(pct=True).to_numpy()
        f['board'] = board_of(f['ts_code'])
        f = f.merge(basic, on='ts_code', how='left')
        f['industry'] = f['industry'].fillna('unknown')
        tops = {}
        for g in GROUPS:
            # rank 1 = highest prediction; ties by ts_code (rows are sorted by ts_code within the day)
            order = f.groupby('signal_date')[g].rank(method='first', ascending=False).to_numpy()
            dec = np.ceil(order / n_day * 10).astype(int)
            v20 = np.ceil(order / n_day * 20).astype(int)
            for bins, label in ((dec, 10), (v20, 20)):
                t = pd.DataFrame({'d': day_index, 'b': bins, 'ex': f['ex'].to_numpy()}).groupby(['d', 'b'])['ex'].mean()
                means = t.groupby(level='b').mean()
                q_rows.append(dict(year=year, group=g, bins=label, **{f'b{int(k)}': float(v) for k, v in means.items()}))
            row = dict(year=year, group=g)
            ex = f['ex'].to_numpy()
            for n in (50, 100):
                sel = order <= n
                daily = pd.Series(ex[sel]).groupby(day_index[sel]).mean()
                row[f'top{n}_excess'] = float(daily.mean())
                if n == 50:
                    daily_top.append(pd.DataFrame({'date': pd.to_datetime(dates[starts[daily.index]]), 'group': g,
                                                   'top50_excess': daily.to_numpy()}))
            row['decile1_excess'] = q_rows[-2]['b1']
            row['decile10_excess'] = q_rows[-2]['b10']
            top_rows.append(row)
            # Rank IC: full, top 10%, upper half, lower half (each within the subset, per day)
            pred, y = f[g].to_numpy(), f['y'].to_numpy()
            ics = {}
            for name, mask in (('full', np.ones(len(f), bool)), ('top10pct', order <= 0.1 * n_day),
                               ('upper_half', order <= 0.5 * n_day), ('lower_half', order > 0.5 * n_day)):
                idx = np.flatnonzero(mask)
                d = day_index[idx]
                b = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
                ics[name] = float(np.nanmean(spearman_by_day(pred[idx], y[idx], b[:-1], b[1:])))
            ic_rows.append(dict(year=year, group=g, **{f'rank_ic_{k}': v for k, v in ics.items()}))
            if g in ('G1_lgbm_final', 'G3_linear_candidate'):
                top = f.loc[order <= 50]
                tops[g] = top
                style_rows.append(dict(year=year, group=g, window='year',
                                       **{f'pct_{c}': float(top[f'pct_{c}'].mean()) for c in STYLE}))
                if year == 2022:
                    w = top.loc[(top['signal_date'] >= window[0]) & (top['signal_date'] <= window[1])]
                    style_rows.append(dict(year=year, group=g, window=f'G1 drawdown {window[0].date()}..{window[1].date()}',
                                           **{f'pct_{c}': float(w[f'pct_{c}'].mean()) for c in STYLE}))
                share = top['board'].value_counts(normalize=True)
                pool = f['board'].value_counts(normalize=True)
                for b in ('Main', 'ChiNext', 'STAR', 'Beijing', 'Other'):
                    board_rows.append(dict(year=year, group=g, board=b, top50_share=float(share.get(b, 0)),
                                           pool_share=float(pool.get(b, 0))))
                ishare = top['industry'].value_counts(normalize=True)
                ipool = f['industry'].value_counts(normalize=True)
                for ind in ishare.index:
                    ind_rows.append(dict(year=year, group=g, industry=ind, top50_share=float(ishare[ind]),
                                         pool_share=float(ipool.get(ind, 0)), overweight=float(ishare[ind] - ipool.get(ind, 0))))
        a = tops['G1_lgbm_final'].groupby('signal_date')['ts_code'].apply(set)
        b = tops['G3_linear_candidate'].groupby('signal_date')['ts_code'].apply(set)
        overlap_rows.append(dict(year=year, g1_g3_top50_overlap=float(np.mean([len(a[d] & b[d]) / 50 for d in a.index]))))
        print(f'[structure {year}] done', flush=True)
    pd.DataFrame(q_rows).to_csv(OUT / 'd1_quantile_excess.csv', index=False)
    pd.DataFrame(top_rows).to_csv(OUT / 'd1_head_excess.csv', index=False)
    pd.DataFrame(ic_rows).to_csv(OUT / 'd1_rank_ic_parts.csv', index=False)
    pd.DataFrame(style_rows).to_csv(OUT / 'd2_top50_style.csv', index=False)
    pd.DataFrame(board_rows).to_csv(OUT / 'd2_top50_board.csv', index=False)
    pd.DataFrame(ind_rows).to_csv(OUT / 'd2_top50_industry.csv', index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / 'd2_g1_g3_overlap.csv', index=False)
    common.save_json(OUT / 'd2_drawdown_window.json', dict(start=str(window[0].date()), end=str(window[1].date()),
                                                           nav_drawdown=float(n22['nav'][trough_i] / n22['nav'][peak_i] - 1)))
    # diagnostic 4: quarterly decay
    dt = pd.concat(daily_top, ignore_index=True)
    dt['quarter'] = dt['date'].dt.to_period('Q').astype(str)
    ic = []
    for y in YEARS:
        for g in GROUPS:
            e = pd.read_parquet(conf.OUT / 'evaluation' / f'eligible_daily_{y}_{g}.parquet')
            ic.append(e[['date', 'rank_ic']].assign(group=g))
    ic = pd.concat(ic, ignore_index=True)
    ic['date'] = pd.to_datetime(ic['date'])
    ic['quarter'] = ic['date'].dt.to_period('Q').astype(str)
    qi = ic.pivot_table(index='quarter', columns='group', values='rank_ic', aggfunc='mean')
    qt = dt.pivot_table(index='quarter', columns='group', values='top50_excess', aggfunc='mean')
    quarterly = pd.DataFrame({'ic_G1': qi['G1_lgbm_final'], 'ic_G1_minus_G2': qi['G1_lgbm_final'] - qi['G2_elastic_D'],
                              'ic_G1_minus_G3': qi['G1_lgbm_final'] - qi['G3_linear_candidate'],
                              'top50_G1': qt['G1_lgbm_final'],
                              'top50_G1_minus_G2': qt['G1_lgbm_final'] - qt['G2_elastic_D'],
                              'top50_G1_minus_G3': qt['G1_lgbm_final'] - qt['G3_linear_candidate']})
    quarterly.to_csv(OUT / 'd4_quarterly.csv')
    print(quarterly.round(4).to_string(), flush=True)


# ---------------------------------------------------------------- diagnostic 3: fixed rounds
def trees_dir(year, seed):
    return OUT / 'fixed_trees' / f'{year}_F3_s{seed}'


def fit_trees(year, seed, threads=8):
    import lightgbm as lgb
    out = trees_dir(year, seed)
    if (out / 'done.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rounds = max(TREE_PLAN[year])
    frozen_rounds = int(common.read_json(rv.review_dir(year, conf.CONFIG, seed) / 'summary.json')['lightgbm']['rounds'])
    data = rv.load_full(year)
    base = cache.load(year, 'full')
    names = common.feature_names(year, 'full')
    params = fin.params_for(conf.CONFIG, s6.build_labels(year)['ratio_D'], cache.ROWS, seed, threads)
    frozen_params = common.read_json(rv.review_dir(year, conf.CONFIG, seed) / 'summary.json')['lightgbm']['params']
    assert params == frozen_params, 'parameters differ from the frozen review fit'
    train_set = lgb.Dataset(data['x'], label=data['d'], feature_name=names, free_raw_data=False, params=params)
    booster = lgb.train(params, train_set, num_boost_round=rounds)
    train_seconds = round(time.perf_counter() - started, 1)
    del train_set
    data['x'] = None
    preds = {}
    for k in sorted(set(TREE_PLAN[year] + [frozen_rounds])):
        preds[k] = np.concatenate([booster.predict(a, num_iteration=k) for a in conf.inp.outer_chunks(base['outer_x'], None)])
    frozen = pd.read_parquet(rv.review_dir(year, conf.CONFIG, seed) / 'outer_predictions.parquet')['lgbm'].to_numpy()
    max_diff = float(np.max(np.abs(preds[frozen_rounds] - frozen)))
    if max_diff > 1e-12:
        raise SystemExit(f'{year} s{seed}: truncated booster at {frozen_rounds} does not reproduce the frozen review '
                         f'predictions (max diff {max_diff}); stopping')
    outer = base['outer'][KEY].copy()
    for k in TREE_PLAN[year]:
        outer[f'trees_{k}'] = preds[k]
    outer.to_parquet(out / 'outer_predictions.parquet', index=False)
    common.save_json(out / 'done.json', dict(year=year, seed=seed, rounds=rounds, trees_built=booster.num_trees(),
                                             frozen_rounds=frozen_rounds, frozen_reproduced_max_abs_diff=max_diff,
                                             train_seconds=train_seconds, total_seconds=round(time.perf_counter() - started, 1)))
    print(f'[trees {year} s{seed}] {rounds} rounds, trees {booster.num_trees()}, frozen@{frozen_rounds} reproduced '
          f'(max diff {max_diff:.1e}), train {train_seconds:.0f}s', flush=True)


def run_sub(args, label):
    print(f'[start] {label} at {time.strftime("%m-%d %H:%M:%S")}', flush=True)
    if subprocess.call([sys.executable, '-u', __file__] + args) != 0:
        raise SystemExit(f'{label} failed; stopping')


def variant_predictions():
    """Three-seed averages per variant; returns {variant: {year: array aligned with predictions/{year}.parquet}}."""
    out = {}
    for year, ks in TREE_PLAN.items():
        meta = pd.read_parquet(PRED / f'{year}.parquet')[KEY]
        frames = [pd.read_parquet(trees_dir(year, s) / 'outer_predictions.parquet') for s in fin.SEEDS]
        outer = cache.load(year, 'full')['outer']
        for fr in frames:
            assert (fr['ts_code'].to_numpy() == outer['ts_code'].to_numpy()).all()
        for k in ks:
            avg = outer[KEY].assign(p=np.mean([fr[f'trees_{k}'].to_numpy() for fr in frames], axis=0))
            avg['signal_date'] = pd.to_datetime(avg['signal_date']).dt.normalize()
            m = meta.merge(avg, on=KEY, how='left', validate='one_to_one')
            assert np.isfinite(m['p']).all()
            out.setdefault(k, {})[year] = m['p'].to_numpy()
    return out


def fixed():
    OUT.mkdir(parents=True, exist_ok=True)
    for year, ks in TREE_PLAN.items():
        for s in fin.SEEDS:
            if not (trees_dir(year, s) / 'done.json').exists():
                run_sub(['--trees', str(year), str(s)], f'fixed trees {year} s{s}')
    preds = variant_predictions()
    # metrics per year: frozen G1 vs variants
    rows = []
    for year in (2022, 2023):
        f = pd.read_parquet(PRED / f'{year}.parquet')
        cols = {'frozen_G1': f['G1_lgbm_final'].to_numpy()}
        for k, by_year in preds.items():
            if year in by_year:
                cols[f'trees_{k}'] = by_year[year]
        for name, p in cols.items():
            g = f.assign(p=p)
            e, _ = eligible_metrics(g, 'p')
            m, _ = common.evaluate_outer(g, 'p')
            rows.append(dict(year=year, variant=name, eligible_rank_ic=e['mean_rank_ic'], all_sample_rank_ic=m['mean_rank_ic'],
                             top50_minus_pool=m['top50_minus_pool_5d_simple_return'], top50_overlap=m['consecutive_day_top50_overlap']))
            print(f'[d3 {year} {name}] eligible {e["mean_rank_ic"]:.5f} top50 {m["top50_minus_pool_5d_simple_return"]:.5f}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'd3_fixed_trees_metrics.csv', index=False)
    # backtests, original impact, 2017-2023 path (2017-2021 frozen G1), report 2022-2023
    variants = {f'2023_trees_{k}': {2023: preds[k][2023]} for k in TREE_PLAN[2023]}
    variants['2022_2023_trees_4500'] = {2022: preds[4500][2022], 2023: preds[4500][2023]}
    backtest(variants)


def backtest(variants):
    from run_elastic_19_control import prepare, buffer_panel, annotate_orders, simulate_released_cash, audit_ledger
    out = OUT / 'backtests'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'progress.json'
    results = common.read_json(path) if path.exists() else {}
    frame = conf.load_predictions()
    template, _, panel, missing, evidence = prepare(conf.backtest_frame(frame, 'G1_lgbm_final'))
    del evidence
    template = template.drop(columns='prediction')
    template['entry_tradable'] = template.entry_tradable.fillna(False) & template.strict_buy
    jobs = {'frozen_G1_check': {}}
    jobs.update(variants)
    for name, repl in jobs.items():
        if name in results:
            continue
        f = frame.copy()
        for year, p in repl.items():
            mask = (f['signal_date'].dt.year == year).to_numpy()
            assert mask.sum() == len(p)
            f.loc[mask, 'G1_lgbm_final'] = p
        signals = conf.backtest_frame(f, 'G1_lgbm_final')[KEY + ['prediction']]
        orders = template.merge(signals, on=KEY, validate='one_to_one')
        assert len(orders) == len(template)
        selected, decision = buffer_panel(orders, panel, 100)
        metrics, nav, trades = simulate_released_cash(orders, selected, aum=1e8, cost_bps=20, impact_coefficient=0.5, round_lots=True)
        trades = annotate_orders(trades, decision, panel, orders)
        audit = audit_ledger(trades, nav, orders, panel, missing)
        assert audit['filled_entry_missing_limits'] == audit['sold_exit_missing_limits'] == 0
        nav['date'] = pd.to_datetime(nav['date'])
        nav.to_parquet(out / f'{name}_nav.parquet', index=False)
        yearly = {b['year']: b['net_return'] for b in audit['annual_boundaries']}
        d = nav.loc[nav['date'].dt.year.isin([2022, 2023]), 'daily_return']
        results[name] = dict(net_return_2017_2023=metrics['net_return'], yearly=yearly,
                             net_return_2022_2023=float(np.prod(1 + d) - 1),
                             max_drawdown_2022_2023=float(np.min(np.cumprod(1 + d.to_numpy()) / np.maximum.accumulate(np.cumprod(1 + d.to_numpy())) - 1)))
        if name == 'frozen_G1_check':
            ref = common.read_json(conf.OUT / 'backtests' / 'progress.json')['G1_lgbm_final|0.5']['net_return']
            if abs(metrics['net_return'] - ref) > 1e-9:
                raise SystemExit(f'frozen G1 backtest not reproduced ({metrics["net_return"]} vs {ref}); stopping')
        common.save_json(path, results)
        print(f'[d3 backtest {name}] 2022 {yearly[2022]:.4f} 2023 {yearly[2023]:.4f} 2022-23 {results[name]["net_return_2022_2023"]:.4f}', flush=True)


def curves():
    rows = []
    for year in range(2017, 2024):
        for seed in fin.SEEDS:
            c = pd.read_parquet(fin.seed_dir(year, conf.CONFIG, seed) / 'early_stop_curve.parquet')
            best = c.loc[c['early_stop_rank_ic'].idxmax()]
            r = dict(year=year, seed=seed, last_round=int(c['iteration'].max()), best_round=int(best['iteration']),
                     best_rank_ic=float(best['early_stop_rank_ic']))
            for k in (10, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000):
                hit = c.loc[c['iteration'] == k, 'early_stop_rank_ic']
                r[f'ic_{k}'] = float(hit.iloc[0]) if len(hit) else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'd3_early_stop_curves.csv', index=False)
    print(t.round(4).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--structure', action='store_true')
    p.add_argument('--trees', nargs=2, type=int)
    p.add_argument('--fixed', action='store_true')
    p.add_argument('--curves', action='store_true')
    a = p.parse_args()
    conf.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.structure:
        structure()
    elif a.trees:
        fit_trees(*a.trees)
    elif a.fixed:
        t = time.perf_counter()
        fixed()
        print(f'[done] diagnostic 3 in {(time.perf_counter() - t) / 3600:.2f} h', flush=True)
    elif a.curves:
        curves()
    else:
        p.print_help()


if __name__ == '__main__':
    main()
