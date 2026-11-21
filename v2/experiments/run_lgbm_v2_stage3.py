"""V2 LightGBM after stage 6a (plan section 14): placebo, Elastic(D), D-label structure / lambda2 recheck (P*D),
stage 3 sampling and binning (P3). Training label = D (daily cross-sectional rank return) throughout; early
stopping and every score stay on the raw label (eligible daily Rank IC for selection).

  python v2/experiments/run_lgbm_v2_stage3.py --placebo        step 1 (stops the pipeline if it fails)
  python v2/experiments/run_lgbm_v2_stage3.py --elastic        step 2
  python v2/experiments/run_lgbm_v2_stage3.py --structure      step 3 -> P*D
  python v2/experiments/run_lgbm_v2_stage3.py --sampling       step 4 -> P3
  python v2/experiments/run_lgbm_v2_stage3.py --summarize      step 5
  python v2/experiments/run_lgbm_v2_stage3.py --all            steps 1-5 in order
  python v2/experiments/run_lgbm_v2_stage3.py --one YEAR VARIANT   one fit (subprocess)
Fits use run_lgbm_v2_stage2.fit_one (cap 6000, patience 200, all-round models, per-round tree counts).
Outputs: quant_data/processed/v2/lgbm_stage3/{year}_{variant}/ ; Elastic(D) in lgbm_cache/{year}_full/elastic_D/.
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

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from lgbm_v2_common import DATA
from run_lgbm_v2_stage2 import PENALTY, YEARS, completed, eligible_metrics, fit_one, paired_t, truncated
import run_lgbm_v2_stage6a as s6

OUT = DATA / 'v2' / 'lgbm_stage3'
PLACEBO_SEED = 20261117
BASE_L1 = 50  # P* lambda1 on the raw-label scale; D uses sig2(50 x ratio_D) x multiplier
DEEP, MEDIUM = ('deep', '0.1pct'), ('medium', '0.1pct')
D_COMBOS = [(0.5, 1000), (1.0, 1000), (2.0, 1000), (1.0, 0), (1.0, 10000)]  # (lambda1 multiplier, lambda2)
SAMPLING = {
    'feature_fraction': {'ff0.1': dict(feature_fraction=0.1), 'ff0.2': dict(feature_fraction=0.2),
                         'ff0.5': dict(feature_fraction=0.5)},
    'bagging_fraction': {'bag0.5': dict(bagging_fraction=0.5, bagging_freq=1),
                         'bag0.7': dict(bagging_fraction=0.7, bagging_freq=1)},
    'max_bin': {'bin63': dict(max_bin=63)},
    'extra_trees': {'xt': dict(extra_trees=True)},
}
SAMPLING_PARAMS = {name: p for item in SAMPLING.values() for name, p in item.items()}


def ratio_d(year):
    return s6.build_labels(year)['ratio_D']


def d_label(year):
    return np.load(s6.LABELS / f'{year}_D.npy')


def deep_l1(year, mult):
    return s6.sig2(BASE_L1 * ratio_d(year)) * mult


def num(v):
    v = float(v)
    return int(v) if v.is_integer() else v


# ---------------------------------------------------------------- variants
def combo_name(mult, l2):
    return f'deepD_x{mult:g}_l2-{l2:g}'


def variant_dir(year, variant):
    """Stage-6a D fits at lambda2 = 1000 are reused for the deep D combinations."""
    if variant.startswith('deepD_x') and variant.endswith('_l2-1000'):
        mult = float(variant.split('_x')[1].split('_l2')[0])
        return s6.OUT / f'{year}_D_x{mult:g}'
    return OUT / f'{year}_{variant}'


def p_star_d():
    p = common.read_json(OUT / 'p_star_d.json')['p_star_d']
    return float(p['mult']), num(p['l2'])


def variant_spec(year, variant):
    """(structure, leaf, lambda1, lambda2, extra, train_label, label_info) for one fit."""
    label = d_label(year)
    info = dict(scheme='D', name=s6.NAMES['D'], label_file=f'labels/{year}_D.npy', std_ratio=ratio_d(year))
    if variant == 'placebo' or variant.startswith('placebo_s'):
        seed = PLACEBO_SEED if variant == 'placebo' else int(variant.split('_s')[1])
        keys = pd.read_parquet(cache.cache_dir(year, 'full') / 'fit_keys.parquet')
        dates = pd.to_datetime(keys['signal_date']).to_numpy()
        starts, ends = common.day_slices(dates)
        rng = np.random.default_rng(seed)
        shuffled = label.copy()
        for s, e in zip(starts, ends):
            shuffled[s:e] = label[s:e][rng.permutation(e - s)]
        assert np.allclose(np.sort(shuffled), np.sort(label)) and not np.array_equal(shuffled, label)
        info.update(placebo=f'D label permuted within each signal date among sampled rows, seed {seed}')
        return (*DEEP, deep_l1(year, 1.0), 1000, {}, shuffled, info)
    if variant.startswith('deepD_x'):
        mult = float(variant.split('_x')[1].split('_l2')[0])
        l2 = num(variant.split('_l2-')[1])
        return (*DEEP, deep_l1(year, mult), l2, {}, label, dict(info, lambda1_multiplier=mult))
    if variant == 'mediumD':
        l1 = s6.sig2(10 * ratio_d(year))
        return (*MEDIUM, l1, 1000, {}, label, dict(info, lambda1_rule='10 x ratio_D, 2 significant digits'))
    mult, l2 = p_star_d()
    names = variant.split('+')
    extra = {}
    for name in names:
        extra.update(SAMPLING_PARAMS[name])
    return (*DEEP, deep_l1(year, mult), l2, extra, label, dict(info, base='P*D', sampling=names))


def run_one(year, variant, threads):
    structure, leaf, l1, l2, extra, label, info = variant_spec(year, variant)
    fit_one(year, structure, leaf, num(l1), l2, OUT / f'{year}_{variant}', threads, extra=extra or None,
            train_label=label, label_info=info, stage=3, tag=f'{variant} (lambda1 {l1:g}, lambda2 {l2:g})')


def run_job(year, variant, threads):
    output = variant_dir(year, variant)
    if completed(output):
        return
    print(f'[start] {year}_{variant} at {time.strftime("%H:%M:%S")}', flush=True)
    code = subprocess.call([sys.executable, '-u', __file__, '--one', str(year), variant, '--threads', str(threads)])
    if code != 0:
        raise SystemExit(f'{year}_{variant} failed with exit code {code}; stopping')


# ---------------------------------------------------------------- scoring helpers
def summary(year, variant):
    return common.read_json(variant_dir(year, variant) / 'summary.json')


def daily(variant, years=YEARS):
    parts = []
    for y in years:
        d = pd.read_parquet(variant_dir(y, variant) / 'eligible_daily_metrics.parquet')
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        parts.append(d.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def nw(a, b):
    joined = a.merge(b, on=['year', 'date'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(joined) != len(a) or len(joined) != len(b):
        raise RuntimeError('Incomplete daily pairing')
    diff = (joined['rank_ic_a'] - joined['rank_ic_b']).to_numpy()
    return float(diff.mean()), common.newey_west_tstat(diff, lag=5)


def scores(variant):
    rows = [summary(y, variant) for y in YEARS]
    elig = np.array([r['eligible_metrics']['mean_rank_ic'] for r in rows])
    full = np.array([r['outer_metrics']['mean_rank_ic'] for r in rows])
    best = [r['lightgbm']['best_iteration'] for r in rows]
    t = [r['diagnostics']['trees'] for r in rows]
    m = [r['outer_metrics'] for r in rows]
    return dict(variant=variant, **{f'eligible_{y}': v for y, v in zip(YEARS, elig)},
                eligible_score=float(elig.mean() - PENALTY * elig.std(ddof=0)),
                **{f'all_{y}': v for y, v in zip(YEARS, full)},
                all_score=float(full.mean() - PENALTY * full.std(ddof=0)),
                negative_fold=bool((elig < 0).any()), truncated=any(truncated(b) for b in best),
                best_iterations=best,
                skipped_rounds=[r['lightgbm'].get('skipped_rounds', 0) for r in rows],
                lambda1_by_year=[r['lightgbm']['params']['lambda_l1'] for r in rows],
                mean_leaves=float(np.mean([x['mean_leaves'] for x in t])),
                mean_depth=float(np.mean([x['mean_depth'] for x in t])),
                fit_minus_es=float(np.mean([r['diagnostics']['fit_minus_early_stop_rank_ic'] for r in rows])),
                top50_minus_pool=float(np.mean([x['top50_minus_pool_5d_simple_return'] for x in m])),
                top50_overlap=float(np.mean([x['consecutive_day_top50_overlap'] for x in m])),
                train_seconds=float(np.mean([r['timing_seconds']['train'] for r in rows])),
                total_seconds=float(np.sum([r['timing_seconds']['total'] for r in rows])))


# ---------------------------------------------------------------- step 1: placebo
def placebo(threads):
    year = 2021
    run_job(year, 'placebo', threads)
    s = summary(year, 'placebo')
    d = pd.read_parquet(variant_dir(year, 'placebo') / 'eligible_daily_metrics.parquet')
    ric = d['rank_ic'].to_numpy(float)
    ric = ric[np.isfinite(ric)]
    mean, t = float(ric.mean()), common.newey_west_tstat(ric, lag=5)
    result = dict(year=year, eligible_mean_rank_ic=mean, nw_t_vs_zero=t, days=len(ric),
                  all_sample_mean_rank_ic=s['outer_metrics']['mean_rank_ic'],
                  best_iteration=s['lightgbm']['best_iteration'], rounds_trained=s['lightgbm']['rounds_trained'],
                  trees_built=s['lightgbm'].get('trees_built'), early_stop_rank_ic=s['lightgbm']['early_stop_best_rank_ic'],
                  inner_fit_rank_ic=s['diagnostics']['inner_fit_rank_ic'],
                  d_x1_same_fold_eligible=summary(year, combo_name(1.0, 1000))['eligible_metrics']['mean_rank_ic'],
                  criterion='|mean| < 0.01 and |NW t| < 2', passed=bool(abs(mean) < 0.01 and abs(t) < 2))
    common.save_json(OUT / 'placebo.json', result)
    print(f'[placebo] eligible mean RankIC {mean:.5f}, NW t {t:.2f} -> {"PASSED" if result["passed"] else "FAILED"}', flush=True)
    if not result['passed']:
        raise SystemExit('Placebo test failed; stopping before any further step')
    return result


PLACEBO_FOLDS = {2021: 20, 2018: 10}  # seeds per fold, consecutive from PLACEBO_SEED (plan section 14 item 7)


def placebo_multi(year, n_seeds, threads):
    """Null distribution for one fold: the same placebo with n consecutive seeds, then the fixed three-part rule."""
    rows = []
    for seed in range(PLACEBO_SEED, PLACEBO_SEED + n_seeds):
        variant = 'placebo' if seed == PLACEBO_SEED else f'placebo_s{seed}'
        run_job(year, variant, threads)
        s = summary(year, variant)
        d = pd.read_parquet(variant_dir(year, variant) / 'eligible_daily_metrics.parquet')
        ric = d['rank_ic'].to_numpy(float)
        ric = ric[np.isfinite(ric)]
        rows.append(dict(seed=seed, eligible_mean_rank_ic=float(ric.mean()), nw_t_vs_zero=common.newey_west_tstat(ric, lag=5),
                         all_sample_mean_rank_ic=s['outer_metrics']['mean_rank_ic'],
                         best_iteration=s['lightgbm']['best_iteration'],
                         early_stop_rank_ic=s['lightgbm']['early_stop_best_rank_ic']))
        print(f'[placebo {year} seed {seed}] eligible {rows[-1]["eligible_mean_rank_ic"]:.5f} t {rows[-1]["nw_t_vs_zero"]:.2f}',
              flush=True)
    table = pd.DataFrame(rows)
    real = summary(year, combo_name(1.0, 1000))['eligible_metrics']['mean_rank_ic']
    vals = table['eligible_mean_rank_ic'].to_numpy()
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    rules = dict(
        rule1_abs_mean_below_20pct_of_real=dict(value=abs(mean), limit=0.2 * real, passed=bool(abs(mean) < 0.2 * real)),
        rule2_every_seed_abs_below_third_of_real=dict(value=float(np.abs(vals).max()), limit=real / 3,
                                                      passed=bool((np.abs(vals) < real / 3).all())),
        rule3_real_above_mean_plus_5sd=dict(value=real, limit=mean + 5 * sd, passed=bool(real > mean + 5 * sd)))
    result = dict(year=year, seeds=[int(s) for s in table['seed']], table=json_records(table), real_d_x1_eligible=real,
                  null_mean=mean, null_std=sd, null_min=float(vals.min()), null_max=float(vals.max()),
                  real_z_vs_null=float((real - mean) / sd), rules=rules,
                  passed=all(r['passed'] for r in rules.values()))
    table.to_csv(OUT / f'placebo_multi_{year}.csv', index=False, encoding='utf-8')
    common.save_json(OUT / f'placebo_multi_{year}.json', result)
    print(f'[placebo {year}] null mean {mean:.5f} sd {sd:.5f} range [{vals.min():.5f}, {vals.max():.5f}]; real {real:.5f} '
          f'(z {result["real_z_vs_null"]:.1f}); rules ' + ', '.join(f'{k.split("_")[0]}={"ok" if v["passed"] else "FAIL"}'
                                                                   for k, v in rules.items())
          + f' -> {"PASSED" if result["passed"] else "FAILED"}', flush=True)
    return result


def placebo_gate(threads):
    results = {year: placebo_multi(year, n, threads) for year, n in PLACEBO_FOLDS.items()}
    gate = dict(passed=all(r['passed'] for r in results.values()),
                failed=[dict(year=y, rules=[k for k, v in r['rules'].items() if not v['passed']])
                        for y, r in results.items() if not r['passed']])
    common.save_json(OUT / 'placebo_gate.json', gate)
    if not gate['passed']:
        raise SystemExit(f'Multi-seed placebo failed: {gate["failed"]}; stopping before steps 2-5')
    print('[placebo gate] both folds passed', flush=True)
    return gate


# ---------------------------------------------------------------- step 2: Elastic(D)
def elastic_d_dir(year):
    return cache.cache_dir(year, 'full') / 'elastic_D'


def elastic_d(year):
    from run_lgbm_v2_pilot import fit_elastic, predict_elastic
    out = elastic_d_dir(year)
    if (out / 'metrics.json').exists():
        return common.read_json(out / 'metrics.json')
    out.mkdir(exist_ok=True)
    data_dir = cache.cache_dir(year, 'full')
    names = common.feature_names(year, 'full')
    x_fit = np.load(data_dir / 'fit_x.npy')
    model = fit_elastic(x_fit, d_label(year), names)  # alpha = 0.1 x critical value recomputed on the D label
    del x_fit
    outer_x = np.load(data_dir / 'outer_x.npy', mmap_mode='r')
    pred = np.concatenate([predict_elastic(model, np.asarray(outer_x[s:s + 200_000]))
                           for s in range(0, len(outer_x), 200_000)])
    outer = pd.read_parquet(data_dir / 'outer_meta.parquet')
    frame = outer.assign(elastic=pred)
    elig, elig_daily = eligible_metrics(frame, 'elastic')
    allm, _ = common.evaluate_outer(frame, 'elastic')
    np.save(out / 'outer_pred.npy', pred)
    elig_daily.to_parquet(out / 'eligible_daily_metrics.parquet', index=False)
    common.save_json(out / 'model.json', dict(coefficient=np.asarray(model['coefficient']).tolist(),
                                              intercept=model['intercept'], alpha=model['alpha'],
                                              nonzero=model['nonzero'], iterations=model['iterations'],
                                              label='D (daily cross-sectional rank return)'))
    metrics = dict(year=year, eligible=elig, all_sample_mean_rank_ic=allm['mean_rank_ic'], alpha=model['alpha'],
                   nonzero=model['nonzero'], iterations=model['iterations'])
    common.save_json(out / 'metrics.json', metrics)
    print(f'[elastic D {year}] eligible RankIC {elig["mean_rank_ic"]:.5f} (alpha {model["alpha"]:.3g}, '
          f'{model["nonzero"]} nonzero)', flush=True)
    return metrics


def elastic_daily(kind, years=YEARS):
    parts = []
    for y in years:
        if kind == 'D':
            d = pd.read_parquet(elastic_d_dir(y) / 'eligible_daily_metrics.parquet')
        else:
            from run_lgbm_v2_stage2 import elastic_eligible
            d = elastic_eligible(y)[1]
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        parts.append(d.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def elastic_step():
    from run_lgbm_v2_stage2 import elastic_eligible
    rows = []
    for year in YEARS:
        m = elastic_d(year)
        rows.append(dict(year=year, elastic_D=m['eligible']['mean_rank_ic'],
                         elastic_raw=elastic_eligible(year)[0]['mean_rank_ic'], alpha_D=m['alpha'], nonzero_D=m['nonzero']))
    table = pd.DataFrame(rows)
    e_d, e_raw = table['elastic_D'].to_numpy(), table['elastic_raw'].to_numpy()
    d_vs_raw = nw(elastic_daily('D'), elastic_daily('raw'))
    lgbm = daily(combo_name(1.0, 1000))
    per_fold = {}
    for y in YEARS:
        per_fold[y] = nw(lgbm.loc[lgbm['year'] == y], elastic_daily('D', [y]))
    pooled = nw(lgbm, elastic_daily('D'))
    result = dict(table=json_records(table), elastic_D_score=float(e_d.mean() - PENALTY * e_d.std(ddof=0)),
                  elastic_raw_score=float(e_raw.mean() - PENALTY * e_raw.std(ddof=0)),
                  elastic_D_vs_raw=dict(mean_diff=d_vs_raw[0], t=d_vs_raw[1]),
                  lgbm_D_x1_vs_elastic_D=dict(per_fold={y: dict(mean_diff=v[0], t=v[1]) for y, v in per_fold.items()},
                                              pooled=dict(mean_diff=pooled[0], t=pooled[1])))
    common.save_json(OUT / 'elastic_d.json', result)
    table.to_csv(OUT / 'elastic_d_table.csv', index=False, encoding='utf-8')
    print(f'[elastic D] score {result["elastic_D_score"]:.5f} (raw {result["elastic_raw_score"]:.5f}); '
          f'LGBM D x1 vs Elastic(D) pooled t {pooled[1]:.2f}', flush=True)
    return result


def json_records(frame):
    import json
    return json.loads(frame.to_json(orient='records'))


# ---------------------------------------------------------------- step 3: structure / lambda2 under D -> P*D
def neighbours(choice, combos):
    m, l2 = choice
    out = []
    same_l2 = sorted(a for a, b in combos if b == l2)
    same_m = sorted(b for a, b in combos if a == m)
    i, j = same_l2.index(m), same_m.index(l2)
    if i > 0: out.append((same_l2[i - 1], l2))
    if i < len(same_l2) - 1: out.append((same_l2[i + 1], l2))
    if j > 0: out.append((m, same_m[j - 1]))
    if j < len(same_m) - 1: out.append((m, same_m[j + 1]))
    return out


def structure_step(threads):
    for mult, l2 in D_COMBOS:
        for year in YEARS:
            run_job(year, combo_name(mult, l2), threads)
    for year in YEARS:
        run_job(year, 'mediumD', threads)
    table = pd.DataFrame([dict(scores(combo_name(m, l2)), mult=m, l2=l2) for m, l2 in D_COMBOS])
    cand = table.loc[~table['negative_fold'] & ~table['truncated']]
    if not len(cand):
        raise RuntimeError('No admissible deep D combination')
    top = cand.sort_values('eligible_score', ascending=False).iloc[0]
    base = daily(top['variant'])
    table['t_vs_T'] = [0.0 if v == top['variant'] else nw(base, daily(v))[1] for v in table['variant']]
    table['mean_diff_T_minus'] = [0.0 if v == top['variant'] else nw(base, daily(v))[0] for v in table['variant']]
    table['in_plateau'] = ~table['negative_fold'] & ~table['truncated'] & (table['t_vs_T'].abs() < 2)
    plateau = table.loc[table['in_plateau']]
    plateau_set = {(r['mult'], r['l2']) for _, r in plateau.iterrows()}
    combos = [(m, l2) for m, l2 in D_COMBOS]
    order, remaining = [], plateau.copy()
    while len(remaining):
        least = remaining['mean_leaves'].min()
        tied = remaining.loc[remaining['mean_leaves'] <= least * 1.05]
        pick = tied.sort_values(['l2', 'mean_leaves'], ascending=[False, True]).iloc[0]
        order.append(pick)
        remaining = remaining.loc[remaining['variant'] != pick['variant']]
    checks, chosen = [], None
    for pick in order:
        nb = neighbours((pick['mult'], pick['l2']), combos)
        missing = [n for n in nb if n not in plateau_set]
        checks.append(dict(variant=pick['variant'], mean_leaves=float(pick['mean_leaves']),
                           neighbours=[list(map(float, n)) for n in nb],
                           neighbours_outside_plateau=[list(map(float, n)) for n in missing]))
        if not missing:
            chosen = pick
            break
    if chosen is None:
        raise RuntimeError('No plateau member passes the neighbourhood check')
    medium = scores('mediumD')
    med_vs = nw(daily('mediumD'), daily(chosen['variant']))
    result = dict(p_star_d=dict(variant=chosen['variant'], mult=float(chosen['mult']), l2=float(chosen['l2'])),
                  T=top['variant'], plateau=list(plateau['variant']), neighbourhood_checks=checks,
                  medium_backup=dict(score=medium['eligible_score'], all_score=medium['all_score'],
                                     mean_leaves=medium['mean_leaves'], top50_minus_pool=medium['top50_minus_pool'],
                                     top50_overlap=medium['top50_overlap'], mean_diff_vs_p_star_d=med_vs[0],
                                     t_vs_p_star_d=med_vs[1], negative_fold=medium['negative_fold'],
                                     truncated=medium['truncated'], lambda1_by_year=medium['lambda1_by_year']))
    table.to_csv(OUT / 'structure_d_table.csv', index=False, encoding='utf-8')
    common.save_json(OUT / 'p_star_d.json', result)
    print(f'[P*D] {chosen["variant"]} (T = {top["variant"]}, plateau {len(plateau)}); medium backup t vs P*D {med_vs[1]:.2f}',
          flush=True)
    return result


# ---------------------------------------------------------------- step 4: sampling and binning -> P3
def baseline_variant():
    mult, l2 = p_star_d()
    return combo_name(mult, l2)


def sampling_step(threads):
    base = baseline_variant()
    for name in SAMPLING_PARAMS:
        for year in YEARS:
            run_job(year, name, threads)
    base_s, base_d = scores(base), daily(base)
    rows, adopted = [], {}
    for item, values in SAMPLING.items():
        for name in values:
            s = scores(name)
            diff, t = nw(daily(name), base_d)
            ok = (s['eligible_score'] > base_s['eligible_score'] and not s['negative_fold'] and not s['truncated']
                  and t >= 2)
            rows.append(dict(s, item=item, mean_diff_vs_base=diff, t_vs_base=t, qualifies=ok))
            if ok and (item not in adopted or s['eligible_score'] > adopted[item]['eligible_score']):
                adopted[item] = dict(name=name, eligible_score=s['eligible_score'])
    decision = dict(base=base, adopted_items={k: v['name'] for k, v in adopted.items()})
    if len(adopted) >= 2:
        combo = '+'.join(v['name'] for v in adopted.values())
        for year in YEARS:
            run_job(year, combo, threads)
        best_single = max(adopted.values(), key=lambda v: v['eligible_score'])['name']
        cs = scores(combo)
        diff, t = nw(daily(combo), daily(best_single))
        rows.append(dict(cs, item='combined', mean_diff_vs_base=nw(daily(combo), base_d)[0],
                         t_vs_base=nw(daily(combo), base_d)[1], qualifies=None))
        decision.update(combined=combo, best_single=best_single, combined_vs_best_single=dict(mean_diff=diff, t=t))
        decision['p3'] = combo if (t >= 2 and not cs['negative_fold'] and not cs['truncated']) else best_single
    elif len(adopted) == 1:
        decision['p3'] = next(iter(adopted.values()))['name']
    else:
        decision['p3'] = base
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'sampling_table.csv', index=False, encoding='utf-8')
    common.save_json(OUT / 'p3.json', decision)
    print(f'[P3] {decision["p3"]} (adopted items: {decision["adopted_items"] or "none"})', flush=True)
    return decision


# ---------------------------------------------------------------- step 5: summary
def f(x, d=4):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def per_fold_t_vs_elastic(variant):
    lg = daily(variant)
    return [nw(lg.loc[lg['year'] == y], elastic_daily('D', [y]))[1] for y in YEARS]


def summarize():
    pl = common.read_json(OUT / 'placebo.json')
    el = common.read_json(OUT / 'elastic_d.json')
    ps = common.read_json(OUT / 'p_star_d.json')
    p3 = common.read_json(OUT / 'p3.json')
    st = pd.read_csv(OUT / 'structure_d_table.csv')
    sa = pd.read_csv(OUT / 'sampling_table.csv')
    base = p3['base']
    L = ['## A. 安慰剂检验（D ×1 配置，D 标签在每个信号日内随机置换；多种子，计划第 14 节第 7 条）', '',
         f"单种子（2021，种子 20261117）：资格池 Rank IC {f(pl['eligible_mean_rank_ic'], 5)}，NW t {f(pl['nw_t_vs_zero'], 2)}，"
         f"按原标准（|均值| < 0.01）未通过，标准随后按评审决定修订为下面的多种子标准。", '',
         '| 折 | 种子数 | 跨种子均值 | 标准差 | 最小 | 最大 | 真实 D ×1 | 真实值的 z | ① |均值| < 20% 真实 | ② 每个 |值| < 1/3 真实 | ③ 真实 > 均值 + 5σ | 结论 |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for year in PLACEBO_FOLDS:
        r = common.read_json(OUT / f'placebo_multi_{year}.json')
        ru = r['rules']
        cell = lambda k: f"{'✓' if ru[k]['passed'] else '✗'} {f(ru[k]['value'], 4)} vs {f(ru[k]['limit'], 4)}"
        L.append(f"| {year} | {len(r['seeds'])} | {f(r['null_mean'], 5)} | {f(r['null_std'], 5)} | {f(r['null_min'], 5)} | "
                 f"{f(r['null_max'], 5)} | {f(r['real_d_x1_eligible'])} | {r['real_z_vs_null']:.1f} | "
                 f"{cell('rule1_abs_mean_below_20pct_of_real')} | {cell('rule2_every_seed_abs_below_third_of_real')} | "
                 f"{cell('rule3_real_above_mean_plus_5sd')} | **{'通过' if r['passed'] else '不通过'}** |")
    L += ['', '各种子明细见 `lgbm_stage3/placebo_multi_2021.csv`、`placebo_multi_2018.csv`。', '',
         '## B. 同标签线性对照 Elastic(D)', '',
         '| 折 | Elastic(D) 资格池 Rank IC | Elastic(原始) 资格池 Rank IC | Elastic(D) alpha | 非零系数 | LightGBM D×1 对 Elastic(D) 差 | NW t |',
         '|---:|---:|---:|---:|---:|---:|---:|']
    for r in el['table']:
        pf = el['lgbm_D_x1_vs_elastic_D']['per_fold'][str(r['year'])]
        L.append(f"| {r['year']} | {f(r['elastic_D'])} | {f(r['elastic_raw'])} | {r['alpha_D']:.3g} | {r['nonzero_D']} | "
                 f"{f(pf['mean_diff'], 5)} | {f(pf['t'], 2)} |")
    po = el['lgbm_D_x1_vs_elastic_D']['pooled']
    L += ['', f"- 选择分数：Elastic(D) **{f(el['elastic_D_score'])}**，Elastic(原始) {f(el['elastic_raw_score'])}；"
              f"Elastic(D) 对 Elastic(原始) 日度差 {f(el['elastic_D_vs_raw']['mean_diff'], 5)}，NW t = {f(el['elastic_D_vs_raw']['t'], 2)}。",
          f"- LightGBM D ×1 对 Elastic(D)，五折拼接：日度差 {f(po['mean_diff'], 5)}，NW t = **{f(po['t'], 2)}**。", '',
          f"## C. D 标签下的结构与 λ2 复核；P*D = **{ps['p_star_d']['variant']}**", '',
          f"T = {ps['T']}；平台（与 T 的 |t| < 2，含 T）= {', '.join(ps['plateau'])}。", '',
          '| 组合 | 逐折 λ1 | 2017 | 2018 | 2019 | 2020 | 2021 | 资格池分数 | 全样本分数 | 平均叶 | 对 T 的 t | 平台 | 负折 | 截断 | best_iteration | 跳过轮次 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|']
    for _, r in st.sort_values('eligible_score', ascending=False).iterrows():
        L.append(f"| {r['variant']} | {r['lambda1_by_year']} | " + ' | '.join(f(r[f'eligible_{y}']) for y in YEARS) +
                 f" | **{f(r['eligible_score'])}** | {f(r['all_score'])} | {r['mean_leaves']:.1f} | {f(r['t_vs_T'], 2)} | "
                 f"{'是' if r['in_plateau'] else ''} | {'是' if r['negative_fold'] else '否'} | {'是' if r['truncated'] else '否'} | "
                 f"{r['best_iterations']} | {r['skipped_rounds']} |")
    L += ['', '规则 4–5 检查顺序：']
    for c in ps['neighbourhood_checks']:
        L.append(f"- {c['variant']}：平均叶子 {c['mean_leaves']:.1f}；邻居 {c['neighbours']}；不在平台的邻居 "
                 f"{c['neighbours_outside_plateau'] or '无'}" + (' → **选中**' if c['variant'] == ps['p_star_d']['variant'] else ' → 退到下一个'))
    mb = ps['medium_backup']
    L += ['', f"备选中 0.1%（D 标签，λ1 = {mb['lambda1_by_year']}，λ2 = 1000）：资格池分数 {f(mb['score'])}、全样本 {f(mb['all_score'])}、"
              f"平均叶子 {mb['mean_leaves']:.1f}、Top50−池 {f(mb['top50_minus_pool'], 5)}、Top50 重合 {f(mb['top50_overlap'], 3)}；"
              f"对 P*D 日度差 {f(mb['mean_diff_vs_p_star_d'], 5)}，NW t = {f(mb['t_vs_p_star_d'], 2)}。中结构继续只作第八阶段备选。", '',
          f"## D. 第三阶段：采样与分箱（基准 = P*D = {base}）", '',
          '| 项 | 组 | 2017 | 2018 | 2019 | 2020 | 2021 | 资格池分数 | 全样本分数 | 对基准差 | 对基准 t | 满足采用条件 | 负折 | 截断 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    bs = scores(base)
    L.append(f"| 基准 | {base} | " + ' | '.join(f(bs[f'eligible_{y}']) for y in YEARS) +
             f" | **{f(bs['eligible_score'])}** | {f(bs['all_score'])} | | | | {'是' if bs['negative_fold'] else '否'} | {'是' if bs['truncated'] else '否'} |")
    for _, r in sa.iterrows():
        L.append(f"| {r['item']} | {r['variant']} | " + ' | '.join(f(r[f'eligible_{y}']) for y in YEARS) +
                 f" | **{f(r['eligible_score'])}** | {f(r['all_score'])} | {f(r['mean_diff_vs_base'], 5)} | {f(r['t_vs_base'], 2)} | "
                 f"{'是' if r['qualifies'] is True else ('' if pd.isna(r['qualifies']) else '否')} | "
                 f"{'是' if r['negative_fold'] else '否'} | {'是' if r['truncated'] else '否'} |")
    L += ['', '### 每组明细（五折平均；对 Elastic(D) 为逐折配对 t）', '',
          '| 组 | 对 Elastic(D) t 2017/2018/2019/2020/2021 | best_iteration | 跳过轮次 | 截断 | 平均叶 | 内部−早停 IC | Top50−池 | Top50 重合 | 平均训练秒 |',
          '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for variant in [base] + list(sa['variant']):
        s = scores(variant)
        tt = per_fold_t_vs_elastic(variant)
        L.append(f"| {variant} | {' / '.join(f(v, 2) for v in tt)} | {'/'.join(map(str, s['best_iterations']))} | "
                 f"{'/'.join(map(str, s['skipped_rounds']))} | {'是' if s['truncated'] else '否'} | {s['mean_leaves']:.1f} | "
                 f"{f(s['fit_minus_es'])} | {f(s['top50_minus_pool'], 5)} | {f(s['top50_overlap'], 3)} | {s['train_seconds']:.0f} |")
    L += ['', '## E. 按收紧规则得到的 P3（只报告，由评审确认）', '',
          f"- 满足采用条件的项：{p3['adopted_items'] or '无'}"]
    if 'combined' in p3:
        L.append(f"- 合并组 {p3['combined']} 对单项最佳 {p3['best_single']}：日度差 {f(p3['combined_vs_best_single']['mean_diff'], 5)}，"
                 f"NW t = {f(p3['combined_vs_best_single']['t'], 2)}")
    L.append(f"- **P3 = {p3['p3']}**" + ('（即保留基准 P*D）' if p3['p3'] == base else ''))
    text = '\n'.join(L) + '\n'
    (OUT / 'stage3_tables.md').write_text(text, encoding='utf-8')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--one', nargs=2, metavar=('YEAR', 'VARIANT'))
    for flag in ('placebo', 'elastic', 'structure', 'sampling', 'summarize', 'all'):
        parser.add_argument(f'--{flag}', action='store_true')
    parser.add_argument('--placebo-multi', action='store_true', help='multi-seed placebo gate (plan section 14 item 7)')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.one:
        year = int(args.one[0])
        if year not in YEARS:
            raise ValueError('Not a search fold')
        run_one(year, args.one[1], args.threads)
        return
    if args.placebo:
        placebo(args.threads)  # original single-seed check (superseded by --placebo-multi, plan section 14 item 7)
    if args.placebo_multi or args.all:
        placebo_gate(args.threads)
    if args.elastic or args.all:
        elastic_step()
    if args.structure or args.all:
        structure_step(args.threads)
    if args.sampling or args.all:
        sampling_step(args.threads)
    if args.summarize or args.all:
        summarize()
    if args.all:
        print('[done] stage 3 round complete', flush=True)


if __name__ == '__main__':
    main()
