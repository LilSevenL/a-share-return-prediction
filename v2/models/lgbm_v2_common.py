"""V2 LightGBM shared data access, sampling, metrics and diagnostics.

Reads only 2011-2023 local data. Never opens 2024+ partitions or the V1 permanent Test.
Sampling, boundaries and metric definitions follow v2/docs/LIGHTGBM_V2_PLAN.md.
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = _REPO_ROOT
V2 = ROOT / 'v2'
for _path in (ROOT, V2):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DATA = ROOT / 'quant_data' / 'processed'
FEATURES = DATA / 'v2' / 'local_features_2011_2023_stable'
TARGETS = DATA / 'targets_5d'
LEGACY = DATA / 'features_50'
PARQUET = DATA / 'parquet'
STOCK_BASIC = ROOT / 'quant_data' / 'combined' / 'stock_basic.csv'
SAMPLE_SEED = 20261025
KEY = ['ts_code', 'signal_date']
LAST_YEAR = 2023

CATALOG = json.loads((_source_path('configs/v2/research_design/factor_catalog_500.json')).read_text(encoding='utf-8'))
FAMILY = {r['name']: r['family'] for r in CATALOG['factors']}


# ---------------------------------------------------------------- utilities
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def peak_rss_gb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, 'peak_wset', info.rss) / 1024 ** 3
    except Exception:
        return float('nan')


class Timer:
    def __init__(self):
        self.records = {}

    def __call__(self, name):
        timer = self

        class _Block:
            def __enter__(self):
                self.start = time.perf_counter()

            def __exit__(self, *exc):
                timer.records[name] = round(time.perf_counter() - self.start, 3)
        return _Block()


# ---------------------------------------------------------------- factor lists
def feature_names(year, kind):
    """Fold-specific active factor list, identical to the V2 linear experiments."""
    stats = read_json(DATA / 'v2' / 'ols_baseline_stable' / f'validation_{year}' / 'preprocessing.json')
    names = list(stats['active_names'])
    if kind == 'full':
        expected = 448
    elif kind == 'compact':
        names = [n for n in names if FAMILY[n] not in ('C', 'M')]
        expected = 333
    else:
        raise ValueError(f'Unknown feature set {kind}')
    if len(names) != expected:
        raise RuntimeError(f'{year} {kind}: expected {expected} factors, found {len(names)}')
    return names


# ---------------------------------------------------------------- duckdb access
def parquet_files(root, partition, first, last):
    if last > LAST_YEAR:
        raise ValueError('2024+ partitions are prohibited')
    files = [p for year in range(first, last + 1) for p in sorted((root / f'{partition}={year}').glob('*.parquet'))]
    if not files:
        raise FileNotFoundError(f'{root} {partition} {first}-{last}')
    return files


def open_connection(output, threads=8, memory='6GB'):
    import duckdb
    con = duckdb.connect()
    con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET memory_limit={quote(memory)}')
    con.execute(f"SET temp_directory={quote(str(output / '_duckdb_tmp'))}")
    con.execute('SET preserve_insertion_order=false')
    return con


def create_views(con, last_year):
    """Views over 2011..last_year only. last_year is the outer year for scoring views."""
    if last_year > LAST_YEAR:
        raise ValueError('2024+ partitions are prohibited')
    sources = {
        'features': (FEATURES, 'feature_year'),
        'targets': (TARGETS, 'signal_year'),
        'legacy': (LEGACY, 'feature_year'),
        'daily': (PARQUET / 'daily', 'year'),
        'adj_factor': (PARQUET / 'adj_factor', 'year'),
    }
    for name, (root, partition) in sources.items():
        files = [str(p) for p in parquet_files(root, partition, 2011, last_year)]
        con.read_parquet(files, hive_partitioning=False).create_view(name, replace=True)
    basic = pd.read_csv(STOCK_BASIC, dtype={'list_date': str})[['ts_code', 'list_date']]
    basic['list_date'] = pd.to_datetime(basic['list_date'], format='%Y%m%d', errors='coerce')
    con.register('basic', basic)


def create_pool(con, year):
    """Training pool for outer year: finite raw label, signal and exit strictly before year."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pool AS
        SELECT f.ts_code, f.signal_date, CAST(t.exit_date AS DATE) exit_date,
               t.target_logret_5d_raw y,
               coalesce(l.feature_valid_count >= 30, false) feature_ok,
               hash(f.ts_code, f.signal_date, {SAMPLE_SEED}) priority
        FROM (SELECT ts_code, CAST(signal_date AS DATE) signal_date FROM features) f
        JOIN targets t ON t.ts_code = f.ts_code AND CAST(t.signal_date AS DATE) = f.signal_date
        LEFT JOIN legacy l ON l.ts_code = f.ts_code AND CAST(l.signal_date AS DATE) = f.signal_date
        WHERE isfinite(t.target_logret_5d_raw)
          AND f.signal_date >= DATE '2011-01-01'
          AND f.signal_date < DATE '{year}-01-01'
          AND t.exit_date < DATE '{year}-01-01'
    """)
    rows, keys = con.execute('SELECT count(*), count(DISTINCT (ts_code, signal_date)) FROM pool').fetchone()
    if rows != keys:
        raise RuntimeError('Duplicate keys in training pool')
    return rows


def create_samples(con, year, rows, early_rows, gate=False):
    """Full training sample, inner fit subset and inner early-stopping set (DATE-hash order)."""
    condition = 'feature_ok' if gate else 'TRUE'
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE full_sample AS
        SELECT * FROM pool WHERE {condition}
        ORDER BY priority, ts_code, signal_date LIMIT {int(rows)}
    """)
    boundary = f'{year - 1}-01-01'
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE inner_fit AS
        SELECT * FROM full_sample
        WHERE signal_date < DATE '{boundary}' AND exit_date < DATE '{boundary}'
    """)
    # Early stopping is an evaluation step: no training gate; exit strictly before outer year.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE early_stop AS
        SELECT * FROM pool
        WHERE year(signal_date) = {year - 1} AND exit_date < DATE '{year}-01-01'
        ORDER BY priority, ts_code, signal_date LIMIT {int(early_rows)}
    """)
    return boundary


def boundary_checks(con, year):
    boundary = f'{year - 1}-01-01'
    outer = f'{year}-01-01'
    checks = {}
    for table in ('full_sample', 'inner_fit', 'early_stop'):
        n, max_signal, max_exit = con.execute(
            f'SELECT count(*), max(signal_date), max(exit_date) FROM {table}').fetchone()
        checks[table] = dict(rows=int(n), max_signal_date=str(max_signal), max_exit_date=str(max_exit))
    fit = checks['inner_fit']
    early = checks['early_stop']
    full = checks['full_sample']
    assert fit['rows'] > 0 and early['rows'] > 0, 'Empty inner fit or early-stopping set'
    assert fit['max_exit_date'] < boundary and fit['max_signal_date'] < boundary, 'Inner fit crosses Y-1'
    assert early['max_exit_date'] < outer, 'Early-stopping labels use outer-year prices'
    assert full['max_exit_date'] < outer and full['max_signal_date'] < outer, 'Training sample crosses outer year'
    low = con.execute(f"SELECT min(signal_date) FROM early_stop").fetchone()[0]
    assert str(low) >= boundary, 'Early-stopping set starts before Y-1'
    checks['assertions'] = 'passed'
    return checks


def fetch_matrix(con, table, names, batch_rows=200_000):
    """Raw float32 features for keys in `table`, sorted by date then code. NaN preserved."""
    total = con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
    columns = ', '.join(f'CAST(f."{n}" AS FLOAT) AS "{n}"' for n in names)
    query = f"""
        SELECT k.ts_code, k.signal_date, k.y, {columns}
        FROM {table} k JOIN features f
          ON f.ts_code = k.ts_code AND CAST(f.signal_date AS DATE) = k.signal_date
        ORDER BY k.signal_date, k.ts_code
    """
    x = np.empty((total, len(names)), dtype=np.float32)
    codes, dates, y = [], [], np.empty(total, dtype=np.float64)
    position = 0
    reader = con.execute(query).fetch_record_batch(batch_rows)
    for batch in reader:
        n = batch.num_rows
        codes.append(batch.column(0).to_pylist())
        dates.append(batch.column(1).to_numpy(zero_copy_only=False))
        y[position:position + n] = batch.column(2).to_numpy(zero_copy_only=False)
        for j in range(len(names)):
            x[position:position + n, j] = batch.column(3 + j).to_numpy(zero_copy_only=False)
        position += n
    if position != total:
        raise RuntimeError(f'Feature join lost rows: {position} of {total}')
    keys = pd.DataFrame({'ts_code': np.concatenate(codes) if codes else [],
                         'signal_date': np.concatenate(dates).astype('datetime64[D]') if dates else []})
    return x, y, keys


def outer_months(con, year, max_months=None):
    months = [r[0] for r in con.execute(f"""
        SELECT DISTINCT month(CAST(signal_date AS DATE)) m FROM features
        WHERE CAST(signal_date AS DATE) >= DATE '{year}-01-01'
          AND CAST(signal_date AS DATE) < DATE '{year + 1}-01-01' ORDER BY m""").fetchall()]
    return months[:max_months] if max_months else months


def fetch_outer_month(con, year, month, names):
    """All feature rows of one outer month with scoring label, eligibility and volatility."""
    columns = ', '.join(f'CAST(f."{n}" AS FLOAT) AS "{n}"' for n in names)
    query = f"""
        SELECT f.ts_code, CAST(f.signal_date AS DATE) signal_date,
               CASE WHEN isfinite(t.target_logret_5d_raw) AND t.exit_date < DATE '{year + 1}-01-01'
                    THEN t.target_logret_5d_raw END y,
               CAST(f.volatility_20d AS DOUBLE) vol20,
               coalesce(d.amount >= 50000 AND b.list_date IS NOT NULL
                        AND date_diff('day', b.list_date, CAST(f.signal_date AS DATE)) >= 60, false) eligible,
               {columns}
        FROM features f
        LEFT JOIN targets t ON t.ts_code = f.ts_code AND CAST(t.signal_date AS DATE) = CAST(f.signal_date AS DATE)
        LEFT JOIN daily d ON d.ts_code = f.ts_code AND CAST(d.trade_date AS DATE) = CAST(f.signal_date AS DATE)
        LEFT JOIN basic b ON b.ts_code = f.ts_code
        WHERE CAST(f.signal_date AS DATE) >= make_date({year}, {month}, 1)
          AND CAST(f.signal_date AS DATE) < make_date({year}, {month}, 1) + INTERVAL 1 MONTH
        ORDER BY signal_date, f.ts_code
    """
    table = con.execute(query).to_arrow_table()
    frame = pd.DataFrame({
        'ts_code': table.column('ts_code').to_pylist(),
        'signal_date': np.asarray(table.column('signal_date').to_numpy()).astype('datetime64[D]'),
        'y': np.asarray(table.column('y').to_numpy(), dtype=float),
        'vol20': np.asarray(table.column('vol20').to_numpy(), dtype=float),
        'eligible': np.asarray(table.column('eligible').to_numpy(), dtype=bool),
    })
    x = np.empty((table.num_rows, len(names)), dtype=np.float32)
    for j, n in enumerate(names):
        x[:, j] = np.asarray(table.column(n).to_numpy(), dtype=np.float32)
    if frame.duplicated(KEY).any():
        raise RuntimeError('Duplicate outer keys')
    return x, frame


# ---------------------------------------------------------------- correctness checks
def label_check(con, table, rows=1000):
    """Recompute ln(adj close[exit] / adj open[entry]) from raw prices for a sample."""
    frame = con.execute(f"""
        WITH s AS (SELECT ts_code, signal_date, y FROM {table}
                   ORDER BY hash(ts_code, signal_date, 42), ts_code, signal_date LIMIT {int(rows)})
        SELECT s.ts_code, s.signal_date, s.y,
               ln(x.close * xa.adj_factor / (e.open * ea.adj_factor)) expected
        FROM s
        JOIN targets t ON t.ts_code = s.ts_code AND CAST(t.signal_date AS DATE) = s.signal_date
        LEFT JOIN daily e ON e.ts_code = s.ts_code AND CAST(e.trade_date AS DATE) = CAST(t.entry_date AS DATE)
        LEFT JOIN adj_factor ea ON ea.ts_code = s.ts_code AND CAST(ea.trade_date AS DATE) = CAST(t.entry_date AS DATE)
        LEFT JOIN daily x ON x.ts_code = s.ts_code AND CAST(x.trade_date AS DATE) = CAST(t.exit_date AS DATE)
        LEFT JOIN adj_factor xa ON xa.ts_code = s.ts_code AND CAST(xa.trade_date AS DATE) = CAST(t.exit_date AS DATE)
    """).fetch_df()
    ok = len(frame) == rows and np.allclose(frame['y'], frame['expected'], rtol=1e-9, atol=1e-10)
    if not ok:
        raise RuntimeError('Independent label recomputation failed')
    return dict(rows=len(frame), max_abs_error=float(np.max(np.abs(frame['y'] - frame['expected']))), status='passed')


def subset_check(keys, year):
    """Compare our DATE-hash sample with saved V2 sample keys; report, never assume."""
    ours = keys[KEY].copy()
    ours['signal_date'] = pd.to_datetime(ours['signal_date']).dt.normalize()
    references = {
        'elastic_gate_ablation_baseline_500k': DATA / 'v2' / 'elastic_gate_ablation' / f'{year}_baseline_keys.parquet',
        'expanded_losses_500k_timestamp_hash': DATA / 'v2' / 'expanded_losses_500k' / f'validation_{year}' / 'sample_keys.parquet',
    }
    report = {}
    for name, path in references.items():
        if not path.exists():
            report[name] = 'reference file missing'
            continue
        ref = pd.read_parquet(path, columns=KEY)
        ref['signal_date'] = pd.to_datetime(ref['signal_date']).dt.normalize()
        overlap = len(ours.merge(ref, on=KEY, how='inner'))
        report[name] = dict(reference_rows=len(ref), our_rows=len(ours), overlap=overlap,
                            reference_is_subset_of_ours=overlap == len(ref),
                            ours_is_subset_of_reference=overlap == len(ours),
                            overlap_share_of_reference=overlap / len(ref) if len(ref) else None)
    return report


# ---------------------------------------------------------------- metrics
def day_slices(dates):
    dates = np.asarray(dates)
    if len(dates) and np.any(dates[1:] < dates[:-1]):
        raise ValueError('Rows must be sorted by date')
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1] if len(dates) else np.array([], int)
    ends = np.r_[starts[1:], len(dates)] if len(dates) else np.array([], int)
    return starts, ends


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float('nan')


class DailyRankIC:
    """Fast repeated daily Rank IC on a fixed, date-sorted evaluation set."""

    def __init__(self, y, dates, min_stocks=20):
        self.slices = [(s, e) for s, e in zip(*day_slices(dates)) if e - s >= min_stocks]
        y = np.asarray(y, float)
        self.y_ranks = [rankdata(y[s:e]) for s, e in self.slices]

    def daily(self, pred):
        pred = np.asarray(pred, float)
        return np.array([_pearson(rankdata(pred[s:e]), r) for (s, e), r in zip(self.slices, self.y_ranks)])

    def __call__(self, pred):
        values = self.daily(pred)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else float('nan')


def newey_west_tstat(series, lag=5):
    """t statistic of the mean with Newey-West (Bartlett) variance; for daily IC differences."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * (lag + 1):
        return float('nan')
    e = x - x.mean()
    variance = np.dot(e, e) / n
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        variance += 2 * weight * np.dot(e[k:], e[:-k]) / n
    if variance <= 0:
        return float('nan')
    return float(x.mean() / math.sqrt(variance / n))


def head_and_turnover(frame, pred_column, top=50):
    """Top-N minus eligible-pool mean simple return, and day-to-day ranking stability."""
    elig = frame.loc[frame['eligible']].sort_values(['signal_date', pred_column, 'ts_code'],
                                                    ascending=[True, False, True])
    spreads, rank_corr, overlap = [], [], []
    previous = None
    for _, day in elig.groupby('signal_date', sort=True):
        y = day['y'].to_numpy()
        head = y[:top]
        if np.isfinite(head).any() and np.isfinite(y).any():
            spreads.append(np.nanmean(np.expm1(head)) - np.nanmean(np.expm1(y)))
        current = day[['ts_code', pred_column]]
        if previous is not None:
            joined = previous.merge(current, on='ts_code', suffixes=('_a', '_b'))
            if len(joined) >= 20:
                rank_corr.append(_pearson(rankdata(joined[pred_column + '_a']), rankdata(joined[pred_column + '_b'])))
            overlap.append(len(set(previous['ts_code'].head(top)) & set(current['ts_code'].head(top))) / top)
        previous = current
    return dict(top50_minus_pool_5d_simple_return=float(np.mean(spreads)) if spreads else None,
                consecutive_day_rank_corr=float(np.nanmean(rank_corr)) if rank_corr else None,
                consecutive_day_top50_overlap=float(np.mean(overlap)) if overlap else None,
                head_days=len(spreads))


def evaluate_outer(frame, pred_column):
    """Common V2 metrics (same function as the linear experiments) plus head and turnover diagnostics."""
    from research_controls import WeightConfig, evaluate_common, sample_weights
    scored = frame.loc[np.isfinite(frame['y'])]
    _, weights = sample_weights(scored['signal_date'].to_numpy(), scored['vol20'].to_numpy(),
                                WeightConfig(date_gamma=0, volatility_power=1))
    metrics, daily = evaluate_common(scored[pred_column].to_numpy(), scored['y'].to_numpy(),
                                     scored['signal_date'].to_numpy(), weights)
    ric = np.array([d['rank_ic'] for d in daily], float)
    metrics['rank_ic_positive_share'] = float(np.mean(ric[np.isfinite(ric)] > 0)) if len(ric) else None
    metrics.update(head_and_turnover(frame, pred_column))
    return metrics, pd.DataFrame(daily)


def timing_selection_split(frame, pred_column):
    """How much of the prediction varies over time (market timing) versus across stocks."""
    scored = frame.loc[np.isfinite(frame['y'])]
    pred = scored[pred_column].to_numpy(float)
    by_day = scored.groupby('signal_date')
    day_mean_pred = by_day[pred_column].transform('mean').to_numpy(float)
    total = float(np.var(pred))
    daily_pred = by_day[pred_column].mean()
    daily_y = by_day['y'].mean()
    return dict(timing_variance_share=float(np.var(day_mean_pred) / total) if total > 0 else None,
                timing_correlation=float(np.corrcoef(daily_pred, daily_y)[0, 1]) if len(daily_pred) > 2 else None,
                mean_cross_sectional_std=float(by_day[pred_column].std().mean()))


# ---------------------------------------------------------------- LightGBM helpers
class RankICEarlyStopping:
    """Evaluate daily Rank IC on the inner early-stopping set every `every` rounds.

    Used both as `feval` (cheap incremental predictions supplied by LightGBM) and as an
    after-iteration callback that stops after `patience` rounds without improvement.
    """

    def __init__(self, y, dates, every=10, patience=200, min_delta=0.0):
        self.metric = DailyRankIC(y, dates)
        self.every, self.patience, self.min_delta = every, patience, min_delta
        self.iteration = 0
        self.last = float('nan')
        self.curve = []
        self.best_score = -float('inf')
        self.best_iteration = 0
        self.total_rounds = None
        self.started = time.perf_counter()
        self.lap_times = []

    def before(self, env):
        self.iteration = env.iteration + 1
        self.total_rounds = env.end_iteration

    def feval(self, preds, _data):
        if self.iteration % self.every == 0 or self.iteration == self.total_rounds:
            self.last = self.metric(preds)
            self.curve.append((self.iteration, self.last, time.perf_counter() - self.started))
        return 'daily_rank_ic', self.last, True

    def after(self, env):
        import lightgbm as lgb
        it = env.iteration + 1
        if it % 100 == 0:
            self.lap_times.append((it, time.perf_counter() - self.started))
        if not (it % self.every == 0 or it == env.end_iteration):
            return
        if math.isfinite(self.last) and self.last > self.best_score + self.min_delta:
            self.best_score, self.best_iteration = self.last, it
        elif it - self.best_iteration >= self.patience:
            raise lgb.callback.EarlyStopException(self.best_iteration - 1, [('early_stop', 'daily_rank_ic', self.best_score, True)])

    def hit_round_limit(self, cap):
        return self.best_iteration > cap - self.patience

    def callbacks(self):
        import functools
        before = functools.partial(RankICEarlyStopping.before, self)
        before.before_iteration = True
        before.order = 10
        after = functools.partial(RankICEarlyStopping.after, self)
        after.before_iteration = False
        after.order = 30
        return [before, after]


def tree_diagnostics(booster, num_iteration):
    model = booster.dump_model(num_iteration=num_iteration)
    leaves, depths = [], []

    def depth(node):
        if 'split_index' not in node:
            return 0
        return 1 + max(depth(node['left_child']), depth(node['right_child']))
    for tree in model['tree_info']:
        leaves.append(tree['num_leaves'])
        depths.append(depth(tree['tree_structure']))
    gains = booster.feature_importance(importance_type='gain', iteration=num_iteration)
    names = booster.feature_name()
    order = np.argsort(gains)[::-1][:20]
    return dict(trees=len(leaves), mean_leaves=float(np.mean(leaves)), max_leaves=int(np.max(leaves)),
                mean_depth=float(np.mean(depths)), max_depth=int(np.max(depths)),
                top20_gain=[(names[i], float(gains[i])) for i in order],
                gain_share_top20=float(gains[order].sum() / gains.sum()) if gains.sum() > 0 else None)


def leaf_date_coverage(booster, x, dates, num_iteration, trees=10, rows=200_000, seed=42):
    """Distinct trading days per leaf on a training subsample, for a spread of trees."""
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(x), min(rows, len(x)), replace=False))
    sub_x, sub_dates = x[index], np.asarray(dates)[index]
    picks = sorted(set(np.linspace(0, max(num_iteration - 1, 0), trees).astype(int)))
    medians, small = [], []
    for t in picks:
        leaf = booster.predict(sub_x, pred_leaf=True, start_iteration=int(t), num_iteration=1).reshape(-1)
        days = pd.DataFrame({'leaf': leaf, 'day': sub_dates}).groupby('leaf')['day'].nunique()
        medians.append(float(days.median()))
        small.append(float((days <= 20).mean()))
    return dict(sampled_rows=len(index), trees_checked=picks, median_days_per_leaf=medians,
                share_leaves_with_20_days_or_fewer=small)
