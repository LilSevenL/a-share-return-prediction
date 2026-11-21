"""Per-fold data cache for the V2 LightGBM search (plan sections 3, 4, 9).

One cache per (outer year, feature set) under quant_data/processed/v2/lgbm_cache/{year}_{set}/:
  fit_x.npy / fit_y.npy / fit_keys.parquet        inner fit set (signal and exit < Y-1-01-01)
  es_x.npy  / es_y.npy  / es_keys.parquet         inner early-stopping set (signal in Y-1, exit < Y-01-01)
  outer_x.npy / outer_meta.parquet                all outer-year rows: ts_code, signal_date, y, vol20, eligible
  full_sample_keys.parquet                        the 2M-row full training sample in priority order
  checks.json                                     boundary assertions, label recheck, subset check
  elastic/                                        same-key Elastic Net control, fitted once per fold
  manifest.json                                   rows, feature names, key/label hashes, generation params

The cache only changes how data are fetched: it calls the same lgbm_v2_common functions
(create_pool, create_samples, boundary_checks, fetch_matrix, fetch_outer_month) as the pilot,
in the same order, and stores their outputs. Loading verifies the manifest and raises on any
mismatch; it never rebuilds silently.

  python v2/models/lgbm_v2_cache.py --year 2021 --feature-set full
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lgbm_v2_common as common
from lgbm_v2_common import DATA, KEY

CACHE = DATA / 'v2' / 'lgbm_cache'
ROWS, EARLY_ROWS, LABEL_ROWS = 2_000_000, 300_000, 1000
CACHE_VERSION = 1
ARRAYS = {'fit_x': 2, 'fit_y': 1, 'es_x': 2, 'es_y': 1, 'outer_x': 2}


def cache_dir(year, feature_set):
    return CACHE / f'{year}_{feature_set}'


def generation_params(year, feature_set):
    return dict(cache_version=CACHE_VERSION, year=int(year), feature_set=feature_set, rows=ROWS,
                early_rows=EARLY_ROWS, sample_seed=common.SAMPLE_SEED, training_gate=False,
                sample_order='hash(ts_code, CAST(signal_date AS DATE), 20261025), ts_code, signal_date',
                row_order='signal_date, ts_code', dtype_x='float32', label_rows=LABEL_ROWS,
                elastic=dict(l1_ratio=0.5, alpha='0.1 x critical', preprocessing_rows=50_000, seed=42))


def key_hash(keys, y=None):
    """sha256 over the canonical text of (ts_code, signal_date[, y bytes]) in stored row order."""
    digest = hashlib.sha256()
    dates = pd.to_datetime(keys['signal_date']).dt.strftime('%Y-%m-%d')
    digest.update('\n'.join(keys['ts_code'].astype(str) + '|' + dates).encode())
    if y is not None:
        digest.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _save_keys(path, keys):
    frame = keys[KEY].copy()
    frame['signal_date'] = pd.to_datetime(frame['signal_date']).astype('datetime64[ns]')
    frame.to_parquet(path, index=False)


def build(year, feature_set, threads=8, memory='6GB'):
    from run_lgbm_v2_pilot import fit_elastic, predict_elastic
    if not 2017 <= year <= 2021:
        raise ValueError('Search caches are restricted to outer years 2017-2021')
    out = cache_dir(year, feature_set)
    if (out / 'manifest.json').exists():
        raise RuntimeError(f'{out} already has a manifest; remove the directory explicitly to rebuild')
    out.mkdir(parents=True, exist_ok=True)
    timing, started = {}, time.perf_counter()
    names = common.feature_names(year, feature_set)
    con = common.open_connection(out, threads, memory)

    t = time.perf_counter()
    common.create_views(con, year)
    pool_rows = common.create_pool(con, year)
    common.create_samples(con, year, ROWS, EARLY_ROWS)
    checks = dict(pool_rows=pool_rows, boundaries=common.boundary_checks(con, year))
    full_keys = con.execute('SELECT ts_code, signal_date FROM full_sample '
                            'ORDER BY priority, ts_code, signal_date').fetch_df()
    _save_keys(out / 'full_sample_keys.parquet', full_keys)
    checks['labels'] = common.label_check(con, 'full_sample', LABEL_ROWS)
    checks['subset'] = common.subset_check(full_keys, year)
    common.save_json(out / 'checks.json', checks)
    timing['pool_sampling_checks'] = round(time.perf_counter() - t, 1)
    print(f'[{year}] pool={pool_rows:,} full={len(full_keys):,} '
          f'fit={checks["boundaries"]["inner_fit"]["rows"]:,} early={checks["boundaries"]["early_stop"]["rows"]:,} '
          f'checks passed ({timing["pool_sampling_checks"]}s)', flush=True)

    hashes = dict(full_sample_keys=key_hash(full_keys))
    shapes = {}
    for prefix, table in (('fit', 'inner_fit'), ('es', 'early_stop')):
        t = time.perf_counter()
        x, y, keys = common.fetch_matrix(con, table, names)
        np.save(out / f'{prefix}_x.npy', x)
        np.save(out / f'{prefix}_y.npy', y)
        _save_keys(out / f'{prefix}_keys.parquet', keys)
        hashes[f'{prefix}_keys_and_y'] = key_hash(keys, y)
        shapes[f'{prefix}_x'], shapes[f'{prefix}_y'] = list(x.shape), list(y.shape)
        timing[f'fetch_{table}'] = round(time.perf_counter() - t, 1)
        print(f'[{year}] {table} {x.shape} {timing[f"fetch_{table}"]}s', flush=True)
        del x, y, keys

    t = time.perf_counter()
    xs, frames = [], []
    for month in common.outer_months(con, year):
        x_out, frame = common.fetch_outer_month(con, year, month, names)
        xs.append(x_out)
        frames.append(frame)
    con.close()
    outer_x = np.concatenate(xs)
    del xs
    meta = pd.concat(frames, ignore_index=True)
    if meta.duplicated(KEY).any() or not meta['signal_date'].is_monotonic_increasing:
        raise RuntimeError('Outer rows must be unique and sorted by date')
    if (pd.to_datetime(meta['signal_date']).dt.year != year).any():
        raise RuntimeError('Outer cache contains rows outside the outer year')
    np.save(out / 'outer_x.npy', outer_x)
    meta['signal_date'] = pd.to_datetime(meta['signal_date']).astype('datetime64[ns]')
    meta.to_parquet(out / 'outer_meta.parquet', index=False)
    hashes['outer_keys_and_y'] = key_hash(meta, meta['y'].to_numpy())
    shapes['outer_x'] = list(outer_x.shape)
    timing['fetch_outer'] = round(time.perf_counter() - t, 1)
    print(f'[{year}] outer {outer_x.shape} {timing["fetch_outer"]}s', flush=True)
    del outer_x

    t = time.perf_counter()
    elastic_dir = out / 'elastic'
    elastic_dir.mkdir(exist_ok=True)
    x_fit, y_fit = np.load(out / 'fit_x.npy'), np.load(out / 'fit_y.npy')
    model = fit_elastic(x_fit, y_fit, names)
    del x_fit
    outer_x = np.load(out / 'outer_x.npy', mmap_mode='r')
    pred = np.concatenate([predict_elastic(model, np.asarray(outer_x[s:s + 200_000]))
                           for s in range(0, len(outer_x), 200_000)])
    np.save(elastic_dir / 'outer_pred.npy', pred)
    frame = meta.assign(elastic=pred)
    metrics, daily = common.evaluate_outer(frame, 'elastic')
    metrics['timing_split'] = common.timing_selection_split(frame, 'elastic')
    daily.to_parquet(elastic_dir / 'daily_metrics.parquet', index=False)
    stats = model['stats']
    common.save_json(elastic_dir / 'model.json', dict(
        coefficient=np.asarray(model['coefficient']).tolist(), intercept=model['intercept'],
        alpha=model['alpha'], nonzero=model['nonzero'], iterations=model['iterations'],
        active_count=model['active_count'], active_names=stats['active_names'], preprocessing=stats))
    common.save_json(elastic_dir / 'metrics.json', metrics)
    timing['elastic'] = round(time.perf_counter() - t, 1)
    print(f'[{year}] elastic RankIC={metrics["mean_rank_ic"]:.5f} {timing["elastic"]}s', flush=True)

    timing['total'] = round(time.perf_counter() - started, 1)
    sources = [Path(common.__file__), Path(__file__), _source_path('v2/experiments/run_lgbm_v2_pilot.py'),
               _source_path('v2/models/research_controls.py'), _source_path('tune_elastic_net.py')]
    manifest = dict(params=generation_params(year, feature_set), feature_names=names,
                    rows=dict(training_pool=pool_rows, full_sample=len(full_keys),
                              inner_fit=shapes['fit_y'][0], early_stop=shapes['es_y'][0],
                              outer=shapes['outer_x'][0], outer_scored=int(np.isfinite(meta['y']).sum())),
                    shapes=shapes, hashes=hashes,
                    file_bytes={p.name: p.stat().st_size for p in sorted(out.glob('*.npy'))},
                    source_sha256_at_build={p.name: common.sha256(p) for p in sources},
                    elastic_outer_mean_rank_ic=metrics['mean_rank_ic'], timing_seconds=timing,
                    test_used=False)
    common.save_json(out / 'manifest.json', manifest)  # written last: its presence marks a complete cache
    print(f'[{year}] cache complete in {timing["total"]}s -> {out}', flush=True)
    return manifest


def load(year, feature_set, mmap_outer=True):
    """Load a fold cache, verifying it against the manifest; raises on any mismatch."""
    out = cache_dir(year, feature_set)
    path = out / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError(f'No complete cache at {out}; build it with lgbm_v2_cache.py')
    manifest = common.read_json(path)
    if manifest['params'] != generation_params(year, feature_set):
        raise RuntimeError(f'{out}: generation params differ from the current definition')
    names = common.feature_names(year, feature_set)
    if manifest['feature_names'] != names:
        raise RuntimeError(f'{out}: feature list differs from the fold definition')
    for name, size in manifest['file_bytes'].items():
        if (out / name).stat().st_size != size:
            raise RuntimeError(f'{out}: {name} size changed')
    data = dict(names=names, manifest=manifest, dir=out)
    for name in ARRAYS:
        mode = 'r' if (name == 'outer_x' and mmap_outer) else None
        data[name] = np.load(out / f'{name}.npy', mmap_mode=mode)
        if list(data[name].shape) != manifest['shapes'][name] or \
                (name.endswith('_x') and data[name].dtype != np.float32):
            raise RuntimeError(f'{out}: {name} shape/dtype mismatch')
    data['fit_keys'] = pd.read_parquet(out / 'fit_keys.parquet')
    data['es_keys'] = pd.read_parquet(out / 'es_keys.parquet')
    data['outer'] = pd.read_parquet(out / 'outer_meta.parquet')
    checks = {'fit_keys_and_y': key_hash(data['fit_keys'], data['fit_y']),
              'es_keys_and_y': key_hash(data['es_keys'], data['es_y']),
              'outer_keys_and_y': key_hash(data['outer'], data['outer']['y'].to_numpy())}
    for name, value in checks.items():
        if manifest['hashes'][name] != value:
            raise RuntimeError(f'{out}: {name} hash mismatch')
    for frame in (data['fit_keys'], data['es_keys'], data['outer']):
        frame['signal_date'] = frame['signal_date'].to_numpy().astype('datetime64[D]')
    data['elastic_pred'] = np.load(out / 'elastic' / 'outer_pred.npy')
    data['elastic_daily'] = pd.read_parquet(out / 'elastic' / 'daily_metrics.parquet')
    data['elastic_metrics'] = common.read_json(out / 'elastic' / 'metrics.json')
    if len(data['elastic_pred']) != len(data['outer']):
        raise RuntimeError(f'{out}: Elastic predictions do not match outer rows')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--year', type=int, nargs='+', required=True)
    parser.add_argument('--feature-set', choices=['full', 'compact'], default='full')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-limit', default='6GB')
    args = parser.parse_args()
    for year in args.year:
        build(year, args.feature_set, args.threads, args.memory_limit)
        load(year, args.feature_set)
        print(f'[{year}] reload and manifest verification passed', flush=True)


if __name__ == '__main__':
    main()
