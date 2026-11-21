"""V2 LightGBM input stage (plan section 16 item 4): factor set x training-sample gate, then daily
cross-sectional rank factors. Training label D throughout; early stopping and every score stay on the raw label
(eligible daily Rank IC for selection). Base configuration = P4 = stage-3 baseline (deep 0.1%, lambda1 x1,
lambda2 10000, ff 0.3, lr 0.02, cap 6000, patience 200, R1).

Variant name: {factor set}_{sample}_{input}_{structure}_x{lambda1 multiplier}
  factor set full (448) | compact (333, columns selected from the 448 cache, order = feature_names(year, 'compact'))
  sample     none | gate (old-50 feature_valid_count >= 30 applied to the training pool before the 2M draw;
             early-stopping set and outer rows unchanged, early-stopping set not gated)
  input      raw | rank ((average rank - 0.5) / valid stocks that day - 0.5 on the full daily cross-section)
  structure  deep (511 leaves, depth -1, lambda1 = sig2(50 x ratio_D) x mult, lambda2 10000)
             medium (127 leaves, depth 8, lambda1 = sig2(10 x ratio_D), lambda2 1000)
full_none_raw_deep_x1 is P4 itself: the stage-3 baseline fits are reused, not refitted.

  python v2/experiments/run_lgbm_v2_input.py --all            gate check -> step 1 -> I1 -> rank cache -> step 2 -> summary
  python v2/experiments/run_lgbm_v2_input.py --one YEAR VARIANT
  python v2/experiments/run_lgbm_v2_input.py --build-gate YEAR
  python v2/experiments/run_lgbm_v2_input.py --build-rank SAMPLE
  python v2/experiments/run_lgbm_v2_input.py --summarize
"""
from __future__ import annotations

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path


import argparse
import gc
import gzip
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import lgbm_v2_cache as cache
import lgbm_v2_common as common
from lgbm_v2_common import DATA, KEY, ROOT, Timer
from run_lgbm_v2_stage1 import LEAF_FRACTIONS, STRUCTURES, lgbm_params
from run_lgbm_v2_stage2 import PENALTY, YEARS, eligible_metrics, paired_t
import run_lgbm_v2_stage6a as s6

OUT = DATA / 'v2' / 'lgbm_input'
STAGE3 = DATA / 'v2' / 'lgbm_stage3'
GATE = ROOT / 'lgbm_gate_cache'      # F: project directory, not through the quant_data\processed junction
RANK = ROOT / 'lgbm_rank_cache'
CAP, PATIENCE, EVERY, TRUNCATION = 6000, 200, 10, 0.95
MIN_GAIN, MIN_T, PLATEAU_DIFF = 0.001, 2.0, 0.001
P4 = 'full_none_raw_deep_x1'
P4_STAGE3 = 'deepD_x1_l2-10000'
GATE_VERSION, RANK_VERSION = 1, 1
STEP1 = ['compact_none_raw_deep_x1', 'compact_none_raw_medium_x1', 'full_gate_raw_deep_x1', 'compact_gate_raw_deep_x1']
GATE_CHECK_YEAR = 2017


# ---------------------------------------------------------------- naming
def parse(variant):
    fs, sample, inp, structure, mult = variant.split('_')
    assert fs in ('full', 'compact') and sample in ('none', 'gate') and inp in ('raw', 'rank')
    assert structure in ('deep', 'medium') and mult.startswith('x')
    return dict(fs=fs, sample=sample, inp=inp, structure=structure, mult=float(mult[1:]))


def cell_of(variant):
    p = parse(variant)
    return f"{p['fs']}_{p['sample']}_{p['inp']}"


def variant_dir(year, variant):
    if variant == P4:
        return STAGE3 / f'{year}_{P4_STAGE3}'
    return OUT / f'{year}_{variant}'


def completed(output):
    path = output / 'summary.json'
    return path.exists() and common.read_json(path).get('status') == 'completed'


def truncated(best):
    return best >= TRUNCATION * CAP


def sig2(x):
    return s6.sig2(x)


def num(v):
    v = float(v)
    return int(v) if v.is_integer() else v


# ---------------------------------------------------------------- D labels
def d_labels_for(year, keys, y_fit, con=None):
    """D label (daily rank of the raw label among all stocks with a finite label that day) for given inner-fit keys.
    Same definition and source as stage 6a build_labels."""
    keys = keys[KEY].copy()
    keys['signal_date'] = pd.to_datetime(keys['signal_date']).dt.normalize()
    first, last = keys['signal_date'].min(), keys['signal_date'].max()
    own = con is None
    if own:
        con = common.open_connection(OUT, 8, '6GB')
        common.create_views(con, year)
    targets = con.execute(f"""
        SELECT ts_code, CAST(signal_date AS DATE) signal_date, CAST(exit_date AS DATE) exit_date,
               target_logret_5d_raw y
        FROM targets
        WHERE isfinite(target_logret_5d_raw)
          AND CAST(signal_date AS DATE) >= DATE '{first.date()}' AND CAST(signal_date AS DATE) <= DATE '{last.date()}'
    """).fetch_df()
    if own:
        con.close()
    targets['signal_date'] = pd.to_datetime(targets['signal_date']).dt.normalize()
    exit_dates = pd.to_datetime(targets['exit_date'])
    assert exit_dates.max() < pd.Timestamp(f'{year}-01-01'), f'{year}: a label for daily statistics exits in the outer year'
    if targets.duplicated(KEY).any():
        raise RuntimeError('Duplicate target keys')
    grouped = targets.groupby('signal_date')['y']
    targets['n_day'] = grouped.transform('size')
    targets['rank'] = (grouped.rank(method='average') - 0.5) / targets['n_day']
    joined = keys.merge(targets, on=KEY, how='left', validate='one_to_one')
    if joined['y'].isna().any():
        raise RuntimeError(f'{year}: inner-fit keys missing from targets')
    if not np.array_equal(joined['y'].to_numpy(), y_fit):
        raise RuntimeError(f'{year}: raw labels from targets differ from the inner-fit labels')
    rank = joined['rank'].to_numpy(np.float64)
    assert np.isfinite(rank).all() and rank.min() > 0 and rank.max() < 1
    info = dict(daily_stat_rows=len(targets), daily_stat_days=int(targets['signal_date'].nunique()),
                max_exit_date_used=str(exit_dates.max().date()), raw_std=float(y_fit.std()),
                rank_std=float(rank.std()), ratio_D=float(rank.std()) / float(y_fit.std()))
    return rank, info


# ---------------------------------------------------------------- gated inner-fit cache
def gate_dir(year):
    return GATE / f'{year}'


def gate_params(year):
    return dict(gate_version=GATE_VERSION, year=int(year), feature_set='full', rows=cache.ROWS,
                early_rows=cache.EARLY_ROWS, sample_seed=common.SAMPLE_SEED, training_gate='legacy feature_valid_count >= 30',
                sample_order='gate applied to the training pool, then hash(ts_code, CAST(signal_date AS DATE), 20261025), '
                             'ts_code, signal_date; first 2M rows',
                row_order='signal_date, ts_code', dtype_x='float32', label_rows=cache.LABEL_ROWS,
                stored='inner fit set only; early-stopping set and outer rows come from lgbm_cache/{year}_full')


def build_gate(year, threads=8):
    out = gate_dir(year)
    if (out / 'manifest.json').exists():
        load_gate(year)
        print(f'[gate {year}] cache exists and verifies', flush=True)
        return
    if out.exists():
        shutil.rmtree(out)  # incomplete build (no manifest): start over
    out.mkdir(parents=True)
    started = time.perf_counter()
    base = common.read_json(cache.cache_dir(year, 'full') / 'manifest.json')
    names = common.feature_names(year, 'full')
    assert base['feature_names'] == names
    con = common.open_connection(out, threads, '6GB')
    common.create_views(con, year)
    pool_rows = common.create_pool(con, year)
    assert pool_rows == base['rows']['training_pool'], 'training pool differs from the base cache'
    gated_pool = con.execute('SELECT count(*) FROM pool WHERE feature_ok').fetchone()[0]
    common.create_samples(con, year, cache.ROWS, cache.EARLY_ROWS, gate=True)
    checks = dict(pool_rows=pool_rows, gated_pool_rows=int(gated_pool), boundaries=common.boundary_checks(con, year))
    bad = con.execute('SELECT count(*) FROM full_sample WHERE NOT feature_ok').fetchone()[0]
    assert bad == 0, 'gated sample contains rows failing the gate'
    full_keys = con.execute('SELECT ts_code, signal_date FROM full_sample ORDER BY priority, ts_code, signal_date').fetch_df()
    assert len(full_keys) == cache.ROWS, f'gated full sample has {len(full_keys)} rows'
    # early-stopping set must be exactly the cached (ungated) one
    es = con.execute('SELECT ts_code, signal_date, y FROM early_stop ORDER BY signal_date, ts_code').fetch_df()
    es['signal_date'] = pd.to_datetime(es['signal_date'])
    assert cache.key_hash(es, es['y'].to_numpy()) == base['hashes']['es_keys_and_y'], 'early-stopping set changed'
    checks['early_stop_identical_to_base'] = True
    checks['labels'] = common.label_check(con, 'full_sample', cache.LABEL_ROWS)
    cache._save_keys(out / 'full_sample_keys.parquet', full_keys)
    t = time.perf_counter()
    x, y, keys = common.fetch_matrix(con, 'inner_fit', names)
    fetch_seconds = round(time.perf_counter() - t, 1)
    np.save(out / 'fit_x.npy', x)
    np.save(out / 'fit_y.npy', y)
    cache._save_keys(out / 'fit_keys.parquet', keys)
    del x
    d, dinfo = d_labels_for(year, keys, y, con)
    con.close()
    np.save(out / 'fit_d.npy', d)
    # overlap with the ungated sample, and D labels identical on shared keys
    base_dir = cache.cache_dir(year, 'full')
    norm = lambda f: f.assign(signal_date=pd.to_datetime(f['signal_date']).dt.normalize())[KEY]
    base_full = norm(pd.read_parquet(base_dir / 'full_sample_keys.parquet'))
    base_fit = norm(pd.read_parquet(base_dir / 'fit_keys.parquet')).assign(i_base=lambda f: np.arange(len(f)))
    ours_fit = norm(keys).assign(i_gate=np.arange(len(keys)))
    full_overlap = len(norm(full_keys).merge(base_full, on=KEY))
    shared = ours_fit.merge(base_fit, on=KEY)
    base_d = np.load(s6.LABELS / f'{year}_D.npy')
    assert np.array_equal(d[shared['i_gate'].to_numpy()], base_d[shared['i_base'].to_numpy()]), \
        'D labels differ from stage 6a on shared keys'
    checks['d_label_equal_to_6a_on_shared_keys'] = int(len(shared))
    gx, bx = np.load(out / 'fit_x.npy', mmap_mode='r'), np.load(base_dir / 'fit_x.npy', mmap_mode='r')
    gi, bi = shared['i_gate'].to_numpy(), shared['i_base'].to_numpy()
    for s in range(0, len(gi), 200_000):
        assert np.array_equal(np.asarray(gx[gi[s:s + 200_000]]), np.asarray(bx[bi[s:s + 200_000]]), equal_nan=True), \
            'gated features differ from the base cache on shared keys'
    checks['features_equal_to_base_on_shared_keys'] = int(len(shared))
    del gx, bx
    overlap = dict(full_sample_overlap=full_overlap, full_sample_overlap_share=full_overlap / cache.ROWS,
                   inner_fit_rows_gate=len(keys), inner_fit_rows_base=len(base_fit),
                   inner_fit_overlap=int(len(shared)), inner_fit_overlap_share_of_gate=len(shared) / len(keys),
                   inner_fit_overlap_share_of_base=len(shared) / len(base_fit))
    common.save_json(out / 'checks.json', checks)
    manifest = dict(params=gate_params(year), feature_names=names,
                    rows=dict(training_pool=pool_rows, gated_pool=int(gated_pool), full_sample=len(full_keys),
                              inner_fit=len(y), early_stop=cache.EARLY_ROWS),
                    shapes=dict(fit_x=[len(y), len(names)], fit_y=[len(y)], fit_d=[len(d)]),
                    hashes=dict(full_sample_keys=cache.key_hash(full_keys), fit_keys_and_y=cache.key_hash(keys, y),
                                fit_d=cache.key_hash(keys, d)),
                    file_bytes={p.name: p.stat().st_size for p in sorted(out.glob('*.npy'))},
                    overlap_with_ungated=overlap, d_label=dinfo, checks=checks,
                    source_sha256_at_build={p.name: common.sha256(p) for p in (Path(__file__), Path(common.__file__))},
                    timing_seconds=dict(fetch_inner_fit=fetch_seconds, total=round(time.perf_counter() - started, 1)),
                    test_used=False)
    common.save_json(out / 'manifest.json', manifest)
    load_gate(year)
    print(f'[gate {year}] pool {pool_rows:,} -> gated {gated_pool:,}; inner fit {len(y):,} '
          f'(ungated {len(base_fit):,}); full-sample overlap {overlap["full_sample_overlap_share"]:.3f}, '
          f'inner-fit overlap {overlap["inner_fit_overlap_share_of_gate"]:.3f}; ratio D {dinfo["ratio_D"]:.3f}; '
          f'{manifest["timing_seconds"]["total"]}s', flush=True)


def load_gate(year):
    out = gate_dir(year)
    manifest = common.read_json(out / 'manifest.json')
    if manifest['params'] != gate_params(year):
        raise RuntimeError(f'{out}: generation params differ')
    if manifest['feature_names'] != common.feature_names(year, 'full'):
        raise RuntimeError(f'{out}: feature list differs')
    for name, size in manifest['file_bytes'].items():
        if (out / name).stat().st_size != size:
            raise RuntimeError(f'{out}: {name} size changed')
    data = dict(manifest=manifest)
    for name in ('fit_x', 'fit_y', 'fit_d'):
        data[name] = np.load(out / f'{name}.npy')
        if list(data[name].shape) != manifest['shapes'][name]:
            raise RuntimeError(f'{out}: {name} shape mismatch')
    assert data['fit_x'].dtype == np.float32
    keys = pd.read_parquet(out / 'fit_keys.parquet')
    if cache.key_hash(keys, data['fit_y']) != manifest['hashes']['fit_keys_and_y'] or \
            cache.key_hash(keys, data['fit_d']) != manifest['hashes']['fit_d']:
        raise RuntimeError(f'{out}: key/label hash mismatch')
    keys['signal_date'] = keys['signal_date'].to_numpy().astype('datetime64[D]')
    data['fit_keys'] = keys
    return data


# ---------------------------------------------------------------- rank-factor cache
def rank_dir(year, sample):
    return RANK / f'{year}_{sample}'


def rank_params(year, sample):
    return dict(rank_version=RANK_VERSION, year=int(year), sample=sample,
                formula='(average rank - 0.5) / valid stocks that day for the factor - 0.5; missing kept missing',
                cross_section='all rows of local_features_2011_2023_stable on the signal date (feature_year <= 2021)',
                ranked_values='DOUBLE source values', dtype_x='float32')


def sample_keys(year, sample):
    """(fit keys, fit y) of the raw-data sample, the cached early-stopping keys/y and outer meta."""
    directory = cache.cache_dir(year, 'full')
    es_keys = pd.read_parquet(directory / 'es_keys.parquet')
    es_y = np.load(directory / 'es_y.npy')
    outer = pd.read_parquet(directory / 'outer_meta.parquet')
    if sample == 'none':
        fit_keys, fit_y = pd.read_parquet(directory / 'fit_keys.parquet'), np.load(directory / 'fit_y.npy')
    else:
        fit_keys, fit_y = pd.read_parquet(gate_dir(year) / 'fit_keys.parquet'), np.load(gate_dir(year) / 'fit_y.npy')
    return dict(fit=(fit_keys, fit_y), es=(es_keys, es_y), outer=(outer[KEY], outer['y'].to_numpy()))


def build_rank(sample, threads=8):
    from scipy.stats import rankdata
    todo = [y for y in YEARS if not (rank_dir(y, sample) / 'manifest.json').exists()]
    for y in YEARS:
        if y not in todo:
            load_rank(y, sample)
    if not todo:
        print(f'[rank {sample}] all folds exist and verify', flush=True)
        return
    names = {y: common.feature_names(y, 'full') for y in todo}
    union = []
    for y in todo:
        union += [n for n in names[y] if n not in union]
    col = {y: np.array([union.index(n) for n in names[y]]) for y in todo}
    # targets: per fold and part, the stored key order and the row position
    plan, arrays, need = {}, {}, 0
    for y in todo:
        out = rank_dir(y, sample)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        keys = sample_keys(y, sample)
        for part, (k, _) in keys.items():
            frame = k[KEY].copy()
            frame['signal_date'] = pd.to_datetime(frame['signal_date']).dt.normalize()
            frame['pos'] = np.arange(len(frame))
            plan[(y, part)] = frame
            need += len(frame) * len(names[y]) * 4
    free = shutil.disk_usage(RANK.parent).free
    print(f'[rank {sample}] need {need / 1024 ** 3:.1f} GB, free {free / 1024 ** 3:.1f} GB on {RANK.parent.drive}', flush=True)
    if need * 1.15 > free:
        raise SystemExit('Not enough disk space for the rank cache; stopping')
    for (y, part), frame in plan.items():
        arrays[(y, part)] = np.lib.format.open_memmap(rank_dir(y, sample) / f'{part}_x.npy', mode='w+',
                                                      dtype=np.float32, shape=(len(frame), len(names[y])))
    filled = {k: np.zeros(len(v), bool) for k, v in plan.items()}
    stats = dict(dates=0, rows=0, max_abs_daily_mean=0.0, min_value=1.0, max_value=-1.0)
    con = common.open_connection(RANK, threads, '8GB')
    started = time.perf_counter()
    for year in range(2011, max(todo) + 1):
        t = time.perf_counter()
        files = [str(p) for p in common.parquet_files(common.FEATURES, 'feature_year', year, year)]
        con.read_parquet(files, hive_partitioning=False).create_view('fy', replace=True)
        columns = ', '.join(f'CAST("{n}" AS DOUBLE) AS "{n}"' for n in union)
        total = con.execute('SELECT count(*) FROM fy').fetchone()[0]
        rel = con.execute(f'SELECT ts_code, CAST(signal_date AS DATE) signal_date, {columns} FROM fy '
                          'ORDER BY signal_date, ts_code')
        x = np.empty((total, len(union)), dtype=np.float64)
        codes, dates, pos = [], [], 0
        for batch in rel.fetch_record_batch(200_000):
            n = batch.num_rows
            codes.append(batch.column(0).to_pylist())
            dates.append(batch.column(1).to_numpy(zero_copy_only=False))
            for j in range(len(union)):
                x[pos:pos + n, j] = batch.column(2 + j).to_numpy(zero_copy_only=False)
            pos += n
        assert pos == total
        frame = pd.DataFrame({'ts_code': np.concatenate(codes),
                              'signal_date': pd.to_datetime(np.concatenate(dates).astype('datetime64[D]'))})
        assert (frame['signal_date'].dt.year == year).all(), f'feature_year={year} holds rows of another year'
        assert frame['signal_date'].max() <= pd.Timestamp('2021-12-31')
        if frame.duplicated(KEY).any():
            raise RuntimeError(f'duplicate feature keys in {year}')
        r = np.empty((total, len(union)), dtype=np.float32)
        starts, ends = common.day_slices(frame['signal_date'].to_numpy())
        for s, e in zip(starts, ends):
            block = x[s:e]
            valid = np.isfinite(block) | np.isinf(block)
            nan = np.isnan(block)
            ranks = rankdata(block, axis=0, nan_policy='omit')
            count = (~nan).sum(axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                value = (ranks - 0.5) / count - 0.5
            value[nan] = np.nan
            fin = value[~nan]
            assert np.isnan(value).sum() == nan.sum() and valid.sum() == (~nan).sum()
            if len(fin):
                assert fin.min() > -0.5 and fin.max() < 0.5, f'{year}: rank value outside (-0.5, 0.5)'
                stats['min_value'] = min(stats['min_value'], float(fin.min()))
                stats['max_value'] = max(stats['max_value'], float(fin.max()))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                means = np.nanmean(value, axis=0)
            means = means[np.isfinite(means)]
            if len(means):
                stats['max_abs_daily_mean'] = max(stats['max_abs_daily_mean'], float(np.abs(means).max()))
                assert np.abs(means).max() < 1e-9, f'{year}: daily mean of rank values is not 0'
            r[s:e] = value.astype(np.float32)
        stats['dates'] += len(starts)
        stats['rows'] += total
        f32 = r[~np.isnan(r)]
        assert f32.min() > -0.5 and f32.max() < 0.5, f'{year}: float32 rank value outside (-0.5, 0.5)'
        assert np.array_equal(np.isnan(r), np.isnan(x)), f'{year}: missing pattern changed'
        del x
        frame['row'] = np.arange(total)
        for (y, part), keys in plan.items():
            sub = keys.loc[keys['signal_date'].dt.year == year]
            if not len(sub):
                continue
            m = sub.merge(frame, on=KEY, how='left', validate='one_to_one')
            if m['row'].isna().any():
                raise RuntimeError(f'{y} {part}: {int(m["row"].isna().sum())} keys of {year} missing from features')
            rows = m['row'].to_numpy(np.int64)
            arrays[(y, part)][m['pos'].to_numpy()] = r[rows][:, col[y]]
            filled[(y, part)][m['pos'].to_numpy()] = True
        del r
        gc.collect()
        print(f'[rank {sample}] year {year}: {total:,} rows, {len(starts)} days, {time.perf_counter() - t:.0f}s', flush=True)
    con.close()
    for k, v in filled.items():
        assert v.all(), f'{k}: rows not filled'
    for a in arrays.values():
        a.flush()
    del arrays
    for y in todo:
        finish_rank(y, sample, names[y], stats, round(time.perf_counter() - started, 1))
    print(f'[rank {sample}] done: {stats}', flush=True)


def finish_rank(year, sample, names, stats, seconds):
    """Validate one fold's rank arrays against the raw cache, then write the manifest (last)."""
    out = rank_dir(year, sample)
    keys = sample_keys(year, sample)
    raw = cache.cache_dir(year, 'full')
    raw_x = dict(fit=(np.load(gate_dir(year) / 'fit_x.npy', mmap_mode='r') if sample == 'gate'
                      else np.load(raw / 'fit_x.npy', mmap_mode='r')),
                 es=np.load(raw / 'es_x.npy', mmap_mode='r'), outer=np.load(raw / 'outer_x.npy', mmap_mode='r'))
    checks, hashes, shapes = {}, {}, {}
    rng = np.random.default_rng(20261118)
    for part, (k, y) in keys.items():
        rx = np.load(out / f'{part}_x.npy', mmap_mode='r')
        assert rx.shape == raw_x[part].shape and rx.dtype == np.float32
        for s in range(0, len(rx), 200_000):
            a, b = np.asarray(rx[s:s + 200_000]), np.asarray(raw_x[part][s:s + 200_000])
            assert np.array_equal(np.isnan(a), np.isnan(b)), f'{year} {part}: missing pattern differs from raw cache'
            fin = a[~np.isnan(a)]
            assert fin.min() > -0.5 and fin.max() < 0.5
        # within a day, strictly larger raw (float32) value -> strictly larger rank (order preserved)
        dates = pd.to_datetime(k['signal_date']).to_numpy()
        starts, ends = common.day_slices(dates)
        picks = rng.choice(len(starts), min(25, len(starts)), replace=False)
        violations = 0
        for i in picks:
            s, e = starts[i], ends[i]
            a, b = np.asarray(rx[s:e]), np.asarray(raw_x[part][s:e])
            for j in range(a.shape[1]):
                ok = ~np.isnan(b[:, j])
                bj, aj = b[ok, j], a[ok, j]
                order = np.argsort(bj, kind='stable')
                bs, as_ = bj[order], aj[order]
                inc = bs[1:] > bs[:-1]
                violations += int((as_[1:][inc] <= as_[:-1][inc]).sum())
        assert violations == 0, f'{year} {part}: rank order violates raw order ({violations})'
        checks[part] = dict(rows=len(rx), days_checked=len(picks), order_violations=0, missing_pattern='identical to raw')
        hashes[f'{part}_keys_and_y'] = cache.key_hash(k, y)
        shapes[f'{part}_x'] = list(rx.shape)
    manifest = dict(params=rank_params(year, sample), feature_names=names, shapes=shapes, hashes=hashes,
                    file_bytes={p.name: p.stat().st_size for p in sorted(out.glob('*.npy'))},
                    checks=checks, build_stats=stats, build_seconds_all_folds=seconds,
                    source_sha256_at_build={p.name: common.sha256(p) for p in (Path(__file__), Path(common.__file__))},
                    test_used=False)
    common.save_json(out / 'manifest.json', manifest)
    load_rank(year, sample)
    print(f'[rank {sample} {year}] verified: {shapes}', flush=True)


def load_rank(year, sample):
    out = rank_dir(year, sample)
    manifest = common.read_json(out / 'manifest.json')
    if manifest['params'] != rank_params(year, sample):
        raise RuntimeError(f'{out}: generation params differ')
    if manifest['feature_names'] != common.feature_names(year, 'full'):
        raise RuntimeError(f'{out}: feature list differs')
    for name, size in manifest['file_bytes'].items():
        if (out / name).stat().st_size != size:
            raise RuntimeError(f'{out}: {name} size changed')
    keys = sample_keys(year, sample)
    for part, (k, y) in keys.items():
        if cache.key_hash(k, y) != manifest['hashes'][f'{part}_keys_and_y']:
            raise RuntimeError(f'{out}: {part} keys differ from the raw sample')
    data = {}
    for part in ('fit', 'es', 'outer'):
        data[f'{part}_x'] = np.load(out / f'{part}_x.npy', mmap_mode='r' if part == 'outer' else None)
        if list(data[f'{part}_x'].shape) != manifest['shapes'][f'{part}_x'] or data[f'{part}_x'].dtype != np.float32:
            raise RuntimeError(f'{out}: {part}_x shape/dtype mismatch')
    return data


# ---------------------------------------------------------------- assembled inputs for one cell
def columns(year, fs):
    full = common.feature_names(year, 'full')
    if fs == 'full':
        return full, None
    names = common.feature_names(year, 'compact')
    idx = np.array([full.index(n) for n in names])
    assert [full[i] for i in idx] == names and len(names) == 333
    return names, idx


def load_cell(year, fs, sample, inp):
    """Inner fit (x, raw y, D label, dates), early-stopping (x, raw y, dates), outer (x memmap, meta, column index)."""
    base = cache.load(year, 'full')
    names, idx = columns(year, fs)
    if base['names'] != common.feature_names(year, 'full'):
        raise RuntimeError('base cache column list differs')
    if sample == 'none':
        fit_x, fit_y, fit_keys = base['fit_x'], base['fit_y'], base['fit_keys']
        info = s6.build_labels(year)
        if info['fit_keys_hash'] != cache.key_hash(pd.read_parquet(base['dir'] / 'fit_keys.parquet'), fit_y):
            raise RuntimeError('stage 6a D label belongs to a different inner fit set')
        fit_d, ratio = np.load(s6.LABELS / f'{year}_D.npy'), info['ratio_D']
        gate_manifest = None
    else:
        g = load_gate(year)
        base['fit_x'] = None
        fit_x, fit_y, fit_keys, fit_d = g['fit_x'], g['fit_y'], g['fit_keys'], g['fit_d']
        ratio = g['manifest']['d_label']['ratio_D']
        gate_manifest = g['manifest']
    es_x, outer_x = base['es_x'], base['outer_x']
    if inp == 'rank':
        r = load_rank(year, sample)
        base['fit_x'] = None
        fit_x, es_x, outer_x = r['fit_x'], r['es_x'], r['outer_x']
    if idx is not None:
        fit_x, es_x = np.ascontiguousarray(fit_x[:, idx]), np.ascontiguousarray(es_x[:, idx])
    assert len(fit_d) == len(fit_y) == len(fit_x) and np.isfinite(fit_d).all()
    return dict(names=names, idx=idx, fit_x=fit_x, fit_y=fit_y, fit_d=fit_d, fit_dates=fit_keys['signal_date'].to_numpy(),
                es_x=es_x, es_y=base['es_y'], es_dates=base['es_keys']['signal_date'].to_numpy(),
                outer_x=outer_x, outer=base['outer'], ratio_D=ratio, base_manifest=base['manifest'],
                gate_manifest=gate_manifest)


def outer_chunks(x, idx, step=200_000):
    for s in range(0, len(x), step):
        a = np.asarray(x[s:s + step])
        yield a if idx is None else a[:, idx]


# ---------------------------------------------------------------- Elastic(D) per cell
def elastic_dir(year, cell):
    if cell == 'full_none_raw':
        return cache.cache_dir(year, 'full') / 'elastic_D'   # stage-3 Elastic(D), same keys / factors / alpha rule
    return OUT / 'elastic' / f'{year}_{cell}'


def elastic_cell(year, cell):
    from run_lgbm_v2_pilot import fit_elastic, predict_elastic
    out = elastic_dir(year, cell)
    if (out / 'metrics.json').exists():
        return common.read_json(out / 'metrics.json')
    out.mkdir(parents=True, exist_ok=True)
    fs, sample, inp = cell.split('_')
    started = time.perf_counter()
    data = load_cell(year, fs, sample, inp)
    model = fit_elastic(data['fit_x'], data['fit_d'], data['names'])  # alpha = 0.1 x critical on the D label
    pred = np.concatenate([predict_elastic(model, a) for a in outer_chunks(data['outer_x'], data['idx'])])
    frame = data['outer'].assign(elastic=pred)
    elig, elig_daily = eligible_metrics(frame, 'elastic')
    allm, _ = common.evaluate_outer(frame, 'elastic')
    np.save(out / 'outer_pred.npy', pred)
    elig_daily.to_parquet(out / 'eligible_daily_metrics.parquet', index=False)
    common.save_json(out / 'model.json', dict(coefficient=np.asarray(model['coefficient']).tolist(),
                                              intercept=model['intercept'], alpha=model['alpha'], nonzero=model['nonzero'],
                                              iterations=model['iterations'], active_names=model['stats']['active_names'],
                                              label='D', cell=cell))
    metrics = dict(year=year, cell=cell, eligible=elig, all_sample_mean_rank_ic=allm['mean_rank_ic'], alpha=model['alpha'],
                   nonzero=model['nonzero'], iterations=model['iterations'], inner_fit_rows=len(data['fit_y']),
                   factors=len(data['names']), seconds=round(time.perf_counter() - started, 1))
    common.save_json(out / 'metrics.json', metrics)
    print(f'[elastic D {year} {cell}] eligible RankIC {elig["mean_rank_ic"]:.5f} ({model["nonzero"]} nonzero, '
          f'{metrics["seconds"]:.0f}s)', flush=True)
    return metrics


# ---------------------------------------------------------------- one LightGBM fit
def drift(x, dates, names, top):
    """Plan section 10 item 3: correlation of each top factor's daily cross-sectional median with time."""
    pos = [names.index(n) for n in top]
    frame = pd.DataFrame(np.asarray(x[:, pos], dtype=np.float64), columns=top)
    frame['d'] = dates
    med = frame.groupby('d', sort=True)[top].median()
    t = np.arange(len(med), dtype=float)
    out = {}
    for n in top:
        v = med[n].to_numpy()
        ok = np.isfinite(v)
        out[n] = float(np.corrcoef(t[ok], v[ok])[0, 1]) if ok.sum() > 2 and np.std(v[ok]) > 0 else float('nan')
    return out


def importance(booster, trees):
    gains = booster.feature_importance(importance_type='gain', iteration=trees)
    return dict(zip(booster.feature_name(), map(float, gains)))


def fit_variant(year, variant, output, threads=8):
    import lightgbm as lgb
    p = parse(variant)
    output.mkdir(parents=True, exist_ok=True)
    timer, started = Timer(), time.perf_counter()
    with timer('load_cache'):
        data = load_cell(year, p['fs'], p['sample'], p['inp'])
    names, ratio = data['names'], data['ratio_D']
    if p['structure'] == 'deep':
        l1, l2 = sig2(50 * ratio) * p['mult'], 10000
    else:
        assert p['mult'] == 1.0
        l1, l2 = sig2(10 * ratio), 1000
    sources = [Path(__file__), Path(common.__file__), Path(cache.__file__), _source_path('v2/experiments/run_lgbm_v2_stage1.py'),
               _source_path('v2/experiments/run_lgbm_v2_stage2.py'), _source_path('v2/experiments/run_lgbm_v2_stage6a.py')]
    config = dict(stage='input', year=year, variant=variant, **p, lambda_l1=l1, lambda_l2=l2, ratio_D=ratio,
                  cap=CAP, patience=PATIENCE, every=EVERY, cache_hashes=data['base_manifest']['hashes'],
                  gate_hashes=(data['gate_manifest'] or {}).get('hashes'), factors=len(names), threads=threads,
                  test_used=False, sources={s.name: common.sha256(s) for s in sources})
    config_path = output / 'config.json'
    if config_path.exists() and common.read_json(config_path) != json.loads(json.dumps(config, default=str)):
        raise RuntimeError(f'{output}: existing output belongs to a different configuration or code version')
    common.save_json(config_path, config)

    x_fit, y_fit, fit_dates, label = data['fit_x'], data['fit_y'], data['fit_dates'], data['fit_d']
    x_es, y_es = data['es_x'], data['es_y']
    params = lgbm_params(STRUCTURES[p['structure']], LEAF_FRACTIONS['0.1pct'], len(y_fit), threads)
    params.update(lambda_l1=float(l1), lambda_l2=float(l2))
    monitor = common.RankICEarlyStopping(y_es, data['es_dates'], every=EVERY, patience=PATIENCE)
    with timer('dataset_construct'):
        train_set = lgb.Dataset(x_fit, label=label, feature_name=names, free_raw_data=False, params=params)
        valid_set = lgb.Dataset(x_es, label=y_es, reference=train_set, params=params)
        train_set.construct()
        valid_set.construct()
    trees_per_round = []

    def record_trees(env):
        trees_per_round.append(env.model.num_trees())
    record_trees.order = 20
    with timer('train'):
        monitor.started = time.perf_counter()
        booster = lgb.train(params, train_set, num_boost_round=CAP, valid_sets=[valid_set], valid_names=['early_stop'],
                            feval=monitor.feval, callbacks=monitor.callbacks() + [record_trees],
                            keep_training_booster=True)
    best, trained = monitor.best_iteration, monitor.iteration
    if best <= 0:
        raise RuntimeError('No valid early-stopping score was recorded')
    trees_built = booster.num_trees()
    monotone = all(b >= a for a, b in zip(trees_per_round, trees_per_round[1:]))
    if len(trees_per_round) != trained or trees_per_round[-1] != trees_built or not monotone:
        raise RuntimeError('Per-round tree counts are inconsistent with the trained booster')
    skipped, best_trees = trained - trees_built, trees_per_round[best - 1]
    print(f'[{year} {variant}] lambda1 {l1:g} lambda2 {l2:g} rounds={trained} best={best} trees@best={best_trees} '
          f'skipped={skipped} es RankIC={monitor.best_score:.5f} train={timer.records["train"]:.0f}s', flush=True)
    with timer('save_full_model'):
        text = booster.model_to_string(num_iteration=-1)
        with gzip.open(output / 'lightgbm_model_all_rounds.txt.gz', 'wt', encoding='utf-8', compresslevel=6) as h:
            h.write(text)
        model_bytes = len(text)
        del text
        np.save(output / 'trees_per_round.npy', np.asarray(trees_per_round, dtype=np.int32))
        common.save_json(output / 'best_iteration.json', dict(best_iteration=best, best_model_trees=best_trees,
                                                              rounds_trained=trained, trees_built=trees_built,
                                                              skipped_rounds=skipped))
    with timer('diagnostics'):
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(y_fit), min(300_000, len(y_fit)), replace=False))
        train_ic = common.DailyRankIC(y_fit[idx], fit_dates[idx])(booster.predict(x_fit[idx], num_iteration=best_trees))
        trees = common.tree_diagnostics(booster, best_trees)
        gains = importance(booster, best_trees)
        diagnostics = dict(trees=trees, leaf_coverage=common.leaf_date_coverage(booster, x_fit, fit_dates, best_trees, trees=10),
                           inner_fit_rank_ic=train_ic, early_stop_rank_ic=monitor.best_score,
                           fit_minus_early_stop_rank_ic=train_ic - monitor.best_score,
                           drift_top20=drift(x_fit, fit_dates, names, [n for n, _ in trees['top20_gain']]))
        common.save_json(output / 'importance_gain.json', gains)
    del train_set, valid_set
    gc.collect()
    with timer('outer_predict'):
        pred = np.concatenate([booster.predict(a, num_iteration=best_trees) for a in outer_chunks(data['outer_x'], data['idx'])])
    if not np.isfinite(pred).all():
        raise RuntimeError('Non-finite outer predictions')
    outer = data['outer'].assign(lgbm=pred)
    with timer('outer_metrics'):
        metrics, daily = common.evaluate_outer(outer, 'lgbm')
        metrics['timing_split'] = common.timing_selection_split(outer, 'lgbm')
        eligible, eligible_daily = eligible_metrics(outer, 'lgbm')
        e_dir = elastic_dir(year, cell_of(variant))
        e_daily = pd.read_parquet(e_dir / 'eligible_daily_metrics.parquet')
        e_daily['date'] = pd.to_datetime(e_daily['date']).dt.normalize()
        comparison = dict(eligible=paired_t(eligible_daily, e_daily),
                          elastic_eligible_mean_rank_ic=common.read_json(e_dir / 'metrics.json')['eligible']['mean_rank_ic'],
                          elastic='Elastic(D) of the same cell (same keys, factors and input)')
    outer[['ts_code', 'signal_date', 'lgbm']].to_parquet(output / 'outer_predictions.parquet', index=False)
    pd.DataFrame(monitor.curve, columns=['iteration', 'early_stop_rank_ic', 'seconds']).to_parquet(
        output / 'early_stop_curve.parquet', index=False)
    daily.to_parquet(output / 'daily_metrics.parquet', index=False)
    eligible_daily.to_parquet(output / 'eligible_daily_metrics.parquet', index=False)
    summary = dict(
        status='completed', stage='input', year=year, variant=variant, **p,
        rows=dict(inner_fit=len(y_fit), early_stop=len(y_es), outer=len(outer),
                  outer_scored=int(np.isfinite(outer['y']).sum()), outer_eligible_scored=eligible['rows']),
        lightgbm=dict(params=params, cap=CAP, rounds_trained=trained, trees_built=trees_built, skipped_rounds=skipped,
                      best_iteration=best, best_model_trees=best_trees, truncated=truncated(best),
                      early_stop_best_rank_ic=monitor.best_score, full_model_text_bytes=model_bytes),
        eligible_metrics=eligible, outer_metrics=metrics, lgbm_vs_elastic_d=comparison, diagnostics=diagnostics,
        timing_seconds=dict(**timer.records, total=round(time.perf_counter() - started, 1)),
        peak_memory_gb=common.peak_rss_gb(),
        machine=dict(platform=platform.platform(), threads=threads, lightgbm=lgb.__version__), test_used=False)
    common.save_json(output / 'summary.json', summary)
    print(f'[{year} {variant}] eligible RankIC={eligible["mean_rank_ic"]:.5f} (Elastic(D) '
          f'{comparison["elastic_eligible_mean_rank_ic"]:.5f}, t={comparison["eligible"]["newey_west_t_lag5"]:.2f}) '
          f'all={metrics["mean_rank_ic"]:.5f} leaves={trees["mean_leaves"]:.1f} total={summary["timing_seconds"]["total"]:.0f}s '
          f'peak={summary["peak_memory_gb"]:.1f}GB', flush=True)
    return summary


# ---------------------------------------------------------------- gate check: the new code path reproduces P4
def gate_check(threads):
    out = OUT / f'_check_{GATE_CHECK_YEAR}_{P4}'
    result_path = OUT / 'gate_check.json'
    if result_path.exists() and common.read_json(result_path).get('passed'):
        return
    if not completed(out):
        run_sub(['--one-check', str(GATE_CHECK_YEAR)], f'check {GATE_CHECK_YEAR} {P4}')
    ref = STAGE3 / f'{GATE_CHECK_YEAR}_{P4_STAGE3}'
    a, b = common.read_json(out / 'summary.json'), common.read_json(ref / 'summary.json')
    ca, cb = pd.read_parquet(out / 'early_stop_curve.parquet'), pd.read_parquet(ref / 'early_stop_curve.parquet')
    pa, pb = pd.read_parquet(out / 'outer_predictions.parquet'), pd.read_parquet(ref / 'outer_predictions.parquet')
    params_a = {k: v for k, v in a['lightgbm']['params'].items()}
    params_b = {k: v for k, v in b['lightgbm']['params'].items()}
    result = dict(year=GATE_CHECK_YEAR,
                  params_equal=params_a == params_b,
                  best_iteration=[a['lightgbm']['best_iteration'], b['lightgbm']['best_iteration']],
                  trees_built=[a['lightgbm']['trees_built'], b['lightgbm']['trees_built']],
                  curve_points=[len(ca), len(cb)],
                  curve_max_abs_diff=float(np.max(np.abs(ca['early_stop_rank_ic'].to_numpy() - cb['early_stop_rank_ic'].to_numpy())))
                  if len(ca) == len(cb) else None,
                  prediction_max_abs_diff=float(np.max(np.abs(pa['lgbm'].to_numpy() - pb['lgbm'].to_numpy()))),
                  keys_equal=bool((pa['ts_code'].to_numpy() == pb['ts_code'].to_numpy()).all()),
                  eligible=[a['eligible_metrics']['mean_rank_ic'], b['eligible_metrics']['mean_rank_ic']])
    result['passed'] = bool(result['params_equal'] and result['best_iteration'][0] == result['best_iteration'][1]
                            and result['trees_built'][0] == result['trees_built'][1] and len(ca) == len(cb)
                            and result['curve_max_abs_diff'] == 0.0 and result['prediction_max_abs_diff'] == 0.0
                            and result['keys_equal'])
    common.save_json(result_path, result)
    print(f'[gate check] {result}', flush=True)
    if not result['passed']:
        raise SystemExit('New fit code does not reproduce the P4 baseline; stopping')


# ---------------------------------------------------------------- driver
def run_sub(args, label):
    print(f'[start] {label} at {time.strftime("%m-%d %H:%M:%S")}', flush=True)
    code = subprocess.call([sys.executable, '-u', __file__] + args)
    if code != 0:
        raise SystemExit(f'{label} failed with exit code {code}; stopping')


def run_job(year, variant, threads):
    if completed(variant_dir(year, variant)):
        return
    if not (elastic_dir(year, cell_of(variant)) / 'metrics.json').exists():
        run_sub(['--elastic', str(year), cell_of(variant)], f'Elastic(D) {year} {cell_of(variant)}')
    run_sub(['--one', str(year), variant, '--threads', str(threads)], f'{year}_{variant}')


def ensure_elastic(cell):
    for y in YEARS:
        if not (elastic_dir(y, cell) / 'metrics.json').exists():
            run_sub(['--elastic', str(y), cell], f'Elastic(D) {y} {cell}')


# ---------------------------------------------------------------- scoring
def summary(year, variant):
    return common.read_json(variant_dir(year, variant) / 'summary.json')


def daily(variant, years=YEARS):
    parts = []
    for y in years:
        d = pd.read_parquet(variant_dir(y, variant) / 'eligible_daily_metrics.parquet')
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        parts.append(d.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def elastic_daily(cell, years=YEARS):
    parts = []
    for y in years:
        d = pd.read_parquet(elastic_dir(y, cell) / 'eligible_daily_metrics.parquet')
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        parts.append(d.assign(year=y))
    return pd.concat(parts, ignore_index=True)


def nw(a, b):
    joined = a.merge(b, on=['year', 'date'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(joined) != len(a) or len(joined) != len(b):
        raise RuntimeError('Incomplete daily pairing')
    diff = (joined['rank_ic_a'] - joined['rank_ic_b']).to_numpy()
    return float(diff.mean()), common.newey_west_tstat(diff, lag=5)


def score_array(values):
    v = np.asarray(values, float)
    return float(v.mean() - PENALTY * v.std(ddof=0))


def scores(variant):
    rows = [summary(y, variant) for y in YEARS]
    elig = np.array([r['eligible_metrics']['mean_rank_ic'] for r in rows])
    full = np.array([r['outer_metrics']['mean_rank_ic'] for r in rows])
    best = [r['lightgbm']['best_iteration'] for r in rows]
    t = [r['diagnostics']['trees'] for r in rows]
    m = [r['outer_metrics'] for r in rows]
    return dict(variant=variant, **{f'eligible_{y}': v for y, v in zip(YEARS, elig)},
                eligible_score=score_array(elig), eligible_mean=float(elig.mean()),
                **{f'all_{y}': v for y, v in zip(YEARS, full)}, all_score=score_array(full),
                negative_fold=bool((elig < 0).any()), truncated=any(truncated(b) for b in best),
                best_iterations=best, skipped_rounds=[r['lightgbm'].get('skipped_rounds', 0) for r in rows],
                lambda1_by_year=[r['lightgbm']['params']['lambda_l1'] for r in rows],
                min_data_in_leaf=[r['lightgbm']['params']['min_data_in_leaf'] for r in rows],
                inner_fit_rows=[r['rows']['inner_fit'] for r in rows],
                mean_leaves=float(np.mean([x['mean_leaves'] for x in t])),
                fit_minus_es=float(np.mean([r['diagnostics']['fit_minus_early_stop_rank_ic'] for r in rows])),
                top50_minus_pool=float(np.mean([x['top50_minus_pool_5d_simple_return'] for x in m])),
                top50_overlap=float(np.mean([x['consecutive_day_top50_overlap'] for x in m])),
                train_seconds=float(np.mean([r['timing_seconds']['train'] for r in rows])),
                total_seconds=float(np.sum([r['timing_seconds']['total'] for r in rows])))


def admissible(s):
    return not s['negative_fold'] and not s['truncated']


def adopt(candidate, base):
    """Plan section 8 item 8: a non-baseline option needs score gain >= 0.001 and NW t >= 2 (and admissible)."""
    c, b = scores(candidate), scores(base)
    diff, t = nw(daily(candidate), daily(base))
    gain = c['eligible_score'] - b['eligible_score']
    ok = bool(admissible(c) and gain >= MIN_GAIN and t >= MIN_T)
    return dict(candidate=candidate, base=base, score_candidate=c['eligible_score'], score_base=b['eligible_score'],
                score_gain=gain, mean_daily_diff=diff, t=t, candidate_admissible=admissible(c), adopted=ok)


def decide_i1():
    path = OUT / 'i1.json'
    fs_cmp = adopt('full_none_raw_deep_x1', 'compact_none_raw_deep_x1')
    compact_ok = admissible(scores('compact_none_raw_deep_x1'))
    fs = 'full' if (fs_cmp['adopted'] or not compact_ok) else 'compact'
    gate_cmp = adopt(f'{fs}_gate_raw_deep_x1', f'{fs}_none_raw_deep_x1')
    base_ok = admissible(scores(f'{fs}_none_raw_deep_x1'))
    sample = 'gate' if (gate_cmp['adopted'] or (not base_ok and admissible(scores(f'{fs}_gate_raw_deep_x1')))) else 'none'
    result = dict(factor_set=fs, sample=sample, i1=f'{fs}_{sample}_raw_deep_x1',
                  factor_set_comparison=fs_cmp, compact_admissible=compact_ok,
                  gate_comparison=gate_cmp, rule='448 kept only if it beats 333 by score >= 0.001 and NW t >= 2; '
                                                  'gate adopted only if it beats no gate by the same threshold')
    common.save_json(path, result)
    print(f'[I1] {result["i1"]}  (448 vs 333: gain {fs_cmp["score_gain"]:+.5f}, t {fs_cmp["t"]:.2f}; '
          f'gate vs none ({fs}): gain {gate_cmp["score_gain"]:+.5f}, t {gate_cmp["t"]:.2f})', flush=True)
    return result


def decide_i2(threads):
    i1 = common.read_json(OUT / 'i1.json')
    fs, sample = i1['factor_set'], i1['sample']
    raw_v, rank_v = f'{fs}_{sample}_raw_deep_x1', f'{fs}_{sample}_rank_deep_x1'
    cmp = adopt(rank_v, raw_v)
    result = dict(i1=raw_v, rank_vs_raw=cmp, adopted=cmp['adopted'])
    if cmp['adopted']:
        mults = [0.5, 1.0, 2.0]
        for m in (0.5, 2.0):
            for y in YEARS:
                run_job(y, f'{fs}_{sample}_rank_deep_x{m:g}', threads)
        table = {m: scores(f'{fs}_{sample}_rank_deep_x{m:g}') for m in mults}
        ok = {m: s for m, s in table.items() if admissible(s)}
        top = max(ok, key=lambda m: ok[m]['eligible_score'])
        top_d = daily(f'{fs}_{sample}_rank_deep_x{top:g}')
        plateau = []
        rows = []
        for m in mults:
            v = f'{fs}_{sample}_rank_deep_x{m:g}'
            diff, t = (0.0, 0.0) if m == top else nw(daily(v), top_d)
            member = m in ok and (m == top or abs(t) < MIN_T or abs(diff) < PLATEAU_DIFF)
            rows.append(dict(mult=m, variant=v, score=table[m]['eligible_score'], diff_vs_top=diff, t_vs_top=t,
                             admissible=m in ok, plateau=member, lambda1_by_year=table[m]['lambda1_by_year']))
            if member:
                plateau.append(m)
        chosen = max(plateau)  # strongest regularisation within the plateau (plan section 8 items 4, 9)
        result.update(lambda1_review=rows, top=top, plateau=plateau, chosen_mult=chosen,
                      i2=f'{fs}_{sample}_rank_deep_x{chosen:g}')
    else:
        result['i2'] = raw_v
    common.save_json(OUT / 'i2.json', result)
    print(f'[I2] {result["i2"]} (rank vs raw: gain {cmp["score_gain"]:+.5f}, t {cmp["t"]:.2f})', flush=True)
    return result


def run_all(threads):
    OUT.mkdir(parents=True, exist_ok=True)
    gate_check(threads)
    for y in YEARS:
        if not (gate_dir(y) / 'manifest.json').exists():
            run_sub(['--build-gate', str(y)], f'gated inner-fit cache {y}')
    for variant in STEP1:
        ensure_elastic(cell_of(variant))
        for y in YEARS:
            run_job(y, variant, threads)
    i1 = decide_i1()
    if not all((rank_dir(y, i1['sample']) / 'manifest.json').exists() for y in YEARS):
        run_sub(['--build-rank', i1['sample']], f'rank cache ({i1["sample"]})')
    ensure_elastic(f"{i1['factor_set']}_{i1['sample']}_rank")
    rank_v = f"{i1['factor_set']}_{i1['sample']}_rank_deep_x1"
    for y in YEARS:
        run_job(y, rank_v, threads)
    decide_i2(threads)
    summarize()
    print('[done] input stage complete', flush=True)


# ---------------------------------------------------------------- summary
def f(x, d=4):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def p4_importance_and_drift(year):
    """For the reused P4 fits: gain importance at the best model and drift of its top 20 (from the base cache)."""
    import lightgbm as lgb
    path = OUT / 'p4_importance' / f'{year}.json'
    if path.exists():
        return common.read_json(path)
    d = variant_dir(year, P4)
    best = common.read_json(d / 'best_iteration.json')
    with gzip.open(d / 'lightgbm_model_all_rounds.txt.gz', 'rt', encoding='utf-8') as h:
        booster = lgb.Booster(model_str=h.read())
    trees = best.get('best_model_trees', best['best_iteration'])
    gains = importance(booster, trees)
    top = [n for n, _ in sorted(gains.items(), key=lambda kv: -kv[1])[:20]]
    s = summary(year, P4)
    assert top == [n for n, _ in s['diagnostics']['trees']['top20_gain']], 'P4 top-20 differs from its summary'
    base = cache.load(year, 'full')
    result = dict(gains=gains, top20=top, drift_top20=drift(base['fit_x'], base['fit_keys']['signal_date'].to_numpy(),
                                                             base['names'], top))
    common.save_json(path, result)
    return result


def top20_and_drift(variant, year):
    if variant == P4:
        r = p4_importance_and_drift(year)
        return r['gains'], r['top20'], r['drift_top20']
    s = summary(year, variant)
    gains = common.read_json(variant_dir(year, variant) / 'importance_gain.json')
    top = [n for n, _ in s['diagnostics']['trees']['top20_gain']]
    return gains, top, s['diagnostics']['drift_top20']


def importance_table(variant):
    """Factors ranked by mean gain share across folds; folds in top 20; mean drift where in top 20."""
    shares, in_top, drifts, spearman = {}, {}, {}, []
    ranks = []
    for y in YEARS:
        gains, top, dr = top20_and_drift(variant, y)
        total = sum(gains.values())
        for n, g in gains.items():
            shares.setdefault(n, []).append(g / total)
        for n in top:
            in_top[n] = in_top.get(n, 0) + 1
            drifts.setdefault(n, []).append(dr[n])
        ranks.append(pd.Series(gains).rank(ascending=False))
    rk = pd.concat(ranks, axis=1).dropna()
    corr = rk.corr(method='spearman').to_numpy()
    fold_spearman = float(corr[np.triu_indices(len(YEARS), 1)].mean())
    table = pd.DataFrame([dict(factor=n, family=common.FAMILY.get(n), mean_gain_share=float(np.mean(v)),
                               folds_in_top20=in_top.get(n, 0),
                               mean_drift_when_top20=float(np.nanmean(drifts[n])) if n in drifts else np.nan)
                          for n, v in shares.items()]).sort_values('mean_gain_share', ascending=False)
    top_drift = [abs(v) for n in in_top for v in drifts[n] if np.isfinite(v)]
    return table.head(20).reset_index(drop=True), fold_spearman, float(np.mean(top_drift)), \
        float(np.mean(np.array(top_drift) > 0.5))


def summarize():
    i1 = common.read_json(OUT / 'i1.json')
    i2 = common.read_json(OUT / 'i2.json')
    fs1, s1 = i1['factor_set'], i1['sample']
    variants = [P4, 'compact_none_raw_deep_x1', 'full_gate_raw_deep_x1', 'compact_gate_raw_deep_x1',
                'compact_none_raw_medium_x1', f'{fs1}_{s1}_rank_deep_x1']
    if i2.get('lambda1_review'):
        variants += [r['variant'] for r in i2['lambda1_review'] if r['mult'] != 1.0]
    p4d = daily(P4)
    rows, fold_rows = [], []
    for v in variants:
        s = scores(v)
        cell = cell_of(v)
        diff, t = (0.0, 0.0) if v == P4 else nw(daily(v), p4d)
        lg = daily(v)
        ed = elastic_daily(cell)
        per_fold_t = [nw(lg.loc[lg['year'] == y], ed.loc[ed['year'] == y])[1] for y in YEARS]
        per_fold_diff = [nw(lg.loc[lg['year'] == y], ed.loc[ed['year'] == y])[0] for y in YEARS]
        pooled = nw(lg, ed)
        e_vals = [common.read_json(elastic_dir(y, cell) / 'metrics.json')['eligible']['mean_rank_ic'] for y in YEARS]
        e_all = [common.read_json(elastic_dir(y, cell) / 'metrics.json')['all_sample_mean_rank_ic'] for y in YEARS]
        rows.append(dict(s, cell=cell, diff_vs_p4=diff, t_vs_p4=t, t_vs_elastic_by_fold=per_fold_t,
                         diff_vs_elastic_by_fold=per_fold_diff, diff_vs_elastic_pooled=pooled[0],
                         t_vs_elastic_pooled=pooled[1], elastic_score=score_array(e_vals),
                         elastic_all_score=score_array(e_all), **{f'elastic_{y}': e for y, e in zip(YEARS, e_vals)}))
        for y in YEARS:
            r = summary(y, v)
            fold_rows.append(dict(variant=v, year=y, eligible_rank_ic=r['eligible_metrics']['mean_rank_ic'],
                                  all_rank_ic=r['outer_metrics']['mean_rank_ic'],
                                  best_iteration=r['lightgbm']['best_iteration'],
                                  skipped_rounds=r['lightgbm'].get('skipped_rounds', 0),
                                  truncated=r['lightgbm']['truncated'], lambda_l1=r['lightgbm']['params']['lambda_l1'],
                                  lambda_l2=r['lightgbm']['params']['lambda_l2'],
                                  min_data_in_leaf=r['lightgbm']['params']['min_data_in_leaf'],
                                  inner_fit_rows=r['rows']['inner_fit'],
                                  mean_leaves=r['diagnostics']['trees']['mean_leaves'],
                                  fit_minus_es=r['diagnostics']['fit_minus_early_stop_rank_ic'],
                                  top50_minus_pool=r['outer_metrics']['top50_minus_pool_5d_simple_return'],
                                  top50_overlap=r['outer_metrics']['consecutive_day_top50_overlap'],
                                  train_seconds=r['timing_seconds']['train'], total_seconds=r['timing_seconds']['total']))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'variants_table.csv', index=False, encoding='utf-8')
    pd.DataFrame(fold_rows).to_csv(OUT / 'variants_by_fold.csv', index=False, encoding='utf-8')
    medium = nw(daily('compact_none_raw_medium_x1'), daily('compact_none_raw_deep_x1'))
    rank_cell, raw_cell = f'{fs1}_{s1}_rank', f'{fs1}_{s1}_raw'
    e_rank_vs_raw = nw(elastic_daily(rank_cell), elastic_daily(raw_cell))
    gates = []
    for y in YEARS:
        m = common.read_json(gate_dir(y) / 'manifest.json')
        gates.append(dict(year=y, **m['rows'], **m['overlap_with_ungated'], ratio_D_gate=m['d_label']['ratio_D'],
                          ratio_D_none=s6.build_labels(y)['ratio_D']))
    gates = pd.DataFrame(gates)
    gates.to_csv(OUT / 'gate_sample_table.csv', index=False, encoding='utf-8')
    imp = {}
    for v in (i1['i1'], f'{fs1}_{s1}_rank_deep_x1'):
        v_ = P4 if v == P4 else v
        imp[v] = importance_table(v_)
        imp[v][0].to_csv(OUT / f'importance_top20_{v}.csv', index=False, encoding='utf-8')
    common.save_json(OUT / 'input_summary.json', dict(
        i1=i1, i2=i2, medium_vs_deep_333=dict(mean_diff=medium[0], t=medium[1]),
        elastic_rank_vs_raw=dict(cells=[rank_cell, raw_cell], mean_diff=e_rank_vs_raw[0], t=e_rank_vs_raw[1]),
        importance={v: dict(fold_spearman=x[1], mean_abs_drift_top20=x[2], share_abs_drift_above_0_5=x[3])
                    for v, x in imp.items()}))

    L = ['## 2×2 与追加组（五折资格池日度 Rank IC；配置 P4，D 标签）', '',
         '| 组 | 2017 | 2018 | 2019 | 2020 | 2021 | 资格池分数 | 全样本分数 | 对 P4 差（日度） | 对 P4 t | 负折 | 截断 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for _, r in table.iterrows():
        L.append(f"| {r['variant']} | " + ' | '.join(f(r[f'eligible_{y}']) for y in YEARS) +
                 f" | **{f(r['eligible_score'])}** | {f(r['all_score'])} | {f(r['diff_vs_p4'], 5)} | {f(r['t_vs_p4'], 2)} | "
                 f"{'是' if r['negative_fold'] else '否'} | {'是' if r['truncated'] else '否'} |")
    L += ['', '## 对同格 Elastic(D)', '',
          '| 组 | Elastic(D) 逐折 | Elastic(D) 分数 | Elastic(D) 全样本分数 | LightGBM − Elastic 拼接差 | 拼接 t | 逐折 t |',
          '|---|---|---:|---:|---:|---:|---|']
    for _, r in table.iterrows():
        L.append(f"| {r['variant']} | {' / '.join(f(r[f'elastic_{y}']) for y in YEARS)} | {f(r['elastic_score'])} | "
                 f"{f(r['elastic_all_score'])} | {f(r['diff_vs_elastic_pooled'], 5)} | {f(r['t_vs_elastic_pooled'], 2)} | "
                 f"{' / '.join(f(v, 2) for v in r['t_vs_elastic_by_fold'])} |")
    L += ['', '## 轮数与诊断（五折）', '',
          '| 组 | 逐折 λ1 | min_data_in_leaf | best_iteration | 跳过轮次 | 平均叶 | 内部−早停 | Top50−池 | Top50 重合 | 平均训练秒 |',
          '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for _, r in table.iterrows():
        L.append(f"| {r['variant']} | {'/'.join(f'{v:g}' for v in r['lambda1_by_year'])} | "
                 f"{'/'.join(map(str, r['min_data_in_leaf']))} | {'/'.join(map(str, r['best_iterations']))} | "
                 f"{'/'.join(map(str, r['skipped_rounds']))} | {r['mean_leaves']:.1f} | {f(r['fit_minus_es'])} | "
                 f"{f(r['top50_minus_pool'], 5)} | {f(r['top50_overlap'], 3)} | {r['train_seconds']:.0f} |")
    L += ['', '## 筛选样本', '',
          '| 折 | 训练池 | 筛选后训练池 | 内部拟合行（筛选） | 内部拟合行（无筛选） | 200 万行键重合率 | 内部拟合键重合（占筛选） | 比例 D 筛选 | 比例 D 无筛选 |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in gates.iterrows():
        L.append(f"| {r['year']} | {r['training_pool']:,} | {r['gated_pool']:,} | {r['inner_fit_rows_gate']:,} | "
                 f"{r['inner_fit_rows_base']:,} | {f(r['full_sample_overlap_share'], 4)} | "
                 f"{f(r['inner_fit_overlap_share_of_gate'], 4)} | {f(r['ratio_D_gate'], 3)} | {f(r['ratio_D_none'], 3)} |")
    fc, gc_ = i1['factor_set_comparison'], i1['gate_comparison']
    L += ['', '## 选择', '',
          f"- 因子集（无筛选）：448 对 333 分数差 {fc['score_gain']:+.5f}，日度差 {fc['mean_daily_diff']:+.5f}，NW t {fc['t']:.2f} → "
          f"{'保留 448' if fs1 == 'full' else '选 333'}。",
          f"- 样本筛选（{fs1}）：筛选对无筛选分数差 {gc_['score_gain']:+.5f}，日度差 {gc_['mean_daily_diff']:+.5f}，NW t {gc_['t']:.2f} → "
          f"{'采用筛选' if s1 == 'gate' else '不筛选'}。",
          f"- **I1 = {i1['i1']}**。",
          f"- 333 中结构对 333 深结构：日度差 {medium[0]:+.5f}，NW t {medium[1]:.2f}。",
          f"- 排名因子对原始因子（{fs1}_{s1}）：分数差 {i2['rank_vs_raw']['score_gain']:+.5f}，日度差 "
          f"{i2['rank_vs_raw']['mean_daily_diff']:+.5f}，NW t {i2['rank_vs_raw']['t']:.2f} → "
          f"{'采用' if i2['adopted'] else '不采用'}。",
          f"- Elastic(D) 排名版对原始版（{fs1}_{s1}）：日度差 {e_rank_vs_raw[0]:+.5f}，NW t {e_rank_vs_raw[1]:.2f}。"]
    if i2.get('lambda1_review'):
        L.append('- λ1 复核（排名因子）：' + '；'.join(
            f"×{r['mult']:g} 分数 {f(r['score'])}、对最高 t {f(r['t_vs_top'], 2)}{'（平台）' if r['plateau'] else ''}"
            for r in i2['lambda1_review']) + f" → ×{i2['chosen_mult']:g}")
    L.append(f"- **I2 = {i2['i2']}**（只报告）。")
    for v, (tab, sp, dr, share) in imp.items():
        L += ['', f'## 重要性前 20：{v}（各折增益占比平均；各折排序 Spearman 均值 {sp:.3f}；'
                  f'前 20 因子漂移 |相关| 均值 {dr:.3f}，> 0.5 的占比 {share:.2f}）', '',
              '| # | 因子 | 族 | 平均增益占比 | 进前 20 的折数 | 漂移（进前 20 时的均值） |', '|---:|---|---|---:|---:|---:|']
        for i, r in tab.iterrows():
            L.append(f"| {i + 1} | {r['factor']} | {r['family']} | {r['mean_gain_share']:.4f} | {r['folds_in_top20']} | "
                     f"{f(r['mean_drift_when_top20'], 3)} |")
    text = '\n'.join(L) + '\n'
    (OUT / 'input_tables.md').write_text(text, encoding='utf-8')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--one', nargs=2, metavar=('YEAR', 'VARIANT'))
    parser.add_argument('--one-check', type=int)
    parser.add_argument('--elastic', nargs=2, metavar=('YEAR', 'CELL'))
    parser.add_argument('--build-gate', type=int)
    parser.add_argument('--build-rank', choices=['none', 'gate'])
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.one:
        year, variant = int(args.one[0]), args.one[1]
        if year not in YEARS or variant == P4:
            raise ValueError('Not an input-stage fit')
        fit_variant(year, variant, variant_dir(year, variant), args.threads)
    elif args.one_check:
        fit_variant(args.one_check, P4, OUT / f'_check_{args.one_check}_{P4}', args.threads)
    elif args.elastic:
        elastic_cell(int(args.elastic[0]), args.elastic[1])
    elif args.build_gate:
        build_gate(args.build_gate, args.threads)
    elif args.build_rank:
        build_rank(args.build_rank, args.threads)
    elif args.summarize:
        summarize()
    elif args.all:
        run_all(args.threads)
    else:
        parser.error('choose an action')


if __name__ == '__main__':
    main()
