"""Build V2 500-column features, preserving V1. No labels or model fitting.

Example: python v2/build_features_500.py --start-date 2023-01-01 --end-date 2023-12-31 --max-stocks 8 --output-dir quant_data/processed/v2/features_500_smoke
Default covers 2011-2024 development data; never opens 2025+ partitions.
"""
from __future__ import annotations

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import time
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from features_500 import ROOT, BASE_FEATURES, FEATURE_COLUMNS, NEW_FEATURES, RETURNS, conditional_features, export_catalog


def atomic_json(path, payload):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_files(root, partition, start_year, end_year):
    files = []
    for directory in sorted(root.glob(partition+'=*')):
        year = int(directory.name.split('=')[1])
        if start_year <= year <= end_year:
            files.extend(sorted(directory.glob('*.parquet')))
    if not files:
        raise FileNotFoundError(root)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start-date', default='2011-01-01')
    p.add_argument('--end-date', default='2024-12-31')
    p.add_argument('--max-stocks', type=int)
    p.add_argument('--output-dir', default=str(ROOT/'quant_data/processed/v2/features_500'))
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    start, end = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    if start > end or end.year > 2024 or args.threads < 1 or (args.max_stocks is not None and args.max_stocks < 1):
        raise ValueError('Require valid dates through 2024 and positive threads/stock limit')
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT/'quant_data/processed/v2'):
        raise ValueError('V2 output directory required')
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 8*1024**3:
        raise RuntimeError('Less than 8 GiB free')
    sources = {
        'base': read_files(ROOT/'quant_data/processed/features_50', 'feature_year', start.year-1, end.year),
        'daily': read_files(ROOT/'quant_data/processed/parquet/daily', 'year', start.year-1, end.year),
        'flows': read_files(ROOT/'quant_data/processed/parquet/moneyflow', 'year', start.year-1, end.year),
    }
    config = dict(arguments=vars(args), features=FEATURE_COLUMNS,
                  code_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), _source_path('v2/features_500.py'),_source_path('build_features.py')]},
                  inputs={k: [dict(path=str(f), bytes=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns) for f in v] for k,v in sources.items()})
    config_path = output/'build_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Configuration changed: use a new output directory')
    atomic_json(config_path, config)
    export_catalog(output)
    progress_path = output/'build_progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else dict(completed=[], rows=0, finite_counts=[0]*500)
    done = set(progress['completed'])
    con = duckdb.connect()
    con.execute(f'SET threads={args.threads}')
    con.execute("SET memory_limit='6GB'")
    con.execute('SET preserve_insertion_order=false')
    con.execute('SET temp_directory=?', [str(output/'_duckdb_tmp')])
    began = time.time()
    try:
        for name, files in sources.items():
            con.read_parquet([str(f) for f in files], hive_partitioning=False).create_view(name)
        con.execute('CREATE TEMP TABLE calendar AS SELECT trade_date, row_number() OVER (ORDER BY trade_date) AS market_index FROM (SELECT DISTINCT CAST(trade_date AS DATE) trade_date FROM daily WHERE trade_date <= ?)', [end])
        # A prior calendar year normally contains over 120 sessions. Check rather
        # than silently shorten the maximum 120-session lookback for late starts.
        prior_sessions = con.execute('SELECT count(*) FROM calendar WHERE trade_date < ?', [start]).fetchone()[0]
        if prior_sessions < 120:
            print(f'Only {prior_sessions} prior market sessions available; insufficient windows stay null', flush=True)
        codes = [r[0] for r in con.execute('SELECT DISTINCT ts_code FROM base WHERE signal_date BETWEEN ? AND ? ORDER BY ts_code', [start,end]).fetchall()]
        if args.max_stocks:
            codes = codes[:args.max_stocks]
        if not codes:
            raise ValueError('No stocks in date range')
        # Materialize one compact selected-input table once. Per-stock scans then
        # avoid rereading 50-column, all-market Parquet files thousands of times.
        con.register('selected_codes', pa.table({'ts_code': codes}))
        print(f'Materializing inputs for {len(codes)} stocks; no targets read', flush=True)
        con.execute('''CREATE TEMP TABLE inputs AS
            SELECT b.*, c.market_index, d.vol AS selector_volume, d.amount AS selector_amount,
              m.buy_lg_amount+m.buy_elg_amount AS selector_large_buy,
              m.buy_lg_amount+m.buy_elg_amount-m.sell_lg_amount-m.sell_elg_amount AS selector_large_net
            FROM base b JOIN selected_codes s USING(ts_code)
            JOIN calendar c ON b.signal_date=c.trade_date
            LEFT JOIN daily d ON b.ts_code=d.ts_code AND b.signal_date=d.trade_date
            LEFT JOIN flows m ON b.ts_code=m.ts_code AND b.signal_date=m.trade_date
            WHERE b.signal_date <= ? ORDER BY b.ts_code, b.signal_date''', [end])
        con.execute('CREATE INDEX stock_idx ON inputs(ts_code)')
        for number, code in enumerate(codes, 1):
            if code in done:
                continue
            if shutil.disk_usage(output).free < 8*1024**3:
                raise RuntimeError('Low disk space; checkpoints retained')
            table = con.execute('SELECT * FROM inputs WHERE ts_code=? ORDER BY signal_date', [code]).to_arrow_table()
            frame = table.to_pandas()
            for column in ('signal_date','financial_ann_date','financial_end_date'):
                frame[column] = pd.to_datetime(frame[column])
            if frame['signal_date'].duplicated().any():
                raise ValueError(f'Duplicate source keys for {code}')
            ann, signal = frame['financial_ann_date'], frame['signal_date']
            if ((ann.notna()) & (ann >= signal)).any():
                raise ValueError('Inherited financial availability violation')
            selectors = {key: frame[col].to_numpy(dtype=float) for key,col in {
                'volume':'selector_volume', 'amount':'selector_amount','turnover':'turnover_rate',
                'large_buy':'selector_large_buy','large_net':'selector_large_net'}.items()}
            returns = {key: frame[value[1]].to_numpy(dtype=float) for key,value in RETURNS.items()}
            new = conditional_features(frame['market_index'].to_numpy(), selectors, returns)
            keep = (frame['signal_date'] >= np.datetime64(start)) & (frame['signal_date'] <= np.datetime64(end))
            frame, new = frame.loc[keep].reset_index(drop=True), new[keep]
            base = frame[list(BASE_FEATURES)].to_numpy(dtype=float)
            if np.isinf(base).any() or np.isinf(new).any():
                raise ValueError('Infinite features')
            all_values = np.column_stack([base,new])
            fields = {name: pa.array(frame[name], type=table.schema.field(name).type) for name in
                      ['ts_code','signal_date','financial_ann_date','financial_end_date',*BASE_FEATURES]}
            fields.update({name: pa.array(new[:,i], mask=~np.isfinite(new[:,i])) for i,name in enumerate(NEW_FEATURES)})
            fields['base_feature_valid_count'] = pa.array(np.isfinite(base).sum(axis=1).astype('int16'))
            fields['feature_valid_count'] = pa.array(np.isfinite(all_values).sum(axis=1).astype('int16'))
            result = pa.table(fields)
            years = frame['signal_date'].dt.year.to_numpy()
            for year in np.unique(years):
                directory = output/f'feature_year={year}'
                directory.mkdir(exist_ok=True)
                target = directory/f'{code}.parquet'
                temp = target.with_suffix('.tmp')
                pq.write_table(result.filter(pa.array(years==year)), temp, compression='zstd')
                temp.replace(target)
            counts = np.isfinite(all_values).sum(axis=0)
            progress['finite_counts'] = (np.array(progress['finite_counts'])+counts).tolist()
            progress['rows'] += len(frame)
            progress['completed'].append(code)
            atomic_json(progress_path, progress)
            print(f'{number}/{len(codes)} {code}: {len(frame):,} rows, 500 features', flush=True)
        counts = np.array(progress['finite_counts'])
        report = dict(status='completed', feature_count=500, original=50, added=450,
                      rows=progress['rows'], stocks=len(progress['completed']),
                      start_date=str(start), end_date=str(end), labels_read=False, models_trained=False,
                      scope='limited-stock verification' if args.max_stocks else 'all available stocks',
                      columns_with_any_finite_value=int((counts>0).sum()),
                      coverage={name: float(count/progress['rows']) for name,count in zip(FEATURE_COLUMNS,counts)},
                      elapsed_minutes=(time.time()-began)/60)
        atomic_json(output/'feature_report.json', report)
        print(f"Completed: {report['rows']:,} rows, {report['stocks']} stocks, 500 feature columns", flush=True)
    finally:
        con.close()


if __name__ == '__main__':
    main()
