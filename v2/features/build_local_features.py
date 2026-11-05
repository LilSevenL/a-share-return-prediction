"""Disk-staged, bounded-memory local survivor-universe feature build.

No API, targets, or 2024+ data. Checkpoints at staging year, stock bucket,
and final month. The repaired factor mathematics are unchanged.
"""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from balanced_features import BASE_FEATURES,CATALOG,COLUMNS,RECORDS,technical_stock,finish_cross_section
from build_balanced_features import files,save_json,save_frame,ROOT
RANK_PARENTS={name for r in RECORDS if r['family']=='I' for name in r['source'].split('; ')}


def safe_remove_tree(path,root):
    path=path.resolve();root=root.resolve()
    if path==root or not path.is_relative_to(root) or not path.name.startswith('stage_year_'):
        raise ValueError('Unsafe scratch cleanup path')
    if path.exists():shutil.rmtree(path)


def calculate_stock(payload):
    code,data,calendar,market,start,end=payload
    observed=pd.DatetimeIndex(data.signal_date)
    if observed.duplicated().any():raise ValueError(f'Duplicate stock dates: {code}')
    cal=calendar[(calendar>=observed.min())&(calendar<=observed.max())]
    d=data.set_index('signal_date').reindex(cal)
    d['rm']=market.rm.reindex(cal);d['market_median_turnover']=market.market_median_turnover.reindex(cal)
    result=technical_stock(d)
    result=result.loc[result.index.isin(observed)&(result.index>=start)&(result.index<end)].copy()
    # Storage compression only. Computations above remain float64.
    # Rank inputs retain full precision until the cross-section is assembled.
    result=result.astype({name:np.float64 if name in RANK_PARENTS else np.float32 for name in COLUMNS}).copy()
    if np.isinf(result.to_numpy()).any():raise ValueError(f'Float32 overflow: {code}')
    result.insert(0,'ts_code',code);return result.reset_index()


def check_space(output,min_gb):
    if shutil.disk_usage(output).free<min_gb*2**30:raise RuntimeError('Disk reserve reached; checkpoints retained')


def write_status(output,state,**updates):
    state.update(updates);state['updated_at']=pd.Timestamp.now(tz='UTC').isoformat()
    save_json(output/'progress.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start-year',type=int,default=2011);p.add_argument('--end-year',type=int,default=2023)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--bucket-size',type=int,default=64)
    p.add_argument('--max-stocks',type=int);p.add_argument('--min-free-gb',type=float,default=8)
    p.add_argument('--output-dir',default=str(ROOT/'quant_data/processed/v2/local_features_2011_2023'))
    args=p.parse_args()
    if not 2011<=args.start_year<=args.end_year<=2023 or args.workers<1 or args.bucket_size<1:raise ValueError('Invalid build scope')
    output=Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT/'quant_data/processed/v2'):raise ValueError('V2 output required')
    output.mkdir(parents=True,exist_ok=True);check_space(output,args.min_free_gb)
    history=max(2010,args.start_year-2);start=pd.Timestamp(f'{args.start_year}-01-01');end=pd.Timestamp(f'{args.end_year+1}-01-01')
    paths={'base':files(ROOT/'quant_data/processed/features_50','feature_year',history,args.end_year)}
    for name in ('daily','adj_factor','daily_basic','moneyflow'):
        paths[name]=files(ROOT/'quant_data/processed/parquet'/name,'year',history,args.end_year)
    sources=[Path(__file__),_source_path('v2/balanced_features.py'),_source_path('v2/stable_rolling.py'),_source_path('v2/features_500.py'),_source_path('v2/build_balanced_features.py'),_source_path('v2/research_design/factor_catalog_500.json')]
    config=dict(args=vars(args),catalog_version=CATALOG['version'],storage='float32; full float64 calculation',
        inputs={k:[dict(path=str(f),bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in fps] for k,fps in paths.items()},
        sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        universe='local available historical data; survivor bias remains; user-authorized restricted research',test_used=False,labels_read=False)
    manifest=output/'build_config.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=config:raise ValueError('Changed configuration/source/input; use a new output directory')
    save_json(manifest,config)
    state=json.loads((output/'progress.json').read_text(encoding='utf-8')) if (output/'progress.json').exists() else dict(staged_years=[],completed_buckets=[],completed_months=[])
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='4GB'")
    con.execute('SET temp_directory=?',[str(output/'_duckdb_tmp')]);con.execute('SET preserve_insertion_order=false')
    for name,fps in paths.items():con.read_parquet([str(f) for f in fps],hive_partitioning=False).create_view(name)
    codes=[r[0] for r in con.execute('SELECT DISTINCT ts_code FROM base WHERE signal_date>=? AND signal_date<? ORDER BY ts_code',[start,end]).fetchall()]
    if args.max_stocks:codes=codes[:args.max_stocks]
    if not codes:raise ValueError('Empty universe')
    selected=pd.DataFrame({'ts_code':codes,'bucket':np.arange(len(codes))//args.bucket_size});con.register('selected',selected)
    total_buckets=int(selected.bucket.max()+1)
    # Context always uses the full locally available pool, even during QA.
    if not all((output/name).exists() for name in ('market.parquet','calendar.parquet')):
        market=con.execute('SELECT signal_date,avg(logret_1d) rm,median(turnover_rate) market_median_turnover FROM base GROUP BY signal_date ORDER BY signal_date').fetch_df()
        save_frame(output/'market.parquet',market)
        calendar=con.execute('SELECT DISTINCT trade_date AS signal_date FROM daily ORDER BY signal_date').fetch_df()
        save_frame(output/'calendar.parquet',calendar)
    market=pd.read_parquet(output/'market.parquet').set_index('signal_date')
    calendar=pd.DatetimeIndex(pd.read_parquet(output/'calendar.parquet').signal_date,name='signal_date')
    for year in range(history,args.end_year+1):
        if year in state['staged_years']:continue
        check_space(output,args.min_free_gb);write_status(output,state,phase='staging',year=year,stocks=len(codes),total_buckets=total_buckets)
        scratch=output/f'stage_year_{year}';safe_remove_tree(scratch,output)
        lo=f'{year}-01-01';hi=f'{year+1}-01-01'
        for name in paths:
            date='signal_date' if name=='base' else 'trade_date'
            con.execute(f"CREATE OR REPLACE TEMP VIEW current_{name} AS SELECT * FROM {name} WHERE {date}>=DATE '{lo}' AND {date}<DATE '{hi}'")
        basecols=','.join(f'b."{c}"' for c in BASE_FEATURES)
        flows=','.join(f'm.{side}_{size}_amount' for side in ('buy','sell') for size in ('sm','md','lg','elg'))
        query=f'''SELECT b.ts_code,b.signal_date,{basecols},s.bucket,
            d.open*a.adj_factor O,d.high*a.adj_factor H,d.low*a.adj_factor L,d.close*a.adj_factor C,
            d.close raw_close,d.pre_close raw_pre_close,d.vol volume,d.amount*1000 amount_yuan,
            db.volume_ratio source_volume_ratio,db.dv_ttm,db.pe_ttm,db.total_mv,db.circ_mv,{flows},m.net_mf_amount
            FROM current_base b JOIN selected s USING(ts_code)
            JOIN current_daily d ON b.ts_code=d.ts_code AND b.signal_date=d.trade_date
            JOIN current_adj_factor a ON b.ts_code=a.ts_code AND b.signal_date=a.trade_date
            LEFT JOIN current_daily_basic db ON b.ts_code=db.ts_code AND b.signal_date=db.trade_date
            LEFT JOIN current_moneyflow m ON b.ts_code=m.ts_code AND b.signal_date=m.trade_date'''
        destination=str(scratch).replace("'","''").replace('\\','/')
        con.execute(f"COPY ({query}) TO '{destination}' (FORMAT PARQUET, PARTITION_BY(bucket), COMPRESSION ZSTD)")
        state['staged_years'].append(year);write_status(output,state)
        print(f'Staged {year}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for bucket in range(total_buckets):
            if bucket in state['completed_buckets']:continue
            check_space(output,args.min_free_gb);write_status(output,state,phase='factors',bucket=bucket)
            fp=[str(f) for year in range(history,args.end_year+1) for f in sorted((output/f'stage_year_{year}'/f'bucket={bucket}').glob('*.parquet'))]
            if not fp:raise ValueError('Missing staged bucket')
            data=con.read_parquet(fp,hive_partitioning=False).order('ts_code,signal_date').fetchdf()
            tasks=((code,part,calendar,market,start,end) for code,part in data.groupby('ts_code',sort=True))
            result=pd.concat(pool.map(calculate_stock,tasks,chunksize=1),ignore_index=True)
            del data
            for year,part in result.groupby(result.signal_date.dt.year):
                save_frame(output/'_factor_cache'/f'year={year}'/f'bucket_{bucket:04d}.parquet',part)
            del result
            state['completed_buckets'].append(bucket);write_status(output,state)
            # Only scratch input files for this verified completed bucket are removed.
            for path in fp:
                f=Path(path).resolve()
                if not f.is_relative_to(output):raise ValueError('Unsafe scratch file path')
                f.unlink()
            print(f'Factor buckets {len(state["completed_buckets"])}/{total_buckets}',flush=True)
    # At most one month of full cross-section is in pandas; statistics are
    # diagnostics only and never used as training preprocessing parameters.
    for year in range(args.start_year,args.end_year+1):
        fps=list((output/'_factor_cache'/f'year={year}').glob('*.parquet'))
        if not fps and all(f'{year}-{month:02d}' in state['completed_months'] for month in range(1,13)):continue
        if not fps:raise ValueError(f'Missing year cache {year}')
        con.read_parquet([str(f) for f in fps],hive_partitioning=False).create_view('year_features',replace=True)
        for month in range(1,13):
            key=f'{year}-{month:02d}'
            if key in state['completed_months']:continue
            check_space(output,args.min_free_gb);write_status(output,state,phase='monthly_cross_section',month=key)
            lo=pd.Timestamp(year=year,month=month,day=1);hi=lo+pd.offsets.MonthBegin(1)
            frame=con.execute('SELECT * FROM year_features WHERE signal_date>=? AND signal_date<? ORDER BY signal_date,ts_code',[lo,hi]).fetch_df()
            if frame.duplicated(['ts_code','signal_date']).any():raise ValueError('Duplicate final keys')
            frame,blocked=finish_cross_section(frame)
            frame[COLUMNS]=frame[COLUMNS].astype(np.float32)
            frame=frame.copy()
            values=frame[COLUMNS].to_numpy();finite=np.isfinite(values)
            if np.isinf(values).any() or frame[blocked].notna().any().any():raise ValueError('Invalid final feature values')
            frame['feature_valid_count']=finite.sum(axis=1).astype('int16')
            save_frame(output/f'feature_year={year}'/f'month_{month:02d}.parquet',frame)
            n=finite.sum(axis=0);diag=dict(month=key,rows=len(frame),stocks=int(frame.ts_code.nunique()),
                finite_counts=n.tolist(),missing_counts=(len(frame)-n).tolist(),blocked_features=blocked,
                min_values=[float(v) if np.isfinite(v) else None for v in frame[COLUMNS].min().to_numpy()],
                max_values=[float(v) if np.isfinite(v) else None for v in frame[COLUMNS].max().to_numpy()],
                duplicate_keys=0,infinite_cells=0)
            (output/'quality').mkdir(exist_ok=True);save_json(output/'quality'/f'{key}.json',diag)
            state['completed_months'].append(key);write_status(output,state)
            print(f'Final {key}: {len(frame)} rows',flush=True)
        # All twelve month outputs and diagnostics have committed.
        for f in fps:f.unlink()
    diagnostics=[json.loads((output/'quality'/f'{key}.json').read_text(encoding='utf-8')) for key in sorted(state['completed_months'])]
    for diag in diagnostics:
        year,month=diag['month'].split('-');target=output/f'feature_year={year}'/f'month_{month}.parquet'
        if not target.exists() or pq.ParquetFile(target).metadata.num_rows!=diag['rows']:
            raise ValueError(f'Missing or damaged completed partition: {target}')
    total=sum(d['rows'] for d in diagnostics);counts=np.sum([d['finite_counts'] for d in diagnostics],axis=0)
    blocked=diagnostics[0]['blocked_features']
    expected=con.execute('SELECT count(*) FROM base JOIN selected USING(ts_code) WHERE signal_date>=? AND signal_date<?',[start,end]).fetchone()[0]
    if total!=expected:raise ValueError(f'Join row loss: expected {expected}, got {total}')
    report=dict(status='completed_with_explicit_gates',rows=total,stocks=len(codes),schema_features=500,
        finite_columns=int((counts>0).sum()),computable_nonfinancial_features=500-len(blocked),blocked_features=blocked,
        coverage={c:float(n/total) for c,n in zip(COLUMNS,counts)},scope='integration_QA_subset' if args.max_stocks else 'local_survivor_research',
        universe_status='known_survivor_bias; user authorized local-data research; not unbiased full-market validation',
        market_context='full locally available source pool',test_used=False,labels_read=False,storage='float32',
        expected_source_rows=int(expected),monthly_partitions=len(diagnostics),quality_summary='keys/infinities/gates checked per month; detailed monthly counts and ranges retained')
    save_json(output/'feature_report.json',report);write_status(output,state,phase='completed',rows=total)
    print(f'COMPLETE: {total} rows, {len(codes)} stocks, {len(diagnostics)} months',flush=True);con.close()


if __name__=='__main__':
    # Prevent numerical libraries in worker processes from multiplying threads.
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
    main()
