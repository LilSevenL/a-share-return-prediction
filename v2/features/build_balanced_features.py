"""Build current balanced factor design. Financial PIT gate is mandatory.

No target data or 2025+ files are opened. Small runs are labelled integration QA.
"""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import argparse
from pathlib import Path
import hashlib
import json
import shutil
import sys
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from balanced_features import CATALOG,COLUMNS,technical_stock,finish_cross_section

ROOT=_REPO_ROOT


def files(root,partition,first,last):
    if last>2024: raise ValueError('No permanent-Test partition access')
    found=[p for year in range(first,last+1) for p in sorted((root/f'{partition}={year}').glob('*.parquet'))]
    if not found: raise FileNotFoundError(root)
    return found


def save_json(path,data):
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8');temp.replace(path)


def save_frame(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),temp,compression='zstd')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start-year',type=int,default=2011);p.add_argument('--end-year',type=int,default=2023)
    p.add_argument('--max-stocks',type=int);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output-dir',default=str(ROOT/'quant_data/processed/v2/balanced_500'))
    args=p.parse_args()
    if not 2011<=args.start_year<=args.end_year<=2024 or args.threads<1 or (args.max_stocks is not None and args.max_stocks<1): raise ValueError('Invalid build scope')
    output=Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT/'quant_data/processed/v2'): raise ValueError('V2 directory required')
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<8*1024**3: raise RuntimeError('Insufficient disk space')
    start=f'{args.start_year}-01-01';end=f'{args.end_year+1}-01-01';history=max(2010,args.start_year-2)
    paths={'base':files(ROOT/'quant_data/processed/features_50','feature_year',history,args.end_year)}
    for name in ('daily','adj_factor','daily_basic','moneyflow'):
        paths[name]=files(ROOT/'quant_data/processed/parquet'/name,'year',history,args.end_year)
    source_files=[Path(__file__),_source_path('v2/balanced_features.py'),_source_path('v2/features_500.py'),_source_path('v2/research_design/factor_catalog_500.json')]
    config=dict(args=vars(args),catalog_version=CATALOG['version'],financial_gate='blocked_pending_PIT_audit',
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        inputs={k:[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in v] for k,v in paths.items()})
    conf=output/'build_config.json'
    if conf.exists() and json.loads(conf.read_text(encoding='utf-8'))!=config: raise ValueError('Changed sources/config; use new output directory')
    save_json(conf,config)
    progress_path=output/'progress.json'
    progress=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else dict(completed_stocks=[])
    con=duckdb.connect(); con.execute(f'SET threads={args.threads}');con.execute("SET memory_limit='6GB'")
    con.execute('SET temp_directory=?',[str(output/'_duckdb_tmp')])
    con.execute('SET preserve_insertion_order=false')
    try:
        for name,fp in paths.items(): con.read_parquet([str(p) for p in fp],hive_partitioning=False).create_view(name)
        con.execute('CREATE TEMP TABLE market AS SELECT signal_date,avg(logret_1d) rm,median(turnover_rate) market_median_turnover FROM base GROUP BY signal_date')
        calendar=pd.DatetimeIndex(con.execute('SELECT DISTINCT trade_date FROM daily ORDER BY trade_date').fetch_df().iloc[:,0],name='signal_date')
        codes=[r[0] for r in con.execute('SELECT DISTINCT ts_code FROM base WHERE signal_date>=? AND signal_date<? ORDER BY ts_code',[start,end]).fetchall()]
        if args.max_stocks:codes=codes[:args.max_stocks]
        if not codes: raise ValueError('No stocks')
        con.register('selected',pa.table({'ts_code':codes}))
        flowcols=', '.join(f'm.{side}_{size}_amount' for side in ('buy','sell') for size in ('sm','md','lg','elg'))
        con.execute(f'''CREATE TEMP TABLE input AS
          SELECT b.*,d.open*a.adj_factor O,d.high*a.adj_factor H,d.low*a.adj_factor L,d.close*a.adj_factor C,
            d.close raw_close,d.pre_close raw_pre_close,d.vol volume,d.amount*1000 amount_yuan,
            db.volume_ratio source_volume_ratio,db.dv_ttm,db.pe_ttm,db.total_mv,db.circ_mv,
            {flowcols},m.net_mf_amount,mk.rm,mk.market_median_turnover
          FROM base b JOIN selected USING(ts_code)
          JOIN daily d ON b.ts_code=d.ts_code AND b.signal_date=d.trade_date
          JOIN adj_factor a ON b.ts_code=a.ts_code AND b.signal_date=a.trade_date
          LEFT JOIN daily_basic db ON b.ts_code=db.ts_code AND b.signal_date=db.trade_date
          LEFT JOIN moneyflow m ON b.ts_code=m.ts_code AND b.signal_date=m.trade_date
          JOIN market mk USING(signal_date)''')
        con.execute('CREATE INDEX input_code ON input(ts_code)')
        market=con.execute('SELECT * FROM market ORDER BY signal_date').fetch_df().set_index('signal_date')
        print(f'Computing balanced catalog for {len(codes)} stocks; financial factors gated',flush=True)
        for n,code in enumerate(codes,1):
            if code in progress['completed_stocks']:continue
            if shutil.disk_usage(output).free<8*1024**3: raise RuntimeError('Low disk; progress retained')
            d=con.execute('SELECT * FROM input WHERE ts_code=? ORDER BY signal_date',[code]).fetch_df()
            if d.signal_date.duplicated().any(): raise ValueError('Duplicate input keys')
            observed=pd.DatetimeIndex(d.signal_date)
            d=d.set_index('signal_date').reindex(calendar)
            d['rm']=market.rm.reindex(calendar);d['market_median_turnover']=market.market_median_turnover.reindex(calendar)
            result=technical_stock(d)
            result=result.loc[result.index.isin(observed)&(result.index>=start)&(result.index<end)].copy()
            result.insert(0,'ts_code',code);result=result.reset_index()
            save_frame(output/'_stock_cache'/f'{code}.parquet',result)
            progress['completed_stocks'].append(code);save_json(progress_path,progress)
            print(f'{n}/{len(codes)} {code}: {len(result)} rows',flush=True)
        con.read_parquet([str(output/'_stock_cache'/f'{c}.parquet') for c in codes],hive_partitioning=False).create_view('stock_features')
        counts=np.zeros(500,dtype=np.int64);total=0;blocked=[]
        for year in range(args.start_year,args.end_year+1):
            for month in range(1,13):
                lo=pd.Timestamp(year=year,month=month,day=1);hi=lo+pd.offsets.MonthBegin(1)
                frame=con.execute('SELECT * FROM stock_features WHERE signal_date>=? AND signal_date<? ORDER BY signal_date,ts_code',[lo,hi]).fetch_df()
                if frame.empty:continue
                frame,blocked=finish_cross_section(frame)
                values=frame[COLUMNS].to_numpy(float);finite=np.isfinite(values)
                if np.isinf(values).any():raise ValueError('Infinite output')
                frame['feature_valid_count']=finite.sum(1).astype('int16')
                counts+=finite.sum(0);total+=len(frame)
                save_frame(output/f'feature_year={year}'/f'month_{month:02d}.parquet',frame)
        report=dict(status='completed_with_explicit_gates',rows=total,stocks=len(codes),schema_features=500,
            computable_nonfinancial_features=500-len(blocked),blocked_features=blocked,
            block_reason='Financial PIT unavailable; negative_earnings_flag cannot be inferred from missing loss-making PE; not imputed',
            universe_status='legacy_survivor_source_not_repaired; require historical-universe data completion before formal training',
            finite_columns=int((counts>0).sum()),coverage={c:float(n/total) for c,n in zip(COLUMNS,counts)},
            scope='integration_QA_subset' if args.max_stocks else 'all_available_stocks',
            market_context='computed on full source universe even in subset QA',test_used=False,labels_read=False)
        save_json(output/'feature_report.json',report)
        print(f'Completed {total} rows; {report["finite_columns"]} columns have data; {len(blocked)} explicitly blocked',flush=True)
    finally:con.close()


if __name__=='__main__':main()
