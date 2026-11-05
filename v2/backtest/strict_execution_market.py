"""Execution-day limit evidence; no changes to signal ranking or target labels."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import duckdb,numpy as np,pandas as pd
from continuous_market import prepare_continuous
from simulation import DATA,source_files

def strict_panel(panel,evidence):
    dates,close,sellable,_,lookup=panel
    can=np.zeros_like(sellable)
    for d,g in evidence.groupby('trade_date'):
        index=lookup[np.datetime64(d,'D')]
        can[g.code_idx.to_numpy(int),index]=g.strict_sell.to_numpy(bool)
    can &= sellable & np.isfinite(close)
    next_exit=np.full(can.shape,-1,dtype=np.int32);nxt=np.full(can.shape[0],-1,dtype=np.int32)
    for j in range(len(dates)-1,-1,-1):nxt[can[:,j]]=j;next_exit[:,j]=nxt
    return dates,close,can,next_exit,lookup

def prepare(signals):
    orders,panel,missing=prepare_continuous(signals)
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='3GB'")
    for source,name in [('daily','d'),('stk_limit','l')]:con.read_parquet(source_files(DATA/'parquet'/source,'year',range(2017,2024)),hive_partitioning=False).create_view(name)
    codes=orders[['ts_code','code_idx']].drop_duplicates();con.register('codes',codes)
    evidence=con.execute('''SELECT d.ts_code,d.trade_date,c.code_idx,d.open entry_raw_open,
        coalesce(l.up_limit>0 AND l.down_limit>0 AND l.up_limit>=l.down_limit,false) limits_known,
        coalesce(l.up_limit>0 AND l.down_limit>0 AND l.up_limit>=l.down_limit AND d.open>0 AND d.open<l.up_limit-1e-6,false) strict_buy,
        coalesce(l.up_limit>0 AND l.down_limit>0 AND l.up_limit>=l.down_limit AND d.close>l.down_limit+1e-6 AND d.vol>0,false) strict_sell
        FROM d JOIN codes c USING(ts_code) LEFT JOIN l USING(ts_code,trade_date)''').fetch_df();con.close()
    assert not evidence.duplicated(['ts_code','trade_date']).any()
    orders=orders.merge(evidence[['ts_code','trade_date','entry_raw_open','strict_buy','limits_known']].rename(columns={'trade_date':'entry_date'}),on=['ts_code','entry_date'],how='left',validate='many_to_one')
    orders['strict_buy']=orders.strict_buy.fillna(False).astype(bool)
    return orders,panel,strict_panel(panel,evidence),missing,evidence
