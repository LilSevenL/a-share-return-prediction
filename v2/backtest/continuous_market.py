"""Join saved yearly predictions to one uninterrupted 2017-2023 market calendar."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import sys
from pathlib import Path
ROOT=_REPO_ROOT;sys.path.insert(0,str(ROOT))
import duckdb
import pandas as pd
from simulation import DATA,source_files
from backtest_validation import build_price_panel


def prepare_continuous(signals):
    if signals.signal_date.dt.year.min()!=2017 or signals.signal_date.dt.year.max()!=2023:raise ValueError('Fixed research period required')
    if signals.duplicated(['ts_code','signal_date']).any():raise ValueError('Duplicate signal keys')
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='4GB'")
    for source,view in [('daily','daily_data'),('adj_factor','adj_data'),('stk_limit','limit_data')]:
        con.read_parquet(source_files(DATA/'parquet'/source,'year',range(2017,2024)),hive_partitioning=False).create_view(view)
    con.register('signals',signals)
    con.execute('CREATE TABLE price_codes AS SELECT ts_code,row_number() OVER(ORDER BY ts_code)-1 code_idx FROM (SELECT DISTINCT ts_code FROM signals)')
    con.execute('CREATE TABLE price_calendar AS SELECT trade_date,row_number() OVER(ORDER BY trade_date)-1 date_idx FROM (SELECT DISTINCT trade_date FROM daily_data)')
    panel=build_price_panel(con)
    con.read_parquet(source_files(DATA/'targets_5d','signal_year',range(2017,2024)),hive_partitioning=False).create_view('targets')
    basic=pd.read_csv(ROOT/'quant_data/combined/stock_basic.csv',dtype={'list_date':str})[['ts_code','list_date']]
    basic.list_date=pd.to_datetime(basic.list_date,format='%Y%m%d',errors='coerce');con.register('basic',basic)
    orders=con.execute('''SELECT s.*,pc.code_idx,d.amount*1000 amount_yuan,
        t.entry_date,t.exit_date scheduled_exit_date,t.entry_adj_open,t.entry_tradable
        FROM signals s JOIN daily_data d ON s.ts_code=d.ts_code AND s.signal_date=d.trade_date
        JOIN price_codes pc ON s.ts_code=pc.ts_code LEFT JOIN basic b ON s.ts_code=b.ts_code
        LEFT JOIN targets t ON s.ts_code=t.ts_code AND s.signal_date=t.signal_date
        WHERE d.amount>=50000 AND b.list_date IS NOT NULL AND date_diff('day',b.list_date,s.signal_date)>=60''').fetch_df()
    missing=con.execute('''SELECT d.ts_code,d.trade_date FROM daily_data d LEFT JOIN limit_data l USING(ts_code,trade_date)
        WHERE l.up_limit IS NULL OR l.down_limit IS NULL''').fetch_df()
    con.close()
    if orders.duplicated(['ts_code','signal_date']).any():raise ValueError('Duplicate orders after join')
    return orders,panel,missing
