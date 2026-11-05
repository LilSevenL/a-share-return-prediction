"""Common daily-bar execution proxy. Ranking precedes execution checks; failed orders stay cash."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from backtest_validation import build_price_panel

ROOT=_REPO_ROOT
DATA=ROOT/'quant_data/processed'


def source_files(directory,partition,years):
    files=[str(p) for y in years for p in sorted(directory.glob(f'{partition}={y}/*.parquet'))]
    if not files:raise FileNotFoundError(directory)
    return files


def prepare_market(signals,year):
    if year>2024:raise ValueError('Permanent Test prohibited')
    con=duckdb.connect();con.execute('SET threads=4')
    for source,view in [('daily','daily_data'),('adj_factor','adj_data'),('stk_limit','limit_data')]:
        con.read_parquet(source_files(DATA/'parquet'/source,'year',[year]),hive_partitioning=False).create_view(view)
    con.register('signals',signals)
    con.execute('CREATE TABLE price_codes AS SELECT ts_code,row_number() OVER(ORDER BY ts_code)-1 code_idx FROM (SELECT DISTINCT ts_code FROM signals)')
    con.execute('CREATE TABLE price_calendar AS SELECT trade_date,row_number() OVER(ORDER BY trade_date)-1 date_idx FROM (SELECT DISTINCT trade_date FROM daily_data)')
    panel=build_price_panel(con)
    con.read_parquet(source_files(DATA/'targets_5d','signal_year',[year]),hive_partitioning=False).create_view('targets')
    basic=pd.read_csv(ROOT/'quant_data/combined/stock_basic.csv',dtype={'list_date':str})[['ts_code','list_date']]
    basic['list_date']=pd.to_datetime(basic.list_date,format='%Y%m%d',errors='coerce');con.register('basic',basic)
    # LEFT JOIN targets: future label availability never determines the ranking universe.
    orders=con.execute('''SELECT s.*,pc.code_idx,d.amount*1000 amount_yuan,
      t.entry_date,t.exit_date scheduled_exit_date,t.entry_adj_open,t.entry_tradable
      FROM signals s JOIN daily_data d ON s.ts_code=d.ts_code AND s.signal_date=d.trade_date
      JOIN price_codes pc ON s.ts_code=pc.ts_code LEFT JOIN basic b ON s.ts_code=b.ts_code
      LEFT JOIN targets t ON s.ts_code=t.ts_code AND s.signal_date=t.signal_date
      WHERE d.amount>=50000 AND b.list_date IS NOT NULL
        AND date_diff('day',b.list_date,s.signal_date)>=60''').fetch_df()
    con.close()
    return orders,panel


def simulate(orders,panel,top_n=50,aum=1e8,cost_bps=20.,participation_cap=.05,impact_coefficient=.5):
    dates,close,_,next_sellable,lookup=panel;n=len(dates)
    if top_n<1 or aum<=0 or cost_bps<0:raise ValueError('Invalid portfolio settings')
    if orders.duplicated(['ts_code','signal_date']).any() or not np.isfinite(orders.prediction).all():raise ValueError('Invalid order keys/predictions')
    sleeves=np.full((5,n),aum/5,dtype=float);busy=np.full(5,-1);records=[];traded=0.;busy_days=0
    for day,rows in orders.groupby('signal_date',sort=True):
        si=lookup.get(np.datetime64(day,'D'));entry=si+1 if si is not None else n
        if entry>=n:continue
        slot=si%5
        if busy[slot]>=entry:busy_days+=1;continue
        capital=sleeves[slot,entry-1];budget=capital/top_n
        cohort=np.full(n-entry,capital);last_exit=entry-1
        chosen=rows.sort_values(['prediction','ts_code'],ascending=[False,True]).head(top_n)
        for row in chosen.itertuples():
            record=dict(signal_date=str(day),ts_code=row.ts_code,budget=float(budget),status='unfilled')
            records.append(record)
            if pd.isna(row.entry_date) or np.datetime64(row.entry_date,'D')!=dates[entry]:continue
            planned=lookup.get(np.datetime64(row.scheduled_exit_date,'D')) if pd.notna(row.scheduled_exit_date) else None
            # No new positions whose planned exit lies outside this evaluation period.
            if planned is None:record['status']='outside_period';continue
            if pd.isna(row.entry_tradable) or not bool(row.entry_tradable):continue
            if not np.isfinite(row.entry_adj_open) or row.entry_adj_open<=0:continue
            if not np.isfinite(row.amount_yuan) or row.amount_yuan<=0:record['status']='missing_liquidity';continue
            participation=budget/row.amount_yuan
            if participation>participation_cap:record['status']='capacity_rejected';continue
            if not np.isfinite(row.volatility_20d) or row.volatility_20d<=0:record['status']='missing_risk';continue
            impact=impact_coefficient*row.volatility_20d*np.sqrt(participation)
            buy=cost_bps/10000*.25+impact;sell=cost_bps/10000*.75+impact
            if sell>=1:record['status']='invalid_cost';continue
            code=int(row.code_idx);actual=int(next_sellable[code,planned]);shares=budget/(row.entry_adj_open*(1+buy))
            values=shares*close[code,entry:].astype(float)
            # Only actual holding dates require a valuation; post-sale prices are irrelevant.
            holding_values=values[:actual-entry+1] if actual>=0 else values
            if not np.isfinite(holding_values).all():raise ValueError('Missing held-position valuation')
            traded+=shares*row.entry_adj_open
            if actual>=0:
                proceeds=shares*close[code,actual]*(1-sell)
                values[actual-entry:]=proceeds;traded+=shares*close[code,actual]
                last_exit=max(last_exit,actual)
            else:last_exit=n-1
            cohort+=values-budget
            record.update(status='filled',entry_date=str(dates[entry]),scheduled_exit_date=str(dates[planned]),
                actual_exit_date=str(dates[actual]) if actual>=0 else None,participation=float(participation),
                buy_cost_rate=float(buy),sell_cost_rate=float(sell),unresolved_at_end=actual<0,
                delayed_exit=actual<0 or actual>planned,buy_notional=float(shares*row.entry_adj_open),
                sell_notional=float(shares*close[code,actual]) if actual>=0 else 0.,
                buy_cost_amount=float(shares*row.entry_adj_open*buy),
                sell_cost_amount=float(shares*close[code,actual]*sell) if actual>=0 else 0.)
        sleeves[slot,entry:]=cohort;busy[slot]=last_exit
    nav=sleeves.sum(axis=0)/aum;previous=np.r_[1.,nav[:-1]];returns=nav/previous-1
    peak=np.maximum.accumulate(np.r_[1.,nav])[1:];std=returns.std(ddof=1)
    filled=sum(r['status']=='filled' for r in records)
    metrics=dict(net_return=float(nav[-1]-1),annualized_return=float(nav[-1]**(252/n)-1),
        sharpe=float(returns.mean()/std*np.sqrt(252)) if std>0 else None,
        max_drawdown=float(np.min(nav/peak-1)),annualized_gross_traded_over_aum=float(traded/aum*252/n),
        orders=len(records),filled_orders=filled,cost_bps=cost_bps,top_n=top_n,aum=aum,
        unresolved_positions=sum(r.get('unresolved_at_end',False) for r in records),
        delayed_exit_orders=sum(r.get('delayed_exit',False) for r in records),blocked_sleeve_days=busy_days,
        annualized_two_way_turnover=float(traded/(aum*nav.mean())*252/n),
        total_cost_amount=float(sum(r.get('buy_cost_amount',0)+r.get('sell_cost_amount',0) for r in records)),
        execution_model='daily_bar_proxy; equal order budgets; no replacement; cash retained; delayed sale; ex-ante impact')
    return metrics,pd.DataFrame({'date':dates,'nav':nav,'daily_return':returns}),pd.DataFrame(records)
