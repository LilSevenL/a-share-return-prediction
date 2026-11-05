"""Fixed five-session review cadence; use only prior-close eligible-pool ranks."""
import numpy as np
import pandas as pd


def buffer_panel(orders,panel,keep_rank):
    if keep_rank<1:raise ValueError('Positive keep threshold required')
    dates,close,sellable,next_sellable,lookup=panel;n=len(dates)
    ranks=np.full(close.shape,np.inf)
    ordered=orders.sort_values(['signal_date','prediction','ts_code'],ascending=[True,False,True]).copy()
    ordered['rank']=ordered.groupby('signal_date').cumcount()+1
    for day,g in ordered.groupby('signal_date',sort=False):
        index=lookup.get(np.datetime64(day,'D'))
        if index is not None:ranks[g.code_idx.to_numpy(dtype=int),index]=g['rank'].to_numpy()
    actual=np.full(close.shape,-1,dtype=np.int32);decision=np.full(close.shape,-1,dtype=np.int32)
    # Store the realized exit date implied by the causal policy, like the existing
    # next-sellable table. Cash is credited only on that date, never when scheduled.
    for day in range(n-1,-1,-1):
        retain=ranks[:,day-1]<=keep_rank if day>0 else np.zeros(len(close),bool)
        actual[:,day]=next_sellable[:,day];decision[:,day]=day
        if day+5<n:
            actual[retain,day]=actual[retain,day+5];decision[retain,day]=decision[retain,day+5]
        else:actual[retain,day]=-1;decision[retain,day]=-1
    return (dates,close,sellable,actual,lookup),decision


def annotate_orders(trades,decision,panel,orders):
    dates,_,_,_,lookup=panel
    codes=orders[['ts_code','code_idx']].drop_duplicates().set_index('ts_code').code_idx.to_dict()
    result=trades.copy();result['buffer_extended']=False;result['liquidity_delayed']=False
    result['exit_decision_date']=None
    for idx,row in result.loc[result.status=='filled'].iterrows():
        planned=lookup[np.datetime64(row.scheduled_exit_date,'D')];d=int(decision[codes[row.ts_code],planned])
        result.at[idx,'buffer_extended']=d<0 or d>planned
        if d>=0:
            result.at[idx,'exit_decision_date']=str(dates[d])
            result.at[idx,'liquidity_delayed']=pd.isna(row.actual_exit_date) or np.datetime64(row.actual_exit_date,'D')>dates[d]
    return result
