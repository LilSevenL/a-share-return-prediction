"""Matched three-way execution ablation on 500k compact signals."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import gc
import numpy as np,pandas as pd
from run_expanded_backtest import signals,read,save,sha,ROOT,DATA
from strict_execution_market import prepare
from lot_execution_simulation import simulate_released_cash,buy_quantity
from holding_buffer import buffer_panel,annotate_orders
from run_continuous_backtest import audit_ledger
OUT=DATA/'v2/execution_hardening_500k';PRIOR=DATA/'v2/expanded_continuous_500k'

def main():
    OUT.mkdir(exist_ok=True);parent=read(PRIOR/'manifest.json')
    for path,digest in parent['sources'].items():assert sha(ROOT/path)==digest
    for item in parent['inputs']:
        p=__import__('pathlib').Path(item['path']);assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns']
    files=['run_execution_hardening.py','strict_execution_market.py','lot_execution_simulation.py','holding_buffer.py','run_continuous_backtest.py','continuous_market.py','simulation.py','run_expanded_backtest.py']
    manifest=dict(sources={str((_source_path('v2/'+n)).relative_to(ROOT)):sha(_source_path('v2/'+n)) for n in files},inputs=parent['inputs'],aum=1e8,cost_bps=20,impact=.5,period=[2017,2023],variants=['baseline','strict_limits','strict_limits_lots'],automatic_selection=False)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    save(OUT/'manifest.json',manifest)
    print('Preparing raw-price and limit evidence',flush=True)
    s=signals('compact','huber');template,original,strict,missing,evidence=prepare(s);keys=s[['ts_code','signal_date']].copy();template=template.drop(columns='prediction');del s
    sell_evidence=evidence[['ts_code','trade_date','strict_sell']].rename(columns={'trade_date':'actual_exit_date'})
    results=[]
    for loss in ['huber','mae']:
        s=signals('compact',loss);pd.testing.assert_frame_equal(keys,s[['ts_code','signal_date']])
        base=template.merge(s[['ts_code','signal_date','prediction']],on=['ts_code','signal_date'],validate='one_to_one');del s
        for variant in manifest['variants']:
            orders=base.copy();panel=original if variant=='baseline' else strict
            if variant!='baseline':orders['entry_tradable']=orders.entry_tradable.fillna(False)&orders.strict_buy
            for policy,threshold in [('fixed5',None),('keep50',50),('keep100',100)]:
                stem=f'{loss}_{variant}_{policy}';dest=OUT/f'{stem}.json'
                if dest.exists():results.append(read(dest));continue
                print('Running '+stem,flush=True)
                selected,decision=buffer_panel(orders,panel,threshold) if threshold else (panel,None)
                m,nav,trades=simulate_released_cash(orders,selected,round_lots=variant=='strict_limits_lots')
                if decision is not None:trades=annotate_orders(trades,decision,panel,orders)
                else:trades['exit_decision_date']=trades.scheduled_exit_date
                np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*1e8)
                audit=audit_ledger(trades,nav,orders,panel,missing)
                filled=trades[trades.status=='filled'].copy()
                if variant=='baseline':
                    old=pd.read_parquet(PRIOR/f'compact_{loss}_{policy}_cost20_nav.parquet');check=nav.copy()
                    old['date']=old.date.astype('datetime64[ns]');check['date']=check.date.astype('datetime64[ns]');pd.testing.assert_frame_equal(old,check,check_exact=True)
                else:
                    filled['signal_date']=pd.to_datetime(filled.signal_date)
                    f=filled.merge(orders[['ts_code','signal_date','strict_buy']],on=['ts_code','signal_date'],validate='one_to_one');assert f.strict_buy.all()
                    sold=filled[filled.actual_exit_date.notna()].copy();sold['actual_exit_date']=pd.to_datetime(sold.actual_exit_date)
                    sold=sold.merge(sell_evidence,on=['ts_code','actual_exit_date'],validate='many_to_one');assert len(sold)==filled.actual_exit_date.notna().sum() and sold.strict_sell.all()
                    assert audit['filled_entry_missing_limits']==audit['sold_exit_missing_limits']==0
                if variant=='strict_limits_lots':
                    assert all(buy_quantity(r.ts_code,r.buy_raw_quantity,r.entry_date)==r.buy_raw_quantity for r in filled.itertuples())
                    np.testing.assert_allclose(filled.buy_raw_quantity*filled.entry_raw_open,filled.buy_notional)
                    assert (filled.budget<=filled.allocated_budget+1e-7).all()
                m.update(loss=loss,policy=policy,variant=variant,stem=stem,audit=audit,quantity_rejections=int((trades.status=='below_minimum_quantity').sum()),annualized_return=float(nav.nav.iloc[-1]**(252/len(nav))-1))
                nav.to_parquet(OUT/f'{stem}_nav.parquet',index=False);trades.to_parquet(OUT/f'{stem}_orders.parquet',index=False);save(dest,m);results.append(m)
                save(OUT/'summary.json',dict(status='running',results=results));print(f'{stem}: {m["net_return"]:.2%}',flush=True)
            gc.collect()
    save(OUT/'summary.json',dict(status='completed',results=results));print('ALL 18 COMPLETE',flush=True)
if __name__=='__main__':main()
