"""Precommitted AUM/impact grid; compact 500k signals, unchanged execution."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import gc
import numpy as np,pandas as pd
from run_expanded_backtest import signals,read,save,sha,ROOT,DATA
from continuous_market import prepare_continuous
from holding_buffer import buffer_panel,annotate_orders
from cash_release_simulation import simulate_released_cash
from run_continuous_backtest import audit_ledger

OUT=DATA/'v2/capacity_stress_500k';PRIOR=DATA/'v2/expanded_continuous_500k'

def normalized_audit(trades,nav,orders,panel,missing,aum):
    # The proven ledger auditor uses 100m monetary units; prices and shares
    # implied by normalized notionals remain consistent. Actual outputs stay raw.
    t=trades.copy();n=nav.copy();scale=1e8/aum
    for col in ['budget','buy_notional','sell_notional','buy_cost_amount','sell_cost_amount']:t[col]*=scale
    n['cash']*=scale;n['holdings']*=scale
    audit=audit_ledger(t,n,orders,panel,missing)
    audit['monetary_unit']='normalized to initial 100 million CNY; raw artifacts use actual AUM'
    return audit

def main():
    OUT.mkdir(exist_ok=True);parent=read(PRIOR/'manifest.json')
    assert read(PRIOR/'artifact_verification.json')['cases']==36
    for path,digest in parent['sources'].items():assert sha(ROOT/path)==digest
    for item in parent['inputs']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns']
    names=['run_capacity_stress.py','run_expanded_backtest.py','continuous_market.py','holding_buffer.py','cash_release_simulation.py','run_continuous_backtest.py','simulation.py']
    manifest=dict(sources={str((_source_path('v2/'+n)).relative_to(ROOT)):sha(_source_path('v2/'+n)) for n in names},inputs=parent['inputs'],aum=[10000000,100000000,500000000],impact=[.5,1.],cost_bps=20,participation_cap=.05,kind='compact',training_rows=500000,period=[2017,2023],policies=['fixed5','keep50','keep100'],losses=['huber','mae'],automatic_selection=False)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    save(OUT/'manifest.json',manifest)
    print('Preparing common market',flush=True)
    s=signals('compact','huber');template,panel,missing=prepare_continuous(s);keys=s[['ts_code','signal_date']].copy();template=template.drop(columns='prediction');del s
    results=[]
    for loss in ['huber','mae']:
        s=signals('compact',loss);pd.testing.assert_frame_equal(keys,s[['ts_code','signal_date']])
        orders=template.merge(s[['ts_code','signal_date','prediction']],on=['ts_code','signal_date'],validate='one_to_one');del s
        for policy,threshold in [('fixed5',None),('keep50',50),('keep100',100)]:
            selected,decision=buffer_panel(orders,panel,threshold) if threshold else (panel,None)
            for aum in manifest['aum']:
                for impact in manifest['impact']:
                    stem=f'{loss}_{policy}_aum{aum}_impact{impact:g}';dest=OUT/f'{stem}.json'
                    if dest.exists():results.append(read(dest));continue
                    print('Running '+stem,flush=True)
                    m,nav,trades=simulate_released_cash(orders,selected,aum=aum,cost_bps=20,impact_coefficient=impact,participation_cap=.05)
                    if decision is not None:trades=annotate_orders(trades,decision,panel,orders)
                    else:trades['exit_decision_date']=trades.scheduled_exit_date
                    assert np.isfinite(nav.nav).all() and (nav.nav>0).all()
                    np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*aum)
                    np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
                    audit=normalized_audit(trades,nav,orders,panel,missing,aum)
                    if aum==100000000 and impact==.5:
                        old=pd.read_parquet(PRIOR/f'compact_{loss}_{policy}_cost20_nav.parquet')
                        old['date']=old.date.astype('datetime64[ns]')
                        check=nav.copy();check['date']=check.date.astype('datetime64[ns]')
                        pd.testing.assert_frame_equal(old,check,check_exact=True)
                    filled=trades[trades.status=='filled'];base=float((filled.buy_notional*.0005+filled.sell_notional*.0015).sum())
                    m.update(loss=loss,policy=policy,aum=aum,impact_coefficient=impact,stem=stem,audit=audit,capacity_rejected=int((trades.status=='capacity_rejected').sum()),base_cost_amount=base,impact_cost_amount=m['total_cost_amount']-base,annualized_return=float(nav.nav.iloc[-1]**(252/len(nav))-1))
                    nav.to_parquet(OUT/f'{stem}_nav.parquet',index=False);trades.to_parquet(OUT/f'{stem}_orders.parquet',index=False);save(dest,m);results.append(m)
                    save(OUT/'summary.json',dict(status='running',results=results));print(f'{stem}: return={m["net_return"]:.2%}, capacity_rejected={m["capacity_rejected"]}',flush=True)
            gc.collect()
    save(OUT/'summary.json',dict(status='completed',results=results))
    print('ALL 36 STRESS CASES COMPLETE',flush=True)
if __name__=='__main__':main()
