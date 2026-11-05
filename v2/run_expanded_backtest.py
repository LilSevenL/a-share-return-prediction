"""Matched full/compact robust signals through common continuous execution."""
from pathlib import Path
import sys,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from continuous_market import prepare_continuous
from holding_buffer import buffer_panel,annotate_orders
from cash_release_simulation import simulate_released_cash
from run_continuous_backtest import audit_ledger
from run_multiyear_losses import read,save,sha
DATA=ROOT/'quant_data/processed';SOURCE=DATA/'v2/expanded_losses_500k';OUT=DATA/'v2/expanded_continuous_500k'

def signals(kind,loss):
    parts=[]
    for year in range(2017,2024):
        f=pd.read_parquet(SOURCE/f'validation_{year}/{kind}_{loss}/predictions.parquet',columns=['ts_code','signal_date','prediction','volatility_20d'])
        assert (f.signal_date.dt.year==year).all() and np.isfinite(f.prediction).all();parts.append(f)
    return pd.concat(parts,ignore_index=True)

def main():
    assert read(SOURCE/'verification.json')['models']==28;OUT.mkdir(exist_ok=True)
    prior=read(DATA/'v2/compact_continuous/manifest.json')
    for path,digest in prior['sources'].items():
        if path!='v2/run_compact_backtest.py':assert sha(ROOT/path)==digest, 'Execution engine changed: '+path
    files=[Path(__file__)]+[ROOT/'v2'/n for n in ['continuous_market.py','holding_buffer.py','cash_release_simulation.py','simulation.py','run_continuous_backtest.py']]+[ROOT/'backtest_validation.py']
    inputs=read(DATA/'v2/ols_continuous/manifest.json')['inputs'].copy()
    inputs += [dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for y in range(2017,2024) for p in [SOURCE/f'validation_{y}/{k}_{l}/predictions.parquet' for k in ['full','compact'] for l in ['huber','mae']]]
    for item in inputs:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns']
    manifest=dict(inputs=inputs,sources={str(p.relative_to(ROOT)):sha(p) for p in files},initial_capital=1e8,annual_reset=False,
        training_rows_per_fold=500000,period=[2017,2023],scope='500k historical development full versus compact; fixed previous execution rules',automatic_selection=False)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    save(OUT/'manifest.json',manifest)
    print('Preparing shared market',flush=True);s=signals('full','huber');template,panel,missing=prepare_continuous(s)
    keys=s[['ts_code','signal_date']].copy();template=template.drop(columns='prediction');del s;gc.collect();results=[]
    for kind in ['full','compact']:
        for loss in ['huber','mae']:
            s=signals(kind,loss);pd.testing.assert_frame_equal(keys,s[['ts_code','signal_date']]);orders=template.merge(s[['ts_code','signal_date','prediction']],on=['ts_code','signal_date'],validate='one_to_one');del s
            for policy,threshold in [('fixed5',None),('keep50',50),('keep100',100)]:
                selected,decision=buffer_panel(orders,panel,threshold) if threshold else (panel,None)
                for scenario,cost,impact in [('gross',0.,0.),('cost20',20.,.5),('cost50',50.,.5)]:
                    stem=f'{kind}_{loss}_{policy}_{scenario}';dest=OUT/(stem+'.json')
                    if dest.exists():results.append(read(dest));continue
                    print('Running '+stem,flush=True)
                    m,nav,trades=simulate_released_cash(orders,selected,cost_bps=cost,impact_coefficient=impact)
                    if decision is not None:trades=annotate_orders(trades,decision,panel,orders)
                    else:trades['exit_decision_date']=trades.scheduled_exit_date
                    assert np.isfinite(nav.nav).all() and (nav.nav>0).all()
                    np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*1e8);np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
                    audit=audit_ledger(trades,nav,orders,panel,missing)
                    m.update(kind=kind,loss=loss,policy=policy,scenario=scenario,annualized_return=float(nav.nav.iloc[-1]**(252/len(nav))-1),audit=audit)
                    nav.to_parquet(OUT/(stem+'_nav.parquet'),index=False);trades.to_parquet(OUT/(stem+'_orders.parquet'),index=False);save(dest,m);results.append(m)
                    save(OUT/'summary.json',dict(status='running',results=results));print(f'{stem}: net={m["net_return"]:.2%}',flush=True)
            del orders;gc.collect()
    save(OUT/'summary.json',dict(status='completed',results=results))
    lines=['# 50万样本连续交易对照','', '2017–2023跨年连续账户，初始1亿元，每折模型均用50万行训练。同一市场、资格与成本；compact同时移除CM组。全部36种场景完整报告。','', '| 损失 | 退出规则 | 成本 | 完整全期收益 | 精简全期收益 | 完整最大回撤 | 精简最大回撤 |','|---|---|---|---:|---:|---:|---:|']
    lookup={(r['kind'],r['loss'],r['policy'],r['scenario']):r for r in results}
    for loss in ['huber','mae']:
        for policy in ['fixed5','keep50','keep100']:
            for scenario in ['gross','cost20','cost50']:
                a=lookup['full',loss,policy,scenario];b=lookup['compact',loss,policy,scenario]
                lines.append(f'| {loss} | {policy} | {scenario} | {a["net_return"]:.2%} | {b["net_return"]:.2%} | {a["max_drawdown"]:.2%} | {b["max_drawdown"]:.2%} |')
    lines+=['','cost20/cost50为20/50bp往返基础费加原冲击代理，gross零成本零冲击。固定五日/前50保留/前100保留均为既有规则。每组现金及每年末持仓独立核算；年末不清仓、末期持仓按2023末价格估值，未假设未来退出。', '', '这属于已查看历史上的开发模拟。幸存者偏差、日线撮合、缺失涨跌停、无历史ST及整手约束等局限仍在。预测改善不保证成本后改善，未据此自动选择最终模型。']
    (OUT/'EXPANDED_BACKTEST_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(OUT/'verification.json',dict(cases=36,cash_and_annual_positions_verified=True,matched_signal_keys=True));print('ALL 36 COMPLETE',flush=True)
if __name__=='__main__':main()
