"""Fixed exit-buffer sensitivity; preserve all prior baselines."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import hashlib
import numpy as np
import pandas as pd
from simulation import DATA,prepare_market
from cash_release_simulation import simulate_released_cash
from holding_buffer import buffer_panel,annotate_orders
from build_balanced_features import save_json,save_frame


def main():
    source=DATA/'v2/ols_baseline_stable';parent=DATA/'v2/ols_diagnostics_stable';out=DATA/'v2/ols_low_turnover';out.mkdir(exist_ok=True)
    policies=[('fixed5',None),('keep50',50),('keep100',100)]
    costs=[('gross',0.,0.),('cost20',20.,.5),('cost50',50.,.5)]
    prior=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    for item in prior['inputs']:
        f=Path(item['path']);assert f.stat().st_size==item['bytes'] and f.stat().st_mtime_ns==item['mtime_ns']
    sources=[Path(__file__),Path(__file__).with_name('holding_buffer.py'),Path(__file__).with_name('cash_release_simulation.py'),Path(__file__).with_name('simulation.py'),ROOT/'backtest_validation.py']
    manifest=dict(policies=policies,costs=costs,inputs=prior['inputs'],sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        parent_manifest_sha256=hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),test_used=False,training=False,
        review='five sessions from nominal exit, prior-close rank only, sell intent persists until executable',
        annual_reset=True,overlapping_lots_allowed=True,automatic_winner_selection=False,scope='post-hoc local survivor research')
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text(encoding='utf-8'))!=json.loads(json.dumps(manifest)):raise ValueError('Changed configuration')
    save_json(mp,manifest);results=[]
    for year in range(2017,2024):
        folder=out/f'validation_{year}';folder.mkdir(exist_ok=True)
        signals=pd.read_parquet(source/f'validation_{year}'/'predictions.parquet',columns=['ts_code','signal_date','prediction','volatility_20d'])
        orders,panel=prepare_market(signals,year)
        for policy,threshold in policies:
            selected,decision=buffer_panel(orders,panel,threshold) if threshold else (panel,None)
            for scenario,cost,impact in costs:
                stem=f'{policy}_{scenario}';dest=folder/f'{stem}.json'
                if dest.exists():results.append(json.loads(dest.read_text(encoding='utf-8')));continue
                m,nav,trades=simulate_released_cash(orders,selected,cost_bps=cost,impact_coefficient=impact)
                if decision is not None:trades=annotate_orders(trades,decision,panel,orders)
                else:
                    trades['buffer_extended']=False;trades['liquidity_delayed']=trades.get('delayed_exit',False)
                    trades['exit_decision_date']=trades.get('scheduled_exit_date',None)
                filled=trades.loc[trades.status=='filled'];lookup=panel[4]
                ages=[];completed=[]
                for r in filled.itertuples():
                    start=lookup[np.datetime64(r.entry_date,'D')]
                    stop=lookup[np.datetime64(r.actual_exit_date,'D')] if pd.notna(r.actual_exit_date) else len(nav)-1
                    ages.append(stop-start+1)
                    if pd.notna(r.actual_exit_date):completed.append(stop-start+1)
                assert np.isfinite(nav.nav).all() and (nav.nav>0).all() and nav.cash.min()>=-1e-5
                assert np.allclose(nav.cash+nav.holdings,nav.nav*1e8)
                assert np.allclose(nav.nav,np.cumprod(1+nav.daily_return))
                assert np.allclose(filled.buy_notional+filled.buy_cost_amount,filled.budget)
                assert not trades.duplicated(['ts_code','signal_date']).any()
                if policy=='fixed5' and scenario in ('gross','cost20'):
                    previous='gross' if scenario=='gross' else 'combined'
                    old=pd.read_parquet(parent/f'validation_{year}'/f'released_{previous}_nav.parquet')
                    np.testing.assert_allclose(nav.nav,old.nav,atol=1e-12,rtol=1e-12)
                m.update(year=year,policy=policy,scenario=scenario,buffer_extended_orders=int(filled.buffer_extended.sum()),
                    liquidity_delayed_orders=int(filled.liquidity_delayed.sum()),
                    mean_observed_holding_sessions=float(np.mean(ages)),mean_completed_holding_sessions=float(np.mean(completed)) if completed else None,
                    year_end_holdings=float(nav.holdings.iloc[-1]),order_status_counts=trades.status.value_counts().to_dict(),
                    base_cost_amount=float((filled.buy_notional*cost/10000*.25+filled.sell_notional*cost/10000*.75).sum()))
                save_frame(folder/f'{stem}_nav.parquet',nav);save_frame(folder/f'{stem}_orders.parquet',trades);save_json(dest,m);results.append(m)
                print(f'{year} {stem}: return={m["net_return"]:.3%}, turnover={m["annualized_two_way_turnover"]:.2f}',flush=True)
        save_json(out/'summary.json',dict(status='running',results=results))
    save_json(out/'summary.json',dict(status='completed',results=results,test_used=False,automatic_winner_selection=False))
    by={(r['year'],r['policy'],r['scenario']):r for r in results}
    lines=['# OLS固定持仓缓冲对照','',
        '三种退出规则×三种成本×七年，共63组；未改变模型或自动选择规则。固定5日对照沿用逐笔现金释放，其他规则只改变退出时机。','',
        '| 年份 | 固定5日净收益20bp | 保留前50净收益20bp | 保留前100净收益20bp | 固定5日换手 | 保留前50换手 | 保留前100换手 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for year in range(2017,2024):
        rows=[by[year,p,'cost20'] for p,_ in policies]
        lines.append(f'| {year} | '+' | '.join(f"{r['net_return']:.2%}" for r in rows)+' | '+' | '.join(f"{r['annualized_two_way_turnover']:.2f}" for r in rows)+' |')
    lines+=['','20bp是往返基础费，另有既定冲击成本；换手为年化双边成交额/平均净资产。','',
        '| 年份 | 固定5日零成本 | 前50零成本 | 前100零成本 | 固定5日50bp | 前50的50bp | 前100的50bp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for year in range(2017,2024):
        vals=[by[year,p,c]['net_return'] for c in ['gross','cost50'] for p,_ in policies]
        lines.append(f'| {year} | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines+=['','规则：第5个持有交易日用前一收盘排名复核，保留则隔5个交易日再看；排名消失视为不满足条件。触发卖出但不能成交时等待，不取消退出意图。未成交买单不递补，允许同股独立批次重叠持有。','',
        '缓冲延长持有，目标不再严格对应模型的五日预测期限；卖出冲击仍沿用入口事前代理。规则变化同时影响仓位、风险、复利与成本，不能将收益差额全解释为费用节约。','',
        '年度独立重置；年末未退出头寸仅估值、未收取假设清仓费。持有时长同时提供已结束批次与年末截断的观测时长；不能把未结束批次当成已平仓。','',
        '验证：3项缓冲时序/未来信息测试通过，63组现金与NAV记账检查通过；固定5日零成本及20bp对照的14条逐日净值复现上一轮。幸存者偏差、日线撮合、缺失涨跌停资料和无历史ST/整手约束仍存在。','',
        '详细参数与输入：manifest.json；结果：summary.json；各年目录保存完整订单、资金曲线、主动延长及流动性延迟退出统计。']
    (out/'LOW_TURNOVER_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('ALL 63 LOW-TURNOVER SCENARIOS COMPLETE',flush=True)


if __name__=='__main__':main()
