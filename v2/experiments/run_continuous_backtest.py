"""Uninterrupted portfolio ledger over saved yearly OLS signals, 2017-2023 only."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import sys
from pathlib import Path
ROOT=_REPO_ROOT;sys.path.insert(0,str(ROOT))
import json
import hashlib
import numpy as np
import pandas as pd
from simulation import DATA
from continuous_market import prepare_continuous
from cash_release_simulation import simulate_released_cash
from holding_buffer import buffer_panel,annotate_orders
from build_balanced_features import save_json,save_frame


def audit_ledger(trades,nav,orders,panel,missing):
    dates,close,_,_,lookup=panel
    filled=trades.loc[trades.status=='filled'].copy()
    filled['signal_date']=pd.to_datetime(filled.signal_date)
    filled=filled.merge(orders[['ts_code','signal_date','entry_adj_open','code_idx']],on=['ts_code','signal_date'],validate='one_to_one')
    entry=pd.to_datetime(filled.entry_date);exit_date=pd.to_datetime(filled.actual_exit_date)
    entry_idx=np.array([lookup[np.datetime64(d,'D')] for d in entry])
    closed=exit_date.notna().to_numpy();exit_idx=np.full(len(filled),len(nav),int)
    exit_idx[closed]=[lookup[np.datetime64(d,'D')] for d in exit_date[closed]]
    movement=np.zeros(len(nav));np.add.at(movement,entry_idx,-filled.budget.to_numpy())
    proceeds=(filled.sell_notional-filled.sell_cost_amount).to_numpy()
    np.add.at(movement,exit_idx[closed],proceeds[closed])
    np.testing.assert_allclose(1e8+movement.cumsum(),nav.cash,rtol=1e-10,atol=1e-5)
    shares=filled.buy_notional.to_numpy()/filled.entry_adj_open.to_numpy();codes=filled.code_idx.to_numpy(dtype=int)
    boundaries=[];previous=1.
    for year in range(2017,2024):
        indexes=np.flatnonzero(pd.DatetimeIndex(dates).year==year);last=int(indexes[-1])
        active=(entry_idx<=last)&(exit_idx>last)
        value=float(np.dot(shares[active],close[codes[active],last]))
        np.testing.assert_allclose(value,nav.holdings.iloc[last],rtol=1e-7,atol=.05)
        end=float(nav.nav.iloc[last]);segment=nav.nav.iloc[indexes].to_numpy()/previous
        peak=np.maximum.accumulate(np.r_[1.,segment])[1:]
        boundaries.append(dict(year=year,net_return=end/previous-1,year_end_nav=end,
            year_end_cash=float(nav.cash.iloc[last]),year_end_holdings=value,
            open_lots=int(active.sum()),open_stocks=int(filled.loc[active,'ts_code'].nunique()),
            exit_pending_lots=int(((pd.to_datetime(filled.exit_decision_date)<=pd.Timestamp(dates[last]))&active).sum()),
            within_year_drawdown=float(np.min(segment/peak-1))))
        previous=end
    assert np.allclose(np.prod([1+b['net_return'] for b in boundaries]),nav.nav.iloc[-1])
    miss=set(zip(missing.ts_code,missing.trade_date.dt.strftime('%Y-%m-%d')))
    missing_buy=sum((c,str(d.date())) in miss for c,d in zip(filled.ts_code,entry))
    missing_sell=sum((c,str(d.date())) in miss for c,d in zip(filled.loc[closed,'ts_code'],exit_date[closed]))
    crossing=int(((entry.dt.year!=exit_date.dt.year)&closed).sum())
    audit=dict(status='passed',cash_reconstructed_from_transactions=True,annual_position_values_verified=True,
        cross_year_completed_lots=crossing,filled_entry_missing_limits=missing_buy,sold_exit_missing_limits=missing_sell,
        final_open_lots=boundaries[-1]['open_lots'],final_open_stocks=boundaries[-1]['open_stocks'],annual_boundaries=boundaries)
    return audit


def main():
    parent=DATA/'v2/ols_low_turnover';source=DATA/'v2/ols_baseline_stable';out=DATA/'v2/ols_continuous';out.mkdir(exist_ok=True)
    prior=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    for item in prior['inputs']:
        p=Path(item['path'])
        if p.stat().st_size!=item['bytes'] or p.stat().st_mtime_ns!=item['mtime_ns']:raise ValueError('Input changed')
    policies=[('fixed5',None),('keep50',50),('keep100',100)];costs=[('gross',0.,0.),('cost20',20.,.5),('cost50',50.,.5)]
    files=[Path(__file__),_source_path('v2/continuous_market.py'),_source_path('v2/cash_release_simulation.py'),_source_path('v2/holding_buffer.py'),_source_path('v2/simulation.py'),_source_path('backtest_validation.py')]
    manifest=dict(policies=policies,costs=costs,inputs=prior['inputs'],sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        period=[2017,2023],initial_capital=1e8,annual_reset=False,model_switch='use saved prediction from signal calendar year',
        sleeve_cycle='global trading index modulo five; never reset at January 1',test_used=False,training=False,
        automatic_winner_selection=False,scope='post-hoc local survivor research; inherited daily-bar assumptions')
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text(encoding='utf-8'))!=json.loads(json.dumps(manifest)):raise ValueError('Configuration changed')
    save_json(mp,manifest)
    signals=[]
    for year in range(2017,2024):
        f=pd.read_parquet(source/f'validation_{year}'/'predictions.parquet',columns=['ts_code','signal_date','prediction','volatility_20d'])
        if not (f.signal_date.dt.year==year).all():raise ValueError('Model year mismatch')
        signals.append(f)
    signals=pd.concat(signals,ignore_index=True)
    if not np.isfinite(signals.prediction).all():raise ValueError('Nonfinite predictions')
    print(f'Preparing continuous market: {len(signals)} signals',flush=True)
    orders,panel,missing=prepare_continuous(signals);del signals
    save_json(out/'input_audit.json',dict(eligible_rows=len(orders),market_days=len(panel[0]),
        first_market_day=str(panel[0][0]),last_market_day=str(panel[0][-1]),missing_limit_source_rows=len(missing)))
    results=[]
    for policy,threshold in policies:
        selected,decision=buffer_panel(orders,panel,threshold) if threshold else (panel,None)
        for scenario,cost,impact in costs:
            stem=f'{policy}_{scenario}';dest=out/f'{stem}.json'
            if dest.exists():results.append(json.loads(dest.read_text(encoding='utf-8')));continue
            print(f'Running {stem}',flush=True)
            m,nav,trades=simulate_released_cash(orders,selected,cost_bps=cost,impact_coefficient=impact)
            if decision is not None:trades=annotate_orders(trades,decision,panel,orders)
            else:
                trades['buffer_extended']=False;trades['liquidity_delayed']=trades.delayed_exit
                trades['exit_decision_date']=trades.scheduled_exit_date
            if not np.isfinite(nav.nav).all() or (nav.nav<=0).any():raise ValueError('Invalid NAV')
            np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*1e8)
            np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
            # Before the final five 2017 sessions, continuity must not change the old first-year path.
            old=pd.read_parquet(parent/'validation_2017'/f'{policy}_{scenario}_nav.parquet')
            np.testing.assert_allclose(nav.nav.iloc[:len(old)-5],old.nav.iloc[:-5],rtol=1e-10,atol=1e-10)
            audit=audit_ledger(trades,nav,orders,panel,missing)
            filled=trades.loc[trades.status=='filled'];std=nav.daily_return.std(ddof=1)
            m.update(policy=policy,scenario=scenario,annualized_return=float(nav.nav.iloc[-1]**(252/len(nav))-1),
                sharpe=float(nav.daily_return.mean()/std*np.sqrt(252)) if std>0 else None,
                final_cash=float(nav.cash.iloc[-1]),final_holdings=float(nav.holdings.iloc[-1]),
                final_nav=float(nav.nav.iloc[-1]),base_cost_amount=float((filled.buy_notional*cost/10000*.25+filled.sell_notional*cost/10000*.75).sum()),
                audit=audit,buffer_extended_lots=int(filled.buffer_extended.sum()),liquidity_delayed_lots=int(filled.liquidity_delayed.sum()))
            save_frame(out/f'{stem}_nav.parquet',nav);save_frame(out/f'{stem}_orders.parquet',trades);save_json(dest,m);results.append(m)
            save_json(out/'summary.json',dict(status='running',results=results))
            print(f'{stem}: total={m["net_return"]:.3%}, annual={m["annualized_return"]:.3%}, maxDD={m["max_drawdown"]:.3%}',flush=True)
    save_json(out/'summary.json',dict(status='completed',results=results,annual_reset=False,test_used=False,automatic_winner_selection=False))
    lines=['# OLS跨年连续持仓回测','',
        '2017–2023年连续账本：初始1亿元仅设置一次，现金、旧仓、退出指令与五组轮换跨年继承；信号按已完成逐年模型切换。没有重新训练或自动选规则。','',
        '| 退出规则 | 成本场景 | 全期净收益 | 年化收益 | 最大回撤 | 日频Sharpe | 双边年化换手 | 末期持仓万元 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['policy']} | {r['scenario']} | {r['net_return']:.2%} | {r['annualized_return']:.2%} | {r['max_drawdown']:.2%} | {r['sharpe']:.3f} | {r['annualized_two_way_turnover']:.2f} | {r['final_holdings']/1e4:.1f} |")
    lines+=['','gross=零成本零冲击；cost20/cost50=20/50bp往返基础费加既定冲击。fixed5=固定5日；keep50/keep100=仍在资格池前50/100名则延长，每5日用昨日收盘排名复核。年化按252交易日，Sharpe未减无风险利率。','',
        '| 年份 | fixed5净收益20bp | keep50净收益20bp | keep100净收益20bp | keep100年末现金万元 | keep100年末持仓万元 | keep100未退出批次 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    chosen={r['policy']:r for r in results if r['scenario']=='cost20'}
    for i,year in enumerate(range(2017,2024)):
        rows=[chosen[p]['audit']['annual_boundaries'][i] for p,_ in policies];b=rows[2]
        lines.append(f"| {year} | "+' | '.join(f"{r['net_return']:.2%}" for r in rows)+f" | {b['year_end_cash']/1e4:.1f} | {b['year_end_holdings']/1e4:.1f} | {b['open_lots']} |")
    lines+=['','年度收益以上年末实际净资产为基准；年度收益复利乘积核对全期净值。年末持仓数为独立批次，不等于不同股票数。末期持仓只是估值，未假设清仓或扣除未来卖出费用。','',
        '核对：跨年资金/未退出测试通过；九组现金均由逐笔预算支出及实际卖出净款独立重建；每年末持仓按订单份额×对应价格独立复算；2017年边界前逐日曲线复现原年度模拟；缺失涨跌停成交事件按情景单列在audit中。','',
        '限制：这是已知历史上的事后研究，仍有幸存者偏差、日线撮合、缺失涨跌停、无历史ST及整手约束；持仓缓冲改变了五日预测期限，卖出冲击仍沿用入口事前估计。末期只读取到2023年，不用2024价格假定完成退出。没有正式市场基准超额收益归因。']
    (out/'CONTINUOUS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('ALL 9 CONTINUOUS SCENARIOS COMPLETE',flush=True)


if __name__=='__main__':main()
