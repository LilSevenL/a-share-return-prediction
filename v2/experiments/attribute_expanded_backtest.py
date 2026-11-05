"""Reconcile annual and stock PnL for all 200k/500k execution cases."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from run_expanded_backtest import signals
from continuous_market import prepare_continuous

ROOT=_REPO_ROOT;BASE=ROOT/'quant_data/processed/v2';OUT=BASE/'expanded_attribution'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    OUT.mkdir(exist_ok=True)
    sources={};inputs=[]
    for name in ['attribute_expanded_backtest.py','run_expanded_backtest.py','continuous_market.py','simulation.py']:
        p=ROOT/'v2'/name;sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in [BASE/'compact_continuous',BASE/'expanded_continuous_500k']:
        assert read(folder/'artifact_verification.json')['cases']==36
        for path,digest in read(folder/'manifest.json')['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        for item in read(folder/'manifest.json')['inputs']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns']
    print('Preparing common valuation panel',flush=True)
    orders,panel,_=prepare_continuous(signals('full','huber'))
    dates,close,_,_,lookup=panel
    entries=orders[['ts_code','signal_date','entry_adj_open','code_idx']].copy()
    del orders
    annual=[];totals=[]
    for size,folder in [(200000,BASE/'compact_continuous'),(500000,BASE/'expanded_continuous_500k')]:
        summary=read(folder/'summary.json');assert summary['status']=='completed'
        for r in summary['results']:
            stem='_'.join(r[k] for k in ['kind','loss','policy','scenario'])
            meta={k:r[k] for k in ['kind','loss','policy','scenario']};meta['training_rows']=size
            paths=[folder/f'{stem}_nav.parquet',folder/f'{stem}_orders.parquet',folder/f'{stem}.json']
            inputs.extend(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths)
            nav=pd.read_parquet(paths[0]);t=pd.read_parquet(paths[1]);t=t[t.status=='filled'].copy()
            t['signal_date']=pd.to_datetime(t.signal_date)
            t=t.merge(entries,on=['ts_code','signal_date'],validate='one_to_one')
            assert len(t)==r['filled_orders']
            entry=pd.to_datetime(t.entry_date);exitdate=pd.to_datetime(t.actual_exit_date)
            ei=np.array([lookup[np.datetime64(d,'D')] for d in entry]);xi=np.full(len(t),len(dates),int)
            closed=exitdate.notna().to_numpy();xi[closed]=[lookup[np.datetime64(d,'D')] for d in exitdate[closed]]
            shares=t.buy_notional.to_numpy()/t.entry_adj_open.to_numpy();codes=t.code_idx.to_numpy(int)
            buy=t.buy_notional.to_numpy();sell=t.sell_notional.to_numpy()
            bc=t.buy_cost_amount.to_numpy();sc=t.sell_cost_amount.to_numpy()
            basebuy=buy*r['cost_bps']/10000*.25;basesell=sell*r['cost_bps']/10000*.75
            assert np.min(bc-basebuy)>-1e-7 and np.min(sc-basesell)>-1e-7
            def marks(day):
                v=np.zeros(len(t));active=(ei<=day)&(xi>day)
                if day>=0:v[active]=shares[active]*close[codes[active],day]
                return v
            by_stock=[];previous=-1;start_nav=1.
            for year in range(2017,2024):
                ix=np.flatnonzero(pd.DatetimeIndex(dates).year==year);last=int(ix[-1])
                entered=(ei>previous)&(ei<=last);exited=(xi>previous)&(xi<=last)
                market=sell*exited-buy*entered+marks(last)-marks(previous)
                costs=bc*entered+sc*exited;basefees=basebuy*entered+basesell*exited
                pnl=market-costs;expected=(nav.nav.iloc[last]-start_nav)*1e8
                np.testing.assert_allclose(pnl.sum(),expected,rtol=1e-8,atol=.1)
                avg_assets=float(nav.nav.iloc[ix].mean()*1e8)
                turnover=float((buy[entered].sum()+sell[exited].sum())/avg_assets*252/len(ix))
                audit=next(a for a in r['audit']['annual_boundaries'] if a['year']==year)
                ret=float(pnl.sum()/(start_nav*1e8));np.testing.assert_allclose(ret,audit['net_return'],atol=1e-9)
                annual.append(dict(**meta,year=year,net_return=ret,net_pnl=float(pnl.sum()),market_pnl=float(market.sum()),base_cost=float(basefees.sum()),impact_cost=float((costs-basefees).sum()),total_cost=float(costs.sum()),annualized_two_way_turnover=turnover,mean_cash_fraction=float((nav.cash.iloc[ix]/(nav.nav.iloc[ix]*1e8)).mean()),start_assets=start_nav*1e8))
                frame=pd.DataFrame(dict(ts_code=t.ts_code,net_pnl=pnl,market_pnl=market,base_cost=basefees,impact_cost=costs-basefees)).groupby('ts_code',as_index=False).sum()
                frame['year']=year;by_stock.append(frame)
                previous=last;start_nav=float(nav.nav.iloc[last])
            stock=pd.concat(by_stock,ignore_index=True)
            stock.to_parquet(OUT/f'{size}_{stem}_annual_stocks.parquet',index=False)
            whole=stock.groupby('ts_code',as_index=False).sum(numeric_only=True).drop(columns='year').sort_values('net_pnl',ascending=False)
            whole.to_parquet(OUT/f'{size}_{stem}_stocks.parquet',index=False)
            positive=whole.loc[whole.net_pnl>0,'net_pnl'];total=float(whole.net_pnl.sum());top10=float(positive.head(10).sum())
            np.testing.assert_allclose(total,r['net_return']*1e8,rtol=1e-8,atol=.1)
            np.testing.assert_allclose(whole.base_cost.sum()+whole.impact_cost.sum(),r['total_cost_amount'],rtol=1e-9,atol=.01)
            np.testing.assert_allclose((buy.sum()+sell.sum())/(nav.nav.mean()*1e8)*252/len(nav),r['annualized_two_way_turnover'],rtol=1e-9)
            totals.append(dict(**meta,net_return=r['net_return'],net_pnl=total,base_cost=float(whole.base_cost.sum()),impact_cost=float(whole.impact_cost.sum()),annualized_two_way_turnover=r['annualized_two_way_turnover'],positive_stock_count=len(positive),traded_stock_count=len(whole),top10_positive_share=top10/float(positive.sum()) if len(positive) else None,top10_net_pnl=top10,pnl_excluding_top10=total-top10))
            print(f'{size} {stem}: reconciled',flush=True)
    a=pd.DataFrame(annual);a.to_parquet(OUT/'annual_attribution.parquet',index=False)
    pd.DataFrame(totals).to_parquet(OUT/'total_attribution.parquet',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(cases=72,annual_cases=504,results=totals),indent=2),encoding='utf-8')
    lines=['# 年度、成本及股票贡献归因','','覆盖20万与50万样本的全部72场景、504个年度账户。逐年净损益=卖出金额−买入金额＋期末持仓−期初持仓−实际费用；包含未退出持仓估值，与保存净值及年度收益逐一核对。','','## 精简模型：20bp基础费加冲击','','| 损失 | 退出规则 | 扩样本年度改善数 | 50万年化双边换手(倍) | 基础费(万元) | 冲击(万元) | 前10占正贡献 | 剔除前10后的静态损益(万元) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in totals:
        if r['training_rows']!=500000 or r['kind']!='compact' or r['scenario']!='cost20':continue
        sub=a[(a.kind=='compact')&(a.loss==r['loss'])&(a.policy==r['policy'])&(a.scenario=='cost20')].pivot(index='year',columns='training_rows',values='net_return')
        wins=int((sub[500000]>sub[200000]).sum())
        lines.append(f'| {r["loss"]} | {r["policy"]} | {wins}/7 | {r["annualized_two_way_turnover"]:.2f} | {r["base_cost"]/1e4:.2f} | {r["impact_cost"]/1e4:.2f} | {r["top10_positive_share"]:.2%} | {r["pnl_excluding_top10"]/1e4:.2f} |')
        lines+=[]
    lines+=['','## 逐年对照（精简、20bp加冲击）','','| 损失 | 规则 | 年份 | 20万年度收益 | 50万年度收益 | 差值(百分点) |','|---|---|---|---:|---:|---:|']
    sub=a[(a.kind=='compact')&(a.scenario=='cost20')].pivot(index=['loss','policy','year'],columns='training_rows',values='net_return')
    for (loss,policy,year),r in sub.iterrows():lines.append(f'| {loss} | {policy} | {year} | {r[200000]:.2%} | {r[500000]:.2%} | {100*(r[500000]-r[200000]):+.2f} |')
    lines+=['','费用是各自真实模拟路径的累计金额，规模和资金复用会随情景改变；不能把零成本与含成本账户终值差全部解释成费用。双边换手为买卖金额之和除以期间平均净资产再年化。前10以事后累计净贡献排名，占比的分母为所有正贡献股票之和；静态剔除仅为集中度诊断，没有补仓或重跑策略，不能当作可交易策略收益。年度收益不可直接相加，金额贡献可以相加。','','仍为已查看历史开发结果，存在幸存者偏差及既有成交假设，未据此冻结最终模型。']
    (OUT/'ATTRIBUTION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(dict(sources=sources,inputs=inputs),indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(cases=72,annual_reconciliations=504,stock_pnl_matches_nav=True,costs_match_saved_ledger=True,turnover_matches_saved_metrics=True),indent=2),encoding='utf-8')
    print('ALL 72 ATTRIBUTIONS VERIFIED',flush=True)
if __name__=='__main__':main()
