"""Verify persisted ledgers and write complete execution ablation report."""
from run_execution_hardening import OUT,PRIOR,ROOT,read,save,sha
from lot_execution_simulation import buy_quantity
import numpy as np,pandas as pd
from pathlib import Path

def main():
    summary=read(OUT/'summary.json');manifest=read(OUT/'manifest.json')
    assert summary['status']=='completed' and len(summary['results'])==18
    for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest
    for item in manifest['inputs']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns']
    seen=set();baselines=0;rounding=0
    for r in summary['results']:
        k=(r['loss'],r['variant'],r['policy']);assert k not in seen;seen.add(k)
        nav=pd.read_parquet(OUT/f'{r["stem"]}_nav.parquet');t=pd.read_parquet(OUT/f'{r["stem"]}_orders.parquet');f=t[t.status=='filled']
        assert r['audit']['status']=='passed' and len(f)==r['filled_orders']
        assert (nav.nav>0).all() and np.isfinite(nav.nav).all() and nav.cash.min()>-1e-5
        np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
        np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*1e8)
        np.testing.assert_allclose(nav.nav.iloc[-1]-1,r['net_return'])
        np.testing.assert_allclose(np.min(nav.nav/np.maximum.accumulate(np.r_[1.,nav.nav])[1:]-1),r['max_drawdown'])
        np.testing.assert_allclose((f.buy_cost_amount+f.sell_cost_amount).sum(),r['total_cost_amount'])
        np.testing.assert_allclose(np.prod([1+a['net_return'] for a in r['audit']['annual_boundaries']]),nav.nav.iloc[-1])
        if r['variant']=='baseline':
            old=pd.read_parquet(PRIOR/f'compact_{r["loss"]}_{r["policy"]}_cost20_nav.parquet')
            old['date']=old.date.astype('datetime64[ns]');nav['date']=nav.date.astype('datetime64[ns]');pd.testing.assert_frame_equal(old,nav,check_exact=True);baselines+=1
        else:assert r['audit']['filled_entry_missing_limits']==r['audit']['sold_exit_missing_limits']==0
        if r['variant']=='strict_limits_lots':
            assert all(buy_quantity(row.ts_code,row.buy_raw_quantity,row.entry_date)==row.buy_raw_quantity for row in f.itertuples())
            np.testing.assert_allclose(f.buy_raw_quantity*f.entry_raw_open,f.buy_notional)
            np.testing.assert_allclose(f.budget,f.buy_notional+f.buy_cost_amount)
            assert (f.budget<=f.allocated_budget+1e-7).all();rounding+=1
    assert seen=={(l,v,p) for l in ['huber','mae'] for v in manifest['variants'] for p in ['fixed5','keep50','keep100']} and baselines==rounding==6
    lookup={(r['loss'],r['variant'],r['policy']):r for r in summary['results']}
    lines=['# 成交假设收紧结果','','2017–2023全期累计收益（非年化），1亿元，20bp基础费加既有冲击；50万样本精简模型。三类退出规则、两种损失、三种执行口径共18场景。','','| 损失 | 退出规则 | 原规则 | 严格涨跌停 | 严格涨跌停+买入数量 | 最终最大回撤 | 数量不足拒单 |','|---|---|---:|---:|---:|---:|---:|']
    for l in ['huber','mae']:
        for p in ['fixed5','keep50','keep100']:
            a,b,c=[lookup[l,v,p] for v in manifest['variants']]
            lines.append(f'| {l} | {p} | {a["net_return"]:.2%} | {b["net_return"]:.2%} | {c["net_return"]:.2%} | {c["max_drawdown"]:.2%} | {c["quantity_rejections"]} |')
    lines+=['','18场景账本通过核验；6个原规则净值精确复现；12个严格场景未在缺失限价的日期成交；6个数量约束场景买入股数、实际资金消耗和剩余现金检查通过。严格场景在运行时还逐笔验证买入开盘价低于涨停价、卖出收盘价高于跌停价。','','严格规则同时处理缺失上下限和成交价格触及涨跌停两件事，不能把其全部差值归因于缺失限价。缺失限价可能是合法不设限日期，拒绝成交是保守假设；原信号排序不变，不替补。买入数量使用未复权价格，之后仍用复权份额估值；公司行动及卖出零股尚未完整模拟，也未实施单笔申报上限/拆单、最小佣金、历史ST和价格笼子。详见 v2/EXECUTION_HARDENING_PLAN.md 的交易所规则依据。收益可能因持仓路径改变而上升，严格处理不保证收益单调下降。','','本轮为既有历史开发数据上的执行敏感性比较，不是新的独立测试，也不意味着已经可实盘。']
    (OUT/'EXECUTION_HARDENING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'verification.json',dict(cases=18,baseline_exact_matches=6,strict_missing_limit_checks=12,raw_buy_quantity_checks=6,saved_accounting_checks=True))
    print('\n'.join(lines[4:13]))
if __name__=='__main__':main()
