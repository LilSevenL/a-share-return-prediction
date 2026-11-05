"""Independent saved-artifact verification and complete stress grid report."""
from run_capacity_stress import OUT,PRIOR,ROOT,read,save,sha
from pathlib import Path
import numpy as np,pandas as pd

def main():
    summary=read(OUT/'summary.json');manifest=read(OUT/'manifest.json')
    assert summary['status']=='completed' and len(summary['results'])==36
    for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest
    for item in manifest['inputs']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns']
    seen=set();baseline_checks=0
    for r in summary['results']:
        key=(r['loss'],r['policy'],r['aum'],r['impact_coefficient']);assert key not in seen;seen.add(key)
        nav=pd.read_parquet(OUT/f'{r["stem"]}_nav.parquet');t=pd.read_parquet(OUT/f'{r["stem"]}_orders.parquet');f=t[t.status=='filled']
        assert r['audit']['status']=='passed' and (nav.nav>0).all() and np.isfinite(nav.nav).all()
        np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*r['aum'])
        np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
        np.testing.assert_allclose(nav.nav.iloc[-1]-1,r['net_return'])
        np.testing.assert_allclose(np.min(nav.nav/np.maximum.accumulate(np.r_[1.,nav.nav])[1:]-1),r['max_drawdown'])
        np.testing.assert_allclose((f.buy_cost_amount+f.sell_cost_amount).sum(),r['total_cost_amount'])
        np.testing.assert_allclose((f.buy_notional*.0005+f.sell_notional*.0015).sum(),r['base_cost_amount'])
        assert (f.participation<=.05+1e-12).all() and r['capacity_rejected']==int((t.status=='capacity_rejected').sum())
        np.testing.assert_allclose(np.prod([1+a['net_return'] for a in r['audit']['annual_boundaries']]),nav.nav.iloc[-1])
        if r['aum']==100000000 and r['impact_coefficient']==.5:
            old=pd.read_parquet(PRIOR/f'compact_{r["loss"]}_{r["policy"]}_cost20_nav.parquet')
            old['date']=old.date.astype('datetime64[ns]');nav['date']=nav.date.astype('datetime64[ns]')
            pd.testing.assert_frame_equal(nav,old,check_exact=True);baseline_checks+=1
    expected={(l,p,a,i) for l in manifest['losses'] for p in manifest['policies'] for a in manifest['aum'] for i in manifest['impact']};assert seen==expected and baseline_checks==6
    lines=['# 资金规模与冲击压力测试','','50万训练样本精简模型，2017–2023历史开发模拟。固定20bp往返基础费、5%参与率上限、三种既有退出规则；每侧冲击=系数×20日波动率×sqrt(信号日参与率)。系数0.5为既有假设，1.0为两倍压力，未经真实成交校准。三档资金为1000万、1亿、5亿元。全部36场景报告，不做自动选优。','','| 损失 | 退出规则 | 资金(亿元) | 冲击系数 | 全期累计收益 | 年化收益 | 最大回撤 | 容量拒单数 | 平均现金占比 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['results']:lines.append(f'| {r["loss"]} | {r["policy"]} | {r["aum"]/1e8:g} | {r["impact_coefficient"]:g} | {r["net_return"]:.2%} | {r["annualized_return"]:.2%} | {r["max_drawdown"]:.2%} | {r["capacity_rejected"]} | {r["mean_cash_fraction"]:.2%} |')
    lines+=['','6个基准场景净值与上一轮逐元素一致；36个场景现金、年末持仓、费用、年度复利及参与率上限通过核验。审计账本货币数值归一到初始1亿元以复用原核验器，原始净值和订单文件均保留实际资金金额。','','容量拒单按信号日成交额与整笔预算判断，未部分成交也未替补，因此扩大规模可能改变实际持仓和现金占比，收益不必单调。沿用日线成交、当前上市样本幸存者偏差、卖出冲击沿用入场估计等限制；本测试不能据此认定真实可管理规模或成本盈亏平衡点。50bp基础费的既有压力结果保留在上一轮报告，本轮不改变基础费。']
    (OUT/'CAPACITY_STRESS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'verification.json',dict(cases=36,baseline_nav_exact_matches=6,ledger_audit=True,source_and_input_checks=True,saved_nav_costs_and_capacity_verified=True))
    print('\n'.join(lines))
if __name__=='__main__':main()
