"""Predeclared capital/cost counterfactuals and label ranking diagnostics."""

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
from simulation import DATA,prepare_market,simulate
from cash_release_simulation import simulate_released_cash
from build_balanced_features import save_json,save_frame


def signal_diagnostic(orders,signals):
    frame=orders[['ts_code','signal_date','prediction']].merge(signals[['ts_code','signal_date','y']],on=['ts_code','signal_date'],validate='one_to_one')
    frame=frame.sort_values(['signal_date','prediction','ts_code'],ascending=[True,False,True])
    days=[]
    for day,g in frame.groupby('signal_date',sort=True):
        row=dict(date=day)
        for name,part in [('top50',g.head(50)),('bottom50',g.tail(50)),('pool',g)]:
            y=part.y.to_numpy();valid=np.isfinite(y)
            row[name+'_selected']=len(y);row[name+'_valid']=int(valid.sum())
            row[name+'_mean_return']=float(np.expm1(y[valid]).mean()) if valid.any() else np.nan
        days.append(row)
    daily=pd.DataFrame(days);common=daily.dropna(subset=['top50_mean_return','pool_mean_return','bottom50_mean_return'])
    result=dict(days=len(common),top50_mean_5d_return=float(common.top50_mean_return.mean()),
        bottom50_mean_5d_return=float(common.bottom50_mean_return.mean()),pool_mean_5d_return=float(common.pool_mean_return.mean()),
        top50_minus_pool=float((common.top50_mean_return-common.pool_mean_return).mean()),
        top50_label_coverage=float(daily.top50_valid.sum()/daily.top50_selected.sum()),
        note='Same-date equal-day average of overlapping simple 5-day raw labels; no cost or tradability filtering; not portfolio PnL')
    return result,daily


def main():
    source=DATA/'v2/ols_baseline_stable';parent=DATA/'v2/ols_backtest_stable';output=DATA/'v2/ols_diagnostics_stable';output.mkdir(exist_ok=True)
    configs=[('gross',0.,0.),('base_only',20.,0.),('impact_only',0.,.5),('combined',20.,.5)]
    parent_manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    for item in parent_manifest['inputs']:
        f=Path(item['path'])
        if f.stat().st_size!=item['bytes'] or f.stat().st_mtime_ns!=item['mtime_ns']:raise ValueError('Parent data changed')
    files=[Path(__file__),_source_path('v2/cash_release_simulation.py'),_source_path('v2/simulation.py'),_source_path('backtest_validation.py')]
    manifest=dict(configs=configs,policies=['locked','released'],parent_manifest_sha256=hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
        inputs=parent_manifest['inputs'],sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        scope='post-hoc diagnostic on local survivor history; not new independent test',test_used=False,training=False,
        capital_rule='released uses prior-close available sleeve cash; unsold lots kept; overlapping lots allowed',automatic_winner_selection=False)
    mp=output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text(encoding='utf-8'))!=json.loads(json.dumps(manifest)):raise ValueError('Diagnostic settings changed')
    save_json(mp,manifest);results=[];signals_summary=[]
    for year in range(2017,2024):
        folder=output/f'validation_{year}';folder.mkdir(exist_ok=True)
        signals=pd.read_parquet(source/f'validation_{year}'/'predictions.parquet')
        orders,panel=prepare_market(signals.drop(columns='y'),year)
        signal,daily=signal_diagnostic(orders,signals);signal['year']=year;signals_summary.append(signal)
        save_json(folder/'signal_diagnostic.json',signal);save_frame(folder/'daily_signal_diagnostic.parquet',daily)
        for policy,fn in [('locked',simulate),('released',simulate_released_cash)]:
            for name,cost,impact in configs:
                stem=f'{policy}_{name}';dest=folder/f'{stem}.json'
                if dest.exists():results.append(json.loads(dest.read_text(encoding='utf-8')));continue
                m,nav,trades=fn(orders,panel,cost_bps=cost,impact_coefficient=impact)
                f=trades.loc[trades.status=='filled']
                base=float((f.buy_notional*cost/10000*.25+f.sell_notional*cost/10000*.75).sum())
                assert np.isfinite(nav.nav).all() and (nav.nav>0).all()
                assert np.allclose(nav.nav,np.cumprod(1+nav.daily_return))
                assert np.allclose(f.buy_notional+f.buy_cost_amount,f.budget)
                if policy=='released':assert np.allclose(nav.cash+nav.holdings,nav.nav*1e8) and nav.cash.min()>=-1e-5
                if policy=='locked' and name=='combined':
                    old=json.loads((parent/f'validation_{year}'/'metrics_cost20.json').read_text(encoding='utf-8'))
                    assert np.isclose(m['net_return'],old['net_return'],rtol=1e-12,atol=1e-12)
                    old_nav=pd.read_parquet(parent/f'validation_{year}'/'nav_cost20.parquet')
                    np.testing.assert_allclose(nav.nav,old_nav.nav,rtol=1e-12,atol=1e-12)
                m.update(year=year,policy=policy,scenario=name,base_cost_amount=base,impact_cost_amount=m['total_cost_amount']-base)
                save_frame(folder/f'{stem}_nav.parquet',nav);save_frame(folder/f'{stem}_orders.parquet',trades);save_json(dest,m);results.append(m)
                print(f'{year} {stem}: {m["net_return"]:.3%}',flush=True)
        save_json(output/'summary.json',dict(status='running',results=results,signal_diagnostics=signals_summary))
    save_json(output/'summary.json',dict(status='completed',results=results,signal_diagnostics=signals_summary,test_used=False,automatic_winner_selection=False))
    by={(r['year'],r['policy'],r['scenario']):r for r in results}
    lines=['# OLS资金与成本诊断','',
        '固定对照，无重新训练或超参数选择。每年独立重置资金；表中是年度模拟收益，不能拼接为真实连续业绩。', '',
        '| 年份 | 整组锁仓零成本 | 整组锁仓仅基础费 | 整组锁仓仅冲击 | 整组锁仓两者 | 逐笔释放零成本 | 逐笔释放仅基础费 | 逐笔释放仅冲击 | 逐笔释放两者 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for year in range(2017,2024):
        vals=[by[year,policy,name]['net_return'] for policy in ['locked','released'] for name,_,_ in configs]
        lines.append('| '+str(year)+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines+=['','基础费用固定为往返20bp；冲击系数固定为0.5。情景差额包含复利、订单资金和容量路径变化，不能简单相加解释为纯成本贡献。各JSON另有实际基础费用与冲击费用金额。', '',
        '| 年份 | 前50五日收益均值 | 全池五日收益均值 | 后50五日收益均值 | 前50减全池 | 前50标签覆盖 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in signals_summary:
        lines.append(f"| {r['year']} | {r['top50_mean_5d_return']:.3%} | {r['pool_mean_5d_return']:.3%} | {r['bottom50_mean_5d_return']:.3%} | {r['top50_minus_pool']:.3%} | {r['top50_label_coverage']:.2%} |")
    lines+=['','标签诊断在资格池先排名，缺失不递补；使用三组都有已知标签的相同日期，日等权汇总。五日收益重叠且未扣成本，不等于组合收益。全池是同一幸存者资格池，不是正式市场基准。', '',
        '新资金规则仍按5组轮换，只复用前一收盘已有现金，未退出头寸继续估值；允许独立批次重叠持有同一股票。它是资金管理敏感性实验，没有自动替换原策略。', '',
        '验证：两项资金记账测试通过；56个情景核对预算/净值恒等式；新规则现金非负且现金+持仓=净资产；旧规则含成本7年逐日净值复现前次结果。', '',
        '全部结果仍受幸存者偏差、日线撮合、缺失涨跌停资料、无历史ST和整手约束影响。本次是已知历史上的事后诊断，不能当作新的独立验证。']
    (output/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('ALL 56 DIAGNOSTIC SCENARIOS COMPLETE',flush=True)


if __name__=='__main__':main()
