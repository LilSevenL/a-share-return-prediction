"""Fixed annual daily-bar simulations of saved OLS validation signals, no fitting."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import sys
from pathlib import Path
ROOT=_REPO_ROOT
sys.path.insert(0,str(ROOT))
import json
import hashlib
import numpy as np
import pandas as pd
import duckdb
from simulation import DATA,source_files,prepare_market,simulate
from build_balanced_features import save_json,save_frame


def main():
    source=DATA/'v2/ols_baseline_stable';output=DATA/'v2/ols_backtest_stable';output.mkdir(exist_ok=True)
    summary=json.loads((source/'verified_summary.json').read_text())
    if summary['status']!='verified':raise ValueError('Unverified OLS results')
    years=list(range(2017,2024));costs=[20.,50.]
    inputs=[source/f'validation_{y}'/'predictions.parquet' for y in years]
    for name in ('daily','adj_factor','stk_limit'):
        inputs += [Path(f) for f in source_files(DATA/'parquet'/name,'year',years)]
    inputs += [Path(f) for f in source_files(DATA/'targets_5d','signal_year',years)]
    inputs.append(ROOT/'quant_data/combined/stock_basic.csv')
    files=[Path(__file__),_source_path('v2/simulation.py'),_source_path('backtest_validation.py'),_source_path('v2/build_balanced_features.py')]
    manifest=dict(years=years,cost_bps=costs,top_n=50,aum=1e8,participation_cap=.05,impact_coefficient=.5,
        costs='round-trip base 20/50 bps allocated 25% buy /75% sell; additional per-side 0.5*past20d_vol*sqrt(order_budget/signal_day_amount)',
        eligibility='signal-day amount >=50 million CNY; listed >=60 calendar days; finite saved predictions',
        sleeves=5,scope='local_survivor_research',training=False,test_used=False,
        year_boundary='independent annual capital reset; no entry with planned exit outside year; unresolved holdings marked at final close, not forced sold',
        execution='rank first; no replacement; unfilled cash; delayed sells lock entire sleeve; signal-day capacity proxy; fractional adjusted shares',
        limitations=['survivorship bias','daily bars do not prove intraday/open execution','unknown price limits use inherited permissive assumption, audited separately',
            'no historical ST exclusion','no board-lot constraint','base costs are fixed scenarios, not historically reconstructed taxes/fees',
            'adjusted-price accounting is a total-return proxy','no benchmark excess return attribution'],
        code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        inputs=[dict(path=str(f),bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in inputs])
    mp=output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Configuration or inputs changed')
    save_json(mp,manifest);results=[]
    for year in years:
        folder=output/f'validation_{year}';folder.mkdir(exist_ok=True)
        if all((folder/f'metrics_cost{c:g}.json').exists() for c in costs):
            results.extend(json.loads((folder/f'metrics_cost{c:g}.json').read_text()) for c in costs);continue
        signals=pd.read_parquet(source/f'validation_{year}'/'predictions.parquet',columns=['ts_code','signal_date','prediction','volatility_20d'])
        if signals.duplicated(['ts_code','signal_date']).any() or not np.isfinite(signals.prediction).all():raise ValueError('Bad signals')
        orders,panel=prepare_market(signals,year)
        if orders.duplicated(['ts_code','signal_date']).any():raise ValueError('Duplicate candidate orders')
        con=duckdb.connect();con.execute('SET threads=4');con.register('orders',orders)
        con.read_parquet(source_files(DATA/'parquet/stk_limit','year',[year]),hive_partitioning=False).create_view('limits')
        con.read_parquet(source_files(DATA/'parquet/daily','year',[year]),hive_partitioning=False).create_view('daily')
        missing_entry=con.execute('''SELECT count(*) FROM orders o LEFT JOIN limits l ON o.ts_code=l.ts_code AND o.entry_date=l.trade_date
            WHERE o.entry_date IS NOT NULL AND o.entry_date<make_date(?,1,1) AND (l.up_limit IS NULL OR l.down_limit IS NULL)''',[year+1]).fetchone()[0]
        missing_limits=con.execute('''SELECT count(*) FROM daily d LEFT JOIN limits l USING(ts_code,trade_date)
            WHERE l.up_limit IS NULL OR l.down_limit IS NULL''').fetchone()[0];con.close()
        audit=dict(year=year,signal_rows=len(signals),eligible_candidate_rows=len(orders),
            eligible_days=int(orders.signal_date.nunique()),market_days=len(panel[0]),
            candidate_entry_missing_limit_rows=int(missing_entry),daily_missing_limit_rows=int(missing_limits),
            ranking_does_not_filter_future_label_or_tradability=True)
        save_json(folder/'execution_input_audit.json',audit)
        print(f'{year}: {len(signals)} predictions, {len(orders)} eligible candidates, missing-limit candidates {missing_entry}',flush=True)
        for cost in costs:
            metrics,nav,trades=simulate(orders,panel,cost_bps=cost)
            if not np.isfinite(nav.nav).all() or (nav.nav<=0).any():raise ValueError('Invalid NAV')
            if not np.allclose(nav.daily_return,nav.nav.to_numpy()/np.r_[1.,nav.nav.to_numpy()[:-1]]-1):raise ValueError('NAV identity failed')
            filled=trades.loc[trades.status=='filled']
            if len(filled) and ((filled.participation>.05+1e-12).any() or not np.allclose(filled.buy_notional+filled.buy_cost_amount,filled.budget)):raise ValueError('Trade accounting failed')
            metrics.update(year=year,order_status_counts=trades.status.value_counts().to_dict(),fill_rate=metrics['filled_orders']/metrics['orders'] if metrics['orders'] else 0,
                input_audit=audit,scope=manifest['scope'])
            save_frame(folder/f'nav_cost{cost:g}.parquet',nav);save_frame(folder/f'orders_cost{cost:g}.parquet',trades)
            save_json(folder/f'metrics_cost{cost:g}.json',metrics);results.append(metrics)
            print(f'{year} cost={cost:g}: return={metrics["net_return"]:.4%}, drawdown={metrics["max_drawdown"]:.4%}, filled={metrics["filled_orders"]}',flush=True)
        save_json(output/'summary.json',dict(status='running',results=results,test_used=False))
    save_json(output/'summary.json',dict(status='completed',results=results,test_used=False,scope=manifest['scope'],annual_reset=True))
    lines=['# OLS含成本分年度回测','',
        '2017–2023年七折已保存预测；无重新训练或按结果调参。各年资金重置为1亿元，前50名、5组轮换、名义5个交易日持有。', '',
        '20/50bp为往返基础成本，另加双边事前波动率/参与率冲击成本。容量上限为订单预算占信号日成交额5%；这不是成交日真实参与率保证。未成交不递补；延迟卖出会锁住整组资金。','',
        '| 年份 | 基础成本bp | 年内净收益 | 最大回撤 | 日频Sharpe | 双边年化换手 | 成交率 | 延迟卖出订单 | 年末未退出 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['year']} | {r['cost_bps']:g} | {r['net_return']:.2%} | {r['max_drawdown']:.2%} | {r['sharpe']:.3f} | {r['annualized_two_way_turnover']:.2f} | {r['fill_rate']:.2%} | {r['delayed_exit_orders']} | {r['unresolved_positions']} |")
    lines+=['','换手为买卖双边成交额/年内平均净资产，再按252个交易日年化。Sharpe未减无风险利率；仅为历史模拟描述，不报告统计显著性。', '',
        '每年独立重置资金；未退出头寸按年末可用收盘价估值，未假设已变现。因此不把各年NAV拼接为可实现的连续资金曲线。', '',
        '限制：本地股票池存在幸存者偏差；日线无法证明开盘排队或盘中可成交；无历史ST过滤及整手约束；固定费用场景不等于逐年实际费税；没有基准超额收益归因。缺失涨跌停数据沿用已有撮合假设，按年输入审计单列，不等于确认可交易。', '',
        '审计文件：manifest.json、summary.json；validation_YYYY内保存输入覆盖、订单状态、成本明细及每日NAV。全部年份限2017–2023，未读取2024及以后市场分区。']
    (output/'BACKTEST_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ALL 14 ANNUAL COST SCENARIOS COMPLETE',flush=True)


if __name__=='__main__':main()
