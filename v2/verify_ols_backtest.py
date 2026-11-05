"""Reopen simulation artifacts and audit accounting and missing-limit exposure."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
import duckdb
from simulation import DATA,source_files
from build_balanced_features import save_json


def main():
    root=DATA/'v2/ols_backtest_stable';summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='completed' or len(summary['results'])!=14:raise ValueError('Incomplete simulation')
    checks=[];con=duckdb.connect()
    for r in summary['results']:
        year=r['year'];cost=r['cost_bps'];folder=root/f'validation_{year}'
        nav=pd.read_parquet(folder/f'nav_cost{cost:g}.parquet');orders=pd.read_parquet(folder/f'orders_cost{cost:g}.parquet')
        filled=orders.loc[orders.status=='filled'].copy()
        assert len(orders)==r['orders'] and len(filled)==r['filled_orders']
        assert not orders.duplicated(['ts_code','signal_date']).any()
        assert np.isclose(nav.nav.iloc[-1]-1,r['net_return'])
        assert np.allclose(nav.nav,np.cumprod(1+nav.daily_return))
        assert np.isclose(filled.buy_cost_amount.sum()+filled.sell_cost_amount.sum(),r['total_cost_amount'])
        assert np.allclose(filled.buy_notional+filled.buy_cost_amount,filled.budget)
        exit_dates=pd.to_datetime(filled.actual_exit_date);planned=pd.to_datetime(filled.scheduled_exit_date)
        assert ((exit_dates>=planned)|exit_dates.isna()).all()
        assert filled.unresolved_at_end.sum()==r['unresolved_positions']
        con.register('filled',filled)
        con.read_parquet(source_files(DATA/'parquet/stk_limit','year',[year]),hive_partitioning=False).create_view('limits',replace=True)
        missing=con.execute('''SELECT
            count(*) FILTER(WHERE e.up_limit IS NULL OR e.down_limit IS NULL),
            count(*) FILTER(WHERE f.actual_exit_date IS NOT NULL AND (x.up_limit IS NULL OR x.down_limit IS NULL))
            FROM filled f LEFT JOIN limits e ON f.ts_code=e.ts_code AND CAST(f.entry_date AS DATE)=e.trade_date
            LEFT JOIN limits x ON f.ts_code=x.ts_code AND CAST(f.actual_exit_date AS DATE)=x.trade_date''').fetchone()
        checks.append(dict(year=year,cost_bps=cost,accounting='passed',filled_entry_missing_limit=int(missing[0]),sold_exit_missing_limit=int(missing[1]),
            blocked_sleeve_days=r['blocked_sleeve_days']))
    con.close();save_json(root/'verification.json',dict(status='passed_accounting',scenarios=checks))
    p=root/'BACKTEST_REPORT.md';text=p.read_text(encoding='utf-8')
    text+='\n## 结果复核与解释\n\n'
    text+='14个情景重新读取后，订单主键、预算与成本记账、NAV复利恒等式、实际退出日期及未退出计数全部通过。\n\n'
    text+=f"实际买入订单缺失涨跌停资料共{sum(c['filled_entry_missing_limit'] for c in checks)}条、实际卖出订单缺失资料共{sum(c['sold_exit_missing_limit'] for c in checks)}条（跨成本情景重复计数）；详见verification.json。\n\n"
    text+='20bp基础成本下仅2021年净收益为正，50bp下七年均为负。不能仅凭正Rank IC认定策略可盈利。尤其2017、2018年整组资金因延迟退出分别跳过182、173个信号日；这是保守的整组锁仓规则，与逐股释放已卖出资金的组合不同。当前成交率只针对已下单记录，不包含这些被跳过的信号日。\n\n'
    text+='下一步先诊断持仓资金释放、换手与冲击成本，以及前50名与整体排序指标之间的差异；随后再做事先固定规则的对照实验。不要直接根据这次成绩调整loss或挑最优参数。\n'
    p.write_text(text,encoding='utf-8');print(json.dumps(checks))


if __name__=='__main__':main()
