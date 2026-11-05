"""Reconcile annual and stock P&L differences from the twelve saved ledgers."""
import duckdb
import numpy as np
import pandas as pd
from run_expanded_losses import DATA, read, save
from simulation import source_files

SOURCE = DATA / "v2/elastic_compact_gate_seeds"
OUT = DATA / "v2/elastic_compact_gate_attribution"
AUM = 1e8
YEARS = range(2017, 2024)


def prices():
    con = duckdb.connect()
    for table in ["daily", "adj_factor"]:
        con.read_parquet(source_files(DATA/"parquet"/table, "year", YEARS),
                         hive_partitioning=False).create_view(table)
    marks = con.execute("""SELECT ts_code, year(trade_date) yr,
        arg_max(CAST(close * adj_factor AS FLOAT), trade_date) price
        FROM daily JOIN adj_factor USING(ts_code,trade_date)
        GROUP BY ts_code,yr""").df().pivot(index="ts_code",columns="yr",values="price").ffill(axis=1)
    con.read_parquet(source_files(DATA/"targets_5d","signal_year",YEARS),
                     hive_partitioning=False).create_view("targets")
    return con, marks


def ledger(con, marks, prefix, metrics):
    orders = pd.read_parquet(SOURCE/f"{prefix}_orders.parquet")
    trades = orders.loc[orders.status=="filled"].copy()
    assert len(trades)==metrics["filled_orders"]
    for name in ["signal_date","entry_date","actual_exit_date"]:
        trades[name] = pd.to_datetime(trades[name])
    con.register("filled", trades)
    trades = con.execute("""SELECT f.*,t.entry_adj_open FROM filled f
        LEFT JOIN targets t USING(ts_code,signal_date)""").df()
    assert len(trades)==metrics["filled_orders"] and (trades.entry_adj_open>0).all()
    nav = pd.read_parquet(SOURCE/f"{prefix}_nav.parquet")
    previous_value = pd.Series(dtype=float)
    start = AUM
    stocks, annual = [], []
    for year in YEARS:
        segment = nav.loc[nav.date.dt.year==year]
        last = segment.date.iloc[-1]
        active = trades.loc[(trades.entry_date<=last)&
                            (trades.actual_exit_date.isna()|(trades.actual_exit_date>last))].copy()
        active["value"] = (active.buy_notional/active.entry_adj_open)*active.ts_code.map(marks[year])
        assert active.value.notna().all()
        value = active.groupby("ts_code").value.sum()
        np.testing.assert_allclose(value.sum(),segment.holdings.iloc[-1],rtol=1e-8,atol=.05)
        buys = trades.loc[trades.entry_date.dt.year==year].groupby("ts_code")
        sells = trades.loc[trades.actual_exit_date.dt.year==year].groupby("ts_code")
        book = pd.concat(dict(end=value,start=previous_value,buy=buys.buy_notional.sum(),
            sell=sells.sell_notional.sum(),buy_cost=buys.buy_cost_amount.sum(),
            sell_cost=sells.sell_cost_amount.sum()),axis=1).fillna(0)
        book["gross"] = book.end-book.start+book.sell-book.buy
        book["cost"] = book.buy_cost+book.sell_cost
        book["net"] = book.gross-book.cost
        end = segment.nav.iloc[-1]*AUM
        np.testing.assert_allclose(book.net.sum(),end-start,rtol=0,atol=.1)
        book.index.name = "ts_code"
        stocks.append(book.reset_index().assign(year=year))
        notional = book.buy.sum()+book.sell.sum()
        fixed_cost = book.buy.sum()*.0005+book.sell.sum()*.0015
        annual.append(dict(year=year,net_return=end/start-1,
            gross_return=book.gross.sum()/start,cost_rate=book.cost.sum()/start,
            fixed_rate=fixed_cost/start,impact_rate=(book.cost.sum()-fixed_cost)/start,
            turnover=notional/(AUM*segment.nav.mean()),gross=book.gross.sum(),
            cost=book.cost.sum(),net=book.net.sum(),avg_nav=segment.nav.mean()))
        start,previous_value=end,value
    annual=pd.DataFrame(annual).set_index("year")
    np.testing.assert_allclose(annual.cost.sum(),metrics["total_cost_amount"])
    np.testing.assert_allclose(annual.net.sum()/AUM,metrics["net_return"],atol=1e-8)
    return annual,pd.concat(stocks)


def main():
    OUT.mkdir(exist_ok=True)
    source=read(SOURCE/"summary.json")
    con,marks=prices()
    annual_rows,stock_rows,summaries=[],[],[]
    for seed in [20261025,20261026,20261027]:
        for impact in [.5,1.]:
            annual,stocks={},{}
            for branch in ["baseline","features_only"]:
                group=f"s{seed}_{branch}"
                annual[branch],stocks[branch]=ledger(con,marks,f"{group}_{impact:g}",
                    source["results"][group][str(impact)])
            delta=annual["features_only"]-annual["baseline"]
            np.testing.assert_allclose(delta.net_return,delta.gross_return-delta.cost_rate,atol=1e-8)
            annual_rows.append(pd.concat([a.add_prefix(k+"_") for k,a in
                {**annual,"delta":delta}.items()],axis=1).reset_index().assign(seed=seed,impact=impact))
            def stock_totals(branch):
                return stocks[branch].groupby("ts_code")[["gross","cost","net"]].sum()
            base,filtered=stock_totals("baseline"),stock_totals("features_only")
            by_stock=filtered.subtract(base,fill_value=0).sort_values("net",ascending=False)
            total=by_stock.net.sum()
            np.testing.assert_allclose(total,delta.net.sum(),atol=.1)
            stock_rows.append(by_stock.reset_index().assign(seed=seed,impact=impact))
            top5=by_stock.net.head(5).sum()
            positive=by_stock.net.clip(lower=0).sum()
            summaries.append(dict(seed=seed,impact=impact,net_pp=100*total/AUM,
                gross_pp=100*delta.gross.sum()/AUM,cost_pp=100*delta.cost.sum()/AUM,
                annual_wins=int((delta.net_return>0).sum()),top_year=int(delta.net.idxmax()),
                top_year_fraction=float(delta.net.max()/total),top5_net_fraction=top5/total,
                top5_positive_fraction=top5/positive,rest_pp=100*(total-top5)/AUM,
                best_stock=by_stock.index[0],positive_stocks=int((by_stock.net>0).sum()),stocks=len(by_stock)))
    con.close()
    years=pd.concat(annual_rows)
    years.to_parquet(OUT/"annual_pairs.parquet",index=False)
    pd.concat(stock_rows).to_parquet(OUT/"stock_differences.parquet",index=False)
    save(OUT/"summary.json",dict(comparisons=summaries))
    lines=["# 333 因子 Elastic Net 收益归因","",
        "2017–2023 年，三个训练种子，两档冲击。逐笔按实际买卖日期记入成交金额与费用，以年末最近复权收盘价计入跨年及期末未卖持仓。12 本账每年均与保存净值核对，金额误差小于 0.1 元。", "",
        "下表以初始 1 亿元归一化。净收益差 = 同一批成交的毛损益差 − 实际成本差。正的成本差代表筛选组多支付费用。毛损益包含持仓路径和资金复利，未模拟免成本策略。", "",
        "| 种子 | 冲击 | 毛损益差 pp | 成本差 pp | 净收益差 pp | 年收益率胜出 | 最大贡献年（占净差） | 前5股占净差 | 扣除前5股后净差 pp |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['seed']} | {r['impact']} | {r['gross_pp']:.2f} | {r['cost_pp']:.2f} | {r['net_pp']:.2f} | {r['annual_wins']}/7 | {r['top_year']} ({r['top_year_fraction']:.1%}) | {r['top5_net_fraction']:.1%} | {r['rest_pp']:.2f} |")
    lines+=["","逐年对照以各组年初净资产计算收益率与成本率，双边换手为当年买卖金额合计除以当年平均净资产。年度收益率差不能直接相加。股票贡献按七年人民币净损益作差，前五股扣除值是账面归因，未模拟重新分配资金。","",
        "| 种子 | 冲击 | 年 | 无筛选净收益 | 筛选净收益 | 毛收益差 pp | 固定费差 pp | 冲击费差 pp | 无筛选双边换手 | 筛选双边换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in years.itertuples():
        lines.append(f"| {r.seed} | {r.impact} | {r.year} | {r.baseline_net_return:.2%} | {r.features_only_net_return:.2%} | {100*r.delta_gross_return:.2f} | {100*r.delta_fixed_rate:.2f} | {100*r.delta_impact_rate:.2f} | {r.baseline_turnover:.2f} | {r.features_only_turnover:.2f} |")
    lines+=["","双倍冲击下贡献差最大的五只股票，括号内为相对初始资金的百分点贡献。"]
    for data in stock_rows:
        if data.impact.iloc[0]==1:
            lines.append("\n"+str(data.seed.iloc[0])+"  "+"、".join(f"{r.ts_code}（{100*r.net/AUM:+.2f}）" for r in data.head().itertuples()))
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
