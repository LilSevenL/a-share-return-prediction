"""Matched isolated board accounts with prespecified capital and slot budgets."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import numpy as np
import pandas as pd
from run_elastic_compact_gate_seeds import (OUT as SOURCE, GROUPS, KEY, read, save, sha,
    prepare, buffer_panel, annotate_orders, simulate_released_cash, audit_ledger)

OUT=SOURCE.parent/"elastic_board_accounts"
BUDGETS={"Main":(.60,30),"ChiNext":(.20,10),"STAR":(.18,9),"Rest":(.02,1)}
MONEY=["budget","allocated_budget","buy_notional","sell_notional","buy_cost_amount","sell_cost_amount"]


def main():
    OUT.mkdir(exist_ok=True)
    files=[SOURCE/str(year)/"predictions.parquet" for year in range(2017,2024)]
    manifest=dict(budgets=BUDGETS,groups=GROUPS,inputs={str(p):sha(p) for p in files},
        script_sha=sha(Path(__file__)),account_rebalancing=False)
    if (OUT/"manifest.json").exists():
        assert read(OUT/"manifest.json")==__import__("json").loads(__import__("json").dumps(manifest))
    save(OUT/"manifest.json",manifest)
    frame=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    orders,_,panel,missing,evidence=prepare(frame[KEY+["volatility_20d",GROUPS[0]]].rename(columns={GROUPS[0]:"prediction"}))
    del evidence
    orders["entry_tradable"]=orders.entry_tradable.fillna(False)&orders.strict_buy
    orders=orders.drop(columns="prediction").merge(frame[KEY+GROUPS],on=KEY,validate="one_to_one")
    codes=orders.ts_code
    orders["board"]=np.select([codes.str.match(r"^68[89]"),codes.str.match(r"^30[01]"),
        codes.str.match(r"^(60|00)")],["STAR","ChiNext","Main"],"Rest")
    results=read(OUT/"progress.json") if (OUT/"progress.json").exists() else {}
    for group in GROUPS:
        for board,(weight,slots) in BUDGETS.items():
            selected_orders=orders.loc[orders.board==board].drop(columns=[g for g in GROUPS if g!=group]).rename(columns={group:"prediction"})
            selected,decision=buffer_panel(selected_orders,panel,2*slots)
            for impact in [.5,1.]:
                name=f"{group}_{board}_{impact:g}"
                if name in results:
                    continue
                metrics,nav,trades=simulate_released_cash(selected_orders,selected,top_n=slots,
                    aum=1e8*weight,cost_bps=20,impact_coefficient=impact,round_lots=True)
                trades=annotate_orders(trades,decision,panel,selected_orders)
                # Existing auditor uses a 100m base. Scale monetary fields only for its checks.
                normalized=trades.copy()
                normalized[MONEY]=normalized[MONEY]/weight
                normalized_nav=nav.copy()
                normalized_nav[["cash","holdings"]]/=weight
                audit=audit_ledger(normalized,normalized_nav,selected_orders,panel,missing)
                assert audit["filled_entry_missing_limits"]==audit["sold_exit_missing_limits"]==0
                np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*1e8*weight,atol=.1)
                nav.to_parquet(OUT/f"{name}_nav.parquet",index=False)
                trades.to_parquet(OUT/f"{name}_orders.parquet",index=False)
                results[name]=dict(**metrics,audit=audit)
                save(OUT/"progress.json",results)
                print(name,round(metrics["net_return"],4),flush=True)
    aggregate=[]
    for group in GROUPS:
        for impact in [.5,1.]:
            account_navs=[pd.read_parquet(OUT/f"{group}_{b}_{impact:g}_nav.parquet") for b in BUDGETS]
            nav=account_navs[0][["date"]].copy()
            nav["nav"]=0.
            nav["cash"]=0.
            nav["holdings"]=0.
            for (board,(weight,_)),account in zip(BUDGETS.items(),account_navs):
                pd.testing.assert_series_equal(nav.date,account.date)
                nav["nav"]+=weight*account.nav
                nav["cash"]+=account.cash
                nav["holdings"]+=account.holdings
            np.testing.assert_allclose((nav.cash+nav.holdings)/1e8,nav.nav,atol=1e-10)
            nav["daily_return"]=nav.nav/np.r_[1.,nav.nav.iloc[:-1]]-1
            peak=np.maximum.accumulate(np.r_[1.,nav.nav])[1:]
            nav.to_parquet(OUT/f"{group}_{impact:g}_combined_nav.parquet",index=False)
            aggregate.append(dict(group=group,impact=impact,net_return=float(nav.nav.iloc[-1]-1),
                max_drawdown=float((nav.nav/peak-1).min())))
    save(OUT/"summary.json",dict(accounts=results,combined=aggregate))
    lines=["# 固定初始板块预算的配对回测","",
        "三个种子各自比较无筛选与完整度筛选的 333 因子模型。初始总资金 1 亿元，主板 60%、创业板 20%、科创板 18%、其余 2%，每天分别最多买入 30、10、9、1 只，板块内保留阈值为其两倍。五个轮转资金账户、20bp、严格成交、整手约束延续既有实现。每个板块独立记账，板块之间不调拨，缺乏合格股票时留现金。","",
        "这是预先指定的诊断配置。各账户随收益独立增长，实际板块权重随盈亏漂移。与原策略相比，板块内买入与保留名额也随预算分配，因此结果检验这套相同约束下的模型表现。","",
        "|种子|冲击|无筛选累计净收益|筛选累计净收益|差 pp|筛选最大回撤|","|---|---:|---:|---:|---:|---:|"]
    combined=pd.DataFrame(aggregate).set_index(["group","impact"])
    for seed in [20261025,20261026,20261027]:
        for impact in [.5,1.]:
            b=combined.loc[(f"s{seed}_baseline",impact)]
            f=combined.loc[(f"s{seed}_features_only",impact)]
            lines.append(f"|{seed}|{impact}|{b.net_return:.2%}|{f.net_return:.2%}|{100*(f.net_return-b.net_return):+.2f}|{f.max_drawdown:.2%}|")
    lines+=["","双倍冲击下各板块账户自身的七年累计收益，未乘初始预算权重。","",
        "|种子|板块|无筛选|筛选|差 pp|","|---|---|---:|---:|---:|"]
    for seed in [20261025,20261026,20261027]:
        for board in BUDGETS:
            b=results[f"s{seed}_baseline_{board}_1"]["net_return"]
            f=results[f"s{seed}_features_only_{board}_1"]["net_return"]
            lines.append(f"|{seed}|{board}|{b:.2%}|{f:.2%}|{100*(f-b):+.2f}|")
    lines+=["","全部 48 个板块账户复核现金流、年末持仓、收益复利和成交价格限制。对账时等比例缩放金额以使用原有 1 亿元审计器，实际模拟始终使用各板块原始预算。原始与合并净值均已保存。结果覆盖已查看的 2017–2023 开发期。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
