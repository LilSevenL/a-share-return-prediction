"""Combine strict execution and doubled impact on the six matched portfolios."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path

import numpy as np
import pandas as pd

from holding_buffer import annotate_orders, buffer_panel
from lot_execution_simulation import buy_quantity, simulate_released_cash
from run_continuous_backtest import audit_ledger
from run_expanded_backtest import DATA, ROOT, read, save, sha, signals
from strict_execution_market import prepare

PRIOR = DATA / "v2/execution_hardening_500k"
OUT = DATA / "v2/joint_execution_stress_500k"
POLICIES = {"fixed5": None, "keep50": 50, "keep100": 100}


def main():
    parent = read(PRIOR / "manifest.json")
    assert read(PRIOR / "verification.json")["cases"] == 18
    for path, digest in parent["sources"].items():
        assert sha(ROOT / path) == digest
    for item in parent["inputs"]:
        stat = Path(item["path"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"])
    OUT.mkdir(exist_ok=True)
    save(OUT / "manifest.json", dict(parent_manifest_sha256=sha(PRIOR / "manifest.json"),
         runner_sha256=sha(Path(__file__)), aum=1e8, cost_bps=20, impact_coefficient=1.,
         participation_cap=.05, round_lots=True, policies=list(POLICIES), period=[2017, 2023]))
    print("Preparing strict execution market", flush=True)
    template, _, panel, missing, evidence = prepare(signals("compact", "huber"))
    template = template.drop(columns="prediction")
    template["entry_tradable"] = template.entry_tradable.fillna(False) & template.strict_buy
    sellable = evidence.set_index(["ts_code", "trade_date"]).strict_sell
    del evidence
    rows = []
    for loss in ["huber", "mae"]:
        predictions = signals("compact", loss)[["ts_code", "signal_date", "prediction"]]
        orders = template.merge(predictions, on=["ts_code","signal_date"], validate="one_to_one")
        assert len(orders) == len(template)
        for policy, threshold in POLICIES.items():
            stem = f"{loss}_{policy}"
            selected, decision = buffer_panel(orders, panel, threshold) if threshold else (panel, None)
            metrics, nav, trades = simulate_released_cash(orders, selected, aum=1e8,
                cost_bps=20, participation_cap=.05, impact_coefficient=1., round_lots=True)
            if decision is None:
                trades["exit_decision_date"] = trades.scheduled_exit_date
            else:
                trades = annotate_orders(trades, decision, panel, orders)
            audit = audit_ledger(trades, nav, orders, panel, missing)
            assert audit["filled_entry_missing_limits"] == audit["sold_exit_missing_limits"] == 0
            filled = trades.loc[trades.status == "filled"].copy()
            filled["signal_date"] = pd.to_datetime(filled.signal_date)
            filled = filled.merge(orders[["ts_code","signal_date","strict_buy","volatility_20d"]],
                                  on=["ts_code","signal_date"], validate="one_to_one")
            assert len(filled) == metrics["filled_orders"] and filled.strict_buy.all()
            sold = filled.dropna(subset=["actual_exit_date"])
            exit_keys = pd.MultiIndex.from_arrays([sold.ts_code, pd.to_datetime(sold.actual_exit_date)])
            assert sellable.loc[exit_keys].all()
            impact = filled.volatility_20d * np.sqrt(filled.participation)
            np.testing.assert_allclose(filled.buy_cost_rate, .0005 + impact)
            np.testing.assert_allclose(filled.sell_cost_rate, .0015 + impact)
            assert all(buy_quantity(t.ts_code,t.buy_raw_quantity,t.entry_date) == t.buy_raw_quantity
                       for t in filled.itertuples())
            np.testing.assert_allclose(filled.buy_raw_quantity * filled.entry_raw_open, filled.buy_notional)
            np.testing.assert_allclose(nav.cash + nav.holdings, nav.nav * 1e8)
            np.testing.assert_allclose(nav.nav, np.cumprod(1 + nav.daily_return))
            old_stem = f"{loss}_strict_limits_lots_{policy}"
            old_nav = pd.read_parquet(PRIOR / f"{old_stem}_nav.parquet")
            np.testing.assert_array_equal(nav.date.to_numpy().astype("datetime64[D]"),
                                          old_nav.date.to_numpy().astype("datetime64[D]"))
            base = float(old_nav.nav.iloc[-1] - 1)
            net = float(nav.nav.iloc[-1] - 1)
            np.testing.assert_allclose(base, read(PRIOR / f"{old_stem}.json")["net_return"])
            np.testing.assert_allclose(net, metrics["net_return"])
            nav.to_parquet(OUT / f"{stem}_nav.parquet", index=False)
            trades.to_parquet(OUT / f"{stem}_orders.parquet", index=False)
            rows.append(dict(loss=loss, policy=policy, base=base, net=net,
                             annual=nav.nav.iloc[-1]**(252/len(nav))-1,
                             drawdown=metrics["max_drawdown"], audit=audit))
            print(f"{stem}: {base:.2%} -> {net:.2%}", flush=True)
    assert len(rows) == 6
    lines = ["# 严格成交与双倍冲击联合回测", "",
             "2017–2023 历史开发期，初始资金 1 亿元，往返基础费 20bp。两列均使用严格涨跌停和原始股数买入约束，冲击系数分别为 0.5 和 1.0。", "",
             "| 损失 | 退出规则 | 原冲击累计收益 | 双倍冲击累计收益 | 变化（百分点） | 双倍冲击年化收益 | 最大回撤 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['loss']} | {r['policy']} | {r['base']:.2%} | {r['net']:.2%} | {(r['net']-r['base'])*100:+.2f} | {r['annual']:.2%} | {r['drawdown']:.2%} |")
    lines += ["", "逐笔检查了成交日限价、原始股数和双边冲击费率，核对了账户现金、持仓、净值及年度复利。冲击系数为研究假设，卖出冲击沿用入场时估计。"]
    (OUT / "JOINT_STRESS_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    save(OUT / "summary.json", dict(status="verified", results=rows))


if __name__ == "__main__":
    main()
