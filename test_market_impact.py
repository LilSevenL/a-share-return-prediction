"""LightGBM Top50 的非线性市场冲击压力测试，仅使用2017-2024 Validation。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from backtest_validation import (
    PROCESSED,
    build_price_panel,
    configure_views,
    period_metrics,
    prepare_output,
    simulate,
    yearly_returns,
)
from stress_test_validation import add_stress_views, load_scenario_trades
from training_common import atomic_json, quote_sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-dir", default=str(PROCESSED / "validation_signals"))
    parser.add_argument("--targets-dir", default=str(PROCESSED / "targets_5d"))
    parser.add_argument("--output-dir", default=str(PROCESSED / "validation_stress_tests"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--impact-coefficient", type=float, default=0.5)
    parser.add_argument("--max-participation", type=float, default=0.05)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = prepare_output(Path(args.output_dir), False, args.min_free_gb)
    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit={quote_sql(args.memory_limit)}")
    con.execute(f"SET temp_directory={quote_sql(str(output / '_impact_tmp'))}")
    con.execute("SET preserve_insertion_order=false")
    configure_views(con, args)
    add_stress_views(con, output)
    print("构建价格矩阵与固定Top50候选……", flush=True)
    dates, close, _, next_sellable, date_to_index = build_price_panel(con)
    trades = load_scenario_trades(con, args.top_n, 50_000.0, False)
    con.register("selected_trades", trades)
    features = str(PROCESSED / "features_50" / "feature_year=*" / "*.parquet").replace("\\", "/")
    trades = con.execute(f"""
        SELECT t.*, f.volatility_20d
        FROM selected_trades t
        LEFT JOIN read_parquet({quote_sql(features)}, hive_partitioning=true) f
          USING (ts_code, signal_date)
    """).fetch_df()
    volatility = pd.to_numeric(trades["volatility_20d"], errors="coerce")
    fallback_volatility = float(volatility.median())
    trades["volatility_20d"] = volatility.fillna(fallback_volatility).clip(lower=0.001, upper=0.20)

    results = []
    for aum_million in [100, 200, 300, 500, 1000]:
        scenario = trades.copy()
        order_rmb = aum_million * 1_000_000 / (5 * args.top_n)
        participation = order_rmb / (scenario["signal_amount_thousand"].to_numpy(float) * 1_000)
        eligible = participation <= args.max_participation
        scenario = scenario.loc[eligible].copy()
        scenario_participation = participation[eligible]
        # 平方根冲击模型：单边冲击=Y*日波动率*sqrt(Q/V)。
        # 将买卖两边的估计冲击合并到有效入场价，再叠加固定20bps往返成本。
        roundtrip_impact = (
            2.0 * args.impact_coefficient
            * scenario["volatility_20d"].to_numpy(float)
            * np.sqrt(scenario_participation)
        )
        scenario["entry_adj_open"] = (
            scenario["entry_adj_open"].to_numpy(float) * (1.0 + roundtrip_impact)
        )
        nav, diagnostics = simulate(
            scenario, args.top_n, dates, close, next_sellable, date_to_index, 20.0, 0.25
        )
        results.append({
            "aum_million_rmb": aum_million,
            "fixed_roundtrip_cost_bps": 20.0,
            "impact_model": "roundtrip impact = 2 * Y * daily_volatility_20d * sqrt(order_value / signal_day_turnover)",
            "impact_coefficient_y": args.impact_coefficient,
            "max_participation": args.max_participation,
            "order_value_rmb_per_stock": order_rmb,
            "dropped_candidate_rows_over_cap": int((~eligible).sum()),
            "mean_participation_after_cap": float(np.mean(scenario_participation)),
            "p95_participation_after_cap": float(np.quantile(scenario_participation, 0.95)),
            "mean_estimated_roundtrip_impact_bps": float(np.mean(roundtrip_impact) * 10_000),
            "p95_estimated_roundtrip_impact_bps": float(np.quantile(roundtrip_impact, 0.95) * 10_000),
            "design_2017_2023": period_metrics(nav, dates, "2017-01-01", "2024-01-01"),
            "confirmation_2024": period_metrics(nav, dates, "2024-01-01", "2025-01-01"),
            "yearly_returns": yearly_returns(nav, dates, 2017, 2024),
            "diagnostics": diagnostics,
        })
        print(
            f"AUM={aum_million}百万元，设计期年化="
            f"{results[-1]['design_2017_2023']['annualized_return']:.2%}", flush=True
        )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "model_and_rule_frozen": "lightgbm_top50",
        "test_used": False,
        "results": results,
        "warnings": [
            "The square-root model is a stress assumption, not a calibrated intraday execution model.",
            "Candidates above the 5% participation cap are skipped without replacement.",
            "2025-2026 permanent Test was not read.",
        ],
    }
    atomic_json(summary, output / "market_impact_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
