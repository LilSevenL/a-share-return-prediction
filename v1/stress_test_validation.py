"""冻结 LightGBM Top50 后的 Validation 风险、容量与稳健性压力测试。

只读取 2017-2024 滚动 Validation 信号；2025-2026 永久 Test 不可见。
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backtest_validation import (
    PARQUET,
    PROCESSED,
    build_price_panel,
    configure_views,
    period_metrics,
    prepare_output,
    simulate,
    yearly_returns,
)
from training_common import PROJECT_DIR, atomic_json, quote_sql


DEFAULT_OUTPUT = PROCESSED / "validation_stress_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-dir", default=str(PROCESSED / "validation_signals"))
    parser.add_argument("--targets-dir", default=str(PROCESSED / "targets_5d"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    return parser.parse_args()


def add_stress_views(con: duckdb.DuckDBPyConnection, output: Path) -> None:
    daily_basic = str(PARQUET / "daily_basic" / "year=*" / "*.parquet").replace("\\", "/")
    con.execute(f"""
        CREATE TEMP VIEW daily_basic_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date, total_mv, circ_mv
        FROM read_parquet({quote_sql(daily_basic)}, hive_partitioning=true)
        WHERE trade_date >= DATE '2017-01-01' AND trade_date < DATE '2025-01-01'
    """)
    # stock_basic 的行业是当前静态分类，只用于暴露诊断，不冒充点时历史行业。
    stock_basic_csv = str(
        PROJECT_DIR / "quant_data" / "combined" / "stock_basic.csv"
    ).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP VIEW stock_meta AS
        SELECT ts_code, industry, market, list_status
        FROM stock_basic
        LEFT JOIN read_csv_auto(
            {quote_sql(stock_basic_csv)}, header=true, all_varchar=true
        ) USING (ts_code)
    """)


def load_scenario_trades(
    con: duckdb.DuckDBPyConnection,
    top_n: int,
    min_amount_thousand: float,
    exclude_restricted_limit_proxy: bool,
) -> pd.DataFrame:
    restricted_clause = "AND NOT restricted_limit_proxy" if exclude_restricted_limit_proxy else ""
    return con.execute(f"""
        WITH raw AS (
            SELECT pc.code_idx::INTEGER AS code_idx,
                   s.ts_code,
                   s.signal_date,
                   t.entry_date,
                   t.exit_date AS scheduled_exit_date,
                   t.entry_adj_open,
                   t.entry_tradable,
                   s.lightgbm_prediction AS prediction,
                   d.amount AS signal_amount_thousand,
                   db.total_mv,
                   COALESCE(m.industry, 'UNKNOWN') AS industry,
                   CASE
                     WHEN l.up_limit IS NULL OR d.pre_close IS NULL OR d.pre_close <= 0 THEN FALSE
                     WHEN l.up_limit / d.pre_close - 1.0 BETWEEN 0.02 AND 0.065 THEN TRUE
                     ELSE FALSE
                   END AS restricted_limit_proxy
            FROM signals s
            JOIN targets t USING (ts_code, signal_date)
            JOIN daily_data d ON d.ts_code=s.ts_code AND d.trade_date=s.signal_date
            JOIN price_codes pc USING (ts_code)
            LEFT JOIN limit_data l ON l.ts_code=s.ts_code AND l.trade_date=s.signal_date
            LEFT JOIN daily_basic_data db ON db.ts_code=s.ts_code AND db.trade_date=s.signal_date
            LEFT JOIN stock_meta m USING (ts_code)
            LEFT JOIN stock_basic b USING (ts_code)
            WHERE d.amount >= {float(min_amount_thousand)}
              AND (b.list_date IS NULL OR DATE_DIFF('day', b.list_date, s.signal_date) >= 60)
              AND s.lightgbm_prediction IS NOT NULL
        ), ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                     PARTITION BY signal_date ORDER BY prediction DESC NULLS LAST, ts_code
                   ) AS signal_rank,
                   PERCENT_RANK() OVER (
                     PARTITION BY signal_date ORDER BY total_mv NULLS LAST
                   ) AS market_cap_percentile
            FROM raw
            WHERE TRUE {restricted_clause}
        )
        SELECT * FROM ranked
        WHERE signal_rank <= {int(top_n)}
          AND entry_tradable
          AND entry_adj_open > 0
          AND scheduled_exit_date < MAKE_DATE(YEAR(signal_date) + 1, 1, 1)
        ORDER BY signal_date, signal_rank
    """).fetch_df()


def scenario_definition(top_n: int) -> list[dict[str, Any]]:
    scenarios = [
        {"id": "base_20bps_50m", "cost_bps": 20.0, "amount": 50_000.0, "st_proxy": False},
        {"id": "high_cost_50bps", "cost_bps": 50.0, "amount": 50_000.0, "st_proxy": False},
        {"id": "liquidity_30m", "cost_bps": 20.0, "amount": 30_000.0, "st_proxy": False},
        {"id": "liquidity_100m", "cost_bps": 20.0, "amount": 100_000.0, "st_proxy": False},
        {"id": "restricted_limit_proxy_excluded", "cost_bps": 20.0, "amount": 50_000.0, "st_proxy": True},
    ]
    # 每只股票买入金额=AUM/(5个袖套*Top-N)。要求不超过信号日成交额的参与率上限。
    for aum_million, participation in [(100, 0.01), (500, 0.01), (1000, 0.01), (500, 0.03), (1000, 0.03)]:
        required_amount = aum_million * 1_000_000 / (5 * top_n) / participation / 1_000
        scenarios.append({
            "id": f"capacity_{aum_million}m_cap{participation*100:g}pct",
            "cost_bps": 20.0,
            "amount": max(50_000.0, required_amount),
            "st_proxy": False,
            "aum_million": aum_million,
            "max_participation": participation,
            "required_amount_thousand": required_amount,
        })
    return scenarios


def benchmark_analysis(nav: np.ndarray, dates: np.ndarray) -> dict[str, Any]:
    source = PROJECT_DIR / "quant_data" / "combined" / "index_daily.csv"
    index_data = pd.read_csv(source, usecols=["ts_code", "trade_date", "close"])
    index_data["trade_date"] = pd.to_datetime(index_data["trade_date"].astype(str))
    index_data = index_data[index_data["ts_code"].isin(["000300.SH", "000905.SH", "000852.SH"])]
    pivot = index_data.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(dates),
        "portfolio": np.r_[0.0, nav[1:] / nav[:-1] - 1.0],
    }).set_index("trade_date").join(pivot.pct_change())
    output: dict[str, Any] = {}
    for start, end, period in [("2017-01-01", "2024-01-01", "design_2017_2023"),
                               ("2024-01-01", "2025-01-01", "confirmation_2024")]:
        part = frame.loc[(frame.index >= start) & (frame.index < end)].dropna()
        period_result: dict[str, Any] = {}
        for code in ["000300.SH", "000905.SH", "000852.SH"]:
            p = part["portfolio"].to_numpy(float)
            b = part[code].to_numpy(float)
            beta = float(np.cov(p, b, ddof=0)[0, 1] / np.var(b)) if np.var(b) > 0 else 0.0
            alpha = float((np.mean(p) - beta * np.mean(b)) * 252)
            active = p - b
            information_ratio = float(np.mean(active) / np.std(active) * math.sqrt(252)) if np.std(active) else 0.0
            period_result[code] = {
                "beta": beta,
                "annualized_jensen_alpha": alpha,
                "correlation": float(np.corrcoef(p, b)[0, 1]),
                "benchmark_total_return": float(np.prod(1.0 + b) - 1.0),
                "portfolio_minus_benchmark_total_return": float(np.prod(1.0 + p) - np.prod(1.0 + b)),
                "information_ratio": information_ratio,
            }
        output[period] = period_result
    return output


def exposure_analysis(trades: pd.DataFrame) -> dict[str, Any]:
    industry = trades["industry"].fillna("UNKNOWN").astype(str)
    shares = industry.value_counts(normalize=True)
    daily = (
        trades.assign(industry=industry)
        .groupby(["signal_date", "industry"]).size()
        .groupby(level=0).apply(lambda x: x / x.sum())
    )
    hhi = daily.pow(2).groupby(level=0).sum()
    total_mv = pd.to_numeric(trades["total_mv"], errors="coerce")
    cap_pct = pd.to_numeric(trades["market_cap_percentile"], errors="coerce")
    return {
        "classification_warning": "industry is the current stock_basic classification, not point-in-time history",
        "top_industries_by_selection_share": {str(k): float(v) for k, v in shares.head(10).items()},
        "unknown_industry_share": float((industry == "UNKNOWN").mean()),
        "mean_daily_industry_hhi": float(hhi.mean()),
        "max_daily_industry_hhi": float(hhi.max()),
        "median_total_mv_ten_thousand_rmb": float(total_mv.median()),
        "mean_market_cap_percentile_within_eligible_universe": float(cap_pct.mean()),
        "median_market_cap_percentile_within_eligible_universe": float(cap_pct.median()),
        "restricted_limit_proxy_selection_share": float(trades["restricted_limit_proxy"].mean()),
    }


def save_nav(path: Path, records: list[pd.DataFrame]) -> None:
    table = pa.Table.from_pandas(pd.concat(records, ignore_index=True), preserve_index=False)
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.top_n <= 0 or args.threads <= 0:
        raise ValueError("top-n和threads必须为正数")
    output = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit={quote_sql(args.memory_limit)}")
    con.execute(f"SET temp_directory={quote_sql(str(output / '_duckdb_tmp'))}")
    con.execute("SET preserve_insertion_order=false")
    configure_views(con, args)
    add_stress_views(con, output)
    print("构建价格和可卖出矩阵……", flush=True)
    dates, close, _, next_sellable, date_to_index = build_price_panel(con)

    trade_cache: dict[tuple[float, bool], pd.DataFrame] = {}
    results: list[dict[str, Any]] = []
    nav_records: list[pd.DataFrame] = []
    base_nav: np.ndarray | None = None
    base_trades: pd.DataFrame | None = None
    for number, scenario in enumerate(scenario_definition(args.top_n), 1):
        print(f"[{number}/10] {scenario['id']}……", flush=True)
        cache_key = (float(scenario["amount"]), bool(scenario["st_proxy"]))
        if cache_key not in trade_cache:
            trade_cache[cache_key] = load_scenario_trades(
                con, args.top_n, cache_key[0], cache_key[1]
            )
        trades = trade_cache[cache_key]
        nav, diagnostics = simulate(
            trades, args.top_n, dates, close, next_sellable, date_to_index,
            float(scenario["cost_bps"]), 0.25,
        )
        row = {
            **scenario,
            "design_2017_2023": period_metrics(nav, dates, "2017-01-01", "2024-01-01"),
            "confirmation_2024": period_metrics(nav, dates, "2024-01-01", "2025-01-01"),
            "yearly_returns": yearly_returns(nav, dates, 2017, 2024),
            "diagnostics": diagnostics,
        }
        results.append(row)
        nav_records.append(pd.DataFrame({
            "trade_date": dates,
            "scenario": scenario["id"],
            "nav": nav,
            "daily_return": np.r_[0.0, nav[1:] / nav[:-1] - 1.0],
        }))
        if scenario["id"] == "base_20bps_50m":
            base_nav, base_trades = nav, trades

    assert base_nav is not None and base_trades is not None
    summary = {
        "generated_at": datetime.now().isoformat(),
        "model_and_rule_frozen": "lightgbm_top50",
        "design_period": "2017-2023",
        "confirmation_period": "2024",
        "test_used": False,
        "scenario_results": results,
        "benchmark_analysis": benchmark_analysis(base_nav, dates),
        "exposure_analysis": exposure_analysis(base_trades),
        "limitations": [
            "The configured proxy returned no stock_st/st records; the historical risk-warning test uses a <=6.5% daily limit proxy and is incomplete for boards whose ST limit is 20%.",
            "Industry is current stock_basic classification rather than point-in-time history.",
            "Capacity scenarios impose participation eligibility but do not estimate nonlinear intraday market impact.",
            "2025-2026 permanent Test was not read.",
        ],
    }
    atomic_json(summary, output / "stress_test_summary.json")
    pd.DataFrame([
        {
            "scenario": row["id"],
            "cost_bps": row["cost_bps"],
            "min_amount_thousand": row["amount"],
            "exclude_restricted_limit_proxy": row["st_proxy"],
            **{f"design_{k}": v for k, v in row["design_2017_2023"].items()},
            **{f"confirm_{k}": v for k, v in row["confirmation_2024"].items()},
            "average_cohort_size": row["diagnostics"]["average_cohort_size"],
        }
        for row in results
    ]).to_csv(output / "scenario_comparison.csv", index=False, encoding="utf-8-sig")
    save_nav(output / "stress_daily_nav.parquet", nav_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
