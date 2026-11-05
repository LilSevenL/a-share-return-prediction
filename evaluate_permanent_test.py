"""冻结规则下对2025-2026永久Test执行唯一一次预测评估与组合回测。"""

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

from backtest_validation import build_price_panel, period_metrics, simulate, yearly_returns
from training_common import PROJECT_DIR, atomic_json, evaluate, quote_sql


PROCESSED = PROJECT_DIR / "quant_data" / "processed"
PARQUET = PROCESSED / "parquet"
TEST_ROOT = PROCESSED / "permanent_test"
SIGNALS = TEST_ROOT / "signals"
OUTPUT = TEST_ROOT / "evaluation"
MODEL_COLUMNS = {
    "mlp_primary": "mlp_prediction",
    "linear_baseline": "linear_prediction",
    "lightgbm_challenger": "lightgbm_prediction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    return parser.parse_args()


def paths(kind: str) -> str:
    values = [
        str(PARQUET / kind / f"year={year}" / "*.parquet").replace("\\", "/")
        for year in [2025, 2026]
    ]
    return "[" + ", ".join(quote_sql(value) for value in values) + "]"


def configure(con: duckdb.DuckDBPyConnection) -> None:
    signal_paths = [
        str(SIGNALS / f"test_signals_{year}.parquet").replace("\\", "/")
        for year in [2025, 2026]
    ]
    target_paths = [
        str(PROCESSED / "targets_5d" / f"signal_year={year}" / "*.parquet").replace("\\", "/")
        for year in [2025, 2026]
    ]
    feature_paths = [
        str(PROCESSED / "features_50" / f"feature_year={year}" / "*.parquet").replace("\\", "/")
        for year in [2025, 2026]
    ]
    sql_array = lambda values: "[" + ", ".join(quote_sql(v) for v in values) + "]"
    con.execute(f"CREATE TEMP VIEW signals AS SELECT * FROM read_parquet({sql_array(signal_paths)})")
    con.execute(f"""
        CREATE TEMP VIEW targets AS
        SELECT * FROM read_parquet({sql_array(target_paths)}, hive_partitioning=true)
        WHERE signal_date <= DATE '2026-08-20'
    """)
    con.execute(f"""
        CREATE TEMP VIEW test_features AS
        SELECT ts_code, signal_date, volatility_20d
        FROM read_parquet({sql_array(feature_paths)}, hive_partitioning=true)
        WHERE signal_date <= DATE '2026-08-20'
    """)
    con.execute(f"""
        CREATE TEMP VIEW daily_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date,
               open, high, low, close, pre_close, amount
        FROM read_parquet({paths('daily')}, hive_partitioning=true)
    """)
    con.execute(f"""
        CREATE TEMP VIEW adj_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date, adj_factor
        FROM read_parquet({paths('adj_factor')}, hive_partitioning=true)
    """)
    con.execute(f"""
        CREATE TEMP VIEW limit_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date, up_limit, down_limit
        FROM read_parquet({paths('stk_limit')}, hive_partitioning=true)
    """)
    stock_basic = str(PROJECT_DIR / "quant_data" / "combined" / "stock_basic.csv").replace("\\", "/")
    con.execute(f"""
        CREATE TEMP VIEW stock_basic AS
        SELECT ts_code, TRY_STRPTIME(CAST(list_date AS VARCHAR), '%Y%m%d')::DATE AS list_date
        FROM read_csv_auto({quote_sql(stock_basic)}, header=true)
    """)
    con.execute("""
        CREATE TEMP TABLE price_codes AS
        SELECT ts_code, ROW_NUMBER() OVER (ORDER BY ts_code)-1 AS code_idx
        FROM (SELECT DISTINCT ts_code FROM daily_data)
    """)
    con.execute("""
        CREATE TEMP TABLE price_calendar AS
        SELECT trade_date, ROW_NUMBER() OVER (ORDER BY trade_date)-1 AS date_idx
        FROM (SELECT DISTINCT trade_date FROM daily_data)
    """)


def predictive_metrics(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    table = con.execute("""
        SELECT s.signal_date, t.target_logret_5d,
               s.mlp_prediction, s.linear_prediction, s.lightgbm_prediction
        FROM signals s JOIN targets t USING (ts_code, signal_date)
        WHERE t.target_valid AND t.target_logret_5d IS NOT NULL
        ORDER BY s.signal_date, s.ts_code
    """).fetchnumpy()
    dates = table["signal_date"]
    target = table["target_logret_5d"].astype(np.float32)
    result: dict[str, Any] = {}
    for model, column in MODEL_COLUMNS.items():
        prediction = table[column].astype(np.float32)
        model_result = {"full_test": evaluate(prediction, target, dates)}
        for year in [2025, 2026]:
            mask = dates.astype("datetime64[Y]").astype(int) + 1970 == year
            model_result[str(year)] = evaluate(prediction[mask], target[mask], dates[mask])
        result[model] = model_result
    return result


def ranked_trades(con: duckdb.DuckDBPyConnection, column: str) -> pd.DataFrame:
    return con.execute(f"""
        WITH raw AS (
            SELECT pc.code_idx::INTEGER AS code_idx, s.ts_code, s.signal_date,
                   t.entry_date, t.exit_date AS scheduled_exit_date,
                   t.entry_adj_open, t.entry_tradable,
                   s.{column} AS prediction, d.amount AS signal_amount_thousand,
                   f.volatility_20d,
                   CASE WHEN l.up_limit IS NOT NULL AND d.pre_close > 0
                              AND l.up_limit / d.pre_close - 1.0 BETWEEN 0.02 AND 0.065
                        THEN TRUE ELSE FALSE END AS restricted_limit_proxy
            FROM signals s
            JOIN targets t USING (ts_code, signal_date)
            JOIN daily_data d ON d.ts_code=s.ts_code AND d.trade_date=s.signal_date
            JOIN price_codes pc USING (ts_code)
            JOIN test_features f USING (ts_code, signal_date)
            LEFT JOIN limit_data l ON l.ts_code=s.ts_code AND l.trade_date=s.signal_date
            LEFT JOIN stock_basic b USING (ts_code)
            WHERE d.amount >= 50000
              AND (b.list_date IS NULL OR DATE_DIFF('day', b.list_date, s.signal_date) >= 60)
              AND s.{column} IS NOT NULL
              AND NOT restricted_limit_proxy
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY signal_date ORDER BY prediction DESC NULLS LAST, ts_code
            ) AS signal_rank
            FROM raw
        )
        SELECT * FROM ranked
        WHERE signal_rank <= 50 AND entry_tradable AND entry_adj_open > 0
        ORDER BY signal_date, signal_rank
    """).fetch_df()


def apply_frozen_impact(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    order_value = 100_000_000 / (5 * 50)
    participation = order_value / (trades["signal_amount_thousand"].to_numpy(float) * 1000)
    eligible = participation <= 0.05
    selected = trades.loc[eligible].copy()
    participation = participation[eligible]
    volatility = pd.to_numeric(selected["volatility_20d"], errors="coerce")
    fallback = float(volatility.median())
    volatility = volatility.fillna(fallback).clip(lower=0.001, upper=0.20).to_numpy(float)
    roundtrip_impact = 2 * 0.5 * volatility * np.sqrt(participation)
    selected["entry_adj_open"] = selected["entry_adj_open"].to_numpy(float) * (1 + roundtrip_impact)
    return selected, {
        "order_value_rmb_per_stock": order_value,
        "dropped_candidate_rows_over_5pct_cap": int((~eligible).sum()),
        "mean_participation": float(np.mean(participation)),
        "p95_participation": float(np.quantile(participation, 0.95)),
        "mean_estimated_roundtrip_impact_bps": float(np.mean(roundtrip_impact) * 10000),
        "p95_estimated_roundtrip_impact_bps": float(np.quantile(roundtrip_impact, 0.95) * 10000),
    }


def benchmark_metrics(nav: np.ndarray, dates: np.ndarray) -> dict[str, Any]:
    data = pd.read_csv(
        PROJECT_DIR / "quant_data" / "combined" / "index_daily.csv",
        usecols=["ts_code", "trade_date", "close"],
    )
    data["trade_date"] = pd.to_datetime(data["trade_date"].astype(str))
    data = data[data["ts_code"].isin(["000300.SH", "000905.SH", "000852.SH"])]
    index_returns = data.pivot(index="trade_date", columns="ts_code", values="close").sort_index().pct_change()
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(dates),
        "portfolio": np.r_[0.0, nav[1:] / nav[:-1] - 1.0],
    }).set_index("trade_date").join(index_returns).dropna()
    output = {}
    for code in ["000300.SH", "000905.SH", "000852.SH"]:
        p, b = frame["portfolio"].to_numpy(float), frame[code].to_numpy(float)
        beta = float(np.cov(p, b, ddof=0)[0, 1] / np.var(b))
        active = p - b
        output[code] = {
            "beta": beta,
            "annualized_jensen_alpha": float((p.mean() - beta * b.mean()) * 252),
            "correlation": float(np.corrcoef(p, b)[0, 1]),
            "benchmark_total_return": float(np.prod(1 + b) - 1),
            "portfolio_minus_benchmark_total_return": float(np.prod(1 + p) - np.prod(1 + b)),
            "information_ratio": float(active.mean() / active.std() * math.sqrt(252)),
        }
    return output


def save_nav(path: Path, records: list[pd.DataFrame]) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pandas(pd.concat(records, ignore_index=True), preserve_index=False),
                   temporary, compression="zstd")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((TEST_ROOT / "PERMANENT_TEST_OPENED.json").read_text(encoding="utf-8"))
    signal_report = json.loads((SIGNALS / "signal_generation_report.json").read_text(encoding="utf-8"))
    if not signal_report.get("completed"):
        raise RuntimeError("永久Test信号尚未完整生成")
    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit={quote_sql(args.memory_limit)}")
    con.execute(f"SET temp_directory={quote_sql(str(output / '_duckdb_tmp'))}")
    con.execute("SET preserve_insertion_order=false")
    configure(con)
    print("计算永久Test预测指标……", flush=True)
    prediction_results = predictive_metrics(con)
    print("构建永久Test价格矩阵……", flush=True)
    dates, close, _, next_sellable, date_to_index = build_price_panel(con)
    portfolio_results = []
    nav_records = []
    primary_nav = None
    for model, column in MODEL_COLUMNS.items():
        print(f"回测{model}……", flush=True)
        trades = ranked_trades(con, column)
        impacted, impact = apply_frozen_impact(trades)
        nav, diagnostics = simulate(
            impacted, 50, dates, close, next_sellable, date_to_index, 20.0, 0.25
        )
        test_years = len(dates) / 252.0
        diagnostics["annualized_one_way_turnover_estimate"] = (
            diagnostics["completed_cohorts"] * 0.2 / test_years
        )
        diagnostics["unresolved_at_test_end"] = diagnostics.pop("unresolved_at_2024_end")
        diagnostics.pop("design_annualized_one_way_turnover_estimate")
        diagnostics.pop("confirmation_one_way_turnover_estimate")
        result = {
            "model": model, "role": (
                "primary" if model == "mlp_primary" else "baseline" if model == "linear_baseline" else "strong_challenger"
            ),
            "full_test": period_metrics(nav, dates, "2025-01-01", "2027-01-01"),
            "yearly_returns": yearly_returns(nav, dates, 2025, 2026),
            "impact": impact, "diagnostics": diagnostics,
        }
        portfolio_results.append(result)
        nav_records.append(pd.DataFrame({
            "trade_date": dates, "model": model, "nav": nav,
            "daily_return": np.r_[0.0, nav[1:] / nav[:-1] - 1.0],
        }))
        if model == "mlp_primary":
            primary_nav = nav
    assert primary_nav is not None
    summary = {
        "generated_at": datetime.now().isoformat(),
        "status": "PERMANENT_TEST_EVALUATED_ONCE",
        "primary_model_remains": "mlp_primary",
        "no_post_test_selection_or_tuning": True,
        "test_period": {"start": str(dates.min()), "end": str(dates.max()), "trading_days": int(len(dates))},
        "frozen_rule": lock["frozen_portfolio_rule"],
        "predictive_metrics": prediction_results,
        "portfolio_results": portfolio_results,
        "primary_benchmark_analysis": benchmark_metrics(primary_nav, dates),
        "warnings": [
            "2026 is partial through the available market-data end date.",
            "Historical ST filtering remains a restricted-limit proxy, not a complete point-in-time ST list.",
            "The square-root impact model is a frozen stress assumption rather than minute-data-calibrated execution cost.",
            "Test results must not be used to retune or switch the declared primary model.",
        ],
    }
    atomic_json(summary, output / "permanent_test_summary.json")
    pd.DataFrame([{
        "model": row["model"], "role": row["role"],
        **{f"portfolio_{k}": v for k, v in row["full_test"].items()},
        **{f"prediction_{k}": v for k, v in prediction_results[row["model"]]["full_test"].items()},
    } for row in portfolio_results]).to_csv(
        output / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    save_nav(output / "daily_nav.parquet", nav_records)
    con.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
