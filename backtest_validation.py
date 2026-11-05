"""严格的2017-2024样本外Validation组合回测；永久Test保持封存。

设计期为2017-2023，2024只用于冻结策略后的确认。信号在t日收盘后产生，
t+1开盘建仓，计划在t+5收盘卖出。使用5个等资金袖套处理重叠持仓；
一字跌停或停牌导致卖出延迟时，该袖套在完全退出前不会建立下一期仓位。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from training_common import PROJECT_DIR, atomic_json, quote_sql


PROCESSED = PROJECT_DIR / "quant_data" / "processed"
PARQUET = PROCESSED / "parquet"
DEFAULT_SIGNALS = PROCESSED / "validation_signals"
DEFAULT_TARGETS = PROCESSED / "targets_5d"
DEFAULT_OUTPUT = PROCESSED / "validation_backtest"
MODEL_COLUMNS = {
    "linear_19": "linear_prediction",
    "lightgbm": "lightgbm_prediction",
    "mlp": "mlp_prediction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-dir", default=str(DEFAULT_SIGNALS))
    parser.add_argument("--targets-dir", default=str(DEFAULT_TARGETS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-n", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--base-cost-bps", type=float, default=20.0)
    parser.add_argument("--cost-scenarios-bps", type=float, nargs="+", default=[10.0, 30.0])
    parser.add_argument("--buy-cost-share", type=float, default=0.25)
    parser.add_argument("--min-amount-thousand", type=float, default=50_000.0)
    parser.add_argument("--min-list-days", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parquet_glob(directory: Path, partition: str) -> str:
    if not directory.is_dir() or not next(directory.rglob("*.parquet"), None):
        raise FileNotFoundError(directory)
    return str(directory / partition / "*.parquet").replace("\\", "/")


def prepare_output(path: Path, force: bool, min_free_gb: float) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free / 1024**3 < min_free_gb:
        raise RuntimeError("磁盘剩余空间低于安全阈值")
    if path.exists() and force:
        backup = path.with_name(f"{path.name}.backup.{datetime.now():%Y%m%d%H%M%S}")
        path.replace(backup)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_views(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    signals = str(Path(args.signals_dir).resolve() / "validation_*.parquet").replace("\\", "/")
    targets = parquet_glob(Path(args.targets_dir).resolve(), "signal_year=*")
    daily = parquet_glob(PARQUET / "daily", "year=*")
    adj = parquet_glob(PARQUET / "adj_factor", "year=*")
    limit_data = parquet_glob(PARQUET / "stk_limit", "year=*")
    stock_basic = str(PROJECT_DIR / "quant_data" / "combined" / "stock_basic.csv").replace("\\", "/")
    con.execute(f"""
        CREATE TEMP VIEW signals AS
        SELECT * FROM read_parquet({quote_sql(signals)})
        WHERE signal_date >= DATE '2017-01-01' AND signal_date < DATE '2025-01-01'
    """)
    if con.execute("SELECT COUNT(*) FROM signals WHERE signal_date >= DATE '2025-01-01'").fetchone()[0]:
        raise RuntimeError("检测到Test年份信号")
    con.execute(f"""
        CREATE TEMP VIEW targets AS
        SELECT * FROM read_parquet({quote_sql(targets)}, hive_partitioning=true)
        WHERE signal_date >= DATE '2017-01-01' AND signal_date < DATE '2025-01-01'
    """)
    con.execute(f"""
        CREATE TEMP VIEW daily_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date,
               open, high, low, close, pre_close, amount
        FROM read_parquet({quote_sql(daily)}, hive_partitioning=true)
        WHERE trade_date >= DATE '2017-01-01' AND trade_date < DATE '2025-01-01'
    """)
    con.execute(f"""
        CREATE TEMP VIEW adj_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date, adj_factor
        FROM read_parquet({quote_sql(adj)}, hive_partitioning=true)
        WHERE trade_date >= DATE '2017-01-01' AND trade_date < DATE '2025-01-01'
    """)
    con.execute(f"""
        CREATE TEMP VIEW limit_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date, up_limit, down_limit
        FROM read_parquet({quote_sql(limit_data)}, hive_partitioning=true)
        WHERE trade_date >= DATE '2017-01-01' AND trade_date < DATE '2025-01-01'
    """)
    con.execute(f"""
        CREATE TEMP VIEW stock_basic AS
        SELECT ts_code,
               TRY_STRPTIME(CAST(list_date AS VARCHAR), '%Y%m%d')::DATE AS list_date
        FROM read_csv_auto({quote_sql(stock_basic)}, header=true)
    """)
    con.execute("""
        CREATE TEMP TABLE price_codes AS
        SELECT ts_code, ROW_NUMBER() OVER (ORDER BY ts_code) - 1 AS code_idx
        FROM (SELECT DISTINCT ts_code FROM daily_data)
    """)
    con.execute("""
        CREATE TEMP TABLE price_calendar AS
        SELECT trade_date, ROW_NUMBER() OVER (ORDER BY trade_date) - 1 AS date_idx
        FROM (SELECT DISTINCT trade_date FROM daily_data)
    """)


def build_price_panel(
    con: duckdb.DuckDBPyConnection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[np.datetime64, int]]:
    dates = con.execute(
        "SELECT trade_date FROM price_calendar ORDER BY date_idx"
    ).fetchnumpy()["trade_date"].astype("datetime64[D]")
    n_codes = int(con.execute("SELECT COUNT(*) FROM price_codes").fetchone()[0])
    table = con.execute("""
        SELECT pc.code_idx::INTEGER AS code_idx,
               cal.date_idx::INTEGER AS date_idx,
               CAST(d.close * a.adj_factor AS FLOAT) AS adj_close,
               CAST(NOT (
                    l.down_limit IS NOT NULL
                    AND ABS(d.high - l.down_limit) <= 1e-6
                    AND ABS(d.low - l.down_limit) <= 1e-6
               ) AS BOOLEAN) AS sellable
        FROM daily_data d
        JOIN adj_data a USING (ts_code, trade_date)
        JOIN price_codes pc USING (ts_code)
        JOIN price_calendar cal USING (trade_date)
        LEFT JOIN limit_data l USING (ts_code, trade_date)
        ORDER BY pc.code_idx, cal.date_idx
    """).to_arrow_table()
    code_idx = table["code_idx"].combine_chunks().to_numpy(zero_copy_only=False)
    date_idx = table["date_idx"].combine_chunks().to_numpy(zero_copy_only=False)
    prices = table["adj_close"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32)
    can_sell = table["sellable"].combine_chunks().to_numpy(zero_copy_only=False)
    close = np.full((n_codes, len(dates)), np.nan, dtype=np.float32)
    sellable = np.zeros((n_codes, len(dates)), dtype=bool)
    close[code_idx, date_idx] = prices
    sellable[code_idx, date_idx] = can_sell
    # 停牌日以最近收盘价估值，但不能卖出。
    for row in range(n_codes):
        valid = np.flatnonzero(np.isfinite(close[row]))
        if not len(valid):
            continue
        indices = np.zeros(len(dates), dtype=np.int32)
        indices[valid] = valid
        np.maximum.accumulate(indices, out=indices)
        close[row, valid[0]:] = close[row, indices[valid[0]:]]
    next_sellable = np.full((n_codes, len(dates)), -1, dtype=np.int32)
    next_index = np.full(n_codes, -1, dtype=np.int32)
    for column in range(len(dates) - 1, -1, -1):
        valid = sellable[:, column] & np.isfinite(close[:, column])
        next_index[valid] = column
        next_sellable[:, column] = next_index
    date_to_index = {value: index for index, value in enumerate(dates)}
    return dates, close, sellable, next_sellable, date_to_index


def load_ranked_trades(
    con: duckdb.DuckDBPyConnection,
    prediction_column: str,
    max_top_n: int,
    min_amount: float,
    min_list_days: int,
) -> pd.DataFrame:
    return con.execute(f"""
        WITH eligible AS (
            SELECT pc.code_idx::INTEGER AS code_idx,
                   s.ts_code,
                   s.signal_date,
                   t.entry_date,
                   t.exit_date AS scheduled_exit_date,
                   t.entry_adj_open,
                   t.entry_tradable,
                   s.{prediction_column} AS prediction,
                   ROW_NUMBER() OVER (
                       PARTITION BY s.signal_date
                       ORDER BY s.{prediction_column} DESC NULLS LAST, s.ts_code
                   ) AS signal_rank
            FROM signals s
            JOIN targets t USING (ts_code, signal_date)
            JOIN daily_data d ON d.ts_code=s.ts_code AND d.trade_date=s.signal_date
            JOIN price_codes pc USING (ts_code)
            LEFT JOIN stock_basic b USING (ts_code)
            WHERE d.amount >= {float(min_amount)}
              AND (b.list_date IS NULL OR DATE_DIFF('day', b.list_date, s.signal_date) >= {int(min_list_days)})
              AND s.{prediction_column} IS NOT NULL
        )
        SELECT * FROM eligible
        WHERE signal_rank <= {int(max_top_n)}
          AND entry_tradable
          AND entry_adj_open > 0
          AND scheduled_exit_date < MAKE_DATE(YEAR(signal_date) + 1, 1, 1)
        ORDER BY signal_date, signal_rank
    """).fetch_df()


def period_metrics(nav: np.ndarray, dates: np.ndarray, start: str, end: str) -> dict[str, float]:
    start_day = np.datetime64(start)
    end_day = np.datetime64(end)
    indices = np.flatnonzero((dates >= start_day) & (dates < end_day))
    if not len(indices):
        raise RuntimeError(f"期间无交易日：{start}至{end}")
    first, last = int(indices[0]), int(indices[-1])
    base = float(nav[first - 1]) if first > 0 else 1.0
    segment = nav[first:last + 1] / base
    previous = np.r_[1.0, segment[:-1]]
    returns = segment / previous - 1.0
    years = len(segment) / 252.0
    total = float(segment[-1] - 1.0)
    annual = float(segment[-1] ** (1.0 / years) - 1.0) if segment[-1] > 0 else -1.0
    volatility = float(np.std(returns, ddof=0) * math.sqrt(252))
    sharpe = float(np.mean(returns) / np.std(returns, ddof=0) * math.sqrt(252)) if np.std(returns) > 0 else 0.0
    running_max = np.maximum.accumulate(segment)
    drawdowns = segment / running_max - 1.0
    max_drawdown = float(drawdowns.min())
    calmar = float(annual / abs(max_drawdown)) if max_drawdown < 0 else float("nan")
    return {
        "trading_days": int(len(segment)),
        "total_return": total,
        "annualized_return": annual,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "positive_day_ratio": float(np.mean(returns > 0)),
    }


def yearly_returns(nav: np.ndarray, dates: np.ndarray, start_year: int, end_year: int) -> dict[str, float]:
    output: dict[str, float] = {}
    for year in range(start_year, end_year + 1):
        indices = np.flatnonzero(
            (dates >= np.datetime64(f"{year}-01-01"))
            & (dates < np.datetime64(f"{year + 1}-01-01"))
        )
        if not len(indices):
            continue
        first, last = int(indices[0]), int(indices[-1])
        base = float(nav[first - 1]) if first > 0 else 1.0
        output[str(year)] = float(nav[last] / base - 1.0)
    return output


def simulate(
    trades: pd.DataFrame,
    top_n: int,
    dates: np.ndarray,
    close: np.ndarray,
    next_sellable: np.ndarray,
    date_to_index: dict[np.datetime64, int],
    total_cost_bps: float,
    buy_cost_share: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = trades.loc[trades["signal_rank"] <= top_n].copy()
    selected["signal_day"] = selected["signal_date"].values.astype("datetime64[D]")
    selected["entry_day"] = selected["entry_date"].values.astype("datetime64[D]")
    selected["scheduled_exit_day"] = selected["scheduled_exit_date"].values.astype("datetime64[D]")
    calendar_lookup = np.asarray([date_to_index.get(value, -1) for value in selected["signal_day"]], dtype=np.int32)
    selected["sleeve"] = calendar_lookup % 5
    buy_cost = total_cost_bps * buy_cost_share / 10_000.0
    sell_cost = total_cost_bps * (1.0 - buy_cost_share) / 10_000.0
    sleeve_nav = np.ones((5, len(dates)), dtype=np.float64)
    cohorts = skipped = delayed = unresolved = holdings = 0
    design_cohorts = confirmation_cohorts = 0
    cohort_sizes: list[int] = []

    for sleeve in range(5):
        data = selected.loc[selected["sleeve"] == sleeve]
        nav_level = 1.0
        last_written = 0
        busy_until = -1
        for _, cohort in data.groupby("signal_day", sort=True):
            entry_day = np.datetime64(cohort["entry_day"].iloc[0], "D")
            entry_index = date_to_index.get(entry_day, -1)
            if entry_index < 0 or entry_index <= busy_until:
                skipped += 1
                continue
            code_indices = cohort["code_idx"].to_numpy(dtype=np.int32)
            scheduled_days = cohort["scheduled_exit_day"].to_numpy(dtype="datetime64[D]")
            scheduled_indices = np.asarray(
                [date_to_index.get(value, -1) for value in scheduled_days], dtype=np.int32
            )
            valid = scheduled_indices >= 0
            if not valid.any():
                skipped += 1
                continue
            code_indices = code_indices[valid]
            scheduled_indices = scheduled_indices[valid]
            entry_prices = cohort["entry_adj_open"].to_numpy(dtype=np.float64)[valid]
            actual_exit = next_sellable[code_indices, scheduled_indices]
            unresolved_mask = actual_exit < 0
            unresolved += int(unresolved_mask.sum())
            actual_exit[unresolved_mask] = len(dates) - 1
            delayed += int((actual_exit > scheduled_indices).sum())
            end_index = int(actual_exit.max())
            day_indices = np.arange(entry_index, end_index + 1, dtype=np.int32)
            price_path = close[code_indices[:, None], day_indices[None, :]].astype(np.float64)
            values = price_path / entry_prices[:, None] / (1.0 + buy_cost)
            for row, exit_index in enumerate(actual_exit):
                relative = int(exit_index - entry_index)
                if relative < 0 or relative >= values.shape[1]:
                    continue
                terminal = values[row, relative]
                if not unresolved_mask[row]:
                    terminal *= 1.0 - sell_cost
                values[row, relative:] = terminal
            cohort_nav = np.nanmean(values, axis=0)
            if not np.isfinite(cohort_nav).all() or cohort_nav[0] <= 0:
                skipped += 1
                continue
            sleeve_nav[sleeve, last_written:entry_index] = nav_level
            sleeve_nav[sleeve, entry_index:end_index + 1] = nav_level * cohort_nav
            nav_level *= float(cohort_nav[-1])
            last_written = end_index + 1
            busy_until = end_index
            cohorts += 1
            if entry_day < np.datetime64("2024-01-01"):
                design_cohorts += 1
            else:
                confirmation_cohorts += 1
            holdings += len(code_indices)
            cohort_sizes.append(len(code_indices))
        sleeve_nav[sleeve, last_written:] = nav_level

    portfolio_nav = sleeve_nav.mean(axis=0)
    diagnostics = {
        "completed_cohorts": cohorts,
        "skipped_cohorts_due_to_busy_sleeve_or_missing_dates": skipped,
        "selected_holdings": holdings,
        "average_cohort_size": float(np.mean(cohort_sizes)) if cohort_sizes else 0.0,
        "delayed_exit_holdings": delayed,
        "unresolved_at_2024_end": unresolved,
        "design_annualized_one_way_turnover_estimate": design_cohorts * 0.2 / 7.0,
        "confirmation_one_way_turnover_estimate": confirmation_cohorts * 0.2,
    }
    return portfolio_nav, diagnostics


def save_daily_nav(path: Path, records: list[pd.DataFrame]) -> None:
    combined = pd.concat(records, ignore_index=True)
    temp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), temp, compression="zstd")
    temp.replace(path)


def main() -> None:
    args = parse_args()
    if args.threads <= 0 or not args.top_n or any(value <= 0 for value in args.top_n):
        raise ValueError("threads和top-n必须为正数")
    if not 0 <= args.buy_cost_share <= 1 or args.base_cost_bps < 0:
        raise ValueError("费用参数无效")
    output = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit={quote_sql(args.memory_limit)}")
    con.execute(f"SET temp_directory={quote_sql(str(output / '_duckdb_tmp'))}")
    con.execute("SET preserve_insertion_order=false")
    configure_views(con, args)
    print("构建2017-2024复权价格与可卖出矩阵……", flush=True)
    dates, close, _, next_sellable, date_to_index = build_price_panel(con)
    base_results: list[dict[str, Any]] = []
    nav_records: list[pd.DataFrame] = []
    trades_by_model: dict[str, pd.DataFrame] = {}
    max_top_n = max(args.top_n)
    for model_name, column in MODEL_COLUMNS.items():
        print(f"加载{model_name}可交易候选……", flush=True)
        trades = load_ranked_trades(
            con, column, max_top_n, args.min_amount_thousand, args.min_list_days
        )
        trades_by_model[model_name] = trades
        for top_n in sorted(set(args.top_n)):
            print(f"    回测{model_name} Top{top_n}，成本{args.base_cost_bps:g}bps", flush=True)
            nav, diagnostics = simulate(
                trades, top_n, dates, close, next_sellable, date_to_index,
                args.base_cost_bps, args.buy_cost_share,
            )
            design = period_metrics(nav, dates, "2017-01-01", "2024-01-01")
            confirmation = period_metrics(nav, dates, "2024-01-01", "2025-01-01")
            identifier = f"{model_name}_top{top_n}_cost{args.base_cost_bps:g}bps"
            base_results.append({
                "config_id": identifier,
                "model": model_name,
                "top_n": top_n,
                "round_trip_cost_bps": args.base_cost_bps,
                "design_2017_2023": design,
                "confirmation_2024": confirmation,
                "yearly_returns": yearly_returns(nav, dates, 2017, 2024),
                "diagnostics": diagnostics,
            })
            nav_records.append(pd.DataFrame({
                "trade_date": dates,
                "config_id": identifier,
                "nav": nav,
                "daily_return": np.r_[0.0, nav[1:] / nav[:-1] - 1.0],
            }))

    # 只使用2017-2023选择；Sharpe近似并列时偏向更大的Top-N以提高容量。
    ranked = sorted(
        base_results,
        key=lambda row: (
            row["design_2017_2023"]["sharpe"],
            row["top_n"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    sensitivity: list[dict[str, Any]] = []
    selected_trades = trades_by_model[selected["model"]]
    for cost in sorted(set(args.cost_scenarios_bps)):
        nav, diagnostics = simulate(
            selected_trades, selected["top_n"], dates, close, next_sellable,
            date_to_index, cost, args.buy_cost_share,
        )
        identifier = f"{selected['model']}_top{selected['top_n']}_cost{cost:g}bps"
        sensitivity.append({
            "config_id": identifier,
            "round_trip_cost_bps": cost,
            "design_2017_2023": period_metrics(nav, dates, "2017-01-01", "2024-01-01"),
            "confirmation_2024": period_metrics(nav, dates, "2024-01-01", "2025-01-01"),
            "yearly_returns": yearly_returns(nav, dates, 2017, 2024),
            "diagnostics": diagnostics,
        })
        nav_records.append(pd.DataFrame({
            "trade_date": dates,
            "config_id": identifier,
            "nav": nav,
            "daily_return": np.r_[0.0, nav[1:] / nav[:-1] - 1.0],
        }))

    summary = {
        "generated_at": datetime.now().isoformat(),
        "method": {
            "design_period": "2017-2023",
            "confirmation_period": "2024",
            "test_used": False,
            "signal_timing": "signal at t close, enter t+1 open, planned exit t+5 close",
            "overlap_handling": "five equal-capital sleeves",
            "blocked_exit_handling": "delay until first non-one-price-limit-down tradable close; sleeve remains busy",
            "entry_filter": {
                "entry_tradable": True,
                "ranking_timing": "rank first using signal-day information; untradable entries are skipped without replacement",
                "minimum_signal_day_amount_thousand": args.min_amount_thousand,
                "minimum_calendar_days_since_listing": args.min_list_days,
                "historical_st_filter": "not applied because historical ST status is unavailable",
            },
            "base_round_trip_cost_bps": args.base_cost_bps,
            "buy_cost_share": args.buy_cost_share,
            "selection_rule": "highest 2017-2023 net Sharpe; tie-break toward larger Top-N",
        },
        "selected_on_design_only": selected["config_id"],
        "base_results": base_results,
        "selected_cost_sensitivity": sensitivity,
        "warnings": [
            "Historical ST status was unavailable and therefore not filtered.",
            "Liquidity filtering uses signal-day amount; capacity still depends on actual portfolio capital.",
            "2025-2026 permanent Test was not read.",
        ],
    }
    atomic_json(summary, output / "backtest_summary.json")
    save_daily_nav(output / "daily_nav.parquet", nav_records)
    pd.DataFrame([
        {
            "config_id": row["config_id"],
            "model": row["model"],
            "top_n": row["top_n"],
            **{f"design_{key}": value for key, value in row["design_2017_2023"].items()},
            **{f"confirm_{key}": value for key, value in row["confirmation_2024"].items()},
        }
        for row in base_results
    ]).to_csv(output / "backtest_comparison.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
