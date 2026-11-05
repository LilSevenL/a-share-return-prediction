"""构建未来 5 个交易日的复权收益预测标签。

标签口径：在 signal_date 收盘后生成信号，于下一市场交易日开盘买入，
并在第 5 个未来市场交易日收盘卖出：

    target_logret_5d = ln(adjusted_close[t+5] / adjusted_open[t+1])

脚本只读取清洗后的 Parquet，不修改原始 Tushare 数据。默认输出到
quant_data/processed/targets_5d，并按 signal_year 分区。

小规模验证：
    python build_targets.py --start-date 20250101 --end-date 20251231 \
        --max-stocks 20 --output-dir quant_data/processed/targets_5d_test --force

全市场构建：
    python build_targets.py
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


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "quant_data" / "processed"
PARQUET_DIR = PROCESSED_DIR / "parquet"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "targets_5d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", default=str(PARQUET_DIR / "daily"))
    parser.add_argument("--adj-factor-dir", default=str(PARQUET_DIR / "adj_factor"))
    parser.add_argument("--stk-limit-dir", default=str(PARQUET_DIR / "stk_limit"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", help="信号开始日期 YYYYMMDD；默认使用全部数据")
    parser.add_argument("--end-date", help="信号结束日期 YYYYMMDD；默认使用全部数据")
    parser.add_argument("--max-stocks", type=int, help="仅处理代码排序后的前 N 只股票")
    parser.add_argument("--threads", type=int, default=4, help="DuckDB 工作线程数")
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB 内存上限，例如 8GB；超出部分写入临时目录",
    )
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument(
        "--price-tolerance",
        type=float,
        default=1e-6,
        help="识别一字涨跌停时的价格比较容差",
    )
    parser.add_argument("--force", action="store_true", help="备份已有结果并重新生成")
    return parser.parse_args()


def validate_day(value: str | None, name: str) -> None:
    if value is None:
        return
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"{name} 必须是 YYYYMMDD 格式")
    datetime.strptime(value, "%Y%m%d")


def ensure_space(path: Path, minimum_gb: float) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path.parent).free / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"磁盘仅剩 {free_gb:.2f}GB，低于安全阈值 {minimum_gb:.2f}GB"
        )
    return free_gb


def parquet_glob(directory: Path) -> str:
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Parquet 目录不存在：{resolved}")
    if not next(resolved.rglob("*.parquet"), None):
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{resolved}")
    return str(resolved / "year=*" / "*.parquet").replace("\\", "/")


def quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temp = path.with_suffix(".tmp.json")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def prepare_destination(output_dir: Path, force: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"输出目录已经存在：{output_dir}。如需重新生成，请添加 --force。"
            )
        backup = output_dir.with_name(
            f"{output_dir.name}.backup.{datetime.now():%Y%m%d%H%M%S}"
        )
        output_dir.replace(backup)
        print(f"已有结果已备份到：{backup}", flush=True)

    staging = output_dir.with_name(f"{output_dir.name}._staging")
    if staging.exists():
        stale = staging.with_name(
            f"{staging.name}.stale.{datetime.now():%Y%m%d%H%M%S}"
        )
        staging.replace(stale)
        print(f"上次未完成的临时目录已保留为：{stale}", flush=True)
    staging.mkdir(parents=True)
    return staging


def create_views(
    connection: duckdb.DuckDBPyConnection,
    daily_glob: str,
    adj_glob: str,
    limit_glob: str,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP VIEW daily_data AS
        SELECT
            ts_code,
            CAST(trade_date AS DATE) AS trade_date,
            open,
            high,
            low,
            close
        FROM read_parquet({quote_sql(daily_glob)}, hive_partitioning=true)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW adj_data AS
        SELECT
            ts_code,
            CAST(trade_date AS DATE) AS trade_date,
            adj_factor
        FROM read_parquet({quote_sql(adj_glob)}, hive_partitioning=true)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW limit_data AS
        SELECT
            ts_code,
            CAST(trade_date AS DATE) AS trade_date,
            up_limit,
            down_limit
        FROM read_parquet({quote_sql(limit_glob)}, hive_partitioning=true)
        """
    )


def build_query(args: argparse.Namespace) -> str:
    signal_filters: list[str] = []
    if args.start_date:
        signal_filters.append(
            f"s.trade_date >= STRPTIME({quote_sql(args.start_date)}, '%Y%m%d')::DATE"
        )
    if args.end_date:
        signal_filters.append(
            f"s.trade_date <= STRPTIME({quote_sql(args.end_date)}, '%Y%m%d')::DATE"
        )
    where_clause = " AND ".join(signal_filters) if signal_filters else "TRUE"

    universe_cte = ""
    universe_join = ""
    if args.max_stocks:
        universe_cte = f"""
        selected_stocks AS (
            SELECT DISTINCT ts_code
            FROM daily_data
            ORDER BY ts_code
            LIMIT {int(args.max_stocks)}
        ),"""
        universe_join = "JOIN selected_stocks u ON u.ts_code = s.ts_code"

    tolerance = float(args.price_tolerance)
    return f"""
    WITH market_calendar AS (
        SELECT
            trade_date,
            LEAD(trade_date, 1) OVER (ORDER BY trade_date) AS entry_date,
            LEAD(trade_date, 5) OVER (ORDER BY trade_date) AS exit_date
        FROM (SELECT DISTINCT trade_date FROM daily_data)
    ),
    {universe_cte}
    joined AS (
        SELECT
            s.ts_code,
            s.trade_date AS signal_date,
            c.entry_date,
            c.exit_date,
            e.open AS entry_open,
            e.high AS entry_high,
            e.low AS entry_low,
            ea.adj_factor AS entry_adj_factor,
            el.up_limit AS entry_up_limit,
            x.close AS exit_close,
            x.high AS exit_high,
            x.low AS exit_low,
            xa.adj_factor AS exit_adj_factor,
            xl.down_limit AS exit_down_limit
        FROM daily_data s
        JOIN market_calendar c ON c.trade_date = s.trade_date
        {universe_join}
        LEFT JOIN daily_data e
          ON e.ts_code = s.ts_code AND e.trade_date = c.entry_date
        LEFT JOIN adj_data ea
          ON ea.ts_code = s.ts_code AND ea.trade_date = c.entry_date
        LEFT JOIN limit_data el
          ON el.ts_code = s.ts_code AND el.trade_date = c.entry_date
        LEFT JOIN daily_data x
          ON x.ts_code = s.ts_code AND x.trade_date = c.exit_date
        LEFT JOIN adj_data xa
          ON xa.ts_code = s.ts_code AND xa.trade_date = c.exit_date
        LEFT JOIN limit_data xl
          ON xl.ts_code = s.ts_code AND xl.trade_date = c.exit_date
        WHERE {where_clause}
    ),
    flags AS (
        SELECT
            *,
            entry_open * entry_adj_factor AS entry_adj_open,
            exit_close * exit_adj_factor AS exit_adj_close,
            (
                entry_up_limit IS NOT NULL
                AND ABS(entry_high - entry_up_limit) <= {tolerance}
                AND ABS(entry_low - entry_up_limit) <= {tolerance}
            ) AS entry_one_price_limit_up,
            (
                exit_down_limit IS NOT NULL
                AND ABS(exit_high - exit_down_limit) <= {tolerance}
                AND ABS(exit_low - exit_down_limit) <= {tolerance}
            ) AS exit_one_price_limit_down,
            COALESCE((
                entry_date IS NOT NULL
                AND entry_open > 0
                AND entry_adj_factor > 0
            ), FALSE) AS entry_price_available,
            COALESCE((
                exit_date IS NOT NULL
                AND exit_close > 0
                AND exit_adj_factor > 0
            ), FALSE) AS exit_price_available
        FROM joined
    ),
    labelled AS (
        SELECT
            *,
            (
                entry_price_available
                AND NOT COALESCE(entry_one_price_limit_up, FALSE)
            ) AS entry_tradable,
            (
                exit_price_available
                AND NOT COALESCE(exit_one_price_limit_down, FALSE)
            ) AS exit_tradable
        FROM flags
    )
    SELECT
        ts_code,
        signal_date,
        entry_date,
        exit_date,
        entry_open,
        entry_adj_factor,
        entry_adj_open,
        exit_close,
        exit_adj_factor,
        exit_adj_close,
        COALESCE(entry_one_price_limit_up, FALSE) AS entry_one_price_limit_up,
        COALESCE(exit_one_price_limit_down, FALSE) AS exit_one_price_limit_down,
        entry_tradable,
        exit_tradable,
        (entry_tradable AND exit_tradable) AS target_valid,
        CASE
            WHEN entry_price_available AND exit_price_available
            THEN LN(exit_adj_close / entry_adj_open)
        END AS target_logret_5d_raw,
        CASE
            WHEN entry_price_available AND exit_price_available
            THEN exit_adj_close / entry_adj_open - 1.0
        END AS target_ret_5d_raw,
        CASE
            WHEN entry_tradable AND exit_tradable
            THEN LN(exit_adj_close / entry_adj_open)
        END AS target_logret_5d,
        CASE
            WHEN entry_tradable AND exit_tradable
            THEN exit_adj_close / entry_adj_open - 1.0
        END AS target_ret_5d,
        CASE
            WHEN entry_date IS NULL OR entry_open IS NULL THEN 'missing_entry_price'
            WHEN entry_adj_factor IS NULL OR entry_adj_factor <= 0 THEN 'missing_entry_adj_factor'
            WHEN COALESCE(entry_one_price_limit_up, FALSE) THEN 'entry_one_price_limit_up'
            WHEN exit_date IS NULL OR exit_close IS NULL THEN 'missing_exit_price'
            WHEN exit_adj_factor IS NULL OR exit_adj_factor <= 0 THEN 'missing_exit_adj_factor'
            WHEN COALESCE(exit_one_price_limit_down, FALSE) THEN 'exit_one_price_limit_down'
            ELSE 'valid'
        END AS target_status,
        CAST(EXTRACT(YEAR FROM signal_date) AS SMALLINT) AS signal_year
    FROM labelled
    """


def main() -> None:
    args = parse_args()
    validate_day(args.start_date, "start-date")
    validate_day(args.end_date, "end-date")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("start-date 不能晚于 end-date")
    if args.max_stocks is not None and args.max_stocks <= 0:
        raise ValueError("max-stocks 必须大于 0")
    if args.threads <= 0:
        raise ValueError("threads 必须大于 0")
    if args.price_tolerance < 0:
        raise ValueError("price-tolerance 不能小于 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    free_gb_before = ensure_space(output_dir, args.min_free_gb)
    daily_glob = parquet_glob(Path(args.daily_dir))
    adj_glob = parquet_glob(Path(args.adj_factor_dir))
    limit_glob = parquet_glob(Path(args.stk_limit_dir))
    staging = prepare_destination(output_dir, args.force)
    database_path = staging.parent / f".{output_dir.name}.duckdb"

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(f"SET threads = {int(args.threads)}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET temp_directory = {quote_sql(str(staging / '_tmp'))}")
        connection.execute(f"SET memory_limit = {quote_sql(args.memory_limit)}")
        create_views(connection, daily_glob, adj_glob, limit_glob)
        query = build_query(args)

        print("正在构建未来5个交易日收益标签……", flush=True)
        copy_sql = f"""
            COPY ({query})
            TO {quote_sql(str(staging).replace(chr(92), '/'))}
            (
                FORMAT PARQUET,
                PARTITION_BY (signal_year),
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000,
                OVERWRITE_OR_IGNORE TRUE
            )
        """
        connection.execute(copy_sql)

        output_glob = str(staging / "signal_year=*" / "*.parquet").replace("\\", "/")
        summary_row = connection.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT ts_code) AS stocks,
                MIN(signal_date) AS first_signal_date,
                MAX(signal_date) AS last_signal_date,
                COUNT(*) FILTER (WHERE target_valid) AS valid_targets,
                COUNT(*) FILTER (WHERE NOT target_valid) AS invalid_targets,
                COUNT(*) FILTER (WHERE entry_one_price_limit_up) AS blocked_entries,
                COUNT(*) FILTER (WHERE exit_one_price_limit_down) AS blocked_exits,
                COUNT(*) FILTER (
                    WHERE target_logret_5d IS NOT NULL
                      AND ABS((EXP(target_logret_5d) - 1.0) - target_ret_5d) > 1e-10
                ) AS return_identity_errors
            FROM read_parquet(?, hive_partitioning=true)
            """,
            [output_glob],
        ).fetchone()
        status_rows = connection.execute(
            """
            SELECT target_status, COUNT(*) AS rows
            FROM read_parquet(?, hive_partitioning=true)
            GROUP BY target_status
            ORDER BY rows DESC
            """,
            [output_glob],
        ).fetchall()
    except Exception:
        print(f"构建失败；临时结果保留在：{staging}", flush=True)
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    if summary_row[0] == 0:
        raise RuntimeError("没有生成任何标签记录")
    if summary_row[8] != 0:
        raise RuntimeError(f"对数收益与普通收益转换校验失败：{summary_row[8]} 行")

    staging.replace(output_dir)
    report = {
        "generated_at": datetime.now().isoformat(),
        "definition": "ln(adjusted_close[t+5] / adjusted_open[t+1])",
        "holding_period": "signal at t close; enter t+1 open; exit t+5 close",
        "signal_start_date": args.start_date,
        "signal_end_date": args.end_date,
        "max_stocks": args.max_stocks,
        "rows": int(summary_row[0]),
        "stocks": int(summary_row[1]),
        "first_signal_date": summary_row[2],
        "last_signal_date": summary_row[3],
        "valid_targets": int(summary_row[4]),
        "invalid_targets": int(summary_row[5]),
        "entry_one_price_limit_up": int(summary_row[6]),
        "exit_one_price_limit_down": int(summary_row[7]),
        "return_identity_errors": int(summary_row[8]),
        "target_status_counts": {name: int(rows) for name, rows in status_rows},
        "free_gb_before": round(free_gb_before, 2),
        "output_size_bytes": sum(
            path.stat().st_size for path in output_dir.rglob("*.parquet")
        ),
    }
    atomic_json(report, output_dir / "target_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"标签数据已保存到：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
