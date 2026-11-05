"""构建 50 个无未来数据泄漏的量化机器学习因子。

输入为 build_parquet.py 生成的清洗后 Parquet。输出按 feature_year 分区，
每行代表一只股票在一个信号日可获得的因子值。滚动价格和成交因子只使用
当日及以前的数据；财务因子只使用 ann_date 严格早于 signal_date 的公告。

小规模验证：
    python build_features.py --start-date 20250101 --end-date 20251231 \
        --max-stocks 20 --output-dir quant_data/processed/features_50_test --force

全市场构建：
    python build_features.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "quant_data" / "processed"
PARQUET_DIR = PROCESSED_DIR / "parquet"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "features_50"

# 固定顺序也是后续训练的特征契约。修改列表时必须同步修改 SQL。
FACTOR_COLUMNS = [
    # 动量与趋势（12）
    "logret_1d", "logret_2d", "logret_3d", "logret_5d", "logret_10d",
    "logret_20d", "logret_60d", "close_ma5_gap", "close_ma20_gap",
    "close_ma60_gap", "ma5_ma20_gap", "ma20_ma60_gap",
    # 风险与价格形态（12）
    "volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d",
    "downside_volatility_20d", "return_skew_20d", "worst_logret_20d",
    "intraday_logret", "overnight_logret", "high_low_range",
    "market_beta_60d", "market_corr_60d",
    # 流动性（7）
    "log_amount", "amount_ratio_5d", "amount_ratio_20d", "volume_ratio_20d",
    "turnover_rate", "turnover_rate_f", "turnover_zscore_20d",
    # 估值与规模（5）
    "log_total_mv", "log_circ_mv", "earnings_yield", "book_to_price",
    "sales_yield",
    # 资金流（6）
    "net_mf_ratio", "large_order_net_ratio", "extra_large_order_net_ratio",
    "medium_order_net_ratio", "small_order_net_ratio", "net_mf_ratio_5d",
    # 财务质量与成长（8）
    "roe", "roa", "grossprofit_margin", "debt_to_assets", "current_ratio",
    "netprofit_yoy", "or_yoy", "ocf_to_or",
]

if len(FACTOR_COLUMNS) != 50 or len(set(FACTOR_COLUMNS)) != 50:
    raise RuntimeError("FACTOR_COLUMNS 必须恰好包含 50 个不重复因子")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", default=str(PARQUET_DIR))
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


def quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parquet_glob(root: Path, interface: str) -> str:
    directory = root.expanduser().resolve() / interface
    if not directory.is_dir() or not next(directory.rglob("*.parquet"), None):
        raise FileNotFoundError(f"缺少已清洗的 {interface} Parquet：{directory}")
    return str(directory / "year=*" / "*.parquet").replace("\\", "/")


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
        print(f"未完成临时目录已保留为：{stale}", flush=True)
    staging.mkdir(parents=True)
    return staging


def create_views(
    connection: duckdb.DuckDBPyConnection, parquet_root: Path
) -> None:
    paths = {
        name: parquet_glob(parquet_root, name)
        for name in ("daily", "adj_factor", "daily_basic", "moneyflow", "fina_indicator")
    }
    connection.execute(
        f"""
        CREATE TEMP VIEW daily_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date,
               open, high, low, close, vol, amount
        FROM read_parquet({quote_sql(paths['daily'])}, hive_partitioning=true)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW adj_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date, adj_factor
        FROM read_parquet({quote_sql(paths['adj_factor'])}, hive_partitioning=true)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW basic_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date,
               turnover_rate, turnover_rate_f, pe_ttm, pb, ps_ttm,
               total_mv, circ_mv
        FROM read_parquet({quote_sql(paths['daily_basic'])}, hive_partitioning=true)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW moneyflow_data AS
        SELECT ts_code, CAST(trade_date AS DATE) AS trade_date,
               buy_sm_amount, sell_sm_amount, buy_md_amount, sell_md_amount,
               buy_lg_amount, sell_lg_amount, buy_elg_amount, sell_elg_amount,
               net_mf_amount
        FROM read_parquet({quote_sql(paths['moneyflow'])}, hive_partitioning=true)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW financial_data AS
        SELECT ts_code, CAST(ann_date AS DATE) AS ann_date,
               CAST(end_date AS DATE) AS end_date, roe, roa, grossprofit_margin,
               debt_to_assets, current_ratio, netprofit_yoy, or_yoy, ocf_to_or
        FROM read_parquet({quote_sql(paths['fina_indicator'])}, hive_partitioning=true)
        """
    )


def build_query(args: argparse.Namespace) -> str:
    final_filters: list[str] = []
    if args.start_date:
        final_filters.append(
            f"v.signal_date >= STRPTIME({quote_sql(args.start_date)}, '%Y%m%d')::DATE"
        )
    if args.end_date:
        final_filters.append(
            f"v.signal_date <= STRPTIME({quote_sql(args.end_date)}, '%Y%m%d')::DATE"
        )
    where_clause = " AND ".join(final_filters) if final_filters else "TRUE"
    sanitized_columns = ",\n".join(
        f"CASE WHEN ISFINITE({name}) THEN {name} END AS {name}"
        for name in FACTOR_COLUMNS
    )
    valid_count = " + ".join(
        f"CAST({name} IS NOT NULL AS INTEGER)" for name in FACTOR_COLUMNS
    )

    return f"""
    WITH base AS (
        SELECT
            d.ts_code,
            d.trade_date AS signal_date,
            c.market_index,
            d.open * a.adj_factor AS adj_open,
            d.high * a.adj_factor AS adj_high,
            d.low * a.adj_factor AS adj_low,
            d.close * a.adj_factor AS adj_close,
            d.vol,
            d.amount,
            b.turnover_rate,
            b.turnover_rate_f,
            b.pe_ttm,
            b.pb,
            b.ps_ttm,
            b.total_mv,
            b.circ_mv,
            10.0 * m.net_mf_amount / NULLIF(d.amount, 0) AS net_mf_ratio_raw,
            10.0 * ((m.buy_lg_amount - m.sell_lg_amount)
                  + (m.buy_elg_amount - m.sell_elg_amount))
                  / NULLIF(d.amount, 0) AS large_order_net_ratio_raw,
            10.0 * (m.buy_elg_amount - m.sell_elg_amount)
                  / NULLIF(d.amount, 0) AS extra_large_order_net_ratio_raw,
            10.0 * (m.buy_md_amount - m.sell_md_amount)
                  / NULLIF(d.amount, 0) AS medium_order_net_ratio_raw,
            10.0 * (m.buy_sm_amount - m.sell_sm_amount)
                  / NULLIF(d.amount, 0) AS small_order_net_ratio_raw
        FROM daily_work d
        JOIN market_calendar c ON c.trade_date = d.trade_date
        JOIN adj_work a ON a.ts_code = d.ts_code AND a.trade_date = d.trade_date
        LEFT JOIN basic_work b ON b.ts_code = d.ts_code AND b.trade_date = d.trade_date
        LEFT JOIN moneyflow_work m ON m.ts_code = d.ts_code AND m.trade_date = d.trade_date
        WHERE a.adj_factor > 0 AND d.open > 0 AND d.high > 0
          AND d.low > 0 AND d.close > 0
    ),
    price_values AS (
        SELECT *, LN(adj_close) AS log_adj_close
        FROM base
        WHERE adj_open > 0 AND adj_high > 0 AND adj_low > 0 AND adj_close > 0
    ),
    lagged AS (
        SELECT *,
            LAG(log_adj_close, 1) OVER w AS lag_log_1,
            LAG(log_adj_close, 2) OVER w AS lag_log_2,
            LAG(log_adj_close, 3) OVER w AS lag_log_3,
            LAG(log_adj_close, 5) OVER w AS lag_log_5,
            LAG(log_adj_close, 10) OVER w AS lag_log_10,
            LAG(log_adj_close, 20) OVER w AS lag_log_20,
            LAG(log_adj_close, 60) OVER w AS lag_log_60,
            LAG(adj_close, 1) OVER w AS lag_close_1,
            LAG(market_index, 1) OVER w AS lag_index_1,
            LAG(market_index, 2) OVER w AS lag_index_2,
            LAG(market_index, 3) OVER w AS lag_index_3,
            LAG(market_index, 5) OVER w AS lag_index_5,
            LAG(market_index, 10) OVER w AS lag_index_10,
            LAG(market_index, 20) OVER w AS lag_index_20,
            LAG(market_index, 60) OVER w AS lag_index_60
        FROM price_values
        WINDOW w AS (PARTITION BY ts_code ORDER BY market_index)
    ),
    returns AS (
        SELECT *,
            CASE WHEN market_index - lag_index_1 = 1 THEN log_adj_close - lag_log_1 END AS logret_1d,
            CASE WHEN market_index - lag_index_2 = 2 THEN log_adj_close - lag_log_2 END AS logret_2d,
            CASE WHEN market_index - lag_index_3 = 3 THEN log_adj_close - lag_log_3 END AS logret_3d,
            CASE WHEN market_index - lag_index_5 = 5 THEN log_adj_close - lag_log_5 END AS logret_5d,
            CASE WHEN market_index - lag_index_10 = 10 THEN log_adj_close - lag_log_10 END AS logret_10d,
            CASE WHEN market_index - lag_index_20 = 20 THEN log_adj_close - lag_log_20 END AS logret_20d,
            CASE WHEN market_index - lag_index_60 = 60 THEN log_adj_close - lag_log_60 END AS logret_60d,
            LN(adj_close / adj_open) AS intraday_logret,
            CASE WHEN market_index - lag_index_1 = 1 THEN LN(adj_open / lag_close_1) END AS overnight_logret,
            (adj_high - adj_low) / NULLIF(adj_open, 0) AS high_low_range
        FROM lagged
    ),
    market_returns AS (
        SELECT signal_date, AVG(logret_1d) AS market_logret_1d
        FROM returns GROUP BY signal_date
    ),
    enriched AS (
        SELECT r.*, mr.market_logret_1d
        FROM returns r LEFT JOIN market_returns mr USING (signal_date)
    ),
    technical_values AS (
        SELECT
            ts_code, signal_date, market_index, logret_1d, logret_2d, logret_3d,
            logret_5d, logret_10d, logret_20d, logret_60d,
            CASE WHEN COUNT(adj_close) OVER w5 = 5
                       AND MAX(market_index) OVER w5 - MIN(market_index) OVER w5 = 4
                 THEN adj_close / AVG(adj_close) OVER w5 - 1.0 END AS close_ma5_gap,
            CASE WHEN COUNT(adj_close) OVER w20 = 20
                       AND MAX(market_index) OVER w20 - MIN(market_index) OVER w20 = 19
                 THEN adj_close / AVG(adj_close) OVER w20 - 1.0 END AS close_ma20_gap,
            CASE WHEN COUNT(adj_close) OVER w60 = 60
                       AND MAX(market_index) OVER w60 - MIN(market_index) OVER w60 = 59
                 THEN adj_close / AVG(adj_close) OVER w60 - 1.0 END AS close_ma60_gap,
            CASE WHEN COUNT(adj_close) OVER w20 = 20
                       AND MAX(market_index) OVER w20 - MIN(market_index) OVER w20 = 19
                 THEN AVG(adj_close) OVER w5 / NULLIF(AVG(adj_close) OVER w20, 0) - 1.0 END AS ma5_ma20_gap,
            CASE WHEN COUNT(adj_close) OVER w60 = 60
                       AND MAX(market_index) OVER w60 - MIN(market_index) OVER w60 = 59
                 THEN AVG(adj_close) OVER w20 / NULLIF(AVG(adj_close) OVER w60, 0) - 1.0 END AS ma20_ma60_gap,
            CASE WHEN COUNT(logret_1d) OVER w5 = 5 THEN STDDEV_SAMP(logret_1d) OVER w5 END AS volatility_5d,
            CASE WHEN COUNT(logret_1d) OVER w10 = 10 THEN STDDEV_SAMP(logret_1d) OVER w10 END AS volatility_10d,
            CASE WHEN COUNT(logret_1d) OVER w20 = 20 THEN STDDEV_SAMP(logret_1d) OVER w20 END AS volatility_20d,
            CASE WHEN COUNT(logret_1d) OVER w60 = 60 THEN STDDEV_SAMP(logret_1d) OVER w60 END AS volatility_60d,
            CASE WHEN COUNT(logret_1d) OVER w20 = 20
                 THEN SQRT(AVG(CASE WHEN logret_1d < 0 THEN POW(logret_1d, 2) ELSE 0 END) OVER w20)
                 END AS downside_volatility_20d,
            CASE WHEN COUNT(logret_1d) OVER w20 = 20 THEN SKEWNESS(logret_1d) OVER w20 END AS return_skew_20d,
            CASE WHEN COUNT(logret_1d) OVER w20 = 20 THEN MIN(logret_1d) OVER w20 END AS worst_logret_20d,
            intraday_logret, overnight_logret, high_low_range,
            CASE WHEN COUNT(logret_1d) OVER w60 = 60
                 THEN REGR_SLOPE(logret_1d, market_logret_1d) OVER w60 END AS market_beta_60d,
            CASE WHEN COUNT(logret_1d) OVER w60 = 60
                 THEN CORR(logret_1d, market_logret_1d) OVER w60 END AS market_corr_60d,
            CASE WHEN amount > 0 THEN LN(amount) END AS log_amount,
            CASE WHEN COUNT(amount) OVER w5 = 5
                 THEN amount / NULLIF(AVG(amount) OVER w5, 0) - 1.0 END AS amount_ratio_5d,
            CASE WHEN COUNT(amount) OVER w20 = 20
                 THEN amount / NULLIF(AVG(amount) OVER w20, 0) - 1.0 END AS amount_ratio_20d,
            CASE WHEN COUNT(vol) OVER w20 = 20
                 THEN vol / NULLIF(AVG(vol) OVER w20, 0) - 1.0 END AS volume_ratio_20d,
            turnover_rate, turnover_rate_f,
            CASE WHEN COUNT(turnover_rate) OVER w20 = 20
                 THEN (turnover_rate - AVG(turnover_rate) OVER w20)
                      / NULLIF(STDDEV_SAMP(turnover_rate) OVER w20, 0) END AS turnover_zscore_20d,
            CASE WHEN total_mv > 0 THEN LN(total_mv) END AS log_total_mv,
            CASE WHEN circ_mv > 0 THEN LN(circ_mv) END AS log_circ_mv,
            CASE WHEN ABS(pe_ttm) > 1e-12 THEN 1.0 / pe_ttm END AS earnings_yield,
            CASE WHEN ABS(pb) > 1e-12 THEN 1.0 / pb END AS book_to_price,
            CASE WHEN ABS(ps_ttm) > 1e-12 THEN 1.0 / ps_ttm END AS sales_yield,
            net_mf_ratio_raw AS net_mf_ratio,
            large_order_net_ratio_raw AS large_order_net_ratio,
            extra_large_order_net_ratio_raw AS extra_large_order_net_ratio,
            medium_order_net_ratio_raw AS medium_order_net_ratio,
            small_order_net_ratio_raw AS small_order_net_ratio,
            CASE WHEN COUNT(net_mf_ratio_raw) OVER w5 = 5
                 THEN AVG(net_mf_ratio_raw) OVER w5 END AS net_mf_ratio_5d
        FROM enriched
        WINDOW
            w5 AS (PARTITION BY ts_code ORDER BY market_index ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
            w10 AS (PARTITION BY ts_code ORDER BY market_index ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
            w20 AS (PARTITION BY ts_code ORDER BY market_index ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
            w60 AS (PARTITION BY ts_code ORDER BY market_index ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
    ),
    financial_events AS (
        SELECT * EXCLUDE (event_rank)
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ts_code, ann_date ORDER BY end_date DESC
            ) AS event_rank
            FROM financial_work
            WHERE ann_date IS NOT NULL
        )
        WHERE event_rank = 1
    ),
    factor_values AS (
        SELECT
            t.* EXCLUDE (market_index),
            f.ann_date AS financial_ann_date,
            f.end_date AS financial_end_date,
            f.roe, f.roa, f.grossprofit_margin, f.debt_to_assets,
            f.current_ratio, f.netprofit_yoy, f.or_yoy, f.ocf_to_or
        FROM technical_values t
        ASOF LEFT JOIN financial_events f
          ON t.ts_code = f.ts_code AND t.signal_date > f.ann_date
    ),
    sanitized AS (
        SELECT
            ts_code, signal_date, financial_ann_date, financial_end_date,
            {sanitized_columns}
        FROM factor_values
    )
    SELECT
        *,
        ({valid_count}) AS feature_valid_count,
        CAST(EXTRACT(YEAR FROM signal_date) AS SMALLINT) AS feature_year
    FROM sanitized v
    WHERE {where_clause}
    """


def validate_output(
    connection: duckdb.DuckDBPyConnection, output_glob: str
) -> dict[str, Any]:
    factor_counts = ", ".join(
        f"COUNT({name}) AS {name}" for name in FACTOR_COLUMNS
    )
    summary = connection.execute(
        f"""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT ts_code) AS stocks,
               MIN(signal_date) AS first_date, MAX(signal_date) AS last_date,
               COUNT(*) - COUNT(DISTINCT (ts_code, signal_date)) AS duplicate_keys,
               COUNT(*) FILTER (
                   WHERE financial_ann_date IS NOT NULL
                     AND financial_ann_date >= signal_date
               ) AS financial_lookahead_rows,
               {factor_counts}
        FROM read_parquet(?, hive_partitioning=true)
        """,
        [output_glob],
    ).fetchone()
    names = [item[0] for item in connection.description]
    values = dict(zip(names, summary))
    if values["rows"] == 0:
        raise RuntimeError("没有生成任何因子记录")
    if int(values["duplicate_keys"]) != 0:
        raise RuntimeError(f"输出存在重复股票日期键：{values['duplicate_keys']}")
    if int(values["financial_lookahead_rows"]) != 0:
        raise RuntimeError(
            f"检测到财务公告日期穿越：{values['financial_lookahead_rows']} 行"
        )
    coverage = {
        name: round(int(values[name]) / int(values["rows"]), 6)
        for name in FACTOR_COLUMNS
    }
    return {
        "rows": int(values["rows"]),
        "stocks": int(values["stocks"]),
        "first_date": values["first_date"],
        "last_date": values["last_date"],
        "duplicate_keys": int(values["duplicate_keys"]),
        "financial_lookahead_rows": int(values["financial_lookahead_rows"]),
        "factor_coverage": coverage,
        "all_null_factors": [name for name, ratio in coverage.items() if ratio == 0],
    }


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

    output_dir = Path(args.output_dir).expanduser().resolve()
    free_gb_before = ensure_space(output_dir, args.min_free_gb)
    staging = prepare_destination(output_dir, args.force)
    database_path = staging.parent / f".{output_dir.name}.duckdb"
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(f"SET threads = {int(args.threads)}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET temp_directory = {quote_sql(str(staging / '_tmp'))}")
        connection.execute(f"SET memory_limit = {quote_sql(args.memory_limit)}")
        create_views(connection, Path(args.parquet_dir))
        stock_limit = f"LIMIT {int(args.max_stocks)}" if args.max_stocks else ""
        connection.execute(
            f"""
            CREATE TEMP TABLE selected_stocks AS
            SELECT DISTINCT ts_code FROM daily_data ORDER BY ts_code {stock_limit}
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE market_calendar AS
            SELECT trade_date,
                   ROW_NUMBER() OVER (ORDER BY trade_date) AS market_index
            FROM (SELECT DISTINCT trade_date FROM daily_data)
            """
        )
        # 先将所有输入裁剪到目标股票和必要回看区间，避免复杂窗口查询
        # 在连接阶段反复扫描全市场 Parquet。
        history_filter = "TRUE"
        if args.start_date:
            history_start = (
                datetime.strptime(args.start_date, "%Y%m%d") - timedelta(days=180)
            ).strftime("%Y%m%d")
            history_filter = (
                f"x.trade_date >= STRPTIME({quote_sql(history_start)}, '%Y%m%d')::DATE"
            )
        for source, work in (
            ("daily_data", "daily_work"),
            ("adj_data", "adj_work"),
            ("basic_data", "basic_work"),
            ("moneyflow_data", "moneyflow_work"),
        ):
            connection.execute(
                f"""
                CREATE TEMP TABLE {work} AS
                SELECT x.* FROM {source} x
                JOIN selected_stocks u ON u.ts_code = x.ts_code
                WHERE {history_filter}
                """
            )
            connection.execute(f"ANALYZE {work}")
        connection.execute(
            """
            CREATE TEMP TABLE financial_work AS
            SELECT x.* FROM financial_data x
            JOIN selected_stocks u ON u.ts_code = x.ts_code
            """
        )
        connection.execute("ANALYZE selected_stocks")
        connection.execute("ANALYZE market_calendar")
        connection.execute("ANALYZE financial_work")
        query = build_query(args)
        print("正在构建 50 个量化因子……", flush=True)
        connection.execute(
            f"""
            COPY ({query}) TO {quote_sql(str(staging).replace(chr(92), '/'))}
            (FORMAT PARQUET, PARTITION_BY (feature_year), COMPRESSION ZSTD,
             ROW_GROUP_SIZE 100000, OVERWRITE_OR_IGNORE TRUE)
            """
        )
        output_glob = str(staging / "feature_year=*" / "*.parquet").replace("\\", "/")
        validation = validate_output(connection, output_glob)
    except Exception:
        print(f"因子构建失败；临时结果保留在：{staging}", flush=True)
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    if validation["all_null_factors"]:
        raise RuntimeError(
            "以下因子全部为空：" + ", ".join(validation["all_null_factors"])
        )
    staging.replace(output_dir)
    report = {
        "generated_at": datetime.now().isoformat(),
        "factor_count": len(FACTOR_COLUMNS),
        "factor_columns": FACTOR_COLUMNS,
        "signal_start_date": args.start_date,
        "signal_end_date": args.end_date,
        "max_stocks": args.max_stocks,
        **validation,
        "free_gb_before": round(free_gb_before, 2),
        "output_size_bytes": sum(
            path.stat().st_size for path in output_dir.rglob("*.parquet")
        ),
    }
    atomic_json(report, output_dir / "feature_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"50因子数据已保存到：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
