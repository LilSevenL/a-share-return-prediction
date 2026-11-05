"""将审计后的 Tushare 股票分片清洗为按接口/年份组织的 Parquet 数据集。

原始文件保持不变。默认读取 quality/raw_manifest.parquet，并优先使用
processed/repairs 中经过验证的修复文件。

小规模验证：
    python build_parquet.py --interfaces daily,fina_indicator,cashflow --max-stocks 20 \
        --output-dir quant_data/processed/parquet_test

全市场转换：
    python build_parquet.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocess_data import PROCESSED_DIR, REPORT_DIR, SPECS, InterfaceSpec, atomic_json, safe_code


DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "parquet"
BATCH_ROWS_PER_YEAR = 20_000
STRING_COLUMNS = {
    "ts_code", "symbol", "name", "ts_name", "area", "industry", "market", "exchange",
    "list_status", "curr_type", "is_hs", "act_name", "act_ent_type", "data_type",
    "concept", "rank_time", "rank_reason", "up_stat", "limit", "report_type",
    "comp_type", "end_type", "update_flag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interfaces", default=",".join(spec.name for spec in SPECS),
        help="逗号分隔接口名",
    )
    parser.add_argument("--manifest", default=str(REPORT_DIR / "raw_manifest.parquet"))
    parser.add_argument("--repairs-dir", default=str(PROCESSED_DIR / "repairs"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-stocks", type=int, help="每个接口只处理前 N 只，用于验证")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    return parser.parse_args()


def ensure_space(path: Path, minimum_gb: float) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(f"磁盘仅剩 {free_gb:.2f}GB，低于安全阈值 {minimum_gb:.2f}GB")
    return free_gb


def canonical_columns(subset: pd.DataFrame) -> list[str]:
    signatures = [value for value in subset["schema_signature"].dropna().astype(str) if value]
    if not signatures:
        raise RuntimeError("没有可用表头")
    signature = Counter(signatures).most_common(1)[0][0]
    return signature.split("|")


def preferred_path(row: Any, repairs_dir: Path) -> Path:
    repair = repairs_dir / row.interface / f"{safe_code(row.ts_code)}.csv.gz"
    return repair if repair.exists() else Path(row.path)


def normalize_types(frame: pd.DataFrame, columns: list[str], spec: InterfaceSpec) -> pd.DataFrame:
    result = frame.reindex(columns=columns).copy()
    for column in columns:
        if column in spec.date_columns:
            raw = result[column].astype("string").str.replace(".0", "", regex=False)
            result[column] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        elif column in STRING_COLUMNS:
            result[column] = result[column].astype("string")
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    return result


def clean_frame(
    frame: pd.DataFrame, expected_code: str, columns: list[str], spec: InterfaceSpec
) -> tuple[pd.DataFrame, dict[str, int]]:
    metrics = {
        "input_rows": len(frame),
        "code_mismatch_dropped": 0,
        "missing_primary_date_dropped": 0,
        "invalid_market_rows_dropped": 0,
        "duplicate_rows_dropped": 0,
    }
    result = normalize_types(frame, columns, spec)
    if "ts_code" in result.columns:
        valid_code = result["ts_code"].astype("string") == expected_code
        metrics["code_mismatch_dropped"] = int((~valid_code).sum())
        result = result[valid_code].copy()

    primary_date = "trade_date" if "trade_date" in spec.date_columns else (
        "ann_date" if "ann_date" in spec.date_columns else spec.date_columns[0]
    )
    missing_date = result[primary_date].isna()
    metrics["missing_primary_date_dropped"] = int(missing_date.sum())
    result = result[~missing_date].copy()

    if all(column in result.columns for column in ("open", "high", "low", "close")):
        positive = (result[["open", "high", "low", "close"]] > 0).all(axis=1)
        consistent = (
            (result["high"] >= result[["open", "low", "close"]].max(axis=1))
            & (result["low"] <= result[["open", "high", "close"]].min(axis=1))
        )
        valid_market = positive & consistent
        if "vol" in result.columns:
            valid_market &= result["vol"].isna() | (result["vol"] >= 0)
        if "amount" in result.columns:
            valid_market &= result["amount"].isna() | (result["amount"] >= 0)
        metrics["invalid_market_rows_dropped"] = int((~valid_market).sum())
        result = result[valid_market].copy()

    keys = [column for column in spec.unique_keys if column in result.columns]
    before = len(result)
    if keys:
        result["__non_null_count"] = result.notna().sum(axis=1)
        result["__source_order"] = range(len(result))
        result = result.sort_values(keys + ["__non_null_count", "__source_order"])
        result = result.drop_duplicates(keys, keep="last")
        result = result.drop(columns=["__non_null_count", "__source_order"])
        result = result.sort_values(keys).reset_index(drop=True)
    metrics["duplicate_rows_dropped"] = before - len(result)
    result["data_year"] = result[primary_date].dt.year.astype("int16")
    return result, metrics


def validate_dataset(path: Path, spec: InterfaceSpec) -> dict[str, Any]:
    glob_path = str(path / "year=*" / "*.parquet").replace("\\", "/")
    con = duckdb.connect()
    try:
        summary = con.execute(
            "SELECT count(*) AS rows, count(DISTINCT ts_code) AS stocks "
            "FROM read_parquet(?, hive_partitioning=true)", [glob_path]
        ).fetchone()
        return {"rows": int(summary[0]), "stocks": int(summary[1])}
    finally:
        con.close()


def convert_interface(
    spec: InterfaceSpec,
    manifest: pd.DataFrame,
    repairs_dir: Path,
    output_dir: Path,
    max_stocks: int | None,
    min_free_gb: float,
) -> dict[str, Any]:
    subset = manifest[(manifest["interface"] == spec.name) & (manifest["status"] == "data")].copy()
    subset = subset.sort_values("ts_code")
    if max_stocks:
        subset = subset.head(max_stocks)
    columns = canonical_columns(subset)
    final_dir = output_dir / spec.name
    if final_dir.exists():
        validation = validate_dataset(final_dir, spec)
        return {
            "interface": spec.name, "status": "existing", "input_files": len(subset),
            "output_rows": validation["rows"], "output_stocks": validation["stocks"],
        }

    staging_root = output_dir / "_staging"
    staging_dir = staging_root / spec.name
    if staging_dir.exists():
        stale = staging_root / f"{spec.name}.stale.{pd.Timestamp.now():%Y%m%d%H%M%S}"
        staging_dir.replace(stale)
    staging_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[int, pq.ParquetWriter] = {}
    output_paths: dict[int, Path] = {}
    pending_tables: dict[int, list[pa.Table]] = {}
    pending_rows: Counter[int] = Counter()
    totals = Counter()
    started_at = time.monotonic()

    def flush_year(year_value: int) -> None:
        tables = pending_tables.get(year_value, [])
        if not tables:
            return
        combined = pa.concat_tables(tables)
        parquet_path = output_paths[year_value]
        if year_value not in writers:
            writers[year_value] = pq.ParquetWriter(
                parquet_path,
                combined.schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        writers[year_value].write_table(
            combined, row_group_size=BATCH_ROWS_PER_YEAR
        )
        pending_tables[year_value] = []
        pending_rows[year_value] = 0

    try:
        for position, row in enumerate(subset.itertuples(index=False), start=1):
            ensure_space(output_dir, min_free_gb)
            source = preferred_path(row, repairs_dir)
            raw = pd.read_csv(source, low_memory=False)
            cleaned, metrics = clean_frame(raw, row.ts_code, columns, spec)
            totals.update(metrics)
            totals["processed_files"] += 1
            if not cleaned.empty:
                totals["output_rows"] += len(cleaned)
                totals["output_stocks"] += 1
                for year, group in cleaned.groupby("data_year", sort=True):
                    year_value = int(year)
                    table = pa.Table.from_pandas(group, preserve_index=False)
                    if year_value not in output_paths:
                        year_dir = staging_dir / f"year={year_value}"
                        year_dir.mkdir(parents=True, exist_ok=True)
                        parquet_path = year_dir / "part-00000.parquet"
                        output_paths[year_value] = parquet_path
                    pending_tables.setdefault(year_value, []).append(table)
                    pending_rows[year_value] += len(group)
                    if pending_rows[year_value] >= BATCH_ROWS_PER_YEAR:
                        flush_year(year_value)
            if position % 100 == 0 or position == len(subset):
                elapsed = time.monotonic() - started_at
                progress = {
                    "updated_at": pd.Timestamp.now().isoformat(),
                    "interface": spec.name,
                    "processed_files": position,
                    "total_files": len(subset),
                    "output_rows": int(totals["output_rows"]),
                    "rows_per_second": round(totals["input_rows"] / elapsed, 1) if elapsed else 0,
                }
                atomic_json(progress, output_dir / "conversion_progress.json")
                print(
                    f"[{spec.name}] {position:,}/{len(subset):,}，输出 {totals['output_rows']:,} 行",
                    flush=True,
                )
        for year_value in list(pending_tables):
            flush_year(year_value)
    finally:
        for writer in writers.values():
            writer.close()

    if not output_paths:
        raise RuntimeError(f"{spec.name} 清洗后没有可写数据")
    staging_dir.replace(final_dir)
    validation = validate_dataset(final_dir, spec)
    if validation["rows"] != totals["output_rows"]:
        raise RuntimeError(
            f"{spec.name} Parquet 行数校验失败：写入 {totals['output_rows']}，读取 {validation['rows']}"
        )
    result = {
        "interface": spec.name,
        "status": "converted",
        "input_files": len(subset),
        **{key: int(value) for key, value in totals.items()},
        "output_rows_verified": validation["rows"],
        "output_stocks_verified": validation["stocks"],
        "year_partitions": len(output_paths),
        "output_size_bytes": sum(path.stat().st_size for path in final_dir.rglob("*.parquet")),
    }
    return result


def main() -> None:
    args = parse_args()
    if args.max_stocks is not None and args.max_stocks <= 0:
        raise ValueError("max-stocks 必须大于 0")
    spec_map = {spec.name: spec for spec in SPECS}
    names = [name.strip() for name in args.interfaces.split(",") if name.strip()]
    unknown = sorted(set(names) - set(spec_map))
    if unknown:
        raise ValueError(f"未知接口：{', '.join(unknown)}")

    manifest = pd.read_parquet(Path(args.manifest))
    repairs_dir = Path(args.repairs_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_space(output_dir, args.min_free_gb)
    results: list[dict[str, Any]] = []
    for name in names:
        result = convert_interface(
            spec_map[name], manifest, repairs_dir, output_dir, args.max_stocks, args.min_free_gb
        )
        results.append(result)
        atomic_json(
            {"generated_at": pd.Timestamp.now().isoformat(), "interfaces": results},
            output_dir / "conversion_report.json",
        )
    print(json.dumps({"interfaces": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
