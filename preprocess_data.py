"""Tushare 全市场原始数据的质量审计与预处理入口。

本脚本不会修改 quant_data/cache、extra_cache 或 combined 中的原始文件。

快速抽样审计：
    python preprocess_data.py audit --sample-per-interface 100

全量行级审计：
    python preprocess_data.py audit --sample-per-interface 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "quant_data"
PROCESSED_DIR = RAW_DIR / "processed"
REPORT_DIR = PROCESSED_DIR / "quality"


@dataclass(frozen=True)
class InterfaceSpec:
    name: str
    cache_group: str
    unique_keys: tuple[str, ...]
    date_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


SPECS = (
    InterfaceSpec("daily", "cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")),
    InterfaceSpec("adj_factor", "cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "adj_factor")),
    InterfaceSpec("daily_basic", "cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "close", "total_mv", "circ_mv")),
    InterfaceSpec("moneyflow", "cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "net_mf_vol", "net_mf_amount")),
    InterfaceSpec("stk_limit", "cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "up_limit", "down_limit")),
    InterfaceSpec("fina_indicator", "cache", ("ts_code", "ann_date", "end_date"),
                  ("ann_date", "end_date"), ("ts_code", "ann_date", "end_date")),
    InterfaceSpec("weekly", "extra_cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")),
    InterfaceSpec("monthly", "extra_cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")),
    InterfaceSpec("income", "extra_cache",
                  ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
                  ("ann_date", "f_ann_date", "end_date"), ("ts_code", "ann_date", "end_date")),
    InterfaceSpec("balancesheet", "extra_cache",
                  ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
                  ("ann_date", "f_ann_date", "end_date"), ("ts_code", "ann_date", "end_date")),
    InterfaceSpec("cashflow", "extra_cache",
                  ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
                  ("ann_date", "f_ann_date", "end_date"), ("ts_code", "ann_date", "end_date")),
    InterfaceSpec("limit_list_d", "extra_cache", ("ts_code", "trade_date", "limit"),
                  ("trade_date",), ("ts_code", "trade_date", "limit")),
    InterfaceSpec("ths_hot", "extra_cache", ("trade_date", "data_type", "ts_code", "rank_time"),
                  ("trade_date",), ("ts_code", "trade_date", "rank")),
    InterfaceSpec("stk_factor_pro", "extra_cache", ("ts_code", "trade_date"), ("trade_date",),
                  ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="生成结构和行级质量报告")
    audit.add_argument(
        "--sample-per-interface",
        type=int,
        default=100,
        help="每接口抽样文件数；0 表示检查全部文件",
    )
    audit.add_argument("--raw-dir", default=str(RAW_DIR), help="quant_data 根目录")
    audit.add_argument("--output-dir", default=str(REPORT_DIR), help="质量报告输出目录")
    repair = subparsers.add_parser("repair", help="补采审计发现的代码错配文件")
    repair.add_argument("--start-date", default="20100101", help="开始日期 YYYYMMDD")
    repair.add_argument("--end-date", default="20260827", help="结束日期 YYYYMMDD")
    repair.add_argument(
        "--output-dir", default=str(PROCESSED_DIR / "repairs"), help="修复数据输出目录"
    )
    return parser.parse_args()


def atomic_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    for attempt in range(1, 6):
        try:
            temp.write_text(payload, encoding="utf-8")
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(attempt * 0.2)


def atomic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)
    temp.replace(path)


def safe_code(ts_code: str) -> str:
    return ts_code.replace(".", "_")


def read_header(path: Path) -> tuple[str, ...]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return tuple(handle.readline().strip().split(","))


def load_universe(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "combined" / "stock_basic.csv"
    frame = pd.read_csv(path, dtype={"ts_code": str, "list_status": str, "list_date": str})
    if "list_status" in frame.columns:
        frame = frame[frame["list_status"] == "L"].copy()
    frame = frame.drop_duplicates("ts_code", keep="last").sort_values("ts_code")
    if frame.empty:
        raise RuntimeError("stock_basic.csv 中没有当前上市股票")
    return frame.reset_index(drop=True)


def discover_manifest(raw_dir: Path, universe: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    structure_issues: list[dict[str, Any]] = []
    for spec in SPECS:
        folder = raw_dir / spec.cache_group / spec.name
        schema_counter: Counter[tuple[str, ...]] = Counter()
        for ts_code in universe["ts_code"]:
            code = safe_code(ts_code)
            gzip_path = folder / f"{code}.csv.gz"
            csv_path = folder / f"{code}.csv"
            empty_path = folder / f"{code}.empty.json"
            existing_data = [path for path in (gzip_path, csv_path) if path.exists()]
            chosen = gzip_path if gzip_path.exists() else csv_path if csv_path.exists() else None
            status = "data" if chosen else "empty" if empty_path.exists() else "missing"
            columns: tuple[str, ...] = ()
            size = 0
            modified = None
            if chosen:
                try:
                    columns = read_header(chosen)
                    schema_counter[columns] += 1
                    size = chosen.stat().st_size
                    modified = pd.Timestamp(chosen.stat().st_mtime, unit="s").isoformat()
                except Exception as exc:
                    status = "unreadable"
                    structure_issues.append(
                        {"interface": spec.name, "ts_code": ts_code, "kind": "header_read_error", "detail": str(exc)}
                    )
            if len(existing_data) > 1:
                structure_issues.append(
                    {"interface": spec.name, "ts_code": ts_code, "kind": "duplicate_cache_variant",
                     "detail": ";".join(str(path) for path in existing_data)}
                )
            records.append(
                {
                    "interface": spec.name,
                    "cache_group": spec.cache_group,
                    "ts_code": ts_code,
                    "status": status,
                    "path": str(chosen.resolve()) if chosen else "",
                    "format": "csv.gz" if chosen and chosen.name.endswith(".gz") else "csv" if chosen else "",
                    "size_bytes": size,
                    "modified_at": modified,
                    "column_count": len(columns),
                    "schema_signature": "|".join(columns),
                }
            )
        if len(schema_counter) > 1:
            structure_issues.append(
                {"interface": spec.name, "ts_code": "*", "kind": "schema_variants",
                 "detail": f"发现 {len(schema_counter)} 种表头"}
            )
    return pd.DataFrame(records), structure_issues


def choose_sample(paths: list[Path], limit: int) -> list[Path]:
    if limit == 0 or len(paths) <= limit:
        return paths
    positions = sorted({round(index * (len(paths) - 1) / (limit - 1)) for index in range(limit)})
    return [paths[position] for position in positions]


def expected_code_from_path(path: Path) -> str:
    name = path.name.split(".csv", 1)[0]
    if name.endswith("_SH"):
        return name[:-3] + ".SH"
    if name.endswith("_SZ"):
        return name[:-3] + ".SZ"
    if name.endswith("_BJ"):
        return name[:-3] + ".BJ"
    return name


def audit_file(path: Path, spec: InterfaceSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    expected_code = expected_code_from_path(path)
    header = read_header(path)
    audit_columns = set(spec.unique_keys) | set(spec.date_columns) | set(spec.required_columns)
    audit_columns.update(("ts_code", "open", "high", "low", "close", "pre_close", "vol", "amount"))
    selected_columns = [column for column in header if column in audit_columns]
    frame = pd.read_csv(
        path,
        usecols=selected_columns,
        low_memory=False,
        dtype={column: "string" for column in spec.date_columns if column in selected_columns},
    )
    result: dict[str, Any] = {
        "interface": spec.name,
        "path": str(path),
        "rows": len(frame),
        "duplicate_key_rows": 0,
        "invalid_date_values": 0,
        "code_mismatch_rows": 0,
        "ohlc_invalid_rows": 0,
        "negative_volume_rows": 0,
        "infinite_numeric_values": 0,
        "min_date": None,
        "max_date": None,
    }
    missing_columns = [column for column in spec.required_columns if column not in header]
    if missing_columns:
        issues.append(
            {"interface": spec.name, "ts_code": expected_code, "kind": "missing_columns",
             "detail": ",".join(missing_columns), "path": str(path)}
        )

    keys = [column for column in spec.unique_keys if column in frame.columns]
    if len(keys) == len(spec.unique_keys):
        duplicate_rows = int(frame.duplicated(keys, keep=False).sum())
        result["duplicate_key_rows"] = duplicate_rows
        if duplicate_rows:
            issues.append(
                {"interface": spec.name, "ts_code": expected_code, "kind": "duplicate_keys",
                 "detail": str(duplicate_rows), "path": str(path)}
            )

    if "ts_code" in frame.columns:
        codes = frame["ts_code"].dropna().astype(str)
        mismatch = int((codes != expected_code).sum())
        result["code_mismatch_rows"] = mismatch
        if mismatch:
            issues.append(
                {"interface": spec.name, "ts_code": expected_code, "kind": "code_mismatch",
                 "detail": str(mismatch), "path": str(path)}
            )

    parsed_dates: list[pd.Series] = []
    for column in spec.date_columns:
        if column not in frame.columns:
            continue
        raw = frame[column].astype("string").str.replace(".0", "", regex=False)
        nonempty = raw.notna() & (raw.str.len() > 0)
        parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        invalid = int((nonempty & parsed.isna()).sum())
        result["invalid_date_values"] += invalid
        parsed_dates.append(parsed)
    if parsed_dates:
        combined_dates = pd.concat(parsed_dates, ignore_index=True).dropna()
        if not combined_dates.empty:
            result["min_date"] = combined_dates.min().date().isoformat()
            result["max_date"] = combined_dates.max().date().isoformat()
    if result["invalid_date_values"]:
        issues.append(
            {"interface": spec.name, "ts_code": expected_code, "kind": "invalid_dates",
             "detail": str(result["invalid_date_values"]), "path": str(path)}
        )

    numeric_candidates = [
        column for column in ("open", "high", "low", "close", "pre_close", "vol", "amount")
        if column in frame.columns
    ]
    numeric: dict[str, pd.Series] = {
        column: pd.to_numeric(frame[column], errors="coerce") for column in numeric_candidates
    }
    infinite = sum(int((series == math.inf).sum() + (series == -math.inf).sum()) for series in numeric.values())
    result["infinite_numeric_values"] = infinite
    if all(column in numeric for column in ("open", "high", "low", "close")):
        max_body = pd.concat([numeric["open"], numeric["close"], numeric["low"]], axis=1).max(axis=1)
        min_body = pd.concat([numeric["open"], numeric["close"], numeric["high"]], axis=1).min(axis=1)
        invalid_ohlc = int(((numeric["high"] < max_body) | (numeric["low"] > min_body)).fillna(False).sum())
        result["ohlc_invalid_rows"] = invalid_ohlc
        if invalid_ohlc:
            issues.append(
                {"interface": spec.name, "ts_code": expected_code, "kind": "ohlc_inconsistent",
                 "detail": str(invalid_ohlc), "path": str(path)}
            )
    negative_volume = 0
    for column in ("vol", "amount"):
        if column in numeric:
            negative_volume += int((numeric[column] < 0).fillna(False).sum())
    result["negative_volume_rows"] = negative_volume
    if negative_volume:
        issues.append(
            {"interface": spec.name, "ts_code": expected_code, "kind": "negative_volume_or_amount",
             "detail": str(negative_volume), "path": str(path)}
        )
    return result, issues


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Tushare 全市场数据质量报告",
        "",
        f"生成时间：{report['generated_at']}",
        f"股票数量：{report['stock_count']:,}",
        f"接口数量：{report['interface_count']}",
        f"审计模式：{report['audit_mode']}",
        "",
        "## 结构完整性",
        "",
        "| 接口 | 数据文件 | 空数据 | 缺失 | 不可读 | 大小GB | 抽查文件 | 抽查行数 | 问题数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["interfaces"]:
        lines.append(
            f"| {row['interface']} | {row['data_files']:,} | {row['empty_markers']:,} | "
            f"{row['missing_files']:,} | {row['unreadable_files']:,} | {row['size_gb']:.3f} | "
            f"{row['audited_files']:,} | {row['audited_rows']:,} | {row['issue_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 结构问题：{report['structure_issue_count']:,} 条。",
            f"- 行级问题：{report['row_issue_count']:,} 条。",
            "- 原始文件未修改；后续清洗结果将写入 quant_data/processed。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(raw_dir)
    print(f"股票列表：{len(universe):,} 只")
    manifest, structure_issues = discover_manifest(raw_dir, universe)
    manifest_path = output_dir / "raw_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False, compression="zstd")
    print(f"原始清单：{len(manifest):,} 条 -> {manifest_path}")

    interface_reports: list[dict[str, Any]] = []
    all_row_issues: list[dict[str, Any]] = []
    all_file_results: list[dict[str, Any]] = []
    audit_mode = "full" if args.sample_per_interface == 0 else f"sample_{args.sample_per_interface}"
    for spec in SPECS:
        subset = manifest[manifest["interface"] == spec.name]
        paths = [Path(value) for value in subset.loc[subset["status"] == "data", "path"]]
        selected = choose_sample(paths, args.sample_per_interface)
        print(f"[{spec.name}] 审计 {len(selected):,}/{len(paths):,} 个数据文件", flush=True)
        spec_results: list[dict[str, Any]] = []
        spec_issues: list[dict[str, Any]] = []
        for position, path in enumerate(selected, start=1):
            try:
                result, issues = audit_file(path, spec)
                spec_results.append(result)
                spec_issues.extend(issues)
            except Exception as exc:
                spec_issues.append(
                    {"interface": spec.name, "ts_code": expected_code_from_path(path),
                     "kind": "file_read_error", "detail": str(exc), "path": str(path)}
                )
            if position % 500 == 0:
                print(f"  {position:,}/{len(selected):,}", flush=True)
        all_file_results.extend(spec_results)
        all_row_issues.extend(spec_issues)
        status_counts = subset["status"].value_counts().to_dict()
        interface_reports.append(
            {
                "interface": spec.name,
                "data_files": int(status_counts.get("data", 0)),
                "empty_markers": int(status_counts.get("empty", 0)),
                "missing_files": int(status_counts.get("missing", 0)),
                "unreadable_files": int(status_counts.get("unreadable", 0)),
                "size_gb": float(subset["size_bytes"].sum() / (1024**3)),
                "schema_variant_count": int(subset.loc[subset["status"] == "data", "schema_signature"].nunique()),
                "audited_files": len(spec_results),
                "audited_rows": int(sum(result["rows"] for result in spec_results)),
                "duplicate_key_rows": int(sum(result["duplicate_key_rows"] for result in spec_results)),
                "invalid_date_values": int(sum(result["invalid_date_values"] for result in spec_results)),
                "code_mismatch_rows": int(sum(result["code_mismatch_rows"] for result in spec_results)),
                "ohlc_invalid_rows": int(sum(result["ohlc_invalid_rows"] for result in spec_results)),
                "negative_volume_rows": int(sum(result["negative_volume_rows"] for result in spec_results)),
                "issue_count": len(spec_issues),
            }
        )
        atomic_json(
            {"updated_at": pd.Timestamp.now().isoformat(), "audit_mode": audit_mode,
             "completed_interfaces": [row["interface"] for row in interface_reports],
             "current_interface": spec.name},
            output_dir / "audit_progress.json",
        )

    issues = structure_issues + all_row_issues
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "as_of_date": date.today().isoformat(),
        "raw_dir": str(raw_dir),
        "stock_count": len(universe),
        "interface_count": len(SPECS),
        "expected_partitions": len(universe) * len(SPECS),
        "audit_mode": audit_mode,
        "structure_issue_count": len(structure_issues),
        "row_issue_count": len(all_row_issues),
        "interfaces": interface_reports,
    }
    atomic_json(report, output_dir / "quality_report.json")
    atomic_json(issues, output_dir / "quality_issues.json")
    pd.DataFrame(all_file_results).to_parquet(
        output_dir / "audited_file_metrics.parquet", index=False, compression="zstd"
    )
    write_markdown(report, output_dir / "quality_report.md")
    print(f"质量报告：{output_dir / 'quality_report.md'}")
    print(f"问题明细：{len(issues):,} 条")


def run_repair(args: argparse.Namespace) -> None:
    import main as base

    output_dir = Path(args.output_dir).expanduser().resolve()
    repairs = (
        ("fina_indicator", "300354.SZ", {}),
        ("cashflow", "301138.SZ", {"limit": 10_000}),
    )
    pro = base.create_pro_client()
    results: list[dict[str, Any]] = []
    for api_name, ts_code, extra_params in repairs:
        frame = getattr(pro, api_name)(
            ts_code=ts_code,
            start_date=args.start_date,
            end_date=args.end_date,
            **extra_params,
        )
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise RuntimeError(f"{api_name}/{ts_code} 修复请求返回空表")
        returned_codes = sorted(frame["ts_code"].dropna().astype(str).unique())
        if returned_codes != [ts_code]:
            raise RuntimeError(
                f"{api_name}/{ts_code} 仍返回错配代码：{returned_codes[:10]}"
            )
        target = output_dir / api_name / f"{safe_code(ts_code)}.csv.gz"
        atomic_gzip_csv(frame, target)
        results.append(
            {
                "interface": api_name,
                "ts_code": ts_code,
                "rows": len(frame),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "path": str(target),
                "validated_unique_codes": returned_codes,
            }
        )
        time.sleep(max(0.2, base.REQUEST_INTERVAL_SECONDS))
    report = {"generated_at": pd.Timestamp.now().isoformat(), "repairs": results}
    atomic_json(report, output_dir / "repair_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        if args.sample_per_interface < 0:
            raise ValueError("sample-per-interface 不能小于 0")
        run_audit(args)
    elif args.command == "repair":
        run_repair(args)


if __name__ == "__main__":
    main()
