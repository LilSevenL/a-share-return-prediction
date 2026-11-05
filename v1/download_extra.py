"""下载常用 Tushare 接口的补充数据。

依赖与代理客户端复用 main.py。默认只补充尚未下载的接口，不重复请求
daily、daily_basic、index_daily、moneyflow 和 stk_limit。

运行：
    python download_extra.py
    python download_extra.py --start-date 20150101 --end-date 20261231
    python download_extra.py --output-dir "F:\\quant_data"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import main as base


REQUEST_INTERVAL_SECONDS = max(0.2, base.REQUEST_INTERVAL_SECONDS)
MAX_RETRIES = 3


@dataclass(frozen=True)
class PerStockSpec:
    api_name: str
    description: str
    unique_keys: tuple[str, ...]
    extra_params: tuple[tuple[str, Any], ...] = ()


PER_STOCK_SPECS = [
    PerStockSpec("weekly", "A 股周线行情", ("ts_code", "trade_date")),
    PerStockSpec("monthly", "A 股月线行情", ("ts_code", "trade_date")),
    PerStockSpec(
        "income",
        "利润表",
        ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
        (("limit", 10_000),),
    ),
    PerStockSpec(
        "balancesheet",
        "资产负债表",
        ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
        (("limit", 10_000),),
    ),
    PerStockSpec(
        "cashflow",
        "现金流量表",
        ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
        (("limit", 10_000),),
    ),
    PerStockSpec(
        "limit_list_d",
        "涨跌停和炸板历史",
        ("ts_code", "trade_date", "limit"),
    ),
    PerStockSpec(
        "ths_hot",
        "同花顺热榜历史",
        ("trade_date", "data_type", "ts_code", "rank_time"),
    ),
    PerStockSpec(
        "stk_factor_pro",
        "专业量化因子",
        ("ts_code", "trade_date"),
    ),
]

QUALITY_KEYS = {
    "stock_basic": ("ts_code", "list_status"),
    "trade_cal": ("exchange", "cal_date"),
    "us_tradecal": ("cal_date",),
    "weekly": ("ts_code", "trade_date"),
    "monthly": ("ts_code", "trade_date"),
    "income": (
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"
    ),
    "balancesheet": (
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"
    ),
    "cashflow": (
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"
    ),
    "limit_list_d": ("ts_code", "trade_date", "limit"),
    "ths_hot": ("trade_date", "data_type", "ts_code", "rank_time"),
    "stk_factor_pro": ("ts_code", "trade_date"),
    "broker_recommend": ("month", "broker", "ts_code"),
    "cyq_chips": ("ts_code", "trade_date", "price"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20100101", help="开始日期 YYYYMMDD")
    parser.add_argument(
        "--end-date", default=date.today().strftime("%Y%m%d"), help="结束日期 YYYYMMDD"
    )
    parser.add_argument(
        "--output-dir",
        default=str(base.DEFAULT_DATA_DIR),
        help="输出目录，默认是项目下的 quant_data",
    )
    parser.add_argument("--force", action="store_true", help="忽略补充数据缓存，重新下载")
    return parser.parse_args()


def validate_date(value: str, name: str) -> None:
    datetime.strptime(value, "%Y%m%d")
    if len(value) != 8:
        raise ValueError(f"{name} 必须使用 YYYYMMDD 格式")


class TushareFetcher:
    def __init__(self, pro: Any) -> None:
        self.pro = pro
        self.last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        remaining = REQUEST_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def fetch(self, api_name: str, fields: str = "", **params: Any) -> pd.DataFrame:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._throttle()
                self.last_request_at = time.monotonic()
                call = getattr(self.pro, api_name)
                frame = call(fields=fields, **params) if fields else call(**params)
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError(f"{api_name} 返回了非 DataFrame 类型：{type(frame)!r}")
                return frame
            except Exception:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(attempt * 2)
        return pd.DataFrame()


def cache_file(cache_dir: Path, api_name: str, partition: str) -> Path:
    safe_partition = partition.replace(".", "_").replace(":", "_")
    return cache_dir / api_name / f"{safe_partition}.csv"


def empty_marker(path: Path) -> Path:
    return path.with_suffix(".empty.json")


def save_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.csv")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)
    empty_marker(path).unlink(missing_ok=True)


def save_empty_marker(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    empty_marker(path).write_text(
        json.dumps({"empty": True, "reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_cached(path: Path, force: bool) -> pd.DataFrame | None:
    if force:
        return None
    if path.exists():
        return pd.read_csv(
            path,
            dtype={
                "ts_code": str,
                "trade_date": str,
                "ann_date": str,
                "f_ann_date": str,
                "end_date": str,
                "cal_date": str,
                "month": str,
            },
            low_memory=False,
        )
    if empty_marker(path).exists():
        return pd.DataFrame()
    return None


def clean_frame(frame: pd.DataFrame, unique_keys: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = [key for key in unique_keys if key in frame.columns]
    frame = frame.drop_duplicates().copy()
    if keys:
        frame = frame.sort_values(keys).reset_index(drop=True)
    return frame


def combine_partitions(
    paths: list[Path], output_path: Path, unique_keys: Iterable[str]
) -> tuple[int, list[str]]:
    valid_paths = [path for path in paths if path.exists()]
    if not valid_paths:
        return 0, []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".tmp.csv")
    wrote_header = False
    total_rows = 0
    columns: list[str] = []
    for path in valid_paths:
        frame = load_cached(path, force=False)
        if frame is None or frame.empty:
            continue
        frame = clean_frame(frame, unique_keys)
        if not wrote_header:
            columns = list(frame.columns)
        frame.to_csv(
            temp,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
            encoding="utf-8-sig" if not wrote_header else "utf-8",
        )
        wrote_header = True
        total_rows += len(frame)

    if wrote_header:
        temp.replace(output_path)
    else:
        temp.unlink(missing_ok=True)
    return total_rows, columns


def download_per_stock(
    fetcher: TushareFetcher,
    spec: PerStockSpec,
    stock_pool: list[tuple[str, str, str]],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    combined_dir: Path,
    force: bool,
) -> dict[str, Any]:
    print(f"\n[{spec.api_name}] {spec.description}", flush=True)
    paths: list[Path] = []
    failures: list[str] = []
    empty_count = 0

    for position, (ts_code, name, _) in enumerate(stock_pool, start=1):
        path = cache_file(cache_dir, spec.api_name, ts_code)
        paths.append(path)
        cached = load_cached(path, force)
        if cached is not None:
            empty_count += int(cached.empty)
            print(
                f"  {position:02d}/{len(stock_pool)} {ts_code} {name}: 缓存 {len(cached):,} 行",
                flush=True,
            )
            continue

        params = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            **dict(spec.extra_params),
        }
        try:
            frame = fetcher.fetch(spec.api_name, **params)
            frame = clean_frame(frame, spec.unique_keys)
            if frame.empty:
                empty_count += 1
                save_empty_marker(path, "接口返回空表")
            else:
                save_csv_atomic(frame, path)
            print(
                f"  {position:02d}/{len(stock_pool)} {ts_code} {name}: 下载 {len(frame):,} 行",
                flush=True,
            )
        except Exception as exc:
            failures.append(f"{ts_code}: {exc}")
            print(
                f"  {position:02d}/{len(stock_pool)} {ts_code} {name}: 失败 - {exc}",
                flush=True,
            )

    rows, columns = combine_partitions(
        paths, combined_dir / f"{spec.api_name}.csv", spec.unique_keys
    )
    print(f"  合并结果：{rows:,} 行", flush=True)
    return {
        "description": spec.description,
        "rows": rows,
        "columns": columns,
        "partition_count": len(paths),
        "empty_partition_count": empty_count,
        "failure_count": len(failures),
        "failures": failures,
    }


def download_stock_basic(
    fetcher: TushareFetcher, cache_dir: Path, combined_dir: Path, force: bool
) -> dict[str, Any]:
    api_name = "stock_basic"
    print(f"\n[{api_name}] 全市场股票列表", flush=True)
    fields = (
        "ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,"
        "curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type"
    )
    paths: list[Path] = []
    failures: list[str] = []
    for status in ("L", "D", "P", "G"):
        path = cache_file(cache_dir, api_name, f"status_{status}")
        paths.append(path)
        cached = load_cached(path, force)
        if cached is not None:
            print(f"  状态 {status}: 缓存 {len(cached):,} 行", flush=True)
            continue
        try:
            frame = fetcher.fetch(
                api_name, fields=fields, exchange="", list_status=status
            )
            frame = clean_frame(frame, ("ts_code", "list_status"))
            if frame.empty:
                save_empty_marker(path, "该上市状态返回空表")
            else:
                save_csv_atomic(frame, path)
            print(f"  状态 {status}: 下载 {len(frame):,} 行", flush=True)
        except Exception as exc:
            failures.append(f"status={status}: {exc}")
            print(f"  状态 {status}: 失败 - {exc}", flush=True)

    rows, columns = combine_partitions(
        paths, combined_dir / f"{api_name}.csv", ("ts_code", "list_status")
    )
    return {
        "description": "全市场上市、退市、暂停上市及待上市股票列表",
        "rows": rows,
        "columns": columns,
        "failure_count": len(failures),
        "failures": failures,
    }


def download_calendars(
    fetcher: TushareFetcher,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    combined_dir: Path,
    force: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    print("\n[trade_cal] A 股交易日历", flush=True)
    paths: list[Path] = []
    failures: list[str] = []
    for exchange in ("SSE", "SZSE"):
        path = cache_file(cache_dir, "trade_cal", exchange)
        paths.append(path)
        cached = load_cached(path, force)
        if cached is None:
            try:
                frame = fetcher.fetch(
                    "trade_cal",
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                )
                frame = clean_frame(frame, ("exchange", "cal_date"))
                save_csv_atomic(frame, path)
                print(f"  {exchange}: 下载 {len(frame):,} 行", flush=True)
            except Exception as exc:
                failures.append(f"{exchange}: {exc}")
        else:
            print(f"  {exchange}: 缓存 {len(cached):,} 行", flush=True)
    rows, columns = combine_partitions(
        paths, combined_dir / "trade_cal.csv", ("exchange", "cal_date")
    )
    results["trade_cal"] = {
        "description": "上交所和深交所交易日历",
        "rows": rows,
        "columns": columns,
        "failure_count": len(failures),
        "failures": failures,
    }

    print("\n[us_tradecal] 美股交易日历（仅日历，不含美股行情）", flush=True)
    path = cache_file(cache_dir, "us_tradecal", "calendar")
    cached = load_cached(path, force)
    failures = []
    if cached is None:
        try:
            frame = fetcher.fetch(
                "us_tradecal", start_date=start_date, end_date=end_date
            )
            frame = clean_frame(frame, ("cal_date",))
            save_csv_atomic(frame, path)
            print(f"  下载 {len(frame):,} 行", flush=True)
        except Exception as exc:
            failures.append(str(exc))
            print(f"  跳过 - {exc}", flush=True)
    else:
        print(f"  缓存 {len(cached):,} 行", flush=True)
    rows, columns = combine_partitions(
        [path], combined_dir / "us_tradecal.csv", ("cal_date",)
    )
    results["us_tradecal"] = {
        "description": "美股交易日历；不包含任何美股行情",
        "rows": rows,
        "columns": columns,
        "failure_count": len(failures),
        "failures": failures,
    }
    return results


def iter_months(start_date: str, end_date: str) -> list[str]:
    start = pd.Period(start_date[:6], freq="M")
    end = pd.Period(end_date[:6], freq="M")
    return [str(period).replace("-", "") for period in pd.period_range(start, end, freq="M")]


def download_broker_recommend(
    fetcher: TushareFetcher,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    combined_dir: Path,
    force: bool,
) -> dict[str, Any]:
    api_name = "broker_recommend"
    print(f"\n[{api_name}] 券商月度金股", flush=True)
    # 官方样例和可用历史从 2021 年开始。
    effective_start = max(start_date, "20210101")
    paths: list[Path] = []
    failures: list[str] = []
    empty_count = 0
    for month in iter_months(effective_start, end_date):
        path = cache_file(cache_dir, api_name, month)
        paths.append(path)
        cached = load_cached(path, force)
        if cached is not None:
            empty_count += int(cached.empty)
            continue
        try:
            frame = fetcher.fetch(api_name, month=month)
            frame = clean_frame(frame, ("month", "broker", "ts_code"))
            if frame.empty:
                empty_count += 1
                save_empty_marker(path, "该月份返回空表")
            else:
                save_csv_atomic(frame, path)
            print(f"  {month}: {len(frame):,} 行", flush=True)
        except Exception as exc:
            failures.append(f"{month}: {exc}")
            print(f"  {month}: 失败 - {exc}", flush=True)

    rows, columns = combine_partitions(
        paths,
        combined_dir / f"{api_name}.csv",
        ("month", "broker", "ts_code"),
    )
    return {
        "description": "券商月度金股",
        "rows": rows,
        "columns": columns,
        "empty_partition_count": empty_count,
        "failure_count": len(failures),
        "failures": failures,
    }


def probe_cyq_chips(
    fetcher: TushareFetcher, cache_dir: Path, combined_dir: Path, force: bool
) -> dict[str, Any]:
    """先用多个股票和日期探测筹码接口，避免代理无数据时产生海量空请求。"""
    api_name = "cyq_chips"
    print(f"\n[{api_name}] 每日筹码分布可用性探测", flush=True)
    probe_cases = [
        ("000001.SZ", "20220429"),
        ("600000.SH", "20240102"),
        ("600519.SH", "20260723"),
    ]
    paths: list[Path] = []
    failures: list[str] = []
    for ts_code, trade_date in probe_cases:
        path = cache_file(cache_dir, api_name, f"probe_{ts_code}_{trade_date}")
        paths.append(path)
        cached = load_cached(path, force)
        if cached is not None:
            print(f"  {ts_code} {trade_date}: 缓存 {len(cached):,} 行", flush=True)
            continue
        try:
            frame = fetcher.fetch(
                api_name, ts_code=ts_code, trade_date=trade_date
            )
            frame = clean_frame(frame, ("ts_code", "trade_date", "price"))
            if frame.empty:
                save_empty_marker(path, "代理端返回空表")
            else:
                save_csv_atomic(frame, path)
            print(f"  {ts_code} {trade_date}: {len(frame):,} 行", flush=True)
        except Exception as exc:
            failures.append(f"{ts_code} {trade_date}: {exc}")
            print(f"  {ts_code} {trade_date}: 失败 - {exc}", flush=True)

    rows, columns = combine_partitions(
        paths, combined_dir / f"{api_name}.csv", ("ts_code", "trade_date", "price")
    )
    available = rows > 0
    return {
        "description": "每日筹码分布",
        "rows": rows,
        "columns": columns,
        "availability_probe_only": True,
        "available_through_proxy": available,
        "note": (
            "三个跨股票、跨年份样本均为空，因此停止批量空请求。"
            if not available
            else "探测到数据；如需全量筹码价位分布，建议单独分月下载。"
        ),
        "failure_count": len(failures),
        "failures": failures,
    }


def inspect_combined_file(path: Path, unique_keys: Iterable[str]) -> dict[str, Any]:
    """只读取宽表的主键列，避免质量检查占用过多内存。"""
    if not path.exists():
        return {"file_exists": False}
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    keys = [key for key in unique_keys if key in columns]
    key_frame = pd.read_csv(path, usecols=keys, dtype=str, low_memory=False)
    date_column = next(
        (
            column
            for column in ("trade_date", "ann_date", "cal_date", "month", "list_date")
            if column in key_frame.columns
        ),
        None,
    )
    return {
        "file_exists": True,
        "rows": len(key_frame),
        "column_count": len(columns),
        "duplicate_business_keys": (
            int(key_frame.duplicated(keys).sum()) if keys else 0
        ),
        "date_column": date_column,
        "min_date": (
            str(key_frame[date_column].min()) if date_column and not key_frame.empty else None
        ),
        "max_date": (
            str(key_frame[date_column].max()) if date_column and not key_frame.empty else None
        ),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    validate_date(args.start_date, "start-date")
    validate_date(args.end_date, "end-date")
    if args.start_date > args.end_date:
        raise ValueError("start-date 不能晚于 end-date")

    data_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = data_dir / "extra_cache"
    combined_dir = data_dir / "combined"
    report_path = data_dir / "extra_download_report.json"
    data_dir.mkdir(parents=True, exist_ok=True)

    fetcher = TushareFetcher(base.create_pro_client())
    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "api_base_url": base.API_BASE_URL,
        "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
        "stock_count": len(base.STOCK_POOL),
        "datasets": {},
        "already_downloaded": [
            "daily",
            "daily_basic",
            "index_daily",
            "moneyflow",
            "stk_limit",
        ],
    }

    report["datasets"]["stock_basic"] = download_stock_basic(
        fetcher, cache_dir, combined_dir, args.force
    )
    report["datasets"].update(
        download_calendars(
            fetcher,
            args.start_date,
            args.end_date,
            cache_dir,
            combined_dir,
            args.force,
        )
    )

    for spec in PER_STOCK_SPECS:
        report["datasets"][spec.api_name] = download_per_stock(
            fetcher,
            spec,
            base.STOCK_POOL,
            args.start_date,
            args.end_date,
            cache_dir,
            combined_dir,
            args.force,
        )

    report["datasets"]["broker_recommend"] = download_broker_recommend(
        fetcher,
        args.start_date,
        args.end_date,
        cache_dir,
        combined_dir,
        args.force,
    )
    report["datasets"]["cyq_chips"] = probe_cyq_chips(
        fetcher, cache_dir, combined_dir, args.force
    )

    report["new_total_rows"] = sum(
        dataset["rows"] for dataset in report["datasets"].values()
    )
    report["total_failures"] = sum(
        dataset["failure_count"] for dataset in report["datasets"].values()
    )
    report["quality"] = {
        api_name: inspect_combined_file(
            combined_dir / f"{api_name}.csv", unique_keys
        )
        for api_name, unique_keys in QUALITY_KEYS.items()
    }
    report["quality_notes"] = [
        "财务报表的重复业务键通常表示同一报告期的更正或重述版本，原始数据予以保留。",
        "回测或机器学习必须依据 f_ann_date 选择当时已经公开的财务版本，避免未来数据泄漏。",
        "broker_recommend 经代理每月只返回一行，应视为代理返回范围而非完整市场覆盖。",
        "cyq_chips 的三个跨股票、跨年份探测样本均为空，未执行海量空请求。",
    ]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 补充接口下载汇总 ===", flush=True)
    for api_name, result in report["datasets"].items():
        print(
            f"{api_name:18s}: {result['rows']:>10,} 行，失败 {result['failure_count']}",
            flush=True,
        )
    print(f"新增数据总行数：{report['new_total_rows']:,}", flush=True)
    print(f"报告：{report_path}", flush=True)


if __name__ == "__main__":
    main()
