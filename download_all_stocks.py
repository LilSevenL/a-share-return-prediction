"""下载全部当前上市 A 股的量化研究数据（分钟线除外）。

运行示例：
    python download_all_stocks.py --start-date 20100101 --end-date 20260827
    python download_all_stocks.py --start-date 20250101 --max-stocks 10
    python download_all_stocks.py --interfaces daily,adj_factor,daily_basic

脚本复用 main.py 的 Tushare SDK 代理配置，按接口/股票保存 gzip CSV 分片。
已有的原始 CSV 缓存或 gzip 分片会被跳过，支持安全中断和断点续传。
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import download_extra as extra
import main as base


REQUEST_INTERVAL_SECONDS = max(0.2, base.REQUEST_INTERVAL_SECONDS)
MAX_RETRIES = 5


@dataclass(frozen=True)
class ApiSpec:
    name: str
    description: str
    unique_keys: tuple[str, ...]
    fields: str = ""
    extra_params: tuple[tuple[str, Any], ...] = ()
    cache_group: str = "cache"


MAIN_SPECS = {
    spec.api_name: ApiSpec(
        spec.api_name,
        spec.description,
        spec.unique_keys,
        fields=spec.fields,
        cache_group="cache",
    )
    for spec in base.DATASETS
}

EXTRA_SPECS = {
    spec.api_name: ApiSpec(
        spec.api_name,
        spec.description,
        spec.unique_keys,
        extra_params=spec.extra_params,
        cache_group="extra_cache",
    )
    for spec in extra.PER_STOCK_SPECS
}

ALL_SPECS = {**MAIN_SPECS, **EXTRA_SPECS}
DEFAULT_INTERFACES = tuple(ALL_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20100101", help="开始日期 YYYYMMDD")
    parser.add_argument(
        "--end-date", default=date.today().strftime("%Y%m%d"), help="结束日期 YYYYMMDD"
    )
    parser.add_argument(
        "--output-dir", default=str(base.DEFAULT_DATA_DIR), help="数据根目录"
    )
    parser.add_argument(
        "--stock-basic",
        default=str(base.DEFAULT_DATA_DIR / "combined" / "stock_basic.csv"),
        help="在线股票列表失败时使用的本地文件",
    )
    parser.add_argument(
        "--interfaces",
        default=",".join(DEFAULT_INTERFACES),
        help="逗号分隔接口名；默认下载上次64只股票使用过的全部逐股票接口",
    )
    parser.add_argument("--max-stocks", type=int, help="仅处理前 N 只，用于连通性测试")
    parser.add_argument("--force", action="store_true", help="覆盖已有缓存分片")
    parser.add_argument(
        "--min-free-gb", type=float, default=10.0, help="磁盘剩余空间安全阈值"
    )
    return parser.parse_args()


def validate_day(value: str, name: str) -> None:
    datetime.strptime(value, "%Y%m%d")
    if len(value) != 8:
        raise ValueError(f"{name} 必须为 YYYYMMDD")


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
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


def cache_paths(data_dir: Path, spec: ApiSpec, ts_code: str) -> tuple[Path, Path, Path]:
    safe_code = ts_code.replace(".", "_")
    root = data_dir / spec.cache_group / spec.name
    return (
        root / f"{safe_code}.csv.gz",
        root / f"{safe_code}.csv",
        root / f"{safe_code}.empty.json",
    )


def has_cache(data_dir: Path, spec: ApiSpec, ts_code: str) -> bool:
    return any(path.exists() for path in cache_paths(data_dir, spec, ts_code))


def clean_frame(frame: pd.DataFrame, spec: ApiSpec) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for column in (
        "trade_date",
        "ann_date",
        "f_ann_date",
        "end_date",
        "list_date",
        "delist_date",
    ):
        if column in result.columns:
            result[column] = result[column].astype("string").str.replace(".0", "", regex=False)
    keys = [key for key in spec.unique_keys if key in result.columns]
    if keys:
        result = result.drop_duplicates(keys, keep="last").sort_values(keys)
    return result.reset_index(drop=True)


def ensure_space(path: Path, minimum_gb: float) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"磁盘仅剩 {free_gb:.2f}GB，低于安全阈值 {minimum_gb:.2f}GB，任务停止"
        )
    return free_gb


class Fetcher:
    def __init__(self, pro: Any) -> None:
        self.pro = pro
        self.last_request_at = 0.0

    def fetch(self, spec: ApiSpec, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        params: dict[str, Any] = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            **dict(spec.extra_params),
        }
        if spec.fields:
            params["fields"] = spec.fields

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                remaining = REQUEST_INTERVAL_SECONDS - (time.monotonic() - self.last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
                self.last_request_at = time.monotonic()
                frame = getattr(self.pro, spec.name)(**params)
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError(f"{spec.name} 返回类型异常：{type(frame)!r}")
                return frame
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise
                wait_seconds = attempt * 2
                print(f"    第 {attempt} 次失败，{wait_seconds} 秒后重试：{exc}")
                time.sleep(wait_seconds)
        return pd.DataFrame()


def load_universe(pro: Any, local_path: Path) -> pd.DataFrame:
    try:
        frame = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
        )
        time.sleep(REQUEST_INTERVAL_SECONDS)
    except Exception as exc:
        print(f"在线股票列表请求失败：{exc}")
        frame = pd.DataFrame()

    if frame.empty:
        if not local_path.exists():
            raise RuntimeError("在线股票列表为空，且本地 stock_basic.csv 不存在")
        frame = pd.read_csv(
            local_path, dtype={"ts_code": str, "list_status": str, "list_date": str}
        )
        frame = frame[frame["list_status"] == "L"].copy()
        print(f"使用本地上市股票列表：{len(frame):,} 只")
    else:
        frame = frame.drop_duplicates("ts_code", keep="last")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(local_path, index=False, encoding="utf-8-sig")
        print(f"股票列表已在线刷新：{len(frame):,} 只")
    return frame.sort_values("ts_code").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    validate_day(args.start_date, "start-date")
    validate_day(args.end_date, "end-date")
    if args.start_date > args.end_date:
        raise ValueError("start-date 不能晚于 end-date")
    if args.max_stocks is not None and args.max_stocks <= 0:
        raise ValueError("max-stocks 必须大于 0")

    names = [name.strip() for name in args.interfaces.split(",") if name.strip()]
    unknown = sorted(set(names) - set(ALL_SPECS))
    if unknown:
        raise ValueError(f"未知接口：{', '.join(unknown)}")
    specs = [ALL_SPECS[name] for name in names]

    data_dir = Path(args.output_dir).expanduser().resolve()
    ensure_space(data_dir, args.min_free_gb)
    pro = base.create_pro_client()
    universe = load_universe(pro, Path(args.stock_basic))
    if args.max_stocks:
        universe = universe.head(args.max_stocks).copy()

    progress_path = data_dir / "all_stocks_progress.json"
    total_tasks = len(universe) * len(specs)
    downloaded = skipped = empty = failed = rows = 0
    disabled_interfaces: set[str] = set()
    started_at = pd.Timestamp.now().isoformat()
    print(f"任务：{len(universe):,} 只股票 × {len(specs)} 个接口 = {total_tasks:,} 个分片")

    fetcher = Fetcher(pro)
    for stock_number, stock in enumerate(universe.itertuples(index=False), start=1):
        ts_code = stock.ts_code
        stock_start = args.start_date
        list_date = str(getattr(stock, "list_date", "") or "").replace(".0", "")
        if len(list_date) == 8 and list_date.isdigit():
            stock_start = max(stock_start, list_date)

        for spec in specs:
            if spec.name in disabled_interfaces:
                skipped += 1
                continue
            gzip_path, _, marker_path = cache_paths(data_dir, spec, ts_code)
            if not args.force and has_cache(data_dir, spec, ts_code):
                skipped += 1
                continue
            ensure_space(data_dir, args.min_free_gb)
            try:
                frame = clean_frame(
                    fetcher.fetch(spec, ts_code, stock_start, args.end_date), spec
                )
                if frame.empty:
                    atomic_json(
                        {"empty": True, "ts_code": ts_code, "api_name": spec.name}, marker_path
                    )
                    empty += 1
                else:
                    atomic_gzip_csv(frame, gzip_path)
                    marker_path.unlink(missing_ok=True)
                    downloaded += 1
                    rows += len(frame)
            except Exception as exc:
                failed += 1
                message = str(exc)
                print(f"[{stock_number}/{len(universe)}] {ts_code} {spec.name} 失败：{message}")
                if base.is_permission_error(message):
                    disabled_interfaces.add(spec.name)
                    print(f"接口 {spec.name} 无权限，后续股票将跳过")

            processed = downloaded + skipped + empty + failed
            atomic_json(
                {
                    "started_at": started_at,
                    "updated_at": pd.Timestamp.now().isoformat(),
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "stock_count": len(universe),
                    "interfaces": names,
                    "total_tasks": total_tasks,
                    "processed": processed,
                    "downloaded_partitions": downloaded,
                    "skipped_partitions": skipped,
                    "empty_partitions": empty,
                    "failed_partitions": failed,
                    "downloaded_rows": rows,
                    "disabled_interfaces": sorted(disabled_interfaces),
                    "last_ts_code": ts_code,
                    "last_interface": spec.name,
                },
                progress_path,
            )
            if processed % 100 == 0:
                free_gb = ensure_space(data_dir, args.min_free_gb)
                print(
                    f"进度 {processed:,}/{total_tasks:,}，新增 {rows:,} 行，"
                    f"失败 {failed:,}，剩余 {free_gb:.2f}GB"
                )

    print(
        f"完成：下载 {downloaded:,} 个分片，跳过 {skipped:,}，空表 {empty:,}，"
        f"失败 {failed:,}，新增 {rows:,} 行。进度文件：{progress_path}"
    )


if __name__ == "__main__":
    main()
