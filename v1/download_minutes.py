"""使用 Tushare stk_mins 下载 A 股历史分钟线。

依赖安装：
    conda activate quant_ml
    python -m pip install --upgrade tushare pandas

示例：
    python download_minutes.py --start-date 20250101 --end-date 20260827 --freq 5min
    python download_minutes.py --start-date 20260801 --end-date 20260827 --freq 1min --max-stocks 10

数据按“频率/股票/月份”保存为 gzip CSV。文件存在即视为完成，因此任务中断后
使用相同参数重跑即可续传。每个请求仅包含一只股票的一个自然月，避免触及
stk_mins 单次最多 8000 行的限制。
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
from typing import Any, Iterator

import pandas as pd

import main as base


REQUEST_INTERVAL_SECONDS = max(0.2, base.REQUEST_INTERVAL_SECONDS)
MAX_RETRIES = 5
ROW_LIMIT = 8_000


@dataclass(frozen=True)
class MonthWindow:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument(
        "--end-date", default=date.today().strftime("%Y%m%d"), help="结束日期 YYYYMMDD"
    )
    parser.add_argument(
        "--freq",
        default="5min",
        choices=("1min", "5min", "15min", "30min", "60min"),
        help="分钟频率；全市场初次下载建议先用 5min",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base.DEFAULT_DATA_DIR / "minutes"),
        help="输出目录",
    )
    parser.add_argument(
        "--stock-basic",
        default=str(base.DEFAULT_DATA_DIR / "combined" / "stock_basic.csv"),
        help="本地股票列表；在线刷新失败时使用",
    )
    parser.add_argument(
        "--include-delisted", action="store_true", help="同时下载已退市股票，默认仅当前上市股票"
    )
    parser.add_argument("--max-stocks", type=int, help="仅处理前 N 只，适合连通性测试")
    parser.add_argument("--force", action="store_true", help="覆盖已有分片")
    parser.add_argument("--dry-run", action="store_true", help="只计算任务规模，不请求分钟数据")
    parser.add_argument(
        "--min-free-gb", type=float, default=10.0, help="磁盘剩余空间低于该值时安全停止"
    )
    return parser.parse_args()


def parse_day(value: str, name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(datetime.strptime(value, "%Y%m%d"))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 YYYYMMDD 格式：{value}") from exc
    return parsed


def month_windows(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[MonthWindow]:
    cursor = start.normalize()
    while cursor <= end:
        month_end = cursor + pd.offsets.MonthEnd(0)
        window_end = min(month_end, end.normalize())
        yield MonthWindow(cursor.strftime("%Y%m"), cursor, window_end)
        cursor = month_end + pd.Timedelta(days=1)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def atomic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)
    temp.replace(path)


def partition_path(output_dir: Path, freq: str, ts_code: str, month: str) -> Path:
    return output_dir / freq / ts_code.replace(".", "_") / f"{month}.csv.gz"


def empty_path(data_path: Path) -> Path:
    return data_path.with_suffix(".empty.json")


def load_stock_universe(
    pro: Any | None, local_path: Path, include_delisted: bool
) -> pd.DataFrame:
    statuses = ["L", "D"] if include_delisted else ["L"]
    frames: list[pd.DataFrame] = []
    if pro is not None:
        for status in statuses:
            try:
                frame = pro.stock_basic(
                    exchange="",
                    list_status=status,
                    fields="ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
                )
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                print(f"stock_basic({status}) 在线刷新失败：{exc}")
            time.sleep(REQUEST_INTERVAL_SECONDS)

    if frames:
        universe = pd.concat(frames, ignore_index=True)
        universe = universe.drop_duplicates(["ts_code", "list_status"], keep="last")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_csv(local_path, index=False, encoding="utf-8-sig")
        print(f"股票列表已在线刷新：{len(universe):,} 只")
    else:
        if not local_path.exists():
            raise RuntimeError("在线 stock_basic 返回空表，且没有可用的本地股票列表")
        universe = pd.read_csv(
            local_path,
            dtype={"ts_code": str, "list_status": str, "list_date": str, "delist_date": str},
        )
        universe = universe[universe["list_status"].isin(statuses)].copy()
        print(f"在线股票列表为空，使用本地列表：{len(universe):,} 只")

    return universe.sort_values("ts_code").reset_index(drop=True)


class MinuteFetcher:
    def __init__(self, pro: Any) -> None:
        self.pro = pro
        self.last_request_at = 0.0

    def _throttle(self) -> None:
        remaining = REQUEST_INTERVAL_SECONDS - (time.monotonic() - self.last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def fetch(self, ts_code: str, freq: str, window: MonthWindow) -> pd.DataFrame:
        start_text = window.start.strftime("%Y-%m-%d 00:00:00")
        end_text = window.end.strftime("%Y-%m-%d 23:59:59")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._throttle()
                self.last_request_at = time.monotonic()
                frame = self.pro.stk_mins(
                    ts_code=ts_code,
                    freq=freq,
                    start_date=start_text,
                    end_date=end_text,
                )
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError(f"stk_mins 返回类型异常：{type(frame)!r}")
                return frame
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise
                wait_seconds = attempt * 2
                print(f"    第 {attempt} 次失败，{wait_seconds} 秒后重试：{exc}")
                time.sleep(wait_seconds)
        return pd.DataFrame()


def clean_minutes(frame: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "code" in result.columns and "ts_code" not in result.columns:
        result = result.rename(columns={"code": "ts_code"})
    if "ts_code" not in result.columns:
        result.insert(0, "ts_code", ts_code)
    if "trade_time" not in result.columns:
        if "time" in result.columns:
            result = result.rename(columns={"time": "trade_time"})
        else:
            raise ValueError("stk_mins 返回结果缺少 trade_time/time 字段")

    result["ts_code"] = result["ts_code"].fillna(ts_code).astype(str)
    result["trade_time"] = pd.to_datetime(result["trade_time"], errors="coerce")
    result = result.dropna(subset=["trade_time"])
    for column in ("open", "high", "low", "close", "vol", "amount"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.drop_duplicates(["ts_code", "trade_time"], keep="last")
    return result.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)


def effective_windows(
    row: Any, requested_start: pd.Timestamp, requested_end: pd.Timestamp
) -> list[MonthWindow]:
    list_date = pd.to_datetime(getattr(row, "list_date", None), format="%Y%m%d", errors="coerce")
    delist_date = pd.to_datetime(
        getattr(row, "delist_date", None), format="%Y%m%d", errors="coerce"
    )
    start = max(requested_start, list_date) if pd.notna(list_date) else requested_start
    end = min(requested_end, delist_date) if pd.notna(delist_date) else requested_end
    return list(month_windows(start, end)) if start <= end else []


def ensure_free_space(path: Path, min_free_gb: float) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"磁盘仅剩 {free_gb:.2f}GB，低于安全阈值 {min_free_gb:.2f}GB，任务已停止"
        )
    return free_gb


def main() -> None:
    args = parse_args()
    start = parse_day(args.start_date, "start-date")
    end = parse_day(args.end_date, "end-date")
    if start > end:
        raise ValueError("start-date 不能晚于 end-date")
    if args.max_stocks is not None and args.max_stocks <= 0:
        raise ValueError("max-stocks 必须大于 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    progress_path = output_dir / f"progress_{args.freq}.json"
    pro = None if args.dry_run else base.create_pro_client()
    universe = load_stock_universe(pro, Path(args.stock_basic), args.include_delisted)
    if args.max_stocks:
        universe = universe.head(args.max_stocks).copy()

    task_count = sum(
        len(effective_windows(row, start, end)) for row in universe.itertuples(index=False)
    )
    print(
        f"任务范围：{len(universe):,} 只股票，{args.freq}，"
        f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d}，约 {task_count:,} 个按月请求"
    )
    if args.dry_run:
        return

    fetcher = MinuteFetcher(pro)
    completed = skipped = empty = failed = rows = 0
    started_at = pd.Timestamp.now().isoformat()

    for stock_index, row in enumerate(universe.itertuples(index=False), start=1):
        ts_code = row.ts_code
        windows = effective_windows(row, start, end)
        for window in windows:
            data_path = partition_path(output_dir, args.freq, ts_code, window.label)
            marker_path = empty_path(data_path)
            if not args.force and (data_path.exists() or marker_path.exists()):
                skipped += 1
                continue
            ensure_free_space(output_dir, args.min_free_gb)
            try:
                frame = clean_minutes(fetcher.fetch(ts_code, args.freq, window), ts_code)
                if len(frame) >= ROW_LIMIT:
                    raise RuntimeError(
                        f"返回 {len(frame):,} 行，可能触及 {ROW_LIMIT:,} 行上限；"
                        "请缩小时间分片后重试"
                    )
                if frame.empty:
                    marker_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(
                        {"ts_code": ts_code, "month": window.label, "empty": True}, marker_path
                    )
                    empty += 1
                else:
                    atomic_gzip_csv(frame, data_path)
                    marker_path.unlink(missing_ok=True)
                    completed += 1
                    rows += len(frame)
            except Exception as exc:
                failed += 1
                print(f"[{stock_index}/{len(universe)}] {ts_code} {window.label} 失败：{exc}")

            done = completed + skipped + empty + failed
            atomic_json(
                {
                    "started_at": started_at,
                    "updated_at": pd.Timestamp.now().isoformat(),
                    "freq": args.freq,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "stock_count": len(universe),
                    "task_count": task_count,
                    "processed": done,
                    "downloaded_partitions": completed,
                    "skipped_partitions": skipped,
                    "empty_partitions": empty,
                    "failed_partitions": failed,
                    "downloaded_rows": rows,
                    "last_ts_code": ts_code,
                    "last_month": window.label,
                },
                progress_path,
            )
            if done % 50 == 0:
                print(
                    f"进度 {done:,}/{task_count:,}，新增 {rows:,} 行，"
                    f"失败 {failed:,}，剩余空间 {ensure_free_space(output_dir, args.min_free_gb):.2f}GB"
                )

    print(
        f"完成：新增分片 {completed:,}，跳过 {skipped:,}，空分片 {empty:,}，"
        f"失败 {failed:,}，新增行数 {rows:,}。进度文件：{progress_path}"
    )


if __name__ == "__main__":
    main()
