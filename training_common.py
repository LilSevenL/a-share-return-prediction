"""三个滚动训练脚本共用的数据、预处理、评估与保存工具。"""

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
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

from build_features import FACTOR_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "quant_data" / "processed" / "model_dataset"


def add_common_args(
    parser: argparse.ArgumentParser, default_output_dir: Path
) -> None:
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument(
        "--folds", nargs="*", help="仅运行指定折，例如 validation_2024；默认全部"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--preprocess-sample-rows", type=int, default=500_000)
    parser.add_argument("--min-feature-coverage", type=float, default=0.05)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-rows", type=int,
        help="仅用于调试；按日期排序后截取Training前N行",
    )
    parser.add_argument(
        "--max-validation-rows", type=int,
        help="仅用于调试；按日期排序后截取Validation前N行",
    )
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true", help="备份旧输出并从头训练")


def validate_common_args(args: argparse.Namespace) -> None:
    if args.threads <= 0:
        raise ValueError("threads 必须大于0")
    if args.preprocess_sample_rows <= 0:
        raise ValueError("preprocess-sample-rows 必须大于0")
    if not 0 <= args.min_feature_coverage <= 1:
        raise ValueError("min-feature-coverage 必须在0到1之间")
    if not 0 <= args.winsor_lower < args.winsor_upper <= 1:
        raise ValueError("缩尾分位数设置无效")
    for value, name in (
        (args.max_train_rows, "max-train-rows"),
        (args.max_validation_rows, "max-validation-rows"),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} 必须大于0")


def quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def atomic_json(payload: Any, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temp.replace(path)


def atomic_text(value: str, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def prepare_output(path: Path, force: bool, minimum_free_gb: float) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path.parent).free / 1024**3
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"磁盘仅剩 {free_gb:.2f}GB，低于安全阈值 {minimum_free_gb:.2f}GB"
        )
    if force and path.exists():
        backup = path.with_name(f"{path.name}.backup.{datetime.now():%Y%m%d%H%M%S}")
        path.replace(backup)
        print(f"旧训练结果已备份到：{backup}", flush=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_progress(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": datetime.now().isoformat(), "completed": [], "results": []}


def load_folds(dataset_dir: Path, requested: list[str] | None) -> list[dict[str, Any]]:
    path = dataset_dir / "rolling_folds.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少滚动折配置：{path}")
    folds = json.loads(path.read_text(encoding="utf-8"))["folds"]
    if requested:
        wanted = set(requested)
        known = {fold["name"] for fold in folds}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError("未知滚动折：" + ", ".join(unknown))
        folds = [fold for fold in folds if fold["name"] in wanted]
    if not folds:
        raise ValueError("没有需要运行的滚动折")
    return folds


def dataset_glob(dataset_dir: Path) -> str:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"模型数据目录不存在：{dataset_dir}")
    if not next(dataset_dir.rglob("*.parquet"), None):
        raise FileNotFoundError(f"模型数据目录中没有Parquet：{dataset_dir}")
    return str(
        dataset_dir / "final_split=*" / "signal_year=*" / "*.parquet"
    ).replace("\\", "/")


def open_connection(output_dir: Path, args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads = {int(args.threads)}")
    con.execute(f"SET memory_limit = {quote_sql(args.memory_limit)}")
    con.execute(f"SET temp_directory = {quote_sql(str(output_dir / '_duckdb_tmp'))}")
    con.execute("SET preserve_insertion_order = false")
    return con


def select_sql(fold: dict[str, Any], split: str) -> str:
    if split == "train":
        return (
            f"signal_date >= DATE {quote_sql(fold['train_start'])} "
            f"AND signal_date < DATE {quote_sql(fold['train_signal_before'])} "
            f"AND exit_date < DATE {quote_sql(fold['train_exit_before'])}"
        )
    return (
        f"signal_date >= DATE {quote_sql(fold['validation_start'])} "
        f"AND signal_date < DATE {quote_sql(fold['validation_signal_before'])} "
        f"AND exit_date < DATE {quote_sql(fold['validation_exit_before'])}"
    )


def load_matrix(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    condition: str,
    row_limit: int | None,
    include_keys: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    key_sql = "ts_code, signal_date," if include_keys else ""
    factors = ", ".join(f"CAST({name} AS FLOAT) AS {name}" for name in FACTOR_COLUMNS)
    limit = f"LIMIT {int(row_limit)}" if row_limit else ""
    table = con.execute(
        f"""
        SELECT {key_sql} {factors}, CAST(target_logret_5d AS FLOAT) AS target
        FROM read_parquet({quote_sql(glob)}, hive_partitioning=true)
        WHERE {condition}
        ORDER BY signal_date, ts_code
        {limit}
        """
    ).to_arrow_table()
    rows = table.num_rows
    if rows == 0:
        raise RuntimeError("当前切分没有样本")
    x = np.empty((rows, len(FACTOR_COLUMNS)), dtype=np.float32)
    for index, name in enumerate(FACTOR_COLUMNS):
        x[:, index] = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    y = table["target"].combine_chunks().to_numpy(zero_copy_only=False).astype(
        np.float32, copy=True
    )
    codes = dates = None
    if include_keys:
        codes = np.asarray(table["ts_code"].combine_chunks().to_pylist(), dtype="U12")
        dates = table["signal_date"].combine_chunks().to_numpy(zero_copy_only=False)
    return x, y, codes, dates


def load_matrix_with_dates(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    condition: str,
    row_limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载因子、标签和日期，但不复制股票代码；用于按日加权训练。"""
    factors = ", ".join(f"CAST({name} AS FLOAT) AS {name}" for name in FACTOR_COLUMNS)
    limit = f"LIMIT {int(row_limit)}" if row_limit else ""
    table = con.execute(
        f"""
        SELECT signal_date, {factors}, CAST(target_logret_5d AS FLOAT) AS target
        FROM read_parquet({quote_sql(glob)}, hive_partitioning=true)
        WHERE {condition}
        ORDER BY signal_date, ts_code
        {limit}
        """
    ).to_arrow_table()
    rows = table.num_rows
    if rows == 0:
        raise RuntimeError("当前切分没有样本")
    x = np.empty((rows, len(FACTOR_COLUMNS)), dtype=np.float32)
    for index, name in enumerate(FACTOR_COLUMNS):
        x[:, index] = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    y = table["target"].combine_chunks().to_numpy(zero_copy_only=False).astype(
        np.float32, copy=True
    )
    dates = table["signal_date"].combine_chunks().to_numpy(zero_copy_only=False)
    return x, y, dates


def fit_preprocessor(x: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    sample = x[
        rng.choice(len(x), args.preprocess_sample_rows, replace=False)
    ] if len(x) > args.preprocess_sample_rows else x
    stats: dict[str, list[Any] | Any] = {
        "median": [], "lower": [], "upper": [], "mean": [], "std": [],
        "coverage": [], "active": [],
    }
    for index in range(len(FACTOR_COLUMNS)):
        values = sample[:, index].astype(np.float64, copy=False)
        finite = values[np.isfinite(values)]
        coverage = float(len(finite) / len(values))
        if len(finite):
            lower, median, upper = np.quantile(
                finite, [args.winsor_lower, 0.5, args.winsor_upper]
            )
            clipped = np.clip(finite, lower, upper)
            mean, std = float(clipped.mean()), float(clipped.std())
        else:
            lower = median = upper = mean = 0.0
            std = 1.0
        active = coverage >= args.min_feature_coverage and math.isfinite(std) and std > 1e-12
        stats["median"].append(float(median))
        stats["lower"].append(float(lower))
        stats["upper"].append(float(upper))
        stats["mean"].append(mean)
        stats["std"].append(std if active else 1.0)
        stats["coverage"].append(coverage)
        stats["active"].append(active)
    stats["factor_names"] = FACTOR_COLUMNS
    stats["active_factors"] = [
        name for name, active in zip(FACTOR_COLUMNS, stats["active"]) if active
    ]
    stats["sample_rows"] = int(len(sample))
    return stats


def transform(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    active = np.asarray(stats["active"], dtype=bool)
    for index in range(x.shape[1]):
        column = x[:, index]
        bad = ~np.isfinite(column)
        if bad.any():
            column[bad] = stats["median"][index]
        np.clip(column, stats["lower"][index], stats["upper"][index], out=column)
        column -= stats["mean"][index]
        column /= stats["std"][index]
    return np.ascontiguousarray(x[:, active], dtype=np.float32)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x0, y0 = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.dot(x0, x0) * np.dot(y0, y0)))
    return float(np.dot(x0, y0) / denominator) if denominator > 0 else float("nan")


def daily_rank_ic(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    values: list[float] = []
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], len(dates)]
    for start, end in zip(starts, ends):
        if end - start < 20:
            continue
        value = pearson(rankdata(pred[start:end]), rankdata(y[start:end]))
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def evaluate(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> dict[str, float]:
    daily_ic, daily_ric, spreads = [], [], []
    _, starts = np.unique(dates, return_index=True)
    ends = np.r_[starts[1:], len(dates)]
    for start, end in zip(starts, ends):
        count = end - start
        if count < 20:
            continue
        p, actual = pred[start:end], y[start:end]
        ic = pearson(p, actual)
        ric = pearson(rankdata(p), rankdata(actual))
        if math.isfinite(ic):
            daily_ic.append(ic)
        if math.isfinite(ric):
            daily_ric.append(ric)
        size = max(1, count // 10)
        order = np.argsort(p)
        spreads.append(float(actual[order[-size:]].mean() - actual[order[:size]].mean()))
    residual = pred.astype(np.float64) - y.astype(np.float64)
    y_var = float(np.sum((y - y.mean()) ** 2))
    return {
        "rows": int(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual**2) / y_var) if y_var > 0 else float("nan"),
        "overall_ic": pearson(pred, y),
        "daily_ic_mean": float(np.mean(daily_ic)),
        "daily_ic_std": float(np.std(daily_ic)),
        "daily_rank_ic_mean": float(np.mean(daily_ric)),
        "daily_rank_ic_std": float(np.std(daily_ric)),
        "daily_rank_ic_positive_ratio": float(np.mean(np.asarray(daily_ric) > 0)),
        "daily_top_bottom_logret_mean": float(np.mean(spreads)),
        "trading_days": int(len(daily_ric)),
    }


def save_predictions(
    path: Path, codes: np.ndarray, dates: np.ndarray, y: np.ndarray, pred: np.ndarray
) -> None:
    temp = path.with_suffix(".tmp.parquet")
    pq.write_table(
        pa.table(
            {
                "ts_code": pa.array(codes), "signal_date": pa.array(dates),
                "target_logret_5d": pa.array(y),
                "prediction": pa.array(pred.astype(np.float32, copy=False)),
            }
        ),
        temp, compression="zstd", row_group_size=100_000,
    )
    temp.replace(path)


def aggregate_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in records]
    if not metrics:
        return {}
    return {
        "folds": len(metrics),
        "mean_daily_ic": float(np.mean([x["daily_ic_mean"] for x in metrics])),
        "mean_daily_rank_ic": float(np.mean([x["daily_rank_ic_mean"] for x in metrics])),
        "mean_top_bottom_logret": float(
            np.mean([x["daily_top_bottom_logret_mean"] for x in metrics])
        ),
        "mean_rmse": float(np.mean([x["rmse"] for x in metrics])),
    }
