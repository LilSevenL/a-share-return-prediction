"""为冻结模型生成2017-2024全股票Validation信号；不读取永久Test。

与训练期Validation预测不同，本脚本从features_50读取当日全部可用特征，
不会按照未来标签是否有效预先删除股票，避免回测中的未来可交易性偏差。
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from build_features import FACTOR_COLUMNS
from train_mlp import MLP, predict
from training_common import PROJECT_DIR, atomic_json, load_folds, quote_sql, transform


PROCESSED = PROJECT_DIR / "quant_data" / "processed"
DEFAULT_FEATURES = PROCESSED / "features_50"
DEFAULT_DATASET = PROCESSED / "model_dataset"
DEFAULT_OUTPUT = PROCESSED / "validation_signals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def feature_glob(directory: Path) -> str:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return str(directory / "feature_year=*" / "*.parquet").replace("\\", "/")


def load_features(
    con: duckdb.DuckDBPyConnection, glob: str, start: str, end: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factors = ", ".join(f"CAST({name} AS FLOAT) AS {name}" for name in FACTOR_COLUMNS)
    table = con.execute(
        f"""
        SELECT ts_code, signal_date, {factors}
        FROM read_parquet({quote_sql(glob)}, hive_partitioning=true)
        WHERE signal_date >= DATE {quote_sql(start)}
          AND signal_date < DATE {quote_sql(end)}
        ORDER BY signal_date, ts_code
        """
    ).to_arrow_table()
    if table.num_rows == 0:
        raise RuntimeError(f"{start}至{end}没有特征")
    x = np.empty((table.num_rows, len(FACTOR_COLUMNS)), dtype=np.float32)
    for index, name in enumerate(FACTOR_COLUMNS):
        x[:, index] = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    codes = np.asarray(table["ts_code"].combine_chunks().to_pylist(), dtype="U12")
    dates = table["signal_date"].combine_chunks().to_numpy(zero_copy_only=False)
    return x, codes, dates


def fold_roots(name: str) -> tuple[Path, Path, Path]:
    if name == "validation_2024":
        linear = PROCESSED / "model_confirmation_2024_linear" / name
        mlp = PROCESSED / "model_confirmation_2024_mlp" / name
    else:
        linear = PROCESSED / "model_tuning_factor_subsets" / name
        mlp = PROCESSED / "model_tuning_mlp_dropout_low_7fold" / name
    lightgbm = PROCESSED / "model_backtest_lightgbm_models" / name
    return linear, lightgbm, mlp


def linear_prediction(x: np.ndarray, root: Path) -> np.ndarray:
    stats = read_json(root / "preprocessing.json")
    transformed = transform(x.copy(), stats)
    model = np.load(root / "selected_at_least_6_model.npz")
    factors = [str(value) for value in model["factors"].tolist()]
    active = list(stats["active_factors"])
    indices = np.asarray([active.index(name) for name in factors], dtype=np.int64)
    result = (
        transformed[:, indices] @ model["coefficient"] + float(model["intercept"])
    ).astype(np.float32)
    del transformed
    return result


def lightgbm_prediction(x: np.ndarray, root: Path) -> np.ndarray:
    stats = read_json(root / "preprocessing.json")
    transformed = transform(x.copy(), stats)
    # LightGBM's native Windows file loader can fail on a path containing
    # Chinese characters. Python can read the same file safely, so load the
    # serialized model text instead of passing the path to the C++ layer.
    model_text = (root / "l1_0_l2_1_model.txt").read_text(encoding="utf-8")
    booster = lgb.Booster(model_str=model_text)
    result = booster.predict(transformed).astype(np.float32)
    del transformed, booster
    return result


def mlp_prediction(x: np.ndarray, root: Path, batch_size: int) -> np.ndarray:
    stats = read_json(root / "final_preprocessing.json")
    transformed = transform(x.copy(), stats)
    payload = torch.load(
        root / "rows_300000" / "mlp_model.pt", map_location="cpu", weights_only=False
    )
    model = MLP(
        int(payload["input_width"]),
        tuple(int(value) for value in payload["architecture"]),
        tuple(float(value) for value in payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    result = predict(model, transformed, torch.device("cpu"), batch_size)
    del transformed, model, payload
    return result


def save_signals(
    path: Path,
    codes: np.ndarray,
    dates: np.ndarray,
    linear: np.ndarray,
    lightgbm: np.ndarray,
    mlp: np.ndarray,
) -> None:
    temp = path.with_suffix(".tmp.parquet")
    pq.write_table(
        pa.table({
            "ts_code": pa.array(codes),
            "signal_date": pa.array(dates),
            "linear_prediction": pa.array(linear),
            "lightgbm_prediction": pa.array(lightgbm),
            "mlp_prediction": pa.array(mlp),
        }),
        temp,
        compression="zstd",
        row_group_size=100_000,
    )
    temp.replace(path)


def main() -> None:
    args = parse_args()
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError("threads和batch-size必须大于0")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free / 1024**3 < args.min_free_gb:
        raise RuntimeError("磁盘剩余空间低于安全阈值")
    folds = load_folds(Path(args.dataset_dir).expanduser().resolve(), args.folds)
    if any(int(fold["name"].rsplit("_", 1)[1]) >= 2025 for fold in folds):
        raise ValueError("本脚本禁止读取2025及以后Test年份")
    glob = feature_glob(Path(args.features_dir).expanduser().resolve())
    torch.set_num_threads(args.threads)
    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit={quote_sql(args.memory_limit)}")
    con.execute(f"SET temp_directory={quote_sql(str(output / '_duckdb_tmp'))}")
    con.execute("SET preserve_insertion_order=false")
    report_path = output / "signal_generation_report.json"
    report = read_json(report_path) if report_path.is_file() else {
        "generated_at": datetime.now().isoformat(), "test_used": False, "folds": []
    }
    reported = {row["name"] for row in report["folds"]}
    try:
        for index, fold in enumerate(folds, 1):
            name = fold["name"]
            destination = output / f"{name}.parquet"
            if destination.is_file():
                print(f"[{index}/{len(folds)}] {name}已有信号，跳过", flush=True)
                continue
            print(f"[{index}/{len(folds)}] {name}加载全股票特征……", flush=True)
            x, codes, dates = load_features(
                con, glob, fold["validation_start"], fold["validation_signal_before"]
            )
            linear_root, lightgbm_root, mlp_root = fold_roots(name)
            linear = linear_prediction(x, linear_root)
            lightgbm_pred = lightgbm_prediction(x, lightgbm_root)
            mlp = mlp_prediction(x, mlp_root, args.batch_size)
            save_signals(destination, codes, dates, linear, lightgbm_pred, mlp)
            row = {
                "name": name,
                "rows": int(len(codes)),
                "start": str(dates.min()),
                "end": str(dates.max()),
            }
            if name not in reported:
                report["folds"].append(row)
                reported.add(name)
            atomic_json(report, report_path)
            print(f"    保存{len(codes):,}行", flush=True)
            del x, codes, dates, linear, lightgbm_pred, mlp
            gc.collect()
    finally:
        con.close()
    report["completed"] = True
    atomic_json(report, report_path)


if __name__ == "__main__":
    main()
