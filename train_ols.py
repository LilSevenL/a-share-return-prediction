"""单独滚动训练OLS多元线性回归；永久Test不会被读取。"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from training_common import (
    FACTOR_COLUMNS, PROJECT_DIR, add_common_args, aggregate_results, atomic_json,
    dataset_glob, evaluate, fit_preprocessor, load_folds, load_matrix,
    load_progress, open_connection, prepare_output, save_predictions, select_sql,
    transform, validate_common_args,
)


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "quant_data" / "processed" / "model_training_ols"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    return parser.parse_args()


def train_ols(x: np.ndarray, y: np.ndarray, chunk: int) -> dict[str, Any]:
    width = x.shape[1]
    xtx = np.zeros((width, width), dtype=np.float64)
    xty = np.zeros(width, dtype=np.float64)
    # 预处理已将X近似中心化；使用Training目标均值作为截距。
    intercept = float(y.mean())
    for start in range(0, len(y), chunk):
        xb = x[start : start + chunk].astype(np.float64, copy=False)
        yb = y[start : start + chunk].astype(np.float64, copy=False) - intercept
        xtx += xb.T @ xb
        xty += xb.T @ yb
    coefficient = np.linalg.pinv(xtx, rcond=1e-10) @ xty
    return {"intercept": intercept, "coefficient": coefficient}


def main() -> None:
    args = parse_args()
    validate_common_args(args)
    if args.chunk_rows <= 0:
        raise ValueError("chunk-rows 必须大于0")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    glob = dataset_glob(dataset_dir)
    folds = load_folds(dataset_dir, args.folds)
    progress_path = output_dir / "training_progress.json"
    progress = load_progress(progress_path)
    completed = set(progress["completed"])
    con = open_connection(output_dir, args)
    started = time.time()
    try:
        for number, fold in enumerate(folds, 1):
            name = fold["name"]
            if name in completed:
                print(f"[{number}/{len(folds)}] {name} 已完成，跳过", flush=True)
                continue
            print(f"[{number}/{len(folds)}] OLS加载 {name}……", flush=True)
            x_train, y_train, _, _ = load_matrix(
                con, glob, select_sql(fold, "train"), args.max_train_rows, False
            )
            x_valid, y_valid, codes, dates = load_matrix(
                con, glob, select_sql(fold, "validation"),
                args.max_validation_rows, True,
            )
            assert codes is not None and dates is not None
            stats = fit_preprocessor(x_train, args)
            x_train, x_valid = transform(x_train, stats), transform(x_valid, stats)
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(stats, fold_dir / "preprocessing.json")
            model = train_ols(x_train, y_train, args.chunk_rows)
            pred = (x_valid @ model["coefficient"] + model["intercept"]).astype(
                np.float32
            )
            np.savez(
                fold_dir / "ols_model.npz", intercept=model["intercept"],
                coefficient=model["coefficient"],
                active_factors=stats["active_factors"],
            )
            metrics = evaluate(pred, y_valid, dates)
            save_predictions(
                fold_dir / "ols_predictions.parquet", codes, dates, y_valid, pred
            )
            atomic_json(metrics, fold_dir / "ols_metrics.json")
            progress["completed"].append(name)
            progress["results"].append({"fold": name, "metrics": metrics})
            completed.add(name)
            atomic_json(progress, progress_path)
            print(
                f"    Training={len(y_train):,} Validation={len(y_valid):,} "
                f"Rank IC={metrics['daily_rank_ic_mean']:.6f}", flush=True,
            )
            del x_train, y_train, x_valid, y_valid, codes, dates
            gc.collect()
    finally:
        con.close()
    summary = {
        "generated_at": datetime.now().isoformat(), "model": "ols",
        "dataset_dir": str(dataset_dir), "target": "target_logret_5d",
        "test_used": False, "folds_requested": [x["name"] for x in folds],
        "chunk_rows": args.chunk_rows,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "aggregate_validation_metrics": aggregate_results(progress["results"]),
    }
    atomic_json(summary, output_dir / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
