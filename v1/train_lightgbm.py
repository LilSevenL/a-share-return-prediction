"""单独滚动训练LightGBM树模型；永久Test不会被读取。"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from training_common import (
    PROJECT_DIR, add_common_args, aggregate_results, atomic_json, atomic_text,
    dataset_glob, evaluate, fit_preprocessor, load_folds, load_matrix,
    load_progress, open_connection, prepare_output, save_predictions, select_sql,
    transform, validate_common_args,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_training_lightgbm"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=1000)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-l1", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    return parser.parse_args()


def validate_model_args(args: argparse.Namespace) -> None:
    if args.learning_rate <= 0 or args.num_leaves < 2:
        raise ValueError("learning-rate必须大于0，num-leaves至少为2")
    if args.min_data_in_leaf <= 0 or args.rounds <= 0 or args.early_stopping <= 0:
        raise ValueError("叶节点样本数、轮数和早停轮数必须大于0")
    if not 0 < args.feature_fraction <= 1 or not 0 < args.bagging_fraction <= 1:
        raise ValueError("feature-fraction和bagging-fraction必须在(0,1]内")


def main() -> None:
    args = parse_args()
    validate_common_args(args)
    validate_model_args(args)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    glob = dataset_glob(dataset_dir)
    folds = load_folds(dataset_dir, args.folds)
    progress_path = output_dir / "training_progress.json"
    progress = load_progress(progress_path)
    completed = set(progress["completed"])
    con = open_connection(output_dir, args)
    started = time.time()
    params = {
        "objective": "regression", "metric": "l2",
        "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction, "bagging_freq": 1,
        "lambda_l1": args.lambda_l1, "lambda_l2": args.lambda_l2,
        "num_threads": args.threads, "seed": args.seed, "verbosity": -1,
    }
    try:
        for number, fold in enumerate(folds, 1):
            name = fold["name"]
            if name in completed:
                print(f"[{number}/{len(folds)}] {name} 已完成，跳过", flush=True)
                continue
            print(f"[{number}/{len(folds)}] LightGBM加载 {name}……", flush=True)
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
            train_set = lgb.Dataset(
                x_train, label=y_train, feature_name=stats["active_factors"]
            )
            valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
            booster = lgb.train(
                params, train_set, num_boost_round=args.rounds,
                valid_sets=[valid_set],
                callbacks=[
                    lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)
                ],
            )
            atomic_text(
                booster.model_to_string(num_iteration=booster.best_iteration),
                fold_dir / "lightgbm_model.txt",
            )
            pred = booster.predict(x_valid, num_iteration=booster.best_iteration).astype(
                np.float32
            )
            metrics = evaluate(pred, y_valid, dates)
            metrics["best_iteration"] = int(booster.best_iteration)
            save_predictions(
                fold_dir / "lightgbm_predictions.parquet", codes, dates, y_valid, pred
            )
            atomic_json(metrics, fold_dir / "lightgbm_metrics.json")
            progress["completed"].append(name)
            progress["results"].append({"fold": name, "metrics": metrics})
            completed.add(name)
            atomic_json(progress, progress_path)
            print(
                f"    Training={len(y_train):,} Validation={len(y_valid):,} "
                f"Rank IC={metrics['daily_rank_ic_mean']:.6f}", flush=True,
            )
            del train_set, valid_set, booster, x_train, y_train, x_valid, y_valid
            del codes, dates
            gc.collect()
    finally:
        con.close()
    summary = {
        "generated_at": datetime.now().isoformat(), "model": "lightgbm",
        "dataset_dir": str(dataset_dir), "target": "target_logret_5d",
        "test_used": False, "folds_requested": [x["name"] for x in folds],
        "parameters": params, "maximum_rounds": args.rounds,
        "early_stopping_rounds": args.early_stopping,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "aggregate_validation_metrics": aggregate_results(progress["results"]),
    }
    atomic_json(summary, output_dir / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
