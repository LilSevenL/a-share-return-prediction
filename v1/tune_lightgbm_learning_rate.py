"""LightGBM第二阶段优化：学习率与最大训练轮数。

固定第一阶段选出的树结构，对多个learning_rate给予充分的轮数上限，
通过每折早停观察实际最佳轮数，再为领先学习率推荐合理的max_rounds。
默认只使用2017-2023，2024确认折和2025-2026 Test均不读取。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from training_common import (
    PROJECT_DIR,
    add_common_args,
    atomic_json,
    dataset_glob,
    evaluate,
    fit_preprocessor,
    load_folds,
    load_matrix,
    load_progress,
    open_connection,
    prepare_output,
    select_sql,
    transform,
    validate_common_args,
)
from tune_ols import DEFAULT_TUNING_FOLDS


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_lightgbm_learning_rate"
)
DEFAULT_LEARNING_RATES = [0.01, 0.02, 0.03, 0.05, 0.08]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=DEFAULT_LEARNING_RATES
    )
    parser.add_argument("--num-leaves", type=int, default=1023)
    parser.add_argument("--min-data-in-leaf", type=int, default=6000)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-l1", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--early-stopping", type=int, default=75)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    args = parser.parse_args()
    if args.folds is None:
        args.folds = DEFAULT_TUNING_FOLDS
    return args


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args)
    if not args.learning_rates or any(
        not np.isfinite(value) or value <= 0 for value in args.learning_rates
    ):
        raise ValueError("learning-rates必须是有限正数")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        raise ValueError("learning-rates中不能有重复值")
    if args.num_leaves < 2 or args.min_data_in_leaf <= 0:
        raise ValueError("num-leaves至少为2，min-data-in-leaf必须大于0")
    if args.rounds <= args.early_stopping or args.early_stopping <= 0:
        raise ValueError("rounds必须大于early-stopping，且二者必须为正数")
    if not 0 < args.feature_fraction <= 1 or not 0 < args.bagging_fraction <= 1:
        raise ValueError("feature-fraction和bagging-fraction必须在(0,1]内")
    if args.stability_penalty < 0:
        raise ValueError("stability-penalty不能为负数")


def learning_rate_id(value: float) -> str:
    return f"lr_{value:.12g}".replace("-", "m").replace(".", "p")


def recommended_round_cap(best_iterations: np.ndarray, patience: int) -> int:
    """给最慢收敛折留出完整早停窗口，并向上取整到50轮。"""
    required = int(best_iterations.max()) + patience
    return max(50, int(math.ceil(required / 50.0) * 50))


def aggregate(results: list[dict[str, Any]], penalty: float, patience: int) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    rates = sorted({float(row["learning_rate"]) for row in results})
    for rate in rates:
        rows = [row for row in results if float(row["learning_rate"]) == rate]
        rank_ics = np.asarray(
            [row["metrics"]["daily_rank_ic_mean"] for row in rows], dtype=np.float64
        )
        ics = np.asarray(
            [row["metrics"]["daily_ic_mean"] for row in rows], dtype=np.float64
        )
        spreads = np.asarray(
            [row["metrics"]["daily_top_bottom_logret_mean"] for row in rows],
            dtype=np.float64,
        )
        rmses = np.asarray(
            [row["metrics"]["rmse"] for row in rows], dtype=np.float64
        )
        iterations = np.asarray(
            [row["best_iteration"] for row in rows], dtype=np.int64
        )
        rank_ic_std = float(rank_ics.std(ddof=0))
        ranking.append({
            "learning_rate": rate,
            "folds": len(rows),
            "mean_daily_ic": float(ics.mean()),
            "mean_daily_rank_ic": float(rank_ics.mean()),
            "std_yearly_rank_ic": rank_ic_std,
            "worst_year_rank_ic": float(rank_ics.min()),
            "positive_year_ratio": float((rank_ics > 0).mean()),
            "mean_top_bottom_logret": float(spreads.mean()),
            "mean_rmse": float(rmses.mean()),
            "min_best_iteration": int(iterations.min()),
            "median_best_iteration": float(np.median(iterations)),
            "mean_best_iteration": float(iterations.mean()),
            "max_best_iteration": int(iterations.max()),
            "recommended_max_rounds": recommended_round_cap(iterations, patience),
            "selection_score": float(rank_ics.mean() - penalty * rank_ic_std),
        })
    return sorted(ranking, key=lambda row: row["selection_score"], reverse=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    glob = dataset_glob(dataset_dir)
    folds = load_folds(dataset_dir, args.folds)
    if any(fold["name"] == "validation_2024" for fold in folds):
        print("提示：显式包含了2024确认折；默认优化不会使用该折。", flush=True)

    experiment_spec = {
        "folds": [fold["name"] for fold in folds],
        "learning_rates": args.learning_rates,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "rounds": args.rounds,
        "early_stopping": args.early_stopping,
        "seed": args.seed,
    }
    progress_path = output_dir / "tuning_progress.json"
    progress = load_progress(progress_path)
    if "experiment_spec" in progress and progress["experiment_spec"] != experiment_spec:
        raise RuntimeError("现有进度属于不同实验；请改输出目录或使用--force")
    progress["experiment_spec"] = experiment_spec
    completed = set(progress["completed"])
    con = open_connection(output_dir, args)
    started = time.time()
    base_params = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "feature_pre_filter": False,
        "num_threads": args.threads,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "data_random_seed": args.seed,
        "verbosity": -1,
    }
    try:
        for fold_number, fold in enumerate(folds, 1):
            name = fold["name"]
            pending = [
                rate for rate in args.learning_rates
                if f"{name}/{learning_rate_id(rate)}" not in completed
            ]
            if not pending:
                print(f"[{fold_number}/{len(folds)}] {name}全部完成，跳过", flush=True)
                continue
            print(
                f"[{fold_number}/{len(folds)}] 加载{name}，待运行{len(pending)}组……",
                flush=True,
            )
            x_train, y_train, _, _ = load_matrix(
                con, glob, select_sql(fold, "train"), args.max_train_rows, False
            )
            x_valid, y_valid, _, dates = load_matrix(
                con, glob, select_sql(fold, "validation"),
                args.max_validation_rows, True,
            )
            assert dates is not None
            preprocessing = fit_preprocessor(x_train, args)
            x_train = transform(x_train, preprocessing)
            x_valid = transform(x_valid, preprocessing)
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            train_set = lgb.Dataset(
                x_train, label=y_train, feature_name=preprocessing["active_factors"]
            )
            valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)

            for rate in pending:
                identifier = learning_rate_id(rate)
                task_id = f"{name}/{identifier}"
                candidate_started = time.time()
                booster = lgb.train(
                    {**base_params, "learning_rate": rate},
                    train_set,
                    num_boost_round=args.rounds,
                    valid_sets=[valid_set],
                    callbacks=[
                        lgb.early_stopping(args.early_stopping, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
                pred = booster.predict(
                    x_valid, num_iteration=booster.best_iteration
                ).astype(np.float32)
                metrics = evaluate(pred, y_valid, dates)
                detail = {
                    "fold": name,
                    "config_id": identifier,
                    "learning_rate": rate,
                    "maximum_rounds": args.rounds,
                    "best_iteration": int(booster.best_iteration),
                    "hit_round_limit": int(booster.best_iteration) >= args.rounds,
                    "elapsed_seconds": round(time.time() - candidate_started, 2),
                    "metrics": metrics,
                }
                atomic_json(detail, fold_dir / f"{identifier}_metrics.json")
                progress["completed"].append(task_id)
                progress["results"].append(detail)
                completed.add(task_id)
                atomic_json(progress, progress_path)
                print(
                    f"    lr={rate:g}: iter={booster.best_iteration} "
                    f"Rank IC={metrics['daily_rank_ic_mean']:.6f} "
                    f"RMSE={metrics['rmse']:.6f}",
                    flush=True,
                )
                del booster, pred
                gc.collect()

            del train_set, valid_set, x_train, y_train, x_valid, y_valid, dates
            gc.collect()
    finally:
        con.close()

    ranking = aggregate(
        progress["results"], args.stability_penalty, args.early_stopping
    )
    expected_folds = len(folds)
    if any(row["folds"] != expected_folds for row in ranking):
        raise RuntimeError("存在未完成候选，已保存进度；重跑相同命令即可续传")
    if any(row["recommended_max_rounds"] > args.rounds for row in ranking):
        raise RuntimeError("轮数上限不足以容纳完整早停窗口，请提高--rounds后重跑")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "lightgbm_learning_rate_and_rounds_stage_2",
        "dataset_dir": str(dataset_dir),
        "target": "target_logret_5d",
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "fixed_parameters": base_params,
        "learning_rates": args.learning_rates,
        "exploration_maximum_rounds": args.rounds,
        "early_stopping_rounds": args.early_stopping,
        "round_cap_rule": (
            "ceil((maximum best_iteration across folds + early_stopping_rounds) / 50) * 50"
        ),
        "selection_rule": (
            "mean_daily_rank_ic - stability_penalty * std_yearly_rank_ic"
        ),
        "stability_penalty": args.stability_penalty,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "best": ranking[0],
        "ranking": ranking,
    }
    atomic_json(summary, output_dir / "tuning_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
