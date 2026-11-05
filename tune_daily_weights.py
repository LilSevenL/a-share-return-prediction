"""第四阶段线性模型优化：比较每日等权的Ridge与Elastic Net。

只改变Training损失中的样本权重：每个交易日的全部股票权重之和相等。
保持此前的预处理与候选超参数不变，以便进行可解释的消融比较。
默认不读取2024确认折，也不读取2025-2026 Test。
"""

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
    PROJECT_DIR,
    add_common_args,
    atomic_json,
    dataset_glob,
    evaluate,
    fit_preprocessor,
    load_folds,
    load_matrix,
    load_matrix_with_dates,
    load_progress,
    open_connection,
    prepare_output,
    select_sql,
    transform,
    validate_common_args,
)
from tune_elastic_net import fit_elastic_net_from_gram
from tune_ols import DEFAULT_TUNING_FOLDS, fit_ridge


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_daily_weights"
)
RIDGE_ALPHA = 1000.0
ELASTIC_L1_RATIO = 0.000001
ELASTIC_ALPHA_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    args = parser.parse_args()
    if args.folds is None:
        args.folds = DEFAULT_TUNING_FOLDS
    return args


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args)
    if args.chunk_rows <= 0 or args.max_iterations <= 0:
        raise ValueError("chunk-rows和max-iterations必须大于0")
    if args.tolerance <= 0:
        raise ValueError("tolerance必须大于0")
    if args.stability_penalty < 0:
        raise ValueError("stability-penalty不能为负数")


def make_daily_equal_weights(dates: np.ndarray) -> tuple[np.ndarray, int]:
    """令每个日期权重和为1，因此每个交易日对损失贡献相同。"""
    _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    weights = np.reciprocal(counts[inverse].astype(np.float64))
    return weights, len(counts)


def weighted_sufficient_statistics(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    chunk_rows: int,
) -> dict[str, Any]:
    """累计每日等权的中心化Gram矩阵与X'y。"""
    width = x.shape[1]
    xtwx = np.zeros((width, width), dtype=np.float64)
    xtwy = np.zeros(width, dtype=np.float64)
    sum_wx = np.zeros(width, dtype=np.float64)
    sum_wy = 0.0
    sum_w = 0.0
    for start in range(0, len(y), chunk_rows):
        stop = start + chunk_rows
        xb = x[start:stop].astype(np.float64, copy=False)
        yb = y[start:stop].astype(np.float64, copy=False)
        wb = weights[start:stop]
        weighted_x = xb * wb[:, None]
        xtwx += xb.T @ weighted_x
        xtwy += xb.T @ (wb * yb)
        sum_wx += weighted_x.sum(axis=0)
        sum_wy += float(np.dot(wb, yb))
        sum_w += float(wb.sum())

    mean_x = sum_wx / sum_w
    mean_y = sum_wy / sum_w
    gram = (xtwx - np.outer(sum_wx, sum_wx) / sum_w) / sum_w
    rhs = (xtwy - sum_wx * sum_wy / sum_w) / sum_w
    gram = (gram + gram.T) * 0.5
    return {
        "rows": len(y),
        "weight_sum": float(sum_w),
        "mean_x": mean_x,
        "mean_y": float(mean_y),
        "gram": gram,
        "rhs": rhs,
    }


def aggregate(
    results: list[dict[str, Any]], stability_penalty: float
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for model_name in sorted({row["model"] for row in results}):
        rows = [row for row in results if row["model"] == model_name]
        rank_ics = np.asarray(
            [row["metrics"]["daily_rank_ic_mean"] for row in rows],
            dtype=np.float64,
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
        rank_ic_std = float(rank_ics.std(ddof=0))
        item: dict[str, Any] = {
            "model": model_name,
            "folds": len(rows),
            "mean_daily_ic": float(ics.mean()),
            "mean_daily_rank_ic": float(rank_ics.mean()),
            "std_yearly_rank_ic": rank_ic_std,
            "worst_year_rank_ic": float(rank_ics.min()),
            "positive_year_ratio": float((rank_ics > 0).mean()),
            "mean_top_bottom_logret": float(spreads.mean()),
            "mean_rmse": float(rmses.mean()),
            "selection_score": float(
                rank_ics.mean() - stability_penalty * rank_ic_std
            ),
        }
        if model_name == "elastic_net_daily_equal":
            item["mean_nonzero_factors"] = float(
                np.mean([row["nonzero_factors"] for row in rows])
            )
            item["all_converged"] = all(row["converged"] for row in rows)
        ranking.append(item)
    return sorted(ranking, key=lambda row: row["selection_score"], reverse=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    glob = dataset_glob(dataset_dir)
    folds = load_folds(dataset_dir, args.folds)
    if any(fold["name"] == "validation_2024" for fold in folds):
        print("提示：本次显式包含2024确认折；默认调参不会使用该折。", flush=True)

    progress_path = output_dir / "tuning_progress.json"
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
            print(f"[{number}/{len(folds)}] 加载 {name}……", flush=True)
            x_train, y_train, train_dates = load_matrix_with_dates(
                con, glob, select_sql(fold, "train"), args.max_train_rows
            )
            x_valid, y_valid, _, valid_dates = load_matrix(
                con,
                glob,
                select_sql(fold, "validation"),
                args.max_validation_rows,
                True,
            )
            assert valid_dates is not None
            preprocessing = fit_preprocessor(x_train, args)
            x_train = transform(x_train, preprocessing)
            x_valid = transform(x_valid, preprocessing)
            weights, trading_days = make_daily_equal_weights(train_dates)
            stats = weighted_sufficient_statistics(
                x_train, y_train, weights, args.chunk_rows
            )
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            atomic_json(
                {
                    "method": "each_trading_day_total_weight_equals_one",
                    "training_rows": len(y_train),
                    "trading_days": trading_days,
                    "weight_sum": stats["weight_sum"],
                    "minimum_daily_count": int(
                        np.unique(train_dates, return_counts=True)[1].min()
                    ),
                    "maximum_daily_count": int(
                        np.unique(train_dates, return_counts=True)[1].max()
                    ),
                },
                fold_dir / "sample_weighting.json",
            )

            fold_results: list[dict[str, Any]] = []
            ridge = fit_ridge(stats, RIDGE_ALPHA)
            ridge_pred = (
                x_valid @ ridge["coefficient"] + ridge["intercept"]
            ).astype(np.float32)
            ridge_metrics = evaluate(ridge_pred, y_valid, valid_dates)
            np.savez(
                fold_dir / "ridge_daily_equal_model.npz",
                alpha=RIDGE_ALPHA,
                intercept=ridge["intercept"],
                coefficient=ridge["coefficient"],
                active_factors=preprocessing["active_factors"],
            )
            atomic_json(ridge_metrics, fold_dir / "ridge_daily_equal_metrics.json")
            fold_results.append(
                {"fold": name, "model": "ridge_daily_equal", "metrics": ridge_metrics}
            )

            max_abs_rhs = float(np.max(np.abs(stats["rhs"])))
            alpha_max = max_abs_rhs / ELASTIC_L1_RATIO
            elastic_alpha = ELASTIC_ALPHA_RATIO * alpha_max
            elastic = fit_elastic_net_from_gram(
                stats,
                elastic_alpha,
                ELASTIC_L1_RATIO,
                np.zeros(len(stats["rhs"]), dtype=np.float64),
                args.max_iterations,
                args.tolerance,
            )
            elastic_pred = (
                x_valid @ elastic["coefficient"] + elastic["intercept"]
            ).astype(np.float32)
            elastic_metrics = evaluate(elastic_pred, y_valid, valid_dates)
            np.savez(
                fold_dir / "elastic_net_daily_equal_model.npz",
                l1_ratio=ELASTIC_L1_RATIO,
                alpha_ratio=ELASTIC_ALPHA_RATIO,
                alpha=elastic_alpha,
                alpha_max=alpha_max,
                intercept=elastic["intercept"],
                coefficient=elastic["coefficient"],
                active_factors=preprocessing["active_factors"],
            )
            elastic_detail = {
                "metrics": elastic_metrics,
                "nonzero_factors": elastic["nonzero_factors"],
                "iterations": elastic["iterations"],
                "converged": elastic["converged"],
            }
            atomic_json(
                elastic_detail, fold_dir / "elastic_net_daily_equal_metrics.json"
            )
            fold_results.append(
                {
                    "fold": name,
                    "model": "elastic_net_daily_equal",
                    **elastic_detail,
                }
            )
            print(
                f"    Ridge Rank IC={ridge_metrics['daily_rank_ic_mean']:.6f}; "
                f"Elastic Net Rank IC={elastic_metrics['daily_rank_ic_mean']:.6f} "
                f"非零={elastic['nonzero_factors']}",
                flush=True,
            )

            progress["completed"].append(name)
            progress["results"].extend(fold_results)
            completed.add(name)
            atomic_json(progress, progress_path)
            del (
                x_train,
                y_train,
                train_dates,
                x_valid,
                y_valid,
                valid_dates,
                weights,
                stats,
            )
            gc.collect()
    finally:
        con.close()

    ranking = aggregate(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "daily_equal_weight_stage_4",
        "dataset_dir": str(dataset_dir),
        "target": "target_logret_5d",
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "changed_component": "training_loss_sample_weights_only",
        "sample_weighting": "each_trading_day_total_weight_equals_one",
        "preprocessing_weighted": False,
        "fixed_parameters": {
            "ridge_alpha": RIDGE_ALPHA,
            "elastic_l1_ratio": ELASTIC_L1_RATIO,
            "elastic_alpha_ratio": ELASTIC_ALPHA_RATIO,
        },
        "selection_rule": (
            "mean_daily_rank_ic - stability_penalty * std_yearly_rank_ic"
        ),
        "stability_penalty": args.stability_penalty,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "ranking": ranking,
    }
    atomic_json(summary, output_dir / "tuning_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
