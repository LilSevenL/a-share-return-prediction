"""第七阶段线性模型优化：比较原始收益与每日市场相对收益Target。

相对Target = 股票未来5日收益 - 同一signal_date全市场未来收益中位数。
该变换只作用于Training标签和Validation评估标签，不作为模型特征。
默认不读取2024确认折和2025-2026 Test。
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
from tune_daily_weights import (
    ELASTIC_ALPHA_RATIO,
    ELASTIC_L1_RATIO,
    RIDGE_ALPHA,
    make_daily_equal_weights,
    weighted_sufficient_statistics,
)
from tune_elastic_net import fit_elastic_net_from_gram
from tune_ols import DEFAULT_TUNING_FOLDS, fit_ridge


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_relative_target"
)
TARGET_MODES = ["raw", "daily_median_relative"]


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


def date_boundaries(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(dates) == 0:
        raise ValueError("日期数组不能为空")
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1]
    ends = np.r_[starts[1:], len(dates)]
    return starts, ends


def make_daily_median_relative_target(
    y: np.ndarray, dates: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """按日减去未来收益中位数；输入必须按日期排序。"""
    if not np.isfinite(y).all():
        raise ValueError("Target中存在非有限值")
    relative = y.astype(np.float32, copy=True)
    starts, ends = date_boundaries(dates)
    medians = np.empty(len(starts), dtype=np.float64)
    max_abs_post_median = 0.0
    for index, (start, end) in enumerate(zip(starts, ends)):
        median = float(np.median(relative[start:end].astype(np.float64, copy=False)))
        medians[index] = median
        relative[start:end] -= np.float32(median)
        max_abs_post_median = max(
            max_abs_post_median,
            abs(float(np.median(relative[start:end].astype(np.float64, copy=False)))),
        )
    report = {
        "method": "target_minus_same_signal_date_cross_sectional_median",
        "rows": int(len(y)),
        "trading_days": int(len(starts)),
        "original_mean": float(np.mean(y, dtype=np.float64)),
        "relative_mean": float(np.mean(relative, dtype=np.float64)),
        "daily_median_mean": float(medians.mean()),
        "daily_median_std": float(medians.std(ddof=0)),
        "daily_median_min": float(medians.min()),
        "daily_median_max": float(medians.max()),
        "max_abs_daily_median_after_transform": max_abs_post_median,
        "used_as_feature": False,
    }
    return relative, report


def aggregate(
    results: list[dict[str, Any]], stability_penalty: float
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    keys = sorted({(row["model"], row["target_mode"]) for row in results})
    for model_name, target_mode in keys:
        rows = [
            row
            for row in results
            if row["model"] == model_name and row["target_mode"] == target_mode
        ]
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
            "target_mode": target_mode,
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
            relative_train, train_relative_report = make_daily_median_relative_target(
                y_train, train_dates
            )
            relative_valid, valid_relative_report = make_daily_median_relative_target(
                y_valid, valid_dates
            )
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            atomic_json(
                {
                    "training": train_relative_report,
                    "validation": valid_relative_report,
                    "relative_target_is_never_a_feature": True,
                },
                fold_dir / "relative_target_report.json",
            )
            atomic_json(
                {
                    "method": "each_trading_day_total_weight_equals_one",
                    "training_rows": len(y_train),
                    "trading_days": trading_days,
                    "weight_sum": float(weights.sum()),
                },
                fold_dir / "sample_weighting.json",
            )

            fold_results: list[dict[str, Any]] = []
            for target_mode in TARGET_MODES:
                if target_mode == "raw":
                    fit_target, evaluation_target = y_train, y_valid
                else:
                    fit_target, evaluation_target = relative_train, relative_valid
                stats = weighted_sufficient_statistics(
                    x_train, fit_target, weights, args.chunk_rows
                )

                ridge = fit_ridge(stats, RIDGE_ALPHA)
                ridge_pred = (
                    x_valid @ ridge["coefficient"] + ridge["intercept"]
                ).astype(np.float32)
                ridge_metrics = evaluate(ridge_pred, evaluation_target, valid_dates)
                np.savez(
                    fold_dir / f"ridge_{target_mode}_model.npz",
                    target_mode=target_mode,
                    alpha=RIDGE_ALPHA,
                    intercept=ridge["intercept"],
                    coefficient=ridge["coefficient"],
                    active_factors=preprocessing["active_factors"],
                )
                atomic_json(
                    ridge_metrics,
                    fold_dir / f"ridge_{target_mode}_metrics.json",
                )
                fold_results.append(
                    {
                        "fold": name,
                        "model": "ridge_daily_equal",
                        "target_mode": target_mode,
                        "metrics": ridge_metrics,
                    }
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
                elastic_metrics = evaluate(
                    elastic_pred, evaluation_target, valid_dates
                )
                np.savez(
                    fold_dir / f"elastic_net_{target_mode}_model.npz",
                    target_mode=target_mode,
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
                    elastic_detail,
                    fold_dir / f"elastic_net_{target_mode}_metrics.json",
                )
                fold_results.append(
                    {
                        "fold": name,
                        "model": "elastic_net_daily_equal",
                        "target_mode": target_mode,
                        **elastic_detail,
                    }
                )
                print(
                    f"    target={target_mode} Ridge="
                    f"{ridge_metrics['daily_rank_ic_mean']:.6f}; Elastic Net="
                    f"{elastic_metrics['daily_rank_ic_mean']:.6f}",
                    flush=True,
                )
                del stats

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
                relative_train,
                relative_valid,
                weights,
            )
            gc.collect()
    finally:
        con.close()

    ranking = aggregate(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "daily_median_relative_target_stage_7",
        "dataset_dir": str(dataset_dir),
        "target": "target_logret_5d",
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "changed_component": "target_only",
        "target_modes": TARGET_MODES,
        "relative_target_definition": (
            "target_logret_5d minus same signal_date cross-sectional median"
        ),
        "relative_target_used_as_feature": False,
        "sample_weighting": "each_trading_day_total_weight_equals_one",
        "factor_transform": "training_global_zscore",
        "target_clipped": False,
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
