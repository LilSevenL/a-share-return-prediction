"""第六阶段线性模型优化：测试每日截面因子Z-score。

缺失值填充、Training缩尾边界和有效因子选择保持原口径；仅将全局均值/标准差
替换为每个signal_date内的截面均值/标准差。默认不读取2024确认折和Test。
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
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_cross_sectional"
)


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


def transform_cross_sectional(
    x: np.ndarray,
    dates: np.ndarray,
    stats: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Training规则填充/缩尾后，按每个交易日做截面Z-score。"""
    active = np.asarray(stats["active"], dtype=bool)
    for index in range(x.shape[1]):
        column = x[:, index]
        bad = ~np.isfinite(column)
        if bad.any():
            column[bad] = stats["median"][index]
        np.clip(column, stats["lower"][index], stats["upper"][index], out=column)
    x = np.ascontiguousarray(x[:, active], dtype=np.float32)

    starts, ends = date_boundaries(dates)
    zero_variance_cells = 0
    max_abs_daily_mean = 0.0
    for start, end in zip(starts, ends):
        block = x[start:end]
        mean = block.mean(axis=0, dtype=np.float64)
        std = block.std(axis=0, dtype=np.float64)
        good = np.isfinite(std) & (std > 1e-12)
        zero_variance_cells += int((~good).sum())
        safe_std = np.where(good, std, 1.0)
        block -= mean.astype(np.float32)
        block /= safe_std.astype(np.float32)
        if (~good).any():
            block[:, ~good] = 0.0
        max_abs_daily_mean = max(
            max_abs_daily_mean,
            float(np.max(np.abs(block.mean(axis=0, dtype=np.float64)))),
        )
    counts = ends - starts
    report = {
        "method": "training_fill_and_clip_then_daily_cross_sectional_zscore",
        "trading_days": int(len(starts)),
        "minimum_daily_count": int(counts.min()),
        "maximum_daily_count": int(counts.max()),
        "zero_variance_factor_days": zero_variance_cells,
        "max_abs_daily_mean_after_transform": max_abs_daily_mean,
        "active_factor_count": int(active.sum()),
    }
    return x, report


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
        if model_name == "elastic_net_cross_sectional":
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
            x_train, train_transform = transform_cross_sectional(
                x_train, train_dates, preprocessing
            )
            x_valid, valid_transform = transform_cross_sectional(
                x_valid, valid_dates, preprocessing
            )
            weights, trading_days = make_daily_equal_weights(train_dates)
            stats = weighted_sufficient_statistics(
                x_train, y_train, weights, args.chunk_rows
            )
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            atomic_json(
                {
                    "training": train_transform,
                    "validation": valid_transform,
                    "no_future_dates_used": True,
                    "validation_uses_training_fill_and_clip_thresholds": True,
                },
                fold_dir / "cross_sectional_transform.json",
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

            ridge = fit_ridge(stats, RIDGE_ALPHA)
            ridge_pred = (
                x_valid @ ridge["coefficient"] + ridge["intercept"]
            ).astype(np.float32)
            ridge_metrics = evaluate(ridge_pred, y_valid, valid_dates)
            np.savez(
                fold_dir / "ridge_cross_sectional_model.npz",
                alpha=RIDGE_ALPHA,
                intercept=ridge["intercept"],
                coefficient=ridge["coefficient"],
                active_factors=preprocessing["active_factors"],
            )
            atomic_json(
                ridge_metrics, fold_dir / "ridge_cross_sectional_metrics.json"
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
                fold_dir / "elastic_net_cross_sectional_model.npz",
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
                elastic_detail, fold_dir / "elastic_net_cross_sectional_metrics.json"
            )

            fold_results = [
                {
                    "fold": name,
                    "model": "ridge_cross_sectional",
                    "metrics": ridge_metrics,
                },
                {
                    "fold": name,
                    "model": "elastic_net_cross_sectional",
                    **elastic_detail,
                },
            ]
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
        "experiment": "daily_cross_sectional_zscore_stage_6",
        "dataset_dir": str(dataset_dir),
        "target": "target_logret_5d",
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "changed_component": "factor_centering_and_scaling_only",
        "factor_transform": (
            "training_fill_and_clip_then_daily_cross_sectional_zscore"
        ),
        "validation_cross_section_uses_same_signal_date_only": True,
        "sample_weighting": "each_trading_day_total_weight_equals_one",
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
