"""第五阶段线性模型优化：比较Training Target缩尾强度。

保持每日等权、预处理和候选模型参数不变。缩尾阈值只由每折Training
目标计算；Validation目标不修改。默认不读取2024确认折和2025-2026 Test。
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
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_target_winsor"
)
DEFAULT_WINSOR_LEVELS = [0.0, 0.005, 0.01]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-winsor-levels",
        type=float,
        nargs="+",
        default=DEFAULT_WINSOR_LEVELS,
        help="对称缩尾比例；0表示不缩尾，0.005表示0.5%%/99.5%%",
    )
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
    if not args.target_winsor_levels:
        raise ValueError("target-winsor-levels不能为空")
    if any(
        not np.isfinite(level) or level < 0 or level >= 0.5
        for level in args.target_winsor_levels
    ):
        raise ValueError("target缩尾比例必须位于0到0.5之间")
    if len(set(args.target_winsor_levels)) != len(args.target_winsor_levels):
        raise ValueError("target-winsor-levels中不能有重复值")


def winsor_label(level: float) -> str:
    if level == 0:
        return "none"
    return f"q_{level:.12g}".replace("-", "m").replace(".", "p")


def clip_training_target(
    y: np.ndarray, level: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if level == 0:
        return y, {
            "level": 0.0,
            "lower_quantile": 0.0,
            "upper_quantile": 1.0,
            "lower_value": None,
            "upper_value": None,
            "clipped_rows": 0,
            "clipped_ratio": 0.0,
        }
    lower, upper = np.quantile(y.astype(np.float64, copy=False), [level, 1.0 - level])
    clipped_rows = int(np.count_nonzero((y < lower) | (y > upper)))
    clipped = np.clip(y, lower, upper).astype(np.float32, copy=False)
    return clipped, {
        "level": level,
        "lower_quantile": level,
        "upper_quantile": 1.0 - level,
        "lower_value": float(lower),
        "upper_value": float(upper),
        "clipped_rows": clipped_rows,
        "clipped_ratio": float(clipped_rows / len(y)),
    }


def aggregate(
    results: list[dict[str, Any]], stability_penalty: float
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    keys = sorted({(row["model"], float(row["winsor_level"])) for row in results})
    for model_name, winsor_level in keys:
        rows = [
            row
            for row in results
            if row["model"] == model_name
            and float(row["winsor_level"]) == winsor_level
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
            "winsor_level": winsor_level,
            "folds": len(rows),
            "mean_clipped_ratio": float(
                np.mean([row["target_clipping"]["clipped_ratio"] for row in rows])
            ),
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
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
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
            for level in args.target_winsor_levels:
                fit_target, clipping = clip_training_target(y_train, level)
                label = winsor_label(level)
                stats = weighted_sufficient_statistics(
                    x_train, fit_target, weights, args.chunk_rows
                )
                atomic_json(clipping, fold_dir / f"target_clipping_{label}.json")

                ridge = fit_ridge(stats, RIDGE_ALPHA)
                ridge_pred = (
                    x_valid @ ridge["coefficient"] + ridge["intercept"]
                ).astype(np.float32)
                ridge_metrics = evaluate(ridge_pred, y_valid, valid_dates)
                np.savez(
                    fold_dir / f"ridge_{label}_model.npz",
                    winsor_level=level,
                    alpha=RIDGE_ALPHA,
                    intercept=ridge["intercept"],
                    coefficient=ridge["coefficient"],
                    active_factors=preprocessing["active_factors"],
                )
                atomic_json(
                    ridge_metrics, fold_dir / f"ridge_{label}_metrics.json"
                )
                fold_results.append(
                    {
                        "fold": name,
                        "model": "ridge_daily_equal",
                        "winsor_level": level,
                        "target_clipping": clipping,
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
                elastic_metrics = evaluate(elastic_pred, y_valid, valid_dates)
                np.savez(
                    fold_dir / f"elastic_net_{label}_model.npz",
                    winsor_level=level,
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
                    fold_dir / f"elastic_net_{label}_metrics.json",
                )
                fold_results.append(
                    {
                        "fold": name,
                        "model": "elastic_net_daily_equal",
                        "winsor_level": level,
                        "target_clipping": clipping,
                        **elastic_detail,
                    }
                )
                print(
                    f"    winsor={level:g} Ridge="
                    f"{ridge_metrics['daily_rank_ic_mean']:.6f}; Elastic Net="
                    f"{elastic_metrics['daily_rank_ic_mean']:.6f}",
                    flush=True,
                )
                del fit_target, stats

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
            )
            gc.collect()
    finally:
        con.close()

    ranking = aggregate(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "target_winsor_stage_5",
        "dataset_dir": str(dataset_dir),
        "target": "target_logret_5d",
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "changed_component": "training_target_clipping_only",
        "validation_target_clipped": False,
        "sample_weighting": "each_trading_day_total_weight_equals_one",
        "target_winsor_levels": args.target_winsor_levels,
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
