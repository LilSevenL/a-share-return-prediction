"""第二阶段线性模型优化：在2017-2023滚动验证中调节Lasso稀疏度。

alpha按每折alpha_max的比例定义，以便不同训练窗口具有可比的稀疏强度。
本脚本默认不读取2024确认折，也不读取2025-2026 Test。
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
    load_progress,
    open_connection,
    prepare_output,
    select_sql,
    transform,
    validate_common_args,
)
from tune_ols import DEFAULT_TUNING_FOLDS, sufficient_statistics


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_lasso"
)
DEFAULT_ALPHA_RATIOS = [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    0.5,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alpha-ratios",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_RATIOS,
        help="每折使用 alpha = ratio * alpha_max",
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
    if not args.alpha_ratios:
        raise ValueError("至少需要一个alpha-ratio")
    if any(
        not np.isfinite(ratio) or ratio <= 0 or ratio >= 1
        for ratio in args.alpha_ratios
    ):
        raise ValueError("alpha-ratio必须是0和1之间的有限数")
    if len(set(args.alpha_ratios)) != len(args.alpha_ratios):
        raise ValueError("alpha-ratios中不能有重复值")


def soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_lasso_from_gram(
    stats: dict[str, Any],
    alpha: float,
    initial: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    """坐标下降求解 1/(2n)||y-Xb||^2 + alpha*||b||_1。"""
    gram = stats["gram"]
    rhs = stats["rhs"]
    coefficient = initial.astype(np.float64, copy=True)
    diagonal = np.diag(gram)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        max_change = 0.0
        for index in range(len(coefficient)):
            if diagonal[index] <= 1e-15:
                coefficient[index] = 0.0
                continue
            old = coefficient[index]
            partial = float(
                rhs[index] - (gram[index] @ coefficient - diagonal[index] * old)
            )
            new = soft_threshold(partial, alpha) / diagonal[index]
            coefficient[index] = new
            max_change = max(max_change, abs(new - old))
        scale = max(1.0, float(np.max(np.abs(coefficient))))
        if max_change <= tolerance * scale:
            converged = True
            break

    intercept = float(stats["mean_y"] - stats["mean_x"] @ coefficient)
    objective_without_constant = float(
        0.5 * coefficient @ gram @ coefficient
        - stats["rhs"] @ coefficient
        + alpha * np.abs(coefficient).sum()
    )
    return {
        "intercept": intercept,
        "coefficient": coefficient,
        "iterations": iteration,
        "converged": converged,
        "nonzero_factors": int(np.count_nonzero(np.abs(coefficient) > 1e-12)),
        "objective_without_constant": objective_without_constant,
    }


def ratio_label(ratio: float) -> str:
    return f"ratio_{ratio:.12g}".replace("-", "m").replace(".", "p")


def aggregate_by_ratio(
    results: list[dict[str, Any]], stability_penalty: float
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    ratios = sorted({float(item["alpha_ratio"]) for item in results})
    for ratio in ratios:
        rows = [item for item in results if float(item["alpha_ratio"]) == ratio]
        rank_ics = np.asarray(
            [item["metrics"]["daily_rank_ic_mean"] for item in rows],
            dtype=np.float64,
        )
        ics = np.asarray(
            [item["metrics"]["daily_ic_mean"] for item in rows], dtype=np.float64
        )
        spreads = np.asarray(
            [item["metrics"]["daily_top_bottom_logret_mean"] for item in rows],
            dtype=np.float64,
        )
        rmses = np.asarray(
            [item["metrics"]["rmse"] for item in rows], dtype=np.float64
        )
        rank_ic_std = float(rank_ics.std(ddof=0))
        ranking.append(
            {
                "alpha_ratio": ratio,
                "folds": len(rows),
                "mean_absolute_alpha": float(
                    np.mean([item["alpha"] for item in rows])
                ),
                "mean_nonzero_factors": float(
                    np.mean([item["nonzero_factors"] for item in rows])
                ),
                "min_nonzero_factors": int(
                    min(item["nonzero_factors"] for item in rows)
                ),
                "max_nonzero_factors": int(
                    max(item["nonzero_factors"] for item in rows)
                ),
                "all_converged": all(item["converged"] for item in rows),
                "max_iterations_used": int(max(item["iterations"] for item in rows)),
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
        )
    return sorted(ranking, key=lambda item: item["selection_score"], reverse=True)


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
            x_train, y_train, _, _ = load_matrix(
                con, glob, select_sql(fold, "train"), args.max_train_rows, False
            )
            x_valid, y_valid, _, dates = load_matrix(
                con,
                glob,
                select_sql(fold, "validation"),
                args.max_validation_rows,
                True,
            )
            assert dates is not None
            preprocessing = fit_preprocessor(x_train, args)
            x_train = transform(x_train, preprocessing)
            x_valid = transform(x_valid, preprocessing)
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            stats = sufficient_statistics(x_train, y_train, args.chunk_rows)
            alpha_max = float(np.max(np.abs(stats["rhs"])))
            if not np.isfinite(alpha_max) or alpha_max <= 0:
                raise RuntimeError(f"{name} 的alpha_max无效：{alpha_max}")

            # 从强正则到弱正则做warm start，可显著加快坐标下降。
            coefficient = np.zeros(len(stats["rhs"]), dtype=np.float64)
            fold_results: list[dict[str, Any]] = []
            for ratio in sorted(args.alpha_ratios, reverse=True):
                alpha = ratio * alpha_max
                model = fit_lasso_from_gram(
                    stats,
                    alpha,
                    coefficient,
                    args.max_iterations,
                    args.tolerance,
                )
                coefficient = model["coefficient"]
                pred = (x_valid @ coefficient + model["intercept"]).astype(np.float32)
                metrics = evaluate(pred, y_valid, dates)
                label = ratio_label(ratio)
                np.savez(
                    fold_dir / f"{label}_model.npz",
                    alpha_ratio=ratio,
                    alpha=alpha,
                    alpha_max=alpha_max,
                    intercept=model["intercept"],
                    coefficient=coefficient,
                    active_factors=preprocessing["active_factors"],
                )
                detail = {
                    "alpha_ratio": ratio,
                    "alpha": alpha,
                    "alpha_max": alpha_max,
                    "iterations": model["iterations"],
                    "converged": model["converged"],
                    "nonzero_factors": model["nonzero_factors"],
                    "objective_without_constant": model["objective_without_constant"],
                    "metrics": metrics,
                }
                atomic_json(detail, fold_dir / f"{label}_metrics.json")
                fold_results.append({"fold": name, **detail})
                print(
                    f"    ratio={ratio:g} 非零={model['nonzero_factors']:2d} "
                    f"Rank IC={metrics['daily_rank_ic_mean']:.6f} "
                    f"收敛={model['converged']}",
                    flush=True,
                )

            progress["completed"].append(name)
            progress["results"].extend(fold_results)
            completed.add(name)
            atomic_json(progress, progress_path)
            del x_train, y_train, x_valid, y_valid, dates, stats
            gc.collect()
    finally:
        con.close()

    ranking = aggregate_by_ratio(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "lasso_stage_2",
        "dataset_dir": str(dataset_dir),
        "target": "target_logret_5d",
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "alpha_ratios": args.alpha_ratios,
        "objective": "squared_error_over_2n_plus_alpha_l1",
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
