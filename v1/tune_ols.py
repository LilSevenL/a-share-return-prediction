"""第一阶段线性模型优化：在2017-2023滚动验证中比较OLS与Ridge。

本脚本只读取各滚动折的Training和Validation，不读取2025-2026 Test。
2024 Validation也默认保留给下一阶段确认，不参与alpha选择。
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


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_ols_ridge"
)
DEFAULT_TUNING_FOLDS = [f"validation_{year}" for year in range(2017, 2024)]
DEFAULT_ALPHAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Ridge强度；0代表无正则的OLS",
    )
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument(
        "--stability-penalty",
        type=float,
        default=0.25,
        help="选择分数 = 平均Rank IC - 此系数×年度Rank IC标准差",
    )
    args = parser.parse_args()
    if args.folds is None:
        args.folds = DEFAULT_TUNING_FOLDS
    return args


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args)
    if args.chunk_rows <= 0:
        raise ValueError("chunk-rows 必须大于0")
    if args.stability_penalty < 0:
        raise ValueError("stability-penalty 不能为负数")
    if not args.alphas:
        raise ValueError("至少需要一个alpha")
    if any(not np.isfinite(alpha) or alpha < 0 for alpha in args.alphas):
        raise ValueError("alpha必须是有限的非负数")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("alphas中不能有重复值")


def sufficient_statistics(
    x: np.ndarray, y: np.ndarray, chunk_rows: int
) -> dict[str, Any]:
    """分块累计统计量，避免额外复制完整Training矩阵。"""
    width = x.shape[1]
    xtx = np.zeros((width, width), dtype=np.float64)
    xty = np.zeros(width, dtype=np.float64)
    sum_x = np.zeros(width, dtype=np.float64)
    sum_y = 0.0
    for start in range(0, len(y), chunk_rows):
        xb = x[start : start + chunk_rows].astype(np.float64, copy=False)
        yb = y[start : start + chunk_rows].astype(np.float64, copy=False)
        xtx += xb.T @ xb
        xty += xb.T @ yb
        sum_x += xb.sum(axis=0)
        sum_y += float(yb.sum())

    n = len(y)
    mean_x = sum_x / n
    mean_y = sum_y / n
    # 使用中心化后的协方差尺度，使alpha不会随样本数量机械变化。
    gram = (xtx - np.outer(sum_x, sum_x) / n) / n
    rhs = (xty - sum_x * sum_y / n) / n
    gram = (gram + gram.T) * 0.5
    return {
        "rows": n,
        "mean_x": mean_x,
        "mean_y": float(mean_y),
        "gram": gram,
        "rhs": rhs,
    }


def fit_ridge(stats: dict[str, Any], alpha: float) -> dict[str, Any]:
    gram = stats["gram"]
    system = gram + np.eye(len(gram), dtype=np.float64) * alpha
    if alpha == 0:
        coefficient = np.linalg.pinv(system, rcond=1e-10) @ stats["rhs"]
    else:
        try:
            coefficient = np.linalg.solve(system, stats["rhs"])
        except np.linalg.LinAlgError:
            coefficient = np.linalg.pinv(system, rcond=1e-10) @ stats["rhs"]
    intercept = float(stats["mean_y"] - stats["mean_x"] @ coefficient)
    return {"intercept": intercept, "coefficient": coefficient}


def alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:.12g}".replace("-", "m").replace(".", "p")


def aggregate_by_alpha(
    results: list[dict[str, Any]], stability_penalty: float
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    alphas = sorted({float(item["alpha"]) for item in results})
    for alpha in alphas:
        rows = [item for item in results if float(item["alpha"]) == alpha]
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
        records.append(
            {
                "alpha": alpha,
                "model": "ols" if alpha == 0 else "ridge",
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
        )
    return sorted(records, key=lambda item: item["selection_score"], reverse=True)


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

            fold_results: list[dict[str, Any]] = []
            for alpha in args.alphas:
                model = fit_ridge(stats, alpha)
                pred = (
                    x_valid @ model["coefficient"] + model["intercept"]
                ).astype(np.float32)
                metrics = evaluate(pred, y_valid, dates)
                label = alpha_label(alpha)
                np.savez(
                    fold_dir / f"{label}_model.npz",
                    alpha=alpha,
                    intercept=model["intercept"],
                    coefficient=model["coefficient"],
                    active_factors=preprocessing["active_factors"],
                )
                atomic_json(metrics, fold_dir / f"{label}_metrics.json")
                record = {"fold": name, "alpha": alpha, "metrics": metrics}
                fold_results.append(record)
                print(
                    f"    alpha={alpha:g} Rank IC="
                    f"{metrics['daily_rank_ic_mean']:.6f} "
                    f"RMSE={metrics['rmse']:.6f}",
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

    ranking = aggregate_by_alpha(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "ols_vs_ridge_stage_1",
        "dataset_dir": str(dataset_dir),
        "target": "target_logret_5d",
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "alphas": args.alphas,
        "ridge_objective": "centered_gram_over_n_plus_alpha_identity",
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
