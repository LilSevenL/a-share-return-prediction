"""第九阶段：固定最佳gamma=0.75，局部精调Elastic Net并比较纯Ridge。"""

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
    PROJECT_DIR, add_common_args, atomic_json, dataset_glob, evaluate,
    fit_preprocessor, load_folds, load_matrix, load_matrix_with_dates,
    load_progress, open_connection, prepare_output, select_sql, transform,
    validate_common_args,
)
from tune_daily_weights import weighted_sufficient_statistics
from tune_elastic_net import fit_elastic_net_from_gram
from tune_ols import DEFAULT_TUNING_FOLDS, fit_ridge
from tune_weight_exponent import make_power_weights


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_elastic_refined"
)
WEIGHT_GAMMA = 0.75
DEFAULT_L1_RATIOS = [1e-7, 1e-6, 1e-5, 1e-4]
DEFAULT_ALPHA_RATIOS = [0.03, 0.05, 0.1, 0.2, 0.3]
DEFAULT_RIDGE_ALPHAS = [100.0, 300.0, 1000.0, 3000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l1-ratios", type=float, nargs="+", default=DEFAULT_L1_RATIOS)
    parser.add_argument(
        "--alpha-ratios", type=float, nargs="+", default=DEFAULT_ALPHA_RATIOS
    )
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=DEFAULT_RIDGE_ALPHAS
    )
    parser.add_argument("--gamma", type=float, default=WEIGHT_GAMMA)
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
    if not 0 <= args.gamma <= 1:
        raise ValueError("gamma必须在0到1之间")
    if args.chunk_rows <= 0 or args.max_iterations <= 0:
        raise ValueError("chunk-rows和max-iterations必须大于0")
    if args.tolerance <= 0 or args.stability_penalty < 0:
        raise ValueError("tolerance必须大于0且stability-penalty不能为负")
    for values, name in (
        (args.l1_ratios, "l1-ratios"),
        (args.alpha_ratios, "alpha-ratios"),
    ):
        if not values or any(
            not np.isfinite(value) or value <= 0 or value >= 1 for value in values
        ):
            raise ValueError(f"{name}必须位于0和1之间")
        if len(set(values)) != len(values):
            raise ValueError(f"{name}不能重复")
    if not args.ridge_alphas or any(
        not np.isfinite(value) or value <= 0 for value in args.ridge_alphas
    ):
        raise ValueError("ridge-alphas必须是正数")


def number_label(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def aggregate(results: list[dict[str, Any]], penalty: float) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for config_id in sorted({row["config_id"] for row in results}):
        rows = [row for row in results if row["config_id"] == config_id]
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
        rmses = np.asarray([row["metrics"]["rmse"] for row in rows], dtype=np.float64)
        std = float(rank_ics.std(ddof=0))
        first = rows[0]
        item: dict[str, Any] = {
            "config_id": config_id,
            "model": first["model"],
            "l1_ratio": first.get("l1_ratio", 0.0),
            "alpha_ratio": first.get("alpha_ratio"),
            "ridge_alpha": first.get("ridge_alpha"),
            "folds": len(rows),
            "mean_daily_ic": float(ics.mean()),
            "mean_daily_rank_ic": float(rank_ics.mean()),
            "std_yearly_rank_ic": std,
            "worst_year_rank_ic": float(rank_ics.min()),
            "positive_year_ratio": float((rank_ics > 0).mean()),
            "mean_top_bottom_logret": float(spreads.mean()),
            "mean_rmse": float(rmses.mean()),
            "selection_score": float(rank_ics.mean() - penalty * std),
        }
        if first["model"] == "elastic_net":
            item["mean_nonzero_factors"] = float(
                np.mean([row["nonzero_factors"] for row in rows])
            )
            item["all_converged"] = all(row["converged"] for row in rows)
        else:
            item["mean_nonzero_factors"] = float(len(rows[0]["active_factors"]))
            item["all_converged"] = True
        ranking.append(item)
    return sorted(ranking, key=lambda row: row["selection_score"], reverse=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    glob = dataset_glob(dataset_dir)
    folds = load_folds(dataset_dir, args.folds)
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
                con, glob, select_sql(fold, "validation"),
                args.max_validation_rows, True,
            )
            assert valid_dates is not None
            preprocessing = fit_preprocessor(x_train, args)
            x_train = transform(x_train, preprocessing)
            x_valid = transform(x_valid, preprocessing)
            weights, weight_report = make_power_weights(train_dates, args.gamma)
            stats = weighted_sufficient_statistics(
                x_train, y_train, weights, args.chunk_rows
            )
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            atomic_json(weight_report, fold_dir / "sample_weighting.json")
            fold_results: list[dict[str, Any]] = []

            for ridge_alpha in args.ridge_alphas:
                model = fit_ridge(stats, ridge_alpha)
                pred = (x_valid @ model["coefficient"] + model["intercept"]).astype(
                    np.float32
                )
                metrics = evaluate(pred, y_valid, valid_dates)
                config_id = f"ridge_alpha_{number_label(ridge_alpha)}"
                np.savez(
                    fold_dir / f"{config_id}_model.npz",
                    gamma=args.gamma, ridge_alpha=ridge_alpha,
                    intercept=model["intercept"], coefficient=model["coefficient"],
                    active_factors=preprocessing["active_factors"],
                )
                detail = {
                    "config_id": config_id, "model": "ridge",
                    "ridge_alpha": ridge_alpha,
                    "active_factors": preprocessing["active_factors"],
                    "metrics": metrics,
                }
                atomic_json(detail, fold_dir / f"{config_id}_metrics.json")
                fold_results.append({"fold": name, **detail})

            max_abs_rhs = float(np.max(np.abs(stats["rhs"])))
            for l1_ratio in args.l1_ratios:
                alpha_max = max_abs_rhs / l1_ratio
                coefficient = np.zeros(len(stats["rhs"]), dtype=np.float64)
                for alpha_ratio in sorted(args.alpha_ratios, reverse=True):
                    alpha = alpha_ratio * alpha_max
                    model = fit_elastic_net_from_gram(
                        stats, alpha, l1_ratio, coefficient,
                        args.max_iterations, args.tolerance,
                    )
                    coefficient = model["coefficient"]
                    pred = (x_valid @ coefficient + model["intercept"]).astype(
                        np.float32
                    )
                    metrics = evaluate(pred, y_valid, valid_dates)
                    config_id = (
                        f"elastic_l1_{number_label(l1_ratio)}_"
                        f"ratio_{number_label(alpha_ratio)}"
                    )
                    np.savez(
                        fold_dir / f"{config_id}_model.npz",
                        gamma=args.gamma, l1_ratio=l1_ratio,
                        alpha_ratio=alpha_ratio, alpha=alpha, alpha_max=alpha_max,
                        intercept=model["intercept"], coefficient=coefficient,
                        active_factors=preprocessing["active_factors"],
                    )
                    detail = {
                        "config_id": config_id, "model": "elastic_net",
                        "l1_ratio": l1_ratio, "alpha_ratio": alpha_ratio,
                        "nonzero_factors": model["nonzero_factors"],
                        "iterations": model["iterations"],
                        "converged": model["converged"], "metrics": metrics,
                    }
                    atomic_json(detail, fold_dir / f"{config_id}_metrics.json")
                    fold_results.append({"fold": name, **detail})
            best_fold = max(
                fold_results,
                key=lambda row: row["metrics"]["daily_rank_ic_mean"],
            )
            print(
                f"    本折最佳={best_fold['config_id']} Rank IC="
                f"{best_fold['metrics']['daily_rank_ic_mean']:.6f}", flush=True,
            )
            progress["completed"].append(name)
            progress["results"].extend(fold_results)
            completed.add(name)
            atomic_json(progress, progress_path)
            del x_train, y_train, train_dates, x_valid, y_valid, valid_dates, weights, stats
            gc.collect()
    finally:
        con.close()
    ranking = aggregate(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "elastic_net_refined_stage_9",
        "dataset_dir": str(dataset_dir),
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "gamma": args.gamma,
        "l1_ratios": args.l1_ratios,
        "alpha_ratios": args.alpha_ratios,
        "ridge_alphas": args.ridge_alphas,
        "stability_penalty": args.stability_penalty,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "best": ranking[0],
        "ranking": ranking,
    }
    atomic_json(summary, output_dir / "tuning_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
