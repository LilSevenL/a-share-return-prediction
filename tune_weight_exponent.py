"""第八阶段：优化每日样本权重指数gamma。

单只股票权重 w_it = N_t ** (-gamma)。gamma=0为普通样本等权，gamma=1为
每个交易日总权重相等。固定此前Elastic Net参数，仅优化权重。默认不读取2024和Test。
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
    PROJECT_DIR, add_common_args, atomic_json, dataset_glob, evaluate,
    fit_preprocessor, load_folds, load_matrix, load_matrix_with_dates,
    load_progress, open_connection, prepare_output, select_sql, transform,
    validate_common_args,
)
from tune_daily_weights import (
    ELASTIC_ALPHA_RATIO, ELASTIC_L1_RATIO, weighted_sufficient_statistics,
)
from tune_elastic_net import fit_elastic_net_from_gram
from tune_ols import DEFAULT_TUNING_FOLDS


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_weight_exponent"
)
DEFAULT_GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
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
    if args.tolerance <= 0 or args.stability_penalty < 0:
        raise ValueError("tolerance必须大于0且stability-penalty不能为负")
    if not args.gammas or any(
        not np.isfinite(gamma) or gamma < 0 or gamma > 1
        for gamma in args.gammas
    ):
        raise ValueError("gammas必须是0到1之间的有限数")
    if len(set(args.gammas)) != len(args.gammas):
        raise ValueError("gammas中不能有重复值")


def make_power_weights(
    dates: np.ndarray, gamma: float
) -> tuple[np.ndarray, dict[str, Any]]:
    _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype(np.float64) ** (-gamma)
    date_total_weights = counts.astype(np.float64) ** (1.0 - gamma)
    return weights, {
        "gamma": gamma,
        "formula": "stock_weight = daily_stock_count ** (-gamma)",
        "trading_days": int(len(counts)),
        "minimum_daily_count": int(counts.min()),
        "maximum_daily_count": int(counts.max()),
        "minimum_date_total_weight": float(date_total_weights.min()),
        "maximum_date_total_weight": float(date_total_weights.max()),
        "max_to_min_date_total_weight_ratio": float(
            date_total_weights.max() / date_total_weights.min()
        ),
        "weight_sum": float(weights.sum()),
    }


def gamma_label(gamma: float) -> str:
    return f"gamma_{gamma:.12g}".replace(".", "p").replace("-", "m")


def aggregate(results: list[dict[str, Any]], penalty: float) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for gamma in sorted({float(row["gamma"]) for row in results}):
        rows = [row for row in results if float(row["gamma"]) == gamma]
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
        ranking.append({
            "gamma": gamma,
            "folds": len(rows),
            "mean_daily_ic": float(ics.mean()),
            "mean_daily_rank_ic": float(rank_ics.mean()),
            "std_yearly_rank_ic": std,
            "worst_year_rank_ic": float(rank_ics.min()),
            "positive_year_ratio": float((rank_ics > 0).mean()),
            "mean_top_bottom_logret": float(spreads.mean()),
            "mean_rmse": float(rmses.mean()),
            "mean_nonzero_factors": float(
                np.mean([row["nonzero_factors"] for row in rows])
            ),
            "all_converged": all(row["converged"] for row in rows),
            "selection_score": float(rank_ics.mean() - penalty * std),
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
        print("提示：显式包含2024；默认调参不会使用该折。", flush=True)
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
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            fold_results: list[dict[str, Any]] = []
            for gamma in args.gammas:
                weights, weight_report = make_power_weights(train_dates, gamma)
                stats = weighted_sufficient_statistics(
                    x_train, y_train, weights, args.chunk_rows
                )
                max_abs_rhs = float(np.max(np.abs(stats["rhs"])))
                alpha_max = max_abs_rhs / ELASTIC_L1_RATIO
                alpha = ELASTIC_ALPHA_RATIO * alpha_max
                model = fit_elastic_net_from_gram(
                    stats, alpha, ELASTIC_L1_RATIO,
                    np.zeros(len(stats["rhs"]), dtype=np.float64),
                    args.max_iterations, args.tolerance,
                )
                pred = (x_valid @ model["coefficient"] + model["intercept"]).astype(
                    np.float32
                )
                metrics = evaluate(pred, y_valid, valid_dates)
                label = gamma_label(gamma)
                np.savez(
                    fold_dir / f"{label}_model.npz",
                    gamma=gamma, l1_ratio=ELASTIC_L1_RATIO,
                    alpha_ratio=ELASTIC_ALPHA_RATIO, alpha=alpha,
                    alpha_max=alpha_max, intercept=model["intercept"],
                    coefficient=model["coefficient"],
                    active_factors=preprocessing["active_factors"],
                )
                detail = {
                    "gamma": gamma,
                    "weight_report": weight_report,
                    "nonzero_factors": model["nonzero_factors"],
                    "iterations": model["iterations"],
                    "converged": model["converged"],
                    "metrics": metrics,
                }
                atomic_json(detail, fold_dir / f"{label}_metrics.json")
                fold_results.append({"fold": name, **detail})
                print(
                    f"    gamma={gamma:g} Rank IC="
                    f"{metrics['daily_rank_ic_mean']:.6f} "
                    f"非零={model['nonzero_factors']}", flush=True,
                )
                del weights, stats
            progress["completed"].append(name)
            progress["results"].extend(fold_results)
            completed.add(name)
            atomic_json(progress, progress_path)
            del x_train, y_train, train_dates, x_valid, y_valid, valid_dates
            gc.collect()
    finally:
        con.close()
    ranking = aggregate(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "weight_exponent_stage_8",
        "dataset_dir": str(dataset_dir),
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "gammas": args.gammas,
        "weight_formula": "stock_weight = daily_stock_count ** (-gamma)",
        "fixed_l1_ratio": ELASTIC_L1_RATIO,
        "fixed_alpha_ratio": ELASTIC_ALPHA_RATIO,
        "stability_penalty": args.stability_penalty,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "best": ranking[0],
        "ranking": ranking,
    }
    atomic_json(summary, output_dir / "tuning_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
