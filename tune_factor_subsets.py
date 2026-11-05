"""第十阶段：比较全部50因子与跨折稳定入选因子子集。"""

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
from tune_ols import DEFAULT_TUNING_FOLDS
from tune_weight_exponent import make_power_weights


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_factor_subsets"
)
DEFAULT_SUBSET_FILE = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_elastic_refined"
    / "stable_factor_subsets.json"
)
WEIGHT_GAMMA = 0.75
L1_RATIO = 1e-7
ALPHA_RATIO = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-file", default=str(DEFAULT_SUBSET_FILE))
    parser.add_argument(
        "--subsets", nargs="+",
        choices=("all_50", "selected_all_7", "selected_at_least_6"),
        default=None,
        help="只运行指定因子子集；默认运行全部三个预定义子集",
    )
    parser.add_argument("--gamma", type=float, default=WEIGHT_GAMMA)
    parser.add_argument("--l1-ratio", type=float, default=L1_RATIO)
    parser.add_argument("--alpha-ratio", type=float, default=ALPHA_RATIO)
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
    if not 0 < args.l1_ratio < 1 or not 0 < args.alpha_ratio < 1:
        raise ValueError("l1-ratio和alpha-ratio必须在0到1之间")
    if args.chunk_rows <= 0 or args.max_iterations <= 0 or args.tolerance <= 0:
        raise ValueError("chunk-rows、max-iterations和tolerance必须大于0")
    if args.stability_penalty < 0:
        raise ValueError("stability-penalty不能为负数")


def load_subsets(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少稳定因子定义：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    subsets = {
        "all_50": [],  # 在每折预处理完成后替换为全部有效因子
        "selected_all_7": list(payload["selected_all_7"]),
        "selected_at_least_6": list(payload["selected_at_least_6"]),
    }
    if not subsets["selected_all_7"] or not subsets["selected_at_least_6"]:
        raise ValueError("稳定因子子集不能为空")
    return subsets, payload


def aggregate(results: list[dict[str, Any]], penalty: float) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for subset in sorted({row["subset"] for row in results}):
        rows = [row for row in results if row["subset"] == subset]
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
            "subset": subset,
            "factor_count": int(rows[0]["factor_count"]),
            "factors": rows[0]["factors"],
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
    subset_file = Path(args.subset_file).expanduser().resolve()
    subsets, subset_source = load_subsets(subset_file)
    if args.subsets is not None:
        subsets = {name: subsets[name] for name in args.subsets}
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
            active_factors = list(preprocessing["active_factors"])
            factor_to_index = {factor: index for index, factor in enumerate(active_factors)}
            fold_subsets = {**subsets, "all_50": active_factors}
            missing = sorted({
                factor for factors in fold_subsets.values() for factor in factors
                if factor not in factor_to_index
            })
            if missing:
                raise RuntimeError(f"{name} 缺少稳定因子：{missing}")
            weights, weight_report = make_power_weights(train_dates, args.gamma)
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(preprocessing, fold_dir / "preprocessing.json")
            atomic_json(weight_report, fold_dir / "sample_weighting.json")
            fold_results: list[dict[str, Any]] = []
            for subset_name, factors in fold_subsets.items():
                indices = np.asarray([factor_to_index[factor] for factor in factors])
                subset_train = np.ascontiguousarray(x_train[:, indices])
                subset_valid = np.ascontiguousarray(x_valid[:, indices])
                stats = weighted_sufficient_statistics(
                    subset_train, y_train, weights, args.chunk_rows
                )
                max_abs_rhs = float(np.max(np.abs(stats["rhs"])))
                alpha_max = max_abs_rhs / args.l1_ratio
                alpha = args.alpha_ratio * alpha_max
                model = fit_elastic_net_from_gram(
                    stats, alpha, args.l1_ratio,
                    np.zeros(len(factors), dtype=np.float64),
                    args.max_iterations, args.tolerance,
                )
                pred = (
                    subset_valid @ model["coefficient"] + model["intercept"]
                ).astype(np.float32)
                metrics = evaluate(pred, y_valid, valid_dates)
                np.savez(
                    fold_dir / f"{subset_name}_model.npz",
                    subset=subset_name, gamma=args.gamma,
                    l1_ratio=args.l1_ratio, alpha_ratio=args.alpha_ratio,
                    alpha=alpha, alpha_max=alpha_max,
                    intercept=model["intercept"], coefficient=model["coefficient"],
                    factors=factors,
                )
                detail = {
                    "subset": subset_name, "factor_count": len(factors),
                    "factors": factors, "nonzero_factors": model["nonzero_factors"],
                    "iterations": model["iterations"],
                    "converged": model["converged"], "metrics": metrics,
                }
                atomic_json(detail, fold_dir / f"{subset_name}_metrics.json")
                fold_results.append({"fold": name, **detail})
                print(
                    f"    {subset_name}({len(factors)}) Rank IC="
                    f"{metrics['daily_rank_ic_mean']:.6f}", flush=True,
                )
                del subset_train, subset_valid, stats
            progress["completed"].append(name)
            progress["results"].extend(fold_results)
            completed.add(name)
            atomic_json(progress, progress_path)
            del x_train, y_train, train_dates, x_valid, y_valid, valid_dates, weights
            gc.collect()
    finally:
        con.close()
    ranking = aggregate(progress["results"], args.stability_penalty)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "stable_factor_subsets_stage_10",
        "dataset_dir": str(dataset_dir),
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": any(
            fold["name"] == "validation_2024" for fold in folds
        ),
        "test_used": False,
        "subset_source_file": str(subset_file),
        "subset_source_config_id": subset_source["source_config_id"],
        "gamma": args.gamma,
        "l1_ratio": args.l1_ratio,
        "alpha_ratio": args.alpha_ratio,
        "stability_penalty": args.stability_penalty,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "best": ranking[0],
        "ranking": ranking,
    }
    atomic_json(summary, output_dir / "tuning_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
