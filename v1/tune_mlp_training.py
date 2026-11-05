"""MLP第二阶段：Training内部按时间早停，再在外层滚动Validation评分。

默认仅运行2017-2023折。每折用验证年前一年作为内部早停集，
选择epoch后在完整Training上重新拟合；2024确认折和Test不参与搜索。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_mlp import MLP, choose_device, predict, save_model
from training_common import (
    PROJECT_DIR, add_common_args, atomic_json, daily_rank_ic, dataset_glob,
    evaluate, fit_preprocessor, load_folds, load_matrix,
    load_matrix_with_dates, load_progress, open_connection, prepare_output,
    quote_sql, save_predictions, select_sql, transform, validate_common_args,
)
from tune_ols import DEFAULT_TUNING_FOLDS


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "quant_data" / "processed" / "model_tuning_mlp_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-rows", type=int, nargs="+", default=[300_000])
    parser.add_argument("--inner-validation-rows", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.20, 0.15, 0.10])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    parser.add_argument(
        "--allow-confirmation-fold", action="store_true",
        help="显式允许只读validation_2024确认折；调参时不要使用",
    )
    args = parser.parse_args()
    if args.folds is None:
        args.folds = DEFAULT_TUNING_FOLDS
    return args


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args)
    if args.max_train_rows is not None:
        raise ValueError("此脚本不支持max-train-rows；请使用train-rows控制抽样")
    if args.max_validation_rows is not None:
        raise ValueError("此脚本要求全量外层Validation，不支持max-validation-rows")
    if not args.train_rows or any(value <= 0 for value in args.train_rows):
        raise ValueError("train-rows必须是正整数")
    if len(set(args.train_rows)) != len(args.train_rows):
        raise ValueError("train-rows不能重复")
    if args.inner_validation_rows <= 0:
        raise ValueError("inner-validation-rows必须为正数")
    if args.epochs <= args.patience or args.patience <= 0:
        raise ValueError("epochs必须大于patience且二者均为正数")
    if args.batch_size <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("batch-size、learning-rate必须为正数，weight-decay不能为负数")
    if len(args.hidden_sizes) != len(args.dropout) or not args.hidden_sizes:
        raise ValueError("hidden-sizes与dropout必须逐层对应且非空")
    if any(size <= 0 for size in args.hidden_sizes):
        raise ValueError("hidden-sizes必须为正整数")
    if any(not 0 <= rate < 1 for rate in args.dropout):
        raise ValueError("dropout比例必须在[0,1)内")


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)


def sample_indices(total: int, count: int, seed: int) -> np.ndarray:
    if total <= count:
        return np.arange(total)
    return np.random.default_rng(seed).choice(total, count, replace=False)


def fit_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    inner: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[MLP, list[dict[str, float]], int]:
    seed_everything(args.seed, args.threads)
    model = MLP(
        x_train.shape[1], tuple(args.hidden_sizes), tuple(args.dropout)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_fn = nn.MSELoss()
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
        row = {"epoch": epoch, "train_mse": loss_sum / len(y_train)}
        if inner is not None:
            x_inner, y_inner, dates_inner = inner
            score = daily_rank_ic(
                predict(model, x_inner, device, args.batch_size), y_inner, dates_inner
            )
            row["inner_rank_ic"] = score
            if math.isfinite(score) and score > best_score + 1e-7:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(row)
        if inner is not None:
            print(
                f"      epoch={epoch} train_mse={row['train_mse']:.6f} "
                f"inner_Rank_IC={row['inner_rank_ic']:.6f}", flush=True,
            )
            if stale >= args.patience:
                break
    if inner is not None and best_epoch == 0:
        raise RuntimeError("内部早停集没有有效Rank IC，不能安全选择epoch")
    return model, history, best_epoch if inner is not None else epochs


def aggregate(results: list[dict[str, Any]], penalty: float) -> list[dict[str, Any]]:
    ranking = []
    for train_rows in sorted({row["train_rows_limit"] for row in results}):
        rows = [row for row in results if row["train_rows_limit"] == train_rows]
        rank_ics = np.asarray(
            [row["metrics"]["daily_rank_ic_mean"] for row in rows], dtype=np.float64
        )
        std = float(rank_ics.std(ddof=0))
        ranking.append({
            "train_rows_limit": train_rows,
            "folds": len(rows),
            "mean_daily_ic": float(np.mean([row["metrics"]["daily_ic_mean"] for row in rows])),
            "mean_daily_rank_ic": float(rank_ics.mean()),
            "std_yearly_rank_ic": std,
            "worst_year_rank_ic": float(rank_ics.min()),
            "positive_year_ratio": float((rank_ics > 0).mean()),
            "mean_top_bottom_logret": float(np.mean([
                row["metrics"]["daily_top_bottom_logret_mean"] for row in rows
            ])),
            "mean_rmse": float(np.mean([row["metrics"]["rmse"] for row in rows])),
            "mean_selected_epoch": float(np.mean([row["selected_epoch"] for row in rows])),
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
    confirmation_fold_used = any(
        fold["name"] == "validation_2024" for fold in folds
    )
    if confirmation_fold_used and not args.allow_confirmation_fold:
        raise ValueError("2024是独立确认折，本调参脚本禁止读取")
    device = choose_device(args.device)
    spec = {
        "folds": [fold["name"] for fold in folds],
        "train_rows": args.train_rows,
        "inner_validation_rows": args.inner_validation_rows,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "seed": args.seed,
        "preprocess_sample_rows": args.preprocess_sample_rows,
        "min_feature_coverage": args.min_feature_coverage,
        "winsor_lower": args.winsor_lower,
        "winsor_upper": args.winsor_upper,
    }
    progress_path = output_dir / "tuning_progress.json"
    progress = load_progress(progress_path)
    if "experiment_spec" in progress:
        previous = progress["experiment_spec"]
        defaults = {
            "hidden_sizes": [64, 32, 16],
            "dropout": [0.20, 0.15, 0.10],
        }
        same_core = all(previous.get(key, defaults.get(key)) == value for key, value in spec.items())
        compatible_extras = (
            previous.get("max_train_rows") is None
            and previous.get("max_validation_rows") is None
            and previous.get("device", str(device)) == str(device)
            and set(previous).issubset(set(spec) | {
                "max_train_rows", "max_validation_rows", "device"
            })
        )
        if not (same_core and compatible_extras):
            raise RuntimeError("现有进度属于不同实验；请改输出目录或使用--force")
    progress["experiment_spec"] = spec
    completed = set(progress["completed"])
    con = open_connection(output_dir, args)
    started = time.time()
    scope = "显式读取2024确认折，不读取Test" if confirmation_fold_used else "内部早停，不读取2024/Test"
    print(f"MLP设备={device}; {scope}", flush=True)
    try:
        for fold_number, fold in enumerate(folds, 1):
            name = fold["name"]
            pending = [n for n in args.train_rows if f"{name}/rows_{n}" not in completed]
            if not pending:
                print(f"[{fold_number}/{len(folds)}] {name}已完成，跳过", flush=True)
                continue
            inner_year = int(name.rsplit("_", 1)[1]) - 1
            inner_start = f"{inner_year}0101"
            inner_end = f"{inner_year + 1}0101"
            cutoff = quote_sql(f"{inner_year}-01-01")
            base = select_sql(fold, "train")
            fit_condition = f"({base}) AND signal_date < DATE {cutoff} AND exit_date < DATE {cutoff}"
            holdout_condition = (
                f"({base}) AND signal_date >= DATE {cutoff}"
            )
            print(f"[{fold_number}/{len(folds)}] {name}: 内部早停年份={inner_year}", flush=True)
            x_early, y_early, _ = load_matrix_with_dates(con, glob, fit_condition, None)
            x_hold, y_hold, dates_hold = load_matrix_with_dates(
                con, glob, holdout_condition, None
            )
            early_sample = sample_indices(len(y_early), max(args.train_rows), args.seed)
            hold_sample = np.sort(sample_indices(
                len(y_hold), args.inner_validation_rows, args.seed
            ))
            inner_stats = fit_preprocessor(x_early, args)
            x_early_sample = transform(x_early[early_sample].copy(), inner_stats)
            x_hold_sample = transform(x_hold[hold_sample].copy(), inner_stats)
            y_early_sample = y_early[early_sample].copy()
            y_hold_sample = y_hold[hold_sample].copy()
            dates_hold_sample = dates_hold[hold_sample].copy()
            del x_early, y_early, x_hold, y_hold, dates_hold
            gc.collect()

            final_condition = select_sql(fold, "train")
            x_full, y_full, _ = load_matrix_with_dates(con, glob, final_condition, None)
            final_sample = sample_indices(len(y_full), max(args.train_rows), args.seed)
            final_stats = fit_preprocessor(x_full, args)
            x_final_sample = transform(x_full[final_sample].copy(), final_stats)
            y_final_sample = y_full[final_sample].copy()
            del x_full, y_full
            gc.collect()

            x_valid, y_valid, codes_valid, dates_valid = load_matrix(
                con, glob, select_sql(fold, "validation"),
                args.max_validation_rows, True,
            )
            assert codes_valid is not None and dates_valid is not None
            x_valid = transform(x_valid, final_stats)
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(inner_stats, fold_dir / "inner_preprocessing.json")
            atomic_json(final_stats, fold_dir / "final_preprocessing.json")
            for count in pending:
                task_id = f"{name}/rows_{count}"
                print(f"    样本上限={count:,}: 内部选epoch", flush=True)
                x_fit = x_early_sample[:count]
                y_fit = y_early_sample[:count]
                inner_model, history, selected_epoch = fit_epochs(
                    x_fit, y_fit, args, device, args.epochs,
                    (x_hold_sample, y_hold_sample, dates_hold_sample),
                )
                del inner_model
                gc.collect()
                print(f"    内部选中epoch={selected_epoch}; 完整Training重新拟合", flush=True)
                final_model, _, _ = fit_epochs(
                    x_final_sample[:count], y_final_sample[:count],
                    args, device, selected_epoch, None,
                )
                pred = predict(final_model, x_valid, device, args.batch_size)
                metrics = evaluate(pred, y_valid, dates_valid)
                candidate_dir = fold_dir / f"rows_{count}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                save_model(final_model, candidate_dir / "mlp_model.pt", final_stats["active_factors"])
                save_predictions(
                    candidate_dir / "validation_predictions.parquet",
                    codes_valid, dates_valid, y_valid, pred,
                )
                atomic_json(history, candidate_dir / "inner_history.json")
                detail = {
                    "fold": name,
                    "inner_validation_year": inner_year,
                    "inner_train_last_signal_before": inner_start,
                    "inner_validation_signal_before": inner_end,
                    "train_rows_limit": count,
                    "inner_train_rows": len(y_fit),
                    "inner_validation_rows": len(y_hold_sample),
                    "final_train_rows": min(count, len(y_final_sample)),
                    "selected_epoch": selected_epoch,
                    "hit_epoch_limit": selected_epoch >= args.epochs - args.patience,
                    "metrics": metrics,
                }
                atomic_json(detail, candidate_dir / "metrics.json")
                progress["completed"].append(task_id)
                progress["results"].append(detail)
                completed.add(task_id)
                atomic_json(progress, progress_path)
                print(f"    外层Rank IC={metrics['daily_rank_ic_mean']:.6f}", flush=True)
                del final_model, pred
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            del (
                x_early_sample, y_early_sample, x_hold_sample, y_hold_sample,
                dates_hold_sample, x_final_sample, y_final_sample, x_valid,
                y_valid, codes_valid, dates_valid,
            )
            gc.collect()
    finally:
        con.close()

    ranking = aggregate(progress["results"], args.stability_penalty)
    if any(row["folds"] != len(folds) for row in ranking):
        raise RuntimeError("存在未完成候选；已保存进度，重跑相同命令即可续传")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "mlp_chronological_inner_early_stopping",
        "dataset_dir": str(dataset_dir),
        "tuning_folds": [fold["name"] for fold in folds],
        "confirmation_fold_used": confirmation_fold_used,
        "test_used": False,
        "experiment_spec": spec,
        "stability_penalty": args.stability_penalty,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "best": ranking[0],
        "ranking": ranking,
    }
    atomic_json(summary, output_dir / "tuning_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
