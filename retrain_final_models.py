"""用2011-2024全部非Test数据重训冻结的OLS、LightGBM与MLP。

MLP为主模型；OLS为基础Baseline；LightGBM为强对照。脚本显式只读取
final_split=training与final_split=validation，绝不打开final_split=test。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import torch

from train_mlp import choose_device, save_model
from training_common import (
    PROJECT_DIR,
    atomic_json,
    atomic_text,
    fit_preprocessor,
    open_connection,
    prepare_output,
    quote_sql,
    transform,
)
from tune_daily_weights import weighted_sufficient_statistics
from tune_elastic_net import fit_elastic_net_from_gram
from tune_mlp_training import fit_epochs, sample_indices
from tune_weight_exponent import make_power_weights


DATASET = PROJECT_DIR / "quant_data" / "processed" / "model_dataset"
OUTPUT = PROJECT_DIR / "quant_data" / "processed" / "final_models_2011_2024"
SUBSET_FILE = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_elastic_refined"
    / "stable_factor_subsets.json"
)
LGBM_HISTORY = (
    PROJECT_DIR / "quant_data" / "processed" / "model_backtest_lightgbm_models"
)
MLP_TUNING = (
    PROJECT_DIR / "quant_data" / "processed" / "model_tuning_mlp_dropout_low_7fold"
)
MLP_CONFIRMATION = (
    PROJECT_DIR / "quant_data" / "processed" / "model_confirmation_2024_mlp"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DATASET))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--preprocess-sample-rows", type=int, default=500_000)
    parser.add_argument("--mlp-train-rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def explicit_sources(dataset: Path) -> tuple[str, str]:
    training = dataset / "final_split=training" / "signal_year=*" / "*.parquet"
    validation = dataset / "final_split=validation" / "signal_year=*" / "*.parquet"
    if not next((dataset / "final_split=training").rglob("*.parquet"), None):
        raise FileNotFoundError(training)
    if not next((dataset / "final_split=validation").rglob("*.parquet"), None):
        raise FileNotFoundError(validation)
    return str(training).replace("\\", "/"), str(validation).replace("\\", "/")


def source_sql(paths: list[str]) -> str:
    return "[" + ", ".join(quote_sql(path) for path in paths) + "]"


def data_stats(con: duckdb.DuckDBPyConnection, sources: list[str]) -> dict[str, Any]:
    source = source_sql(sources)
    row = con.execute(f"""
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT ts_code) AS stocks,
               COUNT(DISTINCT signal_date) AS signal_days,
               MIN(signal_date) AS min_signal_date,
               MAX(signal_date) AS max_signal_date,
               MIN(exit_date) AS min_exit_date,
               MAX(exit_date) AS max_exit_date,
               AVG(target_logret_5d) AS target_mean,
               STDDEV_POP(target_logret_5d) AS target_std,
               APPROX_QUANTILE(target_logret_5d, 0.01) AS target_p01,
               APPROX_QUANTILE(target_logret_5d, 0.50) AS target_median,
               APPROX_QUANTILE(target_logret_5d, 0.99) AS target_p99,
               AVG(feature_valid_count) AS mean_valid_factors,
               MIN(feature_valid_count) AS min_valid_factors,
               MAX(feature_valid_count) AS max_valid_factors
        FROM read_parquet({source}, hive_partitioning=true)
    """).fetchone()
    names = [item[0] for item in con.description]
    result = dict(zip(names, row))
    yearly = con.execute(f"""
        SELECT YEAR(signal_date)::INTEGER AS year, COUNT(*) AS rows,
               COUNT(DISTINCT ts_code) AS stocks
        FROM read_parquet({source}, hive_partitioning=true)
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    result["yearly"] = [
        {"year": int(year), "rows": int(rows), "stocks": int(stocks)}
        for year, rows, stocks in yearly
    ]
    return result


def comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    numeric = ["rows", "stocks", "signal_days", "target_mean", "target_std",
               "target_p01", "target_median", "target_p99", "mean_valid_factors"]
    change: dict[str, Any] = {}
    for key in numeric:
        old, new = before[key], after[key]
        if old is None or new is None:
            continue
        change[key] = {
            "before": old,
            "after": new,
            "absolute_change": new - old,
            "percent_change": ((new / old) - 1.0) if old != 0 else None,
        }
    return change


def load_all_matrix(
    con: duckdb.DuckDBPyConnection, sources: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from build_features import FACTOR_COLUMNS
    factors = ", ".join(f"CAST({name} AS FLOAT) AS {name}" for name in FACTOR_COLUMNS)
    table = con.execute(f"""
        SELECT signal_date, {factors}, CAST(target_logret_5d AS FLOAT) AS target
        FROM read_parquet({source_sql(sources)}, hive_partitioning=true)
        ORDER BY signal_date, ts_code
    """).to_arrow_table()
    x = np.empty((table.num_rows, len(FACTOR_COLUMNS)), dtype=np.float32)
    for index, name in enumerate(FACTOR_COLUMNS):
        x[:, index] = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    y = table["target"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=True)
    dates = table["signal_date"].combine_chunks().to_numpy(zero_copy_only=False)
    return x, y, dates, list(FACTOR_COLUMNS)


def selected_rounds() -> tuple[int, list[int]]:
    values = []
    for path in sorted(LGBM_HISTORY.glob("validation_*/l1_0_l2_1_metrics.json")):
        values.append(int(json.loads(path.read_text(encoding="utf-8"))["best_iteration"]))
    if len(values) != 8:
        raise RuntimeError(f"LightGBM最佳轮数记录应为8折，实际{len(values)}折")
    # 当中位数位于两个整数epoch之间时向下取整，采用更保守的训练长度。
    return int(np.floor(float(np.median(values)))), values


def selected_epochs() -> tuple[int, list[int]]:
    paths = list(MLP_TUNING.glob("validation_*/rows_300000/metrics.json"))
    paths += list(MLP_CONFIRMATION.glob("validation_2024/rows_300000/metrics.json"))
    values = sorted(int(json.loads(path.read_text(encoding="utf-8"))["selected_epoch"]) for path in paths)
    if len(values) != 8:
        raise RuntimeError(f"MLP最佳epoch记录应为8折，实际{len(values)}折")
    return int(round(float(np.median(values)))), values


def file_info(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    args = parse_args()
    if args.threads <= 0 or args.preprocess_sample_rows <= 0 or args.mlp_train_rows <= 0:
        raise ValueError("threads和样本数必须为正数")
    dataset = Path(args.dataset_dir).expanduser().resolve()
    output = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    training_source, validation_source = explicit_sources(dataset)
    before_sources = [training_source]
    after_sources = [training_source, validation_source]
    helper_args = SimpleNamespace(
        threads=args.threads,
        memory_limit=args.memory_limit,
        seed=args.seed,
        preprocess_sample_rows=args.preprocess_sample_rows,
        min_feature_coverage=0.05,
        winsor_lower=0.01,
        winsor_upper=0.99,
    )
    con = open_connection(output, helper_args)
    started = time.time()
    print("统计重训前后数据，但不读取Test……", flush=True)
    before = data_stats(con, before_sources)
    after = data_stats(con, after_sources)
    data_report = {
        "generated_at": datetime.now().isoformat(),
        "before_definition": "2011-2023 final_split=training",
        "after_definition": "2011-2024 final_split=training + validation",
        "test_used": False,
        "before": before,
        "after": after,
        "comparison": comparison(before, after),
    }
    atomic_json(data_report, output / "data_before_after.json")
    print(f"重训前{before['rows']:,}行；重训后{after['rows']:,}行", flush=True)
    print("加载2011-2024全部非Test矩阵……", flush=True)
    x, y, dates, all_factors = load_all_matrix(con, after_sources)
    preprocessing = fit_preprocessor(x, helper_args)
    atomic_json(preprocessing, output / "preprocessing.json")
    x = transform(x, preprocessing)
    active_factors = list(preprocessing["active_factors"])
    if active_factors != all_factors:
        print(f"有效因子{len(active_factors)}/{len(all_factors)}", flush=True)

    artifacts: dict[str, Any] = {}
    configs: dict[str, Any] = {}

    # 1) 19因子Elastic Net线性Baseline。
    subset_payload = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    linear_factors = list(subset_payload["selected_at_least_6"])
    factor_to_index = {name: index for index, name in enumerate(active_factors)}
    missing = [name for name in linear_factors if name not in factor_to_index]
    if missing:
        raise RuntimeError(f"最终线性模型缺少因子：{missing}")
    linear_indices = np.asarray([factor_to_index[name] for name in linear_factors])
    weights, weight_report = make_power_weights(dates, 0.75)
    stats = weighted_sufficient_statistics(
        np.ascontiguousarray(x[:, linear_indices]), y, weights, 200_000
    )
    l1_ratio, alpha_ratio = 1e-7, 0.2
    alpha_max = float(np.max(np.abs(stats["rhs"]))) / l1_ratio
    alpha = alpha_ratio * alpha_max
    linear = fit_elastic_net_from_gram(
        stats, alpha, l1_ratio, np.zeros(len(linear_factors), dtype=np.float64),
        20_000, 1e-8,
    )
    linear_path = output / "linear_elastic_net_19.npz"
    np.savez(
        linear_path, factors=linear_factors, coefficient=linear["coefficient"],
        intercept=linear["intercept"], gamma=0.75, l1_ratio=l1_ratio,
        alpha_ratio=alpha_ratio, alpha=alpha, alpha_max=alpha_max,
    )
    configs["linear_baseline"] = {
        "factors": linear_factors, "factor_count": len(linear_factors),
        "gamma": 0.75, "l1_ratio": l1_ratio, "alpha_ratio": alpha_ratio,
        "alpha": alpha, "nonzero_factors": int(linear["nonzero_factors"]),
        "iterations": int(linear["iterations"]), "converged": bool(linear["converged"]),
        "sample_weighting": weight_report,
    }
    artifacts["linear_baseline"] = file_info(linear_path)
    del weights, stats, linear
    gc.collect()
    print("线性Baseline训练完成", flush=True)

    # 2) 主模型MLP：轮数使用8折内部早停epoch的中位数，样本上限保持冻结的30万。
    mlp_epochs, epoch_history = selected_epochs()
    mlp_indices = sample_indices(len(y), min(args.mlp_train_rows, len(y)), args.seed)
    mlp_args = SimpleNamespace(
        seed=args.seed, threads=args.threads, hidden_sizes=[128, 64, 32],
        dropout=[0.10, 0.075, 0.05], learning_rate=0.0005,
        weight_decay=1e-5, batch_size=4096,
    )
    device = choose_device("auto")
    mlp_model, mlp_history, _ = fit_epochs(
        np.ascontiguousarray(x[mlp_indices]), y[mlp_indices].copy(), mlp_args,
        device, mlp_epochs, None,
    )
    mlp_path = output / "mlp_primary.pt"
    save_model(mlp_model, mlp_path, active_factors)
    atomic_json(mlp_history, output / "mlp_training_history.json")
    sample_years, sample_counts = np.unique(
        dates[mlp_indices].astype("datetime64[Y]").astype(int) + 1970,
        return_counts=True,
    )
    configs["mlp_primary"] = {
        "architecture": [128, 64, 32], "dropout": [0.10, 0.075, 0.05],
        "learning_rate": 0.0005, "batch_size": 4096, "weight_decay": 1e-5,
        "seed": args.seed, "train_rows": int(len(mlp_indices)),
        "eligible_pool_rows": int(len(y)), "epochs": mlp_epochs,
        "validation_selected_epochs": epoch_history,
        "epoch_rule": "floor of median of 8 frozen validation-selected epochs",
        "sample_rows_by_year": {str(int(a)): int(b) for a, b in zip(sample_years, sample_counts)},
        "final_train_mse": float(mlp_history[-1]["train_mse"]),
        "device": str(device),
    }
    artifacts["mlp_primary"] = file_info(mlp_path)
    del mlp_model, mlp_history, mlp_indices
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"MLP主模型训练完成：{mlp_epochs} epochs", flush=True)

    # 3) LightGBM强对照：轮数使用8折最佳轮数的中位数，不使用Test早停。
    lgb_rounds, round_history = selected_rounds()
    params = {
        "objective": "regression", "metric": "l2", "num_leaves": 1023,
        "min_data_in_leaf": 6000, "learning_rate": 0.0075,
        "feature_fraction": 0.3, "bagging_fraction": 1.0, "bagging_freq": 0,
        "lambda_l1": 0.0, "lambda_l2": 1.0, "feature_pre_filter": False,
        "num_threads": args.threads, "seed": args.seed,
        "feature_fraction_seed": args.seed, "bagging_seed": args.seed,
        "data_random_seed": args.seed, "verbosity": -1,
    }
    train_set = lgb.Dataset(x, label=y, feature_name=active_factors)
    evaluation: dict[str, Any] = {}
    booster = lgb.train(
        params, train_set, num_boost_round=lgb_rounds, valid_sets=[train_set],
        valid_names=["train"], callbacks=[lgb.record_evaluation(evaluation), lgb.log_evaluation(0)],
    )
    lgb_path = output / "lightgbm_challenger.txt"
    atomic_text(booster.model_to_string(num_iteration=lgb_rounds), lgb_path)
    configs["lightgbm_challenger"] = {
        "parameters": params, "rounds": lgb_rounds,
        "validation_best_iterations": round_history,
        "round_rule": "rounded median of 8 frozen validation best iterations",
        "final_train_l2": float(evaluation["train"]["l2"][-1]),
    }
    artifacts["lightgbm_challenger"] = file_info(lgb_path)
    del booster, train_set, x, y, dates
    gc.collect()
    print(f"LightGBM强对照训练完成：{lgb_rounds} rounds", flush=True)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "roles": {
            "primary": "mlp_primary",
            "baseline": "linear_baseline",
            "strong_challenger": "lightgbm_challenger",
        },
        "dataset_dir": str(dataset),
        "training_period": "2011-2024",
        "explicitly_read_splits": ["training", "validation"],
        "test_used": False,
        "target": "target_logret_5d",
        "training_rows": int(after["rows"]),
        "training_stocks": int(after["stocks"]),
        "active_factor_count": len(active_factors),
        "configs": configs,
        "artifacts": artifacts,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "post_retrain_metrics": "not computed because no non-Test holdout remains; permanent Test stays sealed",
    }
    atomic_json(manifest, output / "final_training_manifest.json")
    con.close()
    shutil.rmtree(output / "_duckdb_tmp", ignore_errors=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
