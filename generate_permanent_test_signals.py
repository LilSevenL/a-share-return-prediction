"""使用冻结的最终三模型生成2025-2026永久Test全市场信号。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from build_features import FACTOR_COLUMNS
from train_mlp import MLP, predict
from training_common import PROJECT_DIR, atomic_json, quote_sql, transform


PROCESSED = PROJECT_DIR / "quant_data" / "processed"
MODELS = PROCESSED / "final_models_2011_2024"
LOCK = PROCESSED / "permanent_test" / "PERMANENT_TEST_OPENED.json"
OUTPUT = PROCESSED / "permanent_test" / "signals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=65_536)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lock() -> dict:
    if not LOCK.is_file():
        raise RuntimeError("缺少永久Test开启锁定记录")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "PERMANENT_TEST_OPENED":
        raise RuntimeError("永久Test状态异常")
    for name, expected in lock["frozen_file_sha256"].items():
        path = MODELS / name
        if sha256(path) != expected:
            raise RuntimeError(f"冻结文件在Test开启后发生变化：{name}")
    return lock


def load_year(con: duckdb.DuckDBPyConnection, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = str(PROCESSED / "features_50" / f"feature_year={year}" / "*.parquet").replace("\\", "/")
    factors = ", ".join(f"CAST({name} AS FLOAT) AS {name}" for name in FACTOR_COLUMNS)
    date_condition = "signal_date <= DATE '2026-08-20'" if year == 2026 else "TRUE"
    table = con.execute(f"""
        SELECT ts_code, signal_date, {factors}
        FROM read_parquet({quote_sql(path)}, hive_partitioning=true)
        WHERE ({date_condition}) AND feature_valid_count >= 30
        ORDER BY signal_date, ts_code
    """).to_arrow_table()
    x = np.empty((table.num_rows, len(FACTOR_COLUMNS)), dtype=np.float32)
    for index, name in enumerate(FACTOR_COLUMNS):
        x[:, index] = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    codes = np.asarray(table["ts_code"].combine_chunks().to_pylist(), dtype="U12")
    dates = table["signal_date"].combine_chunks().to_numpy(zero_copy_only=False)
    return x, codes, dates


def save(path: Path, codes: np.ndarray, dates: np.ndarray, linear: np.ndarray,
         mlp: np.ndarray, lightgbm: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.table({
        "ts_code": pa.array(codes), "signal_date": pa.array(dates),
        "linear_prediction": pa.array(linear.astype(np.float32, copy=False)),
        "mlp_prediction": pa.array(mlp.astype(np.float32, copy=False)),
        "lightgbm_prediction": pa.array(lightgbm.astype(np.float32, copy=False)),
    }), temporary, compression="zstd", row_group_size=100_000)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError("threads和batch-size必须为正数")
    lock = verify_lock()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((MODELS / "preprocessing.json").read_text(encoding="utf-8"))
    linear = np.load(MODELS / "linear_elastic_net_19.npz", allow_pickle=False)
    linear_indices = np.asarray([
        stats["active_factors"].index(str(name)) for name in linear["factors"]
    ])
    checkpoint = torch.load(MODELS / "mlp_primary.pt", map_location="cpu", weights_only=False)
    mlp_model = MLP(
        checkpoint["input_width"], tuple(checkpoint["architecture"]), tuple(checkpoint["dropout"])
    )
    mlp_model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cpu")
    mlp_model.to(device)
    torch.set_num_threads(args.threads)
    booster = lgb.Booster(model_str=(MODELS / "lightgbm_challenger.txt").read_text(encoding="utf-8"))
    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    records = []
    for year in [2025, 2026]:
        print(f"生成{year}永久Test信号……", flush=True)
        x, codes, dates = load_year(con, year)
        x = transform(x, stats)
        linear_pred = (
            x[:, linear_indices] @ linear["coefficient"] + float(linear["intercept"])
        ).astype(np.float32)
        mlp_pred = predict(mlp_model, x, device, args.batch_size)
        lgb_pred = booster.predict(x).astype(np.float32)
        for name, values in [("linear", linear_pred), ("mlp", mlp_pred), ("lightgbm", lgb_pred)]:
            if not np.isfinite(values).all():
                raise RuntimeError(f"{year} {name}预测含无效值")
        destination = output / f"test_signals_{year}.parquet"
        save(destination, codes, dates, linear_pred, mlp_pred, lgb_pred)
        records.append({
            "year": year, "rows": int(len(codes)),
            "start": str(dates.min()), "end": str(dates.max()),
            "file": str(destination), "sha256": sha256(destination),
        })
        print(f"    {len(codes):,}行", flush=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "permanent_test_opened_recorded_at": lock["recorded_at"],
        "model_roles": {"primary": "mlp", "baseline": "linear", "strong_challenger": "lightgbm"},
        "models_verified_against_lock": True,
        "test_signal_period": "2025-01-02 to 2026-08-20",
        "records": records,
        "completed": True,
    }
    atomic_json(report, output / "signal_generation_report.json")
    con.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
