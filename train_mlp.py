"""单独滚动训练MLP神经网络；永久Test不会被读取。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training_common import (
    PROJECT_DIR, add_common_args, aggregate_results, atomic_json, daily_rank_ic,
    dataset_glob, evaluate, fit_preprocessor, load_folds, load_matrix,
    load_progress, open_connection, prepare_output, save_predictions, select_sql,
    transform, validate_common_args,
)


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "quant_data" / "processed" / "model_training_mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--mlp-max-train-rows", type=int, default=300_000,
        help="每折从完整Training固定随机抽取的最大行数；设为0使用全部",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_model_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs、patience和batch-size必须大于0")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate必须大于0，weight-decay不能小于0")
    if args.mlp_max_train_rows < 0:
        raise ValueError("mlp-max-train-rows不能小于0")


class MLP(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_sizes: tuple[int, ...] = (64, 32, 16),
        dropout: tuple[float, ...] = (0.20, 0.15, 0.10),
    ) -> None:
        super().__init__()
        if not hidden_sizes or len(hidden_sizes) != len(dropout):
            raise ValueError("隐藏层宽度与Dropout比例必须非空且逐层对应")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("隐藏层宽度必须为正整数")
        if any(not 0 <= rate < 1 for rate in dropout):
            raise ValueError("Dropout比例必须在[0,1)内")
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = tuple(dropout)
        layers: list[nn.Module] = []
        input_width = width
        for index, (size, rate) in enumerate(zip(self.hidden_sizes, self.dropout)):
            layers.extend((nn.Linear(input_width, size), nn.ReLU()))
            if index < len(self.hidden_sizes) - 1:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(rate))
            input_width = size
        layers.append(nn.Linear(input_width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def choose_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但PyTorch未检测到可用GPU")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def predict(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            output[start : start + len(xb)] = model(xb).cpu().numpy()
    return output


def train_mlp(
    x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
    y_valid: np.ndarray, dates: np.ndarray, args: argparse.Namespace,
    device: torch.device,
) -> tuple[MLP, list[dict[str, float]]]:
    model = MLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.MSELoss()
    best_score, stale = -float("inf"), 0
    best_state = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
            seen += len(xb)
        valid_pred = predict(model, x_valid, device, args.batch_size)
        score = daily_rank_ic(valid_pred, y_valid, dates)
        history.append(
            {"epoch": epoch, "train_mse": loss_sum / seen, "validation_rank_ic": score}
        )
        print(
            f"    epoch={epoch} train_mse={loss_sum / seen:.8f} "
            f"validation_rank_ic={score:.6f}", flush=True,
        )
        if math.isfinite(score) and score > best_score + 1e-7:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    return model, history


def save_model(model: MLP, path: Path, active_factors: list[str]) -> None:
    temp = path.with_suffix(".tmp.pt")
    torch.save(
        {"state_dict": model.state_dict(), "input_width": len(active_factors),
         "active_factors": active_factors, "architecture": list(model.hidden_sizes),
         "dropout": list(model.dropout)},
        temp,
    )
    temp.replace(path)


def main() -> None:
    args = parse_args()
    validate_common_args(args)
    validate_model_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = prepare_output(Path(args.output_dir), args.force, args.min_free_gb)
    glob = dataset_glob(dataset_dir)
    folds = load_folds(dataset_dir, args.folds)
    progress_path = output_dir / "training_progress.json"
    progress = load_progress(progress_path)
    completed = set(progress["completed"])
    con = open_connection(output_dir, args)
    started = time.time()
    print(f"MLP设备：{device}", flush=True)
    try:
        for number, fold in enumerate(folds, 1):
            name = fold["name"]
            if name in completed:
                print(f"[{number}/{len(folds)}] {name} 已完成，跳过", flush=True)
                continue
            print(f"[{number}/{len(folds)}] MLP加载 {name}……", flush=True)
            x_train, y_train, _, _ = load_matrix(
                con, glob, select_sql(fold, "train"), args.max_train_rows, False
            )
            x_valid, y_valid, codes, dates = load_matrix(
                con, glob, select_sql(fold, "validation"),
                args.max_validation_rows, True,
            )
            assert codes is not None and dates is not None
            stats = fit_preprocessor(x_train, args)
            x_train, x_valid = transform(x_train, stats), transform(x_valid, stats)
            original_rows = len(y_train)
            if args.mlp_max_train_rows and original_rows > args.mlp_max_train_rows:
                rng = np.random.default_rng(args.seed)
                index = rng.choice(
                    original_rows, args.mlp_max_train_rows, replace=False
                )
                x_train, y_train = (
                    np.ascontiguousarray(x_train[index]),
                    np.ascontiguousarray(y_train[index]),
                )
            print(
                f"    MLP训练样本={len(y_train):,}/{original_rows:,}；Validation全量={len(y_valid):,}",
                flush=True,
            )
            fold_dir = output_dir / name
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(stats, fold_dir / "preprocessing.json")
            model, history = train_mlp(
                x_train, y_train, x_valid, y_valid, dates, args, device
            )
            pred = predict(model, x_valid, device, args.batch_size)
            save_model(model, fold_dir / "mlp_model.pt", stats["active_factors"])
            metrics = evaluate(pred, y_valid, dates)
            metrics["best_epoch"] = int(
                max(history, key=lambda item: item["validation_rank_ic"])["epoch"]
            )
            save_predictions(
                fold_dir / "mlp_predictions.parquet", codes, dates, y_valid, pred
            )
            atomic_json(history, fold_dir / "mlp_history.json")
            atomic_json(metrics, fold_dir / "mlp_metrics.json")
            progress["completed"].append(name)
            progress["results"].append({"fold": name, "metrics": metrics})
            completed.add(name)
            atomic_json(progress, progress_path)
            print(f"    最佳Rank IC={metrics['daily_rank_ic_mean']:.6f}", flush=True)
            del model, x_train, y_train, x_valid, y_valid, codes, dates
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        con.close()
    summary = {
        "generated_at": datetime.now().isoformat(), "model": "mlp",
        "dataset_dir": str(dataset_dir), "target": "target_logret_5d",
        "test_used": False, "folds_requested": [x["name"] for x in folds],
        "architecture": [64, 32, 16], "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "weight_decay": args.weight_decay,
        "maximum_training_rows_per_fold": args.mlp_max_train_rows,
        "maximum_epochs": args.epochs, "patience": args.patience,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "aggregate_validation_metrics": aggregate_results(progress["results"]),
    }
    atomic_json(summary, output_dir / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
