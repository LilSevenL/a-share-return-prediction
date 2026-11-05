"""兼容入口：顺序运行拆分后的OLS、LightGBM和MLP训练脚本。

如需调整某个模型的专属参数，请直接运行对应脚本：
    python train_ols.py
    python train_lightgbm.py
    python train_mlp.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS = {
    "ols": PROJECT_DIR / "train_ols.py",
    "lightgbm": PROJECT_DIR / "train_lightgbm.py",
    "mlp": PROJECT_DIR / "train_mlp.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(SCRIPTS),
        default=list(SCRIPTS), help="需要运行的模型；默认三个都运行",
    )
    parser.add_argument("--dataset-dir", help="传递给三个训练脚本的数据目录")
    parser.add_argument("--folds", nargs="*", help="仅运行指定滚动折")
    parser.add_argument("--force", action="store_true", help="各模型备份旧结果后重跑")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model in args.models:
        command = [sys.executable, str(SCRIPTS[model])]
        if args.dataset_dir:
            command.extend(["--dataset-dir", args.dataset_dir])
        if args.folds:
            command.append("--folds")
            command.extend(args.folds)
        if args.force:
            command.append("--force")
        print(f"开始运行 {model}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    print("所选模型训练完成。", flush=True)


if __name__ == "__main__":
    main()
