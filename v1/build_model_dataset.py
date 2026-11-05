"""合并50因子与5日Target，并建立扩展窗口滚动验证方案。

默认方案：
  - 样本起点：2011-01-01
  - 滚动验证：2017年至2024年，每年作为一个Validation折
  - 永久Test：2025-01-01起，模型选择过程中不可使用
  - 最终三组：2011-2023 Training、2024 Validation、2025+ Test

每个边界使用Target的exit_date做purge：只有exit_date早于下一组
开始日期的样本，才允许进入前一组，从而避免未来5日标签跨界。

脚本不进行缩尾、标准化或缺失值拟合。这些步骤必须在每个滚动折的
Training内单独拟合，后续由训练程序完成。
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from build_features import FACTOR_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "quant_data" / "processed"
DEFAULT_FEATURES_DIR = PROCESSED_DIR / "features_50"
DEFAULT_TARGETS_DIR = PROCESSED_DIR / "targets_5d"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "model_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--targets-dir", default=str(DEFAULT_TARGETS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", default="20110101", help="样本起点 YYYYMMDD")
    parser.add_argument(
        "--first-validation-year", type=int, default=2017,
        help="第一个扩展窗口验证年份",
    )
    parser.add_argument(
        "--last-validation-year", type=int, default=2024,
        help="最后一个扩展窗口验证年份；同时作为最终Validation",
    )
    parser.add_argument(
        "--test-start-date", default="20250101",
        help="永久封存Test开始日期 YYYYMMDD",
    )
    parser.add_argument(
        "--min-valid-features", type=int, default=30,
        help="每行至少需要的非空因子数量，范围0至50",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB 内存上限，例如 8GB；超出部分写入临时目录",
    )
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--force", action="store_true", help="备份已有结果并重新生成")
    return parser.parse_args()


def parse_day(value: str, name: str) -> date:
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"{name} 必须是 YYYYMMDD 格式")
    return datetime.strptime(value, "%Y%m%d").date()


def quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parquet_glob(directory: Path, partition_name: str) -> str:
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir() or not next(resolved.rglob("*.parquet"), None):
        raise FileNotFoundError(f"目录中没有Parquet数据：{resolved}")
    return str(resolved / f"{partition_name}=*" / "*.parquet").replace("\\", "/")


def ensure_space(path: Path, minimum_gb: float) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path.parent).free / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"磁盘仅剩 {free_gb:.2f}GB，低于安全阈值 {minimum_gb:.2f}GB"
        )
    return free_gb


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temp = path.with_suffix(".tmp.json")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def prepare_destination(output_dir: Path, force: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"输出目录已存在：{output_dir}。如需重新构建，请添加 --force。"
            )
        backup = output_dir.with_name(
            f"{output_dir.name}.backup.{datetime.now():%Y%m%d%H%M%S}"
        )
        output_dir.replace(backup)
        print(f"已有模型数据已备份到：{backup}", flush=True)
    staging = output_dir.with_name(f"{output_dir.name}._staging")
    if staging.exists():
        stale = staging.with_name(
            f"{staging.name}.stale.{datetime.now():%Y%m%d%H%M%S}"
        )
        staging.replace(stale)
        print(f"未完成临时目录已保留为：{stale}", flush=True)
    staging.mkdir(parents=True)
    return staging


def rolling_folds(
    start_date: date, first_year: int, last_year: int
) -> list[dict[str, str]]:
    folds: list[dict[str, str]] = []
    for validation_year in range(first_year, last_year + 1):
        validation_start = date(validation_year, 1, 1)
        validation_end = date(validation_year + 1, 1, 1)
        folds.append(
            {
                "name": f"validation_{validation_year}",
                "train_start": start_date.isoformat(),
                "train_signal_before": validation_start.isoformat(),
                "train_exit_before": validation_start.isoformat(),
                "validation_start": validation_start.isoformat(),
                "validation_signal_before": validation_end.isoformat(),
                "validation_exit_before": validation_end.isoformat(),
            }
        )
    return folds


def build_query(
    features_glob: str,
    targets_glob: str,
    start_date: date,
    final_validation_year: int,
    test_start: date,
    min_valid_features: int,
) -> str:
    factor_select = ",\n            ".join(
        f"f.{name}" for name in FACTOR_COLUMNS
    )
    final_validation_start = date(final_validation_year, 1, 1)
    return f"""
    WITH joined AS (
        SELECT
            f.ts_code,
            CAST(f.signal_date AS DATE) AS signal_date,
            f.financial_ann_date,
            f.financial_end_date,
            {factor_select},
            f.feature_valid_count,
            CAST(t.entry_date AS DATE) AS entry_date,
            CAST(t.exit_date AS DATE) AS exit_date,
            t.target_logret_5d,
            t.target_ret_5d,
            t.entry_one_price_limit_up,
            t.exit_one_price_limit_down,
            t.target_status
        FROM read_parquet({quote_sql(features_glob)}, hive_partitioning=true) f
        JOIN read_parquet({quote_sql(targets_glob)}, hive_partitioning=true) t
          ON f.ts_code = t.ts_code
         AND CAST(f.signal_date AS DATE) = CAST(t.signal_date AS DATE)
        WHERE CAST(f.signal_date AS DATE) >= DATE {quote_sql(start_date.isoformat())}
          AND t.target_valid
          AND t.target_logret_5d IS NOT NULL
          AND f.feature_valid_count >= {int(min_valid_features)}
    ),
    labelled AS (
        SELECT *,
            CASE
                WHEN signal_date >= DATE {quote_sql(test_start.isoformat())}
                    THEN 'test'
                WHEN signal_date >= DATE {quote_sql(final_validation_start.isoformat())}
                 AND signal_date < DATE {quote_sql(test_start.isoformat())}
                 AND exit_date < DATE {quote_sql(test_start.isoformat())}
                    THEN 'validation'
                WHEN signal_date < DATE {quote_sql(final_validation_start.isoformat())}
                 AND exit_date < DATE {quote_sql(final_validation_start.isoformat())}
                    THEN 'training'
                ELSE 'purged'
            END AS final_split
        FROM joined
    )
    SELECT *, CAST(EXTRACT(YEAR FROM signal_date) AS SMALLINT) AS signal_year
    FROM labelled
    WHERE final_split <> 'purged'
    """


def fold_counts_sql(folds: list[dict[str, str]]) -> tuple[str, list[str]]:
    expressions: list[str] = []
    aliases: list[str] = []
    for index, fold in enumerate(folds):
        train_alias = f"fold_{index}_train"
        validation_alias = f"fold_{index}_validation"
        expressions.extend(
            [
                f"COUNT(*) FILTER (WHERE signal_date >= DATE {quote_sql(fold['train_start'])} "
                f"AND signal_date < DATE {quote_sql(fold['train_signal_before'])} "
                f"AND exit_date < DATE {quote_sql(fold['train_exit_before'])}) AS {train_alias}",
                f"COUNT(*) FILTER (WHERE signal_date >= DATE {quote_sql(fold['validation_start'])} "
                f"AND signal_date < DATE {quote_sql(fold['validation_signal_before'])} "
                f"AND exit_date < DATE {quote_sql(fold['validation_exit_before'])}) AS {validation_alias}",
            ]
        )
        aliases.extend([train_alias, validation_alias])
    return ",\n".join(expressions), aliases


def validate_dataset(
    connection: duckdb.DuckDBPyConnection,
    dataset_glob: str,
    folds: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = connection.execute(
        """
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT ts_code) AS stocks,
               MIN(signal_date) AS first_date,
               MAX(signal_date) AS last_date,
               COUNT(*) - COUNT(DISTINCT (ts_code, signal_date)) AS duplicate_keys,
               COUNT(*) FILTER (WHERE target_logret_5d IS NULL) AS missing_targets,
               COUNT(*) FILTER (WHERE final_split = 'training') AS training_rows,
               COUNT(*) FILTER (WHERE final_split = 'validation') AS validation_rows,
               COUNT(*) FILTER (WHERE final_split = 'test') AS test_rows
        FROM read_parquet(?, hive_partitioning=true)
        """,
        [dataset_glob],
    ).fetchone()
    result = {
        "rows": int(summary[0]),
        "stocks": int(summary[1]),
        "first_date": summary[2],
        "last_date": summary[3],
        "duplicate_keys": int(summary[4]),
        "missing_targets": int(summary[5]),
        "final_split_counts": {
            "training": int(summary[6]),
            "validation": int(summary[7]),
            "test": int(summary[8]),
        },
    }
    if result["rows"] == 0:
        raise RuntimeError("合并后没有可用样本")
    if result["duplicate_keys"]:
        raise RuntimeError(f"模型数据存在重复股票日期键：{result['duplicate_keys']}")
    if result["missing_targets"]:
        raise RuntimeError(f"模型数据存在空Target：{result['missing_targets']}")

    counts_expression, aliases = fold_counts_sql(folds)
    count_values = connection.execute(
        f"SELECT {counts_expression} FROM read_parquet(?, hive_partitioning=true)",
        [dataset_glob],
    ).fetchone()
    count_map = dict(zip(aliases, count_values))
    populated_folds: list[dict[str, Any]] = []
    for index, fold in enumerate(folds):
        populated_folds.append(
            {
                **fold,
                "training_rows": int(count_map[f"fold_{index}_train"]),
                "validation_rows": int(count_map[f"fold_{index}_validation"]),
            }
        )
    return result, populated_folds


def main() -> None:
    args = parse_args()
    start_date = parse_day(args.start_date, "start-date")
    test_start = parse_day(args.test_start_date, "test-start-date")
    if not 0 <= args.min_valid_features <= len(FACTOR_COLUMNS):
        raise ValueError("min-valid-features 必须在0至50之间")
    if args.threads <= 0:
        raise ValueError("threads 必须大于0")
    if args.first_validation_year > args.last_validation_year:
        raise ValueError("first-validation-year不能晚于last-validation-year")
    if start_date >= date(args.first_validation_year, 1, 1):
        raise ValueError("样本起点必须早于第一个Validation年份")
    if date(args.last_validation_year + 1, 1, 1) > test_start:
        raise ValueError("最后Validation年份不能与永久Test重叠")

    features_glob = parquet_glob(Path(args.features_dir), "feature_year")
    targets_glob = parquet_glob(Path(args.targets_dir), "signal_year")
    output_dir = Path(args.output_dir).expanduser().resolve()
    free_gb_before = ensure_space(output_dir, args.min_free_gb)
    staging = prepare_destination(output_dir, args.force)
    database_path = staging.parent / f".{output_dir.name}.duckdb"
    folds = rolling_folds(
        start_date, args.first_validation_year, args.last_validation_year
    )

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(f"SET threads = {int(args.threads)}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET temp_directory = {quote_sql(str(staging / '_tmp'))}")
        connection.execute(f"SET memory_limit = {quote_sql(args.memory_limit)}")
        query = build_query(
            features_glob,
            targets_glob,
            start_date,
            args.last_validation_year,
            test_start,
            args.min_valid_features,
        )
        print("正在合并因子与Target并建立滚动验证数据集……", flush=True)
        connection.execute(
            f"""
            COPY ({query}) TO {quote_sql(str(staging).replace(chr(92), '/'))}
            (FORMAT PARQUET, PARTITION_BY (final_split, signal_year),
             COMPRESSION ZSTD, ROW_GROUP_SIZE 100000, OVERWRITE_OR_IGNORE TRUE)
            """
        )
        dataset_glob = str(
            staging / "final_split=*" / "signal_year=*" / "*.parquet"
        ).replace("\\", "/")
        validation, populated_folds = validate_dataset(
            connection, dataset_glob, folds
        )
    except Exception:
        print(f"构建失败；临时结果保留在：{staging}", flush=True)
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    staging.replace(output_dir)
    report = {
        "generated_at": datetime.now().isoformat(),
        "factor_count": len(FACTOR_COLUMNS),
        "target": "target_logret_5d",
        "sample_start": start_date,
        "test_start": test_start,
        "minimum_valid_features": args.min_valid_features,
        "purge_rule": "previous split requires exit_date < next split start",
        "final_validation_year": args.last_validation_year,
        **validation,
        "rolling_folds": populated_folds,
        "free_gb_before": round(free_gb_before, 2),
        "output_size_bytes": sum(
            path.stat().st_size for path in output_dir.rglob("*.parquet")
        ),
    }
    atomic_json(report, output_dir / "dataset_report.json")
    atomic_json(
        {
            "description": "expanding-window validation; Test is permanently held out",
            "target": "target_logret_5d",
            "folds": populated_folds,
        },
        output_dir / "rolling_folds.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"模型数据已保存到：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
