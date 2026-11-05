"""下载用于量化机器学习的代表性 A 股数据集。

依赖安装：
    python -m pip install --upgrade tushare pandas

运行示例：
    python main.py
    python main.py --start-date 20150101 --end-date 20261231
    python main.py --output-dir "F:\\quant_data"

脚本会按股票、按接口缓存数据，意外中断后再次运行会跳过已有文件。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "quant_data"
DATA_DIR = DEFAULT_DATA_DIR
CACHE_DIR = DATA_DIR / "cache"
COMBINED_DIR = DATA_DIR / "combined"
REPORT_PATH = DATA_DIR / "download_report.json"

# 按用户提供的 Tushare SDK 代理配置。Key 只从环境变量读取，
# 不打印、不写入源码，也不写入下载报告。
API_KEY_ENV_VAR = "TUSHARE_API_KEY"
API_BASE_URL = "https://teajoin.com"
REQUEST_INTERVAL_SECONDS = 0.2

# 覆盖金融、消费、医药、科技、制造、能源、周期、公用事业、交通运输等行业，
# 同时包含主板、创业板和科创板，以及不同上市年份的公司。
STOCK_POOL = [
    ("000001.SZ", "平安银行", "银行"),
    ("000063.SZ", "中兴通讯", "通信设备"),
    ("000100.SZ", "TCL科技", "电子制造"),
    ("000333.SZ", "美的集团", "家用电器"),
    ("000538.SZ", "云南白药", "医药"),
    ("000651.SZ", "格力电器", "家用电器"),
    ("000858.SZ", "五粮液", "食品饮料"),
    ("000895.SZ", "双汇发展", "食品加工"),
    ("001979.SZ", "招商蛇口", "房地产"),
    ("002027.SZ", "分众传媒", "传媒"),
    ("002230.SZ", "科大讯飞", "人工智能"),
    ("002304.SZ", "洋河股份", "食品饮料"),
    ("002352.SZ", "顺丰控股", "物流"),
    ("002371.SZ", "北方华创", "半导体设备"),
    ("002415.SZ", "海康威视", "计算机设备"),
    ("002460.SZ", "赣锋锂业", "新能源材料"),
    ("002475.SZ", "立讯精密", "消费电子"),
    ("002594.SZ", "比亚迪", "汽车"),
    ("002714.SZ", "牧原股份", "农业"),
    ("300014.SZ", "亿纬锂能", "新能源电池"),
    ("300015.SZ", "爱尔眼科", "医疗服务"),
    ("300059.SZ", "东方财富", "金融科技"),
    ("300122.SZ", "智飞生物", "生物医药"),
    ("300124.SZ", "汇川技术", "工业自动化"),
    ("300274.SZ", "阳光电源", "光伏设备"),
    ("300308.SZ", "中际旭创", "光通信"),
    ("300498.SZ", "温氏股份", "农业"),
    ("300750.SZ", "宁德时代", "新能源电池"),
    ("300760.SZ", "迈瑞医疗", "医疗器械"),
    ("600009.SH", "上海机场", "交通运输"),
    ("600028.SH", "中国石化", "石油化工"),
    ("600030.SH", "中信证券", "证券"),
    ("600031.SH", "三一重工", "工程机械"),
    ("600036.SH", "招商银行", "银行"),
    ("600050.SH", "中国联通", "通信运营"),
    ("600089.SH", "特变电工", "电力设备"),
    ("600104.SH", "上汽集团", "汽车"),
    ("600276.SH", "恒瑞医药", "医药"),
    ("600309.SH", "万华化学", "化工"),
    ("600438.SH", "通威股份", "光伏"),
    ("600519.SH", "贵州茅台", "食品饮料"),
    ("600585.SH", "海螺水泥", "建材"),
    ("600690.SH", "海尔智家", "家用电器"),
    ("600887.SH", "伊利股份", "食品饮料"),
    ("600900.SH", "长江电力", "公用事业"),
    ("601012.SH", "隆基绿能", "光伏"),
    ("601088.SH", "中国神华", "煤炭"),
    ("601166.SH", "兴业银行", "银行"),
    ("601318.SH", "中国平安", "保险"),
    ("601398.SH", "工商银行", "银行"),
    ("601668.SH", "中国建筑", "建筑"),
    ("601766.SH", "中国中车", "轨道交通"),
    ("601857.SH", "中国石油", "石油天然气"),
    ("601888.SH", "中国中免", "旅游零售"),
    ("601899.SH", "紫金矿业", "有色金属"),
    ("601919.SH", "中远海控", "航运"),
    ("603259.SH", "药明康德", "医药研发"),
    ("603288.SH", "海天味业", "食品饮料"),
    ("603501.SH", "韦尔股份", "半导体"),
    ("603986.SH", "兆易创新", "半导体"),
    ("688008.SH", "澜起科技", "半导体"),
    ("688036.SH", "传音控股", "消费电子"),
    ("688111.SH", "金山办公", "软件"),
    ("688981.SH", "中芯国际", "半导体制造"),
]

INDEX_POOL = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
}


@dataclass(frozen=True)
class DatasetSpec:
    api_name: str
    fields: str
    unique_keys: tuple[str, ...]
    description: str
    required: bool = False


DATASETS = [
    DatasetSpec(
        "daily",
        "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        ("ts_code", "trade_date"),
        "未复权 OHLCV 基础日线",
        required=True,
    ),
    DatasetSpec(
        "adj_factor",
        "ts_code,trade_date,adj_factor",
        ("ts_code", "trade_date"),
        "复权因子，用于构造前复权或后复权价格",
    ),
    DatasetSpec(
        "daily_basic",
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
        "free_share,total_mv,circ_mv",
        ("ts_code", "trade_date"),
        "换手率、估值、股本和市值等每日指标",
    ),
    DatasetSpec(
        "moneyflow",
        "ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,"
        "buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,"
        "buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,"
        "sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount",
        ("ts_code", "trade_date"),
        "小单到特大单的资金流向",
    ),
    DatasetSpec(
        "stk_limit",
        "ts_code,trade_date,pre_close,up_limit,down_limit",
        ("ts_code", "trade_date"),
        "涨跌停价格，用于判断策略能否实际成交",
    ),
    DatasetSpec(
        "fina_indicator",
        "ts_code,ann_date,end_date,eps,dt_eps,total_revenue_ps,revenue_ps,"
        "netprofit_margin,grossprofit_margin,current_ratio,quick_ratio,"
        "debt_to_assets,roe,roe_waa,roa,assets_turn,inv_turn,ar_turn,"
        "ocf_to_or,netprofit_yoy,or_yoy",
        ("ts_code", "ann_date", "end_date"),
        "季度盈利能力、成长、偿债和运营效率指标",
    ),
]

INDEX_SPEC = DatasetSpec(
    "index_daily",
    "ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount",
    ("ts_code", "trade_date"),
    "市场基准指数日线",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20100101", help="开始日期 YYYYMMDD")
    parser.add_argument(
        "--end-date", default=date.today().strftime("%Y%m%d"), help="结束日期 YYYYMMDD"
    )
    parser.add_argument(
        "--min-daily-rows", type=int, default=100_000, help="日线数据最低行数要求"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR),
        help="数据输出目录，默认是项目下的 quant_data",
    )
    parser.add_argument("--force", action="store_true", help="忽略缓存，重新下载")
    parser.add_argument(
        "--daily-only", action="store_true", help="只下载基础日线，适合积分不足时使用"
    )
    return parser.parse_args()


def validate_date(value: str, name: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"{name} 必须是 YYYYMMDD 格式，当前值：{value}")


def save_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp.csv")
    df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)


def cache_path(api_name: str, ts_code: str) -> Path:
    safe_code = ts_code.replace(".", "_")
    return CACHE_DIR / api_name / f"{safe_code}.csv"


def is_permission_error(message: str) -> bool:
    markers = ("权限", "积分", "2002", "抱歉，您没有访问该接口的权限")
    return any(marker in message for marker in markers)


def create_pro_client() -> Any:
    """按代理服务要求创建 Tushare Pro 客户端。"""
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise RuntimeError(
            f"未检测到环境变量 {API_KEY_ENV_VAR}。请先在当前终端设置 Key，"
            "再运行下载脚本。"
        )

    ts.set_token(api_key)
    # 显式传入 Key，避免某些 SDK 版本读取本地 token 文件时截断内容。
    pro = ts.pro_api(api_key)
    pro._DataApi_token = api_key
    # 当前 SDK 实际使用双下划线私有属性；同时保留上面的代理模板属性。
    pro._DataApi__token = api_key
    pro._DataApi__http_url = API_BASE_URL
    return pro


def call_api(
    pro: Any,
    spec: DatasetSpec,
    ts_code: str,
    start_date: str,
    end_date: str,
    retries: int = 3,
) -> pd.DataFrame:
    params = {
        "ts_code": ts_code,
        "start_date": start_date,
        "end_date": end_date,
        "fields": spec.fields,
    }
    for attempt in range(1, retries + 1):
        try:
            result = getattr(pro, spec.api_name)(**params)
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"接口返回类型异常：{type(result)!r}")
            return result
        except Exception as exc:
            if is_permission_error(str(exc)) or attempt == retries:
                raise
            wait_seconds = attempt * 2
            print(f"    请求失败，{wait_seconds} 秒后重试：{exc}")
            time.sleep(wait_seconds)
    return pd.DataFrame()


def download_per_stock_dataset(
    pro: Any,
    spec: DatasetSpec,
    start_date: str,
    end_date: str,
    force: bool,
) -> tuple[pd.DataFrame, list[str]]:
    failures: list[str] = []
    frames: list[pd.DataFrame] = []
    permission_denied = False

    print(f"\n[{spec.api_name}] {spec.description}")
    for position, (ts_code, name, _) in enumerate(STOCK_POOL, start=1):
        path = cache_path(spec.api_name, ts_code)
        if path.exists() and not force:
            try:
                frame = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
                frames.append(frame)
                print(f"  {position:02d}/{len(STOCK_POOL)} {ts_code} {name}: 缓存 {len(frame)} 行")
                continue
            except (pd.errors.EmptyDataError, OSError):
                path.unlink(missing_ok=True)

        if permission_denied:
            continue

        try:
            frame = call_api(pro, spec, ts_code, start_date, end_date)
            if frame.empty:
                failures.append(f"{ts_code}: 返回空数据")
                print(f"  {position:02d}/{len(STOCK_POOL)} {ts_code} {name}: 空数据")
            else:
                sort_columns = [column for column in ("trade_date", "ann_date", "end_date") if column in frame]
                if sort_columns:
                    frame = frame.sort_values(sort_columns)
                save_csv_atomic(frame, path)
                frames.append(frame)
                print(f"  {position:02d}/{len(STOCK_POOL)} {ts_code} {name}: 下载 {len(frame)} 行")
            time.sleep(REQUEST_INTERVAL_SECONDS)
        except Exception as exc:
            message = str(exc)
            failures.append(f"{ts_code}: {message}")
            print(f"  {position:02d}/{len(STOCK_POOL)} {ts_code} {name}: 失败 - {message}")
            if is_permission_error(message):
                permission_denied = True
                print(f"  {spec.api_name} 无权限，跳过该接口剩余股票。")

    if not frames:
        return pd.DataFrame(), failures

    combined = pd.concat(frames, ignore_index=True)
    available_keys = [key for key in spec.unique_keys if key in combined.columns]
    if available_keys:
        combined = combined.drop_duplicates(subset=available_keys, keep="last")
        combined = combined.sort_values(available_keys).reset_index(drop=True)
    output_path = COMBINED_DIR / f"{spec.api_name}.csv"
    save_csv_atomic(combined, output_path)
    print(f"  合并结果：{len(combined)} 行 -> {output_path}")
    return combined, failures


def download_indices(
    pro: Any, start_date: str, end_date: str, force: bool
) -> tuple[pd.DataFrame, list[str]]:
    failures: list[str] = []
    frames: list[pd.DataFrame] = []
    print(f"\n[index_daily] {INDEX_SPEC.description}")
    for ts_code, name in INDEX_POOL.items():
        path = cache_path("index_daily", ts_code)
        try:
            if path.exists() and not force:
                frame = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
                print(f"  {ts_code} {name}: 缓存 {len(frame)} 行")
            else:
                frame = call_api(pro, INDEX_SPEC, ts_code, start_date, end_date)
                if not frame.empty:
                    save_csv_atomic(frame.sort_values("trade_date"), path)
                print(f"  {ts_code} {name}: 下载 {len(frame)} 行")
                time.sleep(REQUEST_INTERVAL_SECONDS)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            failures.append(f"{ts_code}: {exc}")
            print(f"  {ts_code} {name}: 失败 - {exc}")
            if is_permission_error(str(exc)):
                break

    if not frames:
        return pd.DataFrame(), failures
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(list(INDEX_SPEC.unique_keys), keep="last")
    combined = combined.sort_values(list(INDEX_SPEC.unique_keys)).reset_index(drop=True)
    save_csv_atomic(combined, COMBINED_DIR / "index_daily.csv")
    return combined, failures


def save_stock_pool() -> None:
    frame = pd.DataFrame(STOCK_POOL, columns=["ts_code", "name", "sector"])
    save_csv_atomic(frame, DATA_DIR / "stock_pool.csv")


def main() -> None:
    global DATA_DIR, CACHE_DIR, COMBINED_DIR, REPORT_PATH

    args = parse_args()
    validate_date(args.start_date, "start-date")
    validate_date(args.end_date, "end-date")
    if args.start_date > args.end_date:
        raise ValueError("start-date 不能晚于 end-date")

    DATA_DIR = Path(args.output_dir).expanduser().resolve()
    CACHE_DIR = DATA_DIR / "cache"
    COMBINED_DIR = DATA_DIR / "combined"
    REPORT_PATH = DATA_DIR / "download_report.json"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_stock_pool()
    pro = create_pro_client()

    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "stock_count": len(STOCK_POOL),
        "api_base_url": API_BASE_URL,
        "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
        "datasets": {},
        "notes": [
            "daily 是未复权行情，建模价格应与 adj_factor 配合。",
            "财务数据必须按 ann_date 对齐交易日，不能按 end_date 直接回填，以避免未来数据泄漏。",
            "成交量和成交额的单位以 Tushare 各接口文档为准。",
        ],
    }

    selected_specs = DATASETS[:1] if args.daily_only else DATASETS
    for spec in selected_specs:
        frame, failures = download_per_stock_dataset(
            pro, spec, args.start_date, args.end_date, args.force
        )
        report["datasets"][spec.api_name] = {
            "description": spec.description,
            "rows": len(frame),
            "columns": list(frame.columns),
            "failure_count": len(failures),
            "failures": failures,
        }

    if not args.daily_only:
        index_frame, index_failures = download_indices(
            pro, args.start_date, args.end_date, args.force
        )
        report["datasets"]["index_daily"] = {
            "description": INDEX_SPEC.description,
            "rows": len(index_frame),
            "columns": list(index_frame.columns),
            "failure_count": len(index_failures),
            "failures": index_failures,
        }

    daily_rows = report["datasets"].get("daily", {}).get("rows", 0)
    total_rows = sum(item["rows"] for item in report["datasets"].values())
    report["daily_target"] = args.min_daily_rows
    report["daily_target_met"] = daily_rows >= args.min_daily_rows
    report["total_rows"] = total_rows
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 下载汇总 ===")
    for api_name, result in report["datasets"].items():
        print(f"{api_name:16s}: {result['rows']:>8,} 行，失败 {result['failure_count']} 个")
    print(f"总数据行数：{total_rows:,}")
    print(f"基础日线行数：{daily_rows:,} / 目标 {args.min_daily_rows:,}")
    print(f"报告：{REPORT_PATH}")

    if daily_rows < args.min_daily_rows:
        raise RuntimeError(
            f"基础日线只有 {daily_rows:,} 行，未达到 {args.min_daily_rows:,} 行目标。"
            "请查看 download_report.json 中的 failures。"
        )


if __name__ == "__main__":
    main()
