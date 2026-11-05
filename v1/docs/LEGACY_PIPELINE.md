# Tushare A 股量化数据项目

本项目使用 **Tushare SDK** 下载、审计和清洗 A 股历史数据，为后续量化研究与机器学习建模准备数据集。

当前项目包含两套下载范围：

- `main.py`：最初的 **64 只代表性股票**，用于小规模测试和示例研究。
- `download_all_stocks.py`：全部当前上市 A 股，本次实际下载 **5,550 只股票**，已经包含上述 64 只股票。

全市场非分钟数据、质量审计、异常修复、14个接口的 Parquet 转换、50因子、未来5日标签、滚动验证数据集、三模型优化、最终重训、永久Test以及结果解释均已完成。没有必要重复下载、转换或重新调参。

## 当前项目状态（2026-09-26）

| 阶段 | 状态 | 关键结果 |
| --- | --- | --- |
| 全市场数据 | 已完成 | 5,550只当前上市股票，14类非分钟接口 |
| 模型数据集 | 已完成 | 12,379,334行、5,370只股票、50个候选因子、未来5日标签 |
| 滚动优化 | 已完成 | 2017–2023七折用于方案设计，2024只用于冻结前确认 |
| 最终重训 | 已完成 | 使用2011–2024全部非Test数据重训三个冻结模型 |
| 永久Test | 已完成并封存 | 预测标签覆盖2025-01-02至2026-08-20；结果不得用于重新调参或换模型 |
| 研究输出 | 已完成 | 风险归因、六张图表、结果解释和中文论文初稿 |

永久Test已经执行完毕。后续若产生新模型、因子或交易规则，必须建立新的前瞻样本外区间，不能回到本次Test结果上继续筛选。

## 1. 项目流程

```text
Tushare 接口
    ↓
原始分片 CSV/CSV.GZ（保留不修改）
    ↓
质量审计与问题文件修复
    ↓
清洗并按接口、年份转换为 Parquet
    ↓
按股票代码和可用日期对齐各类数据
    ↓
特征工程、标签构造、时间切分、模型训练与回测
```

目前从原始数据下载到模型评估、组合回测和研究写作的完整链路均已完成。

## 2. 核心脚本分工

| 文件 | 作用 | 当前是否需要运行 |
| --- | --- | --- |
| `main.py` | 下载最初的 64 只代表性股票及市场指数 | 通常不需要；仅用于小样本重新测试 |
| `download_all_stocks.py` | 下载全部当前上市 A 股的 14 类非分钟数据 | 已完成，不需要重复运行 |
| `download_extra.py` | 为 64 只样本股票下载周/月线、财务报表及特色数据 | 已被全市场下载流程覆盖 |
| `download_minutes.py` | 下载历史分钟线 | 当前暂不处理；分钟数据需要独立权限 |
| `preprocess_data.py` | 审计原始文件、生成质量报告、补采已知异常文件 | 全量审计和修复已经完成 |
| `build_parquet.py` | 过滤、去重、统一类型并转换为 Parquet | 全部14个接口已经完成 |
| `build_targets.py` | 生成下一交易日开盘至第5个未来交易日收盘的复权收益标签 | 已完成；除非重建上游数据，否则不需重跑 |
| `build_features.py` | 构建50个趋势、风险、流动性、估值、资金流和财务因子 | 已完成；除非修改因子定义，否则不需重跑 |
| `build_model_dataset.py` | 合并因子与标签并建立滚动Training/Validation/Test时间切分 | 已完成 |
| `train_ols.py` / `train_lightgbm.py` / `train_mlp.py` | 三个初始模型的滚动训练入口 | 已完成；保留用于复现，不应用于Test后调参 |
| `retrain_final_models.py` | 用2011–2024全部非Test数据重训冻结模型 | 已完成 |
| `generate_permanent_test_signals.py` / `evaluate_permanent_test.py` | 生成并评估唯一一次永久Test | 已完成并封存，不要重复用于模型选择 |
| `analyze_test_results.py` | 只读正式Test产物，生成风险归因、统计表和图表 | 可安全重跑；不会修改模型或交易规则 |

## 3. Python 环境

当前已验证的解释器为：

```text
E:\Anaconda\envs\quant_ml\python.exe
```

已验证的软件版本：

- Python 3.12.13
- pandas 3.0.5
- tushare 1.4.29
- pyarrow 25.0.1
- duckdb 1.5.5

安装或更新依赖：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" -m pip install --upgrade `
  tushare pandas pyarrow duckdb
```

如果 PowerShell 能识别 `conda`，也可以先激活环境：

```powershell
conda activate quant_ml
python --version
```

如果出现“无法将 conda 识别为命令”，直接使用上面的完整解释器路径即可。在 VS Code 中选择解释器时，同样选择：

```text
E:\Anaconda\envs\quant_ml\python.exe
```

## 4. Tushare Key 与代理配置

所有下载程序只从环境变量 `TUSHARE_API_KEY` 读取 Key，不会把 Key 打印到终端、写入源码或写入下载报告。不要把真实 Key 放进 README 或提交到版本库。

仅为当前 PowerShell 窗口设置：

```powershell
$env:TUSHARE_API_KEY = "在本机填写自己的 Key"
```

永久写入当前 Windows 用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable(
    "TUSHARE_API_KEY",
    "在本机填写自己的 Key",
    "User"
)
```

永久设置后，需要关闭并重新打开 VS Code/Codex/PowerShell，现有进程才会读到新变量。可以只检查变量是否存在，不要输出 Key 本身：

```powershell
if ($env:TUSHARE_API_KEY) { "TUSHARE_API_KEY 已设置" } else { "尚未设置" }
```

下载程序会自动把 Tushare SDK 请求地址指向：

```text
https://teajoin.com
```

批量请求间隔不低于 0.2 秒，包含重试、原子写入、磁盘空间检查和进度保存。

## 5. 全市场数据下载

全市场下载器是 `download_all_stocks.py`，不是 `main.py`。

本次已经执行并完成的命令相当于：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" download_all_stocks.py `
  --start-date 20100101 `
  --end-date 20260827
```

### 下载结果（2026-08-28）

- 当前上市股票：5,550 只。
- 原 64 只代表性股票：已包含在 5,550 只中，不需要另行合并股票数量。
- 逐股票接口：14 个。
- 总任务：77,700 个“股票 × 接口”分片。
- 失败分片：0。
- 估算累计数据量：约 8,194 万行。
- 原始股票分片及空表标记：约 10.76 GB。
- 最终进度文件：`quant_data/all_stocks_progress.json`。

### 14 个逐股票接口

| 接口 | 内容 | 原始目录 |
| --- | --- | --- |
| `daily` | 未复权日线 OHLCV | `quant_data/cache/daily/` |
| `adj_factor` | 复权因子 | `quant_data/cache/adj_factor/` |
| `daily_basic` | 换手率、估值、股本、市值 | `quant_data/cache/daily_basic/` |
| `moneyflow` | 小单至特大单资金流向 | `quant_data/cache/moneyflow/` |
| `stk_limit` | 每日涨跌停价格 | `quant_data/cache/stk_limit/` |
| `fina_indicator` | 盈利、成长、偿债和运营指标 | `quant_data/cache/fina_indicator/` |
| `weekly` | 周线行情 | `quant_data/extra_cache/weekly/` |
| `monthly` | 月线行情 | `quant_data/extra_cache/monthly/` |
| `income` | 利润表 | `quant_data/extra_cache/income/` |
| `balancesheet` | 资产负债表 | `quant_data/extra_cache/balancesheet/` |
| `cashflow` | 现金流量表 | `quant_data/extra_cache/cashflow/` |
| `limit_list_d` | 涨停、跌停和炸板事件 | `quant_data/extra_cache/limit_list_d/` |
| `ths_hot` | 同花顺历史热榜 | `quant_data/extra_cache/ths_hot/` |
| `stk_factor_pro` | 专业量化因子宽表 | `quant_data/extra_cache/stk_factor_pro/` |

`stock_basic`、交易日历、指数等全市场公共数据只保存一份，不按股票重复保存。`cyq_chips` 在代理测试中持续返回空表，因此没有加入这次全市场逐股票任务。

### 测试少量股票

以后若要验证新的日期范围或接口，先用少量股票测试：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" download_all_stocks.py `
  --start-date 20250101 `
  --end-date 20251231 `
  --interfaces daily,adj_factor,daily_basic `
  --max-stocks 10
```

### 断点续传规则

- 每个接口、每只股票单独保存为 `.csv.gz`。
- 无数据的请求保存为 `.empty.json`，避免重复发起空请求。
- 再次运行时会跳过已有 `.csv.gz`、旧 `.csv` 或 `.empty.json`。
- 进度持续写入 `quant_data/all_stocks_progress.json`。
- 请求失败会重试；单个接口无权限时，会记录并跳过该接口后续股票。
- 磁盘剩余空间默认低于 10 GB 时停止，防止磁盘写满。
- `--force` 会覆盖现有分片，除非明确需要重新下载，否则不要使用。

## 6. 64 只代表性股票下载器

`main.py` 只负责原始的 64 只代表性股票，不是当前全市场数据的主入口。

```powershell
# 默认下载64只股票
& "E:\Anaconda\envs\quant_ml\python.exe" main.py

# 只下载基础日线
& "E:\Anaconda\envs\quant_ml\python.exe" main.py --daily-only

# 指定日期
& "E:\Anaconda\envs\quant_ml\python.exe" main.py `
  --start-date 20150101 `
  --end-date 20261231
```

如需修改这 64 只样本，编辑 `main.py` 中的 `STOCK_POOL`。每项格式为：

```python
("000001.SZ", "平安银行", "银行")
```

## 7. 数据质量审计与修复

原始目录 `quant_data/cache/`、`quant_data/extra_cache/` 和 `quant_data/combined/` 始终保留，不会被预处理程序原地修改。

### 快速抽样审计

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" preprocess_data.py audit `
  --sample-per-interface 100
```

### 全量审计

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" preprocess_data.py audit `
  --sample-per-interface 0
```

审计内容包括：

- 每个接口的股票文件覆盖情况。
- `.empty.json` 空表标记。
- 同时存在旧 `.csv` 和新 `.csv.gz` 的重复缓存版本。
- 表头与必需字段一致性。
- 日期能否正确解析。
- 文件名股票代码与行内 `ts_code` 是否一致。
- 业务主键重复。
- OHLC 高低开收关系和非正价格。
- 负成交量或负成交额。

全量审计结果：

- 缺失文件：0。
- 无法读取文件：0。
- 发现两份股票代码混杂文件：`300354.SZ/fina_indicator`、`301138.SZ/cashflow`。
- 财务接口存在同报告期多版本或重复记录，清洗时按业务主键保留信息最完整的一条。
- 少量日线存在停牌式零价格记录，建模清洗时删除。

主要输出：

- `quant_data/processed/quality/quality_report.md`
- `quant_data/processed/quality/raw_manifest.parquet`
- `quant_data/processed/quality/quality_issues.json`
- `quant_data/processed/quality/audited_file_metrics.parquet`

### 修复已知代码错配文件

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" preprocess_data.py repair
```

修复文件单独写到 `quant_data/processed/repairs/`，原始异常文件不会删除或覆盖。当前已完成：

- `fina_indicator/300354_SZ.csv.gz`
- `cashflow/301138_SZ.csv.gz`

修复报告：`quant_data/processed/repairs/repair_report.json`。

## 8. 清洗并转换为 Parquet

`build_parquet.py` 才是主要的数据清洗和格式转换程序。它读取审计清单，发现修复文件时优先使用修复版本，然后按“接口/年份”输出 Parquet。

### 先用少量股票验证

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" build_parquet.py `
  --interfaces daily,fina_indicator,cashflow `
  --max-stocks 20 `
  --output-dir quant_data/processed/parquet_test
```

### 全市场转换

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" build_parquet.py
```

清洗规则：

1. 只保留行内 `ts_code` 与股票文件名一致的数据。
2. 将交易日、公告日和报告期转换为日期类型。
3. 删除主日期缺失的记录。
4. 删除开、高、低、收小于等于零或 OHLC 关系不成立的行情记录。
5. 删除负成交量和负成交额记录。
6. 按接口业务主键去重，优先保留非空字段最多的版本。
7. 将字符串、日期和数值字段统一为稳定的 Parquet 类型。
8. 增加 `data_year`，按年份分区并使用 ZSTD 压缩。
9. 每个接口完成后使用 DuckDB 校验 Parquet 总行数。

输出结构示例：

```text
quant_data/processed/parquet/
└── daily/
    ├── year=2010/part-00000.parquet
    ├── year=2011/part-00000.parquet
    └── ...
```

转换进度：`quant_data/processed/parquet/conversion_progress.json`。

转换日志：

```text
quant_data/processed/parquet_conversion.log
quant_data/processed/parquet_conversion.error.log
```

程序对单个接口使用临时 `_staging` 目录；该接口全部写入并校验成功后才移动到正式目录。重新运行时会验证并跳过已经完成的接口，因此支持安全续传。

### 全量转换完成情况（2026-09-08）

- 14/14 个接口均已生成正式 Parquet 目录。
- 最后五个接口已重新转换并通过 DuckDB 行数校验。
- `balancesheet`：221,272行，5,379只股票。
- `cashflow`：225,849行，5,375只股票。
- `limit_list_d`：154,696行，5,192只股票。
- `ths_hot`：236,462行，3,066只股票。
- `stk_factor_pro`：13,015,068行，5,341只股票。
- 转换错误日志为空，完成时 F 盘剩余约60.62 GB。

宽表写入采用按年份累计20,000行后批量写入的方式，避免为每只股票生成微小 Parquet row group。这个优化显著降低了长任务后半段的内存增长和写入耗时。

## 9. 构建未来5日收益标签

`build_targets.py` 使用清洗后的 `daily`、`adj_factor` 和 `stk_limit` 数据生成监督学习标签。定义为信号日收盘后形成预测，下一市场交易日开盘买入，第5个未来市场交易日收盘卖出：

```text
target_logret_5d = ln(adjusted_close[t+5] / adjusted_open[t+1])
target_ret_5d    = exp(target_logret_5d) - 1
```

先用20只股票验证：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" build_targets.py `
  --start-date 20250101 `
  --end-date 20251231 `
  --max-stocks 20 `
  --output-dir quant_data/processed/targets_5d_test `
  --force
```

全市场构建：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" build_targets.py
```

标签按 `signal_year` 分区写入 `quant_data/processed/targets_5d/`。主要字段包括：

- `signal_date`：模型产生信号的日期。
- `entry_date`：下一市场交易日。
- `exit_date`：第5个未来市场交易日。
- `target_logret_5d`：考虑可交易约束后的主对数收益标签。
- `target_ret_5d`：方便解释的普通收益率。
- `target_logret_5d_raw`、`target_ret_5d_raw`：只要买卖价格存在就计算的原始标签。
- `entry_one_price_limit_up`：买入日是否一字涨停。
- `exit_one_price_limit_down`：卖出日是否一字跌停。
- `target_valid`：标签是否可以进入训练集。
- `target_status`：无效原因或 `valid`。

股票停牌时不能通过“该股票后面的第5条记录”代替第5个市场交易日。程序首先从全市场日线生成统一交易日序列，再将每条信号映射到严格的 `t+1` 和 `t+5`。买入日一字涨停、卖出日一字跌停、价格缺失或复权因子缺失时，主标签留空并通过状态字段说明原因。

20只股票、2025年测试已经通过：4,852条信号记录，4,847条有效标签，5条下一日一字涨停无效标签，对数收益与普通收益恒等式错误为0。测试报告位于 `quant_data/processed/targets_5d_test/target_report.json`。

全市场标签已于2026-09-08构建完成：13,030,829条信号记录、5,390只股票，日期覆盖2010-01-04至2026-08-27。其中12,881,840条标签有效、148,989条因缺少未来买卖价格或一字涨跌停等原因无效；收益恒等式错误为0。共生成17个年度Parquet文件，约526 MB，完整报告位于 `quant_data/processed/targets_5d/target_report.json`。

标签程序默认设置DuckDB内存上限为8 GB，超出的中间计算会写入F盘临时目录。内存较小的电脑可使用 `--memory-limit 6GB` 调低上限。

## 10. 构建50个量化因子

`build_features.py` 从清洗后的 `daily`、`adj_factor`、`daily_basic`、`moneyflow` 和 `fina_indicator` 构建恰好50个候选因子：

| 因子组 | 数量 | 主要内容 |
| --- | ---: | --- |
| 动量与趋势 | 12 | 1/2/3/5/10/20/60日对数收益、均线偏离 |
| 风险与价格形态 | 12 | 多周期波动率、下行波动、偏度、Beta、市场相关性 |
| 流动性 | 7 | 对数成交额、量价比、换手率及滚动标准分数 |
| 估值与规模 | 5 | 对数市值、盈利收益率、账面市值比、销售收益率 |
| 资金流 | 6 | 净流入和不同订单规模的净流入比例 |
| 财务质量与成长 | 8 | ROE、ROA、毛利率、负债率、增长率和经营现金流 |

小范围验证：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" build_features.py `
  --start-date 20250101 `
  --end-date 20251231 `
  --max-stocks 20 `
  --output-dir quant_data/processed/features_50_test `
  --force
```

全市场构建：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" build_features.py
```

程序只使用信号日及以前的行情。财务数据采用 point-in-time 连接，只允许 `ann_date < signal_date`，防止财务公告穿越。滚动收益会检查市场交易日连续性，股票停牌导致日期不连续时不会把更早的记录冒充前一交易日。数值结果中的无穷值会转换为空值。

输出按 `feature_year` 保存到 `quant_data/processed/features_50/`，并包含 `feature_valid_count`，表示该股票日期50个因子中有多少个非空。完整名称和覆盖率保存在 `feature_report.json`。

20只股票、2025年测试已通过：4,852行、50个因子、重复股票日期键0、财务日期穿越0、全空因子0。测试结果位于 `quant_data/processed/features_50_test/`。

全市场构建已于2026-09-08完成：共13,030,829行、5,390只股票，信号日期为2010-01-04至2026-08-27；重复股票日期键0、财务日期穿越0，50个因子均非全空。正式结果由17个年度Parquet文件组成，约3.26 GB，完整覆盖率见 `quant_data/processed/features_50/feature_report.json`。

全市场窗口计算默认使用 `--memory-limit 8GB`。DuckDB超过该限额时会将中间排序和连接数据写入输出目录下的临时 `_tmp` 目录，成功结束后自动释放，以降低Windows内存不足导致任务中断的风险。可以按机器配置修改，例如 `--memory-limit 6GB`。

这些是未做横截面缩尾和标准化的原始候选因子。缩尾、行业/市值中性化、标准化及最终筛选必须只用训练期拟合，并在构建模型数据集时完成。

## 11. 构建滚动验证数据集

`build_model_dataset.py` 将50个因子和未来5日收益标签按 `ts_code + signal_date` 合并，并采用只沿时间向前推进的扩展窗口（expanding window）验证，禁止随机切分。

默认时间安排如下：

| 用途 | 时间范围 | 说明 |
| --- | --- | --- |
| 因子预热 | 2010年 | 只用于形成60日等滚动因子，不进入模型样本 |
| 滚动训练/验证 | 2011–2024年 | 验证年份从2017逐年滚动到2024；训练期随年份扩展 |
| 最终Training | 2011–2023年 | 用于最终候选模型训练 |
| 最终Validation | 2024年 | 冻结前一次性确认候选方案，不再开展新一轮搜索 |
| 永久Test | 2025-01-02至2026-08-20 | 已完成唯一一次正式评估并封存，不能用于继续调参或切换模型 |

滚动折依次为：2011–2016训练、2017验证；2011–2017训练、2018验证；一直到2011–2023训练、2024验证。数据只保存一份，具体每一折的日期规则和样本数写入 `rolling_folds.json`，不重复复制八份训练数据。

由于主标签使用 `t+1` 开盘到 `t+5` 收盘的收益，每个时间边界都执行 purge：前一组只保留 `exit_date` 严格早于后一组起点的样本。这样可避免一条训练标签的收益区间跨入验证期或测试期。

在全市场运行前，需要先完成全量因子与标签：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" build_features.py
& "E:\Anaconda\envs\quant_ml\python.exe" build_targets.py
& "E:\Anaconda\envs\quant_ml\python.exe" build_model_dataset.py
```

默认输出到 `quant_data/processed/model_dataset/`，按 `final_split` 和 `signal_year` 分区。关键输出包括：

- `final_split=training/`：最终训练集。
- `final_split=validation/`：最终验证集。
- `final_split=test/`：永久测试集。
- `dataset_report.json`：总行数、股票数、重复键、缺失标签和三组行数。
- `rolling_folds.json`：各年度扩展窗口的训练/验证边界及行数。

脚本默认要求每行至少30个非空因子，可用 `--min-valid-features` 修改。它不会提前对全数据做缩尾、缺失值填充或标准化；这些变换必须在每一折的Training中拟合，再原样应用到对应Validation和最终Test，防止数据泄漏。

20只股票、2023–2025年联调已经通过：训练4,725行、验证4,715行、测试4,845行，共14,285行；重复股票日期键0，缺失Target为0。测试结果位于 `quant_data/processed/model_dataset_rolling_test/`。

全市场模型数据集已于2026-09-09构建完成：共12,379,334行、5,370只股票，约3.26 GB。最终Training为9,102,082行（2011-01-04至2023-12-22），Validation为1,207,408行（2024-01-02至2024-12-24），永久Test为2,069,844行（2025-01-02至2026-08-20）。重复股票日期键0、缺失Target为0、时间边界违规0。完整统计位于 `quant_data/processed/model_dataset/dataset_report.json`。

合并脚本默认使用 `--memory-limit 8GB` 控制DuckDB内存；超出部分写入临时目录。正式数据只保存一份，通过 `final_split` 和 `signal_year` 分区，各滚动折根据 `rolling_folds.json` 中的日期规则读取，避免重复复制多年训练集。

## 12. 滚动训练三个模型

训练代码现已按模型拆分，同时共用完全相同的数据与评估逻辑：

- `train_ols.py`：多元线性回归（OLS）Baseline。
- `train_lightgbm.py`：LightGBM树模型Baseline。
- `train_mlp.py`：64→32→16的MLP。
- `training_common.py`：三者共用的数据加载、滚动折、预处理、指标和预测保存，确保比较口径一致。
- `train_models.py`：兼容入口，默认依次调用上述三个训练脚本。

安装新增依赖：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" -m pip install lightgbm torch
```

分别进行全市场训练：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" train_ols.py
& "E:\Anaconda\envs\quant_ml\python.exe" train_lightgbm.py
& "E:\Anaconda\envs\quant_ml\python.exe" train_mlp.py
```

也可以继续使用一键入口：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" train_models.py
```

只运行指定模型：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" train_models.py --models ols
& "E:\Anaconda\envs\quant_ml\python.exe" train_models.py --models lightgbm mlp
```

每个滚动折都只用该折Training估计因子中位数、1%/99%缩尾边界、均值和标准差，再将相同参数应用于对应Validation。OLS与LightGBM使用完整Training；为保证CPU训练可执行，MLP默认从每折完整Training中按固定随机种子抽取300,000行，但Validation保持全量。可通过 `--mlp-max-train-rows` 修改，设为0表示MLP也使用全部Training。

训练以Validation的逐日Rank IC评估MLP并执行Early Stopping。每个独立脚本都有自己的 `training_progress.json`，重新运行相同命令会跳过已完成的滚动折并安全续跑。每折保存预处理参数、模型文件、验证预测和指标；MLP另外保存训练历史。在这一滚动训练阶段，永久Test尚未被读取；它后来只在模型与规则全部冻结后执行了一次正式评估。

20只股票联调已经通过。全市场八折训练已于2026-09-09完成，共生成24个模型、24个验证预测文件和24份指标报告，耗时约9.87分钟。三个模型在八折上的平均结果为：

| 模型 | 平均逐日IC | 平均逐日Rank IC | 平均Top-Bottom 5日对数收益差 | 平均RMSE |
| --- | ---: | ---: | ---: | ---: |
| OLS | 0.04651 | 0.04530 | 0.00856 | 0.06472 |
| LightGBM | 0.05026 | 0.04248 | 0.00989 | 0.06462 |
| MLP | 0.01770 | 0.01738 | 0.00348 | 0.07186 |

原统一脚本完成的正式结果仍完整保留在 `quant_data/processed/model_training/`，拆分不会修改或删除它。三个新脚本已分别通过20只股票的真实滚动折联调；以后单独全量运行时默认写入 `model_training_ols/`、`model_training_lightgbm/` 和 `model_training_mlp/`。这些指标尚未计入手续费、换手率和成交约束，不能直接视为可实现策略收益。

### OLS优化记录总览（十阶段）

这里的“OLS优化”是指从无正则多元线性回归出发，对同一线性模型族逐步加入正则化、日期权重和稳定因子筛选。最终模型已经不是原始OLS，而是19因子Elastic Net线性Baseline。全部参数选择只使用2017–2023七个滚动Validation折，2024与永久Test在优化期间保持隔离。

| 阶段 | 测试内容 | 主要候选或参数变化 | 核心结果 | 决定 |
| --- | --- | --- | --- | --- |
| 1 | OLS与Ridge | `alpha=0–1000` | Ridge `alpha=1000` 的Mean Rank IC为0.06235，高于OLS的0.04402，最差年度也明显改善 | **接受Ridge思想**，用L2缓解50因子共线性 |
| 2 | Lasso | `alpha_ratio=0.0001–0.5` | 最佳0.3的Mean Rank IC为0.05145、平均约14个非零因子，但低于Ridge且存在负年度 | **拒绝纯Lasso**作为最终模型 |
| 3 | Elastic Net | 7档`l1_ratio` × 4档`alpha_ratio` | `l1_ratio=1e-6`、`alpha_ratio=0.1`的Mean Rank IC为0.06343，略高于Ridge | **接受弱L1加L2**，兼顾稳定与有限筛选 |
| 4 | 每日等权 | 普通样本等权对每个交易日总权重相同 | Elastic Net Mean Rank IC由0.06343升至0.06437，选择分数由0.05783升至0.05839 | **接受日期权重思路** |
| 5 | Training Target缩尾 | 不缩尾、0.5%/99.5%、1%/99% | 两档缩尾均降低Rank IC、最差年度和Top-Bottom | **拒绝缩尾**，保留原始未来5日Target |
| 6 | 每日截面因子标准化 | Training全局Z-score对每日截面Z-score | 截面标准化的Elastic Net Mean Rank IC降至0.05917，Top-Bottom也下降 | **拒绝截面Z-score**，保留Training全局标准化 |
| 7 | 市场相对收益Target | 原始收益对减去当日全市场中位数 | 相对Target的Mean Rank IC降至0.05118，最差年度转负 | **拒绝相对Target**，保留绝对收益标签 |
| 8 | 日期权重指数 | `gamma=0、0.25、0.5、0.75、1` | `gamma=0.75`选择分数0.05849，略高于完全每日等权的0.05839 | **接受`gamma=0.75`**，但将微小优势谨慎解释 |
| 9 | Elastic Net局部精调 | 20组Elastic Net加4组Ridge | `l1_ratio=1e-7`、`alpha_ratio=0.2`的Mean Rank IC为0.06476、选择分数0.05873 | **接受精调参数**，七个年度均为正 |
| 10 | 稳定因子子集 | 全部50因子、17因子、19因子 | 19因子Mean Rank IC为0.06594、选择分数0.05941，平均表现最好且更简洁 | **选择19因子候选**，随后在2024一次性确认 |

最终接受的改动包括Elastic Net正则化、`gamma=0.75`日期权重、`l1_ratio=1e-7`、`alpha_ratio=0.2`和19个稳定因子；拒绝纯Lasso、Target缩尾、每日截面Z-score与市场相对收益Target。下面保留每个阶段的详细实验记录和运行入口。

### OLS优化第一阶段：OLS与Ridge

`tune_ols.py` 使用2017–2023七个滚动Validation折比较原始OLS和不同强度的Ridge。该阶段把2024保留为后续确认折，并完全隔离2025–2026永久Test。所有方案沿用相同的Training内预处理；Ridge采用中心化且按样本数归一化的 `X'X / n + alpha * I`，截距不参与惩罚。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_ols.py
```

默认搜索 `alpha = 0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000`；其中0为无正则OLS。选择分数为“2017–2023平均逐日Rank IC - 0.25 × 年度Rank IC标准差”。进度与结果保存在 `quant_data/processed/model_tuning_ols_ridge/`，核心汇总文件为 `tuning_summary.json`；重新运行会按折续传，使用 `--force` 时旧目录会先被备份。

第一阶段全量运行结果：无正则OLS的平均Rank IC为0.04402，`alpha=1000` 为0.06235；`alpha=300` 与1000已经进入近似平台区。该结果仅用于确定下一步候选方案，尚未使用2024确认折，也未读取Test，不能作为最终样本外成绩。

### OLS优化第二阶段：Lasso

`tune_lasso.py` 在相同的2017–2023滚动折上测试Lasso。为使不同长度训练窗口可比较，每折使用 `alpha = alpha_ratio × alpha_max`；坐标下降直接利用分块累计的Gram矩阵，并保存收敛状态和非零因子数量。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_lasso.py
```

默认搜索 `alpha_ratio = 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5`。正式结果保存在 `quant_data/processed/model_tuning_lasso/`。

第二阶段全量运行中，最佳选择分数来自 `alpha_ratio=0.3`：平均Rank IC为0.05145，平均保留14.14个因子，但最差年度Rank IC为-0.00305，七个年度中有六个为正。它优于无正则OLS，但弱于第一阶段Ridge的平均Rank IC 0.06235和最差年度0.01833，因此Lasso暂不作为领先候选。该比较仍未使用2024确认折和永久Test。

### OLS优化第三阶段：Elastic Net

`tune_elastic_net.py` 在相同七折上联合搜索 `l1_ratio` 与 `alpha_ratio`。目标函数同时包含L1稀疏惩罚和L2稳定化惩罚；每折保存全部模型、指标、非零因子数和收敛状态。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_elastic_net.py
```

默认搜索7个L1占比和4个正则强度，共28组参数、196个滚动模型。正式结果保存在 `quant_data/processed/model_tuning_elastic_net/`。

第三阶段最优组合为 `l1_ratio=0.000001, alpha_ratio=0.1`：平均Rank IC为0.06343，年度标准差0.02243，最差年度0.01721，平均保留36.71个因子，七个年度全部为正。它的选择分数0.05783略高于Ridge的0.05693，说明当前最佳方案本质上是“强L2稳定化 + 极弱L1筛选”。该结论尚未使用2024确认折和永久Test。

### OLS优化第四阶段：每日等权

`tune_daily_weights.py` 固定前述Ridge和Elastic Net候选参数，只改变Training损失中的样本权重。股票数量不同的交易日权重总和都设为1，避免后期股票数量更多的日期自动获得更大影响。预处理仍保持原口径，因此本轮是对训练损失权重的单变量消融实验。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_daily_weights.py
```

正式结果保存在 `quant_data/processed/model_tuning_daily_weights/`。每日等权后，Ridge平均Rank IC从0.06235提高至0.06333，选择分数从0.05693提高至0.05753；Elastic Net平均Rank IC从0.06343提高至0.06437，选择分数从0.05783提高至0.05839。两者最差年度也有改善，但平均IC和Top-Bottom收益差略降。由于项目主要目标是稳定的截面排序，每日等权暂时判定为有效改进。该阶段仍未使用2024确认折和永久Test。

### OLS优化第五阶段：Training Target缩尾

`tune_target_winsor.py` 保持每日等权与模型参数不变，只比较不缩尾、0.5%/99.5%缩尾和1%/99%缩尾。阈值仅从每折Training标签计算并只修改拟合目标；Validation标签始终保持原始值。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_target_winsor.py
```

正式结果保存在 `quant_data/processed/model_tuning_target_winsor/`。对Elastic Net，0.5%缩尾使平均Rank IC从0.06437降至0.05801，1%缩尾进一步降至0.05189；Ridge也出现相同下降。说明当前未来5日收益标签的尾部样本包含有用排序信息，因此Target缩尾被判定为无效改动，领先方案继续使用原始Training Target。该阶段仍未使用2024确认折和永久Test。

### OLS优化第六阶段：每日截面因子标准化

`tune_cross_sectional.py` 保持Training拟合的缺失值填充、1%/99%缩尾边界和有效因子筛选，只将全局均值/标准差替换为每个 `signal_date` 内的股票截面Z-score。Validation也只使用同一信号日的股票截面，不会引用未来日期。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_cross_sectional.py
```

正式结果保存在 `quant_data/processed/model_tuning_cross_sectional/`。截面标准化后，Ridge平均Rank IC为0.05946，Elastic Net为0.05917，均低于全局Training标准化配合每日等权的0.06333和0.06437；Top-Bottom收益差也下降。虽然普通IC略有改善，但项目主要目标是稳定截面排序，因此每日截面Z-score被判定为无效改动，领先方案继续使用Training全局标准化。该阶段仍未使用2024确认折和永久Test。

### OLS优化第七阶段：市场相对收益Target

`tune_relative_target.py` 比较原始未来5日收益与“股票未来5日收益减去同一 `signal_date` 全市场未来收益中位数”。相对收益只作为Training标签和Validation评估标签，绝不作为模型特征；生产预测不需要知道未来市场中位数。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_relative_target.py
```

正式结果保存在 `quant_data/processed/model_tuning_relative_target/`。Elastic Net使用相对Target后的平均Rank IC为0.05118，低于原始Target的0.06437，最差年度从0.01817降至-0.00143；Ridge也出现相同下降。相对Target的RMSE较低是因为标签去除了每日共同成分、尺度发生变化，不能与原始Target的RMSE直接比较。因此相对收益Target被判定为无效改动，领先方案继续使用原始未来5日收益标签。该阶段仍未使用2024确认折和永久Test。

### OLS优化第八阶段：每日权重指数

`tune_weight_exponent.py` 将单只股票权重定义为 `N_t ** (-gamma)`，比较 `gamma = 0, 0.25, 0.5, 0.75, 1`。其中0是普通样本等权，1是完全每日等权。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_weight_exponent.py
```

正式结果保存在 `quant_data/processed/model_tuning_weight_exponent/`。最佳 `gamma=0.75`，平均Rank IC为0.06437，选择分数0.05849；完全每日等权的选择分数为0.05839。差异很小，但后续局部调参采用0.75。

### OLS优化第九阶段：Elastic Net局部精调

`refine_elastic_net.py` 固定 `gamma=0.75`，搜索4个极低L1占比、5个alpha比例，并加入4个纯Ridge候选，共24组参数。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" refine_elastic_net.py
```

正式结果保存在 `quant_data/processed/model_tuning_elastic_refined/`。最佳配置为 `l1_ratio=1e-7, alpha_ratio=0.2`：平均Rank IC 0.06476、选择分数0.05873、平均非零因子27.86，七个年度全部为正。跨折稳定因子定义保存在 `stable_factor_subsets.json`。

### OLS优化第十阶段：稳定因子子集

`tune_factor_subsets.py` 使用上述最佳参数比较全部50因子、七折全部入选的17因子，以及至少六折入选的19因子。

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" tune_factor_subsets.py
```

正式结果保存在 `quant_data/processed/model_tuning_factor_subsets/`。19因子子集平均Rank IC为0.06594、选择分数0.05941，优于全部50因子的0.06476和0.05873；但最差年度由0.01673降至0.01310，Top-Bottom也略降。因此当时将19因子作为平均表现最佳的候选、50因子作为更稳健的对照，随后在尚未使用的2024 Validation上完成确认。上述第八至第十阶段当时均未读取2024和永久Test。

### LightGBM优化记录（四阶段）

LightGBM使用2017–2023七个滚动Validation折优化，2024和永久Test在参数搜索期间保持隔离。四个阶段分别由 `tune_lightgbm_structure.py`、`tune_lightgbm_learning_rate.py`、`tune_lightgbm_sampling.py` 和 `tune_lightgbm_regularization.py` 执行。

| 阶段 | 优化内容 | 原始值 | 选定值 | 关键结果与选择原因 |
| --- | --- | --- | --- | --- |
| 1 | 树结构 | `num_leaves=31`、`min_data_in_leaf=1000` | **1023、6000** | Mean Rank IC由0.03354升至0.06041，最差年度由-0.01432升至0.03823；大树表达非线性交互，较大的叶节点样本数同时限制过拟合 |
| 2 | 学习率与轮数 | `learning_rate=0.03`、最多500轮、patience 50 | **0.0075、最多350轮、patience 100** | 0.005的选择分数只高0.000032，但平均训练轮数多约53%；0.0075效率更高且IC、RMSE略好，因此接受0.0075 |
| 3 | 特征与样本采样 | `feature_fraction=0.8`、`bagging_fraction=0.8` | **0.3、1.0**，搜索上限扩为450轮 | `feature_fraction=0.3`的Mean Rank IC为0.06619、选择分数0.06353；全量行采样小幅领先，不需要主动丢弃Training样本 |
| 4 | L1/L2正则化 | `lambda_l1=0.1`、`lambda_l2=1` | **0、1** | L2=1在粗搜和局部搜均第一；非零L1没有稳定优势。增益很小，因此停止继续细调，避免对七折过拟合 |

LightGBM阶段候选最终为：

```text
num_leaves=1023
min_data_in_leaf=6000
learning_rate=0.0075
feature_fraction=0.3
bagging_fraction=1.0
lambda_l1=0
lambda_l2=1
maximum_rounds=450
early_stopping_rounds=100
```

参数冻结后才运行2024确认：Rank IC为0.071338，最佳轮数444/450。随后使用2011–2024重训时，按冻结规则采用244轮。每轮候选、逐折指标、边界搜索和接受/拒绝原因见 `LightGBM_optimize_summary.md`。

### MLP优化记录（八阶段）

MLP的主要实验由 `tune_mlp_training.py` 执行。调参时使用Training内部时间早停选择epoch，再用完整Training重训并只在外层Validation评分，避免直接用同一外层Validation选择epoch又汇报成绩。

| 阶段 | 优化内容 | 参数变化或候选 | 关键结果与结论 |
| --- | --- | --- | --- |
| 1 | 原始基线诊断 | `64→32→16`、Dropout `0.20/0.15/0.10`、LR 0.001、batch 8192、15 epochs | 七折Mean Rank IC 0.02087；训练MSE下降但Validation可能恶化，确认原外层早停存在乐观偏差，只保留为基线 |
| 2 | 内部早停与样本量 | Training内部时间早停；30万对60万行；上限30 epochs、patience 5 | 30万行选择分数0.03643，高于60万的0.03513，且更快；接受30万行与内部早停流程 |
| 3 | 学习率与Batch Size | LR 0.001/0.0005/0.002；batch 8192/4096 | 选定 **LR 0.0005、batch 4096**；七折Mean Rank IC由0.03912升至0.04129，七年全部为正 |
| 4 | 网络容量 | `32→16→8`、`64→32→16`、`128→64→32` | 选定 **128→64→32**；七折Mean Rank IC升至0.04941，是MLP最明显的单阶段提升；小网络因欠拟合被拒绝 |
| 5 | Dropout | 高、中、低三档 | 低Dropout **0.10/0.075/0.05** 的均值略低但年度波动和最差年度更好，选择分数0.04684略高，进入多种子复核 |
| 6 | Weight Decay | 0、`1e-5`、`1e-4`、`1e-3` | 0与`1e-5`在保存精度内一致，更强正则化降低选择分数；保留原`1e-5` |
| 7 | 多随机种子稳定性 | 种子7、42、2026；低Dropout对原Dropout | 低Dropout在9组同种子同年份配对中胜出8次、9/9 Rank IC为正，正式冻结低Dropout |
| 8 | 2024独立确认 | 全部参数冻结后只确认一次 | Rank IC 0.070824、正值率73.84%，内部早停选中第17轮；确认通过后不再修改参数 |

MLP验证阶段的冻结配置为：

```text
hidden_layers=128,64,32
activation=ReLU
dropout=0.10,0.075,0.05
optimizer=AdamW
learning_rate=0.0005
batch_size=4096
weight_decay=0.00001
train_rows_per_fold=300000
maximum_epochs_during_validation=45
early_stopping_patience=5
seed=42
```

最终重训采用2011–2024数据和预先确定的16 epochs。60万样本、小网络、高Dropout、更强Weight Decay以及带意外BatchNorm的无效试验均被拒绝。完整候选表、逐折训练曲线、命令和结果目录见 `MLP_summary.md` 与 `MLP_optimize_summary.md`。

十个优化阶段的参数、完整指标、接受/拒绝结论、最终重训和永久Test结果已集中整理在 `OLS_optimize_summary.md`。最终部署的是19因子Elastic Net线性Baseline，而不是未经正则化的原始OLS；其永久Test Daily Rank IC为0.08601，总收益25.01%，Sharpe为0.739。

LightGBM各阶段的参数搜索、七折结果、2024确认、最终重训和永久Test结果集中整理在 `LightGBM_optimize_summary.md`。冻结模型在永久Test取得Daily Rank IC 0.06764、总收益33.63%、年化19.98%和Sharpe 0.845；该结果只作最终报告，不能用于继续调参。

MLP的完整总结见 `MLP_summary.md`，其中包含八阶段详细优化summary、每组候选参数与指标、接受/拒绝原因、2024确认、最终重训、永久Test、成交诊断与三模型比较；全部逐折训练曲线和实验命令仍保存在 `MLP_optimize_summary.md`。最终冻结配置为 `128→64→32`、Dropout `0.10/0.075/0.05`、`learning_rate=0.0005`、`batch_size=4096`、`weight_decay=1e-5`。低Dropout在9个种子×年份配对中胜出8次且全部Rank IC为正；2024确认Rank IC为0.070824。永久Test中MLP的Daily Rank IC为0.05695，总收益15.65%，Sharpe为0.481；结果不得用于继续调参。

三份模型summary现已采用统一结构：原始参数、每一步候选与参数变化、选择该优化的原因、接受/拒绝依据、2024确认、严格Validation组合回测、2011–2024最终重训、永久Test锁定与信号生成、预测评价、冻结组合回测和完成审计。OLS/线性模型见 `OLS_optimize_summary.md`，LightGBM见 `LightGBM_optimize_summary.md`，MLP见 `MLP_summary.md`。

### 三模型最终冻结配置与永久Test结果

| 角色 | 最终冻结配置 | Test逐日Rank IC | 总收益 | Sharpe | 最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: |
| 线性Baseline | 19因子Elastic Net；`gamma=0.75`、`l1_ratio=1e-7`、`alpha_ratio=0.2` | 0.08601 | 25.01% | 0.739 | -26.89% |
| LightGBM强对照 | 1023叶、`learning_rate=0.0075`、`feature_fraction=0.3`、`lambda_l2=1`、244轮 | 0.06764 | 33.63% | 0.845 | -33.04% |
| MLP预注册主模型 | `128→64→32`；Dropout `0.10/0.075/0.05`；`learning_rate=0.0005`；`batch_size=4096`；16 epochs | 0.05695 | 15.65% | 0.481 | -37.83% |

表中的模型角色在查看永久Test之前已经固定。MLP仍是预注册主模型；线性模型与LightGBM是比较基准。永久Test上对照模型表现更高是研究结论，而不是允许事后更换主模型或重新搜索参数的理由。

三个模型的2024统一确认结果位于 `validation_2024_summary.md` 和 `quant_data/processed/model_confirmation_2024_summary.json`。19因子线性模型、LightGBM、MLP的Rank IC分别为0.077216、0.071338、0.070824，全部为正；LightGBM的IC、Top-Bottom、RMSE与R²最佳，MLP的日度Rank IC正值率最高。

严格Validation组合回测已完成，完整方法与结果见 `validation_backtest_summary.md`。回测使用各滚动折的全市场样本外预测，先按信号日信息排名，再让次日无法买入的股票自然落空；采用5个等资金袖套、停牌与一字跌停延迟退出、复权价格和10/20/30 bps成本情景。只根据2017–2023设计期选择出的方案为LightGBM Top50（20 bps），设计期年化13.04%、Sharpe 0.733、最大回撤-33.86%；冻结后2024总收益35.08%、Sharpe 0.890、最大回撤-36.99%。这项方案设计发生在永久Test之前，当时没有读取2025–2026数据。

Validation压力测试也已完成，详见 `validation_stress_test_summary.md`。50 bps往返成本、1亿元成交额门槛和特殊涨跌幅代理剔除后，设计期收益仍为正。加入20 bps固定成本和保守平方根冲击后，1亿元组合的设计期年化为5.81%、Sharpe为0.390，2024总收益为14.41%；2亿元年化降至2.96%，3亿元仅0.83%，5亿元转负。因此在读取Test前冻结了1亿元资金、Top50、5%单票参与率上限和平方根冲击规则；压力测试阶段没有读取2025–2026永久Test。

最终模型已使用2011–2024全部非Test数据重新训练，详见 `final_retraining_summary.md`。训练池由9,102,082行、5,051只股票扩大到10,309,490行、5,160只股票；Target标准差由6.55%上升到6.80%，尾部也有所扩大。最终角色固定为MLP主模型、19因子线性Baseline和LightGBM强对照；MLP固定30万抽样与16 epochs，LightGBM固定244轮。三个模型均通过重新加载、哈希和非Test样本推理审计；随后永久Test已按冻结配置完成唯一一次正式评估。

2025–2026永久Test现已执行完毕，详见 `permanent_test_summary.md`。在冻结的1亿元Top50、20 bps固定成本和平方根冲击规则下，MLP主模型总收益15.65%、年化9.57%、Sharpe 0.481、最大回撤-37.83%；线性Baseline总收益25.01%、LightGBM强对照总收益33.63%。MLP日均Rank IC为0.05695且74.49%的交易日为正，说明信号有效，但没有超过两个对照模型。MLP仍保持预先声明的主模型身份，永久Test结果不得用于重新调参或切换模型。

永久Test后的结果解释、风险归因、图表和论文初稿也已完成。`analyze_test_results.py` 只读正式Test产物，生成基准Beta/Alpha、尾部风险、回撤、月度/年度收益、模型相关性和六张图表；集中解释见 `results_interpretation.md`，中文论文草稿见 `quant_research_paper_draft.md`。这些分析不改变模型或交易规则。

重新生成永久Test后的只读分析（不会训练或修改模型）：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" analyze_test_results.py
```

## 13. 读取清洗后的数据

### 使用 pandas 读取单个年份

```python
import pandas as pd

daily_2025 = pd.read_parquet(
    r"<PROJECT_ROOT>\quant_data\processed\parquet\daily\year=2025"
)
print(daily_2025.head())
print(daily_2025.shape)
```

### 使用 DuckDB 查询多个年份

```python
import duckdb

sql = r"""
SELECT ts_code, trade_date, close, vol, amount
FROM read_parquet(
    '<PROJECT_ROOT>/quant_data/processed/parquet/daily/year=*/part-*.parquet'
)
WHERE ts_code = '000001.SZ'
  AND trade_date BETWEEN DATE '2020-01-01' AND DATE '2025-12-31'
ORDER BY trade_date
"""

df = duckdb.sql(sql).df()
print(df.head())
```

全市场数据较大，不建议使用 pandas 一次性把所有接口、所有年份加载进内存。优先按年份、股票范围和需要的字段读取，或者使用 DuckDB 直接查询 Parquet。

## 14. 目录结构

```text
项目2/
├── main.py                         # 64只样本股票下载器
├── download_all_stocks.py          # 全市场非分钟下载器
├── download_extra.py               # 64只样本的补充接口
├── download_minutes.py             # 历史分钟线下载器（暂缓）
├── preprocess_data.py              # 数据审计和修复
├── build_parquet.py                # 清洗与Parquet转换
├── build_targets.py                # 未来5个交易日收益标签
├── build_features.py               # 50个候选量化因子
├── build_model_dataset.py          # 滚动验证与最终三组时间切分
├── training_common.py              # 三模型共用加载、预处理与评估
├── train_ols.py                    # 单独训练OLS
├── train_lightgbm.py               # 单独训练LightGBM
├── train_mlp.py                    # 单独训练MLP
├── train_models.py                 # 一键顺序运行三个独立脚本
├── generate_validation_signals.py  # 全市场滚动Validation样本外信号
├── backtest_validation.py          # 严格Validation组合回测
├── validation_backtest_summary.md  # 回测方法、结果与限制总结
├── stress_test_validation.py       # 成本、流动性、容量与暴露压力测试
├── test_market_impact.py           # 非线性市场冲击压力测试
├── validation_stress_test_summary.md # 压力测试结论与冻结建议
├── retrain_final_models.py         # 2011–2024三个冻结模型最终重训
├── final_retraining_summary.md     # 重训前后数据与最终模型清单
├── OLS_optimize_summary.md         # 线性模型逐阶段优化、验证、重训与Test完整总结
├── LightGBM_optimize_summary.md    # LightGBM逐阶段优化、验证、重训与Test完整总结
├── MLP_optimize_summary.md         # MLP逐阶段优化与2024确认记录
├── MLP_summary.md                  # MLP从优化、重训到永久Test的完整总结
├── generate_permanent_test_signals.py # 永久Test全市场预测
├── evaluate_permanent_test.py      # 永久Test唯一一次评估与组合回测
├── permanent_test_summary.md       # 永久Test最终报告
├── analyze_test_results.py         # 只读风险归因与论文图表生成
├── results_interpretation.md       # Test结果解释与风险归因报告
├── quant_research_paper_draft.md   # 中文量化研究论文初稿
├── README.md
└── quant_data/
    ├── cache/                       # 六类基础接口逐股票原始分片
    ├── extra_cache/                 # 八类补充接口逐股票原始分片
    ├── combined/                    # 公共表及早期64只股票合并表
    ├── all_stocks_progress.json     # 全市场下载最终进度
    ├── download_all_stocks.error.log
    ├── processed/
    │   ├── quality/                 # 审计报告和原始文件清单
    │   ├── repairs/                 # 独立保存的问题文件修复版本
    │   ├── parquet_test/            # 20只股票转换测试结果
    │   ├── parquet/                 # 全市场清洗后Parquet
    │   ├── targets_5d_test/          # 20只股票标签测试结果
    │   ├── targets_5d/               # 全市场未来5日标签
    │   ├── features_50_test/         # 20只股票因子测试结果
    │   ├── features_50/              # 全市场50因子
    │   ├── model_dataset_rolling_test/ # 20只股票滚动切分联调结果
    │   ├── model_dataset/             # 全市场Training/Validation/Test
    │   ├── model_training_smoke/      # 三模型小样本联调结果
    │   ├── model_training/            # 八折正式模型、预测和指标
    │   ├── model_training_ols/        # 独立OLS全量输出（运行后生成）
    │   ├── model_training_lightgbm/   # 独立LightGBM全量输出（运行后生成）
    │   ├── model_training_mlp/        # 独立MLP全量输出（运行后生成）
    │   ├── validation_signals/        # 2017–2024全市场样本外信号
    │   ├── validation_backtest/       # 组合回测汇总、比较表与逐日净值
    │   ├── validation_stress_tests/   # 压力情景、冲击模型与逐日净值
    │   ├── final_models_2011_2024/    # 最终三模型、预处理、数据对比与清单
    │   ├── permanent_test/            # Test锁定记录、预测、指标与净值
    │   ├── result_analysis/           # Test只读风险归因、统计表与六张图表
    │   ├── parquet_conversion.log
    │   └── parquet_conversion.error.log
```

原 64 只股票部分目录中可能同时存在旧 `.csv` 和全市场任务生成的 `.csv.gz`。审计清单会识别这种情况，Parquet 转换只选择一个有效来源，不会把两个版本重复加入训练数据。

## 15. 建模前必须注意

### 复权价格

`daily` 是未复权价格。计算历史收益率、技术指标和模型标签时，需要结合 `adj_factor` 构造一致口径的复权价格，不能直接跨除权除息日使用原始收盘价计算收益率。

### 防止未来数据泄漏

- 财务数据不能仅按报告期 `end_date` 回填到历史交易日。
- 应以 `ann_date` 或 `f_ann_date` 判断数据在当时是否已经公开。
- 特征在日期 `t` 的可见时间必须早于下单时间。
- 标签可以使用未来收益，但标签字段绝不能进入特征矩阵。
- 标准化、缺失值填充和特征选择只能在训练集上拟合。
- 数据集应按时间切分，不能随机打乱后再划分训练集和测试集。
- 需要考虑涨跌停、停牌、手续费和无法成交的情况。

### 建议的下一步

模型数据集、滚动切分、三模型优化、严格Validation组合回测、压力测试、2011–2024最终重训、2025–2026永久Test、风险归因、图表和论文初稿均已完成。当前建议按以下顺序收尾：

1. 将 `quant_research_paper_draft.md` 按学校或期刊格式补充作者、课程、摘要要求、引用和附录，并导出最终Word/PDF。
2. 从三份模型summary、`permanent_test_summary.md` 和 `results_interpretation.md` 交叉核对论文中的所有数字，保留Test只读审计边界。
3. 若目标是模拟盘或实盘，补充点时历史ST与行业分类，并用分钟成交数据校准冲击模型；这些属于新研究，必须预先定义新的前瞻样本外评估期。
4. 不要因为永久Test上LightGBM或线性模型优于MLP而回头改主模型、参数、因子或组合规则；本轮永久Test已经关闭。

## 16. 常见问题

### VS Code 找不到 Conda 环境

在“Python: Select Interpreter”中手动选择：

```text
E:\Anaconda\envs\quant_ml\python.exe
```

### 提示没有 `TUSHARE_API_KEY`

如果刚使用永久环境变量命令设置，需要完全重启 VS Code 或 Codex。也可以在当前 PowerShell 临时设置 `$env:TUSHARE_API_KEY` 后再运行。

### 运行下载命令后看起来没有新文件

先查看 `quant_data/all_stocks_progress.json`。断点下载器会跳过已有分片，因此重复运行时可能主要显示“跳过”，这不代表程序失效。

### VPN 关闭后是否还能下载

只要当前网络仍能访问 `https://teajoin.com`，下载就可以继续；如果代理地址不可达，请求会重试并将失败记录到日志。已经写入的分片不会丢失，网络恢复后可以续传。

### 电脑睡眠后是否继续下载

不会。Windows 进入睡眠或休眠后 Python 进程暂停，唤醒后才可能继续。锁屏但不睡眠通常不影响后台下载。

### 是否应该现在下载分钟数据

暂时不建议。全市场多年分钟线体积和请求规模远大于日频数据，并且需要独立分钟权限。应先完成日频数据清洗、特征工程、模型和回测流程，再根据策略是否确实需要分钟级信号决定。
