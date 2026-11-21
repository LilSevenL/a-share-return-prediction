# V2 本地特征扩展到 2024–2026：方案

日期：2026-11-21。评审会话撰写，用户已确认方向。2026-11-21 修订：用户决定 2026 年（截至 2026-08-27）也纳入测试期，一并生成。

## 1. 目的与边界

- 目的：用与 2011–2023 完全相同的代码和因子定义，生成 2024、2025 和 2026 年（截至本地数据最后一天 2026-08-27）的 V2 本地特征，作为 V2 系列模型新的**一次性测试期**。
- 授权：用户于 2026-11-20 确认把 2024–2025 用作 V2 测试期，2026-11-21 决定 2026 年也纳入。之后不再有未使用的备用期；以后的独立检验只能用 2026-08-27 之后新积累的数据。
- 本任务只生成和检查数据：**不读取标签数值、不计算任何因子与收益的关系（IC、分组收益等）、不训练模型**。只允许检查标签文件的覆盖（行数、有效标记计数）。
- 不因为任何因子在 2024–2025 的表现或分布删除、修改或替换因子。
- 不改 V1 文件，不改 2011–2023 的已有特征目录。

## 2. 已知事实（评审会话检查过）

- 源数据：`parquet/daily`、`adj_factor`、`daily_basic`、`moneyflow` 都有 2024、2025、2026 年分区；`features_50`（构建的基础表）和 `targets_5d` 都是 2026-09-08 用同一套代码一次性建到 2026-08-27，所以 2024–2025 与之前年份口径一致。2025 年末信号的 5 日标签会用到 2026 年初价格，已有。
- 构建入口 `build_local_features.py`，之后由 `repair_local_numerics.py` 生成 stable 版本；训练读取的是 stable 版本。
- 年份限制写死在代码里：`build_local_features.py` 要求 `end_year ≤ 2023`；`build_balanced_features.files()` 拒绝 `last > 2024`；`repair_local_numerics.py` 的源和目标目录写死为 2011_2023。
- 预热：构建从 `max(2010, start_year − 2)` 开始读取历史。
- 股票池：本地已有股票（幸存者股票池）。退市股票的历史补采因 API 返回 403 一直未完成（`historical_universe_repair` 状态为 blocked）。2024–2025 年退市较多，测试期的幸存者偏差可能比之前更大，必须量化并写进报告。

## 3. 做法

1. **只做最小代码改动，放在新文件里。** 新建 `build_local_features_ext.py`、`repair_local_numerics_ext.py`（必要时 `build_balanced_features` 的 `files` 包装），相对原文件只允许改：年份上限、输出目录、源和目标目录。因子计算模块 `balanced_features.py`、`stable_rolling.py`、`features_500.py`、`research_design/factor_catalog_500.json` 一律不改，sha256 必须与原 `local_features_2011_2023/build_config.json` 中记录的一致。用 diff 记录新旧文件的全部差异。
2. **一次构建 2022–2026。** `start_year = 2022`、`end_year = 2026`（预热从 2020 年开始；2026 年只到本地数据最后一天）。输出父目录 `quant_data/processed/v2/local_features_2022_2026`，再用修复脚本生成 `local_features_2022_2026_stable`。
3. **重叠核对（关卡）。** 2022、2023 两年与现有 `local_features_2011_2023_stable` 逐项比较：
   - 键（ts_code、signal_date）集合与顺序一致；
   - 市场背景（每日 rm、market_median_turnover）在重叠日期上一致；
   - 500 列逐列：缺失掩码一致，数值一致（float32 存储，容差 |差| ≤ 1e-6 × max(1, |值|)）；51 个隔离列保持隔离。
   - 有任何列不一致：停下，报告该列、不一致行数、最大差、涉及日期与股票；判断是否由预热长度（2 年 vs 原来从 2010 年起）造成。不允许为通过核对改代码或放宽容差。
4. **2024–2026 质量检查**，与原构建同口径：月度键与重复、无穷值、数学范围、449 列可计算 / 51 列隔离、逐因子逐年覆盖率；总行数与源股票日期数一致。另外：
   - 逐因子比较 2024、2025、2026 与 2023 的覆盖率，覆盖率下降超过 10 个百分点的列入提示（只提示，不删因子）；
   - 资金流、估值类字段（moneyflow、daily_basic）在 2024–2026 的缺失情况单独列出。2026 年只有约 8 个月，按月给出行数，确认最后一个月是否完整。
5. **股票池与幸存者偏差**：
   - 2022–2026 每年的股票数、股票日期数；
   - 本地数据中最后交易日早于 2026-08 的股票数（本地能看到的退市或停牌）；
   - 与 `stock_basic` / `status_D` 元数据对照：2024–2026 年元数据中已知退市、但本地没有价格的股票数；
   - 结论写明：测试期的偏差方向（缺少退市股，偏向高估多头收益）与大致规模。
6. **标签覆盖**（只计数，不读数值做任何统计）：2024、2025、2026 每年信号行数、有效标签数、最后一个有效信号日；确认 2025 年 12 月信号的标签存在；2026 年末尾约 5 个交易日没有 5 日标签，属正常，列出具体日期。
7. **使用清单**：写 `quant_data/processed/v2/local_features_manifest.json`，说明 2011–2023 读哪个目录、2024–2026 读哪个目录，各自 build_config 的 sha256，以及"2024–2026（至 2026-08-27）为测试期"。

## 4. 产出

- `v2/FEATURE_EXTENSION_2024_RESULTS.md`：代码差异、重叠核对、质量检查、股票池与幸存者偏差、标签覆盖、使用清单。
- 数据目录：`local_features_2022_2026`、`local_features_2022_2026_stable`、`local_features_manifest.json`。

## 5. 之后

数据通过检查后，测试期的使用另行制定方案：先确定要测的模型（例如冻结的 V2 LightGBM FINAL、改用排名标签的线性候选），写明指标和判定规则，再一次性运行。

The original narrow extension changes are archived in [build diff](feature_extension_build.diff) and [repair diff](feature_extension_repair.diff). These describe the flat-layout research sources. Published import/path adaptations are tracked separately in publication_sources.json.
