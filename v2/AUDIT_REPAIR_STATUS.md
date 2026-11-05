# 因子审计后的修复记录

2026-10-24。本轮落实审计修复，V1、旧特征表和旧模型全部保留。没有启动模型训练或使用2024及以后分区。

## 已修改

1. `balanced_features.py`重算`amount_ratio_5d`、`amount_ratio_20d`、`volume_ratio_20d`、`turnover_zscore_20d`、`net_mf_ratio_5d`。统一在完整市场交易日索引上滚动，停牌缺口不再被旧股票观测行窗口跨过。
2. `negative_earnings_flag`固定隔离为空，不将缺失PE视作亏损。目录版本更新为`balanced_500_audit_repair_v2`；候选总数仍500，隔离51，可计算449。旧版本模型因目录哈希不同不能直接当新版本模型载入。
3. `repair_historical_universe.py`合并现有上市与退市元数据，按上市/退市区间确定历史资格，不按今天是否L状态筛选。缺少退市日期的D状态记录会隔离，冲突日期报错；旧L记录不能覆盖已知退市记录。
4. `run_research.py`增加正式全市场入口检查：未经历史股票池修复验证的数据不能作为正式全市场实验运行。小规模工程验证仍可运行并明确标注。

## 历史股票池与外部阻塞

`quant_data/processed/v2/historical_universe_repair/`保存新名单、元数据来源哈希、待补采任务和访问探测记录。2010–2023区间得到5507个已知股票代码，其中274个不在原主名单，五接口共1370项任务。此前审计的229个缺失代码对应2020–2023区间；时间范围不同，数量不矛盾。

这里是基于当前缓存的历史名单重建，不代表已完全覆盖历史代码变更、重新上市与所有遗漏身份，也不表示价格数据已经补齐。当前主来源中新增274个代码的五类缓存均缺失。

通过Tushare SDK及`https://teajoin.com`探测退市和正常上市股票的已知交易日期，请求均收到HTTP403。检查发现本机SDK将非成功HTTP响应转换为空DataFrame，容易误记为“无历史数据”。新增下载器会检查HTTP状态，401/403立即停止；不会把这类返回写成空历史缓存，也不会记录密钥或响应正文。

本轮`--download`运行已保存HTTP403的安全诊断，未启动1370项批量请求。不能断言是密钥、权限、代理服务或访问限制中的哪一种原因；仅凭403无法区分。访问恢复后仍需补采、清洗合并、重算市场背景/特征及标签，再进行退市退出估值审计。

## 修复后的数据与验证

新表目录：`quant_data/processed/v2/balanced_500_audit_repaired/`。这是原80只股票的公式修复复验表，仍使用旧股票池源，不能称为已解决幸存者偏差。

已重建38623行，449列有数据、51列隔离。与旧表按股票日期逐格核对，数值容差外的变化仅为：5项窗口共1014个单元格按连续日规则改为空值，亏损标记30349个原非空值改为隔离，以及随之更新的有效列计数。其他因子没有发现超出容差的变化，明细保存为`repair_delta.json`。

复审目录：`quant_data/processed/v2/factor_audit_repaired/`。最终复算结果以该目录`summary.json`为准。

复审已完成：67项独立公式/恒等式检查全部无差异（修复前5项失败）；重复键、无穷值、有效列计数不一致均为0，四个构建源码/目录哈希一致。逐因子状态为429项样本统计初检通过、20项覆盖率等待核查、51项隔离。股票池缺少退市价格的结论未改变，不能将公式复审通过解释成全市场审计通过。

测试已通过：历史日期资格/退市/冲突保护与HTTP403处理5项，因子及流程9项，500列模型输入契约4项，共18项。新增停牌测试覆盖本轮修复的全部5列；完整日历参考测试核对比值与均值。

## 恢复操作

在SDK代理访问恢复后，从项目根目录执行：

```powershell
& E:\Anaconda\envs\quant_ml\python.exe v2\repair_historical_universe.py --download
```

脚本先检查已知交易日正常股票，再分年下载并保存断点；年度窗口避免接口行数上限。空数据记录保留为待核查，不自动视作完整历史。下载完成也不会自动宣称数据修复完成或开始训练，仍需清洗合并与审计。

修复样本与复审可复现为：

```powershell
& E:\Anaconda\envs\quant_ml\python.exe v2\build_balanced_features.py --start-year 2022 --end-year 2023 --max-stocks 80 --output-dir quant_data/processed/v2/balanced_500_audit_repaired
& E:\Anaconda\envs\quant_ml\python.exe v2\audit_factors.py --features-dir quant_data/processed/v2/balanced_500_audit_repaired --output-dir quant_data/processed/v2/factor_audit_repaired
```
