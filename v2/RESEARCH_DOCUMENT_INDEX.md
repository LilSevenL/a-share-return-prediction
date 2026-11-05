# V2 研究步骤与文档目录

更新至 2026-11-04，共 38 个阅读阶段。

先阅读[总步骤与结果总结](V2_PROGRESS_SUMMARY.md)，可集中查看阶段进展、模型分数、交易收益与压力测试结果。

本目录按研究流程排列已存在的阶段材料。一次研究阶段可能跨越多次“继续下一步”，编号用于阅读导航，并非聊天轮次。方案记录实验设计，结果报告记录输出，核验JSON和保存的模型、预测、净值、订单提供进一步证据。

| 顺序 | 阶段 | 主要文档 |
|---|---|---|
| 1 | V2总体训练安排 | [训练方案](TRAINING_PLAN.md) |
| 2 | 500因子设计与分组 | [设计目录](research_design/FACTOR_CATALOG_500.md)、[实现说明](FEATURES_500_README.md) |
| 3 | 损失函数设计 | [损失设计](LOSS_DESIGN.md) |
| 4 | 50到500因子迁移 | [迁移说明](MIGRATION_50_TO_500.md)、[集成结果](INTEGRATION_RESULTS.md) |
| 5 | 因子数据审计与修复 | [审计报告](FACTOR_AUDIT_REPORT.md)、[修复状态](AUDIT_REPAIR_STATUS.md) |
| 6 | 使用本地数据构建因子 | [构建说明](LOCAL_DATA_BUILD.md)、[稳定版本质量报告](../reports/v2/local_features_2011_2023_stable/QUALITY_REPORT.md) |
| 7 | OLS基准训练 | [方案](OLS_BASELINE_PLAN.md)、[结果](../reports/v2/ols_baseline_stable/OLS_REPORT.md) |
| 8 | OLS诊断 | [方案](OLS_DIAGNOSTIC_PLAN.md)、[结果](../reports/v2/ols_diagnostics_stable/DIAGNOSTIC_REPORT.md) |
| 9 | 初始交易回测 | [方案](OLS_BACKTEST_PLAN.md)、[结果](../reports/v2/ols_backtest_stable/BACKTEST_REPORT.md) |
| 10 | 降换手与延长持有 | [方案](LOW_TURNOVER_PLAN.md)、[结果](../reports/v2/ols_low_turnover/LOW_TURNOVER_REPORT.md) |
| 11 | 跨年连续账户 | [方案](CONTINUOUS_BACKTEST_PLAN.md)、[结果](../reports/v2/ols_continuous/CONTINUOUS_REPORT.md) |
| 12 | 稳健损失单折试验 | [方案](LOSS_PILOT_PLAN.md)、[结果](../reports/v2/loss_pilot_2017/LOSS_PILOT_REPORT.md) |
| 13 | 求解器优化 | [优化说明](LOSS_SOLVER_OPTIMIZATION.md) |
| 14 | 七年同样本损失比较 | [方案](MULTIYEAR_LOSS_PLAN.md)、[结果](../reports/v2/multiyear_losses_200k/MULTIYEAR_LOSS_REPORT.md) |
| 15 | 日期与股票训练权重 | [方案](STOCK_WEIGHT_PLAN.md)、[结果](../reports/v2/stock_weights_200k/STOCK_WEIGHT_REPORT.md) |
| 16 | OLS因子组消融 | [方案](FACTOR_ABLATION_PLAN.md)、[结果](../reports/v2/factor_ablation_ols_200k/FACTOR_ABLATION_REPORT.md) |
| 17 | 稳健损失因子组消融 | [方案](ROBUST_ABLATION_PLAN.md)、[结果](../reports/v2/robust_ablation_200k/ROBUST_ABLATION_REPORT.md) |
| 18 | 精简候选联合删除C/M | [方案](COMPACT_CANDIDATE_PLAN.md)、[结果](../reports/v2/compact_cm_200k/ROBUST_ABLATION_REPORT.md) |
| 19 | 精简候选连续交易比较 | [结果](../reports/v2/compact_continuous/COMPACT_BACKTEST_REPORT.md) |
| 20 | 每折扩至50万训练样本 | [方案](EXPANDED_SAMPLE_PLAN.md)、[结果](../reports/v2/expanded_losses_500k/EXPANDED_SAMPLE_REPORT.md) |
| 21 | 50万模型连续回测 | [方案](EXPANDED_BACKTEST_PLAN.md)、[完整/精简比较](../reports/v2/expanded_continuous_500k/EXPANDED_BACKTEST_REPORT.md)、[20万/50万比较](../reports/v2/expanded_continuous_500k/SAMPLE_SIZE_COMPARISON.md) |
| 22 | 年度、成本、股票贡献归因 | [结果](../reports/v2/expanded_attribution/ATTRIBUTION_REPORT.md)、[结论](EXPANDED_ATTRIBUTION_FINDINGS.md) |
| 23 | 资金规模与冲击压力 | [结果](../reports/v2/capacity_stress_500k/CAPACITY_STRESS_REPORT.md) |
| 24 | 严格涨跌停及买入数量约束 | [方案及规则依据](EXECUTION_HARDENING_PLAN.md)、[结果](../reports/v2/execution_hardening_500k/EXECUTION_HARDENING_REPORT.md) |
| 25 | 严格成交与双倍冲击联合测试 | [结果](../reports/v2/joint_execution_stress_500k/JOINT_STRESS_REPORT.md) |

同模型比较集中见 [V1/V2 模型对比报告](V1_V2_MODEL_COMPARISON.md)。

## Elastic Net 后续实验

| 顺序 | 阶段 | 主要文档 |
|---|---|---|
| 26 | Elastic每折50万样本训练 | [结果](../reports/v2/elastic_500k/ELASTIC_REPORT.md) |
| 27 | V1/V2 Elastic统一评估及回测 | [比较](../reports/v2/elastic_versions_500k/COMPARISON.md) |
| 28 | 继承V1的19因子名称对照 | [结果](../reports/v2/elastic_19_control/REPORT.md) |
| 29 | 日期权重与正则化交叉实验 | [结果](../reports/v2/elastic_19_factorial/REPORT.md) |
| 30 | 训练样本成员和目标门控审计 | [审计](../reports/v2/elastic_sample_audit/REPORT.md) |
| 31 | 共同池抽样与完整池日期计数比较 | [结果](../reports/v2/elastic_common_weights/REPORT.md)、[结论与下一步](ELASTIC_COMMON_WEIGHTS_FINDINGS.md)、[代码](run_elastic_common_weights.py) |
| 32 | 训练样本因子完整度与未来成交筛选的四组对照 | [结果及逐年误差](../reports/v2/elastic_gate_ablation/REPORT.md)、[代码](run_elastic_gate_ablation.py) |
| 33 | 因子完整度筛选的多种子配对复验 | [逐种子结果与收益差](../reports/v2/elastic_gate_seeds/REPORT.md)、[代码](run_elastic_gate_seeds.py) |
| 34 | 333 因子 Elastic Net 完整度筛选的多种子复验 | [逐种子结果与收益差](../reports/v2/elastic_compact_gate_seeds/REPORT.md)、[代码](run_elastic_compact_gate_seeds.py) |
| 35 | 333 因子完整度筛选收益归因 | [逐年成本、换手及个股贡献](../reports/v2/elastic_compact_gate_attribution/REPORT.md)、[代码](attribute_compact_gate.py) |
| 36 | 2022 年逐日板块与风格归因 | [持仓占比与收益拆解](../reports/v2/elastic_board_style_2022/REPORT.md)、[代码](attribute_compact_board_style.py) |
| 37 | 七年逐日板块与风格归因 | [跨年贡献与风格变化](../reports/v2/elastic_board_style_multiyear/REPORT.md)、[代码](attribute_compact_board_years.py) |
| 38 | 固定初始板块预算的配对回测 | [板块账户与合并收益](../reports/v2/elastic_board_accounts/REPORT.md)、[代码](backtest_compact_board_accounts.py) |

## 文件位置与阅读顺序

设计和方案主要放在 v2 文件夹，运行结果主要放在 quant_data/processed/v2 的各实验目录。部分结果目录中的报告沿用了生成器名称，例如精简联合候选的 ROBUST_ABLATION_REPORT.md，阅读时应以所在实验目录和正文为准。

可先读第 34–38 阶段了解最新的 Elastic 筛选复验、收益归因和独立板块预算结果，再读第 26–33 阶段了解模型对比及样本审计，再回看第2、5、14至18阶段理解因子设计、数据限制和候选形成过程。各阶段文档完整程度不同，有些阶段将方案和结论合并在结果报告中。

这些后续比较使用已查看的历史开发期。报告中的收益需结合资金规模、成本、成交假设和训练样本量一起理解。真正未来验证仍需要冻结候选后取得的新数据。
