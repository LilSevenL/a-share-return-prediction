# 最终模型重新训练总结

生成日期：2026-09-25

## 结论

三个模型已经使用2011–2024全部非Test样本重新训练完成：

- **主模型：MLP**；
- **基础Baseline：19因子Elastic Net线性模型**；
- **强对照模型：LightGBM**。

脚本只显式读取 `final_split=training` 与 `final_split=validation`，没有读取 `final_split=test`。最终训练池由9,102,082行扩大至10,309,490行，增加1,207,408行（13.27%）。三个模型文件均已重新加载并通过2,048行非Test样本推理检查，预测全部为有限值，保存后的SHA-256与清单一致。

## 重新训练前后的数据对比

| 指标 | 重训前：2011–2023 | 重训后：2011–2024 | 变化 |
|---|---:|---:|---:|
| 样本行数 | 9,102,082 | 10,309,490 | +1,207,408（+13.27%） |
| 股票数 | 5,051 | 5,160 | +109（+2.16%） |
| 信号交易日 | 3,154 | 3,391 | +237（+7.51%） |
| 信号日期 | 2011-01-04至2023-12-22 | 2011-01-04至2024-12-24 | 增加2024年 |
| 最晚退出日期 | 2023-12-29 | 2024-12-31 | 增加2024年 |
| Target均值 | 0.0952% | 0.0877% | -0.0074个百分点 |
| Target标准差 | 6.5471% | 6.7973% | +3.82% |
| Target 1%分位数 | -17.4946% | -18.2220% | 左尾扩大 |
| Target中位数 | -0.0115% | -0.0388% | 略微下降 |
| Target 99%分位数 | 18.9424% | 20.0836% | 右尾扩大 |
| 平均有效因子数 | 48.448 | 48.545 | +0.097 |
| 最少/最多有效因子 | 30 / 50 | 30 / 50 | 不变 |

加入2024后，Target均值略降，但波动率和双侧尾部都扩大，说明新增年份提高了训练数据的市场状态多样性，而不是简单增加同分布样本。因子完整度略有提升，全部50个因子在最终预处理中仍然有效。

## MLP主模型

冻结配置：

- 隐藏层：`128 → 64 → 32`；
- 激活函数：ReLU；
- Dropout：`0.10 / 0.075 / 0.05`；
- 学习率：0.0005；
- Batch size：4096；
- Weight decay：`1e-5`；
- 随机种子：42；
- CPU训练；
- 从10,309,490行合格训练池中固定随机抽取300,000行；
- 8个Validation折选中epoch为 `10、11、14、16、17、17、21、23`，中位数16.5按保守规则向下取整为 **16 epochs**。

最终训练样本包含2011–2024所有年份，其中2024抽取35,191行。最终训练MSE为0.004712。该MSE是训练损失，不能当作样本外表现。

模型文件：`quant_data/processed/final_models_2011_2024/mlp_primary.pt`。

## 线性Baseline

线性模型继续使用Validation阶段冻结的19个稳定因子、每日权重指数0.75、`l1_ratio=1e-7`、`alpha_ratio=0.2`。最终19个系数均非零，优化2次迭代收敛。

模型文件：`quant_data/processed/final_models_2011_2024/linear_elastic_net_19.npz`。

## LightGBM强对照

冻结配置：

- `num_leaves=1023`；
- `min_data_in_leaf=6000`；
- `learning_rate=0.0075`；
- `feature_fraction=0.3`；
- `bagging_fraction=1.0`；
- `lambda_l1=0`、`lambda_l2=1`；
- 8折最佳迭代为 `228、193、131、128、259、348、266、444`；
- 最终固定使用其中位数四舍五入后的 **244 rounds**，不使用Test进行Early Stopping。

最终训练L2为0.004399，同样只是训练期损失，不能视为样本外成绩。

模型文件：`quant_data/processed/final_models_2011_2024/lightgbm_challenger.txt`。

## 为什么没有“重训后的Validation指标”

2024原本是最后的Validation确认期，现在已经被纳入最终训练。重训后的模型已经没有剩余的非Test时间段可用于诚实评估。因此：

- 可以比较重训前后的数据规模、分布和模型配置；
- 不能把训练损失与之前的Validation Rank IC、Sharpe直接比较；
- 不能再次用2024评估最终模型，因为模型已经看过2024；
- 下一次样本外评估必须是一次性的2025–2026永久Test。

这不是缺少步骤，而是严格时间序列实验必须遵守的边界。

## 输出文件

- `retrain_final_models.py`：三个冻结模型的一键最终重训脚本。
- `quant_data/processed/final_models_2011_2024/data_before_after.json`：重训前后数据统计与逐项变化。
- `quant_data/processed/final_models_2011_2024/final_training_manifest.json`：模型角色、配置、训练范围、哈希与耗时。
- `quant_data/processed/final_models_2011_2024/preprocessing.json`：2011–2024训练数据拟合的预处理参数。
- `quant_data/processed/final_models_2011_2024/mlp_training_history.json`：MLP 16轮训练损失。
- `quant_data/processed/final_models_2011_2024/mlp_primary.pt`：最终MLP主模型。
- `quant_data/processed/final_models_2011_2024/linear_elastic_net_19.npz`：最终线性Baseline。
- `quant_data/processed/final_models_2011_2024/lightgbm_challenger.txt`：最终LightGBM强对照。

复现命令：

```powershell
& "E:\Anaconda\envs\quant_ml\python.exe" retrain_final_models.py --threads 8
```

2025–2026永久Test已经按冻结配置完成，最终结果见 `permanent_test_summary.md`。MLP主模型保持不变；后续只能进行结果解释和写作，不能再依据Test调参或切换模型。
