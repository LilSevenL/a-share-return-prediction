# V2 线性模型损失函数调整

## 决定与当前状态

保留MSE基准，新增精确MAE和Huber。Huber是优先验证的候选，不预先指定为赢家。三种损失均支持同一套样本权重和L2正则，截距不惩罚。

已经实现 `linear_losses.py` 的损失、梯度与通用线性拟合API，适用于任意因子列数；通过独立数值测试。只运行了合成数据测试，没有市场数据训练，没有读取Test。新设计的500因子表及对应训练数据管线尚未实现完整，因此还不能声称三种损失已接入完整500因子实验。

V1、旧基准结果及其脚本保持原状。MAE/Huber拟合仍然是线性回归，但严格说不再是OLS；有L2时，MSE版本也应称为Ridge或加权Ridge。

## 统一目标

设X已经使用Training拟合的预处理进行变换。每折只用Training目标计算固定尺度：

`s = 1.4826 * median(abs(y_train - median(y_train)))`。

若MAD恰好为0则使用Training标准差；若目标恒定则s=1。这是**目标收益的稳健离散尺度，不是拟合残差sigma的估计**。同一折所有候选共享该尺度，Validation和Test不得参与计算。

模型先预测缩放后的目标：`z_hat = b + X beta`，原目标预测为`s*z_hat`；残差`u=z_hat-y/s`。

统一目标：

`J = sum(w_i * rho(u_i)) / sum(w_i) + lambda/2 * sum(beta_j^2)`。

- MSE训练项：`rho(u)=0.5*u^2`。0.5使其小误差曲率与标准Huber一致；报告的MSE仍是通常的均方误差。不能直接照搬旧脚本的正则化数值。
- MAE训练项：`rho(u)=abs(u)`，不进行平滑近似。无约束理想点预测中，平方损失对应条件均值，绝对损失对应条件中位数；有限线性模型与正则化下只是对应的近似目标。
- Huber训练项：当`abs(u)<=delta`时为`0.5*u^2`，否则为`delta*(abs(u)-0.5*delta)`。

MSE的大误差梯度随误差增长；MAE的非零误差梯度幅度恒定；Huber小误差梯度线性，大误差梯度封顶。**Huber不是只惩罚大误差，也不会忽略大误差；相对MSE，它减轻了大误差的主导作用。**

首轮Huber阈值为`delta=1.35`（缩放单位），只是研究起点。假设某折s=0.03，则实际收益残差阈值为0.0405，即约4.05个百分点的对数收益。不能对原始小数收益直接照抄delta=1并认为已经产生明显稳健作用。

当前方法固定Training目标尺度，不联合优化残差尺度，因此与scikit-learn HuberRegressor的实现并不等价。

## 为什么量化场景不能直接宣布Huber最好

1. 收益分布中的极端值可能是数据错误，也可能是真实行情及选股收益来源。先做数据质量检查，不能把所有大残差都当作坏数据。
2. MSE更重视收益幅度，可能保留重要尾部信号，也可能被少量噪声主导；MAE更稳健，但可能弱化重要的大涨大跌信息。
3. Huber提供折中，但并不直接优化Rank IC、Top50收益或扣成本PnL。训练损失下降不能替代统一预测与交易评价。
4. 高波动股票本来就更容易落入Huber线性区。后续加入股票训练权重时，要检查它与Huber降权的共同效果，不能同时改变二者再把结果全部归因于某一个。
5. Huber主要降低目标残差异常的影响，不能自动解决X里的高杠杆异常、未来数据泄漏或共线性。

不额外对Training收益做缩尾，避免把“修改目标”与“换损失”混在一起。常规因子预处理仍只在Training内拟合。

## 初始实验安排（未执行）

配置保存在 `loss_experiment_plan.json`。

- 第一步固定同一因子集合、时间切分、样本和日权重指数0.75，股票倍数先设1。同一诊断轮固定L2=0.01，比较MSE、MAE、Huber(1.35)。这个L2只是一项初始实验设置，不是最优值。
- 第二步对每种损失给予相同、有限的L2调参预算；例如0.001、0.01、0.1。即使目标缩放一致，不同损失的整体幅度仍不同，单个相同lambda下的结果不应被当作损失函数最终优劣。
- Huber可进一步在开发期比较delta=0.5、1、1.35、2；不自动展开大规模网格。阈值越小越偏向绝对误差，但其整体幅度也变化，需同时审视正则化比例。
- 保持所有模型的评估口径相同：普通Rank IC、固定同一权重定义的加权Rank IC、IC、RMSE、MAE，以及统一交易规则和成本模拟。
- 不跨不同损失比较原始训练loss数值来选赢家。记录每折阈值、Huber线性区样本/权重占比、求解收敛状态和实际训练配置。
- 2017–2023用于开发；2024已看过，不能再称为新未见holdout；已评估永久Test不参与选择。

排序损失、分位数损失和交易成本感知目标可以另设后续实验，但目前先把导师建议的三种回归损失比较清楚。

## 实现与调用

```python
from v2.linear_losses import LossConfig, fit_linear_loss, predict_linear_loss

# X_train/X_valid必须由同一份Training拟合的预处理转换。
# sample_weight只能使用既定的日期/信号时可用股票信息。
model = fit_linear_loss(
    X_train, y_train, sample_weight=train_weights,
    config=LossConfig(loss="huber", delta=1.35, l2=0.01),
)
prediction = predict_linear_loss(model, X_valid)
```

将loss改为`mse`或`mae`即可切换。输出系数已换回原收益目标单位（X仍是预处理后的尺度），包括配置、训练目标尺度、实际Huber阈值和诊断信息，可JSON保存。

MSE用加权正规方程；Huber用解析梯度L-BFGS-B；MAE用ADMM求精确绝对误差目标，不把L-BFGS应用于不可微MAE后假装一定收敛。矩阵运算分块，避免额外复制完整双精度设计矩阵；仍需提供X矩阵/内存映射。尚未完成百万级500列的性能或收敛压测，尤其MAE可能需要较多迭代。

默认不收敛会报错，不能静默把该模型纳入比较。`allow_unconverged=True`仅供诊断，结果仍标明未收敛。

## 数值验证

测试入口：`test_linear_losses.py`，使用合成数据，不是市场训练。检查：

1. Huber函数值与SciPy标准实现一致，阈值边界梯度正确。
2. 三种目标的梯度/非零点次梯度与数值差分相符，截距不惩罚。
3. 无正则MSE与独立增广矩阵加权最小二乘一致。
4. 无正则MAE与独立线性规划的最优目标一致。
5. MAE+L2与独立epigraph二次规划目标一致。
6. 收益单位乘100、全部样本权重乘常数时，预测按应有比例变化。
7. 极端值位置实验、常数目标和退化特征处理。
8. 无效输入及未收敛结果不会被当作有效模型。

## 参考

- [SciPy标准Huber定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.huber.html)。
- [scikit-learn线性模型文档：稳健回归与分位数回归](https://scikit-learn.org/stable/modules/linear_model.html)。
- [HuberRegressor文档：尺度、正则化和残差异常](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)。
