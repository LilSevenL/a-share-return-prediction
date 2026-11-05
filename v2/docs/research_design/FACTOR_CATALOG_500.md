# 按研究方向重新设计的500个候选因子

**状态：审计修复版，500候选中51项隔离；历史股票池补采未完成。旧450个filter组合不是本清单。**

本清单包含原50个因子；新增450个分散到十个方向。数量是本项目初始研究预算，不是重要性实证结论。
符号、缺失值、复权、公告时点和来源规则见同目录README.md。

| 方向 | 数量 | 优先级 | 分配理由 |
|---|---:|---|---|
| M 动量、反转与收益结构 | 80 | 最高 | 与5日预测最直接相关，覆盖趋势持续与短期反转两种相反假设。 |
| R 波动、尾部风险与回撤 | 65 | 高 | 区分相同收益背后的风险状态，并服务交易风险评估。 |
| L 流动性、换手与交易活跃度 | 60 | 高 | 同时影响信号含义、可交易性和成本；日线数据可实现。 |
| P 日线价格形态与价量关系 | 55 | 高 | 补充开高低收和价量配合，避免只看收盘收益。 |
| F 资金流与订单规模结构 | 50 | 中高 | 检验不同订单规模的买卖压力；数据口径和覆盖需审计。 |
| X 市场相对表现与特质收益 | 45 | 中高 | 区分市场共同波动与个股特质信息；不依赖历史行业成分。 |
| Q 财务质量、成长与公告变化 | 45 | 中 | 提供企业状态；慢频信息对5日任务的直接作用需验证。 |
| V 估值、规模与股息 | 40 | 中 | 提供横截面定价和风格背景，避免短周期任务被慢变量占满。 |
| C 条件筛选收益 | 35 | 定向研究 | 保留导师的关键假设，用少量窗口测试，不让参数组合占满库。 |
| I 跨方向交互 | 25 | 后置 | 向线性模型提供少量有经济解释的非线性条件；先验证主效应。 |
| 合计 | 500 | | |

## M 动量、反转与收益结构：80个

与5日预测最直接相关，覆盖趋势持续与短期反转两种相反假设。

内部维度：原有基础因子 14个；多尺度收益 4个；跳过近期的动量 3个；趋势与收益结构 56个；近期冲击位置 3个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F001 | `logret_1d` | 复权收盘价的1个连续市场交易日对数收益。 | `复权收盘价的1个连续市场交易日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F002 | `logret_2d` | 复权收盘价的2个连续市场交易日对数收益。 | `复权收盘价的2个连续市场交易日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F003 | `logret_3d` | 复权收盘价的3个连续市场交易日对数收益。 | `复权收盘价的3个连续市场交易日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F004 | `logret_5d` | 复权收盘价的5个连续市场交易日对数收益。 | `复权收盘价的5个连续市场交易日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F005 | `logret_10d` | 复权收盘价的10个连续市场交易日对数收益。 | `复权收盘价的10个连续市场交易日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F006 | `logret_20d` | 复权收盘价的20个连续市场交易日对数收益。 | `复权收盘价的20个连续市场交易日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F007 | `logret_60d` | 复权收盘价的60个连续市场交易日对数收益。 | `复权收盘价的60个连续市场交易日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F008 | `close_ma5_gap` | 复权收盘价/过去5日复权收盘均价−1，要求完整连续窗口。 | `复权收盘价/过去5日复权收盘均价−1，要求完整连续窗口。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F009 | `close_ma20_gap` | 复权收盘价/过去20日复权收盘均价−1，要求完整连续窗口。 | `复权收盘价/过去20日复权收盘均价−1，要求完整连续窗口。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F010 | `close_ma60_gap` | 复权收盘价/过去60日复权收盘均价−1，要求完整连续窗口。 | `复权收盘价/过去60日复权收盘均价−1，要求完整连续窗口。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F011 | `ma5_ma20_gap` | 过去5日复权收盘均价/过去20日均价−1。 | `过去5日复权收盘均价/过去20日均价−1。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F012 | `ma20_ma60_gap` | 过去20日复权收盘均价/过去60日均价−1。 | `过去20日复权收盘均价/过去60日均价−1。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F013 | `intraday_logret` | ln(当日复权收盘价/当日复权开盘价)。 | `ln(当日复权收盘价/当日复权开盘价)。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F014 | `overnight_logret` | ln(当日复权开盘价/上一市场交易日复权收盘价)。 | `ln(当日复权开盘价/上一市场交易日复权收盘价)。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F015 | `logret_15d` | 累计复权对数收益（15日） | `SUM(r,15)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F016 | `logret_40d` | 累计复权对数收益（40日） | `SUM(r,40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F017 | `logret_120d` | 累计复权对数收益（120日） | `SUM(r,120)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F018 | `logret_250d` | 累计复权对数收益（250日） | `SUM(r,250)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F019 | `skip5_momentum_20d` | 跳过最近5日的动量（20日） | `log(C[t-5]/C[t-20])` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F020 | `skip5_momentum_60d` | 跳过最近5日的动量（60日） | `log(C[t-5]/C[t-60])` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F021 | `skip5_momentum_120d` | 跳过最近5日的动量（120日） | `log(C[t-5]/C[t-120])` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F022 | `trend_slope_5d` | 趋势斜率（5日） | `OLS_SLOPE(log(C),time,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F023 | `trend_slope_10d` | 趋势斜率（10日） | `OLS_SLOPE(log(C),time,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F024 | `trend_slope_20d` | 趋势斜率（20日） | `OLS_SLOPE(log(C),time,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F025 | `trend_slope_60d` | 趋势斜率（60日） | `OLS_SLOPE(log(C),time,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F026 | `trend_r2_5d` | 趋势拟合优度（5日） | `OLS_R2(log(C),time,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F027 | `trend_r2_10d` | 趋势拟合优度（10日） | `OLS_R2(log(C),time,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F028 | `trend_r2_20d` | 趋势拟合优度（20日） | `OLS_R2(log(C),time,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F029 | `trend_r2_60d` | 趋势拟合优度（60日） | `OLS_R2(log(C),time,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F030 | `path_efficiency_5d` | 跨日路径效率（5日） | `abs(SUM(r,5))/SUM(abs(r),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F031 | `path_efficiency_10d` | 跨日路径效率（10日） | `abs(SUM(r,10))/SUM(abs(r),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F032 | `path_efficiency_20d` | 跨日路径效率（20日） | `abs(SUM(r,20))/SUM(abs(r),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F033 | `path_efficiency_60d` | 跨日路径效率（60日） | `abs(SUM(r,60))/SUM(abs(r),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F034 | `return_acceleration_5d` | 前后等长窗口收益加速度（5日） | `MEAN(r,5)-LAG(MEAN(r,5),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F035 | `return_acceleration_10d` | 前后等长窗口收益加速度（10日） | `MEAN(r,10)-LAG(MEAN(r,10),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F036 | `return_acceleration_20d` | 前后等长窗口收益加速度（20日） | `MEAN(r,20)-LAG(MEAN(r,20),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F037 | `return_acceleration_60d` | 前后等长窗口收益加速度（60日） | `MEAN(r,60)-LAG(MEAN(r,60),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F038 | `up_day_share_5d` | 上涨交易日比例（5日） | `MEAN(I(r>0),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F039 | `up_day_share_10d` | 上涨交易日比例（10日） | `MEAN(I(r>0),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F040 | `up_day_share_20d` | 上涨交易日比例（20日） | `MEAN(I(r>0),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F041 | `up_day_share_60d` | 上涨交易日比例（60日） | `MEAN(I(r>0),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F042 | `longest_up_streak_5d` | 窗口内最长连涨（5日） | `MAX_STREAK(r>0,5)/5` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F043 | `longest_up_streak_10d` | 窗口内最长连涨（10日） | `MAX_STREAK(r>0,10)/10` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F044 | `longest_up_streak_20d` | 窗口内最长连涨（20日） | `MAX_STREAK(r>0,20)/20` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F045 | `longest_up_streak_60d` | 窗口内最长连涨（60日） | `MAX_STREAK(r>0,60)/60` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F046 | `longest_down_streak_5d` | 窗口内最长连跌（5日） | `MAX_STREAK(r<0,5)/5` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F047 | `longest_down_streak_10d` | 窗口内最长连跌（10日） | `MAX_STREAK(r<0,10)/10` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F048 | `longest_down_streak_20d` | 窗口内最长连跌（20日） | `MAX_STREAK(r<0,20)/20` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F049 | `longest_down_streak_60d` | 窗口内最长连跌（60日） | `MAX_STREAK(r<0,60)/60` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F050 | `return_autocorr_5d` | 单日收益一阶自相关（5日） | `CORR(r,LAG(r,1),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F051 | `return_autocorr_10d` | 单日收益一阶自相关（10日） | `CORR(r,LAG(r,1),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F052 | `return_autocorr_20d` | 单日收益一阶自相关（20日） | `CORR(r,LAG(r,1),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F053 | `return_autocorr_60d` | 单日收益一阶自相关（60日） | `CORR(r,LAG(r,1),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F054 | `overnight_intraday_spread_5d` | 隔夜与日内收益差（5日） | `SUM(ro-ri,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F055 | `overnight_intraday_spread_10d` | 隔夜与日内收益差（10日） | `SUM(ro-ri,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F056 | `overnight_intraday_spread_20d` | 隔夜与日内收益差（20日） | `SUM(ro-ri,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F057 | `overnight_intraday_spread_60d` | 隔夜与日内收益差（60日） | `SUM(ro-ri,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F058 | `overnight_intraday_cov_5d` | 隔夜与日内协方差（5日） | `COV(ro,ri,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F059 | `overnight_intraday_cov_10d` | 隔夜与日内协方差（10日） | `COV(ro,ri,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F060 | `overnight_intraday_cov_20d` | 隔夜与日内协方差（20日） | `COV(ro,ri,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F061 | `overnight_intraday_cov_60d` | 隔夜与日内协方差（60日） | `COV(ro,ri,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F062 | `overnight_reversal_share_5d` | 隔夜冲击日内反转频率（5日） | `MEAN(I(ro*ri<0),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F063 | `overnight_reversal_share_10d` | 隔夜冲击日内反转频率（10日） | `MEAN(I(ro*ri<0),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F064 | `overnight_reversal_share_20d` | 隔夜冲击日内反转频率（20日） | `MEAN(I(ro*ri<0),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F065 | `overnight_reversal_share_60d` | 隔夜冲击日内反转频率（60日） | `MEAN(I(ro*ri<0),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F066 | `intraday_abs_share_5d` | 日内绝对收益占比（5日） | `SUM(abs(ri),5)/SUM(abs(ri)+abs(ro),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F067 | `intraday_abs_share_10d` | 日内绝对收益占比（10日） | `SUM(abs(ri),10)/SUM(abs(ri)+abs(ro),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F068 | `intraday_abs_share_20d` | 日内绝对收益占比（20日） | `SUM(abs(ri),20)/SUM(abs(ri)+abs(ro),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F069 | `intraday_abs_share_60d` | 日内绝对收益占比（60日） | `SUM(abs(ri),60)/SUM(abs(ri)+abs(ro),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F070 | `slope_change_5d` | 前后窗口趋势斜率变化（5日） | `OLS_SLOPE(log(C),time,5)-LAG(OLS_SLOPE(log(C),time,5),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F071 | `slope_change_10d` | 前后窗口趋势斜率变化（10日） | `OLS_SLOPE(log(C),time,10)-LAG(OLS_SLOPE(log(C),time,10),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F072 | `slope_change_20d` | 前后窗口趋势斜率变化（20日） | `OLS_SLOPE(log(C),time,20)-LAG(OLS_SLOPE(log(C),time,20),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F073 | `slope_change_60d` | 前后窗口趋势斜率变化（60日） | `OLS_SLOPE(log(C),time,60)-LAG(OLS_SLOPE(log(C),time,60),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F074 | `direction_switch_share_5d` | 相邻收益换向频率（5日） | `MEAN(I(r*LAG(r,1)<0),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F075 | `direction_switch_share_10d` | 相邻收益换向频率（10日） | `MEAN(I(r*LAG(r,1)<0),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F076 | `direction_switch_share_20d` | 相邻收益换向频率（20日） | `MEAN(I(r*LAG(r,1)<0),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F077 | `direction_switch_share_60d` | 相邻收益换向频率（60日） | `MEAN(I(r*LAG(r,1)<0),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F078 | `return_ts_rank_20d` | 当日收益历史分位（20日） | `TSRANK(r,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F079 | `return_ts_rank_60d` | 当日收益历史分位（60日） | `TSRANK(r,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F080 | `return_ts_rank_120d` | 当日收益历史分位（120日） | `TSRANK(r,120)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |

## R 波动、尾部风险与回撤：65个

区分相同收益背后的风险状态，并服务交易风险评估。

内部维度：原有基础因子 7个；波动尺度 2个；风险形状与极端事件 56个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F081 | `volatility_5d` | 过去5日单日对数收益样本标准差，要求5个有效收益。 | `过去5日单日对数收益样本标准差，要求5个有效收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F082 | `volatility_10d` | 过去10日单日对数收益样本标准差，要求10个有效收益。 | `过去10日单日对数收益样本标准差，要求10个有效收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F083 | `volatility_20d` | 过去20日单日对数收益样本标准差，要求20个有效收益。 | `过去20日单日对数收益样本标准差，要求20个有效收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F084 | `volatility_60d` | 过去60日单日对数收益样本标准差，要求60个有效收益。 | `过去60日单日对数收益样本标准差，要求60个有效收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F085 | `downside_volatility_20d` | 过去20日收益负值平方的均值开根号，正收益记0。 | `过去20日收益负值平方的均值开根号，正收益记0。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F086 | `return_skew_20d` | 过去20日单日对数收益偏度。 | `过去20日单日对数收益偏度。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F087 | `worst_logret_20d` | 过去20日最小单日对数收益。 | `过去20日最小单日对数收益。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F088 | `volatility_40d` | 收益标准差（40日） | `STD(r,40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F089 | `volatility_120d` | 收益标准差（120日） | `STD(r,120)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F090 | `upside_rms_5d` | 上行均方根（5日） | `sqrt(MEAN(max(r,0)^2,5))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F091 | `upside_rms_10d` | 上行均方根（10日） | `sqrt(MEAN(max(r,0)^2,10))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F092 | `upside_rms_20d` | 上行均方根（20日） | `sqrt(MEAN(max(r,0)^2,20))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F093 | `upside_rms_60d` | 上行均方根（60日） | `sqrt(MEAN(max(r,0)^2,60))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F094 | `downside_rms_5d` | 下行均方根（5日） | `sqrt(MEAN(min(r,0)^2,5))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F095 | `downside_rms_10d` | 下行均方根（10日） | `sqrt(MEAN(min(r,0)^2,10))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F096 | `downside_rms_60d` | 下行均方根（60日） | `sqrt(MEAN(min(r,0)^2,60))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F097 | `downside_rms_120d` | 下行均方根（120日） | `sqrt(MEAN(min(r,0)^2,120))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F098 | `return_kurtosis_5d` | 收益超额峰度（5日） | `KURT_EXCESS(r,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F099 | `return_kurtosis_10d` | 收益超额峰度（10日） | `KURT_EXCESS(r,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F100 | `return_kurtosis_20d` | 收益超额峰度（20日） | `KURT_EXCESS(r,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F101 | `return_kurtosis_60d` | 收益超额峰度（60日） | `KURT_EXCESS(r,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F102 | `return_mad_5d` | 收益中位绝对偏差（5日） | `MEDIAN(abs(r-MEDIAN_WINDOW(r)),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F103 | `return_mad_10d` | 收益中位绝对偏差（10日） | `MEDIAN(abs(r-MEDIAN_WINDOW(r)),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F104 | `return_mad_20d` | 收益中位绝对偏差（20日） | `MEDIAN(abs(r-MEDIAN_WINDOW(r)),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F105 | `return_mad_60d` | 收益中位绝对偏差（60日） | `MEDIAN(abs(r-MEDIAN_WINDOW(r)),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F106 | `return_q05_5d` | 收益5%分位（5日） | `QUANTILE(r,0.05,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F107 | `return_q05_10d` | 收益5%分位（10日） | `QUANTILE(r,0.05,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F108 | `return_q05_20d` | 收益5%分位（20日） | `QUANTILE(r,0.05,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F109 | `return_q05_60d` | 收益5%分位（60日） | `QUANTILE(r,0.05,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F110 | `return_lower_quartile_mean_5d` | 最差四分之一收益均值（5日） | `MEAN_LOWEST(r,ceil(5/4),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F111 | `return_lower_quartile_mean_10d` | 最差四分之一收益均值（10日） | `MEAN_LOWEST(r,ceil(10/4),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F112 | `return_lower_quartile_mean_20d` | 最差四分之一收益均值（20日） | `MEAN_LOWEST(r,ceil(20/4),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F113 | `return_lower_quartile_mean_60d` | 最差四分之一收益均值（60日） | `MEAN_LOWEST(r,ceil(60/4),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F114 | `best_return_5d` | 最大单日收益（5日） | `MAX(r,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F115 | `best_return_10d` | 最大单日收益（10日） | `MAX(r,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F116 | `best_return_20d` | 最大单日收益（20日） | `MAX(r,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F117 | `best_return_60d` | 最大单日收益（60日） | `MAX(r,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F118 | `parkinson_vol_5d` | 高低价区间波动估计（5日） | `sqrt(MEAN(log(H/L)^2,5)/(4*log(2)))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F119 | `parkinson_vol_10d` | 高低价区间波动估计（10日） | `sqrt(MEAN(log(H/L)^2,10)/(4*log(2)))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F120 | `parkinson_vol_20d` | 高低价区间波动估计（20日） | `sqrt(MEAN(log(H/L)^2,20)/(4*log(2)))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F121 | `parkinson_vol_60d` | 高低价区间波动估计（60日） | `sqrt(MEAN(log(H/L)^2,60)/(4*log(2)))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F122 | `garman_klass_vol_5d` | OHLC波动估计（5日） | `sqrt(max(MEAN(0.5*log(H/L)^2-(2*log(2)-1)*ri^2,5),0))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F123 | `garman_klass_vol_10d` | OHLC波动估计（10日） | `sqrt(max(MEAN(0.5*log(H/L)^2-(2*log(2)-1)*ri^2,10),0))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F124 | `garman_klass_vol_20d` | OHLC波动估计（20日） | `sqrt(max(MEAN(0.5*log(H/L)^2-(2*log(2)-1)*ri^2,20),0))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F125 | `garman_klass_vol_60d` | OHLC波动估计（60日） | `sqrt(max(MEAN(0.5*log(H/L)^2-(2*log(2)-1)*ri^2,60),0))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F126 | `maximum_drawdown_5d` | 窗口最大回撤（5日） | `MAX_PEAK_TO_LATER_TROUGH_DRAWDOWN(C,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F127 | `maximum_drawdown_10d` | 窗口最大回撤（10日） | `MAX_PEAK_TO_LATER_TROUGH_DRAWDOWN(C,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F128 | `maximum_drawdown_20d` | 窗口最大回撤（20日） | `MAX_PEAK_TO_LATER_TROUGH_DRAWDOWN(C,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F129 | `maximum_drawdown_60d` | 窗口最大回撤（60日） | `MAX_PEAK_TO_LATER_TROUGH_DRAWDOWN(C,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F130 | `ulcer_index_5d` | 窗口内回撤均方根（5日） | `sqrt(MEAN_WITHIN_WINDOW((C/RUNNING_PEAK(C)-1)^2,5))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F131 | `ulcer_index_10d` | 窗口内回撤均方根（10日） | `sqrt(MEAN_WITHIN_WINDOW((C/RUNNING_PEAK(C)-1)^2,10))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F132 | `ulcer_index_20d` | 窗口内回撤均方根（20日） | `sqrt(MEAN_WITHIN_WINDOW((C/RUNNING_PEAK(C)-1)^2,20))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F133 | `ulcer_index_60d` | 窗口内回撤均方根（60日） | `sqrt(MEAN_WITHIN_WINDOW((C/RUNNING_PEAK(C)-1)^2,60))` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F134 | `volatility_of_volatility_5d` | 波动率变化程度（5日） | `STD(STD(r,5),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F135 | `volatility_of_volatility_10d` | 波动率变化程度（10日） | `STD(STD(r,5),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F136 | `volatility_of_volatility_20d` | 波动率变化程度（20日） | `STD(STD(r,5),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F137 | `volatility_of_volatility_60d` | 波动率变化程度（60日） | `STD(STD(r,5),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F138 | `large_move_share_5d` | 大幅波动日期比例（5日） | `MEAN(I(abs(r)>3*LAG(STD(r,20),1)),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F139 | `large_move_share_10d` | 大幅波动日期比例（10日） | `MEAN(I(abs(r)>3*LAG(STD(r,20),1)),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F140 | `large_move_share_20d` | 大幅波动日期比例（20日） | `MEAN(I(abs(r)>3*LAG(STD(r,20),1)),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F141 | `large_move_share_60d` | 大幅波动日期比例（60日） | `MEAN(I(abs(r)>3*LAG(STD(r,20),1)),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F142 | `overnight_q05_5d` | 隔夜收益左尾（5日） | `QUANTILE(ro,0.05,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F143 | `overnight_q05_10d` | 隔夜收益左尾（10日） | `QUANTILE(ro,0.05,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F144 | `overnight_q05_20d` | 隔夜收益左尾（20日） | `QUANTILE(ro,0.05,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F145 | `overnight_q05_60d` | 隔夜收益左尾（60日） | `QUANTILE(ro,0.05,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |

## L 流动性、换手与交易活跃度：60个

同时影响信号含义、可交易性和成本；日线数据可实现。

内部维度：原有基础因子 7个；当前活跃度 3个；流动性水平与变化 50个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F146 | `log_amount` | 当日正成交额的自然对数。 | `当日正成交额的自然对数。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F147 | `amount_ratio_5d` | 当日成交额/过去5日平均成交额−1。 | `当日成交额/过去5日平均成交额−1。` | v2/features/balanced_features.py；V1快照保留不变；V2审计修复：按连续市场交易日重算，停牌缺口使完整窗口失效 |
| F148 | `amount_ratio_20d` | 当日成交额/过去20日平均成交额−1。 | `当日成交额/过去20日平均成交额−1。` | v2/features/balanced_features.py；V1快照保留不变；V2审计修复：按连续市场交易日重算，停牌缺口使完整窗口失效 |
| F149 | `volume_ratio_20d` | 当日成交量/过去20日平均成交量−1。 | `当日成交量/过去20日平均成交量−1。` | v2/features/balanced_features.py；V1快照保留不变；V2审计修复：按连续市场交易日重算，停牌缺口使完整窗口失效 |
| F150 | `turnover_rate` | 当日换手率，沿用数据源原单位。 | `当日换手率，沿用数据源原单位。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F151 | `turnover_rate_f` | 当日自由流通股换手率，沿用数据源原单位。 | `当日自由流通股换手率，沿用数据源原单位。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F152 | `turnover_zscore_20d` | (当日换手率−过去20日均值)/过去20日样本标准差。 | `(当日换手率−过去20日均值)/过去20日样本标准差。` | v2/features/balanced_features.py；V1快照保留不变；V2审计修复：按连续市场交易日重算，停牌缺口使完整窗口失效 |
| F153 | `log_volume` | 成交量对数 | `log(volume)` | daily.vol；已实现；通过样本审计不等于全市场认证 |
| F154 | `source_volume_ratio` | 数据源量比 | `daily_basic.volume_ratio` | daily_basic.volume_ratio；已实现；通过样本审计不等于全市场认证 |
| F155 | `free_turnover_ratio` | 自由流通换手与总换手比 | `turnover_rate_f/turnover_rate` | daily_basic；已实现；通过样本审计不等于全市场认证 |
| F156 | `log_mean_amount_5d` | 平均成交额对数（5日） | `log(MEAN(amount_yuan,5))` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F157 | `log_mean_amount_10d` | 平均成交额对数（10日） | `log(MEAN(amount_yuan,10))` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F158 | `log_mean_amount_20d` | 平均成交额对数（20日） | `log(MEAN(amount_yuan,20))` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F159 | `log_mean_amount_60d` | 平均成交额对数（60日） | `log(MEAN(amount_yuan,60))` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F160 | `log_mean_amount_120d` | 平均成交额对数（120日） | `log(MEAN(amount_yuan,120))` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F161 | `amount_cv_5d` | 成交额变异系数（5日） | `STD(amount_yuan,5)/MEAN(amount_yuan,5)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F162 | `amount_cv_10d` | 成交额变异系数（10日） | `STD(amount_yuan,10)/MEAN(amount_yuan,10)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F163 | `amount_cv_20d` | 成交额变异系数（20日） | `STD(amount_yuan,20)/MEAN(amount_yuan,20)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F164 | `amount_cv_60d` | 成交额变异系数（60日） | `STD(amount_yuan,60)/MEAN(amount_yuan,60)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F165 | `amount_cv_120d` | 成交额变异系数（120日） | `STD(amount_yuan,120)/MEAN(amount_yuan,120)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F166 | `log_volume_std_5d` | 成交量对数波动（5日） | `STD(log(volume),5)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F167 | `log_volume_std_10d` | 成交量对数波动（10日） | `STD(log(volume),10)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F168 | `log_volume_std_20d` | 成交量对数波动（20日） | `STD(log(volume),20)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F169 | `log_volume_std_60d` | 成交量对数波动（60日） | `STD(log(volume),60)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F170 | `log_volume_std_120d` | 成交量对数波动（120日） | `STD(log(volume),120)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F171 | `amihud_proxy_5d` | Amihud非流动性代理（5日） | `MEAN(abs(r)/amount_yuan,5)*100000000` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F172 | `amihud_proxy_10d` | Amihud非流动性代理（10日） | `MEAN(abs(r)/amount_yuan,10)*100000000` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F173 | `amihud_proxy_20d` | Amihud非流动性代理（20日） | `MEAN(abs(r)/amount_yuan,20)*100000000` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F174 | `amihud_proxy_60d` | Amihud非流动性代理（60日） | `MEAN(abs(r)/amount_yuan,60)*100000000` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F175 | `amihud_proxy_120d` | Amihud非流动性代理（120日） | `MEAN(abs(r)/amount_yuan,120)*100000000` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F176 | `turnover_mean_5d` | 平均换手率（5日） | `MEAN(turnover_rate,5)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F177 | `turnover_mean_10d` | 平均换手率（10日） | `MEAN(turnover_rate,10)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F178 | `turnover_mean_20d` | 平均换手率（20日） | `MEAN(turnover_rate,20)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F179 | `turnover_mean_60d` | 平均换手率（60日） | `MEAN(turnover_rate,60)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F180 | `turnover_mean_120d` | 平均换手率（120日） | `MEAN(turnover_rate,120)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F181 | `turnover_std_5d` | 换手率波动（5日） | `STD(turnover_rate,5)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F182 | `turnover_std_10d` | 换手率波动（10日） | `STD(turnover_rate,10)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F183 | `turnover_std_20d` | 换手率波动（20日） | `STD(turnover_rate,20)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F184 | `turnover_std_60d` | 换手率波动（60日） | `STD(turnover_rate,60)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F185 | `turnover_std_120d` | 换手率波动（120日） | `STD(turnover_rate,120)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F186 | `turnover_surprise_5d` | 当日换手相对前期均值（5日） | `turnover_rate/LAG(MEAN(turnover_rate,5),1)-1` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F187 | `turnover_surprise_10d` | 当日换手相对前期均值（10日） | `turnover_rate/LAG(MEAN(turnover_rate,10),1)-1` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F188 | `turnover_surprise_20d` | 当日换手相对前期均值（20日） | `turnover_rate/LAG(MEAN(turnover_rate,20),1)-1` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F189 | `turnover_surprise_60d` | 当日换手相对前期均值（60日） | `turnover_rate/LAG(MEAN(turnover_rate,60),1)-1` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F190 | `turnover_surprise_120d` | 当日换手相对前期均值（120日） | `turnover_rate/LAG(MEAN(turnover_rate,120),1)-1` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F191 | `zero_return_share_5d` | 零收盘收益日期比例（5日） | `MEAN(I(raw_close=raw_pre_close),5)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F192 | `zero_return_share_10d` | 零收盘收益日期比例（10日） | `MEAN(I(raw_close=raw_pre_close),10)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F193 | `zero_return_share_20d` | 零收盘收益日期比例（20日） | `MEAN(I(raw_close=raw_pre_close),20)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F194 | `zero_return_share_60d` | 零收盘收益日期比例（60日） | `MEAN(I(raw_close=raw_pre_close),60)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F195 | `zero_return_share_120d` | 零收盘收益日期比例（120日） | `MEAN(I(raw_close=raw_pre_close),120)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F196 | `volume_hhi_5d` | 成交量时间集中度（5日） | `SUM(volume^2,5)/SUM(volume,5)^2` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F197 | `volume_hhi_10d` | 成交量时间集中度（10日） | `SUM(volume^2,10)/SUM(volume,10)^2` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F198 | `volume_hhi_20d` | 成交量时间集中度（20日） | `SUM(volume^2,20)/SUM(volume,20)^2` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F199 | `volume_hhi_60d` | 成交量时间集中度（60日） | `SUM(volume^2,60)/SUM(volume,60)^2` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F200 | `volume_hhi_120d` | 成交量时间集中度（120日） | `SUM(volume^2,120)/SUM(volume,120)^2` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F201 | `amount_normalized_slope_5d` | 成交额趋势（5日） | `OLS_SLOPE(amount_yuan,time,5)/MEAN(amount_yuan,5)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F202 | `amount_normalized_slope_10d` | 成交额趋势（10日） | `OLS_SLOPE(amount_yuan,time,10)/MEAN(amount_yuan,10)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F203 | `amount_normalized_slope_20d` | 成交额趋势（20日） | `OLS_SLOPE(amount_yuan,time,20)/MEAN(amount_yuan,20)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F204 | `amount_normalized_slope_60d` | 成交额趋势（60日） | `OLS_SLOPE(amount_yuan,time,60)/MEAN(amount_yuan,60)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F205 | `amount_normalized_slope_120d` | 成交额趋势（120日） | `OLS_SLOPE(amount_yuan,time,120)/MEAN(amount_yuan,120)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |

## P 日线价格形态与价量关系：55个

补充开高低收和价量配合，避免只看收盘收益。

内部维度：原有基础因子 1个；单日K线形态 6个；跨日形态与价量配合 48个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F206 | `high_low_range` | (当日复权最高价−复权最低价)/复权开盘价。 | `(当日复权最高价−复权最低价)/复权开盘价。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F207 | `close_location` | 收盘在日内区间的位置 | `(C-L)/(H-L)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F208 | `upper_shadow` | 上影线占比 | `(H-max(O,C))/(H-L)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F209 | `lower_shadow` | 下影线占比 | `(min(O,C)-L)/(H-L)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F210 | `signed_body` | 有符号实体占比 | `(C-O)/(H-L)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F211 | `gap_to_true_range` | 开盘跳空相对真实振幅 | `(O-LAG(C,1))/TR` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F212 | `close_vs_previous_high` | 收盘相对前日高点 | `(C-LAG(H,1))/LAG(TR,1)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F213 | `close_location_mean_3d` | 平均收盘位置（3日） | `MEAN(close_location,3)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F214 | `close_location_mean_5d` | 平均收盘位置（5日） | `MEAN(close_location,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F215 | `close_location_mean_10d` | 平均收盘位置（10日） | `MEAN(close_location,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F216 | `close_location_mean_20d` | 平均收盘位置（20日） | `MEAN(close_location,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F217 | `close_location_mean_40d` | 平均收盘位置（40日） | `MEAN(close_location,40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F218 | `close_location_mean_60d` | 平均收盘位置（60日） | `MEAN(close_location,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F219 | `upper_shadow_mean_3d` | 平均上影线（3日） | `MEAN(upper_shadow,3)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F220 | `upper_shadow_mean_5d` | 平均上影线（5日） | `MEAN(upper_shadow,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F221 | `upper_shadow_mean_10d` | 平均上影线（10日） | `MEAN(upper_shadow,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F222 | `upper_shadow_mean_20d` | 平均上影线（20日） | `MEAN(upper_shadow,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F223 | `upper_shadow_mean_40d` | 平均上影线（40日） | `MEAN(upper_shadow,40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F224 | `upper_shadow_mean_60d` | 平均上影线（60日） | `MEAN(upper_shadow,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F225 | `lower_shadow_mean_3d` | 平均下影线（3日） | `MEAN(lower_shadow,3)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F226 | `lower_shadow_mean_5d` | 平均下影线（5日） | `MEAN(lower_shadow,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F227 | `lower_shadow_mean_10d` | 平均下影线（10日） | `MEAN(lower_shadow,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F228 | `lower_shadow_mean_20d` | 平均下影线（20日） | `MEAN(lower_shadow,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F229 | `lower_shadow_mean_40d` | 平均下影线（40日） | `MEAN(lower_shadow,40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F230 | `lower_shadow_mean_60d` | 平均下影线（60日） | `MEAN(lower_shadow,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F231 | `breakout_distance_3d` | 相对前期高点突破距离（3日） | `(C-LAG(MAX(H,3),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F232 | `breakout_distance_5d` | 相对前期高点突破距离（5日） | `(C-LAG(MAX(H,5),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F233 | `breakout_distance_10d` | 相对前期高点突破距离（10日） | `(C-LAG(MAX(H,10),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F234 | `breakout_distance_20d` | 相对前期高点突破距离（20日） | `(C-LAG(MAX(H,20),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F235 | `breakout_distance_40d` | 相对前期高点突破距离（40日） | `(C-LAG(MAX(H,40),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F236 | `breakout_distance_60d` | 相对前期高点突破距离（60日） | `(C-LAG(MAX(H,60),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F237 | `breakdown_distance_3d` | 相对前期低点距离（3日） | `(C-LAG(MIN(L,3),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F238 | `breakdown_distance_5d` | 相对前期低点距离（5日） | `(C-LAG(MIN(L,5),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F239 | `breakdown_distance_10d` | 相对前期低点距离（10日） | `(C-LAG(MIN(L,10),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F240 | `breakdown_distance_20d` | 相对前期低点距离（20日） | `(C-LAG(MIN(L,20),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F241 | `breakdown_distance_40d` | 相对前期低点距离（40日） | `(C-LAG(MIN(L,40),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F242 | `breakdown_distance_60d` | 相对前期低点距离（60日） | `(C-LAG(MIN(L,60),1))/MEAN(TR,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F243 | `return_volume_corr_3d` | 收益与成交量变化相关（3日） | `CORR(r,log(volume/LAG(volume,1)),3)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F244 | `return_volume_corr_5d` | 收益与成交量变化相关（5日） | `CORR(r,log(volume/LAG(volume,1)),5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F245 | `return_volume_corr_10d` | 收益与成交量变化相关（10日） | `CORR(r,log(volume/LAG(volume,1)),10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F246 | `return_volume_corr_20d` | 收益与成交量变化相关（20日） | `CORR(r,log(volume/LAG(volume,1)),20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F247 | `return_volume_corr_40d` | 收益与成交量变化相关（40日） | `CORR(r,log(volume/LAG(volume,1)),40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F248 | `return_volume_corr_60d` | 收益与成交量变化相关（60日） | `CORR(r,log(volume/LAG(volume,1)),60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F249 | `up_volume_share_3d` | 上涨日期成交量占比（3日） | `SUM(volume*I(r>0),3)/SUM(volume,3)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F250 | `up_volume_share_5d` | 上涨日期成交量占比（5日） | `SUM(volume*I(r>0),5)/SUM(volume,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F251 | `up_volume_share_10d` | 上涨日期成交量占比（10日） | `SUM(volume*I(r>0),10)/SUM(volume,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F252 | `up_volume_share_20d` | 上涨日期成交量占比（20日） | `SUM(volume*I(r>0),20)/SUM(volume,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F253 | `up_volume_share_40d` | 上涨日期成交量占比（40日） | `SUM(volume*I(r>0),40)/SUM(volume,40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F254 | `up_volume_share_60d` | 上涨日期成交量占比（60日） | `SUM(volume*I(r>0),60)/SUM(volume,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F255 | `close_pressure_3d` | 收盘位置成交额加权（3日） | `SUM((2*close_location-1)*amount_yuan,3)/SUM(amount_yuan,3)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F256 | `close_pressure_5d` | 收盘位置成交额加权（5日） | `SUM((2*close_location-1)*amount_yuan,5)/SUM(amount_yuan,5)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F257 | `close_pressure_10d` | 收盘位置成交额加权（10日） | `SUM((2*close_location-1)*amount_yuan,10)/SUM(amount_yuan,10)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F258 | `close_pressure_20d` | 收盘位置成交额加权（20日） | `SUM((2*close_location-1)*amount_yuan,20)/SUM(amount_yuan,20)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F259 | `close_pressure_40d` | 收盘位置成交额加权（40日） | `SUM((2*close_location-1)*amount_yuan,40)/SUM(amount_yuan,40)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F260 | `close_pressure_60d` | 收盘位置成交额加权（60日） | `SUM((2*close_location-1)*amount_yuan,60)/SUM(amount_yuan,60)` | daily+adj_factor；已实现；通过样本审计不等于全市场认证 |

## F 资金流与订单规模结构：50个

检验不同订单规模的买卖压力；数据口径和覆盖需审计。

内部维度：原有基础因子 6个；订单规模结构 4个；资金流持续性与价格响应 40个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F261 | `net_mf_ratio` | 10×净资金流入额/成交额；10用于万元与千元单位换算。 | `10×净资金流入额/成交额；10用于万元与千元单位换算。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F262 | `large_order_net_ratio` | 10×(大单及超大单买入额−卖出额)/成交额。 | `10×(大单及超大单买入额−卖出额)/成交额。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F263 | `extra_large_order_net_ratio` | 10×(超大单买入额−卖出额)/成交额。 | `10×(超大单买入额−卖出额)/成交额。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F264 | `medium_order_net_ratio` | 10×(中单买入额−卖出额)/成交额。 | `10×(中单买入额−卖出额)/成交额。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F265 | `small_order_net_ratio` | 10×(小单买入额−卖出额)/成交额。 | `10×(小单买入额−卖出额)/成交额。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F266 | `net_mf_ratio_5d` | 过去5日net_mf_ratio平均值。 | `过去5日net_mf_ratio平均值。` | v2/features/balanced_features.py；V1快照保留不变；V2审计修复：按连续市场交易日重算，停牌缺口使完整窗口失效 |
| F267 | `large_buy_share` | 大单及超大单买入占比 | `(buy_lg+buy_elg)/buy_all` | moneyflow.*_amount；已实现；通过样本审计不等于全市场认证 |
| F268 | `extra_large_buy_share` | 超大单买入占比 | `buy_elg/buy_all` | moneyflow.*_amount；已实现；通过样本审计不等于全市场认证 |
| F269 | `order_size_hhi` | 订单规模成交结构集中度 | `SUM_ACROSS_SIZE(((buy_size+sell_size)/(buy_all+sell_all))^2)` | moneyflow.*_amount；已实现；通过样本审计不等于全市场认证 |
| F270 | `large_buy_sell_logratio` | 大单买卖金额对数比 | `log((buy_lg+buy_elg)/(sell_lg+sell_elg))` | moneyflow.*_amount；已实现；通过样本审计不等于全市场认证 |
| F271 | `large_net_mean_5d` | 大单净流入比例均值（5日） | `MEAN(large_order_net_ratio,5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F272 | `large_net_mean_10d` | 大单净流入比例均值（10日） | `MEAN(large_order_net_ratio,10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F273 | `large_net_mean_20d` | 大单净流入比例均值（20日） | `MEAN(large_order_net_ratio,20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F274 | `large_net_mean_40d` | 大单净流入比例均值（40日） | `MEAN(large_order_net_ratio,40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F275 | `large_net_mean_60d` | 大单净流入比例均值（60日） | `MEAN(large_order_net_ratio,60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F276 | `large_net_std_5d` | 大单净流入比例波动（5日） | `STD(large_order_net_ratio,5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F277 | `large_net_std_10d` | 大单净流入比例波动（10日） | `STD(large_order_net_ratio,10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F278 | `large_net_std_20d` | 大单净流入比例波动（20日） | `STD(large_order_net_ratio,20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F279 | `large_net_std_40d` | 大单净流入比例波动（40日） | `STD(large_order_net_ratio,40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F280 | `large_net_std_60d` | 大单净流入比例波动（60日） | `STD(large_order_net_ratio,60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F281 | `large_net_positive_share_5d` | 大单净流入为正的日期比例（5日） | `MEAN(I(large_order_net_ratio>0),5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F282 | `large_net_positive_share_10d` | 大单净流入为正的日期比例（10日） | `MEAN(I(large_order_net_ratio>0),10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F283 | `large_net_positive_share_20d` | 大单净流入为正的日期比例（20日） | `MEAN(I(large_order_net_ratio>0),20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F284 | `large_net_positive_share_40d` | 大单净流入为正的日期比例（40日） | `MEAN(I(large_order_net_ratio>0),40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F285 | `large_net_positive_share_60d` | 大单净流入为正的日期比例（60日） | `MEAN(I(large_order_net_ratio>0),60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F286 | `large_flow_return_corr_5d` | 大单资金流与收益相关（5日） | `CORR(large_order_net_ratio,r,5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F287 | `large_flow_return_corr_10d` | 大单资金流与收益相关（10日） | `CORR(large_order_net_ratio,r,10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F288 | `large_flow_return_corr_20d` | 大单资金流与收益相关（20日） | `CORR(large_order_net_ratio,r,20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F289 | `large_flow_return_corr_40d` | 大单资金流与收益相关（40日） | `CORR(large_order_net_ratio,r,40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F290 | `large_flow_return_corr_60d` | 大单资金流与收益相关（60日） | `CORR(large_order_net_ratio,r,60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F291 | `large_flow_slope_5d` | 大单净流入比例趋势（5日） | `OLS_SLOPE(large_order_net_ratio,time,5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F292 | `large_flow_slope_10d` | 大单净流入比例趋势（10日） | `OLS_SLOPE(large_order_net_ratio,time,10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F293 | `large_flow_slope_20d` | 大单净流入比例趋势（20日） | `OLS_SLOPE(large_order_net_ratio,time,20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F294 | `large_flow_slope_40d` | 大单净流入比例趋势（40日） | `OLS_SLOPE(large_order_net_ratio,time,40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F295 | `large_flow_slope_60d` | 大单净流入比例趋势（60日） | `OLS_SLOPE(large_order_net_ratio,time,60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F296 | `aggregate_net_flow_ratio_5d` | 期间总净流入占成交额（5日） | `SUM(net_flow_yuan,5)/SUM(amount_yuan,5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F297 | `aggregate_net_flow_ratio_10d` | 期间总净流入占成交额（10日） | `SUM(net_flow_yuan,10)/SUM(amount_yuan,10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F298 | `aggregate_net_flow_ratio_20d` | 期间总净流入占成交额（20日） | `SUM(net_flow_yuan,20)/SUM(amount_yuan,20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F299 | `aggregate_net_flow_ratio_40d` | 期间总净流入占成交额（40日） | `SUM(net_flow_yuan,40)/SUM(amount_yuan,40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F300 | `aggregate_net_flow_ratio_60d` | 期间总净流入占成交额（60日） | `SUM(net_flow_yuan,60)/SUM(amount_yuan,60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F301 | `aggregate_large_buy_share_5d` | 期间大单买入占比（5日） | `SUM(buy_lg+buy_elg,5)/SUM(buy_all,5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F302 | `aggregate_large_buy_share_10d` | 期间大单买入占比（10日） | `SUM(buy_lg+buy_elg,10)/SUM(buy_all,10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F303 | `aggregate_large_buy_share_20d` | 期间大单买入占比（20日） | `SUM(buy_lg+buy_elg,20)/SUM(buy_all,20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F304 | `aggregate_large_buy_share_40d` | 期间大单买入占比（40日） | `SUM(buy_lg+buy_elg,40)/SUM(buy_all,40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F305 | `aggregate_large_buy_share_60d` | 期间大单买入占比（60日） | `SUM(buy_lg+buy_elg,60)/SUM(buy_all,60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F306 | `gross_order_imbalance_std_5d` | 买卖金额不平衡波动（5日） | `STD((buy_all-sell_all)/(buy_all+sell_all),5)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F307 | `gross_order_imbalance_std_10d` | 买卖金额不平衡波动（10日） | `STD((buy_all-sell_all)/(buy_all+sell_all),10)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F308 | `gross_order_imbalance_std_20d` | 买卖金额不平衡波动（20日） | `STD((buy_all-sell_all)/(buy_all+sell_all),20)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F309 | `gross_order_imbalance_std_40d` | 买卖金额不平衡波动（40日） | `STD((buy_all-sell_all)/(buy_all+sell_all),40)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F310 | `gross_order_imbalance_std_60d` | 买卖金额不平衡波动（60日） | `STD((buy_all-sell_all)/(buy_all+sell_all),60)` | moneyflow+daily+adj_factor；已实现；通过样本审计不等于全市场认证 |

## X 市场相对表现与特质收益：45个

区分市场共同波动与个股特质信息；不依赖历史行业成分。

内部维度：原有基础因子 2个；市场暴露 4个；个股特质表现 35个；市场活跃度敏感性 4个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F311 | `market_beta_60d` | 过去60日股票收益对当日全市场股票等权收益的回归斜率。 | `过去60日股票收益对当日全市场股票等权收益的回归斜率。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F312 | `market_corr_60d` | 过去60日股票收益与当日全市场股票等权收益的相关系数。 | `过去60日股票收益与当日全市场股票等权收益的相关系数。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F313 | `market_beta_20d` | 市场Beta（20日） | `COV(r,rm,20)/VAR(rm,20)` | daily+adj_factor，同日可观察股票池；已实现；通过样本审计不等于全市场认证 |
| F314 | `market_beta_120d` | 市场Beta（120日） | `COV(r,rm,120)/VAR(rm,120)` | daily+adj_factor，同日可观察股票池；已实现；通过样本审计不等于全市场认证 |
| F315 | `market_corr_20d` | 市场相关性（20日） | `CORR(r,rm,20)` | daily+adj_factor，同日可观察股票池；已实现；通过样本审计不等于全市场认证 |
| F316 | `market_corr_120d` | 市场相关性（120日） | `CORR(r,rm,120)` | daily+adj_factor，同日可观察股票池；已实现；通过样本审计不等于全市场认证 |
| F317 | `beta_adjusted_return_5d` | 固定事前Beta调整收益（5日） | `SUM(r,5)-LAG(beta60,5)*SUM(rm,5)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F318 | `beta_adjusted_return_10d` | 固定事前Beta调整收益（10日） | `SUM(r,10)-LAG(beta60,10)*SUM(rm,10)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F319 | `beta_adjusted_return_20d` | 固定事前Beta调整收益（20日） | `SUM(r,20)-LAG(beta60,20)*SUM(rm,20)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F320 | `beta_adjusted_return_60d` | 固定事前Beta调整收益（60日） | `SUM(r,60)-LAG(beta60,60)*SUM(rm,60)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F321 | `beta_adjusted_return_120d` | 固定事前Beta调整收益（120日） | `SUM(r,120)-LAG(beta60,120)*SUM(rm,120)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F322 | `residual_momentum_5d` | 逐日事前市场残差累计（5日） | `SUM(e,5)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F323 | `residual_momentum_10d` | 逐日事前市场残差累计（10日） | `SUM(e,10)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F324 | `residual_momentum_20d` | 逐日事前市场残差累计（20日） | `SUM(e,20)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F325 | `residual_momentum_60d` | 逐日事前市场残差累计（60日） | `SUM(e,60)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F326 | `residual_momentum_120d` | 逐日事前市场残差累计（120日） | `SUM(e,120)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F327 | `residual_volatility_5d` | 市场残差波动（5日） | `STD(e,5)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F328 | `residual_volatility_10d` | 市场残差波动（10日） | `STD(e,10)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F329 | `residual_volatility_20d` | 市场残差波动（20日） | `STD(e,20)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F330 | `residual_volatility_60d` | 市场残差波动（60日） | `STD(e,60)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F331 | `residual_volatility_120d` | 市场残差波动（120日） | `STD(e,120)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F332 | `residual_positive_share_5d` | 市场残差为正比例（5日） | `MEAN(I(e>0),5)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F333 | `residual_positive_share_10d` | 市场残差为正比例（10日） | `MEAN(I(e>0),10)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F334 | `residual_positive_share_20d` | 市场残差为正比例（20日） | `MEAN(I(e>0),20)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F335 | `residual_positive_share_60d` | 市场残差为正比例（60日） | `MEAN(I(e>0),60)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F336 | `residual_positive_share_120d` | 市场残差为正比例（120日） | `MEAN(I(e>0),120)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F337 | `up_market_capture_20d` | 上涨市场收益捕获率（20日） | `SUM_IF(r,rm>0,20)/SUM_IF(rm,rm>0,20)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F338 | `up_market_capture_40d` | 上涨市场收益捕获率（40日） | `SUM_IF(r,rm>0,40)/SUM_IF(rm,rm>0,40)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F339 | `up_market_capture_60d` | 上涨市场收益捕获率（60日） | `SUM_IF(r,rm>0,60)/SUM_IF(rm,rm>0,60)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F340 | `up_market_capture_120d` | 上涨市场收益捕获率（120日） | `SUM_IF(r,rm>0,120)/SUM_IF(rm,rm>0,120)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F341 | `up_market_capture_250d` | 上涨市场收益捕获率（250日） | `SUM_IF(r,rm>0,250)/SUM_IF(rm,rm>0,250)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F342 | `down_market_capture_20d` | 下跌市场收益捕获率（20日） | `SUM_IF(r,rm<0,20)/SUM_IF(rm,rm<0,20)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F343 | `down_market_capture_40d` | 下跌市场收益捕获率（40日） | `SUM_IF(r,rm<0,40)/SUM_IF(rm,rm<0,40)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F344 | `down_market_capture_60d` | 下跌市场收益捕获率（60日） | `SUM_IF(r,rm<0,60)/SUM_IF(rm,rm<0,60)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F345 | `down_market_capture_120d` | 下跌市场收益捕获率（120日） | `SUM_IF(r,rm<0,120)/SUM_IF(rm,rm<0,120)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F346 | `down_market_capture_250d` | 下跌市场收益捕获率（250日） | `SUM_IF(r,rm<0,250)/SUM_IF(rm,rm<0,250)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F347 | `residual_ts_rank_5d` | 当日特质冲击历史分位（5日） | `TSRANK(e,5)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F348 | `residual_ts_rank_10d` | 当日特质冲击历史分位（10日） | `TSRANK(e,10)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F349 | `residual_ts_rank_20d` | 当日特质冲击历史分位（20日） | `TSRANK(e,20)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F350 | `residual_ts_rank_60d` | 当日特质冲击历史分位（60日） | `TSRANK(e,60)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F351 | `residual_ts_rank_120d` | 当日特质冲击历史分位（120日） | `TSRANK(e,120)` | daily+adj_factor；事前滚动市场回归；已实现；通过样本审计不等于全市场认证 |
| F352 | `market_activity_corr_20d` | 个股收益与市场换手相关（20日） | `CORR(r,market_median_turnover,20)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F353 | `market_activity_corr_40d` | 个股收益与市场换手相关（40日） | `CORR(r,market_median_turnover,40)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F354 | `market_activity_corr_60d` | 个股收益与市场换手相关（60日） | `CORR(r,market_median_turnover,60)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F355 | `market_activity_corr_120d` | 个股收益与市场换手相关（120日） | `CORR(r,market_median_turnover,120)` | daily+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |

## Q 财务质量、成长与公告变化：45个

提供企业状态；慢频信息对5日任务的直接作用需验证。

内部维度：原有基础因子 8个；补充质量指标 4个；财报变化 16个；财务稳定性 4个；财务趋势 4个；成长持续性 2个；信息新鲜度 1个；质量内部组合 6个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F356 | `roe` | 最近可用财务公告的净资产收益率；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的净资产收益率；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F357 | `roa` | 最近可用财务公告的总资产报酬率；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的总资产报酬率；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F358 | `grossprofit_margin` | 最近可用财务公告的毛利率；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的毛利率；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F359 | `debt_to_assets` | 最近可用财务公告的资产负债率；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的资产负债率；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F360 | `current_ratio` | 最近可用财务公告的流动比率；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的流动比率；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F361 | `netprofit_yoy` | 最近可用财务公告的净利润同比增长率；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的净利润同比增长率；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F362 | `or_yoy` | 最近可用财务公告的营业收入同比增长率；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的营业收入同比增长率；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F363 | `ocf_to_or` | 最近可用财务公告的经营现金净流量/营业收入；公告日严格早于信号日，保留V1公告选择及数据源单位。 | `最近可用财务公告的经营现金净流量/营业收入；公告日严格早于信号日，保留V1公告选择及数据源单位。` | V1 build_features.py同名SQL及原有因子表；隔离：历史财务PIT及报告口径待验证 |
| F364 | `netprofit_margin` | 销售净利率 | `PIT(netprofit_margin)` | fina_indicator；隔离：历史财务PIT及报告口径待验证 |
| F365 | `assets_turn` | 总资产周转率 | `PIT(assets_turn)` | fina_indicator；隔离：历史财务PIT及报告口径待验证 |
| F366 | `quick_ratio` | 速动比率 | `PIT(quick_ratio)` | fina_indicator；隔离：历史财务PIT及报告口径待验证 |
| F367 | `cash_profitability_gap` | 现金质量与会计利润差 | `Q(ocf_to_or)-Q(netprofit_margin)` | fina_indicator；隔离：历史财务PIT及报告口径待验证 |
| F368 | `roe_report_delta_1q` | roe较前1季度报告变化 | `PIT(roe,q)-PIT(roe,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F369 | `roe_report_delta_4q` | roe较前4季度报告变化 | `PIT(roe,q)-PIT(roe,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F370 | `roa_report_delta_1q` | roa较前1季度报告变化 | `PIT(roa,q)-PIT(roa,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F371 | `roa_report_delta_4q` | roa较前4季度报告变化 | `PIT(roa,q)-PIT(roa,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F372 | `grossprofit_margin_report_delta_1q` | grossprofit_margin较前1季度报告变化 | `PIT(grossprofit_margin,q)-PIT(grossprofit_margin,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F373 | `grossprofit_margin_report_delta_4q` | grossprofit_margin较前4季度报告变化 | `PIT(grossprofit_margin,q)-PIT(grossprofit_margin,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F374 | `debt_to_assets_report_delta_1q` | debt_to_assets较前1季度报告变化 | `PIT(debt_to_assets,q)-PIT(debt_to_assets,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F375 | `debt_to_assets_report_delta_4q` | debt_to_assets较前4季度报告变化 | `PIT(debt_to_assets,q)-PIT(debt_to_assets,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F376 | `current_ratio_report_delta_1q` | current_ratio较前1季度报告变化 | `PIT(current_ratio,q)-PIT(current_ratio,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F377 | `current_ratio_report_delta_4q` | current_ratio较前4季度报告变化 | `PIT(current_ratio,q)-PIT(current_ratio,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F378 | `netprofit_yoy_report_delta_1q` | netprofit_yoy较前1季度报告变化 | `PIT(netprofit_yoy,q)-PIT(netprofit_yoy,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F379 | `netprofit_yoy_report_delta_4q` | netprofit_yoy较前4季度报告变化 | `PIT(netprofit_yoy,q)-PIT(netprofit_yoy,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F380 | `or_yoy_report_delta_1q` | or_yoy较前1季度报告变化 | `PIT(or_yoy,q)-PIT(or_yoy,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F381 | `or_yoy_report_delta_4q` | or_yoy较前4季度报告变化 | `PIT(or_yoy,q)-PIT(or_yoy,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F382 | `ocf_to_or_report_delta_1q` | ocf_to_or较前1季度报告变化 | `PIT(ocf_to_or,q)-PIT(ocf_to_or,q-1)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F383 | `ocf_to_or_report_delta_4q` | ocf_to_or较前4季度报告变化 | `PIT(ocf_to_or,q)-PIT(ocf_to_or,q-4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F384 | `roe_report_std_8q` | roe八季度波动 | `STD_LAST_REPORTS(PIT(roe),8)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F385 | `roe_report_slope_4q` | roe四季度趋势 | `SLOPE_LAST_REPORTS(PIT(roe),4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F386 | `roa_report_std_8q` | roa八季度波动 | `STD_LAST_REPORTS(PIT(roa),8)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F387 | `roa_report_slope_4q` | roa四季度趋势 | `SLOPE_LAST_REPORTS(PIT(roa),4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F388 | `grossprofit_margin_report_std_8q` | grossprofit_margin八季度波动 | `STD_LAST_REPORTS(PIT(grossprofit_margin),8)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F389 | `grossprofit_margin_report_slope_4q` | grossprofit_margin四季度趋势 | `SLOPE_LAST_REPORTS(PIT(grossprofit_margin),4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F390 | `ocf_to_or_report_std_8q` | ocf_to_or八季度波动 | `STD_LAST_REPORTS(PIT(ocf_to_or),8)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F391 | `ocf_to_or_report_slope_4q` | ocf_to_or四季度趋势 | `SLOPE_LAST_REPORTS(PIT(ocf_to_or),4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F392 | `netprofit_yoy_positive_share_8q` | netprofit_yoy连续报告期正增长比例 | `MEAN_LAST_REPORTS(I(PIT(netprofit_yoy)>0),8)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F393 | `or_yoy_positive_share_8q` | or_yoy连续报告期正增长比例 | `MEAN_LAST_REPORTS(I(PIT(or_yoy)>0),8)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F394 | `financial_report_age` | 最近财务公告距今天数 | `calendar_days(t-latest_visible_ann_date)` | fina_indicator.ann_date；隔离：历史财务PIT及报告口径待验证 |
| F395 | `quality_growth_product` | 盈利与成长配合 | `Q(roe)*Q(netprofit_yoy)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F396 | `cash_growth_product` | 现金质量与营收成长配合 | `Q(ocf_to_or)*Q(or_yoy)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F397 | `leverage_adjusted_profitability` | 杠杆条件下的盈利 | `Q(roe)/(1+abs(Q(debt_to_assets)))` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F398 | `profitability_disagreement` | ROE与ROA排序分歧 | `abs(Q(roe)-Q(roa))` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F399 | `margin_stability_ratio` | 毛利率水平相对波动 | `MEAN_LAST_REPORTS(PIT(grossprofit_margin),4)/STD_LAST_REPORTS(PIT(grossprofit_margin),4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |
| F400 | `cash_stability_ratio` | 经营现金收入比水平相对波动 | `MEAN_LAST_REPORTS(PIT(ocf_to_or),4)/STD_LAST_REPORTS(PIT(ocf_to_or),4)` | fina_indicator+历史公告版本；隔离：历史财务PIT及报告口径待验证 |

## V 估值、规模与股息：40个

提供横截面定价和风格背景，避免短周期任务被慢变量占满。

内部维度：原有基础因子 5个；估值状态与结构 3个；估值历史位置与变化 32个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F401 | `log_total_mv` | 正总市值的自然对数。 | `正总市值的自然对数。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F402 | `log_circ_mv` | 正流通市值的自然对数。 | `正流通市值的自然对数。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F403 | `earnings_yield` | 1/pe_ttm，绝对分母须大于1e−12，保留负估值。 | `1/pe_ttm，绝对分母须大于1e−12，保留负估值。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F404 | `book_to_price` | 1/pb，绝对分母须大于1e−12。 | `1/pb，绝对分母须大于1e−12。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F405 | `sales_yield` | 1/ps_ttm，绝对分母须大于1e−12。 | `1/ps_ttm，绝对分母须大于1e−12。` | V1 build_features.py同名SQL及原有因子表；已实现；通过样本审计不等于全市场认证 |
| F406 | `dividend_yield_ttm` | 滚动股息率 | `daily_basic.dv_ttm` | daily_basic；已实现；通过样本审计不等于全市场认证 |
| F407 | `float_cap_share` | 流通市值占总市值比例 | `circ_mv/total_mv` | daily_basic；已实现；通过样本审计不等于全市场认证 |
| F408 | `negative_earnings_flag` | 亏损估值状态 | `GATED: PIT_TTM_PROFIT_SIGN_REQUIRED` | 待历史可用TTM净利润；当前输出NULL；隔离：数据源亏损PE为空，不能用pe_ttm<0识别；缺失也不直接等同亏损 |
| F409 | `ep_ts_z_20d` | 盈利收益率历史标准分（20日） | `TSZ(earnings_yield,20)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F410 | `ep_ts_z_60d` | 盈利收益率历史标准分（60日） | `TSZ(earnings_yield,60)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F411 | `ep_ts_z_120d` | 盈利收益率历史标准分（120日） | `TSZ(earnings_yield,120)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F412 | `ep_ts_z_250d` | 盈利收益率历史标准分（250日） | `TSZ(earnings_yield,250)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F413 | `bp_ts_z_20d` | 账面市值比历史标准分（20日） | `TSZ(book_to_price,20)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F414 | `bp_ts_z_60d` | 账面市值比历史标准分（60日） | `TSZ(book_to_price,60)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F415 | `bp_ts_z_120d` | 账面市值比历史标准分（120日） | `TSZ(book_to_price,120)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F416 | `bp_ts_z_250d` | 账面市值比历史标准分（250日） | `TSZ(book_to_price,250)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F417 | `sp_ts_z_20d` | 销售市值比历史标准分（20日） | `TSZ(sales_yield,20)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F418 | `sp_ts_z_60d` | 销售市值比历史标准分（60日） | `TSZ(sales_yield,60)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F419 | `sp_ts_z_120d` | 销售市值比历史标准分（120日） | `TSZ(sales_yield,120)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F420 | `sp_ts_z_250d` | 销售市值比历史标准分（250日） | `TSZ(sales_yield,250)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F421 | `dividend_ts_z_20d` | 股息率历史标准分（20日） | `TSZ(dividend_yield_ttm,20)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F422 | `dividend_ts_z_60d` | 股息率历史标准分（60日） | `TSZ(dividend_yield_ttm,60)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F423 | `dividend_ts_z_120d` | 股息率历史标准分（120日） | `TSZ(dividend_yield_ttm,120)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F424 | `dividend_ts_z_250d` | 股息率历史标准分（250日） | `TSZ(dividend_yield_ttm,250)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F425 | `ep_ts_rank_20d` | 盈利收益率历史分位（20日） | `TSRANK(earnings_yield,20)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F426 | `ep_ts_rank_60d` | 盈利收益率历史分位（60日） | `TSRANK(earnings_yield,60)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F427 | `ep_ts_rank_120d` | 盈利收益率历史分位（120日） | `TSRANK(earnings_yield,120)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F428 | `ep_ts_rank_250d` | 盈利收益率历史分位（250日） | `TSRANK(earnings_yield,250)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F429 | `bp_ts_rank_20d` | 账面市值比历史分位（20日） | `TSRANK(book_to_price,20)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F430 | `bp_ts_rank_60d` | 账面市值比历史分位（60日） | `TSRANK(book_to_price,60)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F431 | `bp_ts_rank_120d` | 账面市值比历史分位（120日） | `TSRANK(book_to_price,120)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F432 | `bp_ts_rank_250d` | 账面市值比历史分位（250日） | `TSRANK(book_to_price,250)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F433 | `sp_ts_rank_20d` | 销售市值比历史分位（20日） | `TSRANK(sales_yield,20)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F434 | `sp_ts_rank_60d` | 销售市值比历史分位（60日） | `TSRANK(sales_yield,60)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F435 | `sp_ts_rank_120d` | 销售市值比历史分位（120日） | `TSRANK(sales_yield,120)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F436 | `sp_ts_rank_250d` | 销售市值比历史分位（250日） | `TSRANK(sales_yield,250)` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F437 | `positive_pe_rerating_20d` | 正市盈率重估（20日） | `log(pe_ttm/LAG(pe_ttm,20)); both PE>0` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F438 | `positive_pe_rerating_60d` | 正市盈率重估（60日） | `log(pe_ttm/LAG(pe_ttm,60)); both PE>0` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F439 | `positive_pe_rerating_120d` | 正市盈率重估（120日） | `log(pe_ttm/LAG(pe_ttm,120)); both PE>0` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |
| F440 | `positive_pe_rerating_250d` | 正市盈率重估（250日） | `log(pe_ttm/LAG(pe_ttm,250)); both PE>0` | daily_basic；历史值可用性需复核；已实现；通过样本审计不等于全市场认证 |

## C 条件筛选收益：35个

保留导师的关键假设，用少量窗口测试，不让参数组合占满库。

内部维度：高活跃日期条件收益 30个；条件下的收益结构 5个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F441 | `cond_volume_high_close_sum_10d` | volume最高半数日期高条件日期累计收益（10日） | `S=TOP_HALF(volume,10); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F442 | `cond_volume_high_close_downside_rms_10d` | volume最高半数日期高条件日期下行均方根（10日） | `S=TOP_HALF(volume,10); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F443 | `cond_volume_high_close_sum_20d` | volume最高半数日期高条件日期累计收益（20日） | `S=TOP_HALF(volume,20); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F444 | `cond_volume_high_close_downside_rms_20d` | volume最高半数日期高条件日期下行均方根（20日） | `S=TOP_HALF(volume,20); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F445 | `cond_volume_high_close_sum_60d` | volume最高半数日期高条件日期累计收益（60日） | `S=TOP_HALF(volume,60); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F446 | `cond_volume_high_close_downside_rms_60d` | volume最高半数日期高条件日期下行均方根（60日） | `S=TOP_HALF(volume,60); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F447 | `cond_volume_overnight_intraday_corr_20d` | volume最高10天隔夜与日内相关 | `S=TOP_HALF(volume,20); CORR_SELECTED(ro,ri,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F448 | `cond_amount_high_close_sum_10d` | amount最高半数日期高条件日期累计收益（10日） | `S=TOP_HALF(amount_yuan,10); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F449 | `cond_amount_high_close_downside_rms_10d` | amount最高半数日期高条件日期下行均方根（10日） | `S=TOP_HALF(amount_yuan,10); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F450 | `cond_amount_high_close_sum_20d` | amount最高半数日期高条件日期累计收益（20日） | `S=TOP_HALF(amount_yuan,20); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F451 | `cond_amount_high_close_downside_rms_20d` | amount最高半数日期高条件日期下行均方根（20日） | `S=TOP_HALF(amount_yuan,20); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F452 | `cond_amount_high_close_sum_60d` | amount最高半数日期高条件日期累计收益（60日） | `S=TOP_HALF(amount_yuan,60); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F453 | `cond_amount_high_close_downside_rms_60d` | amount最高半数日期高条件日期下行均方根（60日） | `S=TOP_HALF(amount_yuan,60); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F454 | `cond_amount_overnight_intraday_corr_20d` | amount最高10天隔夜与日内相关 | `S=TOP_HALF(amount_yuan,20); CORR_SELECTED(ro,ri,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F455 | `cond_turnover_high_close_sum_10d` | turnover最高半数日期高条件日期累计收益（10日） | `S=TOP_HALF(turnover_rate,10); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F456 | `cond_turnover_high_close_downside_rms_10d` | turnover最高半数日期高条件日期下行均方根（10日） | `S=TOP_HALF(turnover_rate,10); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F457 | `cond_turnover_high_close_sum_20d` | turnover最高半数日期高条件日期累计收益（20日） | `S=TOP_HALF(turnover_rate,20); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F458 | `cond_turnover_high_close_downside_rms_20d` | turnover最高半数日期高条件日期下行均方根（20日） | `S=TOP_HALF(turnover_rate,20); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F459 | `cond_turnover_high_close_sum_60d` | turnover最高半数日期高条件日期累计收益（60日） | `S=TOP_HALF(turnover_rate,60); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F460 | `cond_turnover_high_close_downside_rms_60d` | turnover最高半数日期高条件日期下行均方根（60日） | `S=TOP_HALF(turnover_rate,60); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F461 | `cond_turnover_overnight_intraday_corr_20d` | turnover最高10天隔夜与日内相关 | `S=TOP_HALF(turnover_rate,20); CORR_SELECTED(ro,ri,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F462 | `cond_large_buy_high_close_sum_10d` | large_buy最高半数日期高条件日期累计收益（10日） | `S=TOP_HALF(buy_lg+buy_elg,10); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F463 | `cond_large_buy_high_close_downside_rms_10d` | large_buy最高半数日期高条件日期下行均方根（10日） | `S=TOP_HALF(buy_lg+buy_elg,10); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F464 | `cond_large_buy_high_close_sum_20d` | large_buy最高半数日期高条件日期累计收益（20日） | `S=TOP_HALF(buy_lg+buy_elg,20); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F465 | `cond_large_buy_high_close_downside_rms_20d` | large_buy最高半数日期高条件日期下行均方根（20日） | `S=TOP_HALF(buy_lg+buy_elg,20); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F466 | `cond_large_buy_high_close_sum_60d` | large_buy最高半数日期高条件日期累计收益（60日） | `S=TOP_HALF(buy_lg+buy_elg,60); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F467 | `cond_large_buy_high_close_downside_rms_60d` | large_buy最高半数日期高条件日期下行均方根（60日） | `S=TOP_HALF(buy_lg+buy_elg,60); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F468 | `cond_large_buy_overnight_intraday_corr_20d` | large_buy最高10天隔夜与日内相关 | `S=TOP_HALF(buy_lg+buy_elg,20); CORR_SELECTED(ro,ri,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F469 | `cond_large_net_high_close_sum_10d` | large_net最高半数日期高条件日期累计收益（10日） | `S=TOP_HALF(buy_lg+buy_elg-sell_lg-sell_elg,10); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F470 | `cond_large_net_high_close_downside_rms_10d` | large_net最高半数日期高条件日期下行均方根（10日） | `S=TOP_HALF(buy_lg+buy_elg-sell_lg-sell_elg,10); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F471 | `cond_large_net_high_close_sum_20d` | large_net最高半数日期高条件日期累计收益（20日） | `S=TOP_HALF(buy_lg+buy_elg-sell_lg-sell_elg,20); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F472 | `cond_large_net_high_close_downside_rms_20d` | large_net最高半数日期高条件日期下行均方根（20日） | `S=TOP_HALF(buy_lg+buy_elg-sell_lg-sell_elg,20); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F473 | `cond_large_net_high_close_sum_60d` | large_net最高半数日期高条件日期累计收益（60日） | `S=TOP_HALF(buy_lg+buy_elg-sell_lg-sell_elg,60); SUM_SELECTED(r,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F474 | `cond_large_net_high_close_downside_rms_60d` | large_net最高半数日期高条件日期下行均方根（60日） | `S=TOP_HALF(buy_lg+buy_elg-sell_lg-sell_elg,60); sqrt(MEAN_SELECTED(min(r,0)^2,S))` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |
| F475 | `cond_large_net_overnight_intraday_corr_20d` | large_net最高10天隔夜与日内相关 | `S=TOP_HALF(buy_lg+buy_elg-sell_lg-sell_elg,20); CORR_SELECTED(ro,ri,S)` | daily+moneyflow+daily_basic+adj_factor；已实现；通过样本审计不等于全市场认证 |

## I 跨方向交互：25个

向线性模型提供少量有经济解释的非线性条件；先验证主效应。

内部维度：跨方向两两交互 25个。

| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |
|---|---|---|---|---|
| F476 | `interaction_01` | 短期收益×换手水平 | `Q(logret_5d)*Q(turnover_mean_20d)` | logret_5d; turnover_mean_20d；已实现；通过样本审计不等于全市场认证 |
| F477 | `interaction_02` | 动量×换手冲击 | `Q(logret_20d)*Q(turnover_surprise_20d)` | logret_20d; turnover_surprise_20d；已实现；通过样本审计不等于全市场认证 |
| F478 | `interaction_03` | 动量×非流动性 | `Q(logret_20d)*Q(amihud_proxy_20d)` | logret_20d; amihud_proxy_20d；已实现；通过样本审计不等于全市场认证 |
| F479 | `interaction_04` | 短期收益×成交容量 | `Q(logret_5d)*Q(log_mean_amount_20d)` | logret_5d; log_mean_amount_20d；已实现；通过样本审计不等于全市场认证 |
| F480 | `interaction_05` | 趋势效率×成交不稳定性 | `Q(path_efficiency_20d)*Q(amount_cv_20d)` | path_efficiency_20d; amount_cv_20d；已实现；通过样本审计不等于全市场认证 |
| F481 | `interaction_06` | 动量×波动状态 | `Q(logret_20d)*Q(volatility_20d)` | logret_20d; volatility_20d；已实现；通过样本审计不等于全市场认证 |
| F482 | `interaction_07` | 短期收益×下行风险 | `Q(logret_5d)*Q(downside_volatility_20d)` | logret_5d; downside_volatility_20d；已实现；通过样本审计不等于全市场认证 |
| F483 | `interaction_08` | 隔夜冲击×换手 | `Q(overnight_logret)*Q(turnover_rate)` | overnight_logret; turnover_rate；已实现；通过样本审计不等于全市场认证 |
| F484 | `interaction_09` | 收盘位置×大单资金压力 | `Q(close_location)*Q(large_order_net_ratio)` | close_location; large_order_net_ratio；已实现；通过样本审计不等于全市场认证 |
| F485 | `interaction_10` | 上影线×持续资金流 | `Q(upper_shadow)*Q(large_net_mean_20d)` | upper_shadow; large_net_mean_20d；已实现；通过样本审计不等于全市场认证 |
| F486 | `interaction_11` | 下影线×大单买入占比 | `Q(lower_shadow)*Q(aggregate_large_buy_share_20d)` | lower_shadow; aggregate_large_buy_share_20d；已实现；通过样本审计不等于全市场认证 |
| F487 | `interaction_12` | 突破×换手冲击 | `Q(breakout_distance_20d)*Q(turnover_surprise_20d)` | breakout_distance_20d; turnover_surprise_20d；已实现；通过样本审计不等于全市场认证 |
| F488 | `interaction_13` | 突破×资金流持续性 | `Q(breakout_distance_20d)*Q(large_net_positive_share_20d)` | breakout_distance_20d; large_net_positive_share_20d；已实现；通过样本审计不等于全市场认证 |
| F489 | `interaction_14` | 动量×大单净流入趋势 | `Q(logret_20d)*Q(large_flow_slope_20d)` | logret_20d; large_flow_slope_20d；已实现；通过样本审计不等于全市场认证 |
| F490 | `interaction_15` | 特质动量×大单资金流 | `Q(residual_momentum_20d)*Q(large_net_mean_20d)` | residual_momentum_20d; large_net_mean_20d；已实现；通过样本审计不等于全市场认证 |
| F491 | `interaction_16` | 特质波动×流通规模 | `Q(residual_volatility_20d)*Q(log_circ_mv)` | residual_volatility_20d; log_circ_mv；已实现；通过样本审计不等于全市场认证 |
| F492 | `interaction_17` | 市场暴露×非流动性 | `Q(market_beta_60d)*Q(amihud_proxy_20d)` | market_beta_60d; amihud_proxy_20d；已实现；通过样本审计不等于全市场认证 |
| F493 | `interaction_18` | 估值×盈利质量 | `Q(earnings_yield)*Q(roe)` | earnings_yield; roe；隔离：历史财务PIT及报告口径待验证 |
| F494 | `interaction_19` | 账面估值×杠杆 | `Q(book_to_price)*Q(debt_to_assets)` | book_to_price; debt_to_assets；隔离：历史财务PIT及报告口径待验证 |
| F495 | `interaction_20` | 收入估值×毛利率 | `Q(sales_yield)*Q(grossprofit_margin)` | sales_yield; grossprofit_margin；隔离：历史财务PIT及报告口径待验证 |
| F496 | `interaction_21` | 动量×估值 | `Q(logret_20d)*Q(earnings_yield)` | logret_20d; earnings_yield；已实现；通过样本审计不等于全市场认证 |
| F497 | `interaction_22` | 动量×利润成长 | `Q(logret_20d)*Q(netprofit_yoy)` | logret_20d; netprofit_yoy；隔离：历史财务PIT及报告口径待验证 |
| F498 | `interaction_23` | 流通规模×交易冲击 | `Q(log_circ_mv)*Q(turnover_surprise_20d)` | log_circ_mv; turnover_surprise_20d；已实现；通过样本审计不等于全市场认证 |
| F499 | `interaction_24` | 股息×风险状态 | `Q(dividend_yield_ttm)*Q(volatility_20d)` | dividend_yield_ttm; volatility_20d；已实现；通过样本审计不等于全市场认证 |
| F500 | `interaction_25` | 公告信息新鲜度×短期收益 | `Q(financial_report_age)*Q(logret_5d)` | financial_report_age; logret_5d；隔离：历史财务PIT及报告口径待验证 |
