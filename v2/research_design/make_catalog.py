"""Design registry only: 500 named candidates in ten research families.

This does NOT implement or run the 500 numerical features. Legacy 450-filter
builder is a separate superseded design. All formulas here are specifications.
"""
from pathlib import Path
import json
import sys
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from features_500 import BASE_FEATURES, BASE_DESCRIPTIONS

FAMILIES = {
    'M': ('动量、反转与收益结构',80,'最高','与5日预测最直接相关，覆盖趋势持续与短期反转两种相反假设。'),
    'R': ('波动、尾部风险与回撤',65,'高','区分相同收益背后的风险状态，并服务交易风险评估。'),
    'L': ('流动性、换手与交易活跃度',60,'高','同时影响信号含义、可交易性和成本；日线数据可实现。'),
    'P': ('日线价格形态与价量关系',55,'高','补充开高低收和价量配合，避免只看收盘收益。'),
    'F': ('资金流与订单规模结构',50,'中高','检验不同订单规模的买卖压力；数据口径和覆盖需审计。'),
    'X': ('市场相对表现与特质收益',45,'中高','区分市场共同波动与个股特质信息；不依赖历史行业成分。'),
    'Q': ('财务质量、成长与公告变化',45,'中','提供企业状态；慢频信息对5日任务的直接作用需验证。'),
    'V': ('估值、规模与股息',40,'中','提供横截面定价和风格背景，避免短周期任务被慢变量占满。'),
    'C': ('条件筛选收益',35,'定向研究','保留导师的关键假设，用少量窗口测试，不让参数组合占满库。'),
    'I': ('跨方向交互',25,'后置','向线性模型提供少量有经济解释的非线性条件；先验证主效应。'),
}
rows=[]


def add(family, dimension, name, label, formula, source, hypothesis, status='待实现；字段已在本地样例schema核对'):
    rows.append(dict(family=family,direction=FAMILIES[family][0],dimension=dimension,
        name=name,label=label,formula=formula,source=source,hypothesis=hypothesis,status=status))


def series(family, dimension, stem, label, windows, formula, source, hypothesis, status='待实现；字段已在本地样例schema核对'):
    for w in windows:
        add(family,dimension,f'{stem}_{w}d',f'{label}（{w}日）',formula.format(w=w),source,hypothesis,status)


base_groups = {
    'M': list(BASE_FEATURES[:12])+['intraday_logret','overnight_logret'],
    'R': list(BASE_FEATURES[12:19]),
    'L': list(BASE_FEATURES[24:31]),
    'P': ['high_low_range'],
    'F': list(BASE_FEATURES[36:42]),
    'X': ['market_beta_60d','market_corr_60d'],
    'Q': list(BASE_FEATURES[42:50]),
    'V': list(BASE_FEATURES[31:36]),
}
assert sorted(sum(base_groups.values(),[])) == sorted(BASE_FEATURES)
for family,names in base_groups.items():
    for name in names:
        add(family,'原有基础因子',name,BASE_DESCRIPTIONS[name],BASE_DESCRIPTIONS[name],
            'V1 build_features.py同名SQL及原有因子表','保留原定义作为可追溯基准；不承诺原50个全部入选最终模型。',
            '已有V1计算；纳入新500表仍须逐列复核')

# M: 14 inherited + 66 new.
series('M','多尺度收益','logret','累计复权对数收益',[15,40,120,250],'SUM(r,{w})','daily+adj_factor','检验不同时间尺度的趋势与反转。')
series('M','跳过近期的动量','skip5_momentum','跳过最近5日的动量',[20,60,120],'log(C[t-5]/C[t-{w}])','daily+adj_factor','分离近期反转与较早趋势，不把收益取负当作新因子。')
momentum_templates = [
    ('trend_slope','趋势斜率','OLS_SLOPE(log(C),time,{w})','识别价格趋势速度。'),
    ('trend_r2','趋势拟合优度','OLS_R2(log(C),time,{w})','区分稳定趋势与噪声震荡。'),
    ('path_efficiency','跨日路径效率','abs(SUM(r,{w}))/SUM(abs(r),{w})','区分跨日单向运动和往返震荡，非盘中路径。'),
    ('return_acceleration','前后等长窗口收益加速度','MEAN(r,{w})-LAG(MEAN(r,{w}),{w})','检验趋势加速或衰减。'),
    ('up_day_share','上涨交易日比例','MEAN(I(r>0),{w})','区分少量大涨与多日持续上涨。'),
    ('longest_up_streak','窗口内最长连涨','MAX_STREAK(r>0,{w})/{w}','衡量上涨持续性。'),
    ('longest_down_streak','窗口内最长连跌','MAX_STREAK(r<0,{w})/{w}','衡量卖压持续性。'),
    ('return_autocorr','单日收益一阶自相关','CORR(r,LAG(r,1),{w})','识别历史延续或交替方向。'),
    ('overnight_intraday_spread','隔夜与日内收益差','SUM(ro-ri,{w})','检验收益发生时段差异，避免同时复制三个可相加收益和。'),
    ('overnight_intraday_cov','隔夜与日内协方差','COV(ro,ri,{w})','识别隔夜冲击被日内延续还是抵消。'),
    ('overnight_reversal_share','隔夜冲击日内反转频率','MEAN(I(ro*ri<0),{w})','衡量隔夜定价被日内修正的频率。'),
    ('intraday_abs_share','日内绝对收益占比','SUM(abs(ri),{w})/SUM(abs(ri)+abs(ro),{w})','区分日内与隔夜主导的股票。'),
    ('slope_change','前后窗口趋势斜率变化','OLS_SLOPE(log(C),time,{w})-LAG(OLS_SLOPE(log(C),time,{w}),{w})','衡量跨日趋势拐点。'),
    ('direction_switch_share','相邻收益换向频率','MEAN(I(r*LAG(r,1)<0),{w})','识别来回震荡程度。'),
]
for stem,label,formula,h in momentum_templates:
    series('M','趋势与收益结构',stem,label,[5,10,20,60],formula,'daily+adj_factor',h)
series('M','近期冲击位置','return_ts_rank','当日收益历史分位',[20,60,120],'TSRANK(r,{w})','daily+adj_factor','衡量当前收益相对自身历史是否极端。')

# R: 7 inherited + 58 new.
series('R','波动尺度','volatility','收益标准差',[40,120],'STD(r,{w})','daily+adj_factor','补充中长窗口风险状态。')
risk_templates = [
    ('upside_rms','上行均方根','sqrt(MEAN(max(r,0)^2,{w}))',[5,10,20,60]),
    ('downside_rms','下行均方根','sqrt(MEAN(min(r,0)^2,{w}))',[5,10,60,120]),
    ('return_kurtosis','收益超额峰度','KURT_EXCESS(r,{w})',[5,10,20,60]),
    ('return_mad','收益中位绝对偏差','MEDIAN(abs(r-MEDIAN_WINDOW(r)),{w})',[5,10,20,60]),
    ('return_q05','收益5%分位','QUANTILE(r,0.05,{w})',[5,10,20,60]),
    ('return_lower_quartile_mean','最差四分之一收益均值','MEAN_LOWEST(r,ceil({w}/4),{w})',[5,10,20,60]),
    ('best_return','最大单日收益','MAX(r,{w})',[5,10,20,60]),
    ('parkinson_vol','高低价区间波动估计','sqrt(MEAN(log(H/L)^2,{w})/(4*log(2)))',[5,10,20,60]),
    ('garman_klass_vol','OHLC波动估计','sqrt(max(MEAN(0.5*log(H/L)^2-(2*log(2)-1)*ri^2,{w}),0))',[5,10,20,60]),
    ('maximum_drawdown','窗口最大回撤','MAX_PEAK_TO_LATER_TROUGH_DRAWDOWN(C,{w})',[5,10,20,60]),
    ('ulcer_index','窗口内回撤均方根','sqrt(MEAN_WITHIN_WINDOW((C/RUNNING_PEAK(C)-1)^2,{w}))',[5,10,20,60]),
    ('volatility_of_volatility','波动率变化程度','STD(STD(r,5),{w})',[5,10,20,60]),
    ('large_move_share','大幅波动日期比例','MEAN(I(abs(r)>3*LAG(STD(r,20),1)),{w})',[5,10,20,60]),
    ('overnight_q05','隔夜收益左尾','QUANTILE(ro,0.05,{w})',[5,10,20,60]),
]
for stem,label,formula,windows in risk_templates:
    series('R','风险形状与极端事件',stem,label,windows,formula,'daily+adj_factor','刻画风险形状；方向不预设为必然负向或正向，部分仅用于控制风险。')

# L: 7 inherited + 53 new.
for name,label,formula,source in [
    ('log_volume','成交量对数','log(volume)','daily.vol'),
    ('source_volume_ratio','数据源量比','daily_basic.volume_ratio','daily_basic.volume_ratio'),
    ('free_turnover_ratio','自由流通换手与总换手比','turnover_rate_f/turnover_rate','daily_basic')]:
    add('L','当前活跃度',name,label,formula,source,'识别交易活跃程度和自由流通结构。')
liquidity_templates = [
    ('log_mean_amount','平均成交额对数','log(MEAN(amount_yuan,{w}))'),
    ('amount_cv','成交额变异系数','STD(amount_yuan,{w})/MEAN(amount_yuan,{w})'),
    ('log_volume_std','成交量对数波动','STD(log(volume),{w})'),
    ('amihud_proxy','Amihud非流动性代理','MEAN(abs(r)/amount_yuan,{w})*100000000'),
    ('turnover_mean','平均换手率','MEAN(turnover_rate,{w})'),
    ('turnover_std','换手率波动','STD(turnover_rate,{w})'),
    ('turnover_surprise','当日换手相对前期均值','turnover_rate/LAG(MEAN(turnover_rate,{w}),1)-1'),
    ('zero_return_share','零收盘收益日期比例','MEAN(I(raw_close=raw_pre_close),{w})'),
    ('volume_hhi','成交量时间集中度','SUM(volume^2,{w})/SUM(volume,{w})^2'),
    ('amount_normalized_slope','成交额趋势','OLS_SLOPE(amount_yuan,time,{w})/MEAN(amount_yuan,{w})'),
]
for stem,label,formula in liquidity_templates:
    series('L','流动性水平与变化',stem,label,[5,10,20,60,120],formula,'daily+daily_basic+adj_factor','检验流动性状态、交易冲击代理和短期收益的关系；日线代理不等于真实买卖价差。')

# P: 1 inherited + 54 new.
price_daily = [
    ('close_location','收盘在日内区间的位置','(C-L)/(H-L)'),
    ('upper_shadow','上影线占比','(H-max(O,C))/(H-L)'),
    ('lower_shadow','下影线占比','(min(O,C)-L)/(H-L)'),
    ('signed_body','有符号实体占比','(C-O)/(H-L)'),
    ('gap_to_true_range','开盘跳空相对真实振幅','(O-LAG(C,1))/TR'),
    ('close_vs_previous_high','收盘相对前日高点','(C-LAG(H,1))/LAG(TR,1)'),
]
for name,label,formula in price_daily:
    add('P','单日K线形态',name,label,formula,'daily+adj_factor','用开高低收区分价格形态，不推断高低点发生顺序。')
price_templates = [
    ('close_location_mean','平均收盘位置','MEAN(close_location,{w})'),
    ('upper_shadow_mean','平均上影线','MEAN(upper_shadow,{w})'),
    ('lower_shadow_mean','平均下影线','MEAN(lower_shadow,{w})'),
    ('breakout_distance','相对前期高点突破距离','(C-LAG(MAX(H,{w}),1))/MEAN(TR,20)'),
    ('breakdown_distance','相对前期低点距离','(C-LAG(MIN(L,{w}),1))/MEAN(TR,20)'),
    ('return_volume_corr','收益与成交量变化相关','CORR(r,log(volume/LAG(volume,1)),{w})'),
    ('up_volume_share','上涨日期成交量占比','SUM(volume*I(r>0),{w})/SUM(volume,{w})'),
    ('close_pressure','收盘位置成交额加权','SUM((2*close_location-1)*amount_yuan,{w})/SUM(amount_yuan,{w})'),
]
for stem,label,formula in price_templates:
    series('P','跨日形态与价量配合',stem,label,[3,5,10,20,40,60],formula,'daily+adj_factor','检验突破、拒绝上涨/下跌及量价配合；非分钟路径。')

# F: 6 inherited + 44 new.
for name,label,formula in [
    ('large_buy_share','大单及超大单买入占比','(buy_lg+buy_elg)/buy_all'),
    ('extra_large_buy_share','超大单买入占比','buy_elg/buy_all'),
    ('order_size_hhi','订单规模成交结构集中度','SUM_ACROSS_SIZE(((buy_size+sell_size)/(buy_all+sell_all))^2)'),
    ('large_buy_sell_logratio','大单买卖金额对数比','log((buy_lg+buy_elg)/(sell_lg+sell_elg))')]:
    add('F','订单规模结构',name,label,formula,'moneyflow.*_amount','检验订单规模代理；不能认定大单交易者就是机构。')
flow_templates = [
    ('large_net_mean','大单净流入比例均值','MEAN(large_order_net_ratio,{w})'),
    ('large_net_std','大单净流入比例波动','STD(large_order_net_ratio,{w})'),
    ('large_net_positive_share','大单净流入为正的日期比例','MEAN(I(large_order_net_ratio>0),{w})'),
    ('large_flow_return_corr','大单资金流与收益相关','CORR(large_order_net_ratio,r,{w})'),
    ('large_flow_slope','大单净流入比例趋势','OLS_SLOPE(large_order_net_ratio,time,{w})'),
    ('aggregate_net_flow_ratio','期间总净流入占成交额','SUM(net_flow_yuan,{w})/SUM(amount_yuan,{w})'),
    ('aggregate_large_buy_share','期间大单买入占比','SUM(buy_lg+buy_elg,{w})/SUM(buy_all,{w})'),
    ('gross_order_imbalance_std','买卖金额不平衡波动','STD((buy_all-sell_all)/(buy_all+sell_all),{w})'),
]
for stem,label,formula in flow_templates:
    series('F','资金流持续性与价格响应',stem,label,[5,10,20,40,60],formula,'moneyflow+daily+adj_factor','检验持续资金压力和价格响应；先检查字段口径、缺失与覆盖。')

# X: 2 inherited + 43 new. Historical industry labels deliberately excluded.
for name,label,formula in [('market_beta','市场Beta','COV(r,rm,{w})/VAR(rm,{w})'),('market_corr','市场相关性','CORR(r,rm,{w})')]:
    series('X','市场暴露',name,label,[20,120],formula,'daily+adj_factor，同日可观察股票池','刻画随市场波动的程度。')
relative_templates = [
    ('beta_adjusted_return','固定事前Beta调整收益','SUM(r,{w})-LAG(beta60,{w})*SUM(rm,{w})',[5,10,20,60,120]),
    ('residual_momentum','逐日事前市场残差累计','SUM(e,{w})',[5,10,20,60,120]),
    ('residual_volatility','市场残差波动','STD(e,{w})',[5,10,20,60,120]),
    ('residual_positive_share','市场残差为正比例','MEAN(I(e>0),{w})',[5,10,20,60,120]),
    ('up_market_capture','上涨市场收益捕获率','SUM_IF(r,rm>0,{w})/SUM_IF(rm,rm>0,{w})',[20,40,60,120,250]),
    ('down_market_capture','下跌市场收益捕获率','SUM_IF(r,rm<0,{w})/SUM_IF(rm,rm<0,{w})',[20,40,60,120,250]),
    ('residual_ts_rank','当日特质冲击历史分位','TSRANK(e,{w})',[5,10,20,60,120]),
]
for stem,label,formula,windows in relative_templates:
    series('X','个股特质表现',stem,label,windows,formula,'daily+adj_factor；事前滚动市场回归','分离共同市场暴露和个股特质信号；不把同日减去市场常数视作新增排序信息。')
series('X','市场活跃度敏感性','market_activity_corr','个股收益与市场换手相关',[20,40,60,120],'CORR(r,market_median_turnover,{w})','daily+daily_basic+adj_factor','识别个股对全市场交易活跃度的敏感性。')

# Q: 8 inherited + 37 new. PIT is a hard gate, not an assumption of availability.
PIT='待实现；财务历史版本、可见公告日和同口径季度匹配必须先审计'
for name,label,formula in [
    ('netprofit_margin','销售净利率','PIT(netprofit_margin)'),
    ('assets_turn','总资产周转率','PIT(assets_turn)'),
    ('quick_ratio','速动比率','PIT(quick_ratio)'),
    ('cash_profitability_gap','现金质量与会计利润差','Q(ocf_to_or)-Q(netprofit_margin)')]:
    add('Q','补充质量指标',name,label,formula,'fina_indicator','检验盈利质量、经营效率及偿债能力。',PIT)
for field in BASE_FEATURES[42:50]:
    for lag in (1,4):
        add('Q','财报变化',f'{field}_report_delta_{lag}q',f'{field}较前{lag}季度报告变化',
            f'PIT({field},q)-PIT({field},q-{lag})','fina_indicator+历史公告版本',
            '同季度/跨季度变化均作候选；保持同一指标口径，不把累计值直接当单季值。',PIT)
for field in ('roe','roa','grossprofit_margin','ocf_to_or'):
    add('Q','财务稳定性',f'{field}_report_std_8q',f'{field}八季度波动',f'STD_LAST_REPORTS(PIT({field}),8)',
        'fina_indicator+历史公告版本','检验质量稳定性，要求8个连续不同报告期。',PIT)
    add('Q','财务趋势',f'{field}_report_slope_4q',f'{field}四季度趋势',f'SLOPE_LAST_REPORTS(PIT({field}),4)',
        'fina_indicator+历史公告版本','检验质量改善趋势，要求4个连续不同报告期。',PIT)
for field in ('netprofit_yoy','or_yoy'):
    add('Q','成长持续性',f'{field}_positive_share_8q',f'{field}连续报告期正增长比例',f'MEAN_LAST_REPORTS(I(PIT({field})>0),8)',
        'fina_indicator+历史公告版本','区分一次性增长和持续增长。',PIT)
add('Q','信息新鲜度','financial_report_age','最近财务公告距今天数','calendar_days(t-latest_visible_ann_date)',
    'fina_indicator.ann_date','控制陈旧财务信息，不将缺失直接当作新公告。',PIT)
for name,label,formula in [
    ('quality_growth_product','盈利与成长配合','Q(roe)*Q(netprofit_yoy)'),
    ('cash_growth_product','现金质量与营收成长配合','Q(ocf_to_or)*Q(or_yoy)'),
    ('leverage_adjusted_profitability','杠杆条件下的盈利','Q(roe)/(1+abs(Q(debt_to_assets)))'),
    ('profitability_disagreement','ROE与ROA排序分歧','abs(Q(roe)-Q(roa))'),
    ('margin_stability_ratio','毛利率水平相对波动','MEAN_LAST_REPORTS(PIT(grossprofit_margin),4)/STD_LAST_REPORTS(PIT(grossprofit_margin),4)'),
    ('cash_stability_ratio','经营现金收入比水平相对波动','MEAN_LAST_REPORTS(PIT(ocf_to_or),4)/STD_LAST_REPORTS(PIT(ocf_to_or),4)')]:
    add('Q','质量内部组合',name,label,formula,'fina_indicator+历史公告版本','财务方向内部非线性组合；不作为新的独立数据来源。',PIT)

# V: 5 inherited + 35 new.
for name,label,formula in [
    ('dividend_yield_ttm','滚动股息率','daily_basic.dv_ttm'),
    ('float_cap_share','流通市值占总市值比例','circ_mv/total_mv'),
    ('negative_earnings_flag','亏损估值状态','I(pe_ttm<0)')]:
    add('V','估值状态与结构',name,label,formula,'daily_basic','补充估值状态；亏损公司不使用正PE重估公式。')
valuation_templates = [
    ('ep_ts_z','盈利收益率历史标准分','TSZ(earnings_yield,{w})'),
    ('bp_ts_z','账面市值比历史标准分','TSZ(book_to_price,{w})'),
    ('sp_ts_z','销售市值比历史标准分','TSZ(sales_yield,{w})'),
    ('dividend_ts_z','股息率历史标准分','TSZ(dividend_yield_ttm,{w})'),
    ('ep_ts_rank','盈利收益率历史分位','TSRANK(earnings_yield,{w})'),
    ('bp_ts_rank','账面市值比历史分位','TSRANK(book_to_price,{w})'),
    ('sp_ts_rank','销售市值比历史分位','TSRANK(sales_yield,{w})'),
    ('positive_pe_rerating','正市盈率重估','log(pe_ttm/LAG(pe_ttm,{w})); both PE>0'),
]
for stem,label,formula in valuation_templates:
    series('V','估值历史位置与变化',stem,label,[20,60,120,250],formula,'daily_basic；历史值可用性需复核','提供相对自身历史估值状态；不把低估值直接等同于未来5日上涨。')

# C: 35 focused filters, not 450 mechanically expanded combinations.
selectors = dict(volume='volume',amount='amount_yuan',turnover='turnover_rate',
                 large_buy='buy_lg+buy_elg',large_net='buy_lg+buy_elg-sell_lg-sell_elg')
for key,expr in selectors.items():
    for w in (10,20,60):
        for stat,label,formula in [('sum','高条件日期累计收益','SUM_SELECTED(r,S)'),
                                   ('downside_rms','高条件日期下行均方根','sqrt(MEAN_SELECTED(min(r,0)^2,S))')]:
            add('C','高活跃日期条件收益',f'cond_{key}_high_close_{stat}_{w}d',f'{key}最高半数日期{label}（{w}日）',
                f'S=TOP_HALF({expr},{w}); {formula}','daily+moneyflow+daily_basic+adj_factor',
                '检验活跃日期收益是否更有信息；不预先认定其余日期都是噪声。',
                '旧条件库有同名实现；需与新数据契约核对后复用')
    add('C','条件下的收益结构',f'cond_{key}_overnight_intraday_corr_20d',f'{key}最高10天隔夜与日内相关',
        f'S=TOP_HALF({expr},20); CORR_SELECTED(ro,ri,S)','daily+moneyflow+daily_basic+adj_factor',
        '检验活跃日期隔夜冲击被日内延续或抵消。')

# I: 25 explicit economic hypotheses, one primary direction each.
pairs = [
 ('logret_5d','turnover_mean_20d','短期收益×换手水平'),
 ('logret_20d','turnover_surprise_20d','动量×换手冲击'),
 ('logret_20d','amihud_proxy_20d','动量×非流动性'),
 ('logret_5d','log_mean_amount_20d','短期收益×成交容量'),
 ('path_efficiency_20d','amount_cv_20d','趋势效率×成交不稳定性'),
 ('logret_20d','volatility_20d','动量×波动状态'),
 ('logret_5d','downside_volatility_20d','短期收益×下行风险'),
 ('overnight_logret','turnover_rate','隔夜冲击×换手'),
 ('close_location','large_order_net_ratio','收盘位置×大单资金压力'),
 ('upper_shadow','large_net_mean_20d','上影线×持续资金流'),
 ('lower_shadow','aggregate_large_buy_share_20d','下影线×大单买入占比'),
 ('breakout_distance_20d','turnover_surprise_20d','突破×换手冲击'),
 ('breakout_distance_20d','large_net_positive_share_20d','突破×资金流持续性'),
 ('logret_20d','large_flow_slope_20d','动量×大单净流入趋势'),
 ('residual_momentum_20d','large_net_mean_20d','特质动量×大单资金流'),
 ('residual_volatility_20d','log_circ_mv','特质波动×流通规模'),
 ('market_beta_60d','amihud_proxy_20d','市场暴露×非流动性'),
 ('earnings_yield','roe','估值×盈利质量'),
 ('book_to_price','debt_to_assets','账面估值×杠杆'),
 ('sales_yield','grossprofit_margin','收入估值×毛利率'),
 ('logret_20d','earnings_yield','动量×估值'),
 ('logret_20d','netprofit_yoy','动量×利润成长'),
 ('log_circ_mv','turnover_surprise_20d','流通规模×交易冲击'),
 ('dividend_yield_ttm','volatility_20d','股息×风险状态'),
 ('financial_report_age','logret_5d','公告信息新鲜度×短期收益'),
]
for i,(a,b,label) in enumerate(pairs,1):
    add('I','跨方向两两交互',f'interaction_{i:02d}',label,f'Q({a})*Q({b})',f'{a}; {b}',
        '让线性模型表达状态依赖；交互先验不等于有效证据，依赖父因子可用性。',
        '待实现；若依赖财务字段则继承PIT审计门槛')

counts=Counter(r['family'] for r in rows)
assert counts == Counter({k:v[1] for k,v in FAMILIES.items()}), counts
assert len(rows)==len({r['name'] for r in rows})==500
assert len({r['formula'] for r in rows})==500, 'Identical textual formulas'
known={r['name'] for r in rows}
assert all(a in known and b in known for a,b,_ in pairs)
assert set(BASE_FEATURES) <= known
rows.sort(key=lambda r:list(FAMILIES).index(r['family']))
for i,r in enumerate(rows,1): r['id']=f'F{i:03d}'
for r in rows:
    financial_names={item['name'] for item in rows if item['family']=='Q'}
    gated=r['family']=='Q' or (r['family']=='I' and any(x in financial_names for x in r['source'].split('; ')))
    r['status']='隔离：历史财务PIT及报告口径待验证' if gated else '已实现；通过样本审计不等于全市场认证'
    if r['name'] in {'amount_ratio_5d','amount_ratio_20d','volume_ratio_20d','turnover_zscore_20d','net_mf_ratio_5d'}:
        r['status']='V2审计修复：按连续市场交易日重算，停牌缺口使完整窗口失效'
        r['source']='v2/balanced_features.py；V1快照保留不变'
    if r['name']=='negative_earnings_flag':
        r['formula']='GATED: PIT_TTM_PROFIT_SIGN_REQUIRED'
        r['status']='隔离：数据源亏损PE为空，不能用pe_ttm<0识别；缺失也不直接等同亏损'
        r['source']='待历史可用TTM净利润；当前输出NULL'
payload=dict(version='balanced_500_audit_repair_v2',status='implemented_with_51_gates_universe_repair_pending',
    target='existing 5-trading-day adjusted next-open to exit-close log return',
    allocation_basis='research priority for this project; not empirically validated importance or portfolio weights',
    families={k:dict(name=v[0],count=v[1],priority=v[2],rationale=v[3]) for k,v in FAMILIES.items()},
    factors=rows)
(HERE/'factor_catalog_500.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 按研究方向重新设计的500个候选因子','',
       '**状态：审计修复版，500候选中51项隔离；历史股票池补采未完成。旧450个filter组合不是本清单。**','',
       '本清单包含原50个因子；新增450个分散到十个方向。数量是本项目初始研究预算，不是重要性实证结论。',
       '符号、缺失值、复权、公告时点和来源规则见同目录README.md。','',
       '| 方向 | 数量 | 优先级 | 分配理由 |','|---|---:|---|---|']
for k,(name,n,priority,why) in FAMILIES.items(): lines.append(f'| {k} {name} | {n} | {priority} | {why} |')
lines += ['| 合计 | 500 | | |','']
for family,(name,n,priority,why) in FAMILIES.items():
    group=[r for r in rows if r['family']==family]
    dims=Counter(r['dimension'] for r in group)
    lines += [f'## {family} {name}：{n}个','',why,'',
              '内部维度：'+'；'.join(f'{d} {c}个' for d,c in dims.items())+'。','',
              '| 编号 | 因子名 | 含义 | 公式/精确定义 | 数据/实现状态 |',
              '|---|---|---|---|---|']
    for r in group:
        lines.append(f"| {r['id']} | `{r['name']}` | {r['label']} | `{r['formula']}` | {r['source']}；{r['status']} |")
    lines.append('')
(HERE/'FACTOR_CATALOG_500.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(total=len(rows),families=counts,inherited_base=50,added=450,unique_names=True,
                     unique_textual_formulas=True,interaction_references_valid=True),ensure_ascii=False,indent=2))
