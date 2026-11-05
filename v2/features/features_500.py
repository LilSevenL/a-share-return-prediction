"""Ordered 500-feature contract and trailing conditional-return calculations."""
from __future__ import annotations

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import json
import sys
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT))
from build_features import FACTOR_COLUMNS as BASE_FEATURES

WINDOWS = (10, 20, 40, 60, 120)
SELECTORS = {
    'volume': ('成交量', 'daily.vol'),
    'amount': ('成交额', 'daily.amount'),
    'turnover': ('换手率', 'daily_basic.turnover_rate (inherited V1 column)'),
    'large_buy': ('大单及超大单合计买入额', 'moneyflow.buy_lg_amount + moneyflow.buy_elg_amount'),
    'large_net': ('大单及超大单合计净流入额', 'moneyflow.buy_lg_amount + moneyflow.buy_elg_amount - moneyflow.sell_lg_amount - moneyflow.sell_elg_amount'),
}
RETURNS = {
    'close': ('收盘到收盘复权对数收益', 'logret_1d'),
    'intraday': ('开盘到收盘复权对数收益', 'intraday_logret'),
    'overnight': ('前收盘到当日开盘复权对数收益', 'overnight_logret'),
}
STATS = {
    'sum': ('累计对数收益', 'sum(r[d] for d in S)'),
    'std': ('样本标准差', 'sqrt(sum((r[d]-mean(r[S]))^2 for d in S)/(k-1))'),
    'downside_rms': ('下行均方根', 'sqrt(sum(min(r[d],0)^2 for d in S)/k)'),
}
BASE_DESCRIPTIONS = {
    **{f'logret_{w}d': f'复权收盘价的{w}个连续市场交易日对数收益。' for w in (1,2,3,5,10,20,60)},
    **{f'close_ma{w}_gap': f'复权收盘价/过去{w}日复权收盘均价−1，要求完整连续窗口。' for w in (5,20,60)},
    'ma5_ma20_gap':'过去5日复权收盘均价/过去20日均价−1。',
    'ma20_ma60_gap':'过去20日复权收盘均价/过去60日均价−1。',
    **{f'volatility_{w}d': f'过去{w}日单日对数收益样本标准差，要求{w}个有效收益。' for w in (5,10,20,60)},
    'downside_volatility_20d':'过去20日收益负值平方的均值开根号，正收益记0。',
    'return_skew_20d':'过去20日单日对数收益偏度。',
    'worst_logret_20d':'过去20日最小单日对数收益。',
    'intraday_logret':'ln(当日复权收盘价/当日复权开盘价)。',
    'overnight_logret':'ln(当日复权开盘价/上一市场交易日复权收盘价)。',
    'high_low_range':'(当日复权最高价−复权最低价)/复权开盘价。',
    'market_beta_60d':'过去60日股票收益对当日全市场股票等权收益的回归斜率。',
    'market_corr_60d':'过去60日股票收益与当日全市场股票等权收益的相关系数。',
    'log_amount':'当日正成交额的自然对数。',
    'amount_ratio_5d':'当日成交额/过去5日平均成交额−1。',
    'amount_ratio_20d':'当日成交额/过去20日平均成交额−1。',
    'volume_ratio_20d':'当日成交量/过去20日平均成交量−1。',
    'turnover_rate':'当日换手率，沿用数据源原单位。',
    'turnover_rate_f':'当日自由流通股换手率，沿用数据源原单位。',
    'turnover_zscore_20d':'(当日换手率−过去20日均值)/过去20日样本标准差。',
    'log_total_mv':'正总市值的自然对数。',
    'log_circ_mv':'正流通市值的自然对数。',
    'earnings_yield':'1/pe_ttm，绝对分母须大于1e−12，保留负估值。',
    'book_to_price':'1/pb，绝对分母须大于1e−12。',
    'sales_yield':'1/ps_ttm，绝对分母须大于1e−12。',
    'net_mf_ratio':'10×净资金流入额/成交额；10用于万元与千元单位换算。',
    'large_order_net_ratio':'10×(大单及超大单买入额−卖出额)/成交额。',
    'extra_large_order_net_ratio':'10×(超大单买入额−卖出额)/成交额。',
    'medium_order_net_ratio':'10×(中单买入额−卖出额)/成交额。',
    'small_order_net_ratio':'10×(小单买入额−卖出额)/成交额。',
    'net_mf_ratio_5d':'过去5日net_mf_ratio平均值。',
    **{k: f'最近可用财务公告的{v}；公告日严格早于信号日，保留V1公告选择及数据源单位。'
       for k,v in [('roe','净资产收益率'),('roa','总资产报酬率'),('grossprofit_margin','毛利率'),
                   ('debt_to_assets','资产负债率'),('current_ratio','流动比率'),
                   ('netprofit_yoy','净利润同比增长率'),('or_yoy','营业收入同比增长率'),
                   ('ocf_to_or','经营现金净流量/营业收入')]},
}
assert set(BASE_DESCRIPTIONS) == set(BASE_FEATURES)


def specs():
    for window in WINDOWS:
        for selector in SELECTORS:
            for side in ('high', 'low'):
                for returns in RETURNS:
                    for statistic in STATS:
                        yield dict(name=f'cond_{selector}_{side}_{returns}_{statistic}_{window}d',
                                   window=window, selector=selector, side=side,
                                   returns=returns, statistic=statistic, selected_days=window//2)


NEW_SPECS = list(specs())
NEW_FEATURES = [s['name'] for s in NEW_SPECS]
FEATURE_COLUMNS = list(BASE_FEATURES) + NEW_FEATURES
assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 500


def conditional_features(market_index, selectors, returns):
    """One stock, strictly date-sorted, returns float64 [n,450].

    Window includes signal day; signal is available after close for next-open use.
    Require a full consecutive-market-session window, all selector observations,
    and all k selected return observations. Never fill missing flows with zero.
    Stable ascending ordering breaks ties by date: older to low, newer to high.
    An entirely tied selector window yields missing, not an arbitrary signal.
    """
    index = np.asarray(market_index)
    n = len(index)
    if n and (not np.isfinite(index).all() or np.any(np.diff(index) <= 0)):
        raise ValueError('market_index must be finite and strictly increasing')
    if any(len(a) != n for a in [*selectors.values(), *returns.values()]):
        raise ValueError('Input length mismatch')
    output = np.full((n, 450), np.nan, dtype=np.float64)
    col = 0
    for window in WINDOWS:
        if n < window:
            col += 90
            continue
        k = window//2
        calendar = sliding_window_view(index, window)
        continuous = calendar[:, -1] - calendar[:, 0] == window-1
        ret_windows = {key: sliding_window_view(np.asarray(returns[key], dtype=float), window) for key in RETURNS}
        for selector in SELECTORS:
            values = sliding_window_view(np.asarray(selectors[selector], dtype=float), window)
            valid = continuous & np.isfinite(values).all(axis=1)
            valid &= np.max(values, axis=1) > np.min(values, axis=1)
            order = np.argsort(values, axis=1, kind='stable')
            for side in ('high', 'low'):
                selected = order[:, -k:] if side == 'high' else order[:, :k]
                for key in RETURNS:
                    r = np.take_along_axis(ret_windows[key], selected, axis=1)
                    usable = valid & np.isfinite(r).all(axis=1)
                    calculated = (r.sum(axis=1), r.std(axis=1, ddof=1),
                                  np.sqrt(np.mean(np.minimum(r, 0)**2, axis=1)))
                    for values_out in calculated:
                        output[window-1:, col] = np.where(usable & np.isfinite(values_out), values_out, np.nan)
                        col += 1
    assert col == 450
    return output


def export_catalog(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = [dict(number=i+1, name=name, family='V1原有因子',
                    definition=BASE_DESCRIPTIONS[name],
                    source='build_features.py / quant_data/processed/features_50')
               for i, name in enumerate(BASE_FEATURES)]
    for i, s in enumerate(NEW_SPECS, 51):
        selection = f"过去{s['window']}个连续市场交易日（含当日）按{SELECTORS[s['selector']][0]}排序，取{'最高' if s['side']=='high' else '最低'}{s['selected_days']}天"
        definition = selection + '，计算' + RETURNS[s['returns']][0] + '的' + STATS[s['statistic']][0]
        records.append(dict(number=i, **s, family='条件收益因子', definition=definition,
                            formula=STATS[s['statistic']][1],
                            selector_source=SELECTORS[s['selector']][1], return_source=RETURNS[s['returns']][1]))
    payload = dict(feature_count=500, original=50, added=450,
                   signal_timing='after close; execution no earlier than next open',
                   missing_policy='full consecutive window; all selector values; all selected returns; otherwise null',
                   tie_policy='ascending selector then ascending date; low first k, high last k; all equal -> null',
                   factors=records)
    (directory/'feature_catalog_500.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# V2 全部500个候选因子', '',
             '编号1–50保留V1定义；51–500为新增条件因子。窗口含信号日，仅供收盘后预测下一交易日使用。', '',
             '新增因子：5个窗口 × 5个筛选条件 × 2组 × 3类收益 × 3种统计量 = 450。', '',
             '完整窗口及选中收益缺失时输出空值；条件全相同时输出空值。并列条件按日期先后排序，低组取前半，高组取后半。', '',
             '累计收益使用选中日对数收益之和；标准差分母k−1；下行均方根为sqrt(mean(min(r,0)^2))。', '',
             '| 编号 | 因子名 | 定义 |', '|---:|---|---|']
    for r in records:
        lines.append(f"| {r['number']} | `{r['name']}` | {r['definition']} |")
    (directory/'FEATURE_CATALOG_500.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return payload


if __name__ == '__main__':
    export_catalog(_REPO_ROOT/'configs/v2')
    print('Exported exactly 500 feature definitions.')
