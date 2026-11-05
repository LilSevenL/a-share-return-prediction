"""Numerical implementation of the balanced catalog. No targets are accepted.

Financial factors are deliberately gated until historical revision provenance
and comparable quarterly definitions are audited. No synthetic replacement.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
try:
    from .stable_rolling import kurtosis as stable_kurtosis, correlation as stable_correlation, trend_r2 as stable_trend_r2
except ImportError:
    from stable_rolling import kurtosis as stable_kurtosis, correlation as stable_correlation, trend_r2 as stable_trend_r2
try:
    from .features_500 import BASE_FEATURES, conditional_features, NEW_FEATURES
except ImportError:
    from features_500 import BASE_FEATURES, conditional_features, NEW_FEATURES

CATALOG=json.loads((Path(__file__).parent/'research_design/factor_catalog_500.json').read_text(encoding='utf-8'))
RECORDS=CATALOG['factors']; COLUMNS=[r['name'] for r in RECORDS]
FINANCIAL_NAMES={r['name'] for r in RECORDS if r['family']=='Q'}
GATED_NAMES=FINANCIAL_NAMES|{'negative_earnings_flag'}


def div(a,b):
    return a/b.replace(0,np.nan) if isinstance(b,pd.Series) else a/(b if b!=0 else np.nan)


def log(x): return np.log(x.where(x>0))


def roll(s,w,op):
    r=s.rolling(w,min_periods=w)
    if op=='std': return r.std(ddof=1)
    if op=='kurt': return stable_kurtosis(s,w)
    if op=='rank': return (r.rank(method='average')-1)/(w-1)
    return getattr(r,op)()


def slope(s,w):
    t=np.arange(w,dtype=float); t-=t.mean()
    return s.rolling(w,min_periods=w).apply(lambda x: np.dot(x,t)/np.dot(t,t),raw=True)


def streak(x):
    longest=current=0
    for value in x:
        current=current+1 if value else 0
        longest=max(current,longest)
    return longest


def cross_rank(frame,name):
    grouped=frame.groupby('signal_date')[name]
    count=grouped.transform('count')
    return ((grouped.rank(method='average')-.5)/count-.5).where(count>=20)


def technical_stock(frame):
    """A single stock reindexed onto the complete market calendar.

    Reindexing makes missing sessions explicit and all rolling operators strict.
    Five legacy liquidity/flow windows are recomputed on the market calendar.
    Other inherited V1 factors remain supplied values; no ffill of prices/flows.
    """
    d=frame.copy(); out={k:d[k].copy() for k in BASE_FEATURES}
    O,H,L,C=[d[k] for k in ('O','H','L','C')]
    r=log(C/C.shift()); ri=log(C/O); ro=log(O/C.shift())
    vol=d.volume; amount=d.amount_yuan; turn=d.turnover_rate
    tr=pd.concat([H-L,(H-C.shift()).abs(),(L-C.shift()).abs()],axis=1).max(axis=1,skipna=False)
    R=lambda s,w,op:roll(s,w,op)
    flag=lambda condition,valid:condition.astype(float).where(valid)
    # Audit repair: an observation-row window can bridge suspensions. Require
    # full market-day windows here, exactly as for newly introduced factors.
    for w in (5,20):out[f'amount_ratio_{w}d']=div(amount,R(amount,w,'mean'))-1
    out['volume_ratio_20d']=div(vol,R(vol,20,'mean'))-1
    out['turnover_zscore_20d']=div(turn-R(turn,20,'mean'),R(turn,20,'std'))
    out['net_mf_ratio_5d']=R(d.net_mf_ratio,5,'mean')
    for w in (15,40,120,250): out[f'logret_{w}d']=R(r,w,'sum')
    for w in (20,60,120): out[f'skip5_momentum_{w}d']=R(r,w-5,'sum').shift(5)
    for w in (5,10,20,60):
        sl=slope(log(C),w)
        variance=R(log(C),w,'var')
        time_var=np.var(np.arange(w),ddof=1)
        out.update({
          f'trend_slope_{w}d':sl,
          f'trend_r2_{w}d':stable_trend_r2(log(C),w),
          f'path_efficiency_{w}d':div(R(r,w,'sum').abs(),R(r.abs(),w,'sum')),
          f'return_acceleration_{w}d':R(r,w,'mean')-R(r,w,'mean').shift(w),
          f'up_day_share_{w}d':R(flag(r>0,r.notna()),w,'mean'),
          f'longest_up_streak_{w}d':flag(r>0,r.notna()).rolling(w,min_periods=w).apply(streak,raw=True)/w,
          f'longest_down_streak_{w}d':flag(r<0,r.notna()).rolling(w,min_periods=w).apply(streak,raw=True)/w,
          f'return_autocorr_{w}d':r.rolling(w,min_periods=w).corr(r.shift()),
          f'overnight_intraday_spread_{w}d':R(ro-ri,w,'sum'),
          f'overnight_intraday_cov_{w}d':ro.rolling(w,min_periods=w).cov(ri),
          f'overnight_reversal_share_{w}d':R(flag(ro*ri<0,ro.notna()&ri.notna()),w,'mean'),
          f'intraday_abs_share_{w}d':div(R(ri.abs(),w,'sum'),R(ri.abs()+ro.abs(),w,'sum')),
          f'slope_change_{w}d':sl-sl.shift(w),
          f'direction_switch_share_{w}d':R(flag(r*r.shift()<0,r.notna()&r.shift().notna()),w,'mean')})
    for w in (20,60,120): out[f'return_ts_rank_{w}d']=R(r,w,'rank')
    for w in (40,120): out[f'volatility_{w}d']=R(r,w,'std')
    for w in (5,10,20,60):
        out.update({
          f'upside_rms_{w}d':np.sqrt(R(r.clip(lower=0)**2,w,'mean')),
          f'return_kurtosis_{w}d':R(r,w,'kurt'),
          f'return_mad_{w}d':r.rolling(w,min_periods=w).apply(lambda x:np.median(np.abs(x-np.median(x))),raw=True),
          f'return_q05_{w}d':r.rolling(w,min_periods=w).quantile(.05),
          f'return_lower_quartile_mean_{w}d':r.rolling(w,min_periods=w).apply(lambda x:np.sort(x)[:math.ceil(w/4)].mean(),raw=True),
          f'best_return_{w}d':R(r,w,'max'),
          f'parkinson_vol_{w}d':np.sqrt(R(log(H/L)**2,w,'mean')/(4*np.log(2))),
          f'garman_klass_vol_{w}d':np.sqrt(R(.5*log(H/L)**2-(2*np.log(2)-1)*ri**2,w,'mean').clip(lower=0)),
          f'maximum_drawdown_{w}d':C.rolling(w,min_periods=w).apply(lambda x:np.max(1-x/np.maximum.accumulate(x)),raw=True),
          f'ulcer_index_{w}d':C.rolling(w,min_periods=w).apply(lambda x:np.sqrt(np.mean((x/np.maximum.accumulate(x)-1)**2)),raw=True),
          f'volatility_of_volatility_{w}d':R(R(r,5,'std'),w,'std'),
          f'large_move_share_{w}d':R(flag(r.abs()>3*R(r,20,'std').shift(),r.notna()&R(r,20,'std').shift().notna()),w,'mean'),
          f'overnight_q05_{w}d':ro.rolling(w,min_periods=w).quantile(.05)})
    for w in (5,10,60,120): out[f'downside_rms_{w}d']=np.sqrt(R(r.clip(upper=0)**2,w,'mean'))
    out.update(log_volume=log(vol),source_volume_ratio=d.source_volume_ratio,
               free_turnover_ratio=div(d.turnover_rate_f,turn))
    for w in (5,10,20,60,120):
        out.update({
         f'log_mean_amount_{w}d':log(R(amount,w,'mean')),
         f'amount_cv_{w}d':div(R(amount,w,'std'),R(amount,w,'mean')),
         f'log_volume_std_{w}d':R(log(vol),w,'std'),
         f'amihud_proxy_{w}d':R(div(r.abs(),amount),w,'mean')*1e8,
         f'turnover_mean_{w}d':R(turn,w,'mean'),f'turnover_std_{w}d':R(turn,w,'std'),
         f'turnover_surprise_{w}d':div(turn,R(turn,w,'mean').shift())-1,
         f'zero_return_share_{w}d':R(flag(d.raw_close==d.raw_pre_close,d.raw_close.notna()&d.raw_pre_close.notna()),w,'mean'),
         f'volume_hhi_{w}d':div(R(vol**2,w,'sum'),R(vol,w,'sum')**2),
         f'amount_normalized_slope_{w}d':div(slope(amount,w),R(amount,w,'mean'))})
    cl=div(C-L,H-L); upper=div(H-pd.concat([O,C],axis=1).max(axis=1,skipna=False),H-L)
    lower=div(pd.concat([O,C],axis=1).min(axis=1,skipna=False)-L,H-L)
    out.update(close_location=cl,upper_shadow=upper,lower_shadow=lower,signed_body=div(C-O,H-L),
               gap_to_true_range=div(O-C.shift(),tr),close_vs_previous_high=div(C-H.shift(),tr.shift()))
    for w in (3,5,10,20,40,60):
        out.update({
          f'close_location_mean_{w}d':R(cl,w,'mean'),f'upper_shadow_mean_{w}d':R(upper,w,'mean'),
          f'lower_shadow_mean_{w}d':R(lower,w,'mean'),
          f'breakout_distance_{w}d':div(C-R(H,w,'max').shift(),R(tr,20,'mean')),
          f'breakdown_distance_{w}d':div(C-R(L,w,'min').shift(),R(tr,20,'mean')),
          f'return_volume_corr_{w}d':stable_correlation(r,log(vol/vol.shift()),w) if w==3 else r.rolling(w,min_periods=w).corr(log(vol/vol.shift())),
          f'up_volume_share_{w}d':div(R(vol*flag(r>0,r.notna()),w,'sum'),R(vol,w,'sum')),
          f'close_pressure_{w}d':div(R((2*cl-1)*amount,w,'sum'),R(amount,w,'sum'))})
    buy=d[[f'buy_{s}_amount' for s in ('sm','md','lg','elg')]].sum(axis=1,min_count=4)
    sell=d[[f'sell_{s}_amount' for s in ('sm','md','lg','elg')]].sum(axis=1,min_count=4)
    big=d.buy_lg_amount+d.buy_elg_amount; big_sell=d.sell_lg_amount+d.sell_elg_amount
    hhi=sum(div(d[f'buy_{s}_amount']+d[f'sell_{s}_amount'],buy+sell)**2 for s in ('sm','md','lg','elg'))
    net=d.large_order_net_ratio
    out.update(large_buy_share=div(big,buy),extra_large_buy_share=div(d.buy_elg_amount,buy),
               order_size_hhi=hhi,large_buy_sell_logratio=log(div(big,big_sell)))
    for w in (5,10,20,40,60):
        out.update({
          f'large_net_mean_{w}d':R(net,w,'mean'),f'large_net_std_{w}d':R(net,w,'std'),
          f'large_net_positive_share_{w}d':R(flag(net>0,net.notna()),w,'mean'),
          f'large_flow_return_corr_{w}d':net.rolling(w,min_periods=w).corr(r),
          f'large_flow_slope_{w}d':slope(net,w),
          f'aggregate_net_flow_ratio_{w}d':div(R(d.net_mf_amount*10000,w,'sum'),R(amount,w,'sum')),
          f'aggregate_large_buy_share_{w}d':div(R(big,w,'sum'),R(buy,w,'sum')),
          f'gross_order_imbalance_std_{w}d':R(div(buy-sell,buy+sell),w,'std')})
    rm=d.rm
    beta60=div(r.rolling(60,min_periods=60).cov(rm),R(rm,60,'var'))
    alpha60=R(r,60,'mean')-beta60*R(rm,60,'mean')
    e=r-alpha60.shift()-beta60.shift()*rm
    for w in (20,120):
        out[f'market_beta_{w}d']=div(r.rolling(w,min_periods=w).cov(rm),R(rm,w,'var'))
        out[f'market_corr_{w}d']=r.rolling(w,min_periods=w).corr(rm)
    for w in (5,10,20,60,120):
        out.update({f'beta_adjusted_return_{w}d':R(r,w,'sum')-beta60.shift(w)*R(rm,w,'sum'),
          f'residual_momentum_{w}d':R(e,w,'sum'),f'residual_volatility_{w}d':R(e,w,'std'),
          f'residual_positive_share_{w}d':R(flag(e>0,e.notna()),w,'mean'),
          f'residual_ts_rank_{w}d':R(e,w,'rank')})
    for w in (20,40,60,120,250):
        for label,cond in [('up',rm>0),('down',rm<0)]:
            ok=rm.notna()&r.notna(); indicator=flag(cond,ok)
            out[f'{label}_market_capture_{w}d']=div(R(r*indicator,w,'sum'),R(rm*indicator,w,'sum')).where(R(indicator,w,'sum')>=3)
    for w in (20,40,60,120): out[f'market_activity_corr_{w}d']=r.rolling(w,min_periods=w).corr(d.market_median_turnover)
    out.update(dividend_yield_ttm=d.dv_ttm,float_cap_share=div(d.circ_mv,d.total_mv),
               negative_earnings_flag=pd.Series(np.nan,index=d.index))
    for w in (20,60,120,250):
        for prefix,s in [('ep',d.earnings_yield),('bp',d.book_to_price),('sp',d.sales_yield),('dividend',d.dv_ttm)]:
            out[f'{prefix}_ts_z_{w}d']=div(s-R(s,w,'mean'),R(s,w,'std'))
            if prefix!='dividend': out[f'{prefix}_ts_rank_{w}d']=R(s,w,'rank')
        out[f'positive_pe_rerating_{w}d']=log(div(d.pe_ttm,d.pe_ttm.shift(w))).where((d.pe_ttm>0)&(d.pe_ttm.shift(w)>0))
    selectors=dict(volume=vol.to_numpy(),amount=amount.to_numpy(),turnover=turn.to_numpy(),
                   large_buy=big.to_numpy(),large_net=(big-big_sell).to_numpy())
    ret=dict(close=r.to_numpy(),intraday=ri.to_numpy(),overnight=ro.to_numpy())
    conditional=conditional_features(np.arange(len(d)),selectors,ret)
    idx={name:i for i,name in enumerate(NEW_FEATURES)}
    for record in RECORDS:
        name=record['name']
        if record['family']=='C' and name in idx: out[name]=pd.Series(conditional[:,idx[name]],index=d.index)
    for key,s in selectors.items():
        values=np.full(len(d),np.nan)
        if len(d)>=20:
            sw=sliding_window_view(s,20); a=sliding_window_view(ro.to_numpy(),20); b=sliding_window_view(ri.to_numpy(),20)
            order=np.argsort(sw,axis=1,kind='stable')[:,-10:]
            aa=np.take_along_axis(a,order,axis=1); bb=np.take_along_axis(b,order,axis=1)
            valid=np.isfinite(sw).all(1)&(sw.max(1)>sw.min(1))&np.isfinite(aa).all(1)&np.isfinite(bb).all(1)
            aa=aa-aa.mean(1)[:,None]; bb=bb-bb.mean(1)[:,None]
            den=np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1))
            v=np.divide((aa*bb).sum(1),den,out=np.full(len(den),np.nan),where=den>0)
            values[19:]=np.where(valid,v,np.nan)
        out[f'cond_{key}_overnight_intraday_corr_20d']=pd.Series(values,index=d.index)
    # Do not activate stale/cumulative financial signals without provenance.
    for name in FINANCIAL_NAMES: out[name]=pd.Series(np.nan,index=d.index)
    for rec in RECORDS:
        if rec['family']=='I': out[rec['name']]=pd.Series(np.nan,index=d.index)
    missing=set(COLUMNS)-set(out)
    if missing: raise RuntimeError(f'Unimplemented factor names: {sorted(missing)}')
    result=pd.DataFrame({name:out[name] for name in COLUMNS},index=d.index).replace([np.inf,-np.inf],np.nan)
    return result


def finish_cross_section(frame):
    """Cross-sectional interactions calculated AFTER all stocks are combined."""
    blocked=[]
    for record in RECORDS:
        if record['family']=='I':
            a,b=record['source'].split('; ')
            if a in GATED_NAMES or b in GATED_NAMES:
                blocked.append(record['name'])
                frame[record['name']]=np.nan
            else:
                frame[record['name']]=cross_rank(frame,a)*cross_rank(frame,b)
    return frame,sorted(GATED_NAMES|set(blocked))
