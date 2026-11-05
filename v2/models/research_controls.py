"""Training-only preprocessing, bounded ex-ante weights and common metrics."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class WeightConfig:
    date_gamma: float=.75
    volatility_power: float=0
    lower: float=.5
    upper: float=2.


def sample_weights(dates,volatility,config=WeightConfig()):
    if not 0<=config.date_gamma<=1 or config.volatility_power<0 or not 0<config.lower<=1<=config.upper:
        raise ValueError('Invalid weights')
    dates=np.asarray(dates);vol=np.asarray(volatility,float)
    if len(dates)!=len(vol) or pd.isna(dates).any():raise ValueError('Invalid dates/volatility shape')
    weights=np.empty(len(dates));stock=np.ones(len(dates))
    for day in np.unique(dates):
        idx=np.flatnonzero(dates==day);v=vol[idx];good=np.isfinite(v)&(v>0)
        if config.volatility_power>0 and good.any():
            raw=np.ones(len(idx));raw[good]=(v[good]/np.median(v[good]))**config.volatility_power
            # Bounded mean-one multipliers preserve daily total AND final caps.
            left,right=0.,1.
            while np.clip(right*raw,config.lower,config.upper).mean()<1:right*=2
            for _ in range(70):
                mid=(left+right)/2
                if np.clip(mid*raw,config.lower,config.upper).mean()<1:left=mid
                else:right=mid
            stock[idx]=np.clip((left+right)/2*raw,config.lower,config.upper)
        weights[idx]=len(idx)**(-config.date_gamma)*stock[idx]
    return weights/weights.mean(),stock


def fit_preprocessing(x,names,min_coverage=.2,corr_threshold=.9995,seed=42,max_rows=50000):
    if isinstance(x,pd.DataFrame):
        if x.columns.duplicated().any() or set(names)-set(x.columns):raise ValueError('Missing or duplicate feature names')
        x=x.loc[:,names].to_numpy()
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    if x.ndim!=2 or x.shape[1]!=len(names) or len(set(names))!=len(names) or not len(x):
        raise ValueError('Training feature schema mismatch')
    sample=x[rng.choice(len(x),min(len(x),max_rows),replace=False)]
    stats=dict(names=list(names),median=[],lower=[],upper=[],mean=[],std=[],active=[],dropped={})
    z=np.empty_like(sample)
    for j,name in enumerate(names):
        finite=sample[np.isfinite(sample[:,j]),j];coverage=len(finite)/len(sample)
        lo,med,hi=np.quantile(finite,[.01,.5,.99]) if len(finite) else (0.,0.,0.)
        clipped=np.clip(np.where(np.isfinite(sample[:,j]),sample[:,j],med),lo,hi)
        mean,std=clipped.mean(),clipped.std()
        active=coverage>=min_coverage and std>1e-12
        for key,val in [('median',med),('lower',lo),('upper',hi),('mean',mean),('std',std if active else 1.)]:stats[key].append(float(val))
        stats['active'].append(bool(active));z[:,j]=(clipped-mean)/(std if active else 1)
        if not active:stats['dropped'][name]='low_coverage_or_constant_in_training'
    kept=[]
    for j in np.flatnonzero(stats['active']):
        if kept:
            corr=np.abs(z[:,kept].T@z[:,j]/len(z))
            if corr.max()>=corr_threshold:
                stats['active'][j]=False;stats['dropped'][names[j]]=f'training_correlation_with:{names[kept[int(corr.argmax())]]}';continue
        kept.append(j)
    if not kept:raise ValueError('No usable training features')
    stats['sample_rows']=len(sample);stats['active_names']=[names[j] for j in kept]
    return stats


def transform_features(x,stats):
    if isinstance(x,pd.DataFrame):
        if x.columns.duplicated().any() or set(stats['names'])-set(x.columns):raise ValueError('Missing or duplicate feature names')
        x=x.loc[:,stats['names']].to_numpy()
    x=np.array(x,dtype=float,copy=True)
    if x.ndim!=2 or x.shape[1]!=len(stats['names']):raise ValueError('Preprocessor feature count mismatch')
    for j in range(x.shape[1]):
        x[:,j]=np.clip(np.where(np.isfinite(x[:,j]),x[:,j],stats['median'][j]),stats['lower'][j],stats['upper'][j])
        x[:,j]=(x[:,j]-stats['mean'][j])/stats['std'][j]
    return x[:,np.asarray(stats['active'],bool)].astype(np.float32)


def weighted_corr(a,b,w):
    a=a-np.average(a,weights=w);b=b-np.average(b,weights=w)
    den=np.sqrt(np.dot(w,a*a)*np.dot(w,b*b))
    return float(np.dot(w,a*b)/den) if den>0 else np.nan


def evaluate_common(pred,y,dates,evaluation_stock_weights,min_stocks=20):
    pred=np.asarray(pred,float);y=np.asarray(y,float);dates=np.asarray(dates);w=np.asarray(evaluation_stock_weights,float)
    if not (np.isfinite(pred).all() and np.isfinite(y).all() and np.isfinite(w).all() and (w>0).all()):raise ValueError('Invalid evaluation values')
    daily=[]
    for day in np.unique(dates):
        ii=np.flatnonzero(dates==day)
        if len(ii)<min_stocks:continue
        p,a=rankdata(pred[ii]),rankdata(y[ii])
        daily.append(dict(date=str(day),n=len(ii),rank_ic=weighted_corr(p,a,np.ones(len(ii))),
            weighted_rank_ic=weighted_corr(p,a,w[ii]),ic=weighted_corr(pred[ii],y[ii],np.ones(len(ii)))))
    frame=pd.DataFrame(daily)
    output=dict(rows=len(y),days=len(daily),rmse=float(np.sqrt(np.mean((pred-y)**2))),mae=float(np.mean(np.abs(pred-y))))
    for col in ('rank_ic','weighted_rank_ic','ic'):
        good=frame[col].dropna().to_numpy() if len(frame) else np.array([])
        output[f'mean_{col}']=float(good.mean()) if len(good) else None
        output[f'{col}_valid_days']=len(good)
    return output,daily
