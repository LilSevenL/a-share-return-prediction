"""Window-local moments: no accumulator state inherited from distant history."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def windows(s,w):
    return sliding_window_view(s.to_numpy(dtype=float),w)


def series(s,w,values):
    result=np.full(len(s),np.nan)
    result[w-1:]=values
    return pd.Series(result,index=s.index)


def kurtosis(s,w):
    if len(s)<w:return s*np.nan
    x=windows(s,w);valid=np.isfinite(x).all(1)
    z=x-x.mean(1)[:,None];m2=np.mean(z*z,axis=1);m4=np.mean(z**4,axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        k=(w-1)/((w-2)*(w-3))*((w+1)*m4/m2**2-3*(w-1))
    return series(s,w,np.where(valid&(m2>0),k,np.nan))


def correlation(a,b,w):
    if len(a)<w:return a*np.nan
    x=windows(a,w);y=windows(b,w)
    valid=np.isfinite(x).all(1)&np.isfinite(y).all(1)
    x=x-x.mean(1)[:,None];y=y-y.mean(1)[:,None]
    den=np.sqrt(np.sum(x*x,axis=1)*np.sum(y*y,axis=1))
    with np.errstate(divide='ignore',invalid='ignore'):v=np.sum(x*y,axis=1)/den
    return series(a,w,np.where(valid&(den>0),np.clip(v,-1,1),np.nan))


def trend_r2(s,w):
    if len(s)<w:return s*np.nan
    x=windows(s,w);valid=np.isfinite(x).all(1)
    z=x-x.mean(1)[:,None];t=np.arange(w,dtype=float);t-=t.mean()
    den=np.sum(z*z,axis=1)*np.sum(t*t)
    with np.errstate(divide='ignore',invalid='ignore'):v=(z@t)**2/den
    return series(s,w,np.where(valid&(den>0),np.clip(v,0,1),np.nan))
