"""Numerically stable, bounded-memory unweighted OLS via tall-skinny QR."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import numpy as np
from scipy import linalg


class StreamingOLS:
    def __init__(self,n_features):
        self.p=n_features;self.r=np.empty((0,n_features+2));self.rows=0;self.completed_files=0

    def update(self,x,y):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
        if x.ndim!=2 or x.shape[1]!=self.p or y.shape!=(len(x),) or not len(x):raise ValueError('Invalid OLS batch')
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite OLS input')
        augmented=np.column_stack([np.ones(len(x)),x,y])
        self.r=linalg.qr(np.vstack([self.r,augmented]),mode='r',check_finite=False,overwrite_a=True)[0][:self.p+2]
        self.rows+=len(x)

    def finish(self):
        if self.rows<=self.p+1:raise ValueError('Insufficient training rows')
        a=self.r[:,:self.p+1];b=self.r[:,-1]
        # Same explicit singular-value tolerance across batches and folds.
        cutoff=np.finfo(float).eps*max(self.rows,self.p+1)
        theta,_,rank,s=linalg.lstsq(a,b,cond=cutoff,lapack_driver='gelsd')
        residual=a@theta-b
        return dict(coefficient=theta[1:].tolist(),intercept=float(theta[0]),
            target_scale=1.,target_scale_method='raw target units; OLS is scale invariant',
            huber_delta_target_units=None,config=dict(loss='mse',l2=0.,sample_weight='uniform'),
            diagnostics=dict(solver='streaming augmented QR then SVD; no normal equations',training_rows=self.rows,
                feature_count=self.p,rank=int(rank),design_columns=self.p+1,singular_value_cutoff=cutoff,
                condition_number=float(s[0]/s[-1]) if s[-1]>0 else None,
                training_rmse=float(np.sqrt(np.dot(residual,residual)/self.rows))))

    def save(self,path):
        temp=path.with_suffix('.tmp.npz');np.savez(temp,r=self.r,rows=self.rows,p=self.p,completed_files=self.completed_files);temp.replace(path)

    @classmethod
    def load(cls,path):
        d=np.load(path);obj=cls(int(d['p']));obj.r=d['r'];obj.rows=int(d['rows']);obj.completed_files=int(d['completed_files']);return obj
