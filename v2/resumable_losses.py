"""Exact ADMM state recovery and explicitly marked Huber warm restart."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,os,time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import linear_losses_optimized as base

def save_state(path,identity,**arrays):
    path=Path(path); temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f,identity=np.array(identity),**arrays)
        f.flush();os.fsync(f.fileno())
    for attempt in range(10):
        try:
            os.replace(temp,path);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)

def fit(x,y,config,checkpoint,run_id,progress=None,checkpoint_every=100):
    config.validate(); y=np.asarray(y,dtype=float)
    if x.ndim!=2 or len(x)!=len(y) or not len(y) or not np.isfinite(y).all():raise ValueError('Invalid training input')
    for _,_,xb in base._blocks(x,config.chunk_rows):
        if not np.isfinite(xb).all():raise ValueError('Nonfinite X')
    identity=hashlib.sha256(json.dumps([run_id,asdict(config)],sort_keys=True).encode()).hexdigest()
    checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    state={}
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as data:
            if str(data['identity'])!=identity:raise ValueError('Checkpoint identity mismatch')
            state={k:data[k].copy() for k in data.files if k!='identity'}
    scale,method=base.target_scale(y);ys=y/scale;w=np.ones(len(y))
    initial,mse_info=base._mse(x,ys,w,config)
    if config.loss=='mse':return base.fit_linear_loss(x,y,config=config)
    start=int(state.get('step',0)); theta=state.get('theta',initial)
    if start>=config.max_iter:raise RuntimeError('Checkpoint exhausted iteration budget')
    if config.loss=='mae':
        n=len(y);rho=config.admm_rho
        matrix=rho*base._gram(x,w,config.chunk_rows)
        matrix[1:,1:]+=config.l2*np.eye(x.shape[1])
        try:
            chol=linalg.cho_factor(matrix,lower=True)
            solve=lambda rhs:linalg.cho_solve(chol,rhs)
        except linalg.LinAlgError:
            inv=linalg.pinvh(matrix,rtol=1e-12);solve=lambda rhs:inv@rhs
        z=state['z'] if state else base._ax(x,theta,config.chunk_rows)-ys
        u=state['u'] if state else np.zeros(n)
        converged=False
        for step in range(start+1,config.max_iter+1):
            theta=solve(rho*base._at(x,ys+z-u,config.chunk_rows)/n)
            residual=base._ax(x,theta,config.chunk_rows)-ys
            v=residual+u;new_z=np.sign(v)*np.maximum(np.abs(v)-w/rho,0)
            u+=residual-new_z
            primal=float(np.linalg.norm(residual-new_z)/np.sqrt(n))
            projections=base._at_pair(x,new_z-z,u,config.chunk_rows)/n
            dual=float(np.max(np.abs(rho*projections[:,0])))
            pt=config.tol*(1+max(np.linalg.norm(residual),np.linalg.norm(new_z))/np.sqrt(n))
            dt=config.tol*(1+float(np.max(np.abs(rho*projections[:,1]))))
            z=new_z
            if primal<=pt and dual<=dt:
                converged=True;break
            if step%checkpoint_every==0:
                save_state(checkpoint,identity,theta=theta,z=z,u=u,step=step)
                if progress:progress(step)
        info=dict(solver='ADMM exact absolute loss; full state resume',iterations=step,converged=converged,
            primal_rms=primal,primal_tolerance=float(pt),dual_gradient_inf=dual,dual_tolerance=float(dt),rho=rho,
            resumed_from_iteration=start)
    else:
        count=start
        def callback(theta):
            nonlocal count
            count+=1
            if count%checkpoint_every==0:
                save_state(checkpoint,identity,theta=theta,step=count)
                if progress:progress(count)
        result=minimize(base.objective_gradient,theta,args=(x,ys,w,config),jac=True,callback=callback,
            method='L-BFGS-B',options=dict(maxiter=config.max_iter-start,gtol=config.tol,ftol=1e-14,maxls=40))
        theta=result.x;grad=float(np.max(np.abs(result.jac)))
        info=dict(solver='L-BFGS-B; coefficient warm restart without curvature history',
            iterations=start+int(result.nit),converged=bool(result.success and grad<=max(10*config.tol,1e-8)),
            gradient_inf=grad,message=str(result.message),resumed_from_iteration=start)
    if not np.isfinite(theta).all() or not info['converged']:raise RuntimeError(f'Unconverged: {info}')
    residual=base._ax(x,theta,config.chunk_rows)*scale-y
    info.update(training_rmse=float(np.sqrt(np.mean(residual**2))),training_mae=float(np.mean(np.abs(residual))),
        normalized_penalized_objective=base.objective_gradient(theta,x,ys,w,config)[0],training_rows=len(y),feature_count=x.shape[1])
    return dict(coefficient=(theta[1:]*scale).tolist(),intercept=float(theta[0]*scale),target_scale=scale,
        target_scale_method=method,huber_delta_target_units=float(config.delta*scale) if config.loss=='huber' else None,
        config=asdict(config),diagnostics=info)
