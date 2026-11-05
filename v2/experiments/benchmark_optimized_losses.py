"""Frozen-sample computation benchmark; no validation-based model selection."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import hashlib,json,time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
import linear_losses_optimized as opt

ROOT=_REPO_ROOT
BASE=ROOT/'quant_data/processed/v2/loss_pilot_2017'
OUT=ROOT/'quant_data/processed/v2/loss_solver_benchmark_blockcache'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    x0=np.load(BASE/'sample_x.npy',mmap_mode='r');y=np.load(BASE/'sample_y.npy')
    need=x0.size*8;available=psutil.virtual_memory().available
    # Explicit bounded cache: never materialize a full large fold silently.
    if need>2*1024**3 or need>available*.25: raise MemoryError('Float64 cache exceeds pilot memory budget')
    manifest=dict(scope='computation equivalence only; same 200k training sample',
        source_hashes={p.name:sha(p) for p in [Path(__file__),Path(opt.__file__)]},
        sample_hashes={n:sha(BASE/n) for n in ['sample_x.npy','sample_y.npy']},
        prediction_atol=1e-6,objective_atol=1e-7,coefficient_atol=1e-5,
        threads=4,cache_bytes=need,selection_on_validation=False)
    expected=read(BASE/'sample_complete.json')
    assert manifest['sample_hashes']['sample_x.npy']==expected['x_sha256']
    assert manifest['sample_hashes']['sample_y.npy']==expected['y_sha256']
    if (OUT/'manifest.json').exists(): assert read(OUT/'manifest.json')==manifest
    save(OUT/'manifest.json',manifest)
    start=time.perf_counter();x=opt.CachedDesign(x0);cache_seconds=time.perf_counter()-start
    results={}
    with threadpool_limits(limits=4):
        for name in ['mse','huber','mae']:
            print(f'Start {name}, cache={need/1024**2:.1f} MiB',flush=True)
            start=time.perf_counter()
            model=opt.fit_linear_loss(x,y,config=opt.LossConfig(loss=name,l2=0.,delta=1.35,max_iter=6000,tol=1e-5))
            elapsed=time.perf_counter()-start
            old=read(BASE/name/'model.json')['linear_model']
            pred_delta=opt.predict_linear_loss(model,x)-opt.predict_linear_loss(old,x)
            coefficient_delta=float(np.max(np.abs(np.r_[model['intercept'],model['coefficient']]-np.r_[old['intercept'],old['coefficient']])))
            objective_delta=abs(model['diagnostics']['normalized_penalized_objective']-old['diagnostics']['normalized_penalized_objective'])
            row=dict(seconds=elapsed,prior_seconds=read(BASE/name/'fit_timing.json')['seconds'],
                max_prediction_difference=float(np.max(np.abs(pred_delta))),
                max_coefficient_difference=coefficient_delta,objective_difference=objective_delta,
                iterations=model['diagnostics']['iterations'],converged=model['diagnostics']['converged'])
            row['equivalent']=bool(row['max_prediction_difference']<=manifest['prediction_atol'] and coefficient_delta<=manifest['coefficient_atol'] and objective_delta<=manifest['objective_atol'] and row['converged'])
            row['speedup_vs_prior_run']=row['prior_seconds']/elapsed
            save(OUT/f'{name}_model.json',model);save(OUT/f'{name}_comparison.json',row)
            results[name]=row;print(name,json.dumps(row),flush=True)
            if not row['equivalent']:raise RuntimeError('Numerical equivalence gate failed')
    save(OUT/'summary.json',dict(results=results,cache_seconds=cache_seconds,
        all_equivalent=all(r['equivalent'] for r in results.values()),
        timing_caveat='prior run comparison, not controlled same-session timing; hardware load may differ',
        full_sample_training=False))
    print('BENCHMARK COMPLETE',flush=True)

if __name__=='__main__': main()
