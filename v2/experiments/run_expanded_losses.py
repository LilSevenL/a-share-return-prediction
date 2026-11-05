"""Seven historical development folds, matched 500k training rows per fold."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import sys,json,hashlib,time,gc,os
ROOT=_REPO_ROOT;sys.path.insert(0,str(ROOT))
import duckdb,numpy as np,pandas as pd
import copy
from balanced_features import RECORDS
from threadpoolctl import threadpool_limits
import linear_losses_optimized as losses
from resumable_losses import fit
from research_controls import transform_features,sample_weights,WeightConfig,evaluate_common
from feature_model import bundle_model,predict_frame

DATA=ROOT/'quant_data/processed';OUT=DATA/'v2/expanded_losses_500k'
FEATURES=DATA/'v2/local_features_2011_2023_stable'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    ff=sorted(FEATURES.glob('feature_year=*/*.parquet'))
    ff=[p for p in ff if 2011<=int(p.parent.name.split('=')[1])<=2023]
    tf=[p for year in range(2011,2024) for p in sorted((DATA/f'targets_5d/signal_year={year}').glob('*.parquet'))]
    refs=[DATA/f'v2/ols_baseline_stable/validation_{year}'/name for year in range(2017,2024) for name in ['preprocessing.json','predictions.parquet']]
    names=['run_expanded_losses.py','resumable_losses.py','linear_losses_optimized.py','linear_losses.py','research_controls.py','feature_model.py','balanced_features.py']
    manifest=dict(years=list(range(2017,2024)),rows_per_fold=500000,l2=0,delta=1.35,tol=1e-5,max_iter=6000,
        sources={n:sha(_source_path('v2/'+n)) for n in names},
        inputs=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in ff+tf+refs],
        scope='matched-sample historical development validation, not full sample or future test',
        sample_seed=20261025,automatic_selection=False)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest,'Inputs/source changed'
    save(OUT/'manifest.json',manifest)
    run_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='3GB'")
    con.read_parquet([str(p) for p in ff],hive_partitioning=False).create_view('features')
    con.read_parquet([str(p) for p in tf],hive_partitioning=False).create_view('targets')
    with threadpool_limits(limits=4):
        for year in range(2017,2024):
            folder=OUT/f'validation_{year}';folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():continue
            print(f'{year}: preparing matched training sample',flush=True)
            baseline=DATA/f'v2/ols_baseline_stable/validation_{year}';stats=read(baseline/'preprocessing.json')
            if not (folder/'sample.json').exists():
                keys=con.execute(f'''SELECT f.ts_code,f.signal_date,t.target_logret_5d_raw y FROM features f JOIN targets t USING(ts_code,signal_date)
                    WHERE f.signal_date<DATE '{year}-01-01' AND t.exit_date<DATE '{year}-01-01' AND isfinite(t.target_logret_5d_raw)
                    ORDER BY hash(f.ts_code,f.signal_date,20261025),f.ts_code,f.signal_date LIMIT 500000''').fetch_df()
                assert len(keys)==500000 and not keys.duplicated(['ts_code','signal_date']).any()
                old_keys=pd.read_parquet(DATA/f'v2/multiyear_losses_200k/validation_{year}/sample_keys.parquet')
                assert len(old_keys.merge(keys[['ts_code','signal_date']],on=['ts_code','signal_date'],validate='one_to_one'))==len(old_keys)
                con.register('sample_keys',keys)
                frame=con.execute('SELECT f.*,k.y FROM features f JOIN sample_keys k USING(ts_code,signal_date) ORDER BY f.signal_date,f.ts_code').fetch_df()
                x=transform_features(frame,stats);y=frame.y.to_numpy(dtype=float)
                assert len(y)==500000
                np.save(folder/'sample_x.npy',x);np.save(folder/'sample_y.npy',y)
                frame[['ts_code','signal_date']].to_parquet(folder/'sample_keys.parquet',index=False)
                save(folder/'sample.json',dict(rows=len(y),features=x.shape[1],x_sha=sha(folder/'sample_x.npy'),y_sha=sha(folder/'sample_y.npy')))
                con.unregister('sample_keys');del keys,frame,x,y;gc.collect()
            sample=read(folder/'sample.json')
            assert sha(folder/'sample_x.npy')==sample['x_sha'] and sha(folder/'sample_y.npy')==sample['y_sha']
            x0=np.load(folder/'sample_x.npy',mmap_mode='r');x=x0;y=np.load(folder/'sample_y.npy')
            models={}
            for name in ['full_huber','full_mae','compact_huber','compact_mae']:
                sub=folder/name;sub.mkdir(exist_ok=True)
                if (sub/'model.json').exists():models[name]=read(sub/'model.json');continue
                print(f'{year}: fitting {name}',flush=True);start=time.perf_counter()
                config=losses.LossConfig(loss=name.split('_')[-1],l2=0,delta=1.35,tol=1e-5,max_iter=6000)
                model_stats=copy.deepcopy(stats)
                if name.startswith('compact'):
                    excluded={r['name'] for r in RECORDS if r['family'] in ('C','M')}
                    model_stats['active']=[a and n not in excluded for a,n in zip(stats['active'],stats['names'])]
                    model_stats['active_names']=[n for a,n in zip(model_stats['active'],stats['names']) if a]
                    for n in excluded:model_stats['dropped'][n]='precommitted_CM_removal'
                keep=[i for i,n in enumerate(stats['active_names']) if n in model_stats['active_names']]
                fit_x=losses.CachedDesign(x[:,keep])
                try:
                    model=fit(fit_x,y,config,sub/'checkpoint.npz',run_id+str(year)+name+sample['x_sha']+sample['y_sha'],
                        lambda step:print(f'{year} {name}: iteration {step}',flush=True))
                except Exception as e:
                    save(sub/'failure.json',dict(error=str(e)));raise
                del fit_x;gc.collect()
                models[name]=bundle_model(model,model_stats);save(sub/'model.json',models[name])
                save(sub/'timing.json',dict(seconds=time.perf_counter()-start))
                print(f'{year} {name}: converged ({model["diagnostics"]["iterations"]})',flush=True)
            del x,x0,y;gc.collect()
            labels=pd.read_parquet(baseline/'predictions.parquet',columns=['ts_code','signal_date','y'])
            predictions={n:[] for n in models}
            for path in sorted((FEATURES/f'feature_year={year}').glob('*.parquet')):
                frame=pd.read_parquet(path)
                for name,model in models.items():
                    s=frame[['ts_code','signal_date','volatility_20d']].copy();s['prediction']=predict_frame(model,frame)
                    predictions[name].append(s)
                del frame
            for name in models:
                signals=pd.concat(predictions[name],ignore_index=True).merge(labels,on=['ts_code','signal_date'],validate='one_to_one')
                assert len(signals)==len(labels) and np.isfinite(signals.prediction).all()
                mask=np.isfinite(signals.y)
                _,ew=sample_weights(signals.signal_date.to_numpy(),signals.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
                metrics,daily=evaluate_common(signals.prediction[mask].to_numpy(),signals.y[mask].to_numpy(),signals.signal_date[mask].to_numpy(),ew[mask])
                sub=folder/name;signals.to_parquet(sub/'predictions.parquet',index=False)
                pd.DataFrame(daily).to_parquet(sub/'daily_metrics.parquet',index=False);save(sub/'metrics.json',metrics)
                print(f'{year} {name}: RankIC={metrics["mean_rank_ic"]:.6f}',flush=True)
            save(folder/'complete.json',dict(year=year,losses=list(models),training_rows=500000))
            del models,predictions,signals,labels;gc.collect()
    con.close()
    rows=[];aggregate={}
    for name in ['full_huber','full_mae','compact_huber','compact_mae']:
        daily=[];n=0;sse=0.;sae=0.
        for year in range(2017,2024):
            sub=OUT/f'validation_{year}'/name;m=read(sub/'metrics.json');rows.append(dict(year=year,loss=name,**m))
            daily.append(pd.read_parquet(sub/'daily_metrics.parquet'));n+=m['rows'];sse+=m['rmse']**2*m['rows'];sae+=m['mae']*m['rows']
        d=pd.concat(daily,ignore_index=True)
        aggregate[name]=dict(rows=n,days=len(d),mean_rank_ic=float(d.rank_ic.mean()),mean_weighted_rank_ic=float(d.weighted_rank_ic.mean()),rmse=float(np.sqrt(sse/n)),mae=sae/n)
    save(OUT/'summary.json',dict(status='completed',folds=rows,aggregate=aggregate,scope=manifest['scope']))
    print('ALL SEVEN FOLDS COMPLETE',flush=True)

if __name__=='__main__':main()
