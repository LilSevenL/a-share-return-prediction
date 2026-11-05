"""Frozen 7-fold matched loss by training-weight comparison."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import sys,json,hashlib,time,gc
ROOT=_REPO_ROOT;sys.path.insert(0,str(ROOT))
import duckdb,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import linear_losses_optimized as losses
from weighted_resumable_losses import fit
from research_controls import transform_features,sample_weights,WeightConfig,evaluate_common
from feature_model import bundle_model
from run_multiyear_losses import read,save,sha

DATA=ROOT/'quant_data/processed';BASE=DATA/'v2/multiyear_losses_200k'
OUT=DATA/'v2/stock_weights_200k';FEATURES=DATA/'v2/local_features_2011_2023_stable'
GROUPS={'date':0.,'date_stock':1.}
def main():
    OUT.mkdir(exist_ok=True)
    assert read(BASE/'verification.json')['all_converged']
    ff=[p for year in range(2011,2024) for p in sorted((FEATURES/f'feature_year={year}').glob('*.parquet'))]
    bp=[p for year in range(2017,2024) for p in (BASE/f'validation_{year}').rglob('*') if p.is_file() and p.suffix in ['.json','.npy','.parquet']]
    names=['run_stock_weights.py','weighted_resumable_losses.py','linear_losses_optimized.py','research_controls.py','feature_model.py','balanced_features.py','run_multiyear_losses.py']
    manifest=dict(years=list(range(2017,2024)),sample_rows=200000,date_gamma=.75,groups=GROUPS,stock_caps=[.5,2.],
        l2=0,delta=1.35,tol=1e-5,max_iter=6000,sources={n:sha(_source_path('v2/'+n)) for n in names},
        inputs=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in ff+bp],
        baseline_manifest_sha=sha(BASE/'manifest.json'),test_used=False,automatic_selection=False,
        scope='historical development; matched 200k samples; 42 new weighted fits, 21 frozen uniform controls')
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest,'Changed inputs or code'
    save(OUT/'manifest.json',manifest);run_id=sha(OUT/'manifest.json')
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='2GB'")
    con.read_parquet([str(p) for p in ff],hive_partitioning=False).create_view('features')
    with threadpool_limits(limits=4):
        for year in range(2017,2024):
            folder=OUT/f'validation_{year}';folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():continue
            prior=BASE/f'validation_{year}';sample=read(prior/'sample.json')
            assert sha(prior/'sample_x.npy')==sample['x_sha'] and sha(prior/'sample_y.npy')==sample['y_sha']
            keys=pd.read_parquet(prior/'sample_keys.parquet');keys['row_id']=np.arange(len(keys));con.register('keys',keys)
            meta=con.execute('SELECT k.*,f.volatility_20d FROM keys k JOIN features f USING(ts_code,signal_date) ORDER BY k.row_id').fetch_df();con.unregister('keys')
            assert len(meta)==200000 and not meta.duplicated(['ts_code','signal_date']).any()
            pd.testing.assert_frame_equal(keys[['ts_code','signal_date']],meta[['ts_code','signal_date']],check_dtype=False)
            assert meta.signal_date.max()<pd.Timestamp(f'{year}-01-01')
            weights={};stock={}
            for group,power in GROUPS.items():weights[group],stock[group]=sample_weights(meta.signal_date.to_numpy(),meta.volatility_20d.to_numpy(),WeightConfig(date_gamma=.75,volatility_power=power))
            audit={};masses={}
            for group,w in weights.items():
                masses[group]=pd.Series(w).groupby(meta.signal_date).sum().to_numpy()
                audit[group]=dict(min=float(w.min()),max=float(w.max()),mean=float(w.mean()),
                    effective_sample_size=float(w.sum()**2/(w@w)),stock_min=float(stock[group].min()),stock_max=float(stock[group].max()),
                    stock_lower_cap_fraction=float(np.mean(stock[group]<=.5+1e-12)),stock_upper_cap_fraction=float(np.mean(stock[group]>=2-1e-12)))
                meta[group+'_weight']=w;meta[group+'_stock_multiplier']=stock[group]
            np.testing.assert_allclose(masses['date'],masses['date_stock'],atol=1e-10)
            audit['daily_mass_max_difference']=float(np.max(np.abs(masses['date']-masses['date_stock'])))
            audit['invalid_volatility_rows']=int((~np.isfinite(meta.volatility_20d)|(meta.volatility_20d<=0)).sum())
            save(folder/'weight_audit.json',audit);meta.to_parquet(folder/'training_weights.parquet',index=False)
            x0=np.load(prior/'sample_x.npy',mmap_mode='r');x=losses.CachedDesign(x0);y=np.load(prior/'sample_y.npy')
            stats=read(prior/'mse/model.json')['preprocessing'];models={}
            for group in GROUPS:
                for loss in ['mse','huber','mae']:
                    key=group+'_'+loss;sub=folder/key;sub.mkdir(exist_ok=True)
                    if (sub/'model.json').exists():models[key]=read(sub/'model.json');continue
                    print(f'{year} {key}: fitting',flush=True);start=time.perf_counter()
                    cfg=losses.LossConfig(loss=loss,l2=0,delta=1.35,tol=1e-5,max_iter=6000)
                    try:
                        model=fit(x,y,cfg,sub/'checkpoint.npz',run_id+str(year)+sample['x_sha']+sample['y_sha'],
                            lambda step:print(f'{year} {key}: iteration {step}',flush=True),sample_weight=weights[group])
                    except Exception as e:save(sub/'failure.json',dict(error=str(e)));raise
                    models[key]=bundle_model(model,stats);save(sub/'model.json',models[key]);save(sub/'timing.json',dict(seconds=time.perf_counter()-start))
                    print(f'{year} {key}: converged {model["diagnostics"]["iterations"]}',flush=True)
            del x,x0,y;gc.collect()
            labels=pd.read_parquet(prior/'mse/predictions.parquet',columns=['ts_code','signal_date','y'])
            predictions={key:[] for key in models}
            for path in sorted((FEATURES/f'feature_year={year}').glob('*.parquet')):
                frame=pd.read_parquet(path);xx=transform_features(frame,stats)
                for key,model in models.items():
                    s=frame[['ts_code','signal_date','volatility_20d']].copy();s['prediction']=losses.predict_linear_loss(model['linear_model'],xx);predictions[key].append(s)
                del frame,xx
            for key in models:
                signals=pd.concat(predictions[key],ignore_index=True).merge(labels,on=['ts_code','signal_date'],validate='one_to_one')
                assert len(signals)==len(labels) and np.isfinite(signals.prediction).all()
                mask=np.isfinite(signals.y);_,ew=sample_weights(signals.signal_date.to_numpy(),signals.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
                metrics,daily=evaluate_common(signals.prediction[mask].to_numpy(),signals.y[mask].to_numpy(),signals.signal_date[mask].to_numpy(),ew[mask])
                sub=folder/key;signals.to_parquet(sub/'predictions.parquet',index=False);pd.DataFrame(daily).to_parquet(sub/'daily_metrics.parquet',index=False);save(sub/'metrics.json',metrics)
                print(f'{year} {key}: RankIC={metrics["mean_rank_ic"]:.6f}',flush=True)
            save(folder/'complete.json',dict(year=year,models=list(models),training_rows=200000))
            del models,predictions,signals,labels,keys,meta,weights,stock;gc.collect()
    con.close();rows=[];aggregates={}
    for group in ['uniform','date','date_stock']:
        for loss in ['mse','huber','mae']:
            key=group+'_'+loss;daily=[];n=0;sse=0;sae=0
            for year in range(2017,2024):
                sub=(BASE/f'validation_{year}'/loss) if group=='uniform' else (OUT/f'validation_{year}'/key)
                m=read(sub/'metrics.json');rows.append(dict(year=year,group=group,loss=loss,**m));daily.append(pd.read_parquet(sub/'daily_metrics.parquet'))
                n+=m['rows'];sse+=m['rmse']**2*m['rows'];sae+=m['mae']*m['rows']
            d=pd.concat(daily,ignore_index=True);aggregates[key]=dict(rows=n,days=len(d),rank_ic=float(d.rank_ic.mean()),weighted_rank_ic=float(d.weighted_rank_ic.mean()),rmse=float(np.sqrt(sse/n)),mae=sae/n)
    save(OUT/'summary.json',dict(status='completed',scope=manifest['scope'],folds=rows,aggregate=aggregates))
    print('ALL WEIGHT COMPARISONS COMPLETE',flush=True)
if __name__=='__main__':main()
