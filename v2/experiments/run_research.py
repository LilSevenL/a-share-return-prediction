"""Frozen V2 comparison: three losses x three stock weights; common metrics and simulation."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import argparse
import hashlib
import json
from pathlib import Path
import sys
import psutil
ROOT=_REPO_ROOT
sys.path.insert(0,str(ROOT))
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from balanced_features import COLUMNS
from build_balanced_features import save_json,save_frame
from linear_losses import LossConfig,fit_linear_loss,predict_linear_loss
from research_controls import WeightConfig,sample_weights,fit_preprocessing,transform_features,evaluate_common
from simulation import source_files,prepare_market,simulate,DATA
from feature_model import validate_input,bundle_model,predict_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir',default=str(DATA/'v2/balanced_500'))
    p.add_argument('--output-dir',default=str(DATA/'v2/research_comparison'))
    p.add_argument('--train-start',type=int,default=2011)
    p.add_argument('--folds',nargs='+',type=int,default=list(range(2017,2024)))
    p.add_argument('--max-train-rows',type=int)
    p.add_argument('--l2',type=float,default=.01)
    p.add_argument('--delta',type=float,default=1.35)
    p.add_argument('--cost-bps',nargs='+',type=float,default=[20.,50.])
    args=p.parse_args()
    if min(args.folds)<=args.train_start or max(args.folds)>2023:raise ValueError('Research folds must be <=2023 and after training start')
    if args.max_train_rows is not None and args.max_train_rows<100:raise ValueError('Training QA sample too small')
    features=Path(args.features_dir).resolve();output=Path(args.output_dir).resolve()
    if not output.is_relative_to(DATA/'v2'):raise ValueError('V2 output directory required')
    output.mkdir(parents=True,exist_ok=True)
    report=json.loads((features/'feature_report.json').read_text(encoding='utf-8'))
    if report['schema_features']!=500 or report['test_used']:raise ValueError('Invalid feature provenance')
    if report['scope']=='all_available_stocks' and report.get('universe_status')!='verified_historical_source':
        raise ValueError('Formal full-universe comparison blocked: historical-universe data are not yet repaired and audited')
    feature_files=source_files(features,'feature_year',range(args.train_start,max(args.folds)+1))
    target_files=source_files(DATA/'targets_5d','signal_year',range(args.train_start,max(args.folds)+1))
    manifest=dict(arguments=vars(args),feature_report=report,
        inputs=[dict(path=f,size=Path(f).stat().st_size,mtime_ns=Path(f).stat().st_mtime_ns) for f in feature_files+target_files],
        code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
        scope='integration_QA_only' if args.max_train_rows or report['scope']!='all_available_stocks' else 'research_validation',
        test_used=False,automatic_winner_selection=False,
        labels='finite raw adjusted log returns; no future tradability filter; exit before fold boundary',
        evaluation_weight='fixed p=1, bounded mean-one per day; identical for every model',
        note='MSE with nonzero L2 is ridge, not unregularized OLS; weights imply weighted regression')
    mf=output/'manifest.json'
    if mf.exists() and json.loads(mf.read_text(encoding='utf-8'))!=manifest:raise ValueError('Run configuration changed; use a new output directory')
    save_json(mf,manifest)
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='6GB'")
    con.read_parquet(feature_files,hive_partitioning=False).create_view('features')
    con.read_parquet(target_files,hive_partitioning=False).create_view('targets')
    results=[]
    with threadpool_limits(limits=4):
        for year in args.folds:
            lo=f'{year}-01-01';hi=f'{year+1}-01-01'
            expected_rows=con.execute('SELECT count(*) FROM features WHERE signal_date>=? AND signal_date<?',[f'{args.train_start}-01-01',hi]).fetchone()[0]
            # Pandas + original/transformed arrays require several copies. Fail before swapping/OOM.
            estimated_bytes=expected_rows*len(COLUMNS)*8*4
            if estimated_bytes>psutil.virtual_memory().available*.65:
                raise MemoryError('Fold exceeds safe in-memory budget. Use a smaller documented research universe or implement disk-backed fitting; --max-train-rows only caps solver rows, not loading.')
            train=con.execute('''SELECT f.*,t.target_logret_5d_raw y FROM features f JOIN targets t USING(ts_code,signal_date)
                WHERE f.signal_date>=? AND f.signal_date<? AND t.exit_date<?
                AND isfinite(t.target_logret_5d_raw) ORDER BY f.signal_date,f.ts_code''',[f'{args.train_start}-01-01',lo,lo]).fetch_df()
            valid=con.execute('''SELECT f.*,CASE WHEN t.exit_date<? THEN t.target_logret_5d_raw ELSE NULL END y
                FROM features f LEFT JOIN targets t USING(ts_code,signal_date)
                WHERE f.signal_date>=? AND f.signal_date<? ORDER BY f.signal_date,f.ts_code''',[hi,lo,hi]).fetch_df()
            if train.empty or valid.empty:raise ValueError('Empty fold')
            validate_input(train);validate_input(valid)
            stats=fit_preprocessing(train,COLUMNS)
            save_json(output/f'preprocessing_{year}.json',stats)
            rng=np.random.default_rng(42)
            chosen=np.sort(rng.choice(len(train),min(len(train),args.max_train_rows or len(train)),replace=False))
            x=transform_features(train.iloc[chosen],stats)
            y=train.y.to_numpy()[chosen];dates=valid.signal_date.to_numpy()
            _,ew=sample_weights(dates,valid.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
            mask=np.isfinite(valid.y.to_numpy());base_signals=valid[['ts_code','signal_date','volatility_20d']].copy()
            market_panel=None
            print(f'{year}: train {len(train)}; fit {len(chosen)}; validation {len(valid)}; active factors {x.shape[1]}',flush=True)
            for power in (0.,.5,1.):
                weights,_=sample_weights(train.signal_date.to_numpy(),train.volatility_20d.to_numpy(),WeightConfig(volatility_power=power))
                for loss in (('ols','mse','mae','huber') if power==0 else ('mse','mae','huber')):
                    name='ols_unweighted' if loss=='ols' else f'{loss}_p{power:g}';folder=output/name;folder.mkdir(exist_ok=True)
                    result_path=folder/f'result_{year}.json'
                    if result_path.exists():results.append(json.loads(result_path.read_text(encoding='utf-8')));continue
                    print(f'Fitting {year} {name}',flush=True)
                    config=LossConfig(loss='mse' if loss=='ols' else loss,l2=0 if loss=='ols' else args.l2,delta=args.delta,max_iter=6000,tol=1e-5)
                    model=fit_linear_loss(x,y,np.ones(len(chosen)) if loss=='ols' else weights[chosen],config)
                    artifact=bundle_model(model,stats)
                    save_json(folder/f'model_{year}.json',artifact)
                    prediction=predict_frame(artifact,valid)
                    metrics,daily=evaluate_common(prediction[mask],valid.y.to_numpy()[mask],dates[mask],ew[mask])
                    signals=base_signals.copy();signals['prediction']=prediction
                    save_frame(folder/f'predictions_{year}.parquet',signals)
                    save_frame(folder/f'daily_metrics_{year}.parquet',pd.DataFrame(daily))
                    # The same pool and execution rules are applied to every fitted model.
                    if market_panel is None:
                        order_template,market_panel=prepare_market(signals,year)
                        order_template=order_template.drop(columns='prediction')
                    orders=order_template.merge(signals[['ts_code','signal_date','prediction']],on=['ts_code','signal_date'],validate='one_to_one')
                    simulations=[]
                    for cost in args.cost_bps:
                        sim,nav,trades=simulate(orders,market_panel,cost_bps=cost)
                        simulations.append(sim)
                        save_frame(folder/f'nav_{year}_cost{cost:g}.parquet',nav)
                        save_frame(folder/f'orders_{year}_cost{cost:g}.parquet',trades)
                    result=dict(year=year,model=name,status='completed',train_rows=len(train),fit_rows=len(chosen),
                        active_factors=x.shape[1],prediction_rows=len(valid),unscored_rows=int((~mask).sum()),
                        predictive_metrics=metrics,simulations=simulations,scope=manifest['scope'])
                    save_json(result_path,result);results.append(result)
                    save_json(output/'summary.json',dict(status='running',results=results,test_used=False))
                    print(f'Completed {name}: Rank IC={metrics["mean_rank_ic"]:.4f}',flush=True)
    con.close()
    save_json(output/'summary.json',dict(status='completed',scope=manifest['scope'],results=results,test_used=False,automatic_winner_selection=False))
    print(f'Completed {len(results)} model/fold comparisons',flush=True)


if __name__=='__main__':main()
