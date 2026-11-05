"""Frozen unweighted OLS baseline on stable local survivor data, bounded-memory QR."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import argparse
import hashlib
import json
from pathlib import Path
import time
import duckdb
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from balanced_features import COLUMNS
from build_balanced_features import ROOT,save_json,save_frame
from research_controls import fit_preprocessing,transform_features,evaluate_common,sample_weights,WeightConfig
from feature_model import bundle_model,predict_frame
from streaming_ols import StreamingOLS

DATA=ROOT/'quant_data/processed'


def read_json(p):return json.loads(p.read_text(encoding='utf-8'))


def audit_labels(con,features,targets,source_config,output):
    con.read_parquet([str(p) for p in features],hive_partitioning=False).create_view('features')
    con.read_parquet([str(p) for p in targets],hive_partitioning=False).create_view('targets')
    dup=con.execute('SELECT count(*) FROM (SELECT ts_code,signal_date FROM targets GROUP BY ALL HAVING count(*)>1)').fetchone()[0]
    if dup:raise ValueError('Duplicate target keys')
    cal=pd.read_parquet(DATA/'v2/local_features_2011_2023/calendar.parquet');cal=cal.sort_values('signal_date')
    cal['expected_entry']=cal.signal_date.shift(-1);cal['expected_exit']=cal.signal_date.shift(-5);con.register('calendar',cal)
    bad=con.execute('''SELECT count(*) FROM targets t JOIN calendar c USING(signal_date)
        WHERE c.expected_exit IS NOT NULL AND (t.entry_date IS DISTINCT FROM c.expected_entry OR t.exit_date IS DISTINCT FROM c.expected_exit)''').fetchone()[0]
    if bad:raise ValueError('Target market-calendar alignment failed')
    # Independent check against original local daily prices; no 2024+ source opened.
    for name in ('daily','adj_factor'):
        con.read_parquet([i['path'] for i in source_config['inputs'][name]],hive_partitioning=False).create_view(name)
    sample=con.execute('''SELECT ts_code,signal_date,entry_date,exit_date,target_logret_5d_raw y FROM targets
        WHERE exit_date<DATE '2024-01-01' AND isfinite(target_logret_5d_raw)
        ORDER BY hash(ts_code,signal_date,42),ts_code,signal_date LIMIT 1000''').fetch_df();con.register('label_sample',sample)
    check=con.execute('''SELECT s.*,ln(x.close*xa.adj_factor/(e.open*ea.adj_factor)) expected
        FROM label_sample s LEFT JOIN daily e ON s.ts_code=e.ts_code AND s.entry_date=e.trade_date
        LEFT JOIN daily x ON s.ts_code=x.ts_code AND s.exit_date=x.trade_date
        LEFT JOIN adj_factor ea ON s.ts_code=ea.ts_code AND s.entry_date=ea.trade_date
        LEFT JOIN adj_factor xa ON s.ts_code=xa.ts_code AND s.exit_date=xa.trade_date''').fetch_df()
    if len(check)!=1000 or not np.allclose(check.y,check.expected,atol=1e-12,rtol=1e-10):raise ValueError('Independent target formula audit failed')
    coverage=con.execute('''SELECT year(f.signal_date) AS signal_year,count(*) feature_rows,
        count(t.ts_code) joined_rows,count(*) FILTER(WHERE isfinite(t.target_logret_5d_raw)) finite_raw_labels,
        count(*) FILTER(WHERE isfinite(t.target_logret_5d_raw) AND NOT t.target_valid) finite_but_not_tradable
        FROM features f LEFT JOIN targets t USING(ts_code,signal_date) GROUP BY 1 ORDER BY 1''').fetch_df()
    audit=dict(status='passed',duplicate_target_keys=int(dup),calendar_mismatches=int(bad),independent_price_checks=len(check),
        formula='ln(adjusted close at t+5 / adjusted open at t+1); market trading days',
        feature_timing='signal after t close; next-day open entry',future_tradability_filter=False,
        yearly_join_coverage=coverage.to_dict('records'),test_used=False)
    save_json(output/'label_audit.json',audit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds',nargs='+',type=int,default=list(range(2017,2024)))
    p.add_argument('--output-dir',default=str(DATA/'v2/ols_baseline_stable'))
    p.add_argument('--batch-rows',type=int,default=20000)
    args=p.parse_args()
    if sorted(set(args.folds))!=args.folds or not all(2017<=y<=2023 for y in args.folds):raise ValueError('Research folds 2017-2023 only')
    if not 1000<=args.batch_rows<=50000:raise ValueError('Invalid batch size')
    features=DATA/'v2/local_features_2011_2023_stable';output=Path(args.output_dir).resolve()
    if not output.is_relative_to(DATA/'v2'):raise ValueError('V2 output required')
    output.mkdir(parents=True,exist_ok=True)
    report=read_json(features/'feature_report.json');quality=read_json(features/'quality_summary.json')
    if report['rows']!=9316878 or report['test_used'] or quality['bound_flag_factor_years'] or read_json(features/'reference_parity.json')['status']!='passed':raise ValueError('Feature gates not passed')
    ff=sorted(features.glob('feature_year=*/month_*.parquet'))
    ff=[f for f in ff if 2011<=int(f.parent.name.split('=')[1])<=max(args.folds)]
    tf=[f for y in range(2011,max(args.folds)+1) for f in sorted((DATA/'targets_5d'/f'signal_year={y}').glob('*.parquet'))]
    names=['run_local_ols.py','streaming_ols.py','research_controls.py','feature_model.py','linear_losses.py','balanced_features.py','stable_rolling.py','build_balanced_features.py']
    manifest=dict(arguments=vars(args),scope='local_survivor_research',model='unweighted OLS with intercept; MSE; L2=0',
        preprocessing=dict(training_only=True,sample_rows=50000,sample='lowest deterministic hash(ts_code,signal_date,42)',
            min_coverage=.2,clip_quantiles=[.01,.99],impute='training sample median',scale='training sample mean/std',corr_threshold=.9995),
        training='ALL eligible training rows; finite raw labels, exit strictly before fold boundary',
        validation='predict every feature row; score finite raw labels with exit before validation year end',
        factor_source=str(features),test_used=False,simulation_run=False,automatic_parameter_selection=False,
        source_sha256={n:hashlib.sha256((_source_path('v2/'+n)).read_bytes()).hexdigest() for n in names},
        factor_provenance_sha256=hashlib.sha256((features/'repair_manifest.json').read_bytes()).hexdigest(),
        inputs=[dict(path=str(f),bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in ff+tf])
    mp=output/'manifest.json'
    if mp.exists() and read_json(mp)!=manifest:raise ValueError('Run source/config/input changed; choose new output')
    save_json(mp,manifest)
    # Conservative bound for pandas monthly frame, preprocessing sample, transforms and QR workspaces.
    budget=4*2**30+120000*500*8*5+50000*500*8*5+args.batch_rows*502*8*5
    if budget>psutil.virtual_memory().available*.8:raise MemoryError('Insufficient headroom for bounded monthly fitting')
    save_json(output/'memory_budget.json',dict(conservative_working_bytes=budget,available_bytes=psutil.virtual_memory().available,
        duckdb_limit='4GB',batch_rows=args.batch_rows,full_training_matrix_loaded=False))
    con=duckdb.connect();con.execute("SET memory_limit='4GB'");con.execute('SET threads=4');con.execute('SET temp_directory=?',[str(output/'_duckdb_tmp')])
    audit_labels(con,ff,tf,read_json(features/'build_config.json'),output)
    results=[]
    with threadpool_limits(limits=4):
        for year in args.folds:
            folder=output/f'validation_{year}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():results.append(read_json(folder/'result.json'));continue
            start=time.perf_counter();boundary=f'{year}-01-01';end=f'{year+1}-01-01'
            con.execute('''CREATE OR REPLACE TEMP VIEW train AS SELECT f.*,t.target_logret_5d_raw y
                FROM features f JOIN targets t USING(ts_code,signal_date)
                WHERE f.signal_date>=DATE '2011-01-01' AND f.signal_date<DATE '%s' AND t.exit_date<DATE '%s'
                AND isfinite(t.target_logret_5d_raw)'''%(boundary,boundary))
            train_count=con.execute('SELECT count(*),max(signal_date) last_signal FROM train').fetchone()
            if not (folder/'preprocessing.json').exists():
                sample=con.execute('SELECT * FROM train ORDER BY hash(ts_code,signal_date,42),ts_code,signal_date LIMIT 50000').fetch_df()
                stats=fit_preprocessing(sample,COLUMNS);stats['sampling_method']=manifest['preprocessing']['sample']
                save_json(folder/'preprocessing.json',stats);del sample
            stats=read_json(folder/'preprocessing.json');active=len(stats['active_names'])
            print(f'{year}: train={train_count[0]}, active={active}',flush=True)
            cp=folder/'qr_checkpoint.npz';fit=StreamingOLS.load(cp) if cp.exists() else StreamingOLS(active)
            train_files=[f for f in ff if int(f.parent.name.split('=')[1])<year]
            if not (folder/'model.json').exists():
                for index,path in enumerate(train_files):
                    if index<fit.completed_files:continue
                    con.read_parquet(str(path),hive_partitioning=False).create_view('month_features',replace=True)
                    frame=con.execute('''SELECT f.*,t.target_logret_5d_raw y FROM month_features f JOIN targets t USING(ts_code,signal_date)
                        WHERE t.exit_date<? AND isfinite(t.target_logret_5d_raw) ORDER BY f.signal_date,f.ts_code''',[boundary]).fetch_df()
                    for offset in range(0,len(frame),args.batch_rows):
                        batch=frame.iloc[offset:offset+args.batch_rows];fit.update(transform_features(batch,stats),batch.y.to_numpy())
                    fit.completed_files=index+1;fit.save(cp)
                    if (index+1)%12==0:print(f'{year}: fitted {index+1}/{len(train_files)} months, {fit.rows} rows',flush=True)
                if fit.rows!=train_count[0]:raise ValueError('Training row count mismatch')
                model=bundle_model(fit.finish(),stats);model['factor_provenance_sha256']=manifest['factor_provenance_sha256']
                save_json(folder/'model.json',model)
            model=read_json(folder/'model.json');predictions=[]
            for path in [f for f in ff if int(f.parent.name.split('=')[1])==year]:
                con.read_parquet(str(path),hive_partitioning=False).create_view('month_features',replace=True)
                frame=con.execute('''SELECT f.*,CASE WHEN t.exit_date<? AND isfinite(t.target_logret_5d_raw) THEN t.target_logret_5d_raw ELSE NULL END y
                    FROM month_features f LEFT JOIN targets t USING(ts_code,signal_date) ORDER BY f.signal_date,f.ts_code''',[end]).fetch_df()
                signals=frame[['ts_code','signal_date','volatility_20d','y']].copy()
                signals['prediction']=np.concatenate([predict_frame(model,frame.iloc[i:i+args.batch_rows]) for i in range(0,len(frame),args.batch_rows)])
                predictions.append(signals)
            signals=pd.concat(predictions,ignore_index=True);mask=np.isfinite(signals.y)
            _,ew=sample_weights(signals.signal_date.to_numpy(),signals.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
            metrics,daily=evaluate_common(signals.prediction[mask].to_numpy(),signals.y[mask].to_numpy(),signals.signal_date[mask].to_numpy(),ew[mask])
            save_frame(folder/'predictions.parquet',signals);save_frame(folder/'daily_metrics.parquet',pd.DataFrame(daily))
            result=dict(year=year,model='ols_unweighted',status='completed',scope=manifest['scope'],train_rows=fit.rows,
                train_last_signal=str(train_count[1]),train_exit_strictly_before=boundary,active_factors=active,
                prediction_rows=len(signals),unscored_rows=int((~mask).sum()),predictive_metrics=metrics,
                diagnostics=model['linear_model']['diagnostics'],elapsed_seconds=time.perf_counter()-start,
                test_used=False,simulation_run=False)
            save_json(folder/'result.json',result);results.append(result)
            save_json(output/'summary.json',dict(status='running',results=results,test_used=False,scope=manifest['scope']))
            print(f'{year} COMPLETE: Rank IC {metrics["mean_rank_ic"]:.6f}',flush=True)
    save_json(output/'summary.json',dict(status='completed',results=results,test_used=False,scope=manifest['scope'],simulation_run=False))
    con.close();print('ALL OLS FOLDS COMPLETE',flush=True)


if __name__=='__main__':main()
