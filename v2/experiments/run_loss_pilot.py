"""Matched 200k-row first-fold engineering comparison; not full-sample validation."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import sys
from pathlib import Path
ROOT=_REPO_ROOT;sys.path.insert(0,str(ROOT))
import json
import hashlib
import time
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import linear_losses as losses
from research_controls import transform_features,sample_weights,WeightConfig,evaluate_common
from feature_model import bundle_model,predict_frame
from simulation import DATA,source_files,prepare_market
from cash_release_simulation import simulate_released_cash
from holding_buffer import buffer_panel
from build_balanced_features import save_json,save_frame


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    out=DATA/'v2/loss_pilot_2017';out.mkdir(exist_ok=True)
    base=DATA/'v2/ols_baseline_stable/validation_2017';features=DATA/'v2/local_features_2011_2023_stable'
    stats=read(base/'preprocessing.json')
    ff=[p for y in range(2011,2018) for p in sorted((features/f'feature_year={y}').glob('*.parquet'))]
    tf=[Path(p) for p in source_files(DATA/'targets_5d','signal_year',range(2011,2018))]
    prior=read(DATA/'v2/ols_backtest_stable/manifest.json')
    market_inputs=[r for r in prior['inputs'] if 'year=2017' in r['path'] or r['path'].endswith('stock_basic.csv')]
    inputs=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in ff+tf+[base/'preprocessing.json',base/'predictions.parquet']]
    inputs+=market_inputs
    source_names=['run_loss_pilot.py','linear_losses.py','research_controls.py','feature_model.py','balanced_features.py','cash_release_simulation.py','holding_buffer.py','simulation.py','build_balanced_features.py']
    source_paths=[_source_path('v2/'+n) for n in source_names]+[_source_path('backtest_validation.py')]
    manifest=dict(scope='matched-sample engineering pilot only',train_years=[2011,2016],validation_year=2017,fit_rows=200000,
        sample='lowest hash(ts_code,signal_date,20261025); tie-break ts_code,date',losses=['mse','huber','mae'],l2=0.,weights='uniform',
        delta=1.35,tol=1e-5,max_iter=6000,inputs=inputs,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        test_used=False,automatic_winner_selection=False,full_sample_ols_comparison=False)
    mp=out/'manifest.json'
    if mp.exists() and read(mp)!=manifest:raise ValueError('Configuration changed; use a new run directory')
    save_json(mp,manifest)
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='4GB'")
    con.read_parquet([str(p) for p in ff],hive_partitioning=False).create_view('features')
    con.read_parquet([str(p) for p in tf],hive_partitioning=False).create_view('targets')
    if not (out/'sample_complete.json').exists():
        keys=con.execute('''SELECT f.ts_code,f.signal_date,t.target_logret_5d_raw y FROM features f JOIN targets t USING(ts_code,signal_date)
            WHERE f.signal_date<DATE '2017-01-01' AND t.exit_date<DATE '2017-01-01' AND isfinite(t.target_logret_5d_raw)
            ORDER BY hash(f.ts_code,f.signal_date,20261025),f.ts_code,f.signal_date LIMIT 200000''').fetch_df()
        if len(keys)!=200000 or keys.duplicated(['ts_code','signal_date']).any():raise ValueError('Wrong training sample')
        con.register('sample_keys',keys)
        frame=con.execute('SELECT f.*,k.y FROM features f JOIN sample_keys k USING(ts_code,signal_date) ORDER BY f.signal_date,f.ts_code').fetch_df()
        x=transform_features(frame,stats);y=frame.y.to_numpy(dtype=float)
        np.save(out/'sample_x.npy',x);np.save(out/'sample_y.npy',y)
        save_frame(out/'sample_keys.parquet',frame[['ts_code','signal_date']])
        save_json(out/'sample_complete.json',dict(rows=len(y),features=x.shape[1],x_sha256=hashlib.sha256((out/'sample_x.npy').read_bytes()).hexdigest(),
            y_sha256=hashlib.sha256((out/'sample_y.npy').read_bytes()).hexdigest()))
        del frame,x,y
    x=np.load(out/'sample_x.npy',mmap_mode='r');y=np.load(out/'sample_y.npy')
    print(f'Prepared shared sample {x.shape}; all losses L2=0',flush=True)
    models={};timings={}
    with threadpool_limits(limits=4):
        for name in manifest['losses']:
            folder=out/name;folder.mkdir(exist_ok=True)
            if (folder/'model.json').exists():models[name]=read(folder/'model.json');continue
            print(f'Fitting {name}',flush=True);start=time.perf_counter();counter={'transpose_passes':0,'objective_calls':0}
            original_at=losses._at;original_objective=losses.objective_gradient
            def traced_at(*args,**kwargs):
                result=original_at(*args,**kwargs);counter['transpose_passes']+=1
                if counter['transpose_passes']%150==0:print(f'{name}: {counter["transpose_passes"]} transpose passes, {time.perf_counter()-start:.1f}s',flush=True)
                return result
            def traced_objective(*args,**kwargs):
                result=original_objective(*args,**kwargs);counter['objective_calls']+=1
                if counter['objective_calls']%25==0:print(f'{name}: {counter["objective_calls"]} objective calls, gradient={np.max(np.abs(result[1])):.3g}',flush=True)
                return result
            losses._at=traced_at;losses.objective_gradient=traced_objective
            try:
                model=losses.fit_linear_loss(x,y,config=losses.LossConfig(loss=name,l2=0.,delta=1.35,max_iter=6000,tol=1e-5))
            except Exception as error:
                save_json(folder/'failure.json',dict(error=str(error),seconds=time.perf_counter()-start,counters=counter));raise
            finally:losses._at=original_at;losses.objective_gradient=original_objective
            bundle=bundle_model(model,stats);bundle['experiment_scope']=manifest['scope'];models[name]=bundle
            save_json(folder/'model.json',bundle);save_json(folder/'fit_timing.json',dict(seconds=time.perf_counter()-start,**counter))
            print(f'{name} CONVERGED: {model["diagnostics"]}',flush=True)
        # Validation labels and pool are exactly the same as the completed OLS fold.
        label_reference=pd.read_parquet(base/'predictions.parquet',columns=['ts_code','signal_date','y'])
        predictions={name:[] for name in models}
        for path in sorted((features/'feature_year=2017').glob('*.parquet')):
            frame=pd.read_parquet(path)
            for name,model in models.items():
                s=frame[['ts_code','signal_date','volatility_20d']].copy();s['prediction']=predict_frame(model,frame)
                predictions[name].append(s)
        results=[];market=None;order_template=None;ranking_results=[]
        for name in manifest['losses']:
            folder=out/name;signals=pd.concat(predictions[name],ignore_index=True).merge(label_reference,on=['ts_code','signal_date'],validate='one_to_one')
            mask=np.isfinite(signals.y);_,ew=sample_weights(signals.signal_date.to_numpy(),signals.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
            metrics,daily=evaluate_common(signals.prediction[mask].to_numpy(),signals.y[mask].to_numpy(),signals.signal_date[mask].to_numpy(),ew[mask])
            save_frame(folder/'predictions.parquet',signals);save_frame(folder/'daily_metrics.parquet',pd.DataFrame(daily))
            save_json(folder/'predictive_metrics.json',metrics);ranking_results.append(dict(loss=name,**metrics))
            if market is None:
                order_template,market=prepare_market(signals.drop(columns='y'),2017);order_template=order_template.drop(columns='prediction')
            orders=order_template.merge(signals[['ts_code','signal_date','prediction']],on=['ts_code','signal_date'],validate='one_to_one')
            for policy,threshold in [('fixed5',None),('keep50',50),('keep100',100)]:
                selected=buffer_panel(orders,market,threshold)[0] if threshold else market
                for scenario,cost,impact in [('gross',0.,0.),('cost20',20.,.5),('cost50',50.,.5)]:
                    m,nav,trades=simulate_released_cash(orders,selected,cost_bps=cost,impact_coefficient=impact)
                    assert np.allclose(nav.cash+nav.holdings,nav.nav*1e8) and nav.cash.min()>=-1e-5
                    m.update(loss=name,policy=policy,scenario=scenario,year=2017,scope=manifest['scope'])
                    stem=f'{policy}_{scenario}';save_json(folder/f'{stem}.json',m);save_frame(folder/f'{stem}_nav.parquet',nav);save_frame(folder/f'{stem}_orders.parquet',trades);results.append(m)
            print(f'{name} evaluation complete: RankIC={metrics["mean_rank_ic"]:.6f}',flush=True)
    con.close();save_json(out/'summary.json',dict(status='completed',scope=manifest['scope'],predictive_metrics=ranking_results,simulations=results,test_used=False,automatic_winner_selection=False))
    lines=['# 首折损失函数试运行','',
        '仅2017验证、2011–2016训练期固定200,000行，三模型同样本同预处理；448因子、等权、L2=0。不是全量训练或七年连续业绩。未按结果选loss或参数。','',
        '| Loss | 日均Rank IC | 加权Rank IC | RMSE | MAE | 迭代数 | 收敛 |','|---|---:|---:|---:|---:|---:|---|']
    for r in ranking_results:
        diag=models[r['loss']]['linear_model']['diagnostics']
        lines.append(f"| {r['loss']} | {r['mean_rank_ic']:.5f} | {r['mean_weighted_rank_ic']:.5f} | {r['rmse']:.5f} | {r['mae']:.5f} | {diag['iterations']} | {diag['converged']} |")
    lines+=['','| Loss | 退出规则 | 零成本净收益 | 20bp加冲击净收益 | 50bp加冲击净收益 | 20bp最大回撤 |','|---|---|---:|---:|---:|---:|']
    lookup={(r['loss'],r['policy'],r['scenario']):r for r in results}
    for name in manifest['losses']:
        for policy in ['fixed5','keep50','keep100']:
            rr=[lookup[name,policy,c] for c in ['gross','cost20','cost50']]
            lines.append(f'| {name} | {policy} | '+' | '.join(f"{r['net_return']:.2%}" for r in rr)+f" | {rr[1]['max_drawdown']:.2%} |")
    lines+=['','Huber delta=1.35×同一训练样本MAD尺度；MAE使用未平滑绝对损失与ADMM。收敛诊断、目标尺度、训练用时在各loss目录。损失不同会改变条件均值/中位数等预测目标，RMSE与MAE需同时查看。','',
        '27组模拟均只覆盖2017并采用同样本边界处理，不与之前全量OLS或跨年连续净值直接归因比较。日线撮合、幸存者偏差、缺失涨跌停和冲击代理等限制继续存在；首折结果不能用作最终模型选择。']
    (out/'LOSS_PILOT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('PILOT COMPLETE',flush=True)


if __name__=='__main__':main()
