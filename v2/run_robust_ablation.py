"""Matched robust-loss rechecks of C/M/F family removals."""
from pathlib import Path
import sys,gc,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from balanced_features import RECORDS
from research_controls import transform_features,sample_weights,WeightConfig,evaluate_common
import linear_losses_optimized as losses
from resumable_losses import fit
from run_multiyear_losses import read,save,sha
DATA=ROOT/'quant_data/processed';BASE=DATA/'v2/multiyear_losses_200k';OUT=DATA/'v2/robust_ablation_200k';FEATURES=DATA/'v2/local_features_2011_2023_stable'
GROUPS=['C','M','F'];LOSSES=['huber','mae']
def main():
    OUT.mkdir(exist_ok=True);family={r['name']:r['family'] for r in RECORDS}
    paths=[Path(__file__)]+[ROOT/'v2'/n for n in ['resumable_losses.py','linear_losses_optimized.py','research_controls.py','balanced_features.py','research_design/factor_catalog_500.json']]
    inputs=[p for y in range(2017,2024) for p in [BASE/f'validation_{y}/sample_x.npy',BASE/f'validation_{y}/sample_y.npy',BASE/f'validation_{y}/sample.json']]
    inputs += [p for y in range(2017,2024) for loss in LOSSES for p in [BASE/f'validation_{y}/{loss}/model.json',BASE/f'validation_{y}/{loss}/predictions.parquet']]
    inputs += [p for y in range(2017,2024) for p in sorted((FEATURES/f'feature_year={y}').glob('*.parquet'))]
    manifest=dict(groups=GROUPS,losses=LOSSES,sample_rows=200000,l2=0,delta=1.35,tol=1e-5,max_iter=6000,
        sources={str(p.relative_to(ROOT)):sha(p) for p in paths},inputs=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in inputs],
        scope='OLS-informed historical development follow-up; no automatic selection')
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    save(OUT/'manifest.json',manifest);run_id=sha(OUT/'manifest.json')
    with threadpool_limits(limits=4):
        for year in range(2017,2024):
            folder=OUT/f'validation_{year}';folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():continue
            prior=BASE/f'validation_{year}';ref=read(prior/'huber/model.json');stats=ref['preprocessing'];names=ref['active_feature_names']
            sample=read(prior/'sample.json');assert sha(prior/'sample_x.npy')==sample['x_sha'] and sha(prior/'sample_y.npy')==sample['y_sha']
            x0=np.load(prior/'sample_x.npy',mmap_mode='r');y=np.load(prior/'sample_y.npy');models={};spec={}
            for group in GROUPS:
                keep=[i for i,n in enumerate(names) if family[n]!=group];spec[group]=dict(indices=keep,active_names=[names[i] for i in keep],removed=len(names)-len(keep))
                subx=x0[:,keep];x=losses.CachedDesign(subx)
                for loss in LOSSES:
                    key=group+'_'+loss;sub=folder/key;sub.mkdir(exist_ok=True)
                    if (sub/'model.json').exists():models[key]=read(sub/'model.json');continue
                    print(f'{year} {key}: fitting {len(keep)} inputs',flush=True)
                    config=losses.LossConfig(loss=loss,l2=0,delta=1.35,tol=1e-5,max_iter=6000)
                    try:
                        model=fit(x,y,config,sub/'checkpoint.npz',run_id+str(year)+key+sample['x_sha']+sample['y_sha'],lambda step:print(f'{year} {key}: {step}',flush=True))
                    except Exception as e:save(sub/'failure.json',dict(error=str(e)));raise
                    models[key]=model;save(sub/'model.json',dict(**model,active_feature_names=spec[group]['active_names']))
                    print(f'{year} {key}: CONVERGED',flush=True)
                del x,subx;gc.collect()
            save(folder/'specifications.json',spec);del x0,y;gc.collect()
            labels=pd.read_parquet(prior/'huber/predictions.parquet',columns=['ts_code','signal_date','y'])
            parts=[]
            for path in sorted((FEATURES/f'feature_year={year}').glob('*.parquet')):
                frame=pd.read_parquet(path);xx=transform_features(frame,stats);s=frame[['ts_code','signal_date','volatility_20d']].copy()
                for key,model in models.items():s[key]=losses.predict_linear_loss(model,xx[:,spec[key.split('_')[0]]['indices']])
                parts.append(s)
            df=pd.concat(parts,ignore_index=True).merge(labels,on=['ts_code','signal_date'],validate='one_to_one');assert len(df)==len(labels)
            mask=np.isfinite(df.y);_,ew=sample_weights(df.signal_date.to_numpy(),df.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
            for key in models:
                assert np.isfinite(df[key]).all()
                m,d=evaluate_common(df[key][mask].to_numpy(),df.y[mask].to_numpy(),df.signal_date[mask].to_numpy(),ew[mask])
                direct=[spearmanr(g[key],g.y).statistic for _,g in df[mask].groupby('signal_date') if len(g)>=20]
                np.testing.assert_allclose(np.nanmean(direct),m['mean_rank_ic'],atol=1e-12)
                sub=folder/key;save(sub/'metrics.json',m);pd.DataFrame(d).to_parquet(sub/'daily_metrics.parquet',index=False)
            df.to_parquet(folder/'predictions.parquet',index=False);save(folder/'complete.json',dict(independent_rank_ic=True,models=list(models)))
            print(f'{year}: COMPLETE',flush=True);del df,parts,models,frame,xx;gc.collect()
    results=[]
    lines=['# Huber与MAE重点因子组复验','', '2017–2023七折，每折20万行，相同训练样本和预处理、等权，移除组后重新拟合。此次组别根据OLS历史结果确定，属于开发复验。增量=完整模型减移除组模型。','', '| 方向 | 损失 | 完整Rank IC | 移除后Rank IC | 保留组增量 | 正增量年份 |','|---|---|---:|---:|---:|---:|']
    for group in GROUPS:
        for loss in LOSSES:
            full=[];drop=[];annual=[]
            for year in range(2017,2024):
                a=pd.read_parquet(BASE/f'validation_{year}/{loss}/daily_metrics.parquet');b=pd.read_parquet(OUT/f'validation_{year}/{group}_{loss}/daily_metrics.parquet')
                assert a.date.equals(b.date);full.extend(a.rank_ic);drop.extend(b.rank_ic);annual.append(dict(year=year,delta=float(a.rank_ic.mean()-b.rank_ic.mean())))
            a=float(np.nanmean(full));b=float(np.nanmean(drop));wins=sum(r['delta']>0 for r in annual)
            results.append(dict(group=group,loss=loss,full_rank_ic=a,removed_rank_ic=b,delta=a-b,positive_years=wins,annual=annual))
            lines.append(f'| {group} | {loss} | {a:.5f} | {b:.5f} | {a-b:+.5f} | {wins}/7 |')
    lines+=['','C=条件筛选收益，M=动量反转，F=资金流。所有42个模型收敛并独立重算Rank IC，日级评分日期与完整模型一致。预测与逐日指标已保存。','', '本轮尚未测试同时删除多组，也未评价扣费交易表现。单组增量不能相加；要精简需另行训练组合候选并统一回测。']
    for p,h in manifest['sources'].items():assert sha(ROOT/p)==h
    save(OUT/'summary.json',dict(results=results,scope=manifest['scope']));save(OUT/'verification.json',dict(models=42,converged=True,independent_rank_ic=True,source_hashes_match=True))
    (OUT/'ROBUST_ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
