"""Matched OLS leave-one-family-out experiments, training-only fixed preprocessing."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import sys,json,gc
ROOT=_REPO_ROOT;sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from balanced_features import CATALOG,RECORDS
from research_controls import transform_features,sample_weights,WeightConfig,evaluate_common
import linear_losses_optimized as losses
from run_multiyear_losses import read,save,sha
DATA=ROOT/'quant_data/processed';BASE=DATA/'v2/multiyear_losses_200k';OUT=DATA/'v2/factor_ablation_ols_200k';FEATURES=DATA/'v2/local_features_2011_2023_stable'
def main():
    OUT.mkdir(exist_ok=True);family={r['name']:r['family'] for r in RECORDS}
    sources=[Path(__file__),_source_path('v2/research_controls.py'),_source_path('v2/linear_losses_optimized.py'),_source_path('v2/research_design/factor_catalog_500.json')]
    inputs=[p for y in range(2017,2024) for p in [BASE/f'validation_{y}/sample_x.npy',BASE/f'validation_{y}/sample_y.npy',BASE/f'validation_{y}/sample.json',BASE/f'validation_{y}/mse/model.json',BASE/f'validation_{y}/mse/predictions.parquet']]
    inputs += [p for y in range(2017,2024) for p in sorted((FEATURES/f'feature_year={y}').glob('*.parquet'))]
    manifest=dict(scope='OLS matched 200k leave-one-family-out historical development',families=CATALOG['families'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},inputs=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in inputs],l2=0,automatic_selection=False)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    save(OUT/'manifest.json',manifest)
    with threadpool_limits(limits=4):
        for year in range(2017,2024):
            folder=OUT/f'validation_{year}';folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():continue
            prior=BASE/f'validation_{year}';ref=read(prior/'mse/model.json');stats=ref['preprocessing'];names=ref['active_feature_names']
            sample=read(prior/'sample.json');assert sha(prior/'sample_x.npy')==sample['x_sha'] and sha(prior/'sample_y.npy')==sample['y_sha']
            x=np.load(prior/'sample_x.npy',mmap_mode='r');y=np.load(prior/'sample_y.npy');models={};specs={}
            for group in ['full']+list(CATALOG['families']):
                keep=np.array([i for i,n in enumerate(names) if group=='full' or family[n]!=group],dtype=int)
                dropped=len(names)-len(keep);specs[group]=dict(active_count=len(keep),removed_active_count=dropped,testable=group=='full' or dropped>0,indices=keep.tolist())
                if group!='full' and not dropped:continue
                print(f'{year}: fitting {group}, {len(keep)} inputs',flush=True)
                sub=folder/group;sub.mkdir(exist_ok=True)
                if (sub/'model.json').exists():models[group]=read(sub/'model.json');continue
                model=losses.fit_linear_loss(x[:,keep],y,config=losses.LossConfig(loss='mse',l2=0,tol=1e-5))
                if group=='full':np.testing.assert_allclose(np.r_[model['intercept'],model['coefficient']],np.r_[ref['linear_model']['intercept'],ref['linear_model']['coefficient']],atol=1e-8)
                models[group]=model;save(sub/'model.json',dict(**model,active_feature_names=[names[i] for i in keep]))
            save(folder/'specifications.json',specs);del x,y;gc.collect()
            labels=pd.read_parquet(prior/'mse/predictions.parquet',columns=['ts_code','signal_date','y','prediction']).rename(columns={'prediction':'reference_prediction'})
            parts=[]
            for path in sorted((FEATURES/f'feature_year={year}').glob('*.parquet')):
                f=pd.read_parquet(path);xx=transform_features(f,stats);s=f[['ts_code','signal_date','volatility_20d']].copy()
                for group,model in models.items():s[group]=losses.predict_linear_loss(model,xx[:,specs[group]['indices']])
                parts.append(s)
            df=pd.concat(parts,ignore_index=True).merge(labels,on=['ts_code','signal_date'],validate='one_to_one')
            assert len(df)==len(labels);np.testing.assert_allclose(df.full,df.reference_prediction,atol=1e-8)
            mask=np.isfinite(df.y);_,ew=sample_weights(df.signal_date.to_numpy(),df.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
            results={}
            for group in models:
                assert np.isfinite(df[group]).all()
                m,d=evaluate_common(df[group][mask].to_numpy(),df.y[mask].to_numpy(),df.signal_date[mask].to_numpy(),ew[mask])
                independent=[spearmanr(g[group],g.y).statistic for _,g in df[mask].groupby('signal_date') if len(g)>=20]
                np.testing.assert_allclose(np.nanmean(independent),m['mean_rank_ic'],atol=1e-12)
                results[group]=m;save(folder/group/'metrics.json',m);pd.DataFrame(d).to_parquet(folder/group/'daily_metrics.parquet',index=False)
            df.to_parquet(folder/'predictions.parquet',index=False);save(folder/'complete.json',dict(year=year,metrics=results,independent_rank_ic_verified=True,full_matches_reference=True))
            print(f'{year}: COMPLETE',flush=True);del parts,df,models,f,xx;gc.collect()
    summary={};lines=['# OLS因子组消融结果','', '七折2017–2023，每折同20万训练行、相同预处理与等权OLS。增量=完整模型减去移除组模型的Rank IC；正值表示该组在完整模型中的条件增量为正。','', '| 移除方向 | 候选数 | 实际移除数 | 完整模型Rank IC增量 | 正增量年份 |','|---|---:|---:|---:|---:|']
    for group,meta in CATALOG['families'].items():
        counts=[];deltas=[];full_daily=[];drop_daily=[];annual=[]
        for year in range(2017,2024):
            folder=OUT/f'validation_{year}';spec=read(folder/'specifications.json')[group];counts.append(spec['removed_active_count'])
            if not spec['testable']:continue
            a=pd.read_parquet(folder/'full/daily_metrics.parquet');b=pd.read_parquet(folder/group/'daily_metrics.parquet');assert a.date.equals(b.date)
            full_daily.extend(a.rank_ic);drop_daily.extend(b.rank_ic);annual.append(dict(year=year,delta=float(a.rank_ic.mean()-b.rank_ic.mean())))
        if not annual:lines.append(f'| {meta["name"]} | {meta["count"]} | 0 | 无可用输入，无法检验 | — |');summary[group]=dict(testable=False);continue
        delta=float(np.nanmean(full_daily)-np.nanmean(drop_daily));wins=sum(r['delta']>0 for r in annual)
        summary[group]=dict(testable=True,delta_rank_ic=delta,positive_years=wins,removed_counts=counts,annual=annual)
        lines.append(f'| {meta["name"]} | {meta["count"]} | {min(counts)}–{max(counts)} | {delta:+.5f} | {wins}/7 |')
    lines+=['','所有模型通过独立Spearman重算，完整OLS与前阶段模型预测一致。每折只改变活跃因子列；删除组后重新拟合。此处并非股票收益的因果贡献，也不保证删除负增量方向后组合一定改善。', '', '财务数据门控继续生效。结果仅为OLS、固定抽样与既有预处理下的历史开发诊断。下一步应对重点组使用鲁棒损失复验，再固定精简候选并进入统一交易回测。']
    save(OUT/'summary.json',summary);save(OUT/'verification.json',dict(folds=7,independent_rank_ic=True,full_prediction_parity=True))
    (OUT/'FACTOR_ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
