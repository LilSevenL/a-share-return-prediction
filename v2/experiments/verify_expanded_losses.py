"""Independent expanded-sample checks and comparison report."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=_REPO_ROOT;DATA=ROOT/'quant_data/processed/v2';OUT=DATA/'expanded_losses_500k'
NAMES=['full_huber','full_mae','compact_huber','compact_mae']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    summary=read(OUT/'summary.json');manifest=read(OUT/'manifest.json')
    assert summary['status']=='completed' and len(summary['folds'])==28
    pooled={name:dict(ic=[],sse=0.,sae=0.,rows=0) for name in NAMES}
    for n,h in manifest['sources'].items():assert hashlib.sha256((_source_path('v2/'+n)).read_bytes()).hexdigest()==h
    for year in range(2017,2024):
        folder=OUT/f'validation_{year}';assert (folder/'complete.json').exists()
        keys=pd.read_parquet(folder/'sample_keys.parquet');oldkeys=pd.read_parquet(DATA/f'multiyear_losses_200k/validation_{year}/sample_keys.parquet')
        assert len(keys)==500000 and not keys.duplicated(['ts_code','signal_date']).any()
        assert len(oldkeys.merge(keys,on=['ts_code','signal_date'],validate='one_to_one'))==200000
        ref=None;scale=None
        baseline=pd.read_parquet(DATA/f'multiyear_losses_200k/validation_{year}/huber/predictions.parquet',columns=['ts_code','signal_date','y','volatility_20d']).sort_values(['signal_date','ts_code']).reset_index(drop=True)
        for name in NAMES:
            sub=folder/name;model=read(sub/'model.json');lm=model['linear_model'];metrics=read(sub/'metrics.json')
            assert lm['diagnostics']['converged'] and lm['diagnostics']['training_rows']==500000
            if scale is None:scale=lm['target_scale']
            assert scale==lm['target_scale'] and lm['config']['l2']==0
            assert len(model['active_feature_names'])==(333 if name.startswith('compact') else 448)
            df=pd.read_parquet(sub/'predictions.parquet').sort_values(['signal_date','ts_code']).reset_index(drop=True)
            assert not df.duplicated(['ts_code','signal_date']).any()
            pd.testing.assert_frame_equal(baseline,df[['ts_code','signal_date','y','volatility_20d']])
            if ref is None:ref=df[['ts_code','signal_date','y','volatility_20d']]
            pd.testing.assert_frame_equal(ref,df[['ts_code','signal_date','y','volatility_20d']])
            assert np.isfinite(df.prediction).all();d=df[np.isfinite(df.y)]
            ic=[spearmanr(g.prediction,g.y).statistic for _,g in d.groupby('signal_date') if len(g)>=20]
            np.testing.assert_allclose(np.nanmean(ic),metrics['mean_rank_ic'],atol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.mean((d.prediction-d.y)**2)),metrics['rmse'],atol=1e-12)
            np.testing.assert_allclose(np.mean(np.abs(d.prediction-d.y)),metrics['mae'],atol=1e-12)
            p=pooled[name];p['ic'].extend(ic);p['sse']+=float(np.sum((d.prediction-d.y)**2));p['sae']+=float(np.sum(np.abs(d.prediction-d.y)));p['rows']+=len(d)
        print(f'{year}: verified',flush=True)
    for name,p in pooled.items():
        a=summary['aggregate'][name]
        assert a['rows']==p['rows'] and a['days']==len(p['ic'])
        np.testing.assert_allclose([a['mean_rank_ic'],a['rmse'],a['mae']],[np.nanmean(p['ic']),np.sqrt(p['sse']/p['rows']),p['sae']/p['rows']],atol=1e-12)
    lines=['# 50万样本稳定性验证','', '七折2017–2023，每折50万训练行，包含前阶段20万样本；预处理沿用训练期参数，所有四候选共用新样本目标尺度。属于历史开发扩样本验证。','', '| 候选 | 20万行Rank IC | 50万行Rank IC | 50万行RMSE | 50万行MAE |','|---|---:|---:|---:|---:|']
    old=read(DATA/'multiyear_losses_200k/summary.json')['aggregate'];compact=read(DATA/'compact_cm_200k/summary.json')['results']
    for name in NAMES:
        loss=name.split('_')[-1];baseline=next(r['removed_rank_ic'] for r in compact if r['loss']==loss) if name.startswith('compact') else old[loss]['mean_rank_ic']
        r=summary['aggregate'][name];lines.append(f'| {name} | {baseline:.5f} | {r["mean_rank_ic"]:.5f} | {r["rmse"]:.5f} | {r["mae"]:.5f} |')
    lines+=['','| 年份 | 完整Huber | 精简Huber | 完整MAE | 精简MAE |','|---|---:|---:|---:|---:|']
    f=pd.DataFrame(summary['folds']).pivot(index='year',columns='loss',values='mean_rank_ic')
    for year,r in f.iterrows():lines.append(f'| {year} | '+ ' | '.join(f'{r[n]:.5f}' for n in ['full_huber','compact_huber','full_mae','compact_mae'])+' |')
    lines+=['','## 比较边界','', '从20万到50万行时，目标尺度由新训练样本重新估计；Huber有效转折点也随之变化，因此前后差异包含样本量与尺度估计的共同影响。每个年度四候选共用同一目标尺度，完整与精简之间的对照仍保持一致。', '', '精简模型删除C类条件过滤收益35项及M类动量/反转80项；这两类来自前期历史开发筛选，其效果仍需要未来数据检验。']
    for loss in ['huber','mae']:
        delta=f['compact_'+loss]-f['full_'+loss]
        lines.append(f'- {loss}: 精简相对完整的年度Rank IC差值范围为 {delta.min():+.5f} 至 {delta.max():+.5f}，正向年份 {int((delta>0).sum())}/7。')
    lines+=['','28个模型收敛，样本嵌套、目标尺度一致、验证键与标签一致、独立Spearman重算通过。本轮尚未更新交易回测，之前20万样本的收益不可归到50万模型。仍非全量训练或真正未来独立验证，不能仅根据历史分数定版。']
    (OUT/'EXPANDED_SAMPLE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(models=28,training_rows_per_fold=500000,nested_sample=True,matched_validation=True,matched_200k_validation=True,independent_rank_ic=True,independent_aggregate=True,source_hashes=True),indent=2),encoding='utf-8')
    print(json.dumps(summary['aggregate'],indent=2))
if __name__=='__main__':main()
