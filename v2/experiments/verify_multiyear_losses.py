"""Independent saved-artifact checks and report; refuses incomplete experiments."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=_REPO_ROOT;OUT=ROOT/'quant_data/processed/v2/multiyear_losses_200k'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    summary=read(OUT/'summary.json');manifest=read(OUT/'manifest.json')
    for n,h in manifest['sources'].items():assert hashlib.sha256((_source_path('v2/'+n)).read_bytes()).hexdigest()==h
    rows=[]
    for year in range(2017,2024):
        folder=OUT/f'validation_{year}';assert (folder/'complete.json').exists()
        reference=None;stats=None;scale=None
        for loss in ['mse','huber','mae']:
            sub=folder/loss;model=read(sub/'model.json');m=read(sub/'metrics.json');linear=model['linear_model']
            assert linear['config']['l2']==0 and linear['diagnostics']['converged'] and linear['diagnostics']['training_rows']==200000
            if stats is None:stats=model['preprocessing'];scale=linear['target_scale']
            assert model['preprocessing']==stats and linear['target_scale']==scale
            df=pd.read_parquet(sub/'predictions.parquet').sort_values(['signal_date','ts_code']).reset_index(drop=True)
            assert not df.duplicated(['signal_date','ts_code']).any() and np.isfinite(df.prediction).all()
            if reference is None:reference=df[['signal_date','ts_code','y']].copy()
            pd.testing.assert_frame_equal(reference,df[['signal_date','ts_code','y']])
            scored=df[np.isfinite(df.y)]
            daily=[spearmanr(g.prediction,g.y).statistic for _,g in scored.groupby('signal_date') if len(g)>=20]
            assert len(scored)==m['rows'] and len(daily)==m['days']
            np.testing.assert_allclose(np.nanmean(daily),m['mean_rank_ic'],atol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.mean((scored.prediction-scored.y)**2)),m['rmse'],atol=1e-12)
            rows.append(dict(year=year,loss=loss,**m))
    check=dict(models=21,folds=7,independent_rank_ic=True,matched_validation_keys_and_labels=True,
        same_fold_preprocessing_and_target_scale=True,all_converged=True,source_hashes_match=True)
    (OUT/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    table=pd.DataFrame(rows).pivot(index='year',columns='loss',values='mean_rank_ic')
    lines=['# 多年份损失函数对照结果','',
        '2017–2023历史开发验证，共21个模型；每折固定20万训练样本，同折同预处理、等权、L2=0。未进行全量鲁棒损失训练或未来独立测试，也未基于结果自动选模型。','',
        '| 验证年 | OLS Rank IC | Huber Rank IC | MAE Rank IC |','|---|---:|---:|---:|']
    for year,r in table.iterrows():lines.append(f'| {year} | {r["mse"]:.5f} | {r["huber"]:.5f} | {r["mae"]:.5f} |')
    lines+=['','| 模型 | 日均Rank IC | 加权Rank IC | RMSE | MAE |','|---|---:|---:|---:|---:|']
    for loss in ['mse','huber','mae']:
        r=summary['aggregate'][loss];lines.append(f'| {loss} | {r["mean_rank_ic"]:.5f} | {r["mean_weighted_rank_ic"]:.5f} | {r["rmse"]:.5f} | {r["mae"]:.5f} |')
    for loss in ['huber','mae']:
        wins=int((table[loss]>table['mse']).sum());lines+=['',f'{loss} 的年度日均Rank IC在{wins}/7个年份高于同样本OLS。该计数是描述统计，不是独立性检验或显著性结论。']
    lines+=['','汇总Rank IC按有效交易日平均，误差指标按评分行数汇总。不同损失函数共用同一评价权重，未用各模型训练权重定义不同评分口径。',
        '', '复核通过：21模型收敛、同折训练尺度与预处理、验证键和标签一致，独立Spearman重算与已保存Rank IC一致。原V1及全量OLS未覆盖。',
        '', '本阶段只回答跨年份预测稳定性；扣除成本后的交易表现仍须统一连续回测。幸存者偏差、缺失财务因子和日线交易模拟限制仍存在。下一阶段可固定股票权重对照方案，最终与损失函数及因子消融一同进入统一交易评价。']
    (OUT/'MULTIYEAR_LOSS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(table.to_string());print(json.dumps(summary['aggregate'],indent=2));print('VERIFIED')

if __name__=='__main__':main()
