"""Audit common metrics, weight conservation and frozen artifacts."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np,pandas as pd
from scipy.stats import rankdata
from research_controls import sample_weights,WeightConfig
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'quant_data/processed/v2/stock_weights_200k';BASE=OUT.parent/'multiyear_losses_200k'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,choices=range(2017,2024));args=parser.parse_args()
    manifest=read(OUT/'manifest.json')
    for n,h in manifest['sources'].items():assert hashlib.sha256((ROOT/'v2'/n).read_bytes()).hexdigest()==h
    audit=[]
    for year in ([args.year] if args.year else range(2017,2024)):
        folder=OUT/f'validation_{year}';assert (folder/'complete.json').exists()
        w=pd.read_parquet(folder/'training_weights.parquet');keys=pd.read_parquet(BASE/f'validation_{year}/sample_keys.parquet')
        pd.testing.assert_frame_equal(w[['ts_code','signal_date']],keys,check_dtype=False)
        np.testing.assert_allclose(w.groupby('signal_date').date_weight.sum(),w.groupby('signal_date').date_stock_weight.sum(),atol=1e-10)
        assert ((w.date_stock_stock_multiplier>=.5)&(w.date_stock_stock_multiplier<=2)).all()
        stats=read(BASE/f'validation_{year}/mse/model.json')['preprocessing']
        scale=read(BASE/f'validation_{year}/mse/model.json')['linear_model']['target_scale']
        ref=pd.read_parquet(BASE/f'validation_{year}/mse/predictions.parquet').sort_values(['signal_date','ts_code']).reset_index(drop=True)
        for group in ['date','date_stock']:
            expected,stock=sample_weights(w.signal_date.to_numpy(),w.volatility_20d.to_numpy(),WeightConfig(date_gamma=.75,volatility_power=1 if group=='date_stock' else 0))
            np.testing.assert_allclose(expected,w[group+'_weight']);np.testing.assert_allclose(stock,w[group+'_stock_multiplier'])
            for loss in ['mse','huber','mae']:
                sub=folder/(group+'_'+loss);model=read(sub/'model.json');linear=model['linear_model'];m=read(sub/'metrics.json')
                assert model['preprocessing']==stats and linear['target_scale']==scale and linear['diagnostics']['converged']
                assert linear['diagnostics']['training_rows']==200000 and linear['config']['l2']==0
                df=pd.read_parquet(sub/'predictions.parquet').sort_values(['signal_date','ts_code']).reset_index(drop=True)
                pd.testing.assert_frame_equal(ref[['signal_date','ts_code','y','volatility_20d']],df[['signal_date','ts_code','y','volatility_20d']])
                assert np.isfinite(df.prediction).all()
                _,ew=sample_weights(df.signal_date.to_numpy(),df.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
                df['ew']=ew;scored=df[np.isfinite(df.y)];daily=[];weighted=[]
                for _,g in scored.groupby('signal_date'):
                    if len(g)<20:continue
                    a=rankdata(g.prediction);b=rankdata(g.y);weights=g.ew.to_numpy();weights=weights/weights.sum()
                    daily.append(float(np.corrcoef(a,b)[0,1]));a=a-np.dot(weights,a);b=b-np.dot(weights,b)
                    weighted.append(float(np.dot(weights,a*b)/np.sqrt(np.dot(weights,a*a)*np.dot(weights,b*b))))
                np.testing.assert_allclose(np.nanmean(daily),m['mean_rank_ic'],atol=1e-12)
                np.testing.assert_allclose(np.nanmean(weighted),m['mean_weighted_rank_ic'],atol=1e-12)
                assert len(scored)==m['rows'] and len(daily)==m['days']
                np.testing.assert_allclose(np.sqrt(np.mean((scored.prediction-scored.y)**2)),m['rmse'],atol=1e-12)
        audit.append(dict(year=year,models=6,matched_keys=True,daily_weight_mass_preserved=True))
        print(f'{year}: six weighted models independently verified',flush=True)
    if args.year:
        print('SINGLE FOLD VERIFIED; full experiment not yet certified');return
    summary=read(OUT/'summary.json')
    (OUT/'verification.json').write_text(json.dumps(dict(new_models=42,uniform_controls=21,folds=audit,source_hashes=True,common_metrics_verified=True),indent=2),encoding='utf-8')
    lines=['# 股票训练权重七年对照','',
        '每折200,000行，2017–2023七折；新增42个加权模型，对照上一阶段21个等权模型。相同样本、预处理、目标尺度与超参数，未进行全量训练或未来独立测试。所有模型在同一验证股票集合和公共评价权重下比较。','',
        '| 模型 | 训练权重 | 普通Rank IC | 公共加权Rank IC | RMSE | MAE |','|---|---|---:|---:|---:|---:|']
    a=summary['aggregate']
    for loss in ['mse','huber','mae']:
        for group in ['uniform','date','date_stock']:
            r=a[group+'_'+loss];label=('OLS' if group=='uniform' else 'WLS') if loss=='mse' else loss
            lines.append(f'| {label} | {group} | {r["rank_ic"]:.5f} | {r["weighted_rank_ic"]:.5f} | {r["rmse"]:.5f} | {r["mae"]:.5f} |')
    lines+=['','## 股票权重的单独增量','',
        '以下比较date_stock减date，日期总权重保持一致；这是给定日期权重后的股票加权效果，未检验所有可能的日期与股票权重交互。',
        '', '| 损失 | 普通Rank IC增量 | 加权Rank IC增量 | 普通Rank IC改善年份数 |','|---|---:|---:|---:|']
    folds=pd.DataFrame(summary['folds'])
    for loss in ['mse','huber','mae']:
        p=folds[folds.loss==loss].pivot(index='year',columns='group',values='mean_rank_ic')
        d=a['date_stock_'+loss];b=a['date_'+loss]
        lines.append(f'| {loss} | {d["rank_ic"]-b["rank_ic"]:+.5f} | {d["weighted_rank_ic"]-b["weighted_rank_ic"]:+.5f} | {int((p.date_stock>p.date).sum())}/7 |')
    lines+=['','## 每年普通Rank IC','', '| 年份 | 损失 | 等权 | 日期权重 | 日期加股票权重 |','|---|---|---:|---:|---:|']
    for year in range(2017,2024):
        for loss in ['mse','huber','mae']:
            r=folds[(folds.year==year)&(folds.loss==loss)].set_index('group')
            lines.append(f'| {year} | {loss} | {r.loc["uniform","mean_rank_ic"]:.5f} | {r.loc["date","mean_rank_ic"]:.5f} | {r.loc["date_stock","mean_rank_ic"]:.5f} |')
    lines+=['','股票乘数范围0.5–2，指数1，日期gamma=0.75。仅使用信号日已知波动率，不使用未来收益。日期权重按抽样训练集当日数量计算，不是时间衰减。因子有效数量仍由原折预处理决定。',
        '', '独立复核42模型收敛、训练样本与验证键、同折目标尺度及预处理、日权重守恒、股票乘数范围，以及普通/公共加权Rank IC重算全部通过。',
        '', '这轮权重是训练损失权重，不是实际组合资金权重。预测分数改善不能直接证明盈利改善；所有模型必须沿用相同交易规则与成本进行连续回测。当前幸存者偏差和数据缺失限制保持不变。没有据此自动选择或替换最终模型。']
    lines+=['','## 权重覆盖与集中度','', '| 年份 | 无有效波动率占比 | 日期权重有效样本量 | 加股票权重有效样本量 |','|---|---:|---:|---:|']
    for year in range(2017,2024):
        r=read(OUT/f'validation_{year}/weight_audit.json')
        lines.append(f'| {year} | {r["invalid_volatility_rows"]/200000:.2%} | {r["date"]["effective_sample_size"]:.0f} | {r["date_stock"]["effective_sample_size"]:.0f} |')
    lines+=['','有效样本量为(sum w)^2/sum(w^2)，用于描述权重集中度，不代表金融时间序列的独立观测数。无有效波动率的行先赋原始中性乘数1，再共同归一；这些行不具备按波动率排序的依据。']
    (OUT/'STOCK_WEIGHT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(a,indent=2));print('VERIFIED')
if __name__=='__main__':main()
