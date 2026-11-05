"""Summarize completed fixed OLS folds; no model selection or extra data reads."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_balanced_features import ROOT,save_json


def main():
    root=ROOT/'quant_data/processed/v2/ols_baseline_stable'
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if summary['status']!='completed':raise ValueError('Full run is not complete')
    results=summary['results']
    if [r['year'] for r in results]!=manifest['arguments']['folds']:raise ValueError('Missing/extra folds')
    daily=[];out=[]
    for r in results:
        folder=root/f'validation_{r["year"]}';model=json.loads((folder/'model.json').read_text())
        p=pd.read_parquet(folder/'predictions.parquet')
        if len(p)!=r['prediction_rows'] or p.duplicated(['ts_code','signal_date']).any() or not np.isfinite(p.prediction).all():raise ValueError('Prediction output failed')
        scored=np.isfinite(p.y)
        if int(scored.sum())!=r['predictive_metrics']['rows']:raise ValueError('Scoring count mismatch')
        if r['train_rows']!=model['linear_model']['diagnostics']['training_rows']:raise ValueError('Training count mismatch')
        daily.append(pd.read_parquet(folder/'daily_metrics.parquet'))
        out.append(dict(year=r['year'],train_rows=r['train_rows'],active_factors=r['active_factors'],
            scored_rows=int(scored.sum()),**r['predictive_metrics'],rank=r['diagnostics']['rank'],
            design_columns=r['diagnostics']['design_columns'],condition_number=r['diagnostics']['condition_number']))
    all_daily=pd.concat(daily,ignore_index=True)
    n=sum(r['scored_rows'] for r in out)
    aggregate=dict(scored_rows=n,validation_days=len(all_daily),
        rmse=float(np.sqrt(sum(r['rows']*r['rmse']**2 for r in out)/n)),
        mae=float(sum(r['rows']*r['mae'] for r in out)/n),
        mean_daily_rank_ic=float(all_daily.rank_ic.mean()),mean_daily_weighted_rank_ic=float(all_daily.weighted_rank_ic.mean()),
        mean_daily_ic=float(all_daily.ic.mean()),positive_rank_ic_years=sum(r['mean_rank_ic']>0 for r in out))
    save_json(root/'verified_summary.json',dict(status='verified',years=out,aggregate=aggregate,
        scope=summary['scope'],test_used=False,simulation_run=False))
    lines=['# 本地修复数据OLS基准结果','',
        '已完成固定的2017–2023年七折扩展窗口。仅本地现有股票研究，仍有幸存者偏差；未使用2024确认期、2025以后Test，未做实际交易模拟。','',
        '模型为带截距的等权OLS，MSE、L2=0；全部合格训练行参与QR拟合。预处理参数仅由当折训练数据的固定50,000行样本估计，未按验证分数调参。','',
        '| 验证年 | 训练行数 | 活跃因子 | 评分行数 | 日均Rank IC | 日均加权Rank IC | RMSE | 数值秩/设计列数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    lines.extend(f"| {r['year']} | {r['train_rows']:,} | {r['active_factors']} | {r['scored_rows']:,} | {r['mean_rank_ic']:.5f} | {r['mean_weighted_rank_ic']:.5f} | {r['rmse']:.5f} | {r['rank']}/{r['design_columns']} |" for r in out)
    lines+=['',f"合计 {n:,} 条可评分预测、{len(all_daily):,} 个验证交易日；日均Rank IC {aggregate['mean_daily_rank_ic']:.5f}，日均加权Rank IC {aggregate['mean_daily_weighted_rank_ic']:.5f}；逐样本汇总RMSE {aggregate['rmse']:.5f}、MAE {aggregate['mae']:.5f}。",'',
        '总体IC按日汇总；RMSE/MAE按样本数汇总。原始目标为五日对数收益。验证期标签跨年或缺失的行仍保留预测，但不计入该年的评分。加权Rank IC采用统一事前波动率权重，OLS训练仍为等权。', '',
        '数值秩不足时SVD给出固定阈值下的最小范数最小二乘解；这是共线性诊断，不能将单个系数直接解释为稳定的经济关系。五日收益重叠，这里不报告独立样本假设下的显著性或交易Sharpe。', '',
        '标签审计：全部目标主键无重复、市场交易日对齐检查通过、1,000条标签与源价格独立复算通过。最终逐折预测主键、有限预测值、评分数和训练行数核对通过。', '',
        '下一步应先用相同股票池和固定交易规则做含成本的历史模拟，再评估MAE/Huber或股票权重的增量价值；预测相关性不代表可实现收益。', '',
        '详细文件：label_audit.json、manifest.json、verified_summary.json；每折validation_YYYY目录内有model.json、preprocessing.json、predictions.parquet、daily_metrics.parquet、result.json。']
    (root/'OLS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(aggregate))


if __name__=='__main__':main()
