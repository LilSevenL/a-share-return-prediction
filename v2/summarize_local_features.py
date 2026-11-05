"""Aggregate committed monthly quality diagnostics without loading full factor matrices."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from balanced_features import COLUMNS,RECORDS
from build_balanced_features import ROOT,save_json,save_frame
from audit_factors import factor_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir',default=str(ROOT/'quant_data/processed/v2/local_features_2011_2023'))
    args=p.parse_args();root=Path(args.features_dir).resolve()
    report=json.loads((root/'feature_report.json').read_text(encoding='utf-8'))
    config=json.loads((root/'build_config.json').read_text(encoding='utf-8'))
    if config['args']['end_year']>2023:raise ValueError('Development years only')
    rows=[];years=[];metadata=[];families={r['name']:r['family'] for r in RECORDS}
    for year in range(config['args']['start_year'],config['args']['end_year']+1):
        data=[]
        for month in range(1,13):
            key=f'{year}-{month:02d}';d=json.loads((root/'quality'/f'{key}.json').read_text(encoding='utf-8'))
            path=root/f'feature_year={year}'/f'month_{month:02d}.parquet';pf=pq.ParquetFile(path)
            if pf.metadata.num_rows!=d['rows'] or set(COLUMNS)-set(pf.schema_arrow.names):raise ValueError('Partition metadata mismatch')
            if d['duplicate_keys'] or d['infinite_cells']:raise ValueError('Failed monthly quality check')
            metadata.append(dict(path=str(path),rows=d['rows'],bytes=path.stat().st_size))
            data.append(d)
        n=sum(d['rows'] for d in data);counts=np.sum([d['finite_counts'] for d in data],axis=0)
        years.append(dict(year=year,rows=n,maximum_monthly_stocks=max(d['stocks'] for d in data),
            computable_columns=int((counts>0).sum())))
        for i,name in enumerate(COLUMNS):
            lows=[d['min_values'][i] for d in data if d['min_values'][i] is not None]
            highs=[d['max_values'][i] for d in data if d['max_values'][i] is not None]
            lo=min(lows) if lows else None;hi=max(highs) if highs else None
            lower,upper=factor_bounds(name,families[name])
            invalid=bool(lo is not None and ((lower is not None and lo<lower-1e-6) or (upper is not None and hi>upper+1e-6)))
            status='gated' if name in report['blocked_features'] else 'review' if invalid or counts[i]/n<.8 or lo==hi else 'basic_checks_passed'
            rows.append(dict(year=year,name=name,family=families[name],rows=n,finite=int(counts[i]),coverage=float(counts[i]/n),
                minimum=lo,maximum=hi,bounds_flag=invalid,status=status))
    annual=pd.DataFrame(rows);save_frame(root/'quality'/'annual_factor_quality.parquet',annual)
    if sum(y['rows'] for y in years)!=report['rows']:raise ValueError('Total row count mismatch')
    summary=dict(status='completed_basic_quality_checks',rows=report['rows'],stocks=report['stocks'],years=years,
        candidates=500,available_candidates=449,gated=51,partitions=len(metadata),output_bytes=sum(m['bytes'] for m in metadata),
        bound_flag_factor_years=int(annual.bounds_flag.sum()),review_factor_years=int((annual.status=='review').sum()),
        training_or_selection=False,test_used=False,survivor_bias_remains=True,
        limitations=['Ranges/coverage/key checks are not a complete independent formula audit',
            'Reference parity and numeric-repair checks are recorded separately; not every full-universe factor was independently recomputed',
            'Monthly/yearly diagnostics are not training preprocessing or alpha selection'],partitions_inventory=metadata)
    save_json(root/'quality_summary.json',summary)
    lines=['# 本地因子数据生成与质量汇总','',
        f"已完成 {report['rows']:,} 条股票日期、{report['stocks']} 只股票、{len(metadata)} 个月度分区。",'',
        '500个候选列，449项可计算、51项隔离。当前股票池有幸存者偏差，仅供本地已有股票样本研究；未训练、未调参、未使用Test。','',
        '| 年份 | 股票日期行数 | 月度股票数最大值 | 有数据的因子列 |','|---|---:|---:|---:|']
    lines.extend(f"| {y['year']} | {y['rows']:,} | {y['maximum_monthly_stocks']} | {y['computable_columns']} |" for y in years)
    lines.extend(['',f"数学范围提示：{summary['bound_flag_factor_years']} 个因子-年份；覆盖/常数/范围待复核：{summary['review_factor_years']} 个因子-年份。",'',
        '覆盖率80%为质量提示阈值，不是自动删因子的标准。长窗口预热、上市历史不足、字段缺失和隔离应分别解释；正式筛选只在训练折内完成。', '',
        '详细年度逐因子统计：quality/annual_factor_quality.parquet；每月范围和有效数：quality/YYYY-MM.json。', '',
        '本汇总检查已提交月份的行数、500列schema及生成时的重复键/无穷值检查，不代表已独立逐行复算全部数据。'])
    (root/'QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('partitions_inventory','limitations')},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
