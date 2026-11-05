"""Check full-build time-series factors against the repaired 80-stock reference."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import json
from pathlib import Path
import numpy as np
import pandas as pd
from balanced_features import RECORDS
from build_balanced_features import ROOT, save_json


def main():
    root=ROOT/'quant_data/processed/v2/local_features_2011_2023_stable'
    reference=ROOT/'quant_data/processed/v2/balanced_500_audit_repaired'
    # Cross-sectional interactions intentionally change when the stock pool changes.
    repaired={f'{prefix}_{w}d' for prefix in ('return_kurtosis','trend_r2') for w in (5,10,20,60)}|{'return_volume_corr_3d'}
    columns=[r['name'] for r in RECORDS if r['family']!='I' and r['name'] not in repaired]
    keys=['ts_code','signal_date']; rows=0; mismatches={}; missing_mismatches={}
    for path in sorted(reference.glob('feature_year=*/month_*.parquet')):
        expected=pd.read_parquet(path,columns=keys+columns).set_index(keys).sort_index()
        actual=pd.read_parquet(root/path.relative_to(reference),columns=keys+columns,
            filters=[('ts_code','in',expected.index.get_level_values('ts_code').unique().tolist())])
        actual=actual.set_index(keys).sort_index()
        if not actual.index.equals(expected.index): raise ValueError(f'Key mismatch: {path}')
        a=actual.to_numpy();b=expected.to_numpy()
        bad=(~np.isclose(a,b,rtol=2e-5,atol=2e-7,equal_nan=True)).sum(axis=0)
        missing=(np.isnan(a)!=np.isnan(b)).sum(axis=0)
        for c,n,m in zip(columns,bad,missing):
            if n:mismatches[c]=mismatches.get(c,0)+int(n)
            if m:missing_mismatches[c]=missing_mismatches.get(c,0)+int(m)
        rows+=len(actual)
    result=dict(rows=rows,columns=len(columns),mismatched_cells_by_factor=mismatches,
        missing_mask_mismatches=missing_mismatches,rtol=2e-5,atol=2e-7,
        excluded='I family: full-pool ranks differ; nine repaired rolling columns: checked separately against independent formulas',
        status='passed' if rows and not mismatches else 'review_required')
    save_json(root/'reference_parity.json',result)
    print(json.dumps(result,ensure_ascii=True))
    if result['status']!='passed': raise ValueError('Reference parity requires review')


if __name__=='__main__':main()
