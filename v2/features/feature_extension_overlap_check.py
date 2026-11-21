"""Feature extension 2024-2026: overlap gate. Compare 2022-2023 of the new
stable build with local_features_2011_2023_stable. Read-only; no labels.
Tolerance fixed by plan: |new-old| <= 1e-6 * max(1, |old|); missing masks equal.
Exit code 1 on any mismatch (all months are still compared and reported).
"""

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from balanced_features import COLUMNS
from build_balanced_features import ROOT,save_json

V2=ROOT/'quant_data/processed/v2'
NEW=V2/'local_features_2022_2026_stable';OLD=V2/'local_features_2011_2023_stable'
NEW_PARENT=V2/'local_features_2022_2026';OLD_PARENT=V2/'local_features_2011_2023'
OUT=V2/'feature_extension_checks'
KEYS=['ts_code','signal_date']


def close(new,old):
    """Return mask-mismatch and value-mismatch boolean arrays."""
    nn=np.isnan(new);on=np.isnan(old);mask=nn!=on
    both=~nn&~on;diff=np.zeros_like(old,dtype=np.float64)
    diff[both]=np.abs(new[both].astype(np.float64)-old[both].astype(np.float64))
    tol=1e-6*np.maximum(1.0,np.abs(np.where(both,old,0).astype(np.float64)))
    return mask,both&(diff>tol),diff


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocked=json.loads((OLD/'feature_report.json').read_text(encoding='utf-8'))['blocked_features']
    blocked_new=json.loads((NEW/'feature_report.json').read_text(encoding='utf-8'))['blocked_features']
    result=dict(tolerance='|new-old| <= 1e-6*max(1,|old|); NaN masks must match',blocked_same=sorted(blocked)==sorted(blocked_new),months=[])
    # Market context on overlapping dates.
    m_new=pd.read_parquet(NEW_PARENT/'market.parquet').set_index('signal_date')
    m_old=pd.read_parquet(OLD_PARENT/'market.parquet').set_index('signal_date')
    dates=m_old.index[(m_old.index>='2022-01-01')&(m_old.index<'2024-01-01')]
    missing_dates=int((~dates.isin(m_new.index)).sum())
    market={}
    for c in ('rm','market_median_turnover'):
        a=m_new[c].reindex(dates).to_numpy(np.float64);b=m_old[c].reindex(dates).to_numpy(np.float64)
        mask,bad,diff=close(a,b)
        market[c]=dict(dates=len(dates),mask_mismatch=int(mask.sum()),value_mismatch=int(bad.sum()),max_abs_diff=float(diff.max()))
    market['missing_dates_in_new']=missing_dates
    c_new=pd.read_parquet(NEW_PARENT/'calendar.parquet').signal_date;c_old=pd.read_parquet(OLD_PARENT/'calendar.parquet').signal_date
    co=c_old[(c_old>='2022-01-01')&(c_old<'2024-01-01')];cn=c_new[(c_new>='2022-01-01')&(c_new<'2024-01-01')]
    market['calendar_same']=bool(len(co)==len(cn) and (co.to_numpy()==cn.to_numpy()).all())
    result['market']=market
    col_stats={c:dict(mask_mismatch=0,value_mismatch=0,max_abs_diff=0.0,dates=set(),stocks=set()) for c in COLUMNS}
    key_ok=True;rows=0;valid_count_mismatch=0;blocked_nonnull=0
    for year in (2022,2023):
        for month in range(1,13):
            name=f'feature_year={year}/month_{month:02d}.parquet'
            a=pd.read_parquet(NEW/name);b=pd.read_parquet(OLD/name)
            same_cols=list(a.columns)==list(b.columns)
            same_keys=len(a)==len(b) and (a.ts_code.to_numpy()==b.ts_code.to_numpy()).all() and (a.signal_date.to_numpy()==b.signal_date.to_numpy()).all()
            m=dict(month=f'{year}-{month:02d}',rows_new=len(a),rows_old=len(b),same_columns=same_cols,same_keys_and_order=bool(same_keys))
            result['months'].append(m)
            if not (same_cols and same_keys):
                key_ok=False
                ka=set(zip(a.ts_code,a.signal_date));kb=set(zip(b.ts_code,b.signal_date))
                m.update(only_new=len(ka-kb),only_old=len(kb-ka),same_set=ka==kb)
                print('KEY MISMATCH',m,flush=True);continue
            rows+=len(a)
            blocked_nonnull+=int(a[blocked].notna().sum().sum()+b[blocked].notna().sum().sum())
            valid_count_mismatch+=int((a.feature_valid_count.to_numpy()!=b.feature_valid_count.to_numpy()).sum())
            A=a[COLUMNS].to_numpy(np.float32);B=b[COLUMNS].to_numpy(np.float32)
            mask,bad,diff=close(A,B);wrong=mask|bad
            for j in np.flatnonzero(wrong.any(0)|(diff.max(0)>0)):
                s=col_stats[COLUMNS[j]];s['mask_mismatch']+=int(mask[:,j].sum());s['value_mismatch']+=int(bad[:,j].sum())
                s['max_abs_diff']=max(s['max_abs_diff'],float(diff[:,j].max()))
                idx=np.flatnonzero(wrong[:,j])
                if len(idx):
                    s['dates'].update(a.signal_date.iloc[idx].dt.strftime('%Y-%m-%d').unique()[:50]);s['stocks'].update(a.ts_code.iloc[idx].unique()[:200])
            print(f'Compared {year}-{month:02d}: {len(a)} rows, mismatching columns so far {sum(1 for s in col_stats.values() if s["mask_mismatch"] or s["value_mismatch"])}',flush=True)
    bad_cols={c:dict(mask_mismatch=s['mask_mismatch'],value_mismatch=s['value_mismatch'],max_abs_diff=s['max_abs_diff'],
        dates_sample=sorted(s['dates'])[:20],first_date=min(s['dates']) if s['dates'] else None,last_date=max(s['dates']) if s['dates'] else None,
        stocks_count_sample=len(s['stocks']),stocks_sample=sorted(s['stocks'])[:20]) for c,s in col_stats.items() if s['mask_mismatch'] or s['value_mismatch']}
    max_diff_all={c:s['max_abs_diff'] for c,s in col_stats.items() if s['max_abs_diff']>0}
    market_ok=market['missing_dates_in_new']==0 and market['calendar_same'] and all(market[c]['mask_mismatch']==0 and market[c]['value_mismatch']==0 for c in ('rm','market_median_turnover'))
    result.update(rows_compared=rows,keys_ok=key_ok,market_ok=market_ok,blocked_nonnull_cells=blocked_nonnull,
        feature_valid_count_mismatch_rows=valid_count_mismatch,mismatching_columns=bad_cols,
        columns_with_nonzero_diff_within_tolerance={c:v for c,v in max_diff_all.items() if c not in bad_cols},
        passed=bool(key_ok and market_ok and not bad_cols and blocked_nonnull==0 and result['blocked_same']))
    save_json(OUT/'overlap_check.json',result)
    print(json.dumps(dict(passed=result['passed'],rows=rows,keys_ok=key_ok,market_ok=market_ok,mismatching_columns=len(bad_cols),
        valid_count_mismatch=valid_count_mismatch,nonzero_within_tol=len(result['columns_with_nonzero_diff_within_tolerance'])),ensure_ascii=False),flush=True)
    sys.exit(0 if result['passed'] else 1)


if __name__=='__main__':main()
