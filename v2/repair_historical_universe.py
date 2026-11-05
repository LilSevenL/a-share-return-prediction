"""Prepare a date-based V2 universe and resumable missing-history jobs.

No current-status L filter. All outputs are isolated. SDK HTTP errors are never
accepted as empty historical datasets. Download only on explicit --download.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from build_balanced_features import save_json,save_frame

INTERFACES=('daily','adj_factor','daily_basic','moneyflow','stk_limit')


def historical_universe(frames,start='20100101',end='20231231'):
    data=pd.concat(frames,ignore_index=True)
    data['list_date']=pd.to_datetime(data.list_date,format='%Y%m%d',errors='coerce')
    data['delist_date']=pd.to_datetime(data.delist_date,format='%Y%m%d',errors='coerce')
    conflicts=data.groupby('ts_code').list_date.nunique()
    if (conflicts>1).any():raise ValueError('Conflicting listing dates; review metadata before use')
    conflicts=data.groupby('ts_code').delist_date.nunique()
    if (conflicts>1).any():raise ValueError('Conflicting delisting dates; review metadata before use')
    # Prefer the row containing a known delisting date over an older active row.
    data=data.sort_values('delist_date',na_position='first').drop_duplicates('ts_code',keep='last')
    excluded=data[data.list_date.isna()|((data.list_status=='D')&data.delist_date.isna())].copy()
    data=data[~data.ts_code.isin(excluded.ts_code)]
    lo=pd.Timestamp(start);hi=pd.Timestamp(end)
    data=data[(data.list_date<=hi)&(data.delist_date.isna()|(data.delist_date>=lo))].copy()
    data['eligible_start']=data.list_date.clip(lower=lo)
    data['eligible_end']=data.delist_date.fillna(hi).clip(upper=hi)
    if (data.eligible_end<data.eligible_start).any():raise ValueError('Invalid listing interval')
    return data.sort_values('ts_code').reset_index(drop=True),excluded


def eligible_on(universe,day):
    day=pd.Timestamp(day)
    return universe[(universe.eligible_start<=day)&(universe.eligible_end>=day)]


class SafeSDK:
    def __init__(self):
        import main
        import tushare as ts
        key=os.environ.get(main.API_KEY_ENV_VAR,'').strip()
        if not key:raise RuntimeError('Configured credential unavailable; do not print or persist it')
        self.pro=ts.pro_api(key);self.pro._DataApi__token=key
        self.pro._DataApi__http_url='https://teajoin.com';self.pro._DataApi__timeout=20
        self.last=0.;self.last_http_status=None

    def fetch(self,interface,**params):
        import tushare.pro.client as client
        original=client.requests.post
        def checked(*a,**kw):
            response=original(*a,**kw);self.last_http_status=response.status_code
            if response.status_code>=400:raise RuntimeError(f'SDK_HTTP_{response.status_code}')
            return response
        for attempt in range(3):
            time.sleep(max(0,.25-(time.monotonic()-self.last)));self.last=time.monotonic()
            try:
                with patch.object(client.requests,'post',checked):
                    frame=getattr(self.pro,interface)(**params)
                if not isinstance(frame,pd.DataFrame):raise TypeError('Invalid SDK response')
                return frame
            except Exception:
                if self.last_http_status in (401,403) or attempt==2:
                    raise RuntimeError(f'API unavailable; interface={interface}; http={self.last_http_status}') from None
                time.sleep(attempt+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default=str(ROOT/'quant_data/processed/v2/historical_universe_repair'))
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args();output=Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT/'quant_data/processed/v2'):raise ValueError('V2 directory required')
    output.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'quant_data/combined/stock_basic.csv']+sorted((ROOT/'quant_data/extra_cache/stock_basic').glob('status_*.csv'))
    frames=[pd.read_csv(p,dtype=str) for p in sources]
    universe,excluded=historical_universe(frames)
    save_frame(output/'universe.parquet',universe);save_frame(output/'metadata_excluded.parquet',excluded)
    # Existing processed universe is explicitly limited to L. Missing additions
    # require data, not merely a change to the roster.
    old=set(frames[0].ts_code);added=universe[~universe.ts_code.isin(old)]
    jobs=[]
    for row in added.itertuples():
        for interface in INTERFACES:
            local=output/'downloads'/interface/f'{row.ts_code}.parquet'
            jobs.append(dict(ts_code=row.ts_code,interface=interface,start_date=row.eligible_start.strftime('%Y%m%d'),
                end_date=row.eligible_end.strftime('%Y%m%d'),path=str(local),status='downloaded_unvalidated' if local.exists() else 'missing'))
    save_json(output/'download_jobs.json',jobs)
    manifest=dict(status='prepared_pending_data',start_date='20100101',end_date='20231231',stocks=len(universe),
        additions=len(added),jobs=len(jobs),excluded_metadata=len(excluded),
        current_status_counts=universe.list_status.value_counts().to_dict(),
        selection_rule='list_date <= signal_date <= delist_date, with open-ended delist_date; not current status L',
        historical_completeness_certified=False,formal_training_ready=False,
        metadata_sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        note='Known cached listed/delisted identities only; relistings/code changes and full historical completeness still require audit')
    save_json(output/'manifest.json',manifest)
    print(f'Prepared {len(universe)} historical identities; {len(added)} additions; {len(jobs)} interface jobs',flush=True)
    if not args.download:return
    sdk=None
    try:
        sdk=SafeSDK()
        probe=sdk.fetch('daily',ts_code='000001.SZ',start_date='20230103',end_date='20230104')
        if probe.empty or not (probe.ts_code=='000001.SZ').all():raise RuntimeError('Known-trading control probe failed; batch not started')
        save_json(output/'access_check.json',dict(status='probe_passed',rows=len(probe),http=sdk.last_http_status))
        for index,job in enumerate(jobs):
            path=Path(job['path'])
            if path.exists():continue
            parts=[]
            # Annual windows remain safely below provider row caps.
            for year in range(int(job['start_date'][:4]),int(job['end_date'][:4])+1):
                lo=max(job['start_date'],f'{year}0101');hi=min(job['end_date'],f'{year}1231')
                data=sdk.fetch(job['interface'],ts_code=job['ts_code'],start_date=lo,end_date=hi)
                if data.empty:continue
                if len(data)>=5000 or not (data.ts_code==job['ts_code']).all():raise ValueError('Response cap/code mismatch')
                data['trade_date']=pd.to_datetime(data.trade_date,format='%Y%m%d',errors='raise')
                if not data.trade_date.between(pd.Timestamp(lo),pd.Timestamp(hi)).all():raise ValueError('Out-of-window response')
                parts.append(data)
            if not parts:
                job['status']='empty_requires_review';save_json(output/'download_jobs.json',jobs);continue
            data=pd.concat(parts,ignore_index=True)
            if data.duplicated(['ts_code','trade_date']).any():raise ValueError('Duplicate raw keys')
            save_frame(path,data.sort_values('trade_date'));job['status']='downloaded_pending_quality_check';job['rows']=len(data)
            save_json(output/'download_jobs.json',jobs)
            print(f'{index+1}/{len(jobs)} {job["interface"]} {job["ts_code"]} rows={len(data)}',flush=True)
        manifest['status']='download_stage_finished_requires_data_audit_and_rebuild';save_json(output/'manifest.json',manifest)
    except Exception as error:
        safe=dict(status='blocked_access_or_validation',http=sdk.last_http_status if sdk else None,
            error_type=type(error).__name__,batch_complete=False,empty_data_not_certified=True)
        save_json(output/'access_check.json',safe)
        manifest['status']='blocked_access_or_validation';save_json(output/'manifest.json',manifest)
        print(json.dumps(safe),flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
