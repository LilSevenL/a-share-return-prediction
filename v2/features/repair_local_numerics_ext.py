"""Extension copy for 2022-2026. Versioned narrow repair of nine unstable rolling columns; no API or labels."""

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import kurtosis as scipy_kurtosis
from stable_rolling import kurtosis,correlation,trend_r2
from balanced_features import COLUMNS
from build_balanced_features import ROOT,save_json,save_frame


def main():
    source=ROOT/'quant_data/processed/v2/local_features_2022_2026'
    dest=ROOT/'quant_data/processed/v2/local_features_2022_2026_stable'
    dest.mkdir(exist_ok=True)
    (dest/'quality').mkdir(exist_ok=True)
    if shutil.disk_usage(dest).free<19*2**30:raise RuntimeError('Insufficient disk space for separate repaired version')
    config=json.loads((source/'build_config.json').read_text(encoding='utf-8'))
    manifest=dict(parent=str(source),parent_manifest_sha256=hashlib.sha256((source/'build_config.json').read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['repair_local_numerics_ext.py','stable_rolling.py','balanced_features.py']},
        note='Window-local kurtosis (zero variance undefined), trend R2 and 3-day return/volume correlation; no interaction parents changed')
    mp=dest/'repair_manifest.json'
    if mp.exists() and json.loads(mp.read_text(encoding='utf-8'))!=manifest:raise ValueError('Repair source changed')
    save_json(mp,manifest);save_json(dest/'build_config.json',dict(config,postprocessing=manifest))
    cols=[f'return_kurtosis_{w}d' for w in (5,10,20,60)]+[f'trend_r2_{w}d' for w in (5,10,20,60)]+['return_volume_corr_3d']
    cache=dest/'corrections.parquet'
    if not cache.exists():
        con=duckdb.connect();con.execute("SET memory_limit='4GB'");con.execute('SET threads=4')
        for name in ('base','daily','adj_factor'):
            con.read_parquet([f['path'] for f in config['inputs'][name]],hive_partitioning=False).create_view(name)
        raw=con.execute('''SELECT b.ts_code,b.signal_date,d.close*a.adj_factor C,d.vol volume
            FROM base b JOIN daily d ON b.ts_code=d.ts_code AND b.signal_date=d.trade_date
            JOIN adj_factor a ON b.ts_code=a.ts_code AND b.signal_date=a.trade_date ORDER BY b.ts_code,b.signal_date''').fetch_df()
        con.close();calendar=pd.DatetimeIndex(pd.read_parquet(source/'calendar.parquet').signal_date,name='signal_date')
        outputs=[];checks=0
        for i,(code,part) in enumerate(raw.groupby('ts_code',sort=True)):
            dates=pd.DatetimeIndex(part.signal_date);cal=calendar[(calendar>=dates.min())&(calendar<=dates.max())]
            d=part.set_index('signal_date').reindex(cal);price=np.log(d.C.where(d.C>0));r=np.log((d.C/d.C.shift()).where(lambda x:x>0))
            vr=np.log((d.volume/d.volume.shift()).where(lambda x:x>0))
            values={f'return_kurtosis_{w}d':kurtosis(r,w) for w in (5,10,20,60)}
            values.update({f'trend_r2_{w}d':trend_r2(price,w) for w in (5,10,20,60)})
            values['return_volume_corr_3d']=correlation(r,vr,3)
            if i%100==0:
                for w in (5,10,20,60):
                    x=r.iloc[-w:].to_numpy()
                    if len(x)==w and np.isfinite(x).all() and np.var(x)>0:
                        np.testing.assert_allclose(values[f'return_kurtosis_{w}d'].iloc[-1],scipy_kurtosis(x,bias=False),rtol=1e-10,atol=1e-10);checks+=1
                    x=price.iloc[-w:].to_numpy()
                    if len(x)==w and np.isfinite(x).all() and np.var(x)>0:
                        np.testing.assert_allclose(values[f'trend_r2_{w}d'].iloc[-1],np.corrcoef(x,np.arange(w))[0,1]**2,rtol=1e-10,atol=1e-10);checks+=1
            result=pd.DataFrame(values,index=cal).loc[lambda x:x.index.isin(dates)&(x.index.year>=2022)].astype('float32')
            result.insert(0,'ts_code',code);outputs.append(result.reset_index())
            if (i+1)%500==0:print(f'Recomputed {i+1} stocks',flush=True)
        correction=pd.concat(outputs,ignore_index=True);del outputs,raw
        save_frame(cache,correction);save_json(dest/'independent_numeric_checks.json',dict(passed=checks,status='passed',method='SciPy unbiased kurtosis and NumPy correlation-squared on real stock windows'))
    else:correction=pd.read_parquet(cache)
    report=json.loads((source/'feature_report.json').read_text(encoding='utf-8'));all_counts=[];changed={c:0 for c in cols}
    keys=['ts_code','signal_date'];total=0
    for path in sorted(source.glob('feature_year=*/month_*.parquet')):
        target=dest/path.relative_to(source);year=int(path.parent.name.split('=')[1]);month=int(path.stem.split('_')[1]);key=f'{year}-{month:02d}'
        qp=dest/'quality'/f'{key}.json'
        if target.exists() and qp.exists():
            diag=json.loads(qp.read_text());all_counts.append(diag['finite_counts']);total+=diag['rows']
            for c,n in diag['repair_changed_cells'].items():changed[c]+=n
            continue
        frame=pd.read_parquet(path).set_index(keys)
        sub=correction.loc[(correction.signal_date.dt.year==year)&(correction.signal_date.dt.month==month)].set_index(keys)
        if len(frame)!=len(sub) or not frame.index.isin(sub.index).all() or not sub.index.is_unique:raise ValueError('Repair key mismatch')
        sub=sub.reindex(frame.index);month_changed={}
        for c in cols:
            n=int((~np.isclose(frame[c],sub[c],rtol=2e-5,atol=2e-7,equal_nan=True)).sum());changed[c]+=n;month_changed[c]=n
            frame[c]=sub[c]
        frame=frame.reset_index();a=frame[COLUMNS].to_numpy();finite=np.isfinite(a)
        if np.isinf(a).any() or frame[report['blocked_features']].notna().any().any():raise ValueError('Invalid repaired values')
        if frame.duplicated(keys).any():raise ValueError('Duplicate keys')
        frame['feature_valid_count']=finite.sum(1).astype('int16');save_frame(target,frame)
        counts=finite.sum(0);all_counts.append(counts);total+=len(frame)
        diag=dict(month=key,rows=len(frame),stocks=int(frame.ts_code.nunique()),finite_counts=counts.tolist(),
            missing_counts=(len(frame)-counts).tolist(),blocked_features=report['blocked_features'],
            min_values=[float(v) if np.isfinite(v) else None for v in frame[COLUMNS].min()],
            max_values=[float(v) if np.isfinite(v) else None for v in frame[COLUMNS].max()],
            duplicate_keys=0,infinite_cells=0,repair_changed_cells=month_changed)
        save_json(qp,diag);save_json(dest/'progress.json',dict(phase='numeric_repair',month=key,rows=total));print(f'Repaired {key}',flush=True)
    if total!=report['rows']:raise ValueError('Row loss')
    counts=np.sum(all_counts,axis=0)
    report.update(coverage={c:float(n/total) for c,n in zip(COLUMNS,counts)},numeric_repair=manifest,repair_changed_cells=changed)
    save_json(dest/'feature_report.json',report);save_json(dest/'progress.json',dict(phase='completed',rows=total))
    print(f'COMPLETE stable repair: {total} rows',flush=True)


if __name__=='__main__':main()
