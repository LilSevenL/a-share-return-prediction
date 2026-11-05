"""Check saved continuous experiment completeness and accounting outputs."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=_REPO_ROOT;OUT=ROOT/'quant_data/processed/v2/expanded_continuous_500k'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    manifest=read(OUT/'manifest.json');summary=read(OUT/'summary.json')
    assert summary['status']=='completed' and len(summary['results'])==36
    for path,h in manifest['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    seen=set();dates=None
    for r in summary['results']:
        key=(r['kind'],r['loss'],r['policy'],r['scenario']);assert key not in seen;seen.add(key)
        stem='_'.join(key);nav=pd.read_parquet(OUT/f'{stem}_nav.parquet');trades=pd.read_parquet(OUT/f'{stem}_orders.parquet')
        assert r['audit']['status']=='passed'
        assert np.isfinite(nav.nav).all() and (nav.nav>0).all() and nav.cash.min()>-1e-4
        np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*1e8)
        np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
        np.testing.assert_allclose(nav.nav.iloc[-1]-1,r['net_return'])
        np.testing.assert_allclose(np.prod([1+b['net_return'] for b in r['audit']['annual_boundaries']]),nav.nav.iloc[-1])
        assert len(trades)>0
    result=dict(cases=36,unique_cases=True,source_hashes=True,saved_nav_accounting=True,annual_compounding=True)
    (OUT/'artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
if __name__=='__main__':main()
