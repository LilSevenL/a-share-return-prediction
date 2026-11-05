"""Derive independent larger-sample experiment, leaving prior provenance intact."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
def main():
    root=Path(__file__).resolve().parent;s=(_source_path('v2/run_multiyear_losses.py')).read_text(encoding='utf-8')
    s=s.replace('import duckdb,numpy as np,pandas as pd','import duckdb,numpy as np,pandas as pd\nimport copy\nfrom balanced_features import RECORDS')
    s=s.replace('multiyear_losses_200k','expanded_losses_500k').replace('run_multiyear_losses.py','run_expanded_losses.py')
    s=s.replace('200000','500000').replace('matched 200k','matched 500k')
    s=s.replace("['mse','huber','mae']","['full_huber','full_mae','compact_huber','compact_mae']")
    s=s.replace("x=losses.CachedDesign(x0);y=np.load(folder/'sample_y.npy')","x=x0;y=np.load(folder/'sample_y.npy')")
    s=s.replace("config=losses.LossConfig(loss=name,", "config=losses.LossConfig(loss=name.split('_')[-1],")
    old="""                try:
                    model=fit(x,y,config,sub/'checkpoint.npz',run_id+str(year)+sample['x_sha']+sample['y_sha'],"""
    new="""                model_stats=copy.deepcopy(stats)
                if name.startswith('compact'):
                    excluded={r['name'] for r in RECORDS if r['family'] in ('C','M')}
                    model_stats['active']=[a and n not in excluded for a,n in zip(stats['active'],stats['names'])]
                    model_stats['active_names']=[n for a,n in zip(model_stats['active'],stats['names']) if a]
                    for n in excluded:model_stats['dropped'][n]='precommitted_CM_removal'
                keep=[i for i,n in enumerate(stats['active_names']) if n in model_stats['active_names']]
                fit_x=losses.CachedDesign(x[:,keep])
                try:
                    model=fit(fit_x,y,config,sub/'checkpoint.npz',run_id+str(year)+name+sample['x_sha']+sample['y_sha'],"""
    assert old in s;s=s.replace(old,new)
    s=s.replace('models[name]=bundle_model(model,stats);','del fit_x;gc.collect()\n                models[name]=bundle_model(model,model_stats);')
    old="""                con.register('sample_keys',keys)"""
    new="""                old_keys=pd.read_parquet(DATA/f'v2/multiyear_losses_200k/validation_{year}/sample_keys.parquet')
                assert len(old_keys.merge(keys[['ts_code','signal_date']],on=['ts_code','signal_date'],validate='one_to_one'))==len(old_keys)
                con.register('sample_keys',keys)"""
    assert old in s;s=s.replace(old,new)
    (_source_path('v2/run_expanded_losses.py')).write_text(s,encoding='utf-8')
if __name__=='__main__':main()
