"""Saved-artifact identity checks after completed robust ablation."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'quant_data/processed/v2/multiyear_losses_200k';OUT=BASE.parent/'robust_ablation_200k'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    manifest=read(OUT/'manifest.json');count=0
    for name,h in manifest['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    for year in range(2017,2024):
        folder=OUT/f'validation_{year}';assert read(folder/'complete.json')['independent_rank_ic']
        df=pd.read_parquet(folder/'predictions.parquet').sort_values(['signal_date','ts_code']).reset_index(drop=True)
        assert not df.duplicated(['signal_date','ts_code']).any()
        specs=read(folder/'specifications.json')
        for loss in ['huber','mae']:
            ref=read(BASE/f'validation_{year}/{loss}/model.json')
            labels=pd.read_parquet(BASE/f'validation_{year}/{loss}/predictions.parquet').sort_values(['signal_date','ts_code']).reset_index(drop=True)
            pd.testing.assert_frame_equal(df[['signal_date','ts_code','y','volatility_20d']],labels[['signal_date','ts_code','y','volatility_20d']])
            for group in ['C','M','F']:
                model=read(folder/f'{group}_{loss}/model.json');diag=model['diagnostics']
                assert model['active_feature_names']==specs[group]['active_names']
                assert diag['converged'] and diag['training_rows']==200000
                assert model['target_scale']==ref['linear_model']['target_scale']
                assert len(model['coefficient'])==len(specs[group]['indices'])
                assert np.isfinite(df[group+'_'+loss]).all();count+=1
    result=dict(models=count,source_hashes=True,validation_keys_and_labels_identical=True,shared_target_scale=True,active_names_verified=True)
    (OUT/'artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
if __name__=='__main__':main()
