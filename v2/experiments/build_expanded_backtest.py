"""Build an isolated runner retaining the previous execution implementation."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path

HERE=Path(__file__).resolve().parent
s=(_source_path('v2/run_compact_backtest.py')).read_text(encoding='utf-8')
start=s.index('DATA=');end=s.index('\ndef main():')
s=s[:start]+'''DATA=ROOT/'quant_data/processed';SOURCE=DATA/'v2/expanded_losses_500k';OUT=DATA/'v2/expanded_continuous_500k'

def signals(kind,loss):
    parts=[]
    for year in range(2017,2024):
        f=pd.read_parquet(SOURCE/f'validation_{year}/{kind}_{loss}/predictions.parquet',columns=['ts_code','signal_date','prediction','volatility_20d'])
        assert (f.signal_date.dt.year==year).all() and np.isfinite(f.prediction).all();parts.append(f)
    return pd.concat(parts,ignore_index=True)
''' +s[end:]
s=s.replace("assert read(COMPACT/'verification.json')['converged'];OUT.mkdir(exist_ok=True)","assert read(SOURCE/'verification.json')['models']==28;OUT.mkdir(exist_ok=True)")
s=s.replace("    files=[Path(__file__)]", "    prior=read(DATA/'v2/compact_continuous/manifest.json')\n    for path,digest in prior['sources'].items():\n        if path!='v2/run_compact_backtest.py':assert sha(ROOT/path)==digest, 'Execution engine changed: '+path\n    files=[Path(__file__)]")
start=s.index('    inputs +=');end=s.index('\n    for item in inputs:',start)
s=s[:start]+"    inputs += [dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for y in range(2017,2024) for p in [SOURCE/f'validation_{y}/{k}_{l}/predictions.parquet' for k in ['full','compact'] for l in ['huber','mae']]]"+s[end:]
s=s.replace("scope='same-sample historical development full versus joint CM removal'","training_rows_per_fold=500000,period=[2017,2023],scope='500k historical development full versus compact; fixed previous execution rules'")
s=s.replace('精简候选连续交易对照','50万样本连续交易对照').replace('每折模型均用20万行训练','每折模型均用50万行训练').replace('COMPACT_BACKTEST_REPORT.md','EXPANDED_BACKTEST_REPORT.md')
(_source_path('v2/run_expanded_backtest.py')).write_text(s,encoding='utf-8')
v=(_source_path('v2/verify_compact_backtest.py')).read_text(encoding='utf-8').replace("v2/compact_continuous'","v2/expanded_continuous_500k'")
(_source_path('v2/verify_expanded_backtest.py')).write_text(v,encoding='utf-8')
print('Built independent 500k backtest runner and verifier')
