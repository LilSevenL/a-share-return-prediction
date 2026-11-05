"""Create isolated lot-aware simulator without changing prior engines."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
p=Path(__file__).resolve().parent
s=(_source_path('v2/cash_release_simulation.py')).read_text(encoding='utf-8')
s=s.replace('impact_coefficient=.5):','impact_coefficient=.5,round_lots=False):')
needle='            stop=actual if actual>=0 else n;values=shares*close[code,entry:stop].astype(float)'
replacement='''            spent=budget
            if round_lots:
                raw_price=row.entry_raw_open
                if not np.isfinite(raw_price) or raw_price<=0:rec['status']='missing_raw_open';continue
                quantity=buy_quantity(row.ts_code,budget/(raw_price*(1+buy)),row.entry_date)
                if quantity<=0:rec['status']='below_minimum_quantity';continue
                notional=quantity*raw_price;spent=notional*(1+buy)
                assert spent<=budget+1e-7
                shares=notional/row.entry_adj_open
                rec.update(allocated_budget=budget,budget=spent,buy_raw_quantity=quantity,entry_raw_open=raw_price)
            stop=actual if actual>=0 else n;values=shares*close[code,entry:stop].astype(float)'''
assert needle in s;s=s.replace(needle,replacement).replace('cash[slot,entry:]-=budget','cash[slot,entry:]-=spent')
s+='''

def buy_quantity(code,affordable,entry_date):
    if code.endswith('.BJ'):
        if pd.Timestamp(entry_date)<pd.Timestamp('2021-11-15'):return 0
        minimum,step=100,1
    elif code.startswith('688') and code.endswith('.SH'):minimum,step=200,1
    elif code.endswith(('.SH','.SZ')):minimum,step=100,100
    else:return 0
    quantity=int(np.floor(affordable/step))*step
    return quantity if quantity>=minimum else 0
'''
(_source_path('v2/lot_execution_simulation.py')).write_text(s,encoding='utf-8')
print('Built isolated lot simulator')
