
# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import unittest
import numpy as np
import pandas as pd
import test_research_pipeline as fixtures
from simulation import simulate


class SimulationAccountingTests(unittest.TestCase):
    def test_cost_ledger_and_post_exit_missing_price(self):
        orders,panel=fixtures.PipelineTests().market()
        panel[1][0,6:]=np.nan
        m,nav,trades=simulate(orders,panel,top_n=1,aum=1000,cost_bps=20,impact_coefficient=0)
        t=trades.iloc[0]
        self.assertAlmostEqual(t.buy_notional+t.buy_cost_amount,t.budget)
        self.assertAlmostEqual(nav.nav.iloc[-1],(800+t.sell_notional-t.sell_cost_amount)/1000)
        self.assertAlmostEqual(m['total_cost_amount'],t.buy_cost_amount+t.sell_cost_amount)

    def test_unresolved_holding_blocks_sleeve_and_is_marked(self):
        orders,panel=fixtures.PipelineTests().market();panel[3][0,5:]=-1
        later=orders.copy();later['signal_date']=pd.Timestamp(panel[0][5]);later['entry_date']=pd.Timestamp(panel[0][6]);later['scheduled_exit_date']=pd.Timestamp(panel[0][7])
        m,_,trades=simulate(pd.concat([orders,later]),panel,top_n=1,aum=1000,cost_bps=0,impact_coefficient=0)
        self.assertEqual(m['blocked_sleeve_days'],1);self.assertEqual(m['unresolved_positions'],1)
        self.assertEqual(m['filled_orders'],1);self.assertEqual(trades.iloc[0].sell_notional,0)


if __name__=='__main__':unittest.main()
