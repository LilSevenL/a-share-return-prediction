import unittest
import numpy as np
import pandas as pd
import test_research_pipeline as fixtures
from simulation import simulate
from cash_release_simulation import simulate_released_cash


class CashReleaseTests(unittest.TestCase):
    def test_no_delay_matches_old_policy(self):
        orders,panel=fixtures.PipelineTests().market()
        a,an,_=simulate(orders,panel,top_n=2,aum=1000,cost_bps=20,impact_coefficient=0)
        b,bn,t=simulate_released_cash(orders,panel,top_n=2,aum=1000,cost_bps=20,impact_coefficient=0)
        np.testing.assert_allclose(an.nav,bn.nav)
        self.assertAlmostEqual(a['total_cost_amount'],b['total_cost_amount'])
        np.testing.assert_allclose(bn.cash+bn.holdings,bn.nav*1000)

    def test_delayed_lot_does_not_freeze_other_cash(self):
        orders,panel=fixtures.PipelineTests().market();orders.entry_tradable=True;panel[3][0,5]=-1
        later=orders.iloc[[1]].copy();later.signal_date=pd.Timestamp(panel[0][5]);later.entry_date=pd.Timestamp(panel[0][6]);later.scheduled_exit_date=pd.Timestamp(panel[0][7])
        all_orders=pd.concat([orders,later],ignore_index=True)
        old,_,_=simulate(all_orders,panel,top_n=2,aum=1000,cost_bps=0,impact_coefficient=0)
        new,nav,trades=simulate_released_cash(all_orders,panel,top_n=2,aum=1000,cost_bps=0,impact_coefficient=0)
        self.assertEqual(old['filled_orders'],2);self.assertEqual(new['filled_orders'],3)
        self.assertEqual(new['unresolved_positions'],1)
        self.assertAlmostEqual(trades.iloc[-1].budget,55.)
        np.testing.assert_allclose(nav.nav.iloc[-1],1.0255)
        self.assertGreaterEqual(nav.cash.min(),0)


if __name__=='__main__':unittest.main()
