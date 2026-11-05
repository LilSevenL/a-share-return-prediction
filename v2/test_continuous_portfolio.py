import unittest
import numpy as np
import pandas as pd
from cash_release_simulation import simulate_released_cash


class ContinuousPortfolioTests(unittest.TestCase):
    def fixture(self):
        dates=pd.bdate_range('2017-12-25',periods=12).to_numpy().astype('datetime64[D]')
        prices=np.full((1,12),100.);prices[0,5:]=110.
        panel=(dates,prices,np.ones((1,12),bool),np.tile(np.arange(12),(1,1)),{d:i for i,d in enumerate(dates)})
        orders=pd.DataFrame([dict(ts_code='A',code_idx=0,signal_date=pd.Timestamp(dates[i]),prediction=1.,entry_date=pd.Timestamp(dates[i+1]),
            scheduled_exit_date=pd.Timestamp(dates[i+5]),entry_adj_open=100.,entry_tradable=True,amount_yuan=1e10,volatility_20d=.02) for i in [0,5]])
        return orders,panel

    def test_cross_year_sale_reinvests_proceeds_without_reset(self):
        orders,panel=self.fixture();m,nav,t=simulate_released_cash(orders,panel,top_n=1,aum=1000,cost_bps=0,impact_coefficient=0)
        self.assertAlmostEqual(nav.holdings.iloc[4],200.)
        self.assertAlmostEqual(t.iloc[1].budget,220.)
        self.assertAlmostEqual(nav.nav.iloc[-1],1.042)
        np.testing.assert_allclose(nav.cash+nav.holdings,nav.nav*1000)

    def test_cross_year_unsold_position_cannot_be_spent(self):
        orders,panel=self.fixture();panel[3][0,5]=-1
        m,nav,t=simulate_released_cash(orders,panel,top_n=1,aum=1000,cost_bps=0,impact_coefficient=0)
        self.assertEqual(m['filled_orders'],1);self.assertEqual(m['unresolved_positions'],1)
        self.assertAlmostEqual(nav.cash.iloc[-1],800.);self.assertAlmostEqual(nav.holdings.iloc[-1],220.)


if __name__=='__main__':unittest.main()
