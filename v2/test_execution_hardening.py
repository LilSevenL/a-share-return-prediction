import unittest
import numpy as np,pandas as pd
from lot_execution_simulation import buy_quantity,simulate_released_cash
from strict_execution_market import strict_panel

class Tests(unittest.TestCase):
    def test_board_quantity_boundaries(self):
        for code,q,expected in [('600000.SH',199.9,100),('000001.SZ',99.9,0),('688001.SH',199.9,0),('688001.SH',201.9,201),('830001.BJ',101.9,101)]:
            self.assertEqual(buy_quantity(code,q,'2023-01-01'),expected)
        self.assertEqual(buy_quantity('830001.BJ',101,'2020-01-01'),0)
    def test_rounding_leaves_cash_and_uses_raw_price(self):
        dates=pd.bdate_range('2023-01-02',periods=7).to_numpy().astype('datetime64[D]');lookup={d:i for i,d in enumerate(dates)}
        panel=(dates,np.full((1,7),20.),np.ones((1,7),bool),np.arange(7)[None,:],lookup)
        orders=pd.DataFrame([dict(ts_code='600000.SH',code_idx=0,signal_date=pd.Timestamp(dates[0]),prediction=1.,entry_date=pd.Timestamp(dates[1]),scheduled_exit_date=pd.Timestamp(dates[5]),entry_adj_open=20.,entry_raw_open=10.,entry_tradable=True,amount_yuan=1e9,volatility_20d=.02)])
        _,nav,t=simulate_released_cash(orders,panel,top_n=1,aum=7500,cost_bps=0,impact_coefficient=0,round_lots=True)
        self.assertEqual(t.iloc[0].buy_raw_quantity,100);self.assertEqual(t.iloc[0].budget,1000)
        self.assertAlmostEqual(nav.cash.iloc[1],6500);self.assertAlmostEqual(nav.holdings.iloc[1],1000)
        self.assertAlmostEqual(nav.nav.iloc[-1],1)
    def test_unknown_or_blocked_limit_delays_exit(self):
        dates=np.arange('2023-01-01','2023-01-05',dtype='datetime64[D]');panel=(dates,np.ones((1,4)),np.ones((1,4),bool),None,{d:i for i,d in enumerate(dates)})
        evidence=pd.DataFrame(dict(trade_date=pd.to_datetime(dates),code_idx=[0]*4,strict_sell=[True,False,True,False]))
        result=strict_panel(panel,evidence)
        np.testing.assert_array_equal(result[3],[[0,2,2,-1]])
if __name__=='__main__':unittest.main()
