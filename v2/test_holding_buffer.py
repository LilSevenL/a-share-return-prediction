import unittest
import numpy as np
import pandas as pd
from holding_buffer import buffer_panel
from cash_release_simulation import simulate_released_cash


class HoldingBufferTests(unittest.TestCase):
    def fixture(self):
        dates=pd.bdate_range('2023-01-02',periods=16).to_numpy().astype('datetime64[D]')
        close=np.full((2,16),100.);next_sale=np.tile(np.arange(16),(2,1))
        panel=(dates,close,np.ones((2,16),bool),next_sale,{d:i for i,d in enumerate(dates)})
        rows=[]
        for i,day in enumerate(dates):
            for code in range(2):
                rows.append(dict(signal_date=pd.Timestamp(day),ts_code=str(code),code_idx=code,prediction=float((code==0) if i<9 else (code==1)),
                    entry_date=pd.Timestamp(dates[i+1]) if i+1<16 else pd.NaT,
                    scheduled_exit_date=pd.Timestamp(dates[i+5]) if i+5<16 else pd.NaT,
                    entry_adj_open=100.,entry_tradable=True,amount_yuan=1e10,volatility_20d=.02))
        return pd.DataFrame(rows),panel

    def test_prior_close_rank_and_fixed_review_interval(self):
        orders,panel=self.fixture();p,d=buffer_panel(orders,panel,1)
        self.assertEqual(d[0,5],10);self.assertEqual(p[3][0,5],10)
        # Rank on the execution date itself is unavailable for that close's decision.
        changed=orders.copy();changed.loc[(changed.signal_date==pd.Timestamp(panel[0][5]))&(changed.ts_code=='0'),'prediction']=-99
        q,_=buffer_panel(changed,panel,1);self.assertEqual(q[3][0,5],10)

    def test_exit_intent_persists_when_sale_blocked(self):
        orders,panel=self.fixture();panel[3][0,10]=12
        p,d=buffer_panel(orders,panel,1)
        self.assertEqual(d[0,5],10);self.assertEqual(p[3][0,5],12)

    def test_future_ranks_do_not_change_past_nav(self):
        orders,panel=self.fixture();p,_=buffer_panel(orders,panel,1)
        _,a,_=simulate_released_cash(orders,p,top_n=1,aum=1000,cost_bps=20,impact_coefficient=.5)
        changed=orders.copy();changed.loc[changed.signal_date>=pd.Timestamp(panel[0][9]),'prediction']*=-1
        q,_=buffer_panel(changed,panel,1)
        _,b,_=simulate_released_cash(changed,q,top_n=1,aum=1000,cost_bps=20,impact_coefficient=.5)
        np.testing.assert_allclose(a.nav.iloc[:10],b.nav.iloc[:10],atol=1e-12)


if __name__=='__main__':unittest.main()
