"""Causal features, bounded weights, fixed metrics, and execution-accounting regression tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
from balanced_features import technical_stock,finish_cross_section,COLUMNS,BASE_FEATURES
from research_controls import sample_weights,WeightConfig,fit_preprocessing,transform_features,evaluate_common
from simulation import simulate


class PipelineTests(unittest.TestCase):
    def test_weights_preserve_daily_mass_and_caps(self):
        dates=np.repeat([1,2],[30,60]);vol=np.geomspace(.00001,10,90)
        a,_=sample_weights(dates,vol,WeightConfig(volatility_power=0))
        b,s=sample_weights(dates,vol,WeightConfig(volatility_power=1))
        self.assertGreaterEqual(s.min(),.5);self.assertLessEqual(s.max(),2)
        for day in (1,2):
            self.assertAlmostEqual(s[dates==day].mean(),1)
            self.assertAlmostEqual(a[dates==day].sum(),b[dates==day].sum())
        order=np.random.default_rng(3).permutation(90)
        c,_=sample_weights(dates[order],vol[order],WeightConfig(volatility_power=1))
        np.testing.assert_allclose(c,b[order])

    def test_training_preprocessing_is_fixed(self):
        rng=np.random.default_rng(4);a=rng.normal(size=(120,3));a[:,2]=a[:,0]
        stats=fit_preprocessing(a,['a','b','duplicate'])
        self.assertEqual(stats['active_names'],['a','b'])
        first=transform_features(a[:1],stats)
        together=transform_features(np.r_[a[:1],np.full((1,3),1e10)],stats)
        np.testing.assert_array_equal(first,together[:1])

    def test_common_metrics_not_training_weights(self):
        y=np.arange(30,dtype=float);pred=y.copy();pred[:3]=pred[:3][::-1]
        metrics,_=evaluate_common(pred,y,np.zeros(30),np.ones(30))
        self.assertAlmostEqual(metrics['mean_rank_ic'],metrics['mean_weighted_rank_ic'])

    def fixture(self):
        rng=np.random.default_rng(1);n=320;index=pd.bdate_range('2020-01-01',periods=n,name='signal_date')
        d=pd.DataFrame({k:rng.uniform(.1,1,n) for k in BASE_FEATURES},index=index)
        c=100*np.exp(np.cumsum(rng.normal(0,.02,n)));d['C']=c;d['O']=c*np.exp(rng.normal(0,.01,n))
        d['H']=d[['C','O']].max(axis=1)*1.01;d['L']=d[['C','O']].min(axis=1)*.99
        d['raw_close']=d.C;d['raw_pre_close']=d.C.shift();d['volume']=np.arange(n)+100
        d['amount_yuan']=d.volume*d.C;d['source_volume_ratio']=1.
        for side in ('buy','sell'):
            for size in ('sm','md','lg','elg'):d[f'{side}_{size}_amount']=rng.uniform(1,10,n)
        for name in ('net_mf_amount','rm','market_median_turnover','dv_ttm','pe_ttm','total_mv','circ_mv'):d[name]=rng.uniform(.01,1,n)
        return d

    def test_feature_prefix_invariance_and_financial_gate(self):
        d=self.fixture();full=technical_stock(d);prefix=technical_stock(d.iloc[:280])
        self.assertEqual(list(full),COLUMNS);self.assertEqual(len(COLUMNS),500)
        np.testing.assert_allclose(full.iloc[:280],prefix,equal_nan=True)
        frame=full.reset_index();frame['ts_code']='synthetic'
        final,blocked=finish_cross_section(frame)
        self.assertEqual(len(blocked),51);self.assertTrue(final[blocked].isna().all().all())
        expected=np.log(d.C/d.C.shift()).tail(15).sum()
        self.assertAlmostEqual(full.logret_15d.iloc[-1],expected)

    def test_missing_market_session_breaks_new_rolling_window(self):
        d=self.fixture();d.loc[d.index[-8],:]=np.nan
        result=technical_stock(d)
        self.assertTrue(np.isnan(result.logret_15d.iloc[-1]))

    def test_repaired_legacy_windows_do_not_bridge_suspension(self):
        d=self.fixture();d.loc[d.index[-3],:]=np.nan
        result=technical_stock(d)
        for name in ('amount_ratio_5d','amount_ratio_20d','volume_ratio_20d','turnover_zscore_20d','net_mf_ratio_5d'):
            self.assertTrue(np.isnan(result[name].iloc[-1]),name)
        self.assertTrue(result.negative_earnings_flag.isna().all())

    def test_repaired_windows_match_complete_calendar_reference(self):
        d=self.fixture();result=technical_stock(d)
        self.assertAlmostEqual(result.amount_ratio_5d.iloc[-1],d.amount_yuan.iloc[-1]/d.amount_yuan.tail(5).mean()-1)
        self.assertAlmostEqual(result.volume_ratio_20d.iloc[-1],d.volume.iloc[-1]/d.volume.tail(20).mean()-1)
        self.assertAlmostEqual(result.net_mf_ratio_5d.iloc[-1],d.net_mf_ratio.tail(5).mean())

    def market(self):
        dates=pd.bdate_range('2023-01-02',periods=8).to_numpy().astype('datetime64[D]')
        prices=np.full((2,8),110.);sell=np.ones((2,8),bool);next_sale=np.tile(np.arange(8),(2,1))
        panel=(dates,prices,sell,next_sale,{d:i for i,d in enumerate(dates)})
        orders=pd.DataFrame([dict(signal_date=pd.Timestamp(dates[0]),ts_code=str(i),code_idx=i,prediction=2-i,
            entry_date=pd.Timestamp(dates[1]),scheduled_exit_date=pd.Timestamp(dates[5]),entry_adj_open=100.,
            entry_tradable=i==0,amount_yuan=1e10,volatility_20d=.02) for i in range(2)])
        return orders,panel

    def test_failed_order_cash_and_no_replacement(self):
        orders,panel=self.market()
        metrics,nav,records=simulate(orders,panel,top_n=2,aum=1000,cost_bps=0,impact_coefficient=0)
        # One of two orders in one of five sleeves earns 10%: total return 1%, not 2%.
        self.assertAlmostEqual(metrics['net_return'],.01)
        orders.loc[0,'entry_tradable']=False;orders.loc[1,'entry_tradable']=True
        metrics,_,records=simulate(orders,panel,top_n=1,aum=1000,cost_bps=0,impact_coefficient=0)
        self.assertEqual(metrics['filled_orders'],0);self.assertEqual(metrics['net_return'],0)

    def test_cost_capacity_and_delayed_exit(self):
        orders,panel=self.market();metrics,_,_=simulate(orders,panel,top_n=2,aum=1000,cost_bps=20,impact_coefficient=0)
        self.assertLess(metrics['net_return'],.01)
        orders['amount_yuan']=1
        metrics,_,_=simulate(orders,panel,top_n=2,aum=1000)
        self.assertEqual(metrics['filled_orders'],0)
        orders,panel=self.market();panel[3][0,5]=7
        _,_,records=simulate(orders,panel,top_n=2,aum=1000,cost_bps=0,impact_coefficient=0)
        self.assertEqual(records.iloc[0].actual_exit_date,str(panel[0][7]))


if __name__=='__main__':unittest.main()
