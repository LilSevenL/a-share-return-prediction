import unittest
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from stable_rolling import kurtosis,correlation,trend_r2


class StableRollingTests(unittest.TestCase):
    def test_independent_window_formulas_and_prefix_invariance(self):
        rng=np.random.default_rng(2)
        a=pd.Series(rng.normal(size=400));b=pd.Series(rng.normal(size=400))
        a.iloc[30:45]=np.nan;a.iloc[80:90]=0
        for w in [5,10,20,60]:
            np.testing.assert_allclose(kurtosis(a,w),a.rolling(w).apply(lambda x:scipy_kurtosis(x,bias=False)),atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(kurtosis(a,w).iloc[260:],kurtosis(a.iloc[200:],w).iloc[60:],atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(trend_r2(a,w).iloc[-1],np.corrcoef(a.iloc[-w:],np.arange(w))[0,1]**2,atol=1e-12)
        np.testing.assert_allclose(correlation(a,b,3).iloc[-1],np.corrcoef(a.iloc[-3:],b.iloc[-3:])[0,1],atol=1e-12)

    def test_constant_and_missing_windows_are_undefined(self):
        a=pd.Series(np.ones(10));b=pd.Series(np.arange(10,dtype=float))
        self.assertTrue(kurtosis(a,5).isna().all())
        self.assertTrue(correlation(a,b,3).isna().all())
        self.assertTrue(trend_r2(a,5).isna().all())
        b.iloc[-1]=np.nan
        self.assertTrue(np.isnan(kurtosis(b,5).iloc[-1]))


if __name__=='__main__':unittest.main()
