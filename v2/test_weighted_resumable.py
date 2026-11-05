import tempfile,unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import linear_losses_optimized as base
from weighted_resumable_losses import fit
from research_controls import WeightConfig,sample_weights

class WeightedTests(unittest.TestCase):
    def test_weighted_fit_and_resume(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(200,8));x[:,1]=x[:,0]+.01*x[:,1]
        y=rng.normal(size=200);y[:20]+=10;w=rng.uniform(.1,3,200);w[:10]=0
        for loss in ['mse','huber','mae']:
            cfg=base.LossConfig(loss=loss,l2=.01,tol=1e-7,max_iter=15000)
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'state.npz';expected=base.fit_linear_loss(x,y,w,cfg)
                def interrupt(step):raise InterruptedError()
                if loss!='mse':
                    with self.assertRaises(InterruptedError):fit(x,y,cfg,p,'test',interrupt,5,w)
                    with self.assertRaises(ValueError):fit(x,y,cfg,p,'test',sample_weight=w+1)
                actual=fit(x,y,cfg,p,'test',sample_weight=w)
                np.testing.assert_allclose(base.predict_linear_loss(actual,x),base.predict_linear_loss(expected,x),atol=2e-5)
    def test_daily_mass_caps_and_no_future_dependence(self):
        dates=np.repeat([1,2],[30,60]);vol=np.geomspace(.00001,10,90);vol[0]=np.nan
        a,_=sample_weights(dates,vol,WeightConfig(date_gamma=.75,volatility_power=0))
        b,s=sample_weights(dates,vol,WeightConfig(date_gamma=.75,volatility_power=1))
        self.assertTrue(np.isfinite(b).all());self.assertTrue(((s>=.5)&(s<=2)).all())
        for day in [1,2]:self.assertAlmostEqual(a[dates==day].sum(),b[dates==day].sum())
        _,prefix=sample_weights(dates[:30],vol[:30],WeightConfig(volatility_power=1))
        np.testing.assert_allclose(prefix,s[:30])

if __name__=='__main__':
    with threadpool_limits(limits=4):unittest.main()
