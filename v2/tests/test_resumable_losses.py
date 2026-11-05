
# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import tempfile,unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import linear_losses_optimized as base
from resumable_losses import fit

class ResumeTests(unittest.TestCase):
    def test_interruption_and_resume(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(200,8));x[:,1]=x[:,0]+.01*x[:,1]
        y=rng.normal(size=200);y[:20]+=10
        for loss in ('mae','huber'):
            cfg=base.LossConfig(loss=loss,l2=0,tol=1e-7,max_iter=15000)
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'state.npz'
                expected=base.fit_linear_loss(x,y,config=cfg)
                def interrupt(step):raise InterruptedError('simulated interruption')
                with self.assertRaises(InterruptedError):fit(x,y,cfg,p,'test',interrupt,checkpoint_every=5)
                with self.assertRaises(ValueError):fit(x,y,cfg,p,'changed')
                actual=fit(x,y,cfg,p,'test')
                self.assertEqual(actual['diagnostics']['resumed_from_iteration'],5)
                np.testing.assert_allclose(base.predict_linear_loss(actual,x),base.predict_linear_loss(expected,x),atol=2e-5)

if __name__=='__main__':
    with threadpool_limits(limits=4):unittest.main()
