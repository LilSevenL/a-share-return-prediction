
# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import unittest
import numpy as np
from streaming_ols import StreamingOLS


class StreamingOLSTests(unittest.TestCase):
    def test_matches_direct_lstsq_across_batches(self):
        rng=np.random.default_rng(72);x=rng.normal(size=(301,9));y=1.2+x@rng.normal(size=9)+rng.normal(size=301)*.1
        expected=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=np.finfo(float).eps*len(x))[0]
        for size in [3,37,301]:
            fit=StreamingOLS(9)
            for i in range(0,len(y),size):fit.update(x[i:i+size],y[i:i+size])
            model=fit.finish();actual=np.r_[model['intercept'],model['coefficient']]
            np.testing.assert_allclose(actual,expected,atol=1e-12)
            self.assertAlmostEqual(model['diagnostics']['training_rmse'],np.sqrt(np.mean((np.column_stack([np.ones(len(x)),x])@expected-y)**2)))

    def test_rank_deficiency_and_checkpoint(self):
        import tempfile
        from pathlib import Path
        rng=np.random.default_rng(2);a=rng.normal(size=(100,2));x=np.column_stack([a,a[:,0]]);y=a[:,0]*2-a[:,1]+.5
        fit=StreamingOLS(3);fit.update(x[:40],y[:40])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.npz';fit.save(p);fit=StreamingOLS.load(p)
        fit.update(x[40:],y[40:]);m=fit.finish()
        self.assertEqual(m['diagnostics']['rank'],3)
        np.testing.assert_allclose(m['intercept']+x@np.array(m['coefficient']),y,atol=1e-12)


if __name__=='__main__':unittest.main()
