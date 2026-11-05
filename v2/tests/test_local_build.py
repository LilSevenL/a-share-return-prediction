
# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from build_local_features import calculate_stock,RANK_PARENTS,safe_remove_tree
from balanced_features import technical_stock,COLUMNS
import test_research_pipeline as fixtures


class LocalBuildTests(unittest.TestCase):
    def test_worker_matches_reference_and_preserves_rank_precision(self):
        d=fixtures.PipelineTests().fixture();d['ts_code']='sample'
        expected=technical_stock(d)
        result=calculate_stock(('sample',d.reset_index(),d.index,d[['rm','market_median_turnover']],d.index.min(),d.index.max()+pd.Timedelta(days=1)))
        np.testing.assert_allclose(result[COLUMNS],expected,rtol=2e-6,atol=1e-7,equal_nan=True)
        for name in RANK_PARENTS:self.assertEqual(result[name].dtype,np.dtype('float64'))

    def test_worker_filters_output_dates_after_warmup(self):
        d=fixtures.PipelineTests().fixture();d['ts_code']='sample';start=d.index[250]
        result=calculate_stock(('sample',d.reset_index(),d.index,d[['rm','market_median_turnover']],start,d.index.max()+pd.Timedelta(days=1)))
        self.assertEqual(len(result),70)
        self.assertTrue(result.logret_250d.notna().all())

    def test_cleanup_cannot_remove_output_or_outside_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'output';root.mkdir()
            with self.assertRaises(ValueError):safe_remove_tree(root,root)
            with self.assertRaises(ValueError):safe_remove_tree(Path(temp)/'stage_year_2020',root)
            with self.assertRaises(ValueError):safe_remove_tree(root/'feature_year=2020',root)
            scratch=root/'stage_year_2020';scratch.mkdir();(scratch/'temp.txt').write_text('scratch')
            safe_remove_tree(scratch,root);self.assertFalse(scratch.exists());self.assertTrue(root.exists())


if __name__=='__main__':unittest.main()
