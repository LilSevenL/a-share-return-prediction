
# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import unittest
import numpy as np
import pandas as pd
from audit_factors import compare,factor_bounds


class AuditTests(unittest.TestCase):
    def test_distinguish_value_and_missing_disagreement(self):
        keys=pd.DataFrame({'ts_code':['x']*4,'signal_date':pd.date_range('2022-01-01',periods=4)})
        result=compare([1,np.nan,3,4],[1,2,np.nan,5],keys,'test','independent')
        self.assertEqual(result['null_mismatches'],2)
        self.assertEqual(result['value_mismatches'],1)
        self.assertEqual(result['mismatches'],3)

    def test_float_rounding_is_not_false_formula_error(self):
        keys=pd.DataFrame({'ts_code':['x'],'signal_date':[pd.Timestamp('2022-01-01')]})
        self.assertEqual(compare([.123456789],[np.float32(.123456789)],keys,'x','x')['mismatches'],0)

    def test_only_mathematical_ranges_are_enforced(self):
        self.assertEqual(factor_bounds('return_volume_corr_20d','P'),(-1.,1.))
        self.assertEqual(factor_bounds('interaction_01','I'),(-.25,.25))
        self.assertEqual(factor_bounds('market_beta_60d','X'),(None,None))
        self.assertEqual(factor_bounds('return_ts_rank_20d','M'),(0.,1.))


if __name__=='__main__':unittest.main()
