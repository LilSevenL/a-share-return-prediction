
# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import unittest
import numpy as np
from features_500 import FEATURE_COLUMNS, NEW_FEATURES, SELECTORS, RETURNS, conditional_features


class FeatureTests(unittest.TestCase):
    def inputs(self, n=150):
        selectors = {k: np.arange(n, dtype=float) for k in SELECTORS}
        returns = {k: (np.arange(n, dtype=float)-75)/1000 for k in RETURNS}
        return np.arange(n), selectors, returns

    def test_count_and_twenty_day_hand_calculation(self):
        self.assertEqual(len(FEATURE_COLUMNS), 500)
        self.assertEqual(len(set(FEATURE_COLUMNS)), 500)
        idx, sel, ret = self.inputs()
        values = conditional_features(idx,sel,ret)
        chosen = ret['close'][10:20]
        for stat, expected in [('sum', chosen.sum()), ('std', chosen.std(ddof=1)),
                               ('downside_rms', np.sqrt(np.mean(np.minimum(chosen,0)**2)))]:
            col = NEW_FEATURES.index(f'cond_volume_high_close_{stat}_20d')
            self.assertAlmostEqual(values[19,col], expected)
        col = NEW_FEATURES.index('cond_volume_low_close_sum_20d')
        self.assertAlmostEqual(values[19,col],ret['close'][:10].sum())

    def test_future_invariance_and_prefix_equivalence(self):
        idx, sel, ret = self.inputs()
        expected = conditional_features(idx,sel,ret)
        prefix = conditional_features(idx[:125],{k:v[:125] for k,v in sel.items()},{k:v[:125] for k,v in ret.items()})
        np.testing.assert_allclose(prefix,expected[:125],equal_nan=True)
        for v in sel.values(): v[125:] *= -100
        for v in ret.values(): v[125:] += 99
        np.testing.assert_allclose(conditional_features(idx,sel,ret)[:125],expected[:125],equal_nan=True)

    def test_missing_conditions_returns_and_calendar_gap(self):
        idx, sel, ret = self.inputs()
        col = NEW_FEATURES.index('cond_volume_high_close_sum_20d')
        sel['volume'][10] = np.nan
        self.assertTrue(np.isnan(conditional_features(idx,sel,ret)[19,col]))
        idx, sel, ret = self.inputs()
        ret['close'][19] = np.nan
        self.assertTrue(np.isnan(conditional_features(idx,sel,ret)[19,col]))
        idx, sel, ret = self.inputs()
        idx[10:] += 1
        self.assertTrue(np.isnan(conditional_features(idx,sel,ret)[19,col]))
        self.assertTrue(np.isfinite(conditional_features(idx,sel,ret)[29,col]))

    def test_ties_and_constant_conditions(self):
        idx, sel, ret = self.inputs()
        sel['volume'][:] = 1
        col = NEW_FEATURES.index('cond_volume_high_close_sum_20d')
        self.assertTrue(np.isnan(conditional_features(idx,sel,ret)[19,col]))
        sel['volume'][:20] = np.repeat([1.,2.],10)
        result = conditional_features(idx,sel,ret)
        self.assertAlmostEqual(result[19,col],ret['close'][10:20].sum())

    def test_short_history_and_duplicate_sessions(self):
        idx, sel, ret = self.inputs(5)
        self.assertTrue(np.isnan(conditional_features(idx,sel,ret)).all())
        idx, sel, ret = self.inputs()
        idx[1] = idx[0]
        with self.assertRaises(ValueError): conditional_features(idx,sel,ret)


if __name__ == '__main__':
    unittest.main()
