"""Numerical checks for new weighting, intercept and evaluation definitions."""
import unittest
import numpy as np
from train_linear_baselines import fit_linear, stock_weights, date_weights, weighted_rank_ic
from training_common import daily_rank_ic


class NumericalTests(unittest.TestCase):
    def test_weighted_solution_matches_augmented_lstsq(self):
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(200, 5)) + 3.0
        y = 1.7 + x @ np.arange(5) + rng.normal(size=200)
        w = rng.uniform(0.1, 3, 200)
        coef, intercept, _ = fit_linear(x, y, w, chunk=37)
        a = np.column_stack([np.ones(len(y)), x])
        expected = np.linalg.lstsq(a * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
        np.testing.assert_allclose(np.r_[intercept, coef], expected, atol=1e-10)

    def test_stock_weights_are_daily_normalized_and_causal(self):
        dates = np.repeat([1, 2], 4)
        vol = np.array([.01, .02, .03, np.nan, .01, .02, .04, .1])
        w = stock_weights(vol, dates)
        np.testing.assert_allclose([w[:4].mean(), w[4:].mean()], 1)
        changed = vol.copy()
        changed[4:] *= 100
        np.testing.assert_allclose(stock_weights(changed, dates)[:4], w[:4])
        self.assertGreater(w[2], w[0])
        dw = date_weights(dates)
        np.testing.assert_allclose((w*dw)[:4].sum(), dw[:4].sum())

    def test_equal_weight_rank_matches_spearman_with_ties(self):
        rng = np.random.default_rng(12)
        p, y = rng.integers(0, 8, (2, 100))
        dates = np.repeat([1, 2], 50)
        self.assertAlmostEqual(weighted_rank_ic(p, y, dates, np.ones(100)), daily_rank_ic(p, y, dates))


if __name__ == '__main__':
    unittest.main()
