"""Synthetic checks for the V2 LightGBM helpers; no market data, no LightGBM training."""

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lgbm_v2_common as common


def fake_lightgbm():
    class EarlyStopException(Exception):
        def __init__(self, best_iteration, best_score):
            super().__init__(best_iteration)
            self.best_iteration, self.best_score = best_iteration, best_score
    module = types.ModuleType('lightgbm')
    module.callback = types.SimpleNamespace(EarlyStopException=EarlyStopException)
    return module, EarlyStopException


class HelperTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.dates = np.repeat(np.arange('2020-01-01', '2020-01-31', dtype='datetime64[D]'), 60)
        self.y = rng.normal(size=len(self.dates))
        self.pred = self.y + rng.normal(scale=2, size=len(self.dates))

    def test_daily_rank_ic_matches_scipy_with_ties(self):
        pred = np.round(self.pred, 1)
        metric = common.DailyRankIC(self.y, self.dates)
        expected = [spearmanr(pred[s:e], self.y[s:e]).statistic for s, e in zip(*common.day_slices(self.dates))]
        np.testing.assert_allclose(metric.daily(pred), expected, atol=1e-12)
        self.assertAlmostEqual(metric(pred), float(np.mean(expected)), places=12)

    def test_unsorted_dates_rejected(self):
        with self.assertRaises(ValueError):
            common.DailyRankIC(self.y, self.dates[::-1])

    def test_newey_west_reduces_to_plain_t_without_lag_and_grows_with_autocorrelation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.1, 1, 500)
        plain = x.mean() / np.sqrt(np.var(x) / len(x))
        self.assertAlmostEqual(common.newey_west_tstat(x, lag=0), plain, places=10)
        smooth = np.convolve(rng.normal(0.05, 1, 504), np.ones(5) / 5, mode='valid')
        self.assertLess(abs(common.newey_west_tstat(smooth, 5)), abs(common.newey_west_tstat(smooth, 0)))

    def test_early_stopping_waits_full_patience_and_reports_best(self):
        module, stop = fake_lightgbm()
        sys.modules['lightgbm'] = module
        try:
            monitor = common.RankICEarlyStopping(self.y, self.dates, every=10, patience=30)
            before, after = monitor.callbacks()
            self.assertTrue(before.before_iteration)
            self.assertFalse(after.before_iteration)
            scores = {10: .1, 20: .2, 30: .15, 40: .19, 50: .2, 60: .18}
            stopped = None
            for it in range(1, 101):
                env = types.SimpleNamespace(iteration=it - 1, end_iteration=100)
                before(env)
                monitor.last = scores.get(it, monitor.last)
                if it % 10 == 0:
                    monitor.curve.append((it, monitor.last, 0))
                try:
                    after(env)
                except stop as exc:
                    stopped = (it, exc.best_iteration)
                    break
            self.assertEqual(monitor.best_iteration, 20)
            self.assertEqual(stopped, (50, 19))  # equal score at 50 is not an improvement
            capped = common.RankICEarlyStopping(self.y, self.dates, patience=200)
            capped.best_iteration = 2900
            self.assertTrue(capped.hit_round_limit(3000))
            capped.best_iteration = 2500
            self.assertFalse(capped.hit_round_limit(3000))
        finally:
            sys.modules.pop('lightgbm', None)

    def test_feval_only_recomputes_on_schedule(self):
        monitor = common.RankICEarlyStopping(self.y, self.dates, every=10, patience=50)
        monitor.total_rounds = 100
        values = []
        for it in range(1, 21):
            monitor.iteration = it
            values.append(monitor.feval(self.pred if it < 15 else self.y, None)[1])
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[9], values[13])
        self.assertAlmostEqual(values[19], 1.0)
        self.assertEqual([c[0] for c in monitor.curve], [10, 20])

    def test_head_turnover_and_timing_split(self):
        frame = pd.DataFrame({'ts_code': np.tile([f'{i:06d}.SZ' for i in range(60)], 30),
                              'signal_date': self.dates, 'y': self.y, 'eligible': True})
        frame['perfect'] = frame['y']
        frame['market'] = frame.groupby('signal_date')['y'].transform('mean')
        head = common.head_and_turnover(frame, 'perfect')
        self.assertGreater(head['top50_minus_pool_5d_simple_return'], 0)
        split = common.timing_selection_split(frame, 'market')
        self.assertAlmostEqual(split['timing_variance_share'], 1.0)
        self.assertAlmostEqual(split['timing_correlation'], 1.0)

    def test_evaluate_outer_uses_common_v2_definition(self):
        frame = pd.DataFrame({'ts_code': np.tile([f'{i:06d}.SZ' for i in range(60)], 30),
                              'signal_date': self.dates, 'y': self.y, 'eligible': True,
                              'vol20': np.abs(self.y) + .01, 'pred': self.pred})
        frame.loc[::7, 'y'] = np.nan
        metrics, daily = common.evaluate_outer(frame, 'pred')
        scored = frame.dropna(subset=['y'])
        expected = scored.groupby('signal_date').apply(lambda g: spearmanr(g.pred, g.y).statistic).mean()
        self.assertAlmostEqual(metrics['mean_rank_ic'], expected, places=10)
        self.assertEqual(len(daily), 30)

    def test_tree_depth_diagnostic(self):
        leaf = lambda i: {'leaf_index': i, 'leaf_value': 0.0}
        tree = {'split_index': 0, 'left_child': leaf(0),
                'right_child': {'split_index': 1, 'left_child': leaf(1), 'right_child': leaf(2)}}

        class Booster:
            def dump_model(self, num_iteration):
                return {'tree_info': [{'num_leaves': 3, 'tree_structure': tree}, {'num_leaves': 1, 'tree_structure': leaf(0)}]}

            def feature_importance(self, importance_type, iteration):
                return np.array([3.0, 1.0])

            def feature_name(self):
                return ['a', 'b']
        result = common.tree_diagnostics(Booster(), 2)
        self.assertEqual(result['max_depth'], 2)
        self.assertEqual(result['mean_leaves'], 2.0)
        self.assertEqual(result['top20_gain'][0][0], 'a')

    def test_feature_names_and_future_years_blocked(self):
        with self.assertRaises(ValueError):
            common.parquet_files(common.FEATURES, 'feature_year', 2023, 2024)


if __name__ == '__main__':
    unittest.main(verbosity=2)
