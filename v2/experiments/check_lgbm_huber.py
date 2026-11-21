"""Synthetic check of LightGBM's huber objective (gradient and hessian), before stage 6a scheme B.

One boosting round, learning_rate 1, a single binary feature (two leaves), no regularization, no
boost_from_average: each leaf value equals -sum(grad) / sum(hess) over its rows. Compare with the formulas
  grad = diff if |diff| <= alpha else alpha * sign(diff),   diff = score - label
  hess = 1 (every row)                       [hypothesis H1]
  hess = 1 inside |diff| <= alpha, 0 outside [hypothesis H0, standard pseudo-Huber-style second derivative]
Also checks a non-zero init_score (diff uses the score) and reports the boost_from_average init value.
"""

# Locate shared modules in the published directory layout.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import numpy as np
import lightgbm as lgb

rng = np.random.default_rng(0)
n = 20_000
x = rng.integers(0, 2, n).astype(float)[:, None]
y = rng.standard_t(2, n) * 0.05 + 0.01 * x[:, 0]
alpha = 0.03


def leaf_values(init):
    params = dict(objective='huber', alpha=alpha, learning_rate=1.0, num_leaves=2, min_data_in_leaf=1,
                  min_sum_hessian_in_leaf=0, lambda_l1=0, lambda_l2=0, boost_from_average=False,
                  max_bin=2, verbosity=-1, seed=1, deterministic=True, force_col_wise=True)
    ds = lgb.Dataset(x, label=y, init_score=init, params=params)
    b = lgb.train(params, ds, num_boost_round=1)
    out = {}
    for g in (0, 1):
        out[g] = float(b.predict(np.array([[float(g)]]), raw_score=True)[0])
    return out


report = []
for name, init in (('init 0', np.zeros(n)), ('init 0.02', np.full(n, 0.02))):
    got = leaf_values(init)
    for g in (0, 1):
        m = x[:, 0] == g
        diff = init[m] - y[m]
        grad = np.where(np.abs(diff) <= alpha, diff, alpha * np.sign(diff))
        h1 = -grad.sum() / m.sum()
        inside = np.abs(diff) <= alpha
        h0 = -grad.sum() / inside.sum()
        report.append((name, g, got[g], h1, h0))
        print(f'{name} leaf {g}: lightgbm {got[g]:.10f} | H1 hess=1 {h1:.10f} (diff {abs(got[g]-h1):.2e}) | '
              f'H0 hess=1{{inside}} {h0:.10f} (diff {abs(got[g]-h0):.2e})')
ok_h1 = all(abs(r[2] - r[3]) < 1e-8 for r in report)
ok_h0 = all(abs(r[2] - r[4]) < 1e-8 for r in report)
params = dict(objective='huber', alpha=alpha, verbosity=-1, num_leaves=2, min_data_in_leaf=1, learning_rate=1e-12)
b = lgb.train(params, lgb.Dataset(x, label=y, params=params), num_boost_round=1)
init_value = float(b.predict(np.array([[0.0]]), raw_score=True)[0])
print(f'boost_from_average init for huber = {init_value:.8f}; mean(y) = {y.mean():.8f}; median(y) = {np.median(y):.8f}')
print(f'RESULT: matches H1 (grad clipped at alpha, hess = 1 for all rows): {ok_h1}; matches H0: {ok_h0}')
print(f'lightgbm {lgb.__version__}')
