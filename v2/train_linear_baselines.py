"""V2 stage 1: full-data OLS/WLS controls on the original 50 features.

This is a baseline experiment, not the expanded-feature or robust-loss stage.
Only the existing final_split=training partition (through 2023) is opened.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from training_common import (
    FACTOR_COLUMNS, add_common_args, atomic_json, evaluate, fit_preprocessor,
    load_folds, load_matrix, load_matrix_with_dates, open_connection,
    prepare_output, save_predictions, select_sql, transform, validate_common_args,
)
from tune_daily_weights import weighted_sufficient_statistics

MODELS = ('ols_equal', 'wls_date', 'wls_date_volatility')


def stock_weights(volatility, dates):
    """Past 20-session volatility / same-day median, clip [0.5,2], daily mean=1.

    No target, future return, industry classification or test data is used.
    Missing volatility receives the neutral raw multiplier 1.
    """
    output = np.ones(len(dates), dtype=np.float64)
    _, starts = np.unique(dates, return_index=True)
    for start, end in zip(starts, np.r_[starts[1:], len(dates)]):
        v = volatility[start:end]
        good = np.isfinite(v) & (v > 0)
        if good.any():
            block = output[start:end]
            block[good] = np.clip(v[good] / np.median(v[good]), 0.5, 2.0)
            block /= block.mean()
    return output


def date_weights(dates):
    _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype(np.float64) ** -0.75
    return weights / weights.mean()


def fit_linear(x, y, weights, chunk=100_000):
    stats = weighted_sufficient_statistics(x, y, weights, chunk)
    coefficient = np.linalg.pinv(stats['gram'], rcond=1e-10) @ stats['rhs']
    intercept = stats['mean_y'] - stats['mean_x'] @ coefficient
    return coefficient, float(intercept), {
        'normal_equation_residual_max': float(np.max(np.abs(stats['gram'] @ coefficient - stats['rhs']))),
        'gram_rank': int(np.linalg.matrix_rank(stats['gram'])),
        'gram_condition_number': float(np.linalg.cond(stats['gram'])),
    }


def weighted_rank_ic(pred, y, dates, weights):
    """Daily weighted Pearson of ordinary average ranks (ties averaged).

    One fixed evaluation weighting rule is applied to EVERY candidate.
    This is explicitly not a weighted-CDF rank definition.
    """
    scores = []
    _, starts = np.unique(dates, return_index=True)
    for start, end in zip(starts, np.r_[starts[1:], len(dates)]):
        if end - start < 20:
            continue
        p, a = rankdata(pred[start:end]), rankdata(y[start:end])
        w = weights[start:end]
        p -= np.average(p, weights=w)
        a -= np.average(a, weights=w)
        denominator = np.sqrt(np.dot(w, p*p) * np.dot(w, a*a))
        if denominator > 0:
            scores.append(float(np.dot(w, p*a) / denominator))
    return float(np.mean(scores)) if scores else None


def aggregate(records):
    output = []
    for model in MODELS:
        rows = [r['metrics'] for r in records if r['model'] == model]
        if rows:
            ric = np.array([r['daily_rank_ic_mean'] for r in rows])
            output.append(dict(model=model, folds=len(rows),
                mean_rank_ic=float(ric.mean()), yearly_rank_ic_std=float(ric.std()),
                mean_weighted_rank_ic=float(np.mean([r['daily_weighted_rank_ic'] for r in rows])),
                mean_rmse=float(np.mean([r['rmse'] for r in rows])),
                mean_mae=float(np.mean([r['mae'] for r in rows]))))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-50',action='store_true',help='Explicitly reproduce the historical 50-factor control; current 500-factor work uses run_research.py')
    add_common_args(parser, ROOT / 'quant_data/processed/v2/linear_baselines_50')
    args = parser.parse_args()
    if not args.legacy_50:
        parser.error('This is the legacy 50-factor experiment. For balanced 500 factors run v2/run_research.py; use --legacy-50 only to reproduce old controls.')
    validate_common_args(args)
    if args.force:
        raise ValueError('Use a new V2 experiment directory instead of --force.')
    if not args.folds:
        args.folds = [f'validation_{year}' for year in range(2017, 2024)]
    if not set(args.folds) <= {f'validation_{y}' for y in range(2017, 2024)}:
        raise ValueError('This stage permits only development folds 2017-2023.')
    dataset = Path(args.dataset_dir).resolve()
    destination = Path(args.output_dir).resolve()
    if not destination.is_relative_to(ROOT / 'quant_data/processed/v2'):
        raise ValueError('Outputs must be inside quant_data/processed/v2.')
    output = prepare_output(destination, False, args.min_free_gb)
    folds = load_folds(dataset, args.folds)
    # Physical partition restriction: permanent Test files are never opened.
    glob = (dataset / 'final_split=training/signal_year=*/*.parquet').as_posix()
    files = sorted((dataset / 'final_split=training').glob('signal_year=*/*.parquet'))
    if not files:
        raise FileNotFoundError(glob)
    config = dict(arguments=vars(args), stage='50-feature MSE baseline controls',
        test_used=False, training_sample='full unless debug row limits specified',
        models=list(MODELS), factors=FACTOR_COLUMNS, gamma=0.75,
        stock_weight='clip(volatility_20d / daily_median, 0.5, 2.0); daily mean normalized to 1',
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT/'training_common.py', ROOT/'tune_daily_weights.py',
             ROOT/'build_features.py', dataset/'rolling_folds.json']},
        input_inventory=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files])
    manifest = output / 'experiment_config.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != config:
        raise ValueError('Configuration/source/input inventory changed; choose a new directory.')
    atomic_json(config, manifest)
    progress_path = output / 'progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {'results': []}
    con = open_connection(output, args)
    started = time.time()
    try:
        with threadpool_limits(limits=args.threads):
            for fold in folds:
                name = fold['name']
                done = {r['model'] for r in progress['results'] if r['fold'] == name}
                if done == set(MODELS):
                    continue
                print(f'{name}: loading full development training data', flush=True)
                x, y, dates = load_matrix_with_dates(con, glob, select_sql(fold, 'train'), args.max_train_rows)
                xv, yv, codes, dv = load_matrix(con, glob, select_sql(fold, 'validation'), args.max_validation_rows, True)
                if not np.isfinite(y).all() or not np.isfinite(yv).all():
                    raise ValueError('Nonfinite targets')
                vol_index = FACTOR_COLUMNS.index('volatility_20d')
                sw = stock_weights(x[:, vol_index], dates)
                sv = stock_weights(xv[:, vol_index], dv)
                dw = date_weights(dates)
                stats = fit_preprocessor(x, args)
                x, xv = transform(x, stats), transform(xv, stats)
                fold_dir = output / name
                fold_dir.mkdir(exist_ok=True)
                atomic_json(stats, fold_dir / 'preprocessing.json')
                print(f'{name}: train={len(y):,}, validation={len(yv):,}, features={x.shape[1]}', flush=True)
                for model in MODELS:
                    if model in done:
                        continue
                    weights = np.ones(len(y)) if model == 'ols_equal' else dw.copy()
                    if model == 'wls_date_volatility':
                        weights *= sw
                    print(f'{name}: fitting {model}', flush=True)
                    coef, intercept, diagnostics = fit_linear(x, y, weights)
                    prediction = (xv @ coef + intercept).astype(np.float32)
                    if not np.isfinite(prediction).all():
                        raise ValueError('Nonfinite predictions')
                    metrics = evaluate(prediction, yv, dv)
                    metrics['daily_weighted_rank_ic'] = weighted_rank_ic(prediction, yv, dv, sv)
                    np.savez(fold_dir / f'{model}.npz', coefficient=coef, intercept=intercept, active_factors=stats['active_factors'])
                    save_predictions(fold_dir / f'{model}_predictions.parquet', codes, dv, yv, prediction)
                    record = dict(fold=name, model=model, training_rows=len(y), metrics=metrics, diagnostics=diagnostics)
                    atomic_json(record, fold_dir / f'{model}_metrics.json')
                    progress['results'].append(record)
                    atomic_json(progress, progress_path)
                    print(f'{name}: {model} Rank IC={metrics["daily_rank_ic_mean"]:.6f}; weighted={metrics["daily_weighted_rank_ic"]:.6f}', flush=True)
                    del weights, prediction
                del x, xv, y, yv, dates, dv, codes, sw, sv, dw
                gc.collect()
    finally:
        con.close()
    summary = dict(status='completed', test_used=False,
        debug_run=bool(args.max_train_rows or args.max_validation_rows),
        elapsed_minutes=(time.time()-started)/60, aggregate=aggregate(progress['results']),
        interpretation='Baseline diagnostics only. No transaction-cost simulation or profitability conclusion. Expanded factors and robust losses are subsequent stages.')
    atomic_json(summary, output / 'summary.json')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
