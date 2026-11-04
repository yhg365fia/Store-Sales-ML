"""Analyze saved backtest predictions, without training or changing inputs.

Default: EMA+Oil versus Trial17. Negative delta = improvement.
Log residual = log1p(prediction) - log1p(actual): positive means overprediction.
This diagnoses forecast errors, not whether actual sales were a surge/drop.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(r'C:\projects\Store Sales-ML\src\backtest_outputs\20261103_155940_882756\predictions.csv')
KEYS = ['date', 'store_nbr', 'family']
MODELS = {'ema7_oil': 'ema7_oil_mix_pred_sales', 'ema7': 'ema7_mix_pred_sales'}


def validate(p, candidate):
    required = KEYS + ['sales', 'baseline_pred_sales', 'analysis_group', candidate]
    missing = sorted(set(required) - set(p.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if p[KEYS + ['analysis_group']].isna().any().any() or p.duplicated(KEYS).any():
        raise ValueError('Missing or duplicate row keys')
    for col in ['sales', 'baseline_pred_sales', candidate]:
        p[col] = pd.to_numeric(p[col], errors='raise')
        if not np.isfinite(p[col]).all() or p[col].lt(0).any():
            raise ValueError(f'Invalid nonnegative values: {col}')
    if 'y_true' in p and not np.allclose(p.sales, p.y_true):
        raise ValueError('sales/y_true mismatch')
    expected_dates = pd.date_range('2016-08-16', '2016-08-31')
    counts = p.groupby(['store_nbr', 'family']).size()
    if (len(p) != 28512 or len(counts) != 1782 or not counts.eq(16).all()
            or set(p.date) != set(expected_dates)):
        raise ValueError('Expected complete August 2016 backtest panel')
    if set(p.analysis_group) != {1, 2, 3, 4} or p.analysis_group.eq(1).sum() != 6048:
        raise ValueError('Unexpected analysis groups')
    if p.groupby('family').analysis_group.nunique().max() != 1:
        raise ValueError('Inconsistent family/group mapping')
    other = p.analysis_group.ne(1)
    if not np.array_equal(p.loc[other, candidate], p.loc[other, 'baseline_pred_sales']):
        raise ValueError('Predictions outside Group1 differ from baseline')
    baseline = np.sqrt(np.mean((np.log1p(p.baseline_pred_sales) - np.log1p(p.sales)) ** 2))
    if not np.isclose(baseline, 0.7477345821606495, atol=1e-9, rtol=0):
        raise ValueError(f'Unexpected baseline score: {baseline}')


def enrich(p, candidate):
    p = p.copy()
    p['horizon_day'] = (p.date - pd.Timestamp('2016-08-16')).dt.days + 1
    p['candidate_pred_sales'] = p[candidate]
    for label, col in [('baseline', 'baseline_pred_sales'), ('candidate', candidate)]:
        p[f'{label}_log_error'] = np.log1p(p[col]) - np.log1p(p.sales)
        p[f'{label}_sq_log_error'] = p[f'{label}_log_error'] ** 2
    p['delta_sq_log_error'] = p.candidate_sq_log_error - p.baseline_sq_log_error
    p['delta_abs_log_error'] = p.candidate_log_error.abs() - p.baseline_log_error.abs()
    p['baseline_direction'] = np.select(
        [p.baseline_log_error.lt(0), p.baseline_log_error.gt(0)],
        ['under', 'over'], default='exact')
    p['candidate_direction'] = np.select(
        [p.candidate_log_error.lt(0), p.candidate_log_error.gt(0)],
        ['under', 'over'], default='exact')
    p['sales_state'] = np.where(p.sales.eq(0), 'zero', 'positive')
    return p


def metrics(b, total_n):
    result = {'rows': len(b), 'actual_sales_sum': float(b.sales.sum())}
    for label in ['baseline', 'candidate']:
        e = b[f'{label}_log_error']
        sq = e ** 2
        pred = b[f'{label}_pred_sales']
        result.update({
            f'{label}_rmsle': float(np.sqrt(sq.mean())),
            f'{label}_mean_log_error': float(e.mean()),
            f'{label}_mean_abs_log_error': float(e.abs().mean()),
            f'{label}_sse_log': float(sq.sum()),
            f'{label}_under_sse_log': float(sq[e.lt(0)].sum()),
            f'{label}_over_sse_log': float(sq[e.gt(0)].sum()),
            f'{label}_under_rows': int(e.lt(0).sum()),
            f'{label}_over_rows': int(e.gt(0).sum()),
            f'{label}_pred_sales_sum': float(pred.sum()),
            f'{label}_sales_bias': float((pred - b.sales).sum()),
        })
    for measure in ['rmsle', 'mean_abs_log_error', 'sse_log', 'under_sse_log', 'over_sse_log']:
        result[f'delta_{measure}'] = result[f'candidate_{measure}'] - result[f'baseline_{measure}']
    result['overall_msle_contribution'] = result['delta_sse_log'] / total_n
    result['sse_reduction_pct'] = (
        -100 * result['delta_sse_log'] / result['baseline_sse_log']
        if result['baseline_sse_log'] > 0 else np.nan)
    result['improved_rows'] = int(b.delta_sq_log_error.lt(-1e-12).sum())
    result['worsened_rows'] = int(b.delta_sq_log_error.gt(1e-12).sum())
    result['unchanged_rows'] = len(b) - result['improved_rows'] - result['worsened_rows']
    return result


def summarize(p, by, total_n):
    rows = []
    grouping = by[0] if len(by) == 1 else by
    for key, b in p.groupby(grouping, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **metrics(b, total_n)})
    return pd.DataFrame(rows)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--candidate', choices=list(MODELS), default='ema7_oil')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'backtest_error_analysis')
    args = parser.parse_args()
    p = pd.read_csv(args.input, parse_dates=['date'])
    validate(p, MODELS[args.candidate])
    p = enrich(p, MODELS[args.candidate])
    g = p.loc[p.analysis_group.eq(1)].copy()
    n = len(p)
    out = args.output_dir.resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.candidate)
    out.mkdir(parents=True, exist_ok=False)
    overall = pd.DataFrame([metrics(p, n)])
    groups = summarize(p, ['analysis_group'], n)
    families = summarize(p, ['analysis_group', 'family'], n).sort_values('delta_sse_log')
    group1_families = families.loc[families.analysis_group.eq(1)]
    reports = {
        'overall': overall,
        'groups': groups,
        'all_families': families,
        'group1_families': group1_families,
        'group1_daily': summarize(g, ['date', 'horizon_day'], n),
        'group1_family_daily': summarize(g, ['family', 'date', 'horizon_day'], n),
        'group1_stores': summarize(g, ['store_nbr'], n),
        'group1_family_stores': summarize(g, ['family', 'store_nbr'], n),
        # Same row subsets for both models, defined by baseline error direction.
        'group1_baseline_direction': summarize(g, ['baseline_direction'], n),
        'group1_direction_transitions': summarize(g, ['baseline_direction', 'candidate_direction'], n),
        'group1_zero_positive': summarize(g, ['sales_state'], n),
        'group1_largest_improvements': g.nsmallest(100, 'delta_sq_log_error'),
        'group1_largest_deteriorations': g.nlargest(100, 'delta_sq_log_error'),
        'group1_row_errors': g,
    }
    if not np.isclose(families.delta_sse_log.sum(), overall.delta_sse_log.iloc[0], atol=1e-8):
        raise ValueError('Error contributions do not reconcile')
    for name, report in reports.items():
        report.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    info = {
        'input': str(args.input.resolve()), 'candidate': args.candidate,
        'baseline': 'Trial17 cached August 2016 backtest',
        'training_performed': False,
        'definitions': {
            'log_error': 'log1p(prediction)-log1p(actual); positive=overprediction',
            'delta': 'candidate minus baseline; negative error delta=improvement',
            'sse_log': 'sum of squared log errors (additive across partitions)',
            'overall_msle_contribution': 'delta_sse_log / all 28512 rows; NOT an RMSLE contribution',
            'under_over_sse': 'each model classified by its own residual sign; inspect transitions too',
            'baseline_direction': 'fixed rows classified by baseline sign for paired comparison',
            'sales_bias': 'sum(predicted sales - actual sales); not the competition metric',
        },
        'limitations': [
            'No surge/drop/volatility labels: these require pre-cutoff sales history.',
            'Different candidate iteration counts; comparison is of complete configurations.',
            '2017-selected settings evaluated retrospectively on 2016; descriptive diagnostic.',
        ],
    }
    (out / 'definitions.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['family', 'baseline_rmsle', 'candidate_rmsle', 'delta_rmsle',
               'delta_sse_log', 'sse_reduction_pct', 'delta_under_sse_log', 'delta_over_sse_log']
    print(f'\nCandidate: {args.candidate}; negative delta = improvement')
    print('\n[GROUPS]\n' + groups[['analysis_group', 'baseline_rmsle', 'candidate_rmsle', 'delta_rmsle']].round(6).to_string(index=False))
    print('\n[GROUP1 FAMILIES: largest total error improvement first]\n'
          + group1_families[columns].round(6).to_string(index=False))
    print('\n[GROUP1: FIXED BASELINE UNDER/OVER SUBSETS]\n'
          + reports['group1_baseline_direction'][['baseline_direction', 'rows', 'baseline_rmsle', 'candidate_rmsle', 'delta_rmsle']].round(6).to_string(index=False))
    print(f'\nSaved to: {out}')


if __name__ == '__main__':
    run()
