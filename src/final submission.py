"""Final Kaggle submission for the EXACT fixed Base23 + Payday2 + Lag364 setup.

Place in src/final_submission.py next to src/modeling.py and src/main.py.
Run from project root:
    python src/final_submission.py
    python src/final_submission.py --run-dir results/base23_payday_lag364/<run_folder>

No Optuna, new features, new folds, or refitting of feature transformations.
The full-train n_estimators is the median best_iteration saved by the
original *fixed* 3-fold experiment (fixed/summary.json).
"""

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# All experiment definitions and original preprocessing are REUSED, not rewritten.
from main import main
from modeling import (
    BASE_FEATURES,
    PAYDAY_FEATURES,
    LAG_DAYS,
    LAG_FEATURE,
    FEATURES,
    CATEGORICAL_FEATURES,
    KEYS,
    TARGET_COL,
    FORECAST_HORIZON,
    FOLDS,
    FIXED_PARAMS,
    OUTPUT_ROOT,
    ROOT,
    prepare_features,
    validate_configuration,
)


def load_fixed_configuration(run_dir=None):
    """Load a completed fixed-CV run, and reject mismatched configurations."""
    if run_dir is not None:
        candidates = [Path(run_dir).expanduser().resolve()]
    else:
        # The run folders created by modeling.py use sortable timestamps.
        candidates = sorted(
            (p.parent.parent for p in OUTPUT_ROOT.glob('*/fixed/summary.json')),
            key=lambda p: p.name,
            reverse=True,
        )

    if not candidates:
        raise FileNotFoundError(
            'No fixed-CV summary found. Check results/base23_payday_lag364/'
            '<run_folder>/fixed/summary.json. This script does not run CV.'
        )

    errors = []
    for candidate in candidates:
        summary_file = candidate / 'fixed' / 'summary.json'
        config_file = candidate / 'run_config.json'
        try:
            if not summary_file.is_file() or not config_file.is_file():
                raise ValueError('fixed/summary.json or run_config.json missing')
            summary = json.loads(summary_file.read_text(encoding='utf-8'))
            config = json.loads(config_file.read_text(encoding='utf-8'))

            if config.get('features') != FEATURES:
                raise ValueError('feature list differs from current modeling.py')
            if config.get('fixed_params') != FIXED_PARAMS:
                raise ValueError('fixed LightGBM parameters differ from current modeling.py')
            if config.get('folds') != FOLDS:
                raise ValueError('CV fold settings differ from current modeling.py')
            if config.get('target') != 'log1p(sales)':
                raise ValueError('target transformation differs')
            if summary.get('features') != len(FEATURES):
                raise ValueError('saved feature count differs')

            n_estimators = int(summary['median_best_iteration'])
            if not 1 <= n_estimators <= FIXED_PARAMS['n_estimators']:
                raise ValueError('invalid median_best_iteration')
            return candidate, summary, n_estimators
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
            errors.append(f'{candidate}: {exc}')
            if run_dir is not None:
                break

    raise RuntimeError(
        'No matching *fixed* experiment found; refusing to guess settings.\n'
        + '\n'.join(errors)
    )


def prepare_test_features(test_raw, train):
    """Construct only sales_lag_364 for test, using known historical sales."""
    required = list(dict.fromkeys(['id', 'date'] + BASE_FEATURES + PAYDAY_FEATURES))
    missing = sorted(set(required) - set(test_raw.columns))
    if missing:
        raise ValueError(f'main() test output missing required columns: {missing}')

    # Keep the original main() features; do not add other features.
    test = test_raw.loc[:, required].copy()
    test['date'] = pd.to_datetime(test['date'], errors='raise')
    if test[['id', 'date'] + KEYS].isna().any().any():
        raise ValueError('Test has null id/date/store_nbr/family')
    if test['id'].duplicated().any() or test.duplicated(KEYS + ['date']).any():
        raise ValueError('Duplicate Test id or store_nbr/family/date')
    if test[BASE_FEATURES + PAYDAY_FEATURES].isna().any().any():
        raise ValueError('Missing base23 or payday features in Test')

    expected_dates = pd.date_range('2017-08-16', periods=FORECAST_HORIZON)
    if (len(test) != 1782 * FORECAST_HORIZON
            or set(test['date']) != set(expected_dates)):
        raise ValueError('Test is not the expected 28,512-row / 16-day forecast panel')
    group_counts = test.groupby(KEYS, observed=True).size()
    if len(group_counts) != 1782 or not group_counts.eq(FORECAST_HORIZON).all():
        raise ValueError('Incomplete Test store_nbr/family panels')

    # Recheck the same deterministic calendar-derived features as modeling.py.
    if not np.array_equal(test['dayofweek'], test['date'].dt.dayofweek):
        raise ValueError('Test dayofweek does not match date')
    if not np.array_equal(test['is_weekend'], (test['date'].dt.dayofweek >= 5).astype(int)):
        raise ValueError('Test is_weekend does not match date')
    if not np.array_equal(test['is_15th'], (test['date'].dt.day == 15).astype(int)):
        raise ValueError('Test is_15th does not match date')
    if not np.array_equal(test['is_month_end'], test['date'].dt.is_month_end.astype(int)):
        raise ValueError('Test is_month_end does not match date')

    # A 364-day lag is from a date already observed before the test begins.
    test['_lag_source_date'] = test['date'] - pd.Timedelta(days=LAG_DAYS)
    if test['_lag_source_date'].max() > train['date'].max():
        raise ValueError('Test lag refers to future sales beyond the train cutoff')

    source = train[KEYS + ['date', 'sales']].rename(
        columns={'date': '_lag_source_date', 'sales': LAG_FEATURE}
    )
    test_ids = test['id'].to_numpy(copy=True)
    test = test.merge(
        source,
        on=KEYS + ['_lag_source_date'],
        how='left',
        sort=False,
        validate='one_to_one',
    )
    if not np.array_equal(test_ids, test['id'].to_numpy()):
        raise ValueError('Test row order/ids changed during lag join')
    if test[LAG_FEATURE].isna().any():
        raise ValueError('Missing sales_lag_364 for Test; check historical dates')
    if len(FEATURES) != 26 or set(FEATURES) - set(test.columns):
        raise ValueError('Feature configuration mismatched after Test lag join')
    return test


def make_model_matrix(train, test):
    """Match modeling.py's prepare_model_data() categorical encoding exactly."""
    x_train = train[FEATURES].copy()
    x_test = test[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        categories = pd.Index(train[col].dropna().unique())
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_test[col] = pd.Categorical(x_test[col], categories=categories)
        if x_test[col].isna().any():
            raise ValueError(f'Unseen Test category: {col}')
    if list(x_train.columns) != FEATURES or list(x_test.columns) != FEATURES:
        raise ValueError('Feature order changed')
    return x_train, x_test


def run(run_dir=None, output_path=None):
    validate_configuration()
    selected_run, fixed_summary, n_estimators = load_fixed_configuration(run_dir)
    params = FIXED_PARAMS.copy()
    # Only the iteration count is determined by the saved FIXED 3-fold result.
    params['n_estimators'] = n_estimators

    print(f'[FIXED CV] {selected_run}')
    print(f'[FEATURES] {len(FEATURES)}: {FEATURES}')
    print(f'[CV RMSLE] {fixed_summary["mean_rmsle"]:.6f}')
    print(f'[PARAMS] Fixed baseline, n_estimators={n_estimators}; Optuna NOT used', flush=True)

    # Reuse the existing project pipeline and train feature engineering verbatim.
    train_raw, test_raw, holidays = main()
    del holidays
    train = prepare_features(train_raw)
    del train_raw
    test = prepare_test_features(test_raw, train)
    del test_raw
    gc.collect()

    if train['date'].max() != pd.Timestamp('2017-08-15'):
        raise ValueError('Train cutoff is not 2017-08-15')
    print(f'[DATA] train={len(train):,} rows; test={len(test):,} rows', flush=True)

    x_train, x_test = make_model_matrix(train, test)
    y_train = train[TARGET_COL]
    model = LGBMRegressor(**params)
    print('[TRAIN] Fitting on ALL available training dates ...', flush=True)
    model.fit(x_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    pred_log = np.asarray(model.predict(x_test), dtype=float)
    if not np.isfinite(pred_log).all():
        raise ValueError('Predictions contain NaN or infinity')

    # Same clipping convention as modeling.py's calculate_rmsle().
    sales_pred = np.expm1(np.maximum(pred_log, 0.0))
    if not np.isfinite(sales_pred).all() or (sales_pred < 0).any():
        raise ValueError('Invalid predicted sales')

    submission = pd.DataFrame({'id': test['id'].to_numpy(), 'sales': sales_pred})
    if submission['id'].duplicated().any() or len(submission) != 28512:
        raise ValueError('Submission must contain 28,512 unique ids')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_path is None:
        output_path = ROOT / 'submissions' / f'submission_base23_payday_lag364_fixed_{timestamp}.csv'
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite an existing submission: {output_path}')
    submission.to_csv(output_path, index=False)

    # Keep a minimal audit log of EXACT settings used for this submission.
    metadata = {
        'source_fixed_run': str(selected_run),
        'source_fixed_cv_rmsle': fixed_summary['mean_rmsle'],
        'train_end': '2017-08-15',
        'test_start': '2017-08-16',
        'test_end': '2017-08-31',
        'n_train_rows': int(len(train)),
        'n_test_rows': int(len(test)),
        'features': FEATURES,
        'params': params,
        'target': 'log1p(sales)',
        'prediction_inverse': 'expm1(max(pred_log, 0))',
        'test_lag': 'sales from the same store_nbr/family exactly 364 calendar days earlier',
        'optuna_used': False,
        'submission_file': str(output_path),
    }
    output_path.with_suffix('.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('\n[SAVED]', output_path)
    print('[SAVED]', output_path.with_suffix('.json'))
    print('[PREVIEW]\n' + submission.head().to_string(index=False))
    print(f'[RANGE] sales min={sales_pred.min():.4f}, max={sales_pred.max():.4f}')
    print('Ready to upload the CSV to Kaggle. No Kaggle score exists until upload.')


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=None,
                        help='Folder of the fixed 3-fold run (containing run_config.json and fixed/summary.json).')
    parser.add_argument('--output', type=Path, default=None,
                        help='Submission CSV destination. Default: timestamped path under submissions/.')
    args = parser.parse_args()
    run(run_dir=args.run_dir, output_path=args.output)


if __name__ == '__main__':
    cli()
