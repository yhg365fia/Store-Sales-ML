"""August 16-31, 2016 diagnostic backtest: Group1 Lag16+EMA7 with/without oil.
Reuse cached Trial17 outside Group1; train two all-family models with fixed rounds.
Requires existing ROOT src/data/results. No other experiment script required.
No early stopping, tuning or submission. Not an independent holdout (2017 CV settings).
--check-only validates features and cached references without training.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

ROOT = Path(r'C:\projects\Store Sales-ML')
CACHE = ROOT / 'results/base23_lag364_lag28_optuna/20261028_182041_683679/cv_cache/07590e4e8d7e52dbe5a3'
sys.path.insert(0, str(ROOT / 'src'))
from main import main as original_pipeline
from feature_engineering import add_lag_features

BASE23 = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion", "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]

LAGS = (364, 28)

EXPERIMENT = "base23_lag364_lag28"

FEATURES = BASE23 + ["sales_lag_364", "sales_lag_28"]

CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]

KEYS = ["store_nbr", "family"]

ROW_KEYS = ["date"] + KEYS

TARGET = "sales_log"

HORIZON = 16

EXPECTED_TRAIN_ROWS = 3_008_016

EXPECTED_VALID_ROWS = 28_512

N_SERIES = 54 * 33

N_ESTIMATORS = 8_000

EARLY_STOPPING_ROUNDS = 150

FOLDS = [
    dict(name="Fold 1", train_end="2017-06-28",
         valid_start="2017-06-29", valid_end="2017-07-14"),
    dict(name="Fold 2", train_end="2017-07-14",
         valid_start="2017-07-15", valid_end="2017-07-30"),
    dict(name="Fold 3", train_end="2017-07-30",
         valid_start="2017-07-31", valid_end="2017-08-15"),
]

AUGUST_BACKTEST = dict(
    name="August 2016 Backtest",
    train_end="2016-08-15",
    valid_start="2016-08-16",
    valid_end="2016-08-31",
)

SPLITS = FOLDS + [AUGUST_BACKTEST]

def check_configuration() -> None:
    if len(BASE23) != 23 or len(set(BASE23)) != 23:
        raise ValueError("BASE23 must have 23 unique columns")
    if len(FEATURES) != 25 or len(set(FEATURES)) != 25:
        raise ValueError("Expected exactly Base23 + lag364 + lag28")
    if FEATURES != BASE23 + ["sales_lag_364", "sales_lag_28"]:
        raise ValueError("Unexpected feature configuration")

    for split in SPLITS:
        cutoff = pd.Timestamp(split["train_end"])
        start = pd.Timestamp(split["valid_start"])
        stop = pd.Timestamp(split["valid_end"])
        if start != cutoff + pd.Timedelta(days=1):
            raise ValueError(f"Nonadjacent cutoff: {split['name']}")
        if (stop - start).days + 1 != HORIZON:
            raise ValueError(f"Validation must be 16 days: {split['name']}")
        for lag in LAGS:
            if stop - pd.Timedelta(days=lag) > cutoff:
                raise ValueError(f"Future target would enter lag{lag}: {split['name']}")

def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    check_configuration()
    required = ["date", "sales"] + BASE23
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"main() missing required Train columns: {missing}")

    df = raw.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["sales"] = pd.to_numeric(df["sales"], errors="raise")
    if (len(df) != EXPECTED_TRAIN_ROWS
            or df["date"].min() != pd.Timestamp("2013-01-01")
            or df["date"].max() != pd.Timestamp("2017-08-15")):
        raise ValueError("Expected complete restored Train 2013-01-01..2017-08-15")
    if df[KEYS + ["date"]].isna().any().any():
        raise ValueError("Null panel key or date")
    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store_nbr/family/date")
    if df[BASE23].isna().any().any():
        raise ValueError("Null in Base23 predictors")
    if (not np.isfinite(df["sales"].to_numpy(dtype=float)).all()
            or (df["sales"] < 0).any()):
        raise ValueError("Invalid sales targets")

    weekday = df["date"].dt.day_of_week
    if not np.array_equal(df["dayofweek"], weekday):
        raise ValueError("dayofweek inconsistent with date")
    if not np.array_equal(df["is_weekend"], (weekday >= 5).astype(int)):
        raise ValueError("is_weekend inconsistent with date")

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    group_counts = df.groupby(KEYS, observed=True).size()
    if len(group_counts) != N_SERIES or group_counts.nunique() != 1:
        raise ValueError("Expected complete 1,782 store/family series")
    gaps = df.groupby(KEYS, observed=True)["date"].diff()
    if (gaps.notna() & gaps.ne(pd.Timedelta(days=1))).any():
        raise ValueError("Missing calendar dates within a store/family series")

    # Reuse the existing feature engineering helper, without changing any lags.
    df = add_lag_features(df, lags=LAGS)
    positions = df.groupby(KEYS, observed=True).cumcount()
    for lag in LAGS:
        col = f"sales_lag_{lag}"
        if col not in df.columns:
            raise ValueError(f"Existing lag helper did not create {col}")
        if not df[col].isna().equals(positions.lt(lag)):
            raise ValueError(f"Unexpected missing positions for {col}")
        if not np.isfinite(df[col].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite values in {col}")
    df[TARGET] = np.log1p(df["sales"])
    print(f"[DATA] Restored train={len(df):,}; computed lags={LAGS}; "
          "initial lag NaNs retained.", flush=True)
    print(f"[FEATURES] {EXPERIMENT}: {len(FEATURES)} predictors", flush=True)
    print("[POLICY] Fixed-round diagnostic backtest; no early stopping.", flush=True)
    return df

def make_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(split["train_end"])
    start = pd.Timestamp(split["valid_start"])
    stop = pd.Timestamp(split["valid_end"])
    train = df.loc[df["date"].le(cutoff)]
    valid = df.loc[df["date"].between(start, stop)]
    counts = valid.groupby(KEYS, observed=True).size()
    if train.empty or train["date"].max() != cutoff:
        raise ValueError(f"Train cutoff mismatch: {split['name']}")
    if (len(valid) != EXPECTED_VALID_ROWS
            or len(counts) != N_SERIES
            or not counts.eq(HORIZON).all()
            or set(valid["date"]) != set(pd.date_range(start, stop))):
        raise ValueError(f"Incomplete 16-day validation panel: {split['name']}")
    for lag in LAGS:
        if valid[f"sales_lag_{lag}"].isna().any():
            raise ValueError(f"Missing valid lag{lag}: {split['name']}")
        if stop - pd.Timedelta(days=lag) > cutoff:
            raise ValueError(f"Leakage in lag{lag}: {split['name']}")
    return train, valid

def make_matrices(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train.loc[:, features].copy()
    x_valid = valid.loc[:, features].copy()
    for col in CATEGORICAL:
        # Category vocabulary is determined by training rows only.
        categories = pd.Index(train[col].dropna().unique())
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_valid[col] = pd.Categorical(x_valid[col], categories=categories)
        if x_valid[col].isna().any():
            raise ValueError(f"Unseen validation category: {col}")
    if list(x_train.columns) != features or list(x_valid.columns) != features:
        raise ValueError("Train/Valid feature ordering changed")
    return x_train, x_valid

def rmsle(y_true, predicted_log) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predicted_log, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite model prediction")
    return float(np.sqrt(np.mean((np.log1p(y) - np.maximum(p, 0)) ** 2)))

PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression', 'learning_rate': 0.023127655409786754, 'n_estimators': 8000, 'num_leaves': 127, 'max_depth': -1, 'min_child_samples': 50, 'min_split_gain': 0.0, 'subsample': 0.7742745551930794, 'subsample_freq': 1, 'colsample_bytree': 0.9621753805232081, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'metric': 'rmse', 'importance_type': 'split', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}


GROUPS = {
    1: ['BEAUTY', 'HARDWARE', 'HOME APPLIANCES', 'LAWN AND GARDEN',
        'LIQUOR,WINE,BEER', 'MAGAZINES', 'SCHOOL AND OFFICE SUPPLIES'],
    2: ['AUTOMOTIVE', 'CELEBRATION', 'HOME AND KITCHEN II', 'PET SUPPLIES',
        'PLAYERS AND ELECTRONICS', 'SEAFOOD'],
    4: ['LINGERIE', 'LADIESWEAR', 'HOME AND KITCHEN I', 'GROCERY II'],
}
NEW_FEATURE = 'group1_lag16'
EMA_FEATURE = 'group1_ema7_shift16'
OIL_FEATURE = 'group1_oil_raw'
NEW_FEATURES = FEATURES + [NEW_FEATURE, EMA_FEATURE, OIL_FEATURE]
BASELINE_DIR = ROOT / 'results/base23_lag364_lag28_optuna/20261028_182041_683679/best_trial_0017'
CV_DIRS = {
    'ema7': ROOT / 'results/group1_lag16_ema7_mix/20261102_214811_779601',
    'ema7_oil': ROOT / 'results/group1_lag16_ema7_oil_raw_mix/20261102_222706_228001',
}
VARIANTS = {'ema7': FEATURES + [NEW_FEATURE, EMA_FEATURE], 'ema7_oil': NEW_FEATURES}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def select_rounds(name, rule):
    directory = CV_DIRS[name]
    cfg = read_json(directory / 'config.json')
    if (cfg['features'] != VARIANTS[name] or cfg['params'] != PARAMS
            or cfg['folds'] != FOLDS or cfg['group1'] != GROUPS[1]):
        raise ValueError(f'{name}: CV configuration mismatch')
    records = read_json(directory / 'best_iterations.json')
    counts = {int(r['fold']): int(r['best_iteration']) for r in records}
    if len(records) != 3 or set(counts) != {1, 2, 3} or min(counts.values()) <= 0:
        raise ValueError('Invalid CV iterations')
    chosen = [1, 2, 3] if rule == 'mean3' else [2, 3]
    rounds = int(np.floor(np.mean([counts[i] for i in chosen]) + 0.5))
    return rounds, dict(rule=rule, folds=chosen, cv_iterations=counts, rounds=rounds)


def read_baseline(valid):
    meta = read_json(BASELINE_DIR / 'august_2016_backtest.json')
    if (meta['features'] != FEATURES or meta['parameters'] != dict(PARAMS, n_estimators=6770)
            or meta['split'] != AUGUST_BACKTEST or meta['early_stopping_used']):
        raise ValueError('Trial17 backtest configuration mismatch')
    cached = pd.read_csv(BASELINE_DIR / 'august_2016_backtest_predictions.csv', parse_dates=['date'])
    if len(cached) != len(valid) or cached.duplicated(ROW_KEYS).any():
        raise ValueError('Baseline row keys mismatch')
    paired = valid[ROW_KEYS + ['sales']].merge(
        cached[ROW_KEYS + ['y_true', 'pred_sales']], on=ROW_KEYS,
        how='left', validate='one_to_one')
    if (not np.allclose(paired.sales, paired.y_true)
            or not np.isfinite(paired.pred_sales).all() or paired.pred_sales.lt(0).any()):
        raise ValueError('Baseline truth/predictions mismatch')
    score = rmsle(paired.sales, np.log1p(paired.pred_sales))
    if not np.isclose(score, 0.7477345821606495, atol=1e-9, rtol=0):
        raise ValueError(f'Baseline score mismatch: {score}')
    return paired.rename(columns={'pred_sales': 'baseline_pred_sales'}), meta


def add_features(df):
    mask = df.family.isin(GROUPS[1])
    grouped = df.groupby(KEYS, observed=True)
    pos = grouped.cumcount()
    df[NEW_FEATURE] = grouped.sales.shift(16).where(mask)
    df[EMA_FEATURE] = grouped.sales.transform(
        lambda x: x.shift(16).ewm(span=7, adjust=False, min_periods=7).mean()).where(mask)
    if not df[NEW_FEATURE].isna().equals(~mask | pos.lt(16)):
        raise ValueError('Lag missing positions mismatch')
    if not df[EMA_FEATURE].isna().equals(~mask | pos.lt(22)):
        raise ValueError('EMA missing positions mismatch')
    oil = pd.read_csv(ROOT / 'data/oil.csv', parse_dates=['date'])
    if oil.date.isna().any() or oil.date.duplicated().any():
        raise ValueError('Invalid oil dates')
    oil['dcoilwtico'] = pd.to_numeric(oil.dcoilwtico, errors='raise')
    if not np.isfinite(oil.dcoilwtico.dropna()).all() or oil.dcoilwtico.dropna().lt(0).any():
        raise ValueError('Invalid oil values')
    # Exact CV policy: contemporaneous competition-provided oil; retain NaNs.
    df[OIL_FEATURE] = df.date.map(oil.set_index('date').dcoilwtico).where(mask)
    cutoff = pd.Timestamp(AUGUST_BACKTEST['train_end'])
    start, stop = map(pd.Timestamp, [AUGUST_BACKTEST['valid_start'], AUGUST_BACKTEST['valid_end']])
    if stop - pd.Timedelta(days=16) > cutoff:
        raise ValueError('Future sales access')
    for _, series in df.loc[mask].groupby(KEYS, observed=True):
        history = series.loc[series.date.le(cutoff)].set_index('date').sales
        target = series.loc[series.date.between(start, stop)]
        dates = target.date - pd.Timedelta(days=16)
        if not np.allclose(history.reindex(dates), target[NEW_FEATURE]):
            raise ValueError('Lag16 history/date mismatch')
        ema = history.ewm(span=7, adjust=False, min_periods=7).mean()
        if not np.allclose(ema.reindex(dates), target[EMA_FEATURE]):
            raise ValueError('EMA history/date mismatch')
    return df


def comparison(p, candidate, reference, label):
    partitions = [('overall', 'ALL', p)]
    for scope, column in [('group', 'analysis_group'), ('family', 'family'), ('horizon_day', 'horizon_day')]:
        partitions += [(scope, str(k), b) for k, b in p.groupby(column, observed=True)]
    rows = []
    for scope, name, b in partitions:
        base = rmsle(b.sales, np.log1p(b[reference]))
        new = rmsle(b.sales, np.log1p(b[candidate]))
        rows.append(dict(comparison=label, scope=scope, name=name, rows=len(b),
                         baseline_rmsle=base, new_rmsle=new, delta=new-base))
    return pd.DataFrame(rows)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--rounds-rule', choices=['mean3', 'mean23'], default='mean3')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'backtest_outputs')
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    choices = {name: select_rounds(name, args.rounds_rule) for name in VARIANTS}
    os.chdir(ROOT)
    raw, test, holidays = original_pipeline()
    df = prepare_features(raw)
    del raw, test, holidays
    df = df.loc[df.date.le(pd.Timestamp(AUGUST_BACKTEST['valid_end']))].copy()
    df = add_features(df)
    train, valid = make_split(df, AUGUST_BACKTEST)
    paired, baseline_meta = read_baseline(valid)
    group_map = {f: g for g, names in GROUPS.items() for f in names}
    paired['analysis_group'] = paired.family.astype(str).map(group_map).fillna(3).astype(int)
    paired['horizon_day'] = (paired.date - pd.Timestamp(AUGUST_BACKTEST['valid_start'])).dt.days + 1
    if paired.analysis_group.eq(1).sum() != 6048:
        raise ValueError('Unexpected Group1 row count')
    if valid.loc[valid.family.isin(GROUPS[1]), OIL_FEATURE].notna().sum() == 0:
        raise ValueError('No validation oil observations')
    for name, (rounds, _) in choices.items():
        print(f'[CONFIG] {name}: {rounds} fixed rounds ({args.rounds_rule})', flush=True)
    print('[QC] Baseline 0.7477345822, keys, truth and history cutoff verified.')
    if args.check_only:
        print('[CHECK ONLY] No training or output files.')
        return
    out = output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', dict(
        split=AUGUST_BACKTEST, baseline=baseline_meta, variants=VARIANTS,
        params=PARAMS, rounds={n: info for n, (_, info) in choices.items()},
        group1=GROUPS[1], training='all families', prediction='Group1 A, others cached B',
        early_stopping=False, oil_policy='same-day raw competition oil; NaNs retained',
        ema_policy='shift16 span7 adjust=False min_periods7',
        lightgbm_version=lightgbm.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Retrospective diagnostic: settings selected on 2017 CV; not independent holdout.'))
    reports = []
    del df
    gc.collect()
    for name, features in VARIANTS.items():
        rounds, _ = choices[name]
        x_train, x_valid = make_matrices(train, valid, features)
        model = LGBMRegressor(**dict(PARAMS, n_estimators=rounds))
        print(f'[FIT] {name}: {len(features)} features, {rounds} rounds, no ES', flush=True)
        model.fit(x_train, train[TARGET], categorical_feature=CATEGORICAL)
        pred_log = np.asarray(model.predict(x_valid, num_iteration=rounds), dtype=float)
        if not np.isfinite(pred_log).all():
            raise ValueError('Nonfinite predictions')
        col = f'{name}_model_pred_sales'
        model_pred = valid[ROW_KEYS].copy()
        model_pred[col] = np.expm1(np.maximum(pred_log, 0))
        if not np.isfinite(model_pred[col]).all():
            raise ValueError('Nonfinite sales predictions')
        paired = paired.merge(model_pred, on=ROW_KEYS, how='left', validate='one_to_one')
        selected = paired.analysis_group.eq(1)
        mix = f'{name}_mix_pred_sales'
        paired[mix] = paired.baseline_pred_sales.copy()
        paired.loc[selected, mix] = paired.loc[selected, col]
        if not np.isfinite(paired[mix]).all():
            raise ValueError('Invalid mixed predictions')
        if not np.array_equal(paired.loc[~selected, mix], paired.loc[~selected, 'baseline_pred_sales']):
            raise ValueError('Non-Group1 predictions changed')
        report = comparison(paired, mix, 'baseline_pred_sales', name + '_vs_trial17')
        report.to_csv(out / f'{name}_comparison.csv', index=False, encoding='utf-8-sig')
        reports.append(report)
        model.booster_.save_model(str(out / f'{name}_model.txt'))
        pd.DataFrame(dict(feature=features,
                          gain_importance=model.booster_.feature_importance(importance_type='gain'),
                          split_importance=model.feature_importances_)).to_csv(
                              out / f'{name}_importance.csv', index=False)
        paired.to_csv(out / 'predictions.csv', index=False)
        print(report.loc[report.scope.isin(['overall', 'group'])].round(6).to_string(index=False))
        del model, x_train, x_valid
        gc.collect()
    oil_report = comparison(paired, 'ema7_oil_mix_pred_sales', 'ema7_mix_pred_sales', 'oil_vs_no_oil')
    oil_report.to_csv(out / 'oil_vs_no_oil.csv', index=False, encoding='utf-8-sig')
    reports.append(oil_report)
    combined = pd.concat(reports, ignore_index=True)
    combined.to_csv(out / 'all_comparisons.csv', index=False, encoding='utf-8-sig')
    summary = combined.loc[combined.scope.isin(['overall', 'group'])]
    summary.to_csv(out / 'summary.csv', index=False, encoding='utf-8-sig')
    write_json(out / 'status.json', dict(status='complete', models_trained=2))
    print('\n[SUMMARY] Negative delta means improvement.\n' + summary.round(6).to_string(index=False))
    print(f'\nSaved to: {out}')


if __name__ == '__main__':
    run()

