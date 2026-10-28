"""
Final Kaggle submission: Base23 + Lag364 + Lag21 (25 features).

Place in src/final_submission.py next to main.py and feature_engineering.py.
Run from the project root:
    python src/final_submission.py --check-only
    python src/final_submission.py

Only ONE LightGBM model is fitted on ALL observed Train dates
(2013-01-01 through 2017-08-15). No CV, backtest, Optuna, ensemble or
Early Stopping is run. The 16-day Test period is 2017-08-16..2017-08-31.

Default training rounds: 3,627, carried over from the lag16 model's
CV median (1,627 / 4,582 / 3,627). This is a provisional choice for
lag21, NOT a verified same-model CV optimum. Override --n-estimators
when a lag21 CV median is available.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from main import main
from feature_engineering import add_lag_features


ROOT = Path(__file__).resolve().parent.parent
TRAIN_START = pd.Timestamp("2013-01-01")
TRAIN_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")
TEST_DATES = pd.date_range(TEST_START, TEST_END, freq="D")
HORIZON = 16
N_STORES = 54
N_FAMILIES = 33
N_SERIES = N_STORES * N_FAMILIES
EXPECTED_TRAIN_ROWS = 3_008_016  # Includes 7,128 restored Christmas rows
EXPECTED_TEST_ROWS = N_SERIES * HORIZON  # 28,512
LAGS = (364, 21)
FULL_TRAIN_N_ESTIMATORS = 4000
CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]
KEYS = ["store_nbr", "family"]

# Exactly the prior Base23 (not Payday29).
BASE23 = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion", "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]
FEATURES = BASE23 + ["sales_lag_364", "sales_lag_21"]

FIXED_PARAMS = dict(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.05,
    n_estimators=FULL_TRAIN_N_ESTIMATORS,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    min_split_gain=0.0,
    subsample=1.0,
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    metric="rmse",
    importance_type="split",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def check_configuration() -> None:
    if len(BASE23) != 23 or len(set(BASE23)) != 23:
        raise ValueError("Expected exactly the original 23 Base features")
    if len(FEATURES) != 25 or len(set(FEATURES)) != 25:
        raise ValueError("Expected exactly Base23 + two unique sales lags")
    forbidden = {
        "is_15th", "is_month_end", "oil_raw", "dcoilwtico",
        "transactions", "transactions_lag_16", "sales_lag_16", "sales_lag_28", "sales", "sales_log",
    }
    if forbidden.intersection(FEATURES):
        raise ValueError("Unexpected Payday/Oil/Transactions/target feature")
    if len(TEST_DATES) != HORIZON or TEST_START != TRAIN_END + pd.Timedelta(days=1):
        raise ValueError("Unexpected forecast horizon")
    for lag in LAGS:
        if TEST_END - pd.Timedelta(days=lag) > TRAIN_END:
            raise ValueError(f"Test lag{lag} would access unknown future sales")


def prepare_full_train(raw: pd.DataFrame) -> pd.DataFrame:
    """Build train lags from observed historical sales only."""
    required = ["date", "sales"] + BASE23
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"main() Train missing columns: {missing}")
    train = raw.loc[:, required].copy()
    train["date"] = pd.to_datetime(train["date"], errors="raise")
    train["sales"] = pd.to_numeric(train["sales"], errors="raise")
    if (len(train) != EXPECTED_TRAIN_ROWS
            or train["date"].min() != TRAIN_START
            or train["date"].max() != TRAIN_END):
        raise ValueError(
            "Expected the complete restored Train panel "
            "2013-01-01..2017-08-15 (3,008,016 rows)"
        )
    if train[KEYS + ["date"]].isna().any().any():
        raise ValueError("Null Train store/family/date")
    if train.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate Train store/family/date")
    if train[BASE23].isna().any().any():
        raise ValueError("Missing Base23 predictors in Train")
    if not np.isfinite(train["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Train contains non-finite sales")
    if (train["sales"] < 0).any():
        raise ValueError("Train sales must be nonnegative")

    dayofweek = train["date"].dt.day_of_week
    if not np.array_equal(train["dayofweek"], dayofweek):
        raise ValueError("Train dayofweek/calendar mismatch")
    if not np.array_equal(train["is_weekend"], (dayofweek >= 5).astype(int)):
        raise ValueError("Train is_weekend/calendar mismatch")

    train = train.sort_values(KEYS + ["date"]).reset_index(drop=True)
    gap = train.groupby(KEYS, observed=True)["date"].diff()
    if (gap.notna() & gap.ne(pd.Timedelta(days=1))).any():
        raise ValueError("Train panel has missing days within a series")
    group_counts = train.groupby(KEYS, observed=True).size()
    if len(group_counts) != N_SERIES or group_counts.nunique() != 1:
        raise ValueError("Train must be a complete 1,782-series daily panel")

    # Use the SAME feature-engineering helper as the previous experiment.
    train = add_lag_features(train, lags=LAGS)
    position = train.groupby(KEYS, observed=True).cumcount()
    for lag in LAGS:
        col = f"sales_lag_{lag}"
        if col not in train.columns:
            raise ValueError(f"Lag helper did not create {col}")
        expected_null = position.lt(lag)
        if not train[col].isna().equals(expected_null):
            raise ValueError(f"Unexpected initial missing values for {col}")
        if not np.isfinite(train[col].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {col} in Train")
    train["sales_log"] = np.log1p(train["sales"])
    print(f"[TRAIN] {len(train):,} rows; dates {TRAIN_START.date()}..{TRAIN_END.date()}")
    return train


def prepare_test(raw: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Keep every Test row. For each lag, join the ACTUAL Train sales at
    (store_nbr, family, test_date - lag) by calendar date.
    No Test sales or future target values enter the predictors.
    """
    required = ["id", "date"] + BASE23
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"main() Test missing columns: {missing}")
    test = raw.loc[:, required].copy().reset_index(drop=True)
    test["date"] = pd.to_datetime(test["date"], errors="raise")
    if len(test) != EXPECTED_TEST_ROWS or set(test["date"]) != set(TEST_DATES):
        raise ValueError("Test must cover all 28,512 rows in 2017-08-16..31")
    if test[["id", "date"] + KEYS].isna().any().any():
        raise ValueError("Test contains null ID/panel key/date")
    if test["id"].duplicated().any() or test.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate Test id or store/family/date")
    if test[BASE23].isna().any().any():
        raise ValueError("Test contains missing Base23 features")

    group_counts = test.groupby(KEYS, observed=True).size()
    if len(group_counts) != N_SERIES or not group_counts.eq(HORIZON).all():
        raise ValueError("Each of 1,782 Test series must have 16 dates")
    dayofweek = test["date"].dt.day_of_week
    if not np.array_equal(test["dayofweek"], dayofweek):
        raise ValueError("Test dayofweek/calendar mismatch")
    if not np.array_equal(test["is_weekend"], (dayofweek >= 5).astype(int)):
        raise ValueError("Test is_weekend/calendar mismatch")

    original_ids = test["id"].to_numpy(copy=True)
    for lag in LAGS:
        col = f"sales_lag_{lag}"
        source_date = f"_source_date_{lag}"
        test[source_date] = test["date"] - pd.Timedelta(days=lag)
        min_date, max_date = test[source_date].min(), test[source_date].max()
        if max_date > TRAIN_END:
            raise ValueError(f"{col} references sales after the Train cutoff")

        # Trim history for a smaller, exact (date, store, family) join.
        history = train.loc[
            train["date"].between(min_date, max_date),
            KEYS + ["date", "sales"],
        ].rename(columns={"date": source_date, "sales": col})
        test = test.merge(
            history,
            on=KEYS + [source_date],
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if not np.array_equal(original_ids, test["id"].to_numpy()):
            raise ValueError(f"Test id/order changed during {col} join")
        if test[col].isna().any() or not np.isfinite(
            test[col].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Missing or invalid Test {col}")
        print(f"[LAG QC] {col}: source dates "
              f"{min_date.date()}..{max_date.date()}, all observed in Train")
        test = test.drop(columns=[source_date])

    if any(c not in test.columns for c in FEATURES):
        raise ValueError("The 25 expected Test features are not available")
    print(f"[TEST] {len(test):,} rows; dates {TEST_START.date()}..{TEST_END.date()}")
    return test


def prepare_model_inputs(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train.loc[:, FEATURES].copy()
    x_test = test.loc[:, FEATURES].copy()
    for col in CATEGORICAL:
        categories = pd.Index(train[col].dropna().unique())
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_test[col] = pd.Categorical(x_test[col], categories=categories)
        if x_test[col].isna().any():
            raise ValueError(f"Unseen category in Test: {col}")
    if list(x_train.columns) != FEATURES or list(x_test.columns) != FEATURES:
        raise ValueError("Train/Test feature order differs")
    return x_train, x_test


def align_submission(test: pd.DataFrame, pred_sales: np.ndarray) -> pd.DataFrame:
    predictions = pd.DataFrame({
        "id": test["id"].to_numpy(copy=True),
        "sales": pred_sales,
    })
    sample_locations = [
        ROOT / "data" / "sample_submission.csv",
        ROOT / "sample_submission.csv",
    ]
    sample_file = next((path for path in sample_locations if path.is_file()), None)
    if sample_file:
        sample = pd.read_csv(sample_file, usecols=["id"])
        if (len(sample) != EXPECTED_TEST_ROWS
                or sample["id"].duplicated().any()
                or set(sample["id"]) != set(predictions["id"])):
            raise ValueError("sample_submission IDs differ from Test IDs")
        submission = sample.merge(
            predictions, on="id", how="left", sort=False,
            validate="one_to_one",
        )
        print(f"[ID QC] Matched sample_submission order: {sample_file}")
    else:
        # Kaggle uses the id column, but deterministic ordering is useful.
        submission = predictions.sort_values("id").reset_index(drop=True)
        print("[ID QC] sample_submission.csv not found; sorted by Test id")
    if (len(submission) != EXPECTED_TEST_ROWS
            or submission["id"].duplicated().any()
            or submission["sales"].isna().any()
            or not np.isfinite(submission["sales"].to_numpy()).all()
            or (submission["sales"] < 0).any()):
        raise ValueError("Submission failed row count/ID/prediction QC")
    if list(submission.columns) != ["id", "sales"]:
        raise ValueError("Submission columns must be exactly id,sales")
    return submission


def run(output_path: Path | None, check_only: bool, n_estimators: int) -> None:
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    check_configuration()
    # main() uses data paths relative to project root.
    os.chdir(ROOT)
    loaded = main()
    if not isinstance(loaded, (tuple, list)) or len(loaded) != 3:
        raise ValueError("Expected main() -> (train_df, test_df, holidays)")
    train_raw, test_raw, holidays = loaded
    del holidays
    train = prepare_full_train(train_raw)
    del train_raw
    test = prepare_test(test_raw, train)
    del test_raw
    x_train, x_test = prepare_model_inputs(train, test)

    print(f"[FEATURES] {len(FEATURES)}: {FEATURES}")
    print(f"[TRAINING] Entire restored Train; fixed iterations={n_estimators}; "
          "no CV/backtest/Optuna/Early Stopping.")
    if check_only:
        print("[CHECK ONLY] All pre-training checks passed. "
              "No model fitted and no submission saved.")
        return

    parameters = {**FIXED_PARAMS, "n_estimators": n_estimators}
    model = LGBMRegressor(**parameters)
    model.fit(
        x_train,
        train["sales_log"],
        categorical_feature=CATEGORICAL,
    )
    if model.booster_.feature_name() != FEATURES:
        raise ValueError("Booster feature order differs from the intended 25")
    pred_log = np.asarray(
        model.predict(x_test, num_iteration=n_estimators),
        dtype=float,
    )
    if len(pred_log) != EXPECTED_TEST_ROWS or not np.isfinite(pred_log).all():
        raise ValueError("Non-finite predictions or incorrect Test row count")
    pred_sales = np.expm1(np.maximum(pred_log, 0.0))
    if not np.isfinite(pred_sales).all() or (pred_sales < 0).any():
        raise ValueError("Invalid nonnegative sales predictions")
    submission = align_submission(test, pred_sales)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"submission_base23_lag364_lag21_"
        f"fixed_fulltrain_iter{n_estimators}_{stamp}.csv"
    )
    destination = (Path(output_path).expanduser().resolve()
                   if output_path is not None
                   else ROOT / "submissions" / filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {destination}")
    # Exclusive creation prevents accidental overwrite of previous submissions.
    with destination.open("x", encoding="utf-8", newline="") as handle:
        submission.to_csv(handle, index=False)

    metadata = dict(
        experiment="Base23 + Lag364 + Lag21",
        train_period=[str(TRAIN_START.date()), str(TRAIN_END.date())],
        test_period=[str(TEST_START.date()), str(TEST_END.date())],
        train_rows=int(len(train)),
        test_rows=int(len(test)),
        feature_count=len(FEATURES),
        feature_names=FEATURES,
        parameters=parameters,
        fixed_cv_best_iterations=None,
        iteration_reference_experiment="base23_lag364_lag16",
        iteration_reference_cv_best_iterations=[1627, 4582, 3627],
        iteration_rule=("provisional: inherited lag16 CV median; lag21 CV not verified"
                        if n_estimators == FULL_TRAIN_N_ESTIMATORS
                        else "explicit command-line override"),
        target="log1p(sales)",
        inverse_prediction="expm1(max(pred_log, 0))",
        test_lags=list(LAGS),
        lag_policy="exact calendar join to observed sales before 2017-08-16",
        no_cv_retraining=True,
        no_backtest_retraining=True,
        no_optuna=True,
        submission_file=str(destination),
    )
    with destination.with_suffix(".json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {destination}")
    print(f"[SAVED] {destination.with_suffix('.json')}")
    print("[PREVIEW]\n" + submission.head().to_string(index=False))
    print(f"[PREDICTION RANGE] min={pred_sales.min():.6f}, "
          f"max={pred_sales.max():.6f}")
    print("Ready to upload CSV to Kaggle. "
          "No Public Score exists until uploaded and evaluated.")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run full data/feature/lag QC without fitting LightGBM.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional exact CSV path; otherwise save under submissions/.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=FULL_TRAIN_N_ESTIMATORS,
        help="Full-training rounds (default: 3627, inherited lag16 CV median; not verified for lag21).",
    )
    args = parser.parse_args()
    run(output_path=args.output, check_only=args.check_only, n_estimators=args.n_estimators)


if __name__ == "__main__":
    cli()
