"""Train Trial17 + family 7/21 uplift and write a Kaggle submission."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(r"C:\projects\Store Sales-ML")

TRAIN_START = pd.Timestamp("2013-01-01")
TRAIN_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

EXPECTED_TRAIN = 3_008_016
EXPECTED_TEST = 28_512
EXPECTED_STORES = 54
EXPECTED_FAMILIES = 33

N_ITERATIONS = 6000
LAGS = (364, 28)

BASE23 = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion", "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day",
    "event_city_foundation", "event_good_friday",
    "event_labor_day", "event_other",
]

BASE_FEATURES = BASE23 + [
    "sales_lag_364",
    "sales_lag_28",
]

UPLIFT_FEATURE = "family_sales_uplift_7d_prev21d"
FEATURES = BASE_FEATURES + [UPLIFT_FEATURE]

CATEGORICAL = [
    "store_nbr", "family", "city",
    "state", "type", "cluster",
]

ROW_KEYS = ["date", "store_nbr", "family"]
TARGET = "sales_log"

PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "learning_rate": 0.023127655409786754,
    "n_estimators": N_ITERATIONS,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "min_split_gain": 0.0,
    "subsample": 0.7742745551930794,
    "subsample_freq": 1,
    "colsample_bytree": 0.9621753805232081,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "metric": "rmse",
    "importance_type": "split",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def progress(env):
    step = env.iteration + 1
    if step % 500 == 0:
        print(
            f"  iteration {step}/{N_ITERATIONS}",
            flush=True,
        )


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def prepare_train(raw, add_lag_features):
    required = ["date", "sales"] + BASE23
    missing = set(required) - set(raw.columns)
    check(not missing, f"Missing train columns: {missing}")

    train = raw[required].copy()
    train["date"] = pd.to_datetime(train["date"])
    train["sales"] = pd.to_numeric(
        train["sales"], errors="raise"
    )

    check(len(train) == EXPECTED_TRAIN, "Invalid train size")
    check(
        train["date"].min() == TRAIN_START
        and train["date"].max() == TRAIN_END,
        "Invalid training dates",
    )
    check(
        not train.duplicated(ROW_KEYS).any(),
        "Duplicate training keys",
    )
    check(
        train["store_nbr"].nunique() == EXPECTED_STORES,
        "Expected 54 stores",
    )
    check(
        train["family"].nunique() == EXPECTED_FAMILIES,
        "Expected 33 families",
    )
    check(
        not train[BASE23].isna().any().any(),
        "Missing Base23 train values",
    )
    check(
        np.isfinite(train["sales"]).all()
        and train["sales"].ge(0).all(),
        "Invalid training sales",
    )

    train = train.sort_values(
        ["store_nbr", "family", "date"]
    ).reset_index(drop=True)

    # Preserve original Trial17 lag generation.
    train = add_lag_features(train, lags=LAGS)

    for lag in LAGS:
        check(
            f"sales_lag_{lag}" in train.columns,
            f"Missing lag {lag}",
        )

    train[TARGET] = np.log1p(train["sales"])

    check(len(BASE_FEATURES) == 25, "Base25 mismatch")

    return train


def prepare_test(original_test, train):
    required = ["id", "date"] + BASE23
    missing = set(required) - set(original_test.columns)
    check(not missing, f"Missing test columns: {missing}")

    test = original_test[required].copy()
    test["date"] = pd.to_datetime(test["date"])
    test["id"] = pd.to_numeric(
        test["id"], errors="raise"
    ).astype("int64")

    check(len(test) == EXPECTED_TEST, "Invalid test size")
    check(
        test["date"].min() == TEST_START
        and test["date"].max() == TEST_END,
        "Invalid test dates",
    )
    check(test["id"].is_unique, "Duplicate test IDs")
    check(
        not test.duplicated(ROW_KEYS).any(),
        "Duplicate test panel keys",
    )
    check(
        not test[BASE23].isna().any().any(),
        "Missing Base23 test values",
    )

    daily_count = test.groupby("date").size()
    check(
        len(daily_count) == 16
        and daily_count.eq(54 * 33).all(),
        "Incomplete daily test panel",
    )

    original_ids = test["id"].reset_index(drop=True)

    # Match test lag values to historical training sales.
    # No recursive use of predicted test sales.
    for lag in LAGS:
        name = f"sales_lag_{lag}"

        history = train[ROW_KEYS + ["sales"]].copy()
        history["date"] += pd.Timedelta(days=lag)
        history = history.rename(
            columns={"sales": name}
        )

        test = test.merge(
            history,
            on=ROW_KEYS,
            how="left",
            sort=False,
            validate="one_to_one",
        )

        check(
            np.isfinite(test[name]).all(),
            f"Invalid test feature {name}",
        )

        del history

    check(
        test["id"].reset_index(drop=True).equals(
            original_ids
        ),
        "Test ID order changed",
    )

    return test


def add_family_uplift(train, test):
    """Calculate daily family uplift and freeze it at the forecast start."""
    grouped = train.groupby(["date", "family"], observed=True)["sales"]
    check(grouped.size().eq(EXPECTED_STORES).all(), "Incomplete family/day store panel")
    dates = pd.date_range(TRAIN_START, TEST_START)
    totals = grouped.sum().unstack("family").reindex(dates)
    check(totals.shape[1] == EXPECTED_FAMILIES, "Unexpected family count")
    check(totals.loc[:TRAIN_END].notna().all().all(), "Missing historical family/day totals")

    recent_7d = totals.shift(1).rolling(7, min_periods=7).mean()
    previous_21d = totals.shift(8).rolling(21, min_periods=21).mean()
    uplift = np.log1p(recent_7d) - np.log1p(previous_21d)

    keys = pd.MultiIndex.from_frame(train[["date", "family"]])
    train[UPLIFT_FEATURE] = uplift.rename_axis(index="date").stack().reindex(keys).to_numpy(dtype=np.float64)
    test[UPLIFT_FEATURE] = test["family"].map(uplift.loc[TEST_START]).to_numpy(dtype=np.float64)
    check(np.isfinite(test[UPLIFT_FEATURE]).all(), "Missing test family uplift")
    print(f"[FEATURE] {UPLIFT_FEATURE}: {len(test):,} test values available", flush=True)


def make_matrices(train, test, features):
    xt = train[features].copy()
    xp = test[features].copy()

    for col in CATEGORICAL:
        categories = pd.Index(
            train[col].dropna().unique()
        )

        xt[col] = pd.Categorical(
            xt[col], categories=categories
        )
        xp[col] = pd.Categorical(
            xp[col], categories=categories
        )

        check(
            not xp[col].isna().any(),
            f"Unknown test category: {col}",
        )

    check(
        not any(
            "oil" in feature.lower()
            or "dcoil" in feature.lower()
            or "promo_high_response" in feature
            for feature in features
        ),
        "Unexpected feature",
    )

    return xt, xp


def train_and_predict(train, test):
    xt, xp = make_matrices(train, test, FEATURES)

    print(
        "\n[MODEL] Trial17 + family 7/21 uplift"
        f"\n  Features: {len(FEATURES)}"
        f"\n  Train rows: {len(xt):,}"
        f"\n  Test rows: {len(xp):,}"
        f"\n  Iterations: {N_ITERATIONS:,}",
        flush=True,
    )

    model = lgb.LGBMRegressor(**PARAMS)

    # Fixed full-data training.
    # No eval_set or early stopping.
    model.fit(
        xt,
        train[TARGET],
        categorical_feature=CATEGORICAL,
        callbacks=[progress],
    )

    actual_rounds = model.booster_.current_iteration()
    check(
        actual_rounds == N_ITERATIONS,
        f"Unexpected rounds: {actual_rounds}",
    )

    prediction_log = np.asarray(
        model.predict(
            xp,
            num_iteration=N_ITERATIONS,
            validate_features=True,
        ),
        dtype=np.float64,
    )

    check(
        np.isfinite(prediction_log).all(),
        "Nonfinite log predictions",
    )

    prediction_sales = np.expm1(
        np.maximum(prediction_log, 0.0)
    )

    check(
        np.isfinite(prediction_sales).all()
        and (prediction_sales >= 0).all(),
        "Invalid sales predictions",
    )

    del xt, xp, model, prediction_log
    gc.collect()

    return prediction_sales


def write_submission(
    test, predictions, sample_path, output_path
):
    check(
        sample_path.is_file(),
        f"Missing sample submission: {sample_path}",
    )

    sample = pd.read_csv(sample_path)

    check(
        list(sample.columns) == ["id", "sales"],
        "Invalid sample submission columns",
    )
    check(
        len(sample) == EXPECTED_TEST
        and sample["id"].is_unique,
        "Invalid sample submission IDs",
    )
    check(
        set(sample["id"]) == set(test["id"]),
        "Sample/test ID mismatch",
    )

    predicted = pd.DataFrame({
        "id": test["id"].to_numpy(),
        "sales": predictions,
    })

    submission = sample[["id"]].merge(
        predicted,
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    check(
        submission["id"].equals(sample["id"]),
        "Submission ID order mismatch",
    )
    check(
        len(submission) == EXPECTED_TEST,
        "Submission row count mismatch",
    )
    check(
        np.isfinite(submission["sales"]).all()
        and submission["sales"].ge(0).all(),
        "Submission contains invalid sales",
    )

    save_csv(submission, output_path)

    # Verify the saved CSV.
    saved = pd.read_csv(output_path)

    check(
        list(saved.columns) == ["id", "sales"]
        and len(saved) == EXPECTED_TEST,
        "Saved CSV structure mismatch",
    )
    check(
        saved["id"].equals(submission["id"]),
        "Saved CSV ID mismatch",
    )
    check(
        np.allclose(
            saved["sales"],
            submission["sales"],
            rtol=1e-12,
            atol=1e-12,
        ),
        "Saved CSV predictions mismatch",
    )

    print(
        f"[SAVED] {output_path}"
        "\n[QC] 28,512 rows | IDs/order/values OK",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "submissions" / f"trial17_family_uplift_6000_{timestamp}"
    )
    sample_path = root / "data" / "sample_submission.csv"
    check(sample_path.is_file(), f"Missing sample submission: {sample_path}")
    check(len(FEATURES) == 26, "Expected 26 features")
    check(
        TEST_START == TRAIN_END + pd.Timedelta(days=1)
        and TEST_END - pd.Timedelta(days=min(LAGS)) <= TRAIN_END,
        "Invalid test cutoff or lag window",
    )

    sys.path.insert(0, str(root / "src"))
    from main import main as original_pipeline
    from feature_engineering import add_lag_features

    os.chdir(root)
    print(f"[ITERATIONS] {N_ITERATIONS:,} fixed rounds")
    print("[OUTPUT]", output)
    raw, original_test, holidays = original_pipeline()
    train = prepare_train(raw, add_lag_features)
    test = prepare_test(original_test, train)
    del raw, original_test, holidays
    gc.collect()
    add_family_uplift(train, test)

    if args.check_only:
        print("[CHECK ONLY] Features prepared.")
        return

    predictions = train_and_predict(train, test)
    write_submission(test, predictions, sample_path, output / "submission_trial17_family_uplift_6000.csv")
    metadata = {
        "params": PARAMS,
        "features": FEATURES,
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[COMPLETE]", output)


if __name__ == "__main__":
    main()
