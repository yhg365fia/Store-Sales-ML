"""Fold 1: train the ratio + uplift model and compare fixed store assignments."""

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
TRAIN_END = pd.Timestamp("2017-06-28")
VALID_START = pd.Timestamp("2017-06-29")
VALID_END = pd.Timestamp("2017-07-14")
FOLD1_REFERENCE = Path("src/trial17_fold1_backtest_results/20261111_140112_017715")


EXPECTED_STORES = 54
EXPECTED_FAMILIES = 33
MAX_ITERATIONS = 8000
EARLY_STOPPING_ROUNDS = 150
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
RATIO_FEATURE = "store_sales_mean_ratio_7d_28d"
UPLIFT_FEATURE = "family_sales_uplift_7d_prev21d"
RATIO_UPLIFT_FEATURES = BASE_FEATURES + [RATIO_FEATURE, UPLIFT_FEATURE]
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
    "n_estimators": MAX_ITERATIONS,
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

OLD_CV_STORES = {4, 9, 11, 18, 21, 25, 31, 37, 47, 51, 52}
NEW_CV_STORES = {9, 21, 23, 25, 31, 35, 37, 47, 49}
NEW_RATIO_UPLIFT_STORES = {3, 4, 8, 11, 12, 17, 18, 34, 51, 52}
# Compare the two models per store on Fold 3, with stores 4 and 34 assigned to ratio_uplift.
TWO_MODEL_RATIO_UPLIFT_STORES = {
    1, 3, 4, 8, 11, 12, 17, 18, 20, 22, 30, 32, 34, 36, 42, 43, 48, 50, 51, 52,
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def progress(env):
    step = env.iteration + 1
    if step % 500 == 0:
        print(
            f"  iteration {step}/{MAX_ITERATIONS}",
            flush=True,
        )


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def prepare_backtest(raw, add_lag_features):
    required = ["date", "sales"] + BASE23
    missing = set(required) - set(raw.columns)
    check(not missing, f"Missing train columns: {missing}")

    train = raw.loc[pd.to_datetime(raw["date"]).le(VALID_END), required].copy()
    train["date"] = pd.to_datetime(train["date"])
    train["sales"] = pd.to_numeric(
        train["sales"], errors="raise"
    )

    check(
        train["date"].min() == TRAIN_START
        and train["date"].max() == VALID_END,
        "Invalid backtest panel dates",
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

    panel_size = EXPECTED_STORES * EXPECTED_FAMILIES
    daily_counts = train.groupby("date").size().reindex(pd.date_range(TRAIN_START, VALID_END))
    check(daily_counts.eq(panel_size).all(), "Incomplete daily backtest panel")

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

    train_mask = train["date"].le(TRAIN_END)
    valid = train.loc[~train_mask].copy()
    train = train.loc[train_mask].copy()
    check(len(valid) == 16 * panel_size, "Unexpected validation row count")
    for lag in LAGS:
        check(VALID_END - pd.Timedelta(days=lag) <= TRAIN_END, f"Validation lag {lag} crosses cutoff")
        check(valid[f"sales_lag_{lag}"].notna().all(), f"Missing validation lag {lag}")
    return train, valid


def add_extra_features(train, test):
    """
    Compute all additional features causally.

    Training:
        feature(t) uses sales strictly before date t.

    Test:
        feature values are frozen at VALID_START.
        All 16 forecast days use the same historical cutoff.
    """

    dates = pd.date_range(TRAIN_START, VALID_START)

    # --------------------------------------------
    # Store-level daily sales totals
    # --------------------------------------------
    store_group = train.groupby(
        ["date", "store_nbr"],
        observed=True,
    )["sales"]

    counts = store_group.size()
    check(
        counts.eq(EXPECTED_FAMILIES).all(),
        "Incomplete store/day family panel",
    )

    store_totals = (
        store_group.sum()
        .unstack("store_nbr")
        .reindex(dates)
    )

    check(
        not store_totals.loc[
            TRAIN_START:TRAIN_END
        ].isna().any().any(),
        "Missing historical store/day totals",
    )

    store_history = store_totals.shift(1)

    # --------------------------------------------
    # Store mean ratio 7d / 28d
    # --------------------------------------------
    store_mean28 = store_history.rolling(
        28, min_periods=28
    ).mean()

    store_mean7 = store_history.rolling(
        7, min_periods=7
    ).mean()

    store_ratio = (
        store_mean7 /
        store_mean28.where(store_mean28.gt(0))
    ) - 1.0

    store_ratio = store_ratio.replace(
        [np.inf, -np.inf], np.nan
    )

    # --------------------------------------------
    # Family-level daily sales totals
    # --------------------------------------------
    family_group = train.groupby(
        ["date", "family"],
        observed=True,
    )["sales"]

    counts = family_group.size()
    check(
        counts.eq(EXPECTED_STORES).all(),
        "Incomplete family/day store panel",
    )

    family_totals = (
        family_group.sum()
        .unstack("family")
        .reindex(dates)
    )

    check(
        not family_totals.loc[
            TRAIN_START:TRAIN_END
        ].isna().any().any(),
        "Missing historical family/day totals",
    )

    # --------------------------------------------
    # Family 7d vs previous 21d uplift
    # --------------------------------------------
    recent7 = family_totals.shift(1).rolling(
        7, min_periods=7
    ).mean()

    previous21 = family_totals.shift(8).rolling(
        21, min_periods=21
    ).mean()

    family_uplift = (
        np.log1p(recent7)
        - np.log1p(previous21)
    )

    # --------------------------------------------
    # Map historical features to training rows
    # --------------------------------------------
    store_keys = pd.MultiIndex.from_frame(
        train[["date", "store_nbr"]]
    )

    family_keys = pd.MultiIndex.from_frame(
        train[["date", "family"]]
    )

    def assign_train_feature(frame, name, keys):
        long_values = frame.rename_axis(
            index="date"
        ).stack()

        train[name] = long_values.reindex(
            keys
        ).to_numpy(dtype=np.float64)

    assign_train_feature(
        store_ratio, RATIO_FEATURE, store_keys
    )
    assign_train_feature(
        family_uplift, UPLIFT_FEATURE, family_keys
    )

    # --------------------------------------------
    # Freeze features at first test day
    # --------------------------------------------
    test[RATIO_FEATURE] = (
        test["store_nbr"]
        .map(store_ratio.loc[VALID_START])
        .to_numpy(dtype=np.float64)
    )

    test[UPLIFT_FEATURE] = (
        test["family"]
        .map(family_uplift.loc[VALID_START])
        .to_numpy(dtype=np.float64)
    )

    for name in (RATIO_FEATURE, UPLIFT_FEATURE):
        nonmissing = int(test[name].notna().sum())
        print(
            f"[FEATURE] {name}: "
            f"{nonmissing:,}/{len(test):,} "
            "validation values available",
            flush=True,
        )

    del (
        store_keys, family_keys,
        store_totals, family_totals,
        store_history, store_mean7,
        store_mean28,
        recent7, previous21,
        store_ratio, family_uplift,
    )

    gc.collect()
    return train, test


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


def train_and_predict(train, valid):
    features = RATIO_UPLIFT_FEATURES
    xt, xp = make_matrices(train, valid, features)

    print(
        "\n[MODEL] ratio_uplift"
        f"\n  Features: {len(features)}"
        f"\n  Train rows: {len(xt):,}"
        f"\n  Validation rows: {len(xp):,}"
        f"\n  Max iterations: {MAX_ITERATIONS:,}",
        flush=True,
    )

    model = lgb.LGBMRegressor(**PARAMS)

    model.fit(
        xt,
        train[TARGET],
        eval_set=[(xp, valid[TARGET])],
        eval_names=["valid"],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True), progress],
    )

    best_iteration = int(model.best_iteration_)
    check(
        1 <= best_iteration <= MAX_ITERATIONS,
        f"Invalid best iteration: {best_iteration}",
    )

    prediction_log = np.asarray(
        model.predict(
            xp,
            num_iteration=best_iteration,
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

    return prediction_sales, best_iteration


def rmsle(actual, predicted):
    error = np.log1p(predicted) - np.log1p(actual)
    return float(np.sqrt(np.mean(error ** 2)))


def load_reference_predictions(valid, reference_dir):
    reference_columns = {
        "baseline_pred_sales": "trial17",
        "experiment_pred_sales": "cv",
    }
    reference = pd.read_csv(
        reference_dir / "fold1_paired_predictions.csv",
        usecols=ROW_KEYS + ["sales", "onpromotion", *reference_columns],
        parse_dates=["date"],
    ).rename(columns=reference_columns)
    predictions = valid[ROW_KEYS + ["sales", "onpromotion"]].merge(
        reference,
        on=ROW_KEYS,
        how="outer",
        suffixes=("", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    check(predictions["_merge"].eq("both").all(), "Fold 1 reference keys do not match")
    for column in ("sales", "onpromotion"):
        saved_column = f"{column}_reference"
        check(
            np.allclose(predictions[column], predictions[saved_column], rtol=1e-12, atol=1e-12),
            f"Fold 1 reference {column} does not match",
        )
        predictions = predictions.drop(columns=saved_column)
    # Preserve the validation row order used by LightGBM predictions.
    predictions = predictions.drop(columns="_merge").set_index(ROW_KEYS)
    predictions = predictions.reindex(pd.MultiIndex.from_frame(valid[ROW_KEYS])).reset_index()

    summary = pd.read_csv(reference_dir / "overall_comparison.csv")
    fold1 = summary.loc[summary["fold"].eq("fold1")]
    check(len(fold1) == 1, "Expected one Fold 1 reference summary row")
    fold1 = fold1.iloc[0]
    best_iterations = {}
    for name, prefix in (("trial17", "baseline"), ("cv", "experiment")):
        predicted = predictions[name]
        check(np.isfinite(predicted).all() and predicted.ge(0).all(), f"Invalid saved {name} predictions")
        score = rmsle(predictions["sales"], predicted)
        check(np.isclose(score, fold1[f"{prefix}_rmsle"], rtol=0, atol=1e-7), f"Saved {name} RMSLE does not match")
        best_iterations[name] = int(fold1[f"{prefix}_best_iteration"])
        print(f"[REUSED] {name} | RMSLE {score:.6f} | best iteration {best_iterations[name]:,}", flush=True)
    return predictions, best_iterations


def apply_store_assignments(predictions):
    stores = predictions["store_nbr"]
    check(NEW_CV_STORES.isdisjoint(NEW_RATIO_UPLIFT_STORES), "Overlapping three-model store assignments")
    assigned_stores = OLD_CV_STORES | NEW_CV_STORES | NEW_RATIO_UPLIFT_STORES | TWO_MODEL_RATIO_UPLIFT_STORES
    check(assigned_stores.issubset(set(stores)), "Invalid store assignment")
    predictions["old_cv_11"] = predictions["cv"].where(
        stores.isin(OLD_CV_STORES), predictions["trial17"]
    )
    predictions["new_three_models"] = predictions["trial17"]
    predictions.loc[stores.isin(NEW_CV_STORES), "new_three_models"] = predictions["cv"]
    predictions.loc[stores.isin(NEW_RATIO_UPLIFT_STORES), "new_three_models"] = predictions["ratio_uplift"]
    predictions["new_two_models"] = predictions["ratio_uplift"].where(
        stores.isin(TWO_MODEL_RATIO_UPLIFT_STORES), predictions["cv"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    reference_dir = root / FOLD1_REFERENCE
    output = (
        args.output_dir.resolve() if args.output_dir is not None
        else root / "src" / "trial17_fold1_model_routing_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    sys.path.insert(0, str(root / "src"))
    from main import main as original_pipeline
    from feature_engineering import add_lag_features

    os.chdir(root)
    print(f"[FOLD 1] Train through {TRAIN_END.date()} | Valid {VALID_START.date()} to {VALID_END.date()}", flush=True)
    print(f"[ITERATIONS] Max {MAX_ITERATIONS:,} | early stopping {EARLY_STOPPING_ROUNDS} on validation log RMSE", flush=True)
    print("[OUTPUT]", output, flush=True)

    raw, test, holidays = original_pipeline()
    train, valid = prepare_backtest(raw, add_lag_features)
    del raw, test, holidays
    gc.collect()
    train, valid = add_extra_features(train, valid)
    print(f"[CHECK] {len(train):,} train rows | {len(valid):,} validation rows", flush=True)
    predictions, best_iterations = load_reference_predictions(valid, reference_dir)
    if args.check_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    config = {
        "train_end": str(TRAIN_END.date()),
        "valid_start": str(VALID_START.date()),
        "valid_end": str(VALID_END.date()),
        "params": PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_metric": "validation log RMSE",
        "trained_model_features": RATIO_UPLIFT_FEATURES,
        "reference_dir": str(reference_dir),
        "old_cv_stores": sorted(OLD_CV_STORES),
        "new_cv_stores": sorted(NEW_CV_STORES),
        "new_ratio_uplift_stores": sorted(NEW_RATIO_UPLIFT_STORES),
        "two_model_ratio_uplift_stores": sorted(TWO_MODEL_RATIO_UPLIFT_STORES),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    actual = predictions["sales"].to_numpy()
    predictions["ratio_uplift"], best_iterations["ratio_uplift"] = train_and_predict(train, valid)
    apply_store_assignments(predictions)

    score_columns = ["trial17", "cv", "ratio_uplift", "old_cv_11", "new_three_models", "new_two_models"]
    summary = pd.DataFrame({
        "model": score_columns,
        "rmsle": [rmsle(actual, predictions[name].to_numpy()) for name in score_columns],
        "best_iteration": pd.array([best_iterations.get(name) for name in score_columns], dtype="Int64"),
    })
    baseline = summary.loc[summary["model"].eq("old_cv_11"), "rmsle"].iloc[0]
    summary["delta_vs_old_cv_11"] = summary["rmsle"] - baseline
    three_model_score = summary.loc[summary["model"].eq("new_three_models"), "rmsle"].iloc[0]
    summary["delta_vs_new_three_models"] = summary["rmsle"] - three_model_score
    save_csv(predictions, output / "predictions.csv")
    save_csv(summary, output / "overall_comparison.csv")
    for key in ("store_nbr", "family"):
        group_scores = [
            {key: value, **{name: rmsle(rows["sales"], rows[name]) for name in score_columns}}
            for value, rows in predictions.groupby(key, observed=True)
        ]
        save_csv(pd.DataFrame(group_scores), output / f"{key}_comparison.csv")
    print(summary.round(6).to_string(index=False), flush=True)
    print("[COMPLETE]", output, flush=True)


if __name__ == "__main__":
    main()
