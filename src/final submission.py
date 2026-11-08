
"""
Store Sales - Trial 17 Baseline + optimized Oil Raw submission.

Baseline:
    Original Trial 17 (Base23 + Lag364 + Lag28)
    Original parameters, 6770 fixed rounds.

Oil Raw:
    2026-11-07 optimized 30-trial experiment
    Base23 + Lag364 + Lag28 + oil_raw
    Saved best parameters and final_rounds.

Both models:
    - Train on ALL 33 families (3,008,016 rows).
    - Predict ALL 28,512 test rows.
    - No Optuna, no early stopping, no new tuning.

Routing:
    Oil Raw: 9 families
    Baseline: remaining 24 families

Outputs:
    Kaggle submission: id,sales
    Model prediction audit CSV
    Model configuration JSON
    Trained LightGBM model files
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

BASE_RUN = (
    ROOT
    / "results"
    / "base23_lag364_lag28_optuna"
    / "20261028_182041_683679"
)

OIL_RUN = Path(
    r"C:\Users\gey82\OneDrive\문서\ChatGPT\판매량 시계열 캐글"
    r"\trial17_oil_raw_results\20261107_031844_450587"
)

SUBMISSIONS_DIR = ROOT / "submissions"

sys.path.insert(0, str(SRC))

from main import main as original_pipeline
from feature_engineering import add_lag_features


# ============================================================
# 2. FEATURES
# ============================================================

BASE23 = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion", "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day",
    "event_city_foundation", "event_good_friday",
    "event_labor_day", "event_other",
]

LAGS = (364, 28)

BASE_FEATURES = BASE23 + [
    "sales_lag_364",
    "sales_lag_28",
]

OIL_FEATURES = BASE_FEATURES + ["oil_raw"]

CATEGORICAL = [
    "store_nbr", "family", "city",
    "state", "type", "cluster",
]

KEYS = ["store_nbr", "family"]
ROW_KEYS = ["date"] + KEYS

TARGET = "sales_log"

TRAIN_START = pd.Timestamp("2013-01-01")
TRAIN_END = pd.Timestamp("2017-08-15")

TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

EXPECTED_TRAIN = 3_008_016
EXPECTED_TEST = 28_512

EXPECTED_FAMILIES = 33
EXPECTED_STORES = 54
FAMILY_TEST_ROWS = 864


# ============================================================
# 3. FINAL ROUTING
# ============================================================

OIL_RAW_FAMILIES = {
    "BABY CARE",
    "BEAUTY",
    "BEVERAGES",
    "BOOKS",
    "GROCERY I",
    "HARDWARE",
    "HOME APPLIANCES",
    "LIQUOR,WINE,BEER",
    "SCHOOL AND OFFICE SUPPLIES",
}

# Remaining 24 families use Trial 17 Baseline.


# ============================================================
# 4. CONFIGURATION / PARAMETERS
# ============================================================

def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_baseline_config():
    """
    Original Trial 17 parameters and fixed training rounds.
    """

    cfg = load_json(BASE_RUN / "best_params.json")

    if cfg.get("best_trial_number") != 17:
        raise ValueError("Wrong Baseline trial number")

    if cfg.get("features") != BASE_FEATURES:
        raise ValueError("Baseline feature mismatch")

    params = dict(cfg["full_cv_params"])

    rounds = int(cfg["backtest_n_estimators"])

    if rounds <= 0:
        raise ValueError("Invalid Baseline rounds")

    params["n_estimators"] = rounds

    return params, rounds, cfg


def load_oil_config():
    """
    Reuse the optimized Oil Raw parameters from the
    completed 2026-11-07 30-trial experiment.

    Never runs Optuna.
    """

    best = load_json(OIL_RUN / "best_params.json")

    if best.get("features") != OIL_FEATURES:
        raise ValueError("Oil Raw feature mismatch")

    params = dict(best["params"])
    rounds = int(best["final_rounds"])

    if rounds <= 0:
        raise ValueError("Invalid Oil Raw rounds")

    params["n_estimators"] = rounds

    return params, rounds, best


def check_configuration():
    assert len(BASE23) == 23
    assert len(BASE_FEATURES) == 25
    assert len(OIL_FEATURES) == 26

    assert len(set(BASE_FEATURES)) == 25
    assert len(set(OIL_FEATURES)) == 26

    assert len(OIL_RAW_FAMILIES) == 9

    for lag in LAGS:
        if TEST_END - pd.Timedelta(days=lag) > TRAIN_END:
            raise ValueError(f"Future target leakage: lag {lag}")


# ============================================================
# 5. PREPARE FULL TRAIN
# ============================================================

def prepare_train(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Retain original main() preprocessing and
    add_lag_features() behavior.
    """

    required = ["date", "sales"] + BASE23

    missing = set(required) - set(raw.columns)

    if missing:
        raise ValueError(f"Missing train columns: {missing}")

    df = raw.loc[:, required].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="raise")

    if len(df) != EXPECTED_TRAIN:
        raise ValueError("Unexpected train row count")

    if (
        df.date.min() != TRAIN_START
        or df.date.max() != TRAIN_END
    ):
        raise ValueError("Unexpected training period")

    if df.duplicated(ROW_KEYS).any():
        raise ValueError("Duplicate train panel keys")

    if df[BASE23].isna().any().any():
        raise ValueError("Missing Base23 features")

    if (
        not np.isfinite(df.sales.to_numpy(dtype=float)).all()
        or (df.sales < 0).any()
    ):
        raise ValueError("Invalid train sales")

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)

    counts = df.groupby(KEYS, observed=True).size()

    if len(counts) != 54 * 33 or counts.nunique() != 1:
        raise ValueError("Incomplete training panel")

    # Original lag feature engineering.
    df = add_lag_features(df, lags=LAGS)

    positions = df.groupby(KEYS, observed=True).cumcount()

    for lag in LAGS:
        col = f"sales_lag_{lag}"

        if col not in df.columns:
            raise ValueError(f"Missing {col}")

        expected_na = positions.lt(lag)

        if not df[col].isna().equals(expected_na):
            raise ValueError(f"Unexpected NaN pattern: {col}")

    df[TARGET] = np.log1p(df["sales"])

    print(
        f"[TRAIN] {len(df):,} rows | "
        f"{df.date.min().date()} ~ {df.date.max().date()}"
    )

    return df


# ============================================================
# 6. PREPARE TEST / HISTORICAL LAGS
# ============================================================

def prepare_test(
    original_test: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:

    required = ["id", "date"] + BASE23

    missing = set(required) - set(original_test.columns)

    if missing:
        raise ValueError(f"Missing test features: {missing}")

    test = original_test.loc[:, required].copy()

    test["date"] = pd.to_datetime(test["date"])
    test["id"] = pd.to_numeric(
        test["id"], errors="raise"
    ).astype("int64")

    if len(test) != EXPECTED_TEST:
        raise ValueError("Unexpected test row count")

    if (
        test.date.min() != TEST_START
        or test.date.max() != TEST_END
    ):
        raise ValueError("Unexpected test dates")

    if (
        test.id.duplicated().any()
        or test.duplicated(ROW_KEYS).any()
    ):
        raise ValueError("Duplicate test IDs / keys")

    if test[BASE23].isna().any().any():
        raise ValueError("Missing test Base23 feature")

    family_counts = test.groupby(
        "family", observed=True
    ).size()

    if (
        len(family_counts) != EXPECTED_FAMILIES
        or not family_counts.eq(FAMILY_TEST_ROWS).all()
    ):
        raise ValueError("Unexpected family distribution")

    if test.store_nbr.nunique() != EXPECTED_STORES:
        raise ValueError("Unexpected number of stores")

    if not OIL_RAW_FAMILIES.issubset(set(test.family)):
        raise ValueError("Missing Oil Raw routing family")

    original_ids = test["id"].copy()

    # Historical sales-based test lags.
    # No recursive prediction is necessary.
    for lag in LAGS:
        col = f"sales_lag_{lag}"

        history = train[
            ROW_KEYS + ["sales"]
        ].copy()

        history["date"] += pd.Timedelta(days=lag)

        history = history.rename(
            columns={"sales": col}
        )

        test = test.merge(
            history,
            on=ROW_KEYS,
            how="left",
            sort=False,
            validate="one_to_one",
        )

        if test[col].isna().any():
            raise ValueError(f"Missing historical {col}")

        if not np.isfinite(
            test[col].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Invalid historical {col}")

    if not test["id"].equals(original_ids):
        raise ValueError("Test row order changed")

    print(
        f"[TEST] {len(test):,} rows | "
        "Lag364 / Lag28 OK"
    )

    return test


# ============================================================
# 7. OIL RAW PREPROCESSING
# ============================================================

def extract_train_oil(raw: pd.DataFrame) -> pd.Series:
    """
    Exactly reuse main()'s preprocessed historical oil.

    Train oil must not be recomputed from raw oil.csv.
    """

    if "dcoilwtico" not in raw.columns:
        raise ValueError(
            "main() must provide preprocessed dcoilwtico"
        )

    oil = raw[["date", "dcoilwtico"]].copy()

    oil["date"] = pd.to_datetime(oil["date"])
    oil["dcoilwtico"] = pd.to_numeric(
        oil["dcoilwtico"], errors="raise"
    )

    if oil["dcoilwtico"].isna().any():
        raise ValueError("Missing train oil")

    if not np.isfinite(
        oil.dcoilwtico.to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid train oil")

    if (
        oil.groupby("date")["dcoilwtico"]
        .nunique()
        .ne(1)
        .any()
    ):
        raise ValueError("Inconsistent same-day train oil")

    daily = (
        oil.drop_duplicates("date")
        .set_index("date")["dcoilwtico"]
        .sort_index()
    )

    if not daily.index.equals(
        pd.date_range(TRAIN_START, TRAIN_END)
    ):
        raise ValueError("Incomplete historical oil dates")

    return daily


def extend_test_oil(
    train_oil: pd.Series,
    original_test: pd.DataFrame,
) -> pd.Series:
    """
    Same competition-test oil policy as final submission:

    1. Prefer fully preprocessed test oil from main().
    2. Otherwise extend historical oil with oil.csv.
    3. Time-interpolate test dates.
    4. Preserve historical train values exactly.

    This is retrospective competition preprocessing,
    not a real-time future oil forecast.
    """

    future_dates = pd.date_range(TEST_START, TEST_END)

    future = None

    if "dcoilwtico" in original_test.columns:
        temp = original_test[
            ["date", "dcoilwtico"]
        ].copy()

        temp["date"] = pd.to_datetime(temp["date"])
        temp["dcoilwtico"] = pd.to_numeric(
            temp["dcoilwtico"], errors="coerce"
        )

        if (
            temp.groupby("date")["dcoilwtico"]
            .nunique(dropna=True)
            .gt(1)
            .any()
        ):
            raise ValueError("Conflicting test oil values")

        candidate = (
            temp.drop_duplicates("date")
            .set_index("date")["dcoilwtico"]
            .reindex(future_dates)
        )

        if np.isfinite(
            candidate.to_numpy(dtype=float)
        ).all():
            future = candidate

            print("[OIL] Using preprocessed test oil")

    if future is None:
        oil_path = ROOT / "data" / "oil.csv"

        if not oil_path.is_file():
            raise FileNotFoundError(oil_path)

        obs = pd.read_csv(
            oil_path,
            usecols=["date", "dcoilwtico"],
        )

        obs["date"] = pd.to_datetime(obs["date"])
        obs["dcoilwtico"] = pd.to_numeric(
            obs["dcoilwtico"], errors="coerce"
        )

        obs = obs.loc[
            obs.date.between(TEST_START, TEST_END)
        ]

        if obs.dcoilwtico.notna().sum() == 0:
            raise ValueError("No future oil observations")

        if (
            obs.groupby("date")["dcoilwtico"]
            .nunique(dropna=True)
            .gt(1)
            .any()
        ):
            raise ValueError("Conflicting raw test oil")

        observations = (
            obs.drop_duplicates("date")
            .set_index("date")["dcoilwtico"]
            .reindex(future_dates)
        )

        combined = pd.concat(
            [train_oil, observations]
        ).sort_index()

        combined = combined.interpolate(
            method="time",
            limit_direction="both",
        )

        future = combined.reindex(future_dates)

        print("[OIL] oil.csv + time interpolation")

    if (
        future.isna().any()
        or not np.isfinite(
            future.to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("Invalid future oil")

    if (future < 0).any():
        raise ValueError("Negative future oil")

    complete = pd.concat(
        [train_oil, future]
    ).sort_index()

    if not complete.loc[
        train_oil.index
    ].equals(train_oil):
        raise ValueError("Historical oil was modified")

    if not complete.index.equals(
        pd.date_range(TRAIN_START, TEST_END)
    ):
        raise ValueError("Incomplete oil timeline")

    return complete


def attach_oil_raw(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oil: pd.Series,
):
    train["oil_raw"] = train.date.map(oil)
    test["oil_raw"] = test.date.map(oil)

    for name, frame in [
        ("TRAIN", train),
        ("TEST", test),
    ]:
        values = frame["oil_raw"].to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(f"Invalid {name} Oil Raw")

    print("[OIL] Same-day raw oil feature OK")


# ============================================================
# 8. MATRICES
# ============================================================

def make_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
):

    x_train = train.loc[:, features].copy()
    x_test = test.loc[:, features].copy()

    for col in CATEGORICAL:
        categories = pd.Index(
            train[col].dropna().unique()
        )

        x_train[col] = pd.Categorical(
            x_train[col],
            categories=categories,
        )

        x_test[col] = pd.Categorical(
            x_test[col],
            categories=categories,
        )

        if x_test[col].isna().any():
            raise ValueError(
                f"Unseen test category: {col}"
            )

    if (
        list(x_train.columns) != features
        or list(x_test.columns) != features
    ):
        raise ValueError("Feature order mismatch")

    return x_train, x_test


# ============================================================
# 9. MODEL TRAINING
# ============================================================

def progress(env):
    iteration = env.iteration + 1

    if iteration % 500 == 0:
        print(
            f"  iteration {iteration}/{env.end_iteration}",
            flush=True,
        )


def save_model(model, directory, name, features):
    """
    Unicode-safe LightGBM model serialization.
    Mirrors the resume script's save policy.
    """

    path = directory / f"{name}_model.txt"
    temp = path.with_suffix(".txt.tmp")

    model_text = model.booster_.model_to_string()

    temp.write_text(
        model_text,
        encoding="utf-8",
    )

    restored = lightgbm.Booster(
        model_str=temp.read_text(encoding="utf-8")
    )

    if (
        restored.num_trees()
        != model.booster_.num_trees()
    ):
        raise ValueError("Model save round-trip failed")

    temp.replace(path)

    importance = pd.DataFrame({
        "feature": features,
        "gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
    })

    importance.to_csv(
        directory / f"{name}_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )


def fit_predict(
    name,
    train,
    test,
    features,
    params,
    rounds,
    model_dir,
):
    x_train, x_test = make_matrices(
        train, test, features
    )

    model_params = dict(params)
    model_params["n_estimators"] = rounds

    print(
        f"\n[FIT] {name}\n"
        f"  Train rows : {len(x_train):,}\n"
        f"  Features   : {len(features)}\n"
        f"  Rounds     : {rounds}\n"
        f"  EarlyStop  : OFF\n"
        f"  Optuna     : OFF",
        flush=True,
    )

    model = LGBMRegressor(**model_params)

    model.fit(
        x_train,
        train[TARGET],
        categorical_feature=CATEGORICAL,
        callbacks=[progress],
    )

    pred_log = np.asarray(
        model.predict(
            x_test,
            num_iteration=rounds,
        ),
        dtype=float,
    )

    if not np.isfinite(pred_log).all():
        raise ValueError(f"{name}: invalid prediction")

    # Same log-target inverse transform.
    predictions = np.expm1(
        np.maximum(pred_log, 0)
    )

    if (
        not np.isfinite(predictions).all()
        or (predictions < 0).any()
    ):
        raise ValueError(f"{name}: invalid sales output")

    save_model(
        model,
        model_dir,
        name,
        features,
    )

    print(
        f"[PREDICT] {name}: "
        f"{len(predictions):,} test predictions",
        flush=True,
    )

    del model, x_train, x_test, pred_log
    gc.collect()

    return predictions


# ============================================================
# 10. MODEL ROUTING
# ============================================================

def route_predictions(
    test,
    baseline_predictions,
    oil_predictions,
):

    result = test[
        ["id", "date", "store_nbr", "family"]
    ].copy()

    result["baseline_pred_sales"] = baseline_predictions
    result["oil_raw_pred_sales"] = oil_predictions

    is_oil = result.family.isin(
        OIL_RAW_FAMILIES
    )

    result["selected_model"] = np.where(
        is_oil,
        "oil_raw",
        "baseline",
    )

    result["sales"] = np.where(
        is_oil,
        result["oil_raw_pred_sales"],
        result["baseline_pred_sales"],
    )

    counts = result.groupby(
        "selected_model"
    ).size().to_dict()

    expected = {
        "baseline": 24 * FAMILY_TEST_ROWS,
        "oil_raw": 9 * FAMILY_TEST_ROWS,
    }

    if counts != expected:
        raise ValueError(
            f"Wrong routing distribution: {counts}"
        )

    if result.id.duplicated().any():
        raise ValueError("Duplicate routed IDs")

    if not np.isfinite(
        result.sales.to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid routed sales")

    for family, group in result.groupby("family"):
        expected_model = (
            "oil_raw"
            if family in OIL_RAW_FAMILIES
            else "baseline"
        )

        if not group.selected_model.eq(
            expected_model
        ).all():
            raise ValueError(
                f"Incorrect route: {family}"
            )

    print("\n[ROUTING]")
    print("  Baseline : 24 families / 20,736 rows")
    print("  Oil Raw  :  9 families /  7,776 rows")

    return result


# ============================================================
# 11. SUBMISSION / AUDIT OUTPUTS
# ============================================================

def write_outputs(
    result,
    baseline_params,
    baseline_rounds,
    oil_params,
    oil_rounds,
    oil_config,
    model_dir,
    stamp,
):

    SUBMISSIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission_path = (
        SUBMISSIONS_DIR
        / f"submission_trial17_oilraw9_{stamp}.csv"
    )

    audit_path = (
        SUBMISSIONS_DIR
        / f"model_predictions_trial17_oilraw9_{stamp}.csv"
    )

    metadata_path = (
        SUBMISSIONS_DIR
        / f"model_config_trial17_oilraw9_{stamp}.json"
    )

    sample = pd.read_csv(
        ROOT / "data" / "sample_submission.csv"
    )

    if (
        len(sample) != EXPECTED_TEST
        or not sample.id.is_unique
        or set(sample.id) != set(result.id)
    ):
        raise ValueError(
            "Sample submission IDs mismatch"
        )

    # Preserve sample_submission ID order exactly.
    aligned = sample[["id"]].merge(
        result,
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    if (
        not aligned.id.equals(sample.id)
        or aligned.sales.isna().any()
    ):
        raise ValueError(
            "Submission ID alignment failed"
        )

    rows = [
        {
            "id": int(i),
            "sales": float(sales),
        }
        for i, sales in zip(
            aligned["id"],
            aligned["sales"],
        )
    ]

    # Kaggle-only CSV: id,sales.
    with submission_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["id", "sales"],
        )

        writer.writeheader()
        writer.writerows(rows)

    # Full prediction trace.
    aligned.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "submission": str(submission_path),
        "baseline": {
            "source": str(BASE_RUN),
            "trial": 17,
            "features": BASE_FEATURES,
            "params": baseline_params,
            "rounds": baseline_rounds,
        },
        "oil_raw": {
            "source": str(OIL_RUN),
            "best_trial_number": oil_config.get(
                "best_trial_number"
            ),
            "features": OIL_FEATURES,
            "params": oil_params,
            "rounds": oil_rounds,
            "fold23_rmsle": oil_config.get(
                "fold23_rmsle"
            ),
        },
        "training": {
            "train_rows": EXPECTED_TRAIN,
            "models": 2,
            "each_model_all_families": True,
            "early_stopping": False,
            "optuna": False,
        },
        "routing": {
            "oil_raw": sorted(OIL_RAW_FAMILIES),
            "baseline": "remaining 24 families",
        },
        "oil_test_policy": (
            "preprocessed main() test oil if complete; "
            "otherwise oil.csv with time interpolation"
        ),
        "models_directory": str(model_dir),
    }

    with metadata_path.open(
        "x",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # Reload and verify actual Kaggle file.
    with submission_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        saved = list(csv.DictReader(f))

    if len(saved) != EXPECTED_TEST:
        raise ValueError(
            "Saved submission row count mismatch"
        )

    for saved_row, expected in zip(saved, rows):
        if (
            int(saved_row["id"]) != expected["id"]
            or float(saved_row["sales"])
            != expected["sales"]
        ):
            raise ValueError(
                "Submission round-trip mismatch"
            )

    print("\n[SUBMISSION QC]")
    print("  Rows       : 28,512")
    print("  Columns    : id, sales")
    print("  Unique IDs : OK")
    print("  Finite     : OK")
    print("  Nonnegative: OK")
    print("  Alignment  : OK")
    print("  Reload     : OK")

    print("\n[OUTPUT FILES]")
    print("Submission:", submission_path)
    print("Audit     :", audit_path)
    print("Metadata  :", metadata_path)
    print("Models    :", model_dir)

    return submission_path


# ============================================================
# 12. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate pipeline without training",
    )

    args = parser.parse_args()

    check_configuration()

    baseline_params, baseline_rounds, _ = (
        load_baseline_config()
    )

    oil_params, oil_rounds, oil_cfg = (
        load_oil_config()
    )

    print("\n===== SAVED MODEL CONFIG =====")
    print(
        f"Baseline Trial17: "
        f"{baseline_rounds} rounds / 25 features"
    )

    print(
        f"Optimized Oil Raw: "
        f"{oil_rounds} rounds / 26 features"
    )

    print("Routing: Oil Raw 9 + Baseline 24")

    os.chdir(ROOT)

    # Original preprocessing pipeline.
    raw, original_test, holidays = (
        original_pipeline()
    )

    if not isinstance(raw, pd.DataFrame):
        raise TypeError("main() train must be DataFrame")

    if not isinstance(original_test, pd.DataFrame):
        raise TypeError("main() test must be DataFrame")

    # Preprocessed oil from the original pipeline.
    historical_oil = extract_train_oil(raw)

    full_oil = extend_test_oil(
        historical_oil,
        original_test,
    )

    # Original Base23 + lag pipeline.
    train = prepare_train(raw)

    test = prepare_test(
        original_test,
        train,
    )

    attach_oil_raw(
        train,
        test,
        full_oil,
    )

    del raw, original_test, holidays
    del historical_oil, full_oil
    gc.collect()

    if args.check_only:

        for name, features in {
            "baseline": BASE_FEATURES,
            "oil_raw": OIL_FEATURES,
        }.items():

            xt, xp = make_matrices(
                train,
                test,
                features,
            )

            print(
                f"[CHECK] {name}: "
                f"train={xt.shape}, "
                f"test={xp.shape}"
            )

            del xt, xp
            gc.collect()

        print("\n[CHECK OK] No training executed.")
        return

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    model_dir = (
        SUBMISSIONS_DIR
        / f"models_trial17_oilraw9_{stamp}"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Model A: Original Trial 17 Baseline
    # Full 33-family training
    # --------------------------------------------------------

    baseline_predictions = fit_predict(
        name="baseline",
        train=train,
        test=test,
        features=BASE_FEATURES,
        params=baseline_params,
        rounds=baseline_rounds,
        model_dir=model_dir,
    )

    # --------------------------------------------------------
    # Model B: Optimized 2026-11-07 Oil Raw
    # Full 33-family training
    # --------------------------------------------------------

    oil_predictions = fit_predict(
        name="oil_raw",
        train=train,
        test=test,
        features=OIL_FEATURES,
        params=oil_params,
        rounds=oil_rounds,
        model_dir=model_dir,
    )

    # --------------------------------------------------------
    # Final prediction routing
    # --------------------------------------------------------

    result = route_predictions(
        test,
        baseline_predictions,
        oil_predictions,
    )

    del train, test
    del baseline_predictions, oil_predictions
    gc.collect()

    # --------------------------------------------------------
    # Kaggle submission + full audit
    # --------------------------------------------------------

    write_outputs(
        result=result,
        baseline_params=baseline_params,
        baseline_rounds=baseline_rounds,
        oil_params=oil_params,
        oil_rounds=oil_rounds,
        oil_config=oil_cfg,
        model_dir=model_dir,
        stamp=stamp,
    )

    print("\n[COMPLETE] Final submission created.")


if __name__ == "__main__":
    main()
