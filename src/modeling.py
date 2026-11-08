
"""
Oil Raw Optuna Best Parameters -> Oil EMA7 experiment

- Parameters: Oil Raw 30-Trial Optuna best_params.json
- Features: Base23 + Lag364 + Lag28 + Oil EMA7
- Oil Raw excluded from model inputs
- Original main.py preprocessing
- Original feature_engineering.py lag generation
- Fold2, Fold3, 2016 Backtest
- All 33 families
- No Optuna / Early Stopping for each split / No submission
"""

from pathlib import Path
import argparse
import gc
import json
import os
import hashlib

import numpy as np
import pandas as pd
import lightgbm as lgb

from main import main as original_pipeline
from feature_engineering import add_lag_features


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OIL_OPTUNA_RUN = (
    ROOT
    / "src"
    / "trial17_oil_raw_results"
    / "20261107_031844_450587"
)

OUTPUT = OIL_OPTUNA_RUN / "ema7_only_cv_results"

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
BASE_FEATURES = BASE23 + ["sales_lag_364", "sales_lag_28"]
FEATURES = BASE_FEATURES + ["oil_ema_7"]

CATEGORICAL = [
    "store_nbr", "family", "city", "state", "type", "cluster"
]

KEYS = ["store_nbr", "family"]
ROW_KEYS = ["date"] + KEYS
TARGET = "sales_log"

EXPECTED_ROWS = 3_008_016
EXPECTED_VALID_ROWS = 28_512

# Upper limit only: actual rounds are selected by validation Early Stopping.
MAX_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 150

SPECS = {
    "fold2": {
        "train_end": "2017-07-14",
        "valid_start": "2017-07-15",
        "valid_end": "2017-07-30",
    },
    "fold3": {
        "train_end": "2017-07-30",
        "valid_start": "2017-07-31",
        "valid_end": "2017-08-15",
    },
    "backtest": {
        "train_end": "2016-08-15",
        "valid_start": "2016-08-16",
        "valid_end": "2016-08-31",
    },
}


# ============================================================
# File helpers
# ============================================================

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Load OIL RAW Optuna parameters
# ============================================================

def load_oil_optuna(run_dir):

    path = run_dir / "best_params.json"

    if not path.is_file():
        raise FileNotFoundError(
            "Oil Raw Optuna best_params.json not found:\n"
            f"{path}\n"
            "Specify the actual Oil Raw Optuna result directory "
            "using --run-dir. Baseline parameters will NOT be used."
        )

    best = json.loads(path.read_text(encoding="utf-8-sig"))

    # Verify the tuned feature configuration.
    expected_tuned_features = BASE_FEATURES + ["oil_raw"]

    if best.get("features") != expected_tuned_features:
        raise ValueError(
            "Saved Optuna features differ from expected "
            "Base25 + oil_raw configuration."
        )

    if not isinstance(best.get("params"), dict):
        raise ValueError("Missing Oil Raw Optuna parameters.")

    params = dict(best["params"])
    # Do not reuse any stored iteration from the Optuna run.
    params.pop("n_estimators", None)

    if len(BASE23) != 23 or len(FEATURES) != 26:
        raise ValueError("Unexpected feature count.")

    print("\n========== PARAMETER SOURCE ==========")
    print("Source: OIL RAW OPTUNA 30-TRIAL")
    print("Best trial:", best.get("best_trial_number"))
    print("Optuna Fold23 RMSLE:", best.get("fold23_rmsle"))
    print("Iteration selection: per-split Early Stopping")
    print("Maximum estimators:", MAX_ESTIMATORS)
    print("Early stopping patience:", EARLY_STOPPING_ROUNDS)
    print("Input features:", len(FEATURES))
    print("Oil Raw: EXCLUDED")
    print("Oil EMA7: INCLUDED")
    print("Optuna rerun: NO")
    print("======================================\n")

    return params, best, path


# ============================================================
# Original feature preparation
# ============================================================

def prepare_features(raw):

    required = ["date", "sales"] + BASE23

    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    df = raw.loc[:, required].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"])

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected restored training rows: {len(df)}"
        )

    if (
        df["date"].min() != pd.Timestamp("2013-01-01")
        or df["date"].max() != pd.Timestamp("2017-08-15")
    ):
        raise ValueError("Unexpected train date range.")

    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store/family/date rows.")

    if df[BASE23].isna().any().any():
        raise ValueError("Missing Base23 feature values.")

    if df["family"].nunique() != 33:
        raise ValueError("Expected all 33 families.")

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)

    # Original lag feature engineering.
    df = add_lag_features(df, lags=LAGS)

    for lag in LAGS:
        col = f"sales_lag_{lag}"
        if col not in df.columns:
            raise ValueError(f"Missing original lag: {col}")

    df[TARGET] = np.log1p(df["sales"])

    return df


# ============================================================
# Oil EMA7: existing preprocessed dcoilwtico only
# ============================================================

def add_oil_ema7(df, raw):

    if "dcoilwtico" not in raw.columns:
        raise ValueError(
            "main.py must return preprocessed dcoilwtico."
        )

    daily = raw[["date", "dcoilwtico"]].copy()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["dcoilwtico"] = pd.to_numeric(
        daily["dcoilwtico"]
    )

    if daily["dcoilwtico"].isna().any():
        raise ValueError("Preprocessed oil contains NaN.")

    if not np.isfinite(daily["dcoilwtico"]).all():
        raise ValueError("Nonfinite preprocessed oil.")

    if daily.groupby("date")["dcoilwtico"].nunique().ne(1).any():
        raise ValueError("Different oil values on the same date.")

    daily = (
        daily.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    expected_dates = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D",
    )

    if not pd.DatetimeIndex(daily["date"]).equals(expected_dates):
        raise ValueError("Missing daily oil dates.")

    # The ONLY new model feature.
    daily["oil_ema_7"] = (
        daily["dcoilwtico"]
        .ewm(span=7, adjust=False, min_periods=7)
        .mean()
    )

    oil_map = daily.set_index("date")["oil_ema_7"]

    df["oil_ema_7"] = df["date"].map(oil_map)

    # The first six days are intentionally NaN.
    bad = (
        df["date"].gt(pd.Timestamp("2013-01-06"))
        & df["oil_ema_7"].isna()
    )

    if bad.any():
        raise ValueError("Unexpected EMA7 missing values.")

    print("[OIL] Source: preprocessed dcoilwtico")
    print("[OIL] EMA span=7, adjust=False, min_periods=7")
    print("[OIL] Oil Raw NOT included in model features")

    return df


# ============================================================
# Original split structure
# ============================================================

def make_split(df, spec):

    cutoff = pd.Timestamp(spec["train_end"])
    start = pd.Timestamp(spec["valid_start"])
    end = pd.Timestamp(spec["valid_end"])

    if start != cutoff + pd.Timedelta(days=1):
        raise ValueError("Invalid split boundary.")

    if (end - start).days + 1 != 16:
        raise ValueError("Expected 16-day validation.")

    train = df.loc[df["date"].le(cutoff)]
    valid = df.loc[df["date"].between(start, end)]

    if len(valid) != EXPECTED_VALID_ROWS:
        raise ValueError("Invalid validation row count.")

    if valid["family"].nunique() != 33:
        raise ValueError("Validation is missing families.")

    for lag in LAGS:
        if end - pd.Timedelta(days=lag) > cutoff:
            raise ValueError(f"Target leakage in Lag{lag}.")

        if valid[f"sales_lag_{lag}"].isna().any():
            raise ValueError(f"Missing validation Lag{lag}.")

    return train, valid


def make_matrices(train, valid):

    xt = train[FEATURES].copy()
    xv = valid[FEATURES].copy()

    for col in CATEGORICAL:
        categories = pd.Index(train[col].dropna().unique())

        xt[col] = pd.Categorical(
            xt[col], categories=categories
        )

        xv[col] = pd.Categorical(
            xv[col], categories=categories
        )

        if xv[col].isna().any():
            raise ValueError(f"Unknown category in {col}.")

    if "oil_raw" in xt.columns or "dcoilwtico" in xt.columns:
        raise ValueError("Oil Raw entered training matrix.")

    return xt, xv


# ============================================================
# RMSLE and per-family report
# ============================================================

def rmsle(actual, predicted):

    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)

    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite scores/predictions.")

    if (y < 0).any():
        raise ValueError("Negative actual sales.")

    p = np.maximum(p, 0)

    return float(
        np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2))
    )


def score_predictions(pred, split):

    rows = []

    groups = [("ALL", pred)]
    groups += list(
        pred.groupby("family", observed=True, sort=True)
    )

    for family, group in groups:
        rows.append({
            "split": split,
            "family": family,
            "n_rows": len(group),
            "rmsle": rmsle(
                group["sales"],
                group["pred_sales"],
            ),
        })

    result = pd.DataFrame(rows)

    if len(result) != 34:
        raise ValueError("Expected ALL + 33 family scores.")

    return result


# ============================================================
# Unicode-safe model save
# ============================================================

def save_model(model, path):

    temp = path.with_suffix(".txt.tmp")

    # Store the best iteration, not the extra trees trained during patience.
    text = model.booster_.model_to_string(num_iteration=model.best_iteration_)
    temp.write_text(text, encoding="utf-8")

    restored = lgb.Booster(
        model_str=temp.read_text(encoding="utf-8")
    )

    if restored.num_trees() != model.best_iteration_:
        raise ValueError("Model save verification failed.")

    temp.replace(path)


# ============================================================
# Train each split
# ============================================================

def progress(env):

    iteration = env.iteration + 1

    if iteration % 500 == 0:
        print(
            f"  iteration {iteration}/{env.end_iteration}",
            flush=True,
        )


def eval_rmsle_log(y_true, y_pred):
    """Exact competition RMSLE using the existing log1p target."""
    # The existing prediction pipeline clips negative log predictions to 0.
    pred_log = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    true_log = np.asarray(y_true, dtype=float)
    score = float(np.sqrt(np.mean((true_log - pred_log) ** 2)))
    return "rmsle", score, False


def train_split(df, name, params, output):

    train, valid = make_split(df, SPECS[name])
    xt, xv = make_matrices(train, valid)

    print(f"\n========== {name.upper()} ==========")
    print(f"Train rows: {len(train):,}")
    print(f"Valid rows: {len(valid):,}")
    print(f"Features: {len(FEATURES)}")
    print(f"Maximum rounds: {MAX_ESTIMATORS} (not a fixed iteration)")
    print("Parameters: Oil Raw Optuna Best")
    print(f"Early stopping: ON (patience={EARLY_STOPPING_ROUNDS})", flush=True)

    model_params = dict(params)
    model_params["n_estimators"] = MAX_ESTIMATORS
    # Use only competition-equivalent RMSLE for early stopping.
    model_params["metric"] = "None"

    model = lgb.LGBMRegressor(**model_params)

    model.fit(
        xt,
        train[TARGET],
        eval_set=[(xv, valid[TARGET])],
        eval_names=["valid"],
        eval_metric=eval_rmsle_log,
        categorical_feature=CATEGORICAL,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=True,
            ),
            progress,
        ],
    )

    best_iteration = int(model.best_iteration_)
    if best_iteration <= 0:
        raise ValueError("Early Stopping did not provide a best iteration.")
    print(f"Best iteration ({name}): {best_iteration}", flush=True)
    if best_iteration >= MAX_ESTIMATORS - EARLY_STOPPING_ROUNDS:
        print("[WARNING] Best iteration is near the maximum rounds; "
              "consider raising MAX_ESTIMATORS.", flush=True)

    prediction_log = model.predict(
        xv,
        num_iteration=best_iteration,
    )

    if not np.isfinite(prediction_log).all():
        raise ValueError("Invalid model predictions.")

    predicted_sales = np.expm1(
        np.maximum(prediction_log, 0)
    )

    pred = valid[ROW_KEYS + ["sales"]].copy()
    pred["pred_sales"] = predicted_sales

    scores = score_predictions(pred, name)

    save_csv(pred, output / f"{name}_predictions.csv")
    save_csv(scores, output / f"{name}_scores.csv")

    save_model(model, output / f"{name}_model.txt")

    importance = pd.DataFrame({
        "feature": FEATURES,
        "gain": model.booster_.feature_importance("gain"),
        "split": model.booster_.feature_importance("split"),
    })

    save_csv(
        importance.sort_values("gain", ascending=False),
        output / f"{name}_importance.csv",
    )

    print("\n[SCORES]")
    print(scores.round(6).to_string(index=False), flush=True)

    del model, train, valid, xt, xv, pred
    gc.collect()

    return scores


# ============================================================
# Summary
# ============================================================

def summarize(reports, output):

    combined = pd.concat(reports, ignore_index=True)

    fold23 = (
        combined[combined["split"].isin(["fold2", "fold3"])]
        .groupby("family", as_index=False)["rmsle"]
        .mean()
    )

    fold23["split"] = "fold23_mean"

    all_scores = pd.concat(
        [combined, fold23],
        ignore_index=True,
    )

    save_csv(
        all_scores,
        output / "all_validation_backtest_scores.csv",
    )

    family_table = (
        all_scores.pivot(
            index="family",
            columns="split",
            values="rmsle",
        )
        .reindex(
            columns=["fold2", "fold3", "fold23_mean", "backtest"]
        )
        .reset_index()
    )

    save_csv(
        family_table,
        output / "family_error_comparison.csv",
    )

    print("\n========== OVERALL RMSLE ==========")
    print(
        all_scores[all_scores["family"] == "ALL"]
        [["split", "rmsle"]]
        .round(6)
        .to_string(index=False)
    )

    print("\n========== FAMILY RMSLE ==========")
    print(family_table.round(6).to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=OIL_OPTUNA_RUN,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = run_dir / "ema7_only_cv_results"

    params, best, params_path = load_oil_optuna(
        run_dir
    )

    os.chdir(ROOT)

    raw, test, holidays = original_pipeline()

    df = prepare_features(raw)
    df = add_oil_ema7(df, raw)

    del raw, test, holidays
    gc.collect()

    for name in SPECS:
        tr, va = make_split(df, SPECS[name])
        xt, xv = make_matrices(tr, va)
        print(
            f"[CHECK] {name}: "
            f"train={len(tr):,} valid={len(va):,} "
            f"features={xt.shape[1]} "
            f"max_rounds={MAX_ESTIMATORS} early_stopping={EARLY_STOPPING_ROUNDS}",
            flush=True,
        )
        del tr, va, xt, xv

    gc.collect()

    if args.check_only:
        print("[CHECK OK] No models trained.")
        return

    output.mkdir(parents=True, exist_ok=True)

    save_json({
        "parameter_source": str(params_path),
        "parameter_sha256": sha256(params_path),
        "parameter_experiment": "Oil Raw Optuna 30-trial",
        "best_trial_number": best.get("best_trial_number"),
        "original_tuned_features": best["features"],
        "experiment_features": FEATURES,
        "params": params,
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_metric": "rmsle_log_target_clipped",
        "oil": "EMA7 from preprocessed dcoilwtico",
        "ema_span": 7,
        "ema_adjust": False,
        "ema_min_periods": 7,
        "optuna_rerun": False,
        "early_stopping": True,
        "family_routing": False,
        "lightgbm_version": lgb.__version__,
    }, output / "config.json")

    reports = []

    for name in SPECS:
        reports.append(
            train_split(df, name, params, output)
        )

    summarize(reports, output)

    print("\n[COMPLETE]")
    print("Saved:", output)


if __name__ == "__main__":
    main()
