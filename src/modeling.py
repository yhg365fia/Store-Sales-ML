"""Controlled 16-day CV experiments: Base23 + sales_lag_364, plus ONE addition.

Put in src/modeling_overnight_ablation.py, next to main.py/feature_engineering.py.
Run at project root:
    python src/modeling_overnight_ablation.py --check-only
    python src/modeling_overnight_ablation.py
    python src/modeling_overnight_ablation.py --run-dir results/base23_lag364_overnight/<run_id>

Exactly the existing project's 3 folds, LightGBM FIXED_PARAMS, early stopping,
log1p target, categorical handling and RMSLE. No Optuna, no submission.
The *only predictor changes* are oil raw, lagged transactions, sales lag16, or
sales lag28, added ONE AT A TIME to a shared Base23 + sales_lag_364.

IMPORTANT: Transactions on the SAME validation date would not exist in Kaggle
Test. The transactions experiment therefore uses transactions_lag_16 (known at
all 16 prediction dates), not raw future-day transactions. Existing main()
preprocessing/imputation remains unchanged; review its interpolation policy
if strict fold-cutoff availability is required.
"""

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from main import main
from feature_engineering import add_lag_features


# ============================================================================
# 1. FROZEN configuration copied from the supplied modeling.py (2026-10-26)
# ============================================================================
BASE_FEATURES = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion", "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]
CATEGORICAL_FEATURES = ["store_nbr", "family", "city", "state", "type", "cluster"]
KEYS = ["store_nbr", "family"]
TARGET_COL = "sales_log"
FORECAST_HORIZON = 16
N_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 150
FOLDS = [
    {"fold": "Fold 1", "train_end": "2017-06-28", "valid_start": "2017-06-29", "valid_end": "2017-07-14"},
    {"fold": "Fold 2", "train_end": "2017-07-14", "valid_start": "2017-07-15", "valid_end": "2017-07-30"},
    {"fold": "Fold 3", "train_end": "2017-07-30", "valid_start": "2017-07-31", "valid_end": "2017-08-15"},
]
FIXED_PARAMS = dict(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.05,
    n_estimators=N_ESTIMATORS,
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

# ONLY this baseline / these four modifications are compared.
# Payday (is_15th, is_month_end) is never a model input.
LAG_364 = "sales_lag_364"
OIL_FEATURE = "oil_raw"
TRANSACTIONS_FEATURE = "transactions_lag_16"
FEATURE_SETS = {
    "base23_lag364": BASE_FEATURES + [LAG_364],
    "add_oil_raw": BASE_FEATURES + [LAG_364, OIL_FEATURE],
    "add_transactions": BASE_FEATURES + [LAG_364, TRANSACTIONS_FEATURE],
    "add_sales_lag16": BASE_FEATURES + [LAG_364, "sales_lag_16"],
    "add_sales_lag28": BASE_FEATURES + [LAG_364, "sales_lag_28"],
}

# main() output schema was NOT provided. These are expected raw-source names.
# If main() uses different names, change ONLY these aliases and record them.
OIL_SOURCE_CANDIDATES = ("dcoilwtico", "oil_price", "oil_raw", "oil", "oil_price_raw")
TRANSACTIONS_SOURCE_CANDIDATES = ("transactions", "transactions_filled", "transaction")
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results" / "base23_lag364_overnight"


def validate_configuration():
    if len(BASE_FEATURES) != 23 or len(set(BASE_FEATURES)) != 23:
        raise ValueError("BASE_FEATURES must contain exactly the original 23 features")
    if len(FEATURE_SETS) != 5 or any(len(v) != (24 if k == "base23_lag364" else 25)
                                     for k, v in FEATURE_SETS.items()):
        raise ValueError("Expected the 24-feature baseline plus four 25-feature variants")
    expected = BASE_FEATURES + [LAG_364]
    if FEATURE_SETS["base23_lag364"] != expected:
        raise ValueError("Baseline features changed")
    for name, cols in FEATURE_SETS.items():
        if cols[:24] != expected or len(set(cols)) != len(cols):
            raise ValueError(f"Non-controlled feature change in {name}")
        if any(x in cols for x in ("is_15th", "is_month_end", "sales", TARGET_COL)):
            raise ValueError(f"Payday/target unexpectedly included in {name}")
    if len(FOLDS) != 3:
        raise ValueError("Exactly the original three folds are required")
    for fold in FOLDS:
        end = pd.Timestamp(fold["train_end"])
        start = pd.Timestamp(fold["valid_start"])
        stop = pd.Timestamp(fold["valid_end"])
        if start != end + pd.Timedelta(days=1) or (stop - start).days + 1 != FORECAST_HORIZON:
            raise ValueError(f"Invalid 16-day fold: {fold['fold']}")
        for lag in (16, 28, 364):
            if stop - pd.Timedelta(days=lag) > end:
                raise ValueError(f"lag{lag} would access validation targets: {fold['fold']}")


def resolve_source(columns, names, label):
    hits = [s for s in names if s in columns]
    if len(hits) != 1:
        raise ValueError(
            f"Cannot safely identify {label} source column: candidates found={hits}. "
            f"Expected exactly one of {names}; actual columns={list(columns)}. "
            "Check main() output and edit SOURCE_CANDIDATES at the top of this file."
        )
    return hits[0]


def prepare_features(df):
    validate_configuration()
    oil_source = resolve_source(df.columns, OIL_SOURCE_CANDIDATES, "oil raw")
    txn_source = resolve_source(df.columns, TRANSACTIONS_SOURCE_CANDIDATES, "transactions")
    required = list(dict.fromkeys(["date", "sales"] + BASE_FEATURES + [oil_source, txn_source]))
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"main() missing required columns: {missing}")

    # Never feed other columns from main() to the model.
    df = df.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df[KEYS + ["date"]].isna().any().any():
        raise ValueError("Null store_nbr/family/date")
    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store_nbr/family/date")
    if df[BASE_FEATURES].isna().any().any():
        raise ValueError("Null in original 23 predictors")
    if df["date"].max() != pd.Timestamp("2017-08-15"):
        raise ValueError("Expected last training date 2017-08-15")
    for c in ("sales", oil_source, txn_source):
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite raw values in {c}; inspect main() imputation")
    if (df["sales"] < 0).any() or (df[txn_source] < 0).any():
        raise ValueError("Negative sales or transactions")
    if not np.array_equal(df["dayofweek"], df["date"].dt.dayofweek):
        raise ValueError("Calendar dayofweek inconsistent")
    if not np.array_equal(df["is_weekend"], (df["date"].dt.dayofweek >= 5).astype(int)):
        raise ValueError("Calendar is_weekend inconsistent")

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    diffs = df.groupby(KEYS, observed=True)["date"].diff()
    if (diffs.notna() & diffs.ne(pd.Timedelta(days=1))).any():
        raise ValueError("Non-contiguous dates; lag shifts would not mean calendar days")

    # Exactly the original lag helper and raw-sales definition.
    # All three lags are calculated once; only one experimental column is selected.
    df = add_lag_features(df, lags=(364, 16, 28))
    for d in (364, 16, 28):
        col = f"sales_lag_{d}"
        if col not in df.columns:
            raise ValueError(f"add_lag_features() did not create {col}")
        expected_na = df.groupby(KEYS, observed=True).cumcount().lt(d)
        if not df[col].isna().equals(expected_na):
            raise ValueError(f"Unexpected missing positions in {col}")

    # External predictors. Oil is the RAW oil value after existing main() QC.
    # Transactions is shifted 16 days within each store/family history:
    # every value used for a 16-day horizon was known at the prediction cutoff.
    df[OIL_FEATURE] = df[oil_source].astype(float)
    df[TRANSACTIONS_FEATURE] = df.groupby(KEYS, observed=True)[txn_source].shift(16)
    expected_txn_na = df.groupby(KEYS, observed=True).cumcount().lt(16)
    if not df[TRANSACTIONS_FEATURE].isna().equals(expected_txn_na):
        raise ValueError("transactions_lag_16 unexpected null positions")

    # Unchanged target definition.
    df[TARGET_COL] = np.log1p(df["sales"])
    print(f"[DATA] train rows={len(df):,}; oil source={oil_source}; transactions source={txn_source}", flush=True)
    print("[RULE] Raw transactions on validation dates NOT used. "+
          "Transactions experiment adds transactions_lag_16 only.", flush=True)
    print("[RULE] No Payday predictors in any variant; Optuna disabled.", flush=True)
    return df, {"oil_source": oil_source, "transactions_source": txn_source}


def make_fold(df, fold):
    end = pd.Timestamp(fold["train_end"])
    start = pd.Timestamp(fold["valid_start"])
    stop = pd.Timestamp(fold["valid_end"])
    train_df = df.loc[df["date"].le(end)]
    valid_df = df.loc[df["date"].between(start, stop)]
    counts = valid_df.groupby(KEYS, observed=True).size()
    if (train_df.empty or train_df["date"].max() != end
            or set(valid_df["date"]) != set(pd.date_range(start, stop))
            or len(valid_df) != 28_512 or len(counts) != 1782
            or not counts.eq(FORECAST_HORIZON).all()):
        raise ValueError(f"Incomplete 16-day validation panel in {fold['fold']}")
    for col in (LAG_364, "sales_lag_16", "sales_lag_28", TRANSACTIONS_FEATURE, OIL_FEATURE):
        if valid_df[col].isna().any():
            raise ValueError(f"{fold['fold']}: {col} has missing validation values")
    # Sanity: target/txn lag values used for the 16-day interval are historical.
    if stop - pd.Timedelta(days=16) > end:
        raise ValueError("Feature lookahead detected")
    return train_df, valid_df


def prepare_model_data(train_df, valid_df, features):
    x_train = train_df.loc[:, features].copy()
    x_valid = valid_df.loc[:, features].copy()
    for col in CATEGORICAL_FEATURES:
        categories = pd.Index(train_df[col].dropna().unique())
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_valid[col] = pd.Categorical(x_valid[col], categories=categories)
        if x_valid[col].isna().any():
            raise ValueError(f"New category in validation: {col}")
    if list(x_train.columns) != features or list(x_valid.columns) != features:
        raise ValueError("Predictor column order changed")
    return x_train, x_valid


def calculate_rmsle(y_true, pred_log):
    y_true = np.asarray(y_true, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    if not np.isfinite(pred_log).all():
        raise ValueError("Nonfinite model predictions")
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.maximum(pred_log, 0)) ** 2)))


def write_json_atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv_atomic(path, df):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def current_config(data_sources=None):
    return {
        "purpose": "Base23+lag364 baseline; four single-addition fixed-parameter experiments",
        "feature_sets": FEATURE_SETS,
        "folds": FOLDS,
        "fixed_params": FIXED_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "target": "log1p(sales)",
        "metric": "RMSLE on sales via clipping predicted log at zero",
        "transactions_transform": "transactions_lag_16, not future-date transactions",
        "sales_lag_transform": "existing add_lag_features; raw sales; shifts 16/28/364",
        "main_preprocessing": "unchanged; main() supplies imputed raw oil and transactions",
        "source_candidates": {"oil": list(OIL_SOURCE_CANDIDATES), "transactions": list(TRANSACTIONS_SOURCE_CANDIDATES)},
        "data_sources": data_sources,
        "optuna": False,
        "submission": False,
    }


def initialize_output(out_dir, sources, resume):
    config_path = out_dir / "run_config.json"
    config = current_config(sources)
    if resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot resume: {config_path} missing")
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError("Refusing to resume: config or source columns changed")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(config_path, config)


def train_one(train_df, valid_df, features, fold_name):
    x_train, x_valid = prepare_model_data(train_df, valid_df, features)
    model = LGBMRegressor(**FIXED_PARAMS.copy())
    print(f"[FIT] {fold_name}: train={len(train_df):,}, valid={len(valid_df):,}", flush=True)
    model.fit(
        x_train, train_df[TARGET_COL],
        eval_set=[(x_valid, valid_df[TARGET_COL])],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=True), log_evaluation(200)],
    )
    best_iteration = int(model.best_iteration_)
    prediction = model.predict(x_valid, num_iteration=best_iteration)
    score = calculate_rmsle(valid_df["sales"], prediction)
    names = model.booster_.feature_name()
    if names != features:
        raise ValueError("Model booster feature order is not the specified order")
    gain = model.booster_.feature_importance(importance_type="gain", iteration=best_iteration)
    split = model.booster_.feature_importance(importance_type="split", iteration=best_iteration)
    importance = pd.DataFrame({"feature": names, "gain": gain, "split": split})
    del model, x_train, x_valid, prediction
    gc.collect()
    return best_iteration, score, importance


def run(df, out_dir):
    print("\n[EXPERIMENT] Base23+lag364 / oil_raw / transactions_lag_16 / lag16 / lag28")
    print("[CONTROL] same 3 folds, full training cutoff, target, params, seed, early stop")
    print(f"[FITS] 5 configurations x 3 folds = 15 fits (barring resume skips)", flush=True)
    for experiment, features in FEATURE_SETS.items():
        exp_dir = out_dir / experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        for fold in FOLDS:
            number = int(fold["fold"].split()[-1])
            fold_path = exp_dir / f"fold_{number}.json"
            importance_path = exp_dir / f"importance_fold_{number}.csv"
            if fold_path.exists() and importance_path.exists():
                done = json.loads(fold_path.read_text(encoding="utf-8"))
                if done.get("feature_names") != features or done.get("fold") != fold:
                    raise ValueError(f"Mismatch in existing result: {fold_path}")
                print(f"[SKIP] {experiment} {fold['fold']} already completed", flush=True)
                continue
            if fold_path.exists() or importance_path.exists():
                raise ValueError(f"Partial/mismatched saved pair for {experiment}/{fold['fold']}")
            print("\n" + "=" * 76)
            print(f"[START] {experiment}, {fold['fold']} | feature_count={len(features)}")
            print(f"[CHANGED] {features[24:] if len(features) > 24 else 'baseline only'}")
            print("=" * 76, flush=True)
            tr, va = make_fold(df, fold)
            best_iteration, rmsle, importance = train_one(tr, va, features, fold["fold"])
            result = {
                "experiment": experiment, "fold": fold, "feature_names": features,
                "feature_count": len(features), "train_rows": int(len(tr)),
                "valid_rows": int(len(va)), "best_iteration": best_iteration,
                "rmsle": rmsle,
            }
            write_csv_atomic(importance_path, importance)
            write_json_atomic(fold_path, result)
            write_current_summary(out_dir)
            print(f"[DONE] {experiment}/{fold['fold']}: RMSLE={rmsle:.6f} | best_iteration={best_iteration}", flush=True)
            del tr, va
            gc.collect()
        write_current_summary(out_dir)
    write_current_summary(out_dir)
    print(f"\n[FINISHED] Results saved: {out_dir.resolve()}", flush=True)


def write_current_summary(out_dir):
    rows = []
    for exp, features in FEATURE_SETS.items():
        for fold in FOLDS:
            p = out_dir / exp / f"fold_{int(fold['fold'].split()[-1])}.json"
            if not p.is_file():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["feature_names"] != features or r["fold"] != fold:
                raise ValueError(f"Saved experiment mismatch: {p}")
            rows.append({
                "experiment": exp, "fold": fold["fold"], "features": len(features),
                "train_end": fold["train_end"], "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"], "rmsle": r["rmsle"],
                "best_iteration": r["best_iteration"], "valid_rows": r["valid_rows"],
            })
    if not rows:
        return
    results = pd.DataFrame(rows)
    write_csv_atomic(out_dir / "fold_results_partial.csv", results)
    summary = (results.groupby("experiment", sort=False)
               .agg(folds_completed=("fold", "count"), mean_rmsle=("rmsle", "mean"),
                    min_rmsle=("rmsle", "min"), max_rmsle=("rmsle", "max"),
                    median_best_iteration=("best_iteration", "median"))
               .reset_index())
    summary["fold_gap_max_minus_min"] = summary["max_rmsle"] - summary["min_rmsle"]
    stds = results.groupby("experiment")["rmsle"].std(ddof=1)
    summary["std_rmsle_ddof1"] = summary["experiment"].map(stds)
    baseline = summary.loc[summary["experiment"] == "base23_lag364"]
    if len(baseline) and int(baseline.iloc[0]["folds_completed"]) == 3:
        ref = float(baseline.iloc[0]["mean_rmsle"])
        summary["delta_vs_base23_lag364"] = summary["mean_rmsle"] - ref
        summary.loc[summary["folds_completed"] != 3, "delta_vs_base23_lag364"] = np.nan
    else:
        summary["delta_vs_base23_lag364"] = np.nan
    write_csv_atomic(out_dir / "summary_partial.csv", summary)
    if (len(summary) == len(FEATURE_SETS) and summary["folds_completed"].eq(3).all()):
        write_csv_atomic(out_dir / "ablation_summary.csv", summary)
        write_csv_atomic(out_dir / "ablation_fold_results.csv", results)
    print("\n[PROGRESS]\n" + summary.to_string(index=False), flush=True)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only", action="store_true", help="Prepare/validate pipeline only, no training")
    p.add_argument("--run-dir", type=Path, default=None, help="Resume an existing timestamped results folder")
    a = p.parse_args()
    if a.check_only and a.run_dir is not None:
        p.error("--check-only cannot be combined with --run-dir")
    validate_configuration()
    train, test, holidays = main()
    del test, holidays
    gc.collect()
    train, sources = prepare_features(train)
    for fold in FOLDS:
        make_fold(train, fold)
    print("[PREFLIGHT PASS] all required source columns, lags, validation panels are valid", flush=True)
    if a.check_only:
        return
    if a.run_dir is None:
        out = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        initialize_output(out, sources, resume=False)
    else:
        out = a.run_dir.expanduser().resolve()
        initialize_output(out, sources, resume=True)
    print(f"[RESULT FOLDER] {out.resolve()}", flush=True)
    run(train, out)


if __name__ == "__main__":
    cli()
