"""2016-08-16~31 Store Sales backtest for three frozen feature sets.

Place this file at ``src/backtest.py`` beside main.py and feature_engineering.py.

Default ``early_stop`` reproduces the previous validation-style training: the
2016 validation labels select LightGBM's best iteration. For a genuinely
2016 scoring-only holdout, use ``--mode fixed --iterations-json FILE`` and select
the iteration counts *before* inspecting the 2016 labels. Note: the feature
sets and parameters have already been informed by later project experiments;
this is not a fully prospectively designed, never-seen historical holdout.

Only Kaggle competition availability is assumed for oil: oil.csv provides the
16 validation dates, which is different from real-time 16-day forecasting.
Transactions used in validation are exclusively 16-day-old historical values.

Outputs: run_config.json, backtest_summary.csv, backtest_fold_results.csv,
backtest_daily_rmsle.csv, and per-model prediction/importance CSVs.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TRAIN_END = pd.Timestamp("2016-08-15")
VALID_START = pd.Timestamp("2016-08-16")
VALID_END = pd.Timestamp("2016-08-31")
HORIZON = 16
KEYS = ["store_nbr", "family"]
CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]

BASE23 = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion", "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]

FEATURE_SETS = {
    "base23_lag364_lag28": BASE23 + ["sales_lag_364", "sales_lag_28"],
    "add_transactions_lag16": BASE23 + [
        "sales_lag_364", "sales_lag_28", "transactions_lag_16"
    ],
    "add_oil_raw": BASE23 + ["sales_lag_364", "sales_lag_28", "oil_raw"],
}
BASELINE = "base23_lag364_lag28"

MODEL_PARAMS = dict(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.05,
    n_estimators=8000,
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
EARLY_STOPPING = 150
EXPECTED_VALID_ROWS = 54 * 33 * HORIZON


def assert_static_config() -> None:
    if len(BASE23) != 23 or len(set(BASE23)) != 23:
        raise AssertionError("Base23 column list changed")
    if list(FEATURE_SETS) != [BASELINE, "add_transactions_lag16", "add_oil_raw"]:
        raise AssertionError("Experiments unexpectedly changed")
    if [len(v) for v in FEATURE_SETS.values()] != [25, 26, 26]:
        raise AssertionError("Feature counts must be 25, 26, 26")
    if any("is_15th" in cols or "is_month_end" in cols for cols in FEATURE_SETS.values()):
        raise AssertionError("Payday accidentally included")
    if (VALID_END - VALID_START).days + 1 != HORIZON:
        raise AssertionError("Validation window must contain exactly 16 days")
    if TRAIN_END + pd.Timedelta(days=1) != VALID_START:
        raise AssertionError("Training and validation windows must be adjacent")
    for lag in (16, 28, 364):
        if VALID_END - pd.Timedelta(days=lag) > TRAIN_END:
            raise AssertionError(f"Lag {lag} can see validation targets")


def nonfinite_check(series: pd.Series, name: str, allow_na: bool = False) -> None:
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if allow_na:
        if np.isinf(values).any():
            raise ValueError(f"{name}: infinite values")
    elif not np.isfinite(values).all():
        raise ValueError(f"{name}: missing or infinite values")


def load_train_frame() -> pd.DataFrame:
    # Import the *project's* preprocessing; do not replace its Base23 features.
    from main import main

    result = main()
    if not isinstance(result, (tuple, list)) or not result:
        raise TypeError("main() must return (train, test, holidays)")
    original = result[0]
    if not isinstance(original, pd.DataFrame):
        raise TypeError("First main() return value must be the training DataFrame")
    needed = list(dict.fromkeys(["date", "sales"] + BASE23))
    missing = [c for c in needed if c not in original.columns]
    if missing:
        raise ValueError(f"main() lacks Base23 / target columns: {missing}")

    # This restriction occurs *before* generating any sales/transaction lags.
    dates = pd.to_datetime(original["date"], errors="raise")
    df = original.loc[dates.le(VALID_END), needed].copy()
    df["date"] = dates.loc[dates.le(VALID_END)]
    del original, result
    gc.collect()

    if df.empty or df.date.max() != VALID_END:
        raise ValueError("2016-08-31 is missing from train.csv")
    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store/family/date rows")
    if df[BASE23 + ["date", "sales"]].isna().any().any():
        raise ValueError("Missing Base23, date or sales values")
    nonfinite_check(df.sales, "sales")
    if (df.sales < 0).any():
        raise ValueError("Negative training sales")
    if not np.array_equal(df.dayofweek.to_numpy(), df.date.dt.dayofweek.to_numpy()):
        raise ValueError("dayofweek disagrees with date")
    if not np.array_equal(df.is_weekend.to_numpy(),
                          df.date.dt.dayofweek.ge(5).astype(int).to_numpy()):
        raise ValueError("is_weekend disagrees with date")

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    groups = df.groupby(KEYS, sort=False, observed=True)
    jumps = groups.date.diff()
    if (jumps.notna() & jumps.ne(pd.Timedelta(days=1))).any():
        raise ValueError("One or more store/family series are not daily-contiguous")
    print(f"[DATA] Rows through 2016-08-31: {len(df):,}")
    return df


def attach_sales_lags(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve the source project's exact lag helper and raw-sales definition.
    from feature_engineering import add_lag_features

    df = add_lag_features(df, lags=(364, 16, 28))
    group_pos = df.groupby(KEYS, sort=False, observed=True).cumcount()
    for lag in (364, 16, 28):
        col = f"sales_lag_{lag}"
        if col not in df:
            raise ValueError(f"add_lag_features did not create {col}")
        expected = group_pos.lt(lag)
        if not df[col].isna().equals(expected):
            raise ValueError(f"{col}: missing positions differ from expected initial history")
        nonfinite_check(df[col], col, allow_na=True)
    return df


def attach_cutoff_transactions(df: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Reconstruct transactions using *only* observed rows at/before 2016-08-15.

    This intentionally bypasses any imputation main() might have performed
    using 2016-09..2017 information. Policy: sales_total==0 -> zero;
    remaining gaps -> per-store time interpolation, bounded by TRAIN_END.
    """
    if not raw_path.is_file():
        raise FileNotFoundError(f"transactions.csv missing: {raw_path}")
    raw = pd.read_csv(raw_path, usecols=["date", "store_nbr", "transactions"])
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    raw = raw.loc[raw.date.le(TRAIN_END)].copy()
    if raw.duplicated(["store_nbr", "date"]).any():
        raise ValueError("Duplicate raw transactions store/date")
    raw["transactions"] = pd.to_numeric(raw.transactions, errors="raise")
    nonfinite_check(raw.transactions, "observed transactions", allow_na=True)
    if raw.transactions.lt(0).any():
        raise ValueError("Negative observed transaction values")

    historical = df.loc[df.date.le(TRAIN_END)]
    daily = (historical.groupby(["store_nbr", "date"], observed=True, sort=False)
             .sales.sum().reset_index(name="daily_sales"))
    daily = daily.merge(raw, on=["store_nbr", "date"], how="left", validate="one_to_one")
    missing0 = int(daily.transactions.isna().sum())
    zero_mask = daily.transactions.isna() & daily.daily_sales.eq(0)
    daily.loc[zero_mask, "transactions"] = 0.0
    n_zeros = int(zero_mask.sum())
    daily = daily.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    daily["transactions_filled"] = np.nan

    # Time interpolation never consults transaction rows after 2016-08-15.
    for _, idx in daily.groupby("store_nbr", observed=True, sort=False).indices.items():
        block = daily.iloc[idx]
        series = pd.Series(block.transactions.to_numpy(dtype=float),
                           index=pd.DatetimeIndex(block.date))
        filled = series.interpolate(method="time", limit_direction="both")
        daily.loc[idx, "transactions_filled"] = filled.to_numpy(dtype=float)
    nonfinite_check(daily.transactions_filled, "bounded transactions")
    if daily.transactions_filled.lt(0).any():
        raise ValueError("Negative imputed transactions")

    # Validation dates lack same-day transactions by design. Shift after merge
    # so all 16 validation days reference <=2016-08-15 observations only.
    df = df.merge(daily[["store_nbr", "date", "transactions_filled"]],
                  on=["store_nbr", "date"], how="left", validate="many_to_one",
                  sort=False)
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    df["transactions_lag_16"] = (
        df.groupby(KEYS, observed=True, sort=False).transactions_filled.shift(16)
    )
    expected_na = df.groupby(KEYS, observed=True, sort=False).cumcount().lt(16)
    if not df.transactions_lag_16.isna().equals(expected_na):
        raise ValueError("transactions_lag_16 has unexpected missing values")
    print(f"[TXN] Raw <= 2016-08-15; missing={missing0:,}; "
          f"zero_sales_zero={n_zeros:,}; interpolated={missing0-n_zeros:,}")
    return df.drop(columns=["transactions_filled"])


def attach_bounded_oil(df: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Use Kaggle-provided oil only up to the end of this 16-day block."""
    if not raw_path.is_file():
        raise FileNotFoundError(f"oil.csv missing: {raw_path}")
    raw = pd.read_csv(raw_path, usecols=["date", "dcoilwtico"])
    raw.date = pd.to_datetime(raw.date, errors="raise")
    if raw.duplicated("date").any():
        raise ValueError("Duplicate oil dates")
    raw = raw.loc[raw.date.le(VALID_END)].copy()
    raw["dcoilwtico"] = pd.to_numeric(raw.dcoilwtico, errors="raise")
    nonfinite_check(raw.dcoilwtico, "observed oil", allow_na=True)
    dates = pd.date_range(df.date.min(), VALID_END, freq="D")
    series = raw.set_index("date").dcoilwtico.reindex(dates)
    missing = int(series.isna().sum())
    # This may use 2016-08-16..31 oil to interpolate within the holdout;
    # allowed by the Kaggle-provided-oil policy, not by a live 16-day forecast.
    series = series.interpolate(method="time", limit_direction="both")
    if series.isna().any():
        raise ValueError("Oil is missing throughout the bounded period")
    df["oil_raw"] = df.date.map(series).astype(float)
    nonfinite_check(df.oil_raw, "bounded oil")
    print(f"[OIL] Observations through 2016-08-31; daily gaps filled={missing:,}")
    return df


def prepare_backtest(raw_transactions: Path, raw_oil: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_static_config()
    df = load_train_frame()
    df = attach_sales_lags(df)
    df = attach_cutoff_transactions(df, raw_transactions)
    df = attach_bounded_oil(df, raw_oil)
    df["sales_log"] = np.log1p(df.sales.astype(float))

    train = df.loc[df.date.le(TRAIN_END)].copy()
    valid = df.loc[df.date.between(VALID_START, VALID_END)].copy()
    if train.empty or train.date.max() != TRAIN_END:
        raise ValueError("Train cutoff 2016-08-15 not represented")
    if len(valid) != EXPECTED_VALID_ROWS:
        raise ValueError(f"Expected {EXPECTED_VALID_ROWS:,} valid rows; got {len(valid):,}")
    counts = valid.groupby(KEYS, observed=True).size()
    if len(counts) != 1782 or not counts.eq(16).all():
        raise ValueError("Incomplete 54-store x 33-family x 16-day validation panel")
    if set(valid.date) != set(pd.date_range(VALID_START, VALID_END, freq="D")):
        raise ValueError("Some validation dates are missing")
    if valid[["sales_lag_364", "sales_lag_28", "sales_lag_16",
              "transactions_lag_16", "oil_raw"]].isna().any().any():
        raise ValueError("Validation features must not contain NaN")
    for col in ("sales_lag_364", "sales_lag_28", "transactions_lag_16", "oil_raw"):
        nonfinite_check(valid[col], col)
    print(f"[SPLIT] Train <= {TRAIN_END.date()}: {len(train):,}; "
          f"Valid {VALID_START.date()}..{VALID_END.date()}: {len(valid):,}")
    print("[QC] Sales/transactions lags cannot access 2016-08-16..31 targets")
    print("[POLICY] Oil is Kaggle-provided by date; NOT realistic live-future availability")
    return train, valid


def make_xy(train: pd.DataFrame, valid: pd.DataFrame,
            features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    xtrain = train[features].copy()
    xvalid = valid[features].copy()
    for c in CATEGORICAL:
        categories = pd.Index(train[c].dropna().unique())
        xtrain[c] = pd.Categorical(xtrain[c], categories=categories)
        xvalid[c] = pd.Categorical(xvalid[c], categories=categories)
        if xvalid[c].isna().any():
            raise ValueError(f"Unknown validation category: {c}")
    if xtrain.columns.tolist() != features or xvalid.columns.tolist() != features:
        raise ValueError("Feature column order was modified")
    return xtrain, xvalid


def get_fixed_iterations(path: Path | None, mode: str) -> dict[str, int]:
    if mode != "fixed":
        return {}
    if path is None or not path.is_file():
        raise ValueError("--mode fixed requires --iterations-json FILE")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != set(FEATURE_SETS):
        raise ValueError(f"Iterations JSON must contain exactly: {list(FEATURE_SETS)}")
    if any(type(v) is not int or not (1 <= v <= 8000) for v in values.values()):
        raise ValueError("Every n_estimators value must be an integer in 1..8000")
    return values


def rmsle_from_log(y_true: np.ndarray, pred_log: np.ndarray) -> float:
    if not np.isfinite(pred_log).all():
        raise ValueError("Model returned NaN/Inf predictions")
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.maximum(pred_log, 0)) ** 2)))


def save_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_csv(path: Path, value: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    value.to_csv(temp, index=False)
    temp.replace(path)


def run_one(train: pd.DataFrame, valid: pd.DataFrame, name: str,
            mode: str, fixed: dict[str, int], out: Path) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    features = FEATURE_SETS[name]
    xtrain, xvalid = make_xy(train, valid, features)
    args = MODEL_PARAMS.copy()
    if mode == "fixed":
        args["n_estimators"] = fixed[name]
    model = LGBMRegressor(**args)
    if mode == "early_stop":
        model.fit(
            xtrain, train.sales_log,
            eval_set=[(xvalid, valid.sales_log)],
            eval_metric="rmse",
            categorical_feature=CATEGORICAL,
            callbacks=[early_stopping(EARLY_STOPPING, verbose=True), log_evaluation(200)],
        )
        iterations = int(model.best_iteration_ or args["n_estimators"])
    else:
        # 2016 validation targets are never passed to model.fit in fixed mode.
        model.fit(xtrain, train.sales_log, categorical_feature=CATEGORICAL)
        iterations = int(args["n_estimators"])

    raw_pred_log = np.asarray(model.predict(xvalid, num_iteration=iterations), dtype=float)
    ytrue = valid.sales.to_numpy(dtype=float)
    score = rmsle_from_log(ytrue, raw_pred_log)
    names = list(model.booster_.feature_name())
    if names != features:
        raise ValueError(f"Trained booster feature order changed for {name}")

    pred = valid[["date", "store_nbr", "family", "sales"]].copy()
    pred["prediction"] = np.expm1(np.maximum(raw_pred_log, 0))
    pred["squared_log_error"] = (
        np.log1p(pred.sales.to_numpy(dtype=float)) - np.maximum(raw_pred_log, 0)
    ) ** 2
    pred["experiment"] = name
    save_csv(out / f"predictions_{name}.csv", pred)

    imp = pd.DataFrame({
        "feature": names,
        "gain": model.booster_.feature_importance(importance_type="gain", iteration=iterations),
        "split": model.booster_.feature_importance(importance_type="split", iteration=iterations),
    })
    save_csv(out / f"feature_importance_{name}.csv", imp)

    row = dict(
        experiment=name, feature_count=len(features), train_end=str(TRAIN_END.date()),
        valid_start=str(VALID_START.date()), valid_end=str(VALID_END.date()),
        train_rows=len(train), valid_rows=len(valid), mode=mode,
        best_iteration=iterations, rmsle=score,
    )
    save_json(out / f"result_{name}.json", row)
    print(f"[RESULT] {name}: RMSLE={score:.6f}, iterations={iterations}", flush=True)
    del model, xtrain, xvalid, pred, imp
    gc.collect()
    return row


def collect_reports(out: Path) -> None:
    rows = [json.loads((out / f"result_{name}.json").read_text(encoding="utf-8"))
            for name in FEATURE_SETS]
    folds = pd.DataFrame(rows)
    base = folds.loc[folds.experiment.eq(BASELINE), "rmsle"].iloc[0]
    summary = folds[["experiment", "feature_count", "rmsle", "best_iteration", "mode"]].copy()
    summary["delta_vs_base"] = summary.rmsle - float(base)
    save_csv(out / "backtest_fold_results.csv", folds)
    save_csv(out / "backtest_summary.csv", summary)

    day_rows = []
    for name in FEATURE_SETS:
        p = pd.read_csv(out / f"predictions_{name}.csv", parse_dates=["date"])
        daily = p.groupby("date", as_index=False).squared_log_error.mean()
        daily["rmsle"] = np.sqrt(daily.squared_log_error)
        daily["experiment"] = name
        day_rows.append(daily[["experiment", "date", "rmsle"]])
    save_csv(out / "backtest_daily_rmsle.csv", pd.concat(day_rows, ignore_index=True))
    print("\n2016 AUGUST BACKTEST (negative delta = improvement)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions-csv", type=Path,
                        default=ROOT / "data" / "transactions.csv")
    parser.add_argument("--oil-csv", type=Path, default=ROOT / "data" / "oil.csv")
    parser.add_argument("--mode", choices=["early_stop", "fixed"], default="early_stop",
                        help="early_stop reproduces CV; fixed leaves August labels untouched")
    parser.add_argument("--iterations-json", type=Path,
                        help="Required in fixed mode; preselected per-model iteration counts")
    parser.add_argument("--check-only", action="store_true", help="QC without model fitting")
    parser.add_argument("--run-dir", type=Path, help="Resume an existing result directory")
    args = parser.parse_args()
    assert_static_config()
    fixed = get_fixed_iterations(args.iterations_json, args.mode)
    train, valid = prepare_backtest(args.transactions_csv, args.oil_csv)
    for name, cols in FEATURE_SETS.items():
        make_xy(train, valid, cols)
        print(f"[CHECK] {name}: {len(cols)} features ready")
    if args.check_only:
        print("[CHECK COMPLETE] No models trained; no output files written")
        return

    out = (args.run_dir.expanduser().resolve() if args.run_dir else
           ROOT / "results" / "august_2016_backtest" /
           datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    config = dict(
        training_cutoff=str(TRAIN_END.date()), validation=[str(VALID_START.date()),
                                                           str(VALID_END.date())],
        feature_sets=FEATURE_SETS, params=MODEL_PARAMS,
        early_stopping_rounds=EARLY_STOPPING, mode=args.mode,
        fixed_iterations=fixed, raw_transactions=str(args.transactions_csv.resolve()),
        raw_oil=str(args.oil_csv.resolve()),
        policy="transactions strictly through train cutoff; oil through validation end",
        metric="RMSLE after clipping predicted log at zero",
    )
    if args.run_dir:
        path = out / "run_config.json"
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Refusing resume: run_config.json mismatch")
    else:
        out.mkdir(parents=True, exist_ok=False)
        save_json(out / "run_config.json", config)
    print(f"[OUTPUT] {out}")
    if args.mode == "early_stop":
        print("[CAUTION] August labels select best_iteration: diagnostic VALIDATION, "
              "not an untouched holdout.")
    else:
        print("[HOLDOUT] Frozen iteration counts; August labels are scoring-only. "
              "Feature design itself was informed by later project experiments.")

    for name in FEATURE_SETS:
        previous = out / f"result_{name}.json"
        if previous.is_file():
            old = json.loads(previous.read_text(encoding="utf-8"))
            if (old.get("experiment") != name or old.get("mode") != args.mode or
                    old.get("train_end") != str(TRAIN_END.date()) or
                    old.get("feature_count") != len(FEATURE_SETS[name]) or
                    (args.mode == "fixed" and old.get("best_iteration") != fixed[name]) or
                    not (out / f"predictions_{name}.csv").is_file()):
                raise ValueError(f"Cannot resume inconsistent result: {previous}")
            print(f"[SKIP] Previously completed: {name}")
            continue
        run_one(train, valid, name, args.mode, fixed, out)
    collect_reports(out)
    print("[FINISHED] Three 2016 August backtest models completed")


if __name__ == "__main__":
    cli()
