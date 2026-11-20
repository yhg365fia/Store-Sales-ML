"""Trial17 Zero Sales + promotion-safe post-processing: Folds 1-3 and August 2016.

Place as C:/projects/Store Sales-ML/src/experiment.py. Run from project root:
    python "src/experiment.py"

Existing Trial17 fold2/fold3 baseline predictions are read from --reference-dir.
Missing fold1 and 2016 August backtest baseline predictions are TRAINED once
with the original Trial17 25 features, Optuna parameters and early stopping.
They are stored in --cache-dir so subsequent runs do not retrain them.
Only predictions are changed by the 16/28/60/140-day zero-sales rules;
if ANY future-day promotion exists for a store/family, promo-safe policies
preserve all 16 original predictions for that series.

Fold1-3 are pooled into oof_comparison.csv; backtest is reported separately.
Validation/test labels never contribute to zero-history or promotion masks.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_DIR = (
    ROOT / "results" / "trial17_store_cv_28d_results" / "20261110_192821_456115"
)

KEYS = ["store_nbr", "family"]
ROW_KEYS = ["date", *KEYS]
WINDOWS = (16, 28, 60, 140)

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
FEATURES = BASE23 + ["sales_lag_364", "sales_lag_28"]
CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]
TRIAL17_PARAMS = {
    "boosting_type": "gbdt", "objective": "regression",
    "learning_rate": 0.023127655409786754,
    "n_estimators": 8000, "num_leaves": 127,
    "max_depth": -1, "min_child_samples": 50, "min_split_gain": 0.0,
    "subsample": 0.7742745551930794, "subsample_freq": 1,
    "colsample_bytree": 0.9621753805232081,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "metric": "rmse", "importance_type": "split",
    "random_state": 42, "n_jobs": -1, "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 150
DEFAULT_CACHE_DIR = ROOT / "results" / "zero_sales_postprocess" / "trial17_cached_missing_splits"

# Fold2/Fold3 use existing cached predictions; missing split baselines are fitted once.
SPLITS = {
    "fold1": ("2017-06-29", "2017-07-14"),
    "fold2": ("2017-07-15", "2017-07-30"),
    "fold3": ("2017-07-31", "2017-08-15"),
    "backtest": ("2016-08-16", "2016-08-31"),
}


def rmsle(y: np.ndarray, prediction_log: np.ndarray) -> float:
    """Exactly log1p-sales RMSLE after clipping negative sales predictions."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction_log, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("RMSLE inputs must be finite")
    if (y < 0).any():
        raise ValueError("True sales cannot be negative")
    return float(np.sqrt(np.mean((np.log1p(y) - np.maximum(p, 0.0)) ** 2)))


def load_history() -> pd.DataFrame:
    """Use the original project's restored daily sales grid (main.py)."""
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)  # project main.py may use relative data paths
    from main import main as existing_main

    train, test, holidays = existing_main()
    required = ROW_KEYS + ["sales", *BASE23]
    missing = sorted(set(required) - set(train.columns))
    if missing:
        raise ValueError(f"main.py train is missing columns: {missing}")

    # Keep original features for fitting missing baseline splits; Zero Sales
    # masks themselves use only past date/store/family/sales.
    hist = train.loc[:, list(dict.fromkeys(required))].copy()
    del train, test, holidays
    gc.collect()

    hist["date"] = pd.to_datetime(hist["date"])
    hist["sales"] = pd.to_numeric(hist["sales"], errors="raise")
    if hist["sales"].isna().any() or (hist["sales"] < 0).any():
        raise ValueError("Sales in history must be nonmissing and nonnegative")
    if hist.duplicated(ROW_KEYS).any():
        raise ValueError("History has duplicated date/store/family rows")
    if not hist["date"].between("2013-01-01", "2017-08-15").all():
        raise ValueError("Unexpected history date range")

    print(f"[HISTORY] {len(hist):,} rows; {hist['date'].min().date()} .. {hist['date'].max().date()}")
    return hist



def generate_missing_baselines(history: pd.DataFrame,
                               missing: list[tuple[str, tuple[str, str], Path]],
                               cache_dir: Path) -> None:
    """Reproduce Trial17 baseline only for unavailable splits.

    Important: previous fold2/3 cached baseline predictions are never
    retrained/changed here. The original source uses 25 Base23+Lag364+Lag28
    features and LightGBM early stopping on un-clipped log RMSE.
    """
    if not missing:
        return
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "Missing splits require LightGBM: pip install lightgbm"
        ) from exc

    from feature_engineering import add_lag_features

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("\n[TRAIN MISSING] Building original Trial17 lag features once ...", flush=True)
    # The legacy add_lag_features() expects time-sorted store/family series.
    df = history.sort_values([*KEYS, "date"]).reset_index(drop=True).copy()
    df = add_lag_features(df, lags=LAGS)
    missing_cols = sorted(set(FEATURES) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Lag feature builder did not create {missing_cols}")
    df["sales_log"] = np.log1p(df["sales"])

    for name, dates, target_path in missing:
        start, end = (pd.Timestamp(s) for s in dates)
        cutoff = start - pd.Timedelta(days=1)
        print(f"\n[TRAIN {name.upper()}] train <= {cutoff.date()} "
              f"| valid={start.date()}..{end.date()} | 25 Trial17 features",
              flush=True)
        for lag in LAGS:
            if end - pd.Timedelta(days=lag) > cutoff:
                raise AssertionError(f"Lag {lag} uses unknown validation sales")
        tr = df.loc[df["date"] <= cutoff]
        va = df.loc[df["date"].between(start, end)]
        if len(va) != 54 * 33 * 16:
            raise ValueError(f"Unexpected {name} validation rows: {len(va):,}")
        if va[["sales_lag_28", "sales_lag_364"]].isna().any().any():
            raise ValueError(f"Unexpected missing validation lags: {name}")

        xt, xv = tr[FEATURES].copy(), va[FEATURES].copy()
        for col in CATEGORICAL:
            cats = pd.Index(tr[col].dropna().unique())
            xt[col] = pd.Categorical(xt[col], categories=cats)
            xv[col] = pd.Categorical(xv[col], categories=cats)
            if xv[col].isna().any():
                raise ValueError(f"Unknown {col} category in {name} validation")

        model = lgb.LGBMRegressor(**TRIAL17_PARAMS)
        model.fit(
            xt, tr["sales_log"],
            eval_set=[(xv, va["sales_log"])],
            eval_names=["valid"], eval_metric="rmse",
            categorical_feature=CATEGORICAL,
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS,
                                   first_metric_only=True, verbose=True),
                lgb.log_evaluation(period=500),
            ],
        )
        best = int(model.best_iteration_)
        if not 1 <= best <= 8000:
            raise ValueError(f"Invalid best_iteration for {name}: {best}")
        predictions = np.asarray(model.predict(xv, num_iteration=best), dtype=np.float64)
        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite predictions for {name}")
        baseline_score = rmsle(va["sales"].to_numpy(dtype=float), predictions)
        result = va[ROW_KEYS + ["sales", "onpromotion"]].copy()
        result["baseline_pred_log_unclipped"] = predictions
        # Write atomically so interruption never leaves a partly valid cache.
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".csv.part")
        result.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, target_path)
        meta = {
            "split": name,
            "train_end": str(cutoff.date()),
            "valid_start": str(start.date()), "valid_end": str(end.date()),
            "n_train": len(tr), "n_valid": len(va),
            "best_iteration": best, "baseline_rmsle": baseline_score,
            "features": FEATURES, "parameters": TRIAL17_PARAMS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "note": "Trial17 Base23 + Lag364 + Lag28; log1p target; un-clipped log RMSE early stopping",
        }
        target_path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[CACHED {name.upper()}] best_iteration={best}, "
              f"RMSLE={baseline_score:.6f} => {target_path}", flush=True)
        del xt, xv, model, predictions, result, tr, va
        gc.collect()
    del df
    gc.collect()


def read_cached_fold(path: Path, start: pd.Timestamp, end: pd.Timestamp,
                     history: pd.DataFrame) -> pd.DataFrame:
    required = ROW_KEYS + ["sales", "onpromotion", "baseline_pred_log_unclipped"]
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path} has no columns: {missing}")

    # Promotion counts are required for the exception experiment: never
    # assume missing promotion data means there was no promotion.
    valid = pd.read_csv(path, usecols=required, parse_dates=["date"])
    valid["onpromotion"] = pd.to_numeric(valid["onpromotion"], errors="raise")
    if (not np.isfinite(valid["onpromotion"].to_numpy(dtype=float)).all()
            or (valid["onpromotion"] < 0).any()):
        raise ValueError(f"Promotion data is missing/invalid: {path}")
    if valid.duplicated(ROW_KEYS).any():
        raise ValueError(f"Duplicate keys in {path}")
    if not valid["date"].between(start, end).all():
        raise ValueError(f"Wrong validation dates in {path}")
    if valid["date"].nunique() != 16:
        raise ValueError(f"Validation must cover 16 dates: {path}")

    # Catch mismatched preprocessing or cached predictions before scoring.
    actual = history.loc[history["date"].between(start, end), ROW_KEYS + ["sales"]]
    check = valid[ROW_KEYS + ["sales"]].merge(
        actual, on=ROW_KEYS, how="left", validate="one_to_one", sort=False,
        suffixes=("_saved", "_history"), indicator=True
    )
    if (check["_merge"] != "both").any():
        raise ValueError(f"Saved validation keys do not match main.py: {path}")
    if not np.allclose(check["sales_saved"], check["sales_history"],
                       rtol=1e-11, atol=1e-11):
        raise ValueError(f"Saved sales disagree with current main.py: {path}")

    if valid["baseline_pred_log_unclipped"].isna().any():
        raise ValueError("Missing cached baseline predictions")
    if not np.isfinite(valid["baseline_pred_log_unclipped"].to_numpy()).all():
        raise ValueError("Nonfinite cached baseline predictions")
    return valid


def zero_mask(history: pd.DataFrame, valid: pd.DataFrame,
              start: pd.Timestamp, days: int) -> tuple[np.ndarray, int]:
    """True where ALL exactly preceding `days` calendar days were observed as zero.

    Only sales from strictly before the first validation/test day are examined.
    Neither the true validation targets nor any sales inside the 16-day
    forecast horizon participate in mask construction.
    """
    begin = start - pd.Timedelta(days=days)
    recent = history.loc[
        (history["date"] >= begin) & (history["date"] < start),
        ROW_KEYS + ["sales"]
    ]

    stats = recent.groupby(KEYS, observed=True).agg(
        n_rows=("sales", "size"),
        n_days=("date", "nunique"),
        n_nonmissing=("sales", "count"),
        sales_sum=("sales", "sum"),
    )
    eligible = stats.loc[
        stats["n_rows"].eq(days)
        & stats["n_days"].eq(days)
        & stats["n_nonmissing"].eq(days)
        & stats["sales_sum"].eq(0)
    ]

    idx = pd.MultiIndex.from_frame(valid[KEYS])
    mask = np.asarray(idx.isin(eligible.index), dtype=bool)
    return mask, len(eligible)


def forecast_promotion_mask(valid: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Flag every row in a series if ANY forecast day is promotional.

    Each (store_nbr, family) is expected to have a full 16-day horizon;
    thus promotion protection is intentionally series-wide, not day-wise.
    """
    promoted = valid.loc[valid["onpromotion"].gt(0), KEYS].drop_duplicates()
    protected_index = pd.MultiIndex.from_frame(promoted)
    row_index = pd.MultiIndex.from_frame(valid[KEYS])
    return np.asarray(row_index.isin(protected_index), dtype=bool), len(promoted)


def distinct_groups(valid: pd.DataFrame, mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    return int(valid.loc[mask, KEYS].drop_duplicates().shape[0])


def one_split(fold: str, dates: tuple[str, str], path: Path,
              history: pd.DataFrame, output: Path):
    start, end = map(pd.Timestamp, dates)
    valid = read_cached_fold(path, start, end, history)
    y = valid["sales"].to_numpy(dtype=np.float64)
    baseline = valid["baseline_pred_log_unclipped"].to_numpy(dtype=np.float64)
    baseline_score = rmsle(y, baseline)
    promo_mask, n_promoted_groups = forecast_promotion_mask(valid)
    print(f"\n[{fold.upper()}] baseline RMSLE={baseline_score:.6f} | valid={len(valid):,}")
    print(f"  [PROMO] promoted series={n_promoted_groups:,}; "
          f"protected forecast rows={promo_mask.sum():,}")

    # Cache masks: all experiments reuse exactly the same history and model
    # predictions. Each 'promo_safe' version modifies only the post-processing.
    zero_masks = {days: zero_mask(history, valid, start, days)
                  for days in WINDOWS}
    policies = [("baseline", None, False)]
    policies += [(f"zero_{days}", days, False) for days in WINDOWS]
    policies += [(f"zero_{days}_promo_safe", days, True) for days in WINDOWS]

    results = []
    changed_rows = []
    by_family = []
    all_preds: dict[str, np.ndarray] = {}
    log_actual = np.log1p(y)
    before_sqerr = (log_actual - np.maximum(baseline, 0.0)) ** 2

    for rule, days, promo_safe in policies:
        if days is None:
            initial = np.zeros(len(valid), dtype=bool)
            mask = initial
            n_series = 0
        else:
            initial, n_series = zero_masks[days]
            mask = initial & ~promo_mask if promo_safe else initial

        adjusted = baseline.copy()
        adjusted[mask] = 0.0  # log1p(0) = 0
        all_preds[rule] = adjusted
        score = rmsle(y, adjusted)
        changed = int(mask.sum())
        exempted = initial & promo_mask if promo_safe else np.zeros(len(valid), dtype=bool)
        after_sqerr = (log_actual - np.maximum(adjusted, 0.0)) ** 2

        results.append({
            "split": fold,
            "rule": rule,
            "n_validation_rows": int(len(valid)),
            "n_zero_groups": int(n_series),
            "n_changed_groups": distinct_groups(valid, mask),
            "n_changed_rows": changed,
            "n_changed_actual_nonzero": int(np.count_nonzero(y[mask] > 0)),
            "n_promo_exempt_groups": distinct_groups(valid, exempted),
            "n_promo_exempt_rows": int(exempted.sum()),
            "baseline_rmsle": baseline_score,
            "rmsle": score,
            "delta_vs_baseline": score - baseline_score,
            "changed_rows_baseline_rmsle": rmsle(y[mask], baseline[mask]),
            "changed_rows_rule_rmsle": rmsle(y[mask], adjusted[mask]),
            "changed_rows_sum_delta_log_sqerr": float(
                np.sum(after_sqerr[mask] - before_sqerr[mask])
            ),
        })

        print(f"  {rule:23s} | eligible_groups={n_series:4d} "
              f"| changed={changed:5d} | promo_exempt={exempted.sum():5d} "
              f"| true>0={np.count_nonzero(y[mask] > 0):5d} "
              f"| RMSLE={score:.6f} | Δ={score - baseline_score:+.6f}")

        if rule != "baseline" and changed:
            c = valid.loc[mask, ROW_KEYS + ["sales", "onpromotion"]].copy()
            c["split"] = fold
            c["rule"] = rule
            c["baseline_prediction"] = np.expm1(np.maximum(baseline[mask], 0.0))
            c["adjusted_prediction"] = 0.0
            c["delta_log_sqerr"] = after_sqerr[mask] - before_sqerr[mask]
            changed_rows.append(c)

        for family, row_idx in valid.groupby("family", observed=True).indices.items():
            i = np.asarray(row_idx, dtype=np.int64)
            base_family = rmsle(y[i], baseline[i])
            adjusted_family = rmsle(y[i], adjusted[i])
            by_family.append({
                "split": fold,
                "rule": rule,
                "family": family,
                "n_rows": len(i),
                "n_zeroed": int(mask[i].sum()),
                "n_promo_exempt_rows": int(exempted[i].sum()),
                "baseline_rmsle": base_family,
                "rmsle": adjusted_family,
                "delta_vs_baseline": adjusted_family - base_family,
            })

    if changed_rows:
        pd.concat(changed_rows, ignore_index=True).to_csv(
            output / f"{fold}_changed_rows.csv", index=False, encoding="utf-8-sig")

    return pd.DataFrame(results), pd.DataFrame(by_family), y, all_preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
                        help="Folder with fold2_paired_predictions.csv etc.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Where newly trained Fold1/backtest predictions are kept")
    parser.add_argument("--no-train-missing", action="store_true",
                        help="Only score available cached folds; do not fit missing baselines")
    parser.add_argument("--retrain-missing", action="store_true",
                        help="Rebuild fold1/backtest predictions in --cache-dir")
    args = parser.parse_args()

    reference_dir = args.reference_dir.expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(
            f"Trial17 cache directory not found: {reference_dir}\n"
            "Specify the actual directory with --reference-dir. No model will be retrained."
        )

    cache_dir = args.cache_dir.expanduser().resolve()
    requested = list(SPLITS.items())
    available = []
    missing = []
    for fold, dates in requested:
        ref_path = reference_dir / f"{fold}_paired_predictions.csv"
        cached = cache_dir / f"{fold}_paired_predictions.csv"
        # Always preserve original experiment caches for Folds 2 and 3.
        if ref_path.is_file():
            available.append((fold, dates, ref_path))
        elif cached.is_file() and not args.retrain_missing:
            available.append((fold, dates, cached))
        else:
            missing.append((fold, dates, cached))

    if args.no_train_missing and missing:
        print("[SKIPPED] Missing split caches:", ", ".join(x[0] for x in missing))
    elif missing:
        print("[RETRAIN REQUIRED]", ", ".join(x[0] for x in missing))

    if not available and (not missing or args.no_train_missing):
        raise FileNotFoundError("No usable reference/cache files. Remove --no-train-missing to fit missing splits")

    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              ROOT / "results" / "zero_sales_postprocess" /
              datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    print("[REFERENCE]", reference_dir)
    print("[CACHE]", cache_dir)
    print("[OUTPUT]", output)
    print("[POLICY] Zero 16/28/60/140 and promo-safe variants")
    print("[PROMO EXCEPTION] Any promotion in 16-day horizon => keep all 16 original predictions for that store/family")
    print("[POLICY] Train baseline ONLY for missing splits; no retraining for cached baseline predictions")
    print("[POLICY] All Zero Sales masks use pre-validation history")

    history = load_history()
    if missing and not args.no_train_missing:
        generate_missing_baselines(history, missing, cache_dir)
        available.extend(missing)
    available.sort(key=lambda x: list(SPLITS).index(x[0]))
    print("[SPLITS]", ", ".join(f"{fold} ({'reference' if path.parent == reference_dir else 'cached'})"
                                    for fold, _, path in available))
    comparisons, families = [], []
    pooled_y: list[np.ndarray] = []
    pooled_p: dict[str, list[np.ndarray]] = {"baseline": []}
    for days in WINDOWS:
        pooled_p[f"zero_{days}"] = []
        pooled_p[f"zero_{days}_promo_safe"] = []

    for fold, dates, path in available:
        comparison, family, y, all_preds = one_split(fold, dates, path, history, output)
        comparisons.append(comparison)
        families.append(family)
        if fold.startswith("fold"):
            pooled_y.append(y)
            for rule, pred in all_preds.items():
                pooled_p[rule].append(pred)

    comparison_df = pd.concat(comparisons, ignore_index=True)
    comparison_df.to_csv(output / "fold_comparison.csv", index=False, encoding="utf-8-sig")
    backtest_comparison = comparison_df.loc[comparison_df["split"].eq("backtest")].copy()
    if not backtest_comparison.empty:
        backtest_comparison.to_csv(output / "backtest_comparison.csv",
                                   index=False, encoding="utf-8-sig")
        print("\n[BACKTEST COMPARISON] (Aug 16-31, 2016; NOT pooled into OOF)")
        print(backtest_comparison[["rule", "n_changed_rows", "rmsle", "delta_vs_baseline"]]
              .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    pd.concat(families, ignore_index=True).to_csv(
        output / "family_comparison.csv", index=False, encoding="utf-8-sig")

    if pooled_y:
        pooled_actual = np.concatenate(pooled_y)
        pooled_base = rmsle(pooled_actual, np.concatenate(pooled_p["baseline"]))
        overall = []
        for rule, predictions in pooled_p.items():
            score = rmsle(pooled_actual, np.concatenate(predictions))
            changed = int(comparison_df.loc[
                comparison_df["split"].str.startswith("fold")
                & comparison_df["rule"].eq(rule), "n_changed_rows"
            ].sum())
            overall.append({
                "rule": rule,
                "n_validation_rows": len(pooled_actual),
                "n_changed_rows": changed,
                "baseline_rmsle": pooled_base,
                "rmsle": score,
                "delta_vs_baseline": score - pooled_base,
            })
        overall_df = pd.DataFrame(overall)
        overall_df.to_csv(output / "oof_comparison.csv", index=False, encoding="utf-8-sig")
        print("\n[OOF COMPARISON] (only cached folds, excluding backtest)")
        print(overall_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n[SAVED]", output)
    print("  fold_comparison.csv: per-fold RMSLE and changed rows")
    print("  oof_comparison.csv: pooled OOF RMSLE for available folds")
    print("  backtest_comparison.csv: August 2016 RMSLE if scored")
    print("  cache files: missing split Trial17 raw log predictions (reusable)")
    print("  family_comparison.csv: 33-family comparison")
    print("  *_changed_rows.csv: actual rows switched to zero, with per-row error effects")
    print("  promo_safe = keep original predictions for any series with a forecast-period promotion")
    missing = [fold for fold in SPLITS if fold not in [a[0] for a in available]]
    if missing:
        print("[NOTE] Unavailable splits were skipped:", ", ".join(missing))
        print("       No scores are claimed for skipped splits.")


if __name__ == "__main__":
    main()
