"""Fold 2/3 ablations against Trial17 on all stores and families."""

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

PROJECT_ROOT = Path(r"C:\projects\Store Sales-ML")
REFERENCE_DIR = (
    PROJECT_ROOT / "results" / "trial17_store_cv_28d_results" / "20261110_192821_456115"
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import main as original_pipeline
from feature_engineering import add_lag_features


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
RATIO_FEATURE = "store_sales_mean_ratio_7d_28d"
FAMILY_UPLIFT = "family_sales_uplift_7d_prev21d"
CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]
KEYS = ["store_nbr", "family"]
ROW_KEYS = ["date"] + KEYS
TARGET = "sales_log"
EXPECTED_FAMILIES = 33
EXPECTED_STORES = 54
EXPECTED_ROWS = 3_008_016
EXPECTED_VALID_ROWS = 28_512
MAX_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 150

EXPERIMENTS = {
    "store_ratio_only": [RATIO_FEATURE],
    "family_uplift_only": [FAMILY_UPLIFT],
    "store_ratio_and_family_uplift": [RATIO_FEATURE, FAMILY_UPLIFT],
}
FEATURE_SETS = {model: BASE_FEATURES + added_features for model, added_features in EXPERIMENTS.items()}
SUBMISSION_STORES = {4, 9, 11, 18, 21, 25, 31, 37, 47, 51, 52}

TRIAL17_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "learning_rate": 0.023127655409786754,
    "n_estimators": 8000,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "min_split_gain": 0.0,
    "subsample": 0.7742745551930794,
    "subsample_freq": 1,
    "colsample_bytree": 0.9621753805232081,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "metric": "rmse",  # unchanged: un-clipped log RMSE for early stopping
    "importance_type": "split",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

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
}


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def save_model(model: lgb.LGBMRegressor, path: Path) -> None:
    """Save only the best boosting iterations (same as source script)."""
    temp = path.with_suffix(".txt.tmp")
    best_iteration = int(model.best_iteration_)
    content = model.booster_.model_to_string(num_iteration=best_iteration)
    temp.write_text(content, encoding="utf-8")
    restored = lgb.Booster(model_str=temp.read_text(encoding="utf-8"))
    if restored.num_trees() != best_iteration:
        raise ValueError("Saved LightGBM model iteration count mismatch")
    temp.replace(path)


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "sales"] + BASE23
    absent = [col for col in required if col not in raw.columns]
    if absent:
        raise ValueError(f"Required original main.py columns missing: {absent}")

    df = raw.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"])

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected restored rows: {len(df):,} (expected {EXPECTED_ROWS:,})")
    if (df["date"].min() != pd.Timestamp("2013-01-01") or
            df["date"].max() != pd.Timestamp("2017-08-15")):
        raise ValueError("Unexpected restored training date range")
    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store/family/date in preprocessed data")
    if df[BASE23].isna().any().any():
        raise ValueError("Unexpected NaNs in Base23 columns")
    if df["family"].nunique() != EXPECTED_FAMILIES:
        raise ValueError(f"Expected {EXPECTED_FAMILIES} families")
    if df["sales"].isna().any() or (df["sales"] < 0).any():
        raise ValueError("Sales must be present and nonnegative")

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    df = add_lag_features(df, lags=LAGS)  # EXACT ORIGINAL function
    for lag in LAGS:
        if f"sales_lag_{lag}" not in df.columns:
            raise ValueError(f"Lag{lag} not created by feature_engineering.py")

    df[TARGET] = np.log1p(df["sales"])
    if len(BASE_FEATURES) != 25:
        raise AssertionError("Expected 25 Trial17 features")
    return df


def add_store_sales_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Return store ratios indexed by the first day excluded from the history."""
    last_train_end = max(pd.Timestamp(spec["train_end"]) for spec in SPECS.values())
    history = df.loc[df["date"].le(last_train_end), ["date", "store_nbr", "sales"]]
    grouped = history.groupby(["date", "store_nbr"], observed=True)["sales"]
    if not grouped.size().eq(EXPECTED_FAMILIES).all():
        raise ValueError("Store sales totals require one row per family and date")
    totals = grouped.sum().unstack("store_nbr")
    dates = pd.date_range(totals.index.min(), last_train_end + pd.Timedelta(days=1))
    history_only = totals.reindex(dates).shift(1)

    mean_7d = history_only.rolling(7, min_periods=7).mean()
    mean_28d = history_only.rolling(28, min_periods=28).mean()
    ratio = (mean_7d / mean_28d.where(mean_28d.gt(0))) - 1
    by_date_store = ratio.rename_axis(index="date", columns="store_nbr").stack()
    keys = pd.MultiIndex.from_frame(df[["date", "store_nbr"]])
    df[RATIO_FEATURE] = by_date_store.reindex(keys).to_numpy(dtype=np.float64)
    return ratio


def add_family_sales_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Return family uplift indexed by the first day excluded from its history."""
    last_train_end = max(pd.Timestamp(spec["train_end"]) for spec in SPECS.values())
    history = df.loc[df["date"].le(last_train_end), ["date", "family", "sales"]]
    grouped = history.groupby(["date", "family"], observed=True)["sales"]
    if not grouped.size().eq(EXPECTED_STORES).all():
        raise ValueError("Family sales totals require one row per store and date")
    totals = grouped.sum().unstack("family")
    dates = pd.date_range(totals.index.min(), last_train_end + pd.Timedelta(days=1))
    totals = totals.reindex(dates)

    recent_7d = totals.shift(1).rolling(7, min_periods=7).mean()
    previous_21d = totals.shift(8).rolling(21, min_periods=21).mean()
    uplift = np.log1p(recent_7d) - np.log1p(previous_21d)
    by_date_family = uplift.rename_axis(index="date", columns="family").stack()
    keys = pd.MultiIndex.from_frame(df[["date", "family"]])
    df[FAMILY_UPLIFT] = by_date_family.reindex(keys).to_numpy(dtype=np.float64)
    return uplift


def make_split(
    df: pd.DataFrame, spec: dict, store_ratio: pd.DataFrame, family_uplift: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(spec["train_end"])
    start = pd.Timestamp(spec["valid_start"])
    end = pd.Timestamp(spec["valid_end"])
    if start != cutoff + pd.Timedelta(days=1) or (end - start).days + 1 != 16:
        raise ValueError("Invalid expanding-window 16-day validation")

    train = df.loc[df["date"] <= cutoff]
    valid = df.loc[df["date"].between(start, end)].copy()
    if len(valid) != EXPECTED_VALID_ROWS or valid["family"].nunique() != EXPECTED_FAMILIES:
        raise ValueError("Invalid validation row count or missing families")

    for lag in LAGS:
        name = f"sales_lag_{lag}"
        if end - pd.Timedelta(days=lag) > cutoff:
            raise ValueError(f"Validation {name} would leak validation target sales")
        if valid[name].isna().any():
            raise ValueError(f"Missing validation {name}")

    valid[RATIO_FEATURE] = valid["store_nbr"].map(store_ratio.loc[start]).to_numpy(dtype=np.float64)
    valid[FAMILY_UPLIFT] = valid["family"].map(family_uplift.loc[start]).to_numpy(dtype=np.float64)
    return train, valid


def make_matrices(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xt = train[features].copy()
    xv = valid[features].copy()
    for col in CATEGORICAL:
        categories = pd.Index(train[col].dropna().unique())
        xt[col] = pd.Categorical(xt[col], categories=categories)
        xv[col] = pd.Categorical(xv[col], categories=categories)
        if xv[col].isna().any():
            raise ValueError(f"Unknown categorical levels in validation: {col}")
    if any("oil" in name.lower() or "dcoil" in name.lower() for name in features):
        raise AssertionError("Oil feature included unexpectedly")
    if any("promo_high_response" in name for name in features):
        raise AssertionError("promo_high_response is NOT allowed")
    return xt, xv


def metrics(y_sales: np.ndarray, pred_log: np.ndarray) -> dict:
    y = np.asarray(y_sales, dtype=np.float64)
    p = np.asarray(pred_log, dtype=np.float64)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid data to score")
    if (y < 0).any():
        raise ValueError("Negative true sales")
    true_log = np.log1p(y)
    clipped_log = np.maximum(p, 0.0)
    error = clipped_log - true_log  # positive=overprediction, negative=underprediction
    return {
        "n_rows": len(y),
        "rmsle": float(np.sqrt(np.mean(error ** 2))),
        "mean_log_error": float(np.mean(error)),
        "rmse_log_unclipped": float(np.sqrt(np.mean((p - true_log) ** 2))),
    }


def progress(env) -> None:
    step = env.iteration + 1
    if step % 500 == 0:
        print(f"  iteration {step}/{env.end_iteration}", flush=True)


def train_one(
    train: pd.DataFrame, valid: pd.DataFrame,
    fold: str, model_name: str, output: Path,
    save_models: bool = True,
) -> tuple[np.ndarray, dict]:
    features = FEATURE_SETS[model_name]
    xt, xv = make_matrices(train, valid, features)
    print(f"\n[{fold.upper()} / {model_name}] "
          f"train={len(train):,}, valid={len(valid):,}, features={len(features)}",
          flush=True)

    model = lgb.LGBMRegressor(**TRIAL17_PARAMS)
    model.fit(
        xt,
        train[TARGET],
        eval_set=[(xv, valid[TARGET])],
        eval_names=["valid"],
        eval_metric="rmse",  # UNCLIPPED log RMSE, original early stopping
        categorical_feature=CATEGORICAL,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True, verbose=True,
            ),
            progress,
        ],
    )

    best = int(model.best_iteration_)
    if not 1 <= best <= MAX_ESTIMATORS:
        raise ValueError(f"Invalid best_iteration_ for {fold} {model_name}: {best}")
    pred_log = np.asarray(model.predict(xv, num_iteration=best), dtype=np.float64)
    scored = metrics(valid["sales"].to_numpy(), pred_log)
    result = {
        "fold": fold,
        "model": model_name,
        "n_features": len(features),
        "n_train": len(train),
        "n_valid": len(valid),
        "best_iteration": best,
        **scored,
    }
    print(f"[RESULT] best_iteration={best}; "
          f"RMSLE={scored['rmsle']:.6f}; "
          f"mean_log_error={scored['mean_log_error']:+.6f}; "
          f"raw_log_RMSE={scored['rmse_log_unclipped']:.6f}", flush=True)
    if best > MAX_ESTIMATORS - EARLY_STOPPING_ROUNDS:
        print("[NOTE] Best iteration is near max rounds", flush=True)

    if save_models:
        save_model(model, output / f"{fold}_{model_name}_model.txt")

    importance = pd.DataFrame({
        "feature": features,
        "gain": model.booster_.feature_importance("gain"),
        "split": model.booster_.feature_importance("split"),
    })
    save_csv(importance.sort_values("gain", ascending=False),
             output / f"{fold}_{model_name}_importance.csv")

    del xt, xv, model, importance
    gc.collect()
    return pred_log, result


def scored_pair(
    y: np.ndarray, b: np.ndarray, e: np.ndarray, fold: str, **labels
) -> dict:
    base = metrics(y, b)
    exp = metrics(y, e)
    return {
        "fold": fold,
        **labels,
        "n_rows": len(y),
        "baseline_rmsle": base["rmsle"],
        "experiment_rmsle": exp["rmsle"],
        "delta_rmsle": exp["rmsle"] - base["rmsle"],
        "improvement_rmsle": base["rmsle"] - exp["rmsle"],
        "baseline_mean_log_error": base["mean_log_error"],
        "experiment_mean_log_error": exp["mean_log_error"],
        "delta_mean_log_error": exp["mean_log_error"] - base["mean_log_error"],
        "delta_abs_mean_log_error": abs(exp["mean_log_error"]) - abs(base["mean_log_error"]),
    }


def row_group_report(
    valid: pd.DataFrame, base: np.ndarray, exp: np.ndarray,
    fold: str, column: str,
) -> pd.DataFrame:
    """Position-safe group evaluation (same samples for baseline and test)."""
    actual = valid["sales"].to_numpy(dtype=np.float64)
    groups = valid.groupby(column, sort=True, observed=True).indices
    out = []
    for label, indices in groups.items():
        i = np.asarray(indices, dtype=np.int64)
        out.append(scored_pair(actual[i], base[i], exp[i], fold, **{column: label}))
    return pd.DataFrame(out)


def load_trial17_predictions(
    valid: pd.DataFrame, fold: str, reference_dir: Path, reference_scores: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    path = reference_dir / f"{fold}_paired_predictions.csv"
    checked_columns = ["sales", "onpromotion"]
    columns = ROW_KEYS + checked_columns + ["baseline_pred_log_unclipped"]
    saved = pd.read_csv(path, usecols=columns, parse_dates=["date"])
    if len(saved) != len(valid) or saved.duplicated(ROW_KEYS).any():
        raise ValueError(f"Invalid Trial17 prediction keys: {path}")
    saved = saved.set_index(ROW_KEYS).reindex(pd.MultiIndex.from_frame(valid[ROW_KEYS]))
    for col in checked_columns:
        if not np.allclose(saved[col], valid[col], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Trial17 predictions disagree with current {col}: {path}")
    pred = saved["baseline_pred_log_unclipped"].to_numpy(dtype=np.float64)
    scored = metrics(valid["sales"].to_numpy(), pred)
    reference = reference_scores.loc[fold]
    if not np.isclose(scored["rmsle"], reference["rmsle"], rtol=0, atol=1e-10):
        raise ValueError(f"Saved predictions do not reproduce the Trial17 RMSLE: {path}")
    return pred, {**reference.to_dict(), **scored, "fold": fold, "model": "trial17"}


RF_REFERENCE = PROJECT_ROOT / "results/trial17_cv_family721_oilraw_results/20261119_164707_823423"
CV = "store_sales_cv_28d"
FEATURES = BASE_FEATURES + [CV]
CANDIDATE_WEIGHT = 0.10


def routing(ref, cfg):
    store = ref.store_nbr.isin(cfg["store_cv"])
    family = ref.family.isin(cfg["family_cv"])
    uplift = ref.family.isin(cfg["family_uplift"])
    sw, fw = cfg["store_weight"], cfg["family_weight"]
    store_pred = np.where(store, ref.cv, ref.trial17)
    family_pred = np.select([family, uplift], [ref.cv, ref.uplift], default=ref.trial17)
    baseline = sw * np.log1p(store_pred) + fw * np.log1p(family_pred)
    np.testing.assert_allclose(baseline, np.log1p(ref.blend_original), atol=1e-9)
    return baseline, sw * store.astype(float) + fw * family.astype(float)


def make_ridge(features):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [col for col in features if col not in CATEGORICAL]
    numeric_steps = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(with_mean=False),
    )
    preprocess = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CATEGORICAL),
            ("numeric", numeric_steps, numeric),
        ],
        sparse_threshold=1.0,
    )
    return make_pipeline(preprocess, Ridge(alpha=1.0, solver="lsqr", tol=1e-5))


def candidate_log_predictions(method, train, valid, fold):
    print(f"[{fold}] {method.upper()} | {len(train):,} training rows", flush=True)
    if method == "ridge":
        estimator = make_ridge(FEATURES)
        estimator.fit(train[FEATURES], train[TARGET])
        prediction = estimator.predict(valid[FEATURES])
        if not np.isfinite(prediction).all():
            raise ValueError(f"{fold}: nonfinite Ridge prediction")
        return prediction.clip(min=0)

    from sklearn.ensemble import RandomForestRegressor
    x, v = train[FEATURES].copy(), valid[FEATURES].copy()
    for col in CATEGORICAL:
        levels = pd.Index(train[col].dropna().unique())
        x[col] = pd.Categorical(x[col], categories=levels).codes
        v[col] = pd.Categorical(v[col], categories=levels).codes
    # Missing historical lags/CV use -1; observed sales lags and CV are nonnegative.
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=30,
        max_features=0.5, max_samples=0.3, random_state=42, n_jobs=-1,
    )
    print(f"{fold}: training RF on {len(train):,} rows", flush=True)
    rf.fit(x.fillna(-1).to_numpy(dtype=np.float32), train[TARGET])
    return rf.predict(v.fillna(-1).to_numpy(dtype=np.float32)).clip(min=0)


def run_candidate_fold23(method, reference_dir, output_dir, check_only):
    os.chdir(PROJECT_ROOT)
    cfg = json.loads((reference_dir / "config.json").read_text(encoding="utf-8-sig"))
    refs = {}
    for fold in SPECS:
        ref = pd.read_csv(reference_dir / f"{fold}_predictions.csv", parse_dates=["date"])
        routing(ref, cfg)
        if len(ref) != EXPECTED_VALID_ROWS or ref.family.nunique() != EXPECTED_FAMILIES:
            raise ValueError(f"{fold}: incomplete all-family reference")
        refs[fold] = ref
    if check_only:
        print("Fold 2/3 references and original routing verified.")
        return

    raw, test, holidays = original_pipeline()
    del test, holidays
    df = prepare_features(raw)
    del raw
    gc.collect()
    daily = df.groupby(["date", "store_nbr"], observed=True).sales.sum().unstack()
    daily = daily.asfreq("D")
    history = daily.shift(1).rolling(28, min_periods=28)
    mean = history.mean()
    cv = history.std(ddof=1).div(mean.where(mean > 0))
    keys = pd.MultiIndex.from_frame(df[["date", "store_nbr"]])
    df[CV] = cv.stack().reindex(keys).to_numpy()

    output = output_dir or PROJECT_ROOT / f"results/{method}_cv_fold23" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    scores = []
    for fold, spec in SPECS.items():
        train = df.loc[df.date <= spec["train_end"]]
        valid = df.loc[df.date.between(spec["valid_start"], spec["valid_end"])].copy()
        valid[CV] = valid.store_nbr.map(cv.loc[spec["valid_start"]]).astype(float)
        ref = valid[ROW_KEYS + ["sales"]].merge(
            refs[fold], on=ROW_KEYS, how="left", validate="one_to_one",
            suffixes=("_pipeline", ""), sort=False,
        )
        assert len(ref) == len(refs[fold]) == EXPECTED_VALID_ROWS
        np.testing.assert_allclose(ref.sales, ref.sales_pipeline)
        baseline, cv_share = routing(ref, cfg)

        candidate_log = candidate_log_predictions(method, train, valid, fold)
        cv_log = np.log1p(ref.cv)
        mixed_cv = cv_log + CANDIDATE_WEIGHT * (candidate_log - cv_log)
        mixed_final = baseline + cv_share * (mixed_cv - cv_log)
        actual = np.log1p(ref.sales)
        for scope, mask, before, after in [
            (f"{method}_only_all_rows_vs_cv", np.ones(len(ref), dtype=bool), cv_log, candidate_log),
            (f"{method}_only_all_rows_vs_final", np.ones(len(ref), dtype=bool), baseline, candidate_log),
            ("cv_component", cv_share > 0, cv_log, mixed_cv),
            ("final_all_rows", np.ones(len(ref), dtype=bool), baseline, mixed_final),
        ]:
            old = np.sqrt(np.mean((before[mask] - actual[mask]) ** 2))
            new = np.sqrt(np.mean((after[mask] - actual[mask]) ** 2))
            scores.append(dict(fold=fold, scope=scope, rows=int(mask.sum()),
                               baseline_rmsle=old, candidate_rmsle=new, delta=new-old))
        pred = ref[ROW_KEYS + ["sales", "cv", "blend_original"]].copy()
        pred["cv_share"] = cv_share
        pred[f"{method}_sales"] = np.expm1(candidate_log)
        pred[f"final_{method}_blend"] = np.expm1(mixed_final)
        save_csv(pred, output / f"{fold}_predictions.csv")
        save_csv(pd.DataFrame(scores), output / "scores.csv")
        del train, valid
        gc.collect()
    print(pd.DataFrame(scores).to_string(index=False))
    print(output)



def fit_fixed_or_early(xt, yt, xv, yv, n_iter=None, extra_params=None, init=None, init_v=None):
    """n_iter=None: early stopping on (xv, yv). Otherwise fixed n_iter rounds (valid only logged).
    init/init_v: optional base-margin (e.g. Ridge log prediction); returned prediction includes it."""
    params = {**TRIAL17_PARAMS, **(extra_params or {})}
    callbacks = [lgb.log_evaluation(500)]
    if n_iter is None:
        callbacks.insert(0, lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False))
    else:
        params["n_estimators"] = int(n_iter)
    model = lgb.LGBMRegressor(**params)
    model.fit(xt, yt, init_score=init, eval_set=[(xv, yv)], eval_names=["valid"], eval_metric="rmse",
              eval_init_score=None if init_v is None else [init_v],
              categorical_feature=CATEGORICAL, callbacks=callbacks)
    best = int(n_iter) if n_iter is not None else int(model.best_iteration_ or MAX_ESTIMATORS)
    pred = model.predict(xv, num_iteration=best)
    return (pred if init_v is None else pred + init_v), best


def run_variant_fold23(variant, reference_dir, output_dir, iter_mode="early", ridge_features=25):
    """Single global LightGBM variant vs Trial17 on folds 2/3 (25 features, TRIAL17_PARAMS, no post-processing).

    variant: "linear_tree" (linear_tree=True, linear_lambda=1.0) or
             "ridge_residual" (Ridge log prediction as init_score; final = Ridge + trees, Ridge in-sample on train).
    iter_mode: "early" (early stopping on scored fold, like Trial17) or "cross" (Trial17's other-fold iteration).
    """
    import time
    os.chdir(PROJECT_ROOT)
    other = {"fold2": "fold3", "fold3": "fold2"}
    reference_scores = pd.read_csv(reference_dir / "model_scores_and_iterations.csv")
    reference_scores = reference_scores.loc[reference_scores["model"].eq("baseline")].set_index("fold")
    raw, test, holidays = original_pipeline()
    df = prepare_features(raw)
    del raw, test, holidays
    gc.collect()
    store_ratio = add_store_sales_ratio(df)
    family_uplift = add_family_sales_uplift(df)
    ridge_cols = BASE_FEATURES
    cv = None
    if variant == "ridge_residual" and ridge_features == 26:
        daily = df.groupby(["date", "store_nbr"], observed=True).sales.sum().unstack().asfreq("D")
        history = daily.shift(1).rolling(28, min_periods=28)
        mean = history.mean()
        cv = history.std(ddof=1).div(mean.where(mean > 0))
        df[CV] = cv.stack().reindex(pd.MultiIndex.from_frame(df[["date", "store_nbr"]])).to_numpy()
        ridge_cols = FEATURES

    extra = {"linear_tree": True, "linear_lambda": 1.0} if variant == "linear_tree" else {}
    output = output_dir or PROJECT_ROOT / f"results/{variant}_fold23" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    save_json({"variant": variant, "features": BASE_FEATURES, "params": {**TRIAL17_PARAMS, **extra},
               "iter_mode": iter_mode, "ridge_features": ridge_cols if variant == "ridge_residual" else None,
               "ridge": "alpha=1.0 lsqr, one-hot categoricals, median impute + indicator, scaler; in-sample on train",
               "folds": SPECS, "reference_dir": str(reference_dir), "post_processing": "none"},
              output / "config.json")
    print(f"{variant} | Fold 2 + Fold 3 | iter_mode={iter_mode} | output {output}", flush=True)

    overall = []
    for n_fold, (fold, spec) in enumerate(SPECS.items(), 1):
        t_fold = time.time()
        train, valid = make_split(df, spec, store_ratio, family_uplift)
        if cv is not None:
            valid[CV] = valid.store_nbr.map(cv.loc[spec["valid_start"]]).astype(float)
        base, base_score = load_trial17_predictions(valid, fold, reference_dir, reference_scores)
        print(f"\n[{fold.upper()} {n_fold}/2] Trial17 RMSLE={base_score['rmsle']:.6f} | "
              f"train={len(train):,} valid={len(valid):,}", flush=True)
        init = init_v = None
        ridge_score = None
        if variant == "ridge_residual":
            t0 = time.time()
            ridge = make_ridge(ridge_cols)
            ridge.fit(train[ridge_cols], train[TARGET])
            init = ridge.predict(train[ridge_cols])
            init_v = ridge.predict(valid[ridge_cols])
            ridge_score = metrics(valid["sales"].to_numpy(), init_v)["rmsle"]
            print(f"[{fold.upper()}] Ridge({len(ridge_cols)}) RMSLE={ridge_score:.6f} | "
                  f"train-period RMSE={np.sqrt(np.mean((init - train[TARGET]) ** 2)):.6f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        n_iter = int(reference_scores.loc[other[fold], "best_iteration"]) if iter_mode == "cross" else None
        if iter_mode == "cross":
            print(f"[{fold.upper()}] fixed {n_iter} iterations (Trial17 {other[fold]})", flush=True)
            # Fair baseline: Trial17 with the same cross-fold iteration (cached from the per-group rerun).
            cache = PROJECT_ROOT / "results/trial17_fixediter_cache" / f"{fold}_iter{n_iter}.npy"
            if not cache.exists():
                raise FileNotFoundError(f"Run the fixed-iteration Trial17 first: {cache}")
            base = np.load(cache)
            base_score = {**metrics(valid["sales"].to_numpy(), base), "best_iteration": n_iter}
            print(f"[{fold.upper()}] Trial17 fixed-iter RMSLE={base_score['rmsle']:.6f}", flush=True)
        xt, xv = make_matrices(train, valid, BASE_FEATURES)
        t0 = time.time()
        pred, best = fit_fixed_or_early(xt, train[TARGET], xv, valid[TARGET], n_iter, extra, init, init_v)
        del xt, xv
        gc.collect()
        actual = valid["sales"].to_numpy(dtype=np.float64)
        row = {**scored_pair(actual, base, pred, fold, model=variant),
               "baseline_best_iteration": base_score["best_iteration"], "experiment_best_iteration": best,
               "ridge_only_rmsle": ridge_score, "minutes": (time.time() - t_fold) / 60}
        overall.append(row)
        print(f"[{fold.upper()} DONE] {variant} RMSLE={row['experiment_rmsle']:.6f} | Trial17 "
              f"{row['baseline_rmsle']:.6f} | diff {row['delta_rmsle']:+.6f} | best_iter={best} | "
              f"train {(time.time() - t0) / 60:.1f} min", flush=True)
        groups = []
        for column in ["store_nbr", "family"]:
            report = row_group_report(valid, base, pred, fold, column)
            report.insert(1, "group_type", column)
            groups.append(report.rename(columns={column: "group"}))
        save_csv(pd.concat(groups, ignore_index=True), output / f"{fold}_group_comparison.csv")
        paired = valid[ROW_KEYS + ["sales", "onpromotion"]].copy()
        paired["baseline_pred_log_unclipped"] = base
        paired["experiment_pred_log_unclipped"] = pred
        if init_v is not None:
            paired["ridge_pred_log"] = init_v
        save_csv(paired, output / f"{fold}_paired_predictions.csv")
        save_csv(pd.DataFrame(overall), output / "overall_comparison.csv")
        del train, valid, paired
        gc.collect()
    print(pd.DataFrame(overall).round(6).to_string(index=False))
    print("[COMPLETE]", output, flush=True)


BT_SPECS = {
    "backtest1": {"train_end": "2016-07-30", "valid_start": "2016-07-31", "valid_end": "2016-08-15"},
    "backtest2": {"train_end": "2016-08-15", "valid_start": "2016-08-16", "valid_end": "2016-08-31"},
}


def run_group_fold23(group_col, reference_dir, output_dir, fixed_from=None, specs=None):
    """Per-group LightGBM (one model per store or family) vs global Trial17, folds 2/3.

    Only the training unit changes: same 25 features (constant ones kept), TRIAL17_PARAMS,
    early stopping on each group's own validation rows, no post-processing.
    fixed_from: earlier early-stopping run dir; each model (groups and global Trial17) then uses
    the best_iteration chosen on the OTHER fold, with no early stopping on the scored fold.
    """
    other = {"fold2": "fold3", "fold3": "fold2"}
    fixed_iters = None
    if fixed_from is not None:
        prev = pd.read_csv(fixed_from / "group_iterations.csv")
        fixed_iters = {(r.fold, r[group_col]): int(r.best_iteration) for _, r in prev.iterrows()}
    import time
    os.chdir(PROJECT_ROOT)
    reference_scores = pd.read_csv(reference_dir / "model_scores_and_iterations.csv")
    reference_scores = reference_scores.loc[reference_scores["model"].eq("baseline")].set_index("fold")
    raw, test, holidays = original_pipeline()
    df = prepare_features(raw)
    del raw, test, holidays
    gc.collect()
    store_ratio = add_store_sales_ratio(df)
    family_uplift = add_family_sales_uplift(df)

    tag = "store" if group_col == "store_nbr" else "family"
    retrain_base = specs is not None  # backtests: no saved Trial17 with identical settings -> retrain in-script
    specs = specs or SPECS
    suffix = ("_fixediter" if fixed_iters else "") + ("_backtest" if retrain_base else "_fold23")
    output = output_dir or PROJECT_ROOT / f"results/per_{tag}_lgbm{suffix}" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    save_json({"group_col": group_col, "features": BASE_FEATURES, "params": TRIAL17_PARAMS,
               "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "folds": specs,
               "trial17": "retrained in-script, early stopping on scored window" if retrain_base else "saved predictions",
               "reference_dir": str(reference_dir), "post_processing": "none",
               "iterations": (f"fixed: other fold's best_iteration from {fixed_from}; Trial17 global "
                              "retrained with other fold's best_iteration" if fixed_iters
                              else "early stopping on scored fold")}, output / "config.json")
    print(f"Per-{tag} LightGBM | Fold 2 + Fold 3 | output {output}", flush=True)

    overall, groups_out, iters = [], [], []
    for fold, spec in specs.items():
        t_fold = time.time()
        train, valid = make_split(df, spec, store_ratio, family_uplift)
        if retrain_base:
            print(f"\n[{fold.upper()}] Trial17 global train (early stopping) | train={len(train):,}", flush=True)
            xt, xv = make_matrices(train, valid, BASE_FEATURES)
            base, n_base = fit_fixed_or_early(xt, train[TARGET], xv, valid[TARGET])
            del xt, xv
            gc.collect()
            base_score = {**metrics(valid["sales"].to_numpy(), base), "best_iteration": n_base}
            print(f"[{fold.upper()}] Trial17 best_iter={n_base} | {(time.time() - t_fold) / 60:.1f} min", flush=True)
        else:
            base, base_score = load_trial17_predictions(valid, fold, reference_dir, reference_scores)
        print(f"\n[{fold.upper()}] Trial17 RMSLE={base_score['rmsle']:.6f}", flush=True)
        if fixed_iters:
            n_base = int(reference_scores.loc[other[fold], "best_iteration"])
            print(f"[{fold.upper()}] Trial17 global retrain, fixed {n_base} iterations "
                  f"(from {other[fold]})", flush=True)
            t0 = time.time()
            cache = PROJECT_ROOT / "results/trial17_fixediter_cache" / f"{fold}_iter{n_base}.npy"
            if cache.exists():
                base = np.load(cache)
                print(f"[{fold.upper()}] reuse cached Trial17 fixed-iter predictions {cache}", flush=True)
            else:
                xt, xv = make_matrices(train, valid, BASE_FEATURES)
                base, _ = fit_fixed_or_early(xt, train[TARGET], xv, valid[TARGET], n_base)
                del xt, xv
                gc.collect()
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, base)
            base_score = {**metrics(valid["sales"].to_numpy(), base), "best_iteration": n_base}
            print(f"[{fold.upper()}] Trial17 fixed-iter RMSLE={base_score['rmsle']:.6f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        pred = np.full(len(valid), np.nan)
        labels = sorted(valid[group_col].unique())
        for n, label in enumerate(labels, 1):
            t0 = time.time()
            tr = train.loc[train[group_col].eq(label)]
            vmask = valid[group_col].eq(label).to_numpy()
            va = valid.loc[vmask]
            xt, xv = make_matrices(tr, va, BASE_FEATURES)
            print(f"[{fold.upper()} {tag} {n}/{len(labels)}: {label}] train={len(tr):,} valid={len(va):,}", flush=True)
            n_fixed = fixed_iters[(other[fold], label)] if fixed_iters else None
            pred[vmask], best = fit_fixed_or_early(xt, tr[TARGET], xv, va[TARGET], n_fixed)
            g_base = metrics(va["sales"].to_numpy(), base[vmask])["rmsle"]
            g_exp = metrics(va["sales"].to_numpy(), pred[vmask])["rmsle"]
            iters.append(dict(fold=fold, **{group_col: label}, best_iteration=best, seconds=time.time() - t0))
            print(f"  best_iter={best} | RMSLE group {g_exp:.6f} (Trial17 {g_base:.6f}, "
                  f"diff {g_exp - g_base:+.6f}) | {time.time() - t0:.1f}s | fold elapsed {time.time() - t_fold:.0f}s",
                  flush=True)
            del xt, xv, tr, va
            gc.collect()
        actual = valid["sales"].to_numpy(dtype=np.float64)
        overall.append({**scored_pair(actual, base, pred, fold, model=f"per_{tag}"),
                        "baseline_best_iteration": base_score["best_iteration"],
                        "blend50_rmsle": metrics(actual, 0.5 * base + 0.5 * pred)["rmsle"],
                        "minutes": (time.time() - t_fold) / 60})
        for column in ["store_nbr", "family"]:
            report = row_group_report(valid, base, pred, fold, column)
            report.insert(1, "group_type", column)
            report = report.rename(columns={column: "group"})
            groups_out.append(report)
        paired = valid[ROW_KEYS + ["sales", "onpromotion"]].copy()
        paired["baseline_pred_log_unclipped"] = base
        paired["experiment_pred_log_unclipped"] = pred
        save_csv(paired, output / f"{fold}_paired_predictions.csv")
        save_csv(pd.DataFrame(overall), output / "overall_comparison.csv")
        save_csv(pd.concat(groups_out, ignore_index=True), output / "group_comparison.csv")
        save_csv(pd.DataFrame(iters), output / "group_iterations.csv")
        print(f"[{fold.upper()} DONE] blend50 RMSLE={overall[-1]['blend50_rmsle']:.6f}", flush=True)
        print(f"[{fold.upper()} DONE] per_{tag} RMSLE={overall[-1]['experiment_rmsle']:.6f} | "
              f"Trial17 {overall[-1]['baseline_rmsle']:.6f} | diff {overall[-1]['delta_rmsle']:+.6f} | "
              f"{overall[-1]['minutes']:.1f} min", flush=True)
        del train, valid, paired
        gc.collect()
    print(pd.DataFrame(overall).round(6).to_string(index=False))
    print("[COMPLETE]", output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--ridge-cv-fold23", action="store_true", help="Evaluate Ridge on folds 2 and 3")
    modes.add_argument("--rf-cv-fold23", action="store_true",
                        help="Compare standalone RF and the fixed routed blend on folds 2/3")
    modes.add_argument("--group-fold23", choices=["store_nbr", "family"],
                        help="One Trial17 LightGBM per store or per family on folds 2/3")
    parser.add_argument("--fixed-iter-from", type=Path,
                        help="With --group-fold23: earlier run dir; use the other fold's best_iteration")
    modes.add_argument("--variant-fold23", choices=["linear_tree", "ridge_residual"],
                        help="Single global LightGBM variant vs Trial17 on folds 2/3")
    parser.add_argument("--iter-mode", choices=["early", "cross"], default="early")
    parser.add_argument("--backtests", nargs="+", choices=list(BT_SPECS),
                        help="With --group-fold23: run on 2016 backtests instead (Trial17 retrained in-script)")
    parser.add_argument("--ridge-features", type=int, choices=[25, 26], default=25)
    args = parser.parse_args()
    if args.variant_fold23:
        run_variant_fold23(args.variant_fold23, (args.reference_dir or REFERENCE_DIR).resolve(),
                           args.output_dir.resolve() if args.output_dir else None,
                           args.iter_mode, args.ridge_features)
        return
    if args.group_fold23:
        run_group_fold23(args.group_fold23, (args.reference_dir or REFERENCE_DIR).resolve(),
                         args.output_dir.resolve() if args.output_dir else None,
                         args.fixed_iter_from.resolve() if args.fixed_iter_from else None,
                         {k: BT_SPECS[k] for k in args.backtests} if args.backtests else None)
        return
    if args.rf_cv_fold23 or args.ridge_cv_fold23:
        method = "ridge" if args.ridge_cv_fold23 else "rf"
        run_candidate_fold23(method, (args.reference_dir or RF_REFERENCE).resolve(),
                      args.output_dir.resolve() if args.output_dir else None, args.check_only)
        return
    reference_dir = (args.reference_dir or REFERENCE_DIR).resolve()
    output = (
        args.output_dir.resolve() if args.output_dir is not None
        else PROJECT_ROOT / "results" / "trial17_single_feature_results"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    print("Fold 2 + Fold 3 | Trial17 feature experiments", flush=True)
    for model, added_features in EXPERIMENTS.items():
        print(f"  {model}: Base25 + {' + '.join(added_features)}", flush=True)
    print("Output:", output, flush=True)

    reference_scores = pd.read_csv(reference_dir / "model_scores_and_iterations.csv")
    reference_scores = reference_scores.loc[reference_scores["model"].eq("baseline")].set_index("fold")
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
    raw, test, holidays = original_pipeline()
    df = prepare_features(raw)
    del raw, test, holidays
    gc.collect()

    store_ratio = add_store_sales_ratio(df)
    family_uplift = add_family_sales_uplift(df)
    if not args.check_only:
        save_json({
            "reference_dir": str(reference_dir),
            "features": {"trial17": BASE_FEATURES, **FEATURE_SETS},
            "params": TRIAL17_PARAMS,
            "folds": SPECS,
            "validation_features": "Both features use only sales before the fold and are frozen for 16 days.",
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_metric": "unclipped log RMSE",
            "reported_metric": "clipped-sales RMSLE",
            "submission_stores": sorted(SUBMISSION_STORES),
        }, output / "config.json")

    model_scores = []
    overall = []
    store_reports = []
    family_reports = []
    selected_reports = []
    baseline_scores = {}
    columns = [
        "fold", "model", "baseline_rmsle", "experiment_rmsle", "delta_rmsle",
        "baseline_best_iteration", "experiment_best_iteration",
    ]
    for model, added_features in EXPERIMENTS.items():
        print(f"\n[EXPERIMENT] {model}", flush=True)
        for fold, spec in SPECS.items():
            train, valid = make_split(df, spec, store_ratio, family_uplift)
            base, baseline_score = load_trial17_predictions(valid, fold, reference_dir, reference_scores)
            print(f"[CHECK] {fold}: {len(train):,} train / {len(valid):,} valid rows; "
                  f"Trial17 RMSLE={baseline_score['rmsle']:.6f}; "
                  f"feature coverage={valid[added_features].notna().sum().to_dict()}",
                  flush=True)
            if args.check_only:
                continue

            baseline_scores[fold] = baseline_score
            actual = valid["sales"].to_numpy(dtype=np.float64)
            selected = valid["store_nbr"].isin(SUBMISSION_STORES).to_numpy()
            pred, score = train_one(train, valid, fold, model, output, save_models=not args.no_save_models)
            model_scores.append(score)
            overall.append({
                **scored_pair(actual, base, pred, fold, model=model, feature=" + ".join(added_features)),
                "baseline_best_iteration": baseline_score["best_iteration"],
                "experiment_best_iteration": score["best_iteration"],
            })
            for column, reports, filename in [
                ("store_nbr", store_reports, "store_comparison"),
                ("family", family_reports, "family_comparison"),
            ]:
                report = row_group_report(valid, base, pred, fold, column)
                report.insert(1, "model", model)
                reports.append(report)
                save_csv(report, output / f"{fold}_{model}_{filename}.csv")

            selected_reports.append(scored_pair(actual[selected], base[selected], pred[selected], fold, model=model))
            paired = valid[ROW_KEYS + ["sales", "onpromotion"] + added_features].copy()
            paired["baseline_pred_log_unclipped"] = base
            paired["experiment_pred_log_unclipped"] = pred
            paired["baseline_pred_sales"] = np.expm1(np.maximum(base, 0))
            paired["experiment_pred_sales"] = np.expm1(np.maximum(pred, 0))
            paired["baseline_log_error"] = np.maximum(base, 0) - np.log1p(actual)
            paired["experiment_log_error"] = np.maximum(pred, 0) - np.log1p(actual)
            save_csv(paired, output / f"{fold}_{model}_paired_predictions.csv")
            del train, valid, paired
            gc.collect()

        if args.check_only:
            continue

        comparison = pd.DataFrame(overall)
        save_csv(comparison, output / "overall_comparison.csv")
        save_csv(pd.DataFrame([*baseline_scores.values(), *model_scores]), output / "model_scores_and_iterations.csv")
        save_csv(pd.concat(store_reports, ignore_index=True), output / "store_comparison.csv")
        save_csv(pd.concat(family_reports, ignore_index=True), output / "family_comparison.csv")
        save_csv(pd.DataFrame(selected_reports), output / "submission_11_stores_comparison.csv")
        print(f"\n[RESULTS SAVED] {model} | {output}", flush=True)
        experiment_summary = comparison.loc[comparison["model"].eq(model), columns]
        print(experiment_summary.round(6).to_string(index=False), flush=True)

    if args.check_only:
        print("[CHECK OK] Both fold references and feature generation verified.", flush=True)
        return

    print("[COMPLETE]", output, flush=True)


if __name__ == "__main__":
    main()
