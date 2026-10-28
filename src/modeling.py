"""
Optuna: Base23 + sales_lag_364 + sales_lag_28 (25 features).
Save as src/modeling_optuna.py beside existing main.py/feature_engineering.py.
Existing preprocessing, restored daily panel, category handling and log target
are retained. Each trial evaluates all three original 2017 folds with ES 150.
Objective: arithmetic mean of three clipped-prediction RMSLE scores.
Trial 0 uses the original fixed parameters as a baseline.

The selected trial's mean CV best_iteration (rounded) fixes the 2016 August
backtest training rounds. NO backtest early stopping or parameter selection.
Since parameters and iterations were selected using 2017 data, the 2016
backtest remains diagnostic, not an independent historical holdout.

Default: 20 TOTAL completed trials, sequential (each is 3 LightGBM fits).
python src/modeling_optuna.py --check-only
python src/modeling_optuna.py --n-trials 20
python src/modeling_optuna.py --run-dir results/.../RUN --n-trials 30
Same run-dir resumes SQLite history and compatible per-fold prediction caches.
Do not launch concurrent processes for the same run-dir.
Outputs: study.db, trials.csv, best_params.json, cv_summary.csv,
backtest summary, and keyed validation predictions for every fitted fold.
This script does not create a Kaggle submission.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import inspect
import optuna
import lightgbm
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from main import main
from feature_engineering import add_lag_features


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results" / "base23_lag364_lag28_optuna"

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
FEATURE_SETS = {EXPERIMENT: FEATURES}
CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]
KEYS = ["store_nbr", "family"]
TARGET = "sales_log"
HORIZON = 16
EXPECTED_TRAIN_ROWS = 3_008_016
EXPECTED_VALID_ROWS = 28_512
N_SERIES = 54 * 33
N_ESTIMATORS = 8_000
EARLY_STOPPING_ROUNDS = 150

CV_FOLDS = [
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


def check_configuration() -> None:
    if len(BASE23) != 23 or len(set(BASE23)) != 23:
        raise ValueError("The Base23 list must contain 23 unique columns")
    if len(FEATURES) != 25 or len(set(FEATURES)) != 25:
        raise ValueError("Expected exactly Base23 + lag364 + lag28")
    if FEATURES != BASE23 + ["sales_lag_364", "sales_lag_28"]:
        raise ValueError("Unexpected feature configuration")

    for split in CV_FOLDS + [AUGUST_BACKTEST]:
        cutoff = pd.Timestamp(split["train_end"])
        start = pd.Timestamp(split["valid_start"])
        stop = pd.Timestamp(split["valid_end"])
        if start != cutoff + pd.Timedelta(days=1):
            raise ValueError(f"Nonadjacent cutoff: {split['name']}")
        if (stop - start).days + 1 != HORIZON:
            raise ValueError(f"Validation must be 16 days: {split['name']}")
        for lag in LAGS:
            if stop - pd.Timedelta(days=lag) > cutoff:
                raise ValueError(f"Future target would enter lag{lag}")


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    check_configuration()
    required = ["date", "sales"] + BASE23
    missing = [column for column in required if column not in raw.columns]
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
        raise ValueError("Missing calendar dates within one of the series")

    # Create all raw-sales lags ONCE and isolate each predictor set later.
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
    print(f"[DATA] Restored train={len(df):,}; "
          f"computed lags={LAGS}; initial lag NaNs retained.")
    print("[FEATURE SETS]")
    for name, cols in FEATURE_SETS.items():
        print(f"  {name}: {len(cols)} features")
    print("[POLICY] Optuna LightGBM; exactly Base23 + lag364 + lag28.")
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
            raise ValueError(f"Missing Valid lag{lag}: {split['name']}")
        if stop - pd.Timedelta(days=lag) > cutoff:
            raise ValueError(f"Leakage in lag{lag}: {split['name']}")
    return train, valid


def make_matrices(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train.loc[:, features].copy()
    x_valid = valid.loc[:, features].copy()
    for col in CATEGORICAL:
        categories = pd.Index(train[col].dropna().unique())
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_valid[col] = pd.Categorical(x_valid[col], categories=categories)
        if x_valid[col].isna().any():
            raise ValueError(f"Unseen validation category: {col}")
    if list(x_train.columns) != features or list(x_valid.columns) != features:
        raise ValueError("Train/Valid feature ordering changed")
    return x_train, x_valid


def rmsle(y_true: pd.Series, predicted_log: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predicted_log, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite model prediction")
    return float(np.sqrt(np.mean((np.log1p(y) - np.maximum(p, 0)) ** 2)))


def atomic_json(path: Path, record: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, data: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".tmp")
    data.to_csv(temp, index=False)
    temp.replace(path)


def data_sha256(df: pd.DataFrame) -> str:
    cols = list(dict.fromkeys(
        ["date", "sales", TARGET] +
        [name for features in FEATURE_SETS.values() for name in features]
    ))
    hashes = pd.util.hash_pandas_object(
        df.loc[:, cols], index=False
    ).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


SEARCH_SPACE = {
    "learning_rate": [0.02, 0.08],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [20, 50, 100, 200],
    "colsample_bytree": [0.7, 1.0],
    "subsample": [0.7, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0, 5.0, 10.0],
    "reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
}


def suggest_params(trial):
    result = {}
    for name, values in SEARCH_SPACE.items():
        if name == "learning_rate":
            result[name] = trial.suggest_float(name, *values, log=True)
        elif name in ("colsample_bytree", "subsample"):
            result[name] = trial.suggest_float(name, *values)
        else:
            result[name] = trial.suggest_categorical(name, values)
    return result


def build_params(sampled, max_rounds, threads):
    params = {**FIXED_PARAMS, **sampled}
    params.update(n_estimators=max_rounds, n_jobs=threads)
    params["subsample_freq"] = 1 if params["subsample"] < 1.0 else 0
    return params


def rounded_mean_iterations(records):
    if len(records) != 3:
        raise ValueError("Exactly three completed CV folds are required")
    values = [r["best_iteration"] for r in records]
    if any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("Invalid CV best_iteration")
    return int(np.floor(np.mean(values) + 0.5))


def evaluate_fold(df, split, params, folder, use_early_stopping, save_model=False):
    """Reuse only exact configuration-matched results in this validated run."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = ("fold_" + split["name"].split()[-1]
            if use_early_stopping else "august_2016_backtest")
    path = folder / f"{stem}.json"
    pred_path = folder / f"{stem}_predictions.csv"
    model_path = folder / f"{stem}_model.txt"
    expected = dict(experiment=EXPERIMENT, split=split, features=FEATURES,
                    parameters=params, early_stopping_used=use_early_stopping)
    if path.exists() and pred_path.exists() and (not save_model or model_path.exists()):
        record = json.loads(path.read_text(encoding="utf-8"))
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Incompatible cached result: {path}")
        if (not np.isfinite(record["rmsle"])
                or not 1 <= record["best_iteration"] <= params["n_estimators"]):
            raise ValueError(f"Invalid cached metrics: {path}")
        print(f"[CACHE] {split['name']}: RMSLE={record['rmsle']:.6f}", flush=True)
        return record

    train, valid = make_split(df, split)
    x_train, x_valid = make_matrices(train, valid, FEATURES)
    model = LGBMRegressor(**params)
    fit_args = dict(categorical_feature=CATEGORICAL)
    if use_early_stopping:
        # New LightGBM releases expose eval_X/eval_y; older versions use eval_set.
        if "eval_X" in inspect.signature(model.fit).parameters:
            fit_args.update(eval_X=x_valid, eval_y=valid[TARGET])
        else:
            fit_args["eval_set"] = [(x_valid, valid[TARGET])]
        fit_args.update(eval_metric="rmse", callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            log_evaluation(500),
        ])
    print(f"[FIT] {split['name']}; train={len(train):,}; "
          f"rounds={params['n_estimators']}; ES={use_early_stopping}", flush=True)
    # Backtest receives neither evaluation labels nor early-stopping callbacks.
    model.fit(x_train, train[TARGET], **fit_args)
    best_iteration = (int(model.best_iteration_) if use_early_stopping
                      else int(params["n_estimators"]))
    if model.booster_.feature_name() != FEATURES:
        raise ValueError("Unexpected fitted feature ordering")
    pred_log = np.asarray(model.predict(x_valid, num_iteration=best_iteration))
    score = rmsle(valid["sales"], pred_log)
    predictions = valid[["date"] + KEYS + ["sales"]].copy()
    predictions.rename(columns={"sales": "y_true"}, inplace=True)
    predictions["pred_log_raw"] = pred_log
    predictions["pred_sales"] = np.expm1(np.maximum(pred_log, 0))
    if not np.isfinite(predictions["pred_sales"]).all():
        raise ValueError("Nonfinite inverse-transformed predictions")
    atomic_csv(pred_path, predictions)
    importance = pd.DataFrame({
        "feature": FEATURES,
        "gain": model.booster_.feature_importance("gain", iteration=best_iteration),
    }).sort_values("gain", ascending=False)
    atomic_csv(folder / f"{stem}_importance.csv", importance)
    if save_model:
        temporary = model_path.with_suffix(".tmp")
        model.booster_.save_model(str(temporary), num_iteration=best_iteration)
        temporary.replace(model_path)
    record = {**expected, "feature_count": len(FEATURES),
              "train_rows": len(train), "valid_rows": len(valid),
              "best_iteration": best_iteration, "rmsle": score,
              "hit_iteration_cap": bool(use_early_stopping and best_iteration >= params["n_estimators"]),
              "predictions_file": str(pred_path.resolve())}
    atomic_json(path, record)
    print(f"[RESULT] {split['name']}: RMSLE={score:.6f}, "
          f"iterations={best_iteration}", flush=True)
    del model, train, valid, x_train, x_valid, predictions, importance, pred_log
    gc.collect()
    return record


def cache_folder(out, params):
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    return out / "cv_cache" / key


def source_fingerprints():
    paths = [Path(__file__).resolve(), ROOT / "src" / "main.py",
             ROOT / "src" / "feature_engineering.py"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(args):
    if args.n_trials < 1 or args.max_rounds < 1 or args.threads == 0:
        raise ValueError("Positive trial/round counts and nonzero threads required")
    os.chdir(ROOT)
    loaded = main()
    if not isinstance(loaded, (list, tuple)) or len(loaded) != 3:
        raise ValueError("Expected main() -> (train, test, holidays)")
    raw, test, holidays = loaded
    del loaded, test, holidays
    df = prepare_features(raw)
    del raw
    gc.collect()
    for split in CV_FOLDS + [AUGUST_BACKTEST]:
        train, valid = make_split(df, split)
        print(f"[QC] {split['name']}: {len(train):,} / {len(valid):,}")
        del train, valid
    if args.check_only:
        print("[CHECK ONLY] Passed. No training or study creation.")
        return
    config = dict(
        schema_version=1, experiment=EXPERIMENT, features=FEATURES,
        folds=CV_FOLDS, backtest=AUGUST_BACKTEST,
        fixed_params=build_params({}, args.max_rounds, args.threads),
        search_space=SEARCH_SPACE, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        objective="mean of 3-fold RMSLE, predictions clipped at zero in log space",
        early_stopping_metric="unclipped log-space RMSE (original pipeline)",
        backtest_rule="round(mean(best trial's 3 CV best_iterations)); no ES",
        data_sha256=data_sha256(df), source_sha256=source_fingerprints(),
        versions={"lightgbm": lightgbm.__version__, "optuna": optuna.__version__,
                  "pandas": pd.__version__, "numpy": np.__version__},
        save_models=args.save_models,
    )
    if args.run_dir:
        out = args.run_dir.expanduser().resolve()
        prior = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        if prior != config:
            raise ValueError("Resume rejected: data/code/settings/package versions differ")
    else:
        out = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "run_config.json", config)
    print(f"[OUTPUT] {out}", flush=True)
    print("[NOTE] 2016 backtest is diagnostic: parameters/rounds selected in 2017.")
    study = optuna.create_study(
        study_name=EXPERIMENT, direction="minimize",
        storage="sqlite:///" + (out / "study.db").as_posix(),
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    # Change the RNG stream on resume; SQLite restores history, not sampler RNG state.
    study.sampler = optuna.samplers.TPESampler(seed=42 + len(study.trials), n_startup_trials=5)
    if not study.trials:
        study.enqueue_trial({k: FIXED_PARAMS[k] for k in SEARCH_SPACE},
                            user_attrs={"role": "original_fixed_baseline"})

    def objective(trial):
        params = build_params(suggest_params(trial), args.max_rounds, args.threads)
        folder = cache_folder(out, params)
        trial.set_user_attr("cache_folder", str(folder.resolve()))
        trial.set_user_attr("full_params", params)
        records = []
        for split in CV_FOLDS:
            records.append(evaluate_fold(df, split, params, folder, True, args.save_models))
            trial.set_user_attr("fold_records", records)
        mean_score = float(np.mean([r["rmsle"] for r in records]))
        trial.set_user_attr("backtest_n_estimators", rounded_mean_iterations(records))
        return mean_score

    def checkpoint(study, trial):
        atomic_csv(out / "trials.csv", study.trials_dataframe())

    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.n_trials - completed)
    print(f"[STUDY] completed={completed}; additional trials={remaining}", flush=True)
    if remaining:
        # Sequential trials limit memory use; LightGBM uses args.threads internally.
        study.optimize(objective, n_trials=remaining, n_jobs=1,
                       callbacks=[checkpoint], gc_after_trial=True)
    best = study.best_trial
    params = best.user_attrs["full_params"]
    records = best.user_attrs["fold_records"]
    fixed_n = rounded_mean_iterations(records)
    scores = [r["rmsle"] for r in records]
    iterations = [r["best_iteration"] for r in records]
    best_out = out / f"best_trial_{best.number:04d}"
    best_out.mkdir(exist_ok=True)
    result = dict(experiment=EXPERIMENT, feature_count=25, features=FEATURES,
                  best_trial_number=best.number, best_params=best.params,
                  full_cv_params=params, mean_3fold=float(np.mean(scores)),
                  std_3fold_ddof1=float(np.std(scores, ddof=1)),
                  mean_fold23=float(np.mean(scores[1:])),
                  cv_best_iterations=iterations, mean_best_iteration=float(np.mean(iterations)),
                  backtest_n_estimators=fixed_n,
                  final_training_params={**params, "n_estimators": fixed_n},
                  best_trial_cache=best.user_attrs["cache_folder"],
                  backtest_note="Diagnostic: parameters/iterations selected with 2017 CV; no 2016 ES.")
    atomic_json(out / "best_params.json", result)
    atomic_json(best_out / "best_params.json", result)
    cv_rows = [{"fold": r["split"]["name"], "rmsle": r["rmsle"],
                "best_iteration": r["best_iteration"],
                "hit_iteration_cap": r["hit_iteration_cap"],
                "predictions_file": r["predictions_file"]} for r in records]
    atomic_csv(out / "cv_fold_results.csv", pd.DataFrame(cv_rows))
    summary = {k: result[k] for k in ("experiment", "best_trial_number", "mean_3fold",
                                    "std_3fold_ddof1", "mean_fold23", "mean_best_iteration",
                                    "backtest_n_estimators")}
    for i, r in enumerate(records, 1):
        summary[f"fold_{i}_rmsle"] = r["rmsle"]
        summary[f"fold_{i}_best_iteration"] = r["best_iteration"]
    atomic_csv(out / "cv_summary.csv", pd.DataFrame([summary]))
    print("[BEST CV]\n" + pd.DataFrame(cv_rows).to_string(index=False), flush=True)
    print(f"[ITERATIONS] {iterations} -> mean={np.mean(iterations):.3f} -> {fixed_n}")
    if any(r["hit_iteration_cap"] for r in records):
        print("[NOTE] A best fold reached max-rounds; its optimum may exceed the cap.")
    if not args.skip_backtest:
        backtest = evaluate_fold(df, AUGUST_BACKTEST,
                                 {**params, "n_estimators": fixed_n},
                                 best_out, False, args.save_models)
        backtest.update(fold_best_iterations=iterations,
                        iteration_rule="rounded arithmetic mean of selected trial's 3 CV best_iterations",
                        evaluation_type=result["backtest_note"])
        atomic_json(best_out / "august_2016_backtest.json", backtest)
        summary["august_2016_backtest_rmsle"] = backtest["rmsle"]
    else:
        summary["august_2016_backtest_rmsle"] = None
    atomic_csv(out / "comparison_summary.csv", pd.DataFrame([summary]))
    atomic_csv(out / "trials.csv", study.trials_dataframe())
    print("[SUMMARY]\n" + pd.DataFrame([summary]).to_string(index=False))
    print(f"[FINISHED] {out}", flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Target total completed trials, including baseline (default 20).")
    parser.add_argument("--max-rounds", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--run-dir", type=Path, help="Resume an existing compatible run directory.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--save-models", action="store_true",
                        help="Also save every fitted model; potentially substantial disk usage.")
    run(parser.parse_args())


if __name__ == "__main__":
    cli()
