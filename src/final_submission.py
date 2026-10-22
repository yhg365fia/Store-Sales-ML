"""Kaggle Store Sales: Payday baseline (29) + Oil Raw (1).

Run from project root:
    python src/final_submission.py

Requires the project's existing main.py and feature_engineering.py.
"""
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from main import main
from feature_engineering import LAGS, LAG_FEATURES, add_lag_features

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "submissions"
TARGET = "sales"
HORIZON = 16
MAX_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 150

BASE_FEATURES = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion",
    "holiday_count", "event_christmas", "event_new_year",
    "event_carnival", "event_mothers_day", "event_independence_day",
    "event_city_foundation", "event_good_friday", "event_labor_day",
    "event_other",
]
PAYDAY_FEATURES = ["is_15th", "is_month_end"]
CATEGORICAL_FEATURES = [
    "store_nbr", "family", "city", "state", "type", "cluster",
]
FEATURES = BASE_FEATURES + list(LAG_FEATURES) + ["dayofyear"] + PAYDAY_FEATURES + ["dcoilwtico"]
FOLDS = [
    ("Fold 1", "2017-06-28", "2017-06-29", "2017-07-14"),
    ("Fold 2", "2017-07-14", "2017-07-15", "2017-07-30"),
    ("Fold 3", "2017-07-30", "2017-07-31", "2017-08-15"),
]


def build_model(n_estimators=MAX_ESTIMATORS):
    return LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        learning_rate=0.05,
        n_estimators=n_estimators,
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


def check_columns(frame, cols, label):
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} 필수 컬럼 누락: {missing}")


def prepare_data():
    train, test, _holidays = main()
    train = train.copy()
    test = test.copy()
    check_columns(train, ["date", "sales", *FEATURES[:23], "dayofyear", *PAYDAY_FEATURES, "dcoilwtico"], "Train")
    check_columns(test, ["id", "date", *FEATURES[:23], "dayofyear", *PAYDAY_FEATURES, "dcoilwtico"], "Test")
    if len(FEATURES) != 30 or len(set(FEATURES)) != 30:
        raise AssertionError("피처는 중복 없이 30개여야 합니다.")
    if tuple(LAGS) != (16, 21, 28) or list(LAG_FEATURES) != ["sales_lag_16", "sales_lag_21", "sales_lag_28"]:
        raise AssertionError("Sales lag 구성은 (16, 21, 28)이어야 합니다.")

    for label, frame in (("Train", train), ("Test", test)):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["dcoilwtico"] = pd.to_numeric(frame["dcoilwtico"], errors="coerce")
        bad_oil = ~np.isfinite(frame["dcoilwtico"].to_numpy(dtype=float))
        print(f"{label} Oil missing/nonfinite: {int(bad_oil.sum()):,} / {len(frame):,}")
        if label == "Train" and bad_oil.any():
            raise ValueError("Train Oil Raw에 결측/무한대가 있습니다. 기존 oil 전처리를 확인하세요.")
        if frame[["date", "store_nbr", "family"]].isna().any().any():
            raise ValueError(f"{label}: date/store_nbr/family 키에 결측치가 있습니다.")
        if frame.duplicated(["date", "store_nbr", "family"]).any():
            raise ValueError(f"{label}: date × store_nbr × family 중복이 있습니다.")

    # oil.csv는 대회에서 Test 기간 값까지 제공된다.
    # main()에서 cutoff(2017-08-15)로 Test Oil이 빠졌더라도 원본에서 복구한다.
    oil_candidates = [ROOT / "data" / "oil.csv", ROOT / "oil.csv", ROOT / "data" / "raw" / "oil.csv"]
    oil_path = next((p for p in oil_candidates if p.exists()), None)
    if oil_path is None:
        if test["dcoilwtico"].isna().any():
            raise FileNotFoundError("Test Oil 결측이 있습니다. data/oil.csv 경로를 확인하세요.")
    else:
        oil = pd.read_csv(oil_path, parse_dates=["date"])
        check_columns(oil, ["date", "dcoilwtico"], "oil.csv")
        oil = oil.sort_values("date").drop_duplicates("date", keep="last")
        oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
        daily_oil = oil.set_index("date")["dcoilwtico"].asfreq("D")
        daily_oil = daily_oil.interpolate(method="time", limit_direction="both")
        test["dcoilwtico"] = test["date"].map(daily_oil)
        print(f"Test Oil loaded from: {oil_path}")
    if not np.isfinite(test["dcoilwtico"].to_numpy(dtype=float)).all():
        raise ValueError("Test Oil이 여전히 결측/무한대입니다. 원본 oil.csv를 확인하세요.")
    if test["id"].isna().any() or test["id"].duplicated().any():
        raise ValueError("Test id에 결측치 또는 중복이 있습니다.")
    if train["sales"].isna().any() or (train["sales"] < 0).any():
        raise ValueError("Train sales에 결측치 또는 음수가 있습니다.")
    if train["date"].max() != pd.Timestamp("2017-08-15"):
        raise ValueError("Train 종료일이 2017-08-15가 아닙니다.")
    if test["date"].min() != pd.Timestamp("2017-08-16") or test["date"].max() != pd.Timestamp("2017-08-31"):
        raise ValueError("Test 기간은 2017-08-16~2017-08-31이어야 합니다.")

    # 중요: Test sales는 알 수 없으므로 결측으로 설정. 절대로 0으로 채우지 않는다.
    # 16~28일 lag는 모든 Test 날짜에서 과거 Train의 실제 sales만 참조한다.
    train["_is_test"] = False
    test["_is_test"] = True
    test["sales"] = np.nan
    joined = pd.concat([train, test], ignore_index=True, sort=False)
    joined = joined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    if joined.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Train/Test 결합 후 시계열 키가 중복됩니다.")
    gaps = joined.groupby(["store_nbr", "family"], observed=True)["date"].diff()
    if (gaps.dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("store × family 날짜가 연속되지 않습니다. 기존 날짜 복원을 확인하세요.")

    joined = add_lag_features(joined, lags=LAGS)
    train = joined.loc[~joined["_is_test"]].copy()
    test = joined.loc[joined["_is_test"]].copy()
    test = test.sort_values("id").reset_index(drop=True)

    for col in LAG_FEATURES:
        if test[col].isna().any():
            raise ValueError(f"Test에 {col} 결측치가 있습니다. Lag 생성 로직을 확인하세요.")
    check_columns(train, FEATURES, "Train features")
    check_columns(test, FEATURES, "Test features")
    other_features = [f for f in FEATURES if f not in LAG_FEATURES]
    if train[other_features].isna().any().any() or test[FEATURES].isna().any().any():
        raise ValueError("일반 피처에 결측치가 있습니다. main()의 전처리를 확인하세요.")

    # 학습 전체에서 동일한 category 목록을 유지한다.
    for col in CATEGORICAL_FEATURES:
        cats = pd.Index(train[col].dropna().unique())
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
        if test[col].isna().any():
            raise ValueError(f"Test의 {col}에 학습에서 보지 못한 범주가 있습니다.")

    train["sales_log"] = np.log1p(train["sales"].to_numpy(dtype=float))
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print(f"Oil missing: train={train['dcoilwtico'].isna().sum()}, test={test['dcoilwtico'].isna().sum()}")
    return train, test


def run_cv(train):
    fold_results = []
    for name, train_end, valid_start, valid_end in FOLDS:
        tr = train.loc[train["date"] <= pd.Timestamp(train_end)]
        va = train.loc[train["date"].between(valid_start, valid_end)].copy()
        if tr.empty or va.empty or va["date"].nunique() != HORIZON:
            raise ValueError(f"{name}: 16일 검증 데이터가 올바르지 않습니다.")
        # 기존 oil_raw 검증과 동일하게 각 날짜의 Oil 원본 피처를 사용한다.
        print(f"\n{name}: train <= {train_end} | valid {valid_start} ~ {valid_end}")
        model = build_model()
        model.fit(
            tr[FEATURES], tr["sales_log"],
            eval_set=[(va[FEATURES], va["sales_log"])],
            eval_metric="rmse",
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=True), log_evaluation(200)],
        )
        best = int(model.best_iteration_)
        pred_log = np.maximum(0, model.predict(va[FEATURES], num_iteration=best))
        rmsle = float(np.sqrt(np.mean((va["sales_log"].to_numpy() - pred_log) ** 2)))
        fold_results.append({"fold": name, "best_iteration": best, "rmsle": rmsle})
        print(f"{name}: RMSLE={rmsle:.6f}, best_iteration={best}")
        del model, tr, va
        gc.collect()
    result = pd.DataFrame(fold_results)
    print("\nCV results:\n", result.to_string(index=False))
    print(f"Mean RMSLE: {result['rmsle'].mean():.6f}")
    return result


def main_submit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = prepare_data()
    cv = run_cv(train)
    n_estimators = int(np.median(cv["best_iteration"].to_numpy()))
    print(f"\nFinal fit: full train, n_estimators={n_estimators}")
    model = build_model(n_estimators=n_estimators)
    model.fit(train[FEATURES], train["sales_log"], categorical_feature=CATEGORICAL_FEATURES)

    pred_log = model.predict(test[FEATURES])
    if not np.isfinite(pred_log).all():
        raise ValueError("Test 예측값에 NaN/무한대가 있습니다.")
    predictions = np.expm1(np.maximum(0, pred_log))
    submission = pd.DataFrame({"id": test["id"].to_numpy(), "sales": predictions})
    if len(submission) != len(test) or submission["id"].duplicated().any() or (submission["sales"] < 0).any():
        raise ValueError("제출 파일 QC 실패")
    sub_path = OUTPUT_DIR / "submission_oil_raw.csv"
    submission.to_csv(sub_path, index=False)
    cv.to_csv(OUTPUT_DIR / "oil_raw_cv.csv", index=False)
    (OUTPUT_DIR / "oil_raw_config.json").write_text(
        json.dumps({"features": FEATURES, "folds": cv.to_dict(orient="records"),
                    "final_n_estimators": n_estimators,
                    "model_params": model.get_params()}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\nSubmission preview:\n", submission.head().to_string(index=False))
    print(f"\nSaved: {sub_path.resolve()}")
    print(f"Rows: {len(submission):,} | NaN: {submission.isna().sum().sum()}")


if __name__ == "__main__":
    main_submit()
