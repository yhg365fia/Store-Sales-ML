"""Group Statistics F: store_nbr x dayofweek (past daily store totals).
Run: python src/experiment_f.py
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from main import main
from feature_engineering import LAGS, LAG_FEATURES, add_lag_features

BASE_FEATURES = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion",
    "holiday_count", "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]
PAYDAY_FEATURES = ["is_15th", "is_month_end"]
CATEGORICAL_FEATURES = ["store_nbr", "family", "city", "state", "type", "cluster"]
BASELINE_FEATURES = BASE_FEATURES + list(LAG_FEATURES) + ["dayofyear"] + PAYDAY_FEATURES
F_FEATURE = "store_dayofweek_mean"
EXPERIMENT_FEATURES = BASELINE_FEATURES + [F_FEATURE]

N_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 150
FORECAST_HORIZON = 16
RETRAIN_BASELINE = False  # True: baseline도 3-fold 재학습(총 6모델)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "group_stats_F"

FOLDS = [
    {"fold": "Fold 1", "train_end": "2017-06-28", "valid_start": "2017-06-29", "valid_end": "2017-07-14"},
    {"fold": "Fold 2", "train_end": "2017-07-14", "valid_start": "2017-07-15", "valid_end": "2017-07-30"},
    {"fold": "Fold 3", "train_end": "2017-07-30", "valid_start": "2017-07-31", "valid_end": "2017-08-15"},
]
REFERENCE_BASELINE = {
    "Fold 1": 0.382044,
    "Fold 2": 0.405838,
    "Fold 3": 0.394152,
}


def prepare_features(df):
    if tuple(LAGS) != (16, 21, 28):
        raise ValueError("LAGS는 (16, 21, 28)이어야 합니다.")
    if len(BASELINE_FEATURES) != 29 or len(set(EXPERIMENT_FEATURES)) != 30:
        raise ValueError("피처 구성은 Baseline 29개 + F 1개여야 합니다.")

    df = df.copy()
    required = ["date", "sales", *BASE_FEATURES, "dayofyear", *PAYDAY_FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    keys = ["store_nbr", "family"]
    if df[["date", *keys]].isna().any().any():
        raise ValueError("날짜/매장/상품군에 결측치가 있습니다.")
    if df.duplicated(["date", *keys]).any():
        raise ValueError("date × store_nbr × family 중복이 있습니다.")
    if not np.isfinite(df["sales"].to_numpy(dtype=float)).all() or (df["sales"] < 0).any():
        raise ValueError("sales에 음수·결측·무한대가 있습니다.")
    if not df["dayofweek"].eq(df["date"].dt.dayofweek).all():
        raise ValueError("요일 피처가 실제 날짜와 다릅니다.")
    if not df["dayofyear"].eq(df["date"].dt.dayofyear).all():
        raise ValueError("dayofyear 피처가 실제 날짜와 다릅니다.")

    df = df.sort_values([*keys, "date"]).reset_index(drop=True)
    diffs = df.groupby(keys, observed=True)["date"].diff()
    if not diffs.dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("매장 × 상품군별 날짜가 연속되지 않습니다.")
    df = add_lag_features(df, lags=LAGS)
    df["sales_log"] = np.log1p(df["sales"])

    positions = df.groupby(keys, observed=True).cumcount()
    for lag in LAGS:
        if not df[f"sales_lag_{lag}"].isna().eq(positions < lag).all():
            raise ValueError(f"sales_lag_{lag} 초기 결측 위치가 잘못됐습니다.")
    return df


def make_fold(df, spec):
    train_end = pd.Timestamp(spec["train_end"])
    valid_start = pd.Timestamp(spec["valid_start"])
    valid_end = pd.Timestamp(spec["valid_end"])
    if valid_start != train_end + pd.Timedelta(days=1):
        raise ValueError("Fold 경계가 연속되지 않습니다.")
    if (valid_end - valid_start).days + 1 != FORECAST_HORIZON:
        raise ValueError("Validation은 16일이어야 합니다.")
    if min(LAGS) < FORECAST_HORIZON:
        raise ValueError("16일 예측에서 Lag가 16 미만입니다.")

    train_df = df[df["date"] <= train_end].copy()
    valid_df = df[df["date"].between(valid_start, valid_end)].copy()
    if train_df.empty or valid_df.empty or train_df["date"].max() != train_end:
        raise ValueError("비어 있는 Fold 또는 잘못된 종료일입니다.")
    counts = valid_df.groupby(["store_nbr", "family"], observed=True).size()
    if not counts.eq(FORECAST_HORIZON).all():
        raise ValueError("검증 그룹별 16일 데이터가 모두 있어야 합니다.")
    return train_df, valid_df


def add_store_dayofweek_mean(train_df, valid_df):
    """해당 매장의 '하루 전체 상품군 sales 총합'의 과거 동일 요일 평균.

    Train: 같은 매장의 당일 sales를 집계하되, 집계 결과에서 현재 날짜를 빼고 계산.
    Valid: 오직 Fold Train 종료일까지 같은 매장·요일의 일별 총합 평균.
    """
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    daily = (
        train_df.groupby(["date", "store_nbr"], as_index=False, observed=True)
        .agg(daily_store_sales=("sales", "sum"))
    )
    daily["dayofweek"] = daily["date"].dt.dayofweek
    daily = daily.sort_values(["store_nbr", "dayofweek", "date"]).reset_index(drop=True)
    group = daily.groupby(["store_nbr", "dayofweek"], observed=True)

    # cumsum에서 '오늘 판매량'을 제거. 첫 등장 요일은 NaN.
    previous_count = group.cumcount()
    previous_sum = group["daily_store_sales"].cumsum() - daily["daily_store_sales"]
    daily[F_FEATURE] = previous_sum / previous_count.replace(0, np.nan)

    # 전체 Fold Train의 '하루 매장 총합'을 요일별로 평균.
    # Validation 날짜의 sales는 전혀 참조하지 않음.
    valid_mapping = (
        daily.groupby(["store_nbr", "dayofweek"], observed=True, as_index=False)
        .agg(**{F_FEATURE: ("daily_store_sales", "mean")})
    )
    train_df = train_df.merge(
        daily[["date", "store_nbr", F_FEATURE]],
        on=["date", "store_nbr"], how="left", sort=False, validate="many_to_one"
    )
    valid_df = valid_df.merge(
        valid_mapping,
        on=["store_nbr", "dayofweek"], how="left", sort=False, validate="many_to_one"
    )
    if not np.isfinite(train_df[F_FEATURE].dropna()).all():
        raise ValueError("Train F 피처에 무한대가 있습니다.")
    if valid_df[F_FEATURE].isna().any() or not np.isfinite(valid_df[F_FEATURE]).all():
        raise ValueError("Validation F 피처에 결측치/무한대가 있습니다.")
    if valid_df.groupby(["store_nbr", "dayofweek"], observed=True)[F_FEATURE].nunique().gt(1).any():
        raise ValueError("Validation 요일별 평균이 16일 구간 내에서 변했습니다.")
    return train_df, valid_df


def model_inputs(train_df, valid_df, features):
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    for col in CATEGORICAL_FEATURES:
        categories = pd.Index(train_df[col].dropna().unique())
        X_train[col] = pd.Categorical(X_train[col], categories=categories)
        X_valid[col] = pd.Categorical(X_valid[col], categories=categories)
    allowed_na = set(LAG_FEATURES)
    if F_FEATURE in features:
        allowed_na.add(F_FEATURE)
    nonnullable = [f for f in features if f not in allowed_na]
    if X_train[nonnullable].isna().any().any() or X_valid[nonnullable].isna().any().any():
        raise ValueError("Lag/F 초기 결측 이외의 결측치가 있습니다.")
    if X_valid[list(LAG_FEATURES)].isna().any().any():
        raise ValueError("Validation Lag가 결측입니다.")
    return X_train, X_valid


def build_model():
    return LGBMRegressor(
        boosting_type="gbdt", objective="regression", learning_rate=0.05,
        n_estimators=N_ESTIMATORS, num_leaves=31, max_depth=-1,
        min_child_samples=50, min_split_gain=0.0,
        subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
        reg_alpha=0.0, reg_lambda=0.0, metric="rmse",
        importance_type="split", random_state=42, n_jobs=-1, verbosity=-1,
    )


def rmsle(y_true, prediction_log):
    prediction_log = np.asarray(prediction_log, dtype=float)
    if not np.isfinite(prediction_log).all():
        raise ValueError("예측 결과에 결측치/무한대가 있습니다.")
    prediction_log = np.maximum(prediction_log, 0)
    return float(np.sqrt(np.mean((np.log1p(np.asarray(y_true)) - prediction_log) ** 2)))


def train_one(train_df, valid_df, features, importance=False):
    X_train, X_valid = model_inputs(train_df, valid_df, features)
    model = build_model()
    model.fit(
        X_train, train_df["sales_log"],
        eval_set=[(X_valid, valid_df["sales_log"])],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS), log_evaluation(200)],
    )
    best = int(model.best_iteration_)
    score = rmsle(valid_df["sales"].to_numpy(), model.predict(X_valid, num_iteration=best))
    if importance:
        booster = model.booster_
        g = booster.feature_importance(importance_type="gain", iteration=best)
        s = booster.feature_importance(importance_type="split", iteration=best)
        importance_df = pd.DataFrame({"feature": booster.feature_name(), "gain": g, "split": s})
        importance_df["gain_pct"] = importance_df["gain"] / max(importance_df["gain"].sum(), 1e-12) * 100
        importance_df["split_pct"] = importance_df["split"] / max(importance_df["split"].sum(), 1e-12) * 100
    else:
        importance_df = None
    del model, X_train, X_valid
    gc.collect()
    return best, score, importance_df


def run_experiment():
    raw_train, raw_test, holidays = main()
    del raw_test, holidays
    gc.collect()
    df = prepare_features(raw_train)
    del raw_train
    gc.collect()
    print("\nEXPERIMENT F: store_nbr × dayofweek")
    print("F: 과거 동일 요일의 매장 하루 총판매량 평균")
    print(f"Features: {len(BASELINE_FEATURES)} -> {len(EXPERIMENT_FEATURES)}")
    print(f"n_estimators={N_ESTIMATORS}, early_stopping={EARLY_STOPPING_ROUNDS}")
    print("Baseline retraining:", RETRAIN_BASELINE)
    results = []
    importances = []
    for spec in FOLDS:
        print("\n" + "=" * 80)
        print(spec["fold"])
        print("=" * 80)
        tr, va = make_fold(df, spec)
        tr, va = add_store_dayofweek_mean(tr, va)
        print("Train/Valid:", tr["date"].min().date(), tr["date"].max().date(),
              " / ", va["date"].min().date(), va["date"].max().date())
        print("F missing train/valid:", tr[F_FEATURE].isna().sum(), va[F_FEATURE].isna().sum())

        if RETRAIN_BASELINE:
            baseline_iter, baseline_score, _ = train_one(tr, va, BASELINE_FEATURES)
        else:
            baseline_iter = None
            baseline_score = REFERENCE_BASELINE[spec["fold"]]

        iteration, score, imp = train_one(tr, va, EXPERIMENT_FEATURES, importance=True)
        imp.insert(0, "fold", spec["fold"])
        importances.append(imp)
        results.append({
            "fold": spec["fold"], "baseline_rmsle": baseline_score,
            "F_rmsle": score, "delta_F_minus_baseline": score - baseline_score,
            "baseline_best_iteration": baseline_iter, "F_best_iteration": iteration,
        })
        print(f"Baseline={baseline_score:.6f} | F={score:.6f} | Δ={score-baseline_score:+.6f}")
        del tr, va, imp
        gc.collect()

    fold_results = pd.DataFrame(results)
    all_importances = pd.concat(importances, ignore_index=True)
    mean_importance = (
        all_importances.groupby("feature", as_index=False)
        .agg(mean_gain_pct=("gain_pct", "mean"), mean_split_pct=("split_pct", "mean"),
             mean_gain=("gain", "mean"), mean_split=("split", "mean"))
        .sort_values("mean_gain_pct", ascending=False)
        .reset_index(drop=True)
    )
    mean_importance.insert(0, "gain_rank", np.arange(1, len(mean_importance)+1))
    mean_importance["split_rank"] = mean_importance["mean_split_pct"].rank(ascending=False, method="min").astype(int)
    print("\n===== 3-FOLD RESULT =====")
    print(fold_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Average Baseline: {fold_results['baseline_rmsle'].mean():.6f}")
    print(f"Average F:        {fold_results['F_rmsle'].mean():.6f}")
    print(f"Average delta:    {fold_results['delta_F_minus_baseline'].mean():+.6f}")
    print(f"Improved folds:   {(fold_results['delta_F_minus_baseline'] < 0).sum()}/3")
    print("\n===== FEATURE IMPORTANCE (3-FOLD AVERAGE) =====")
    print(mean_importance.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nF row:")
    print(mean_importance.loc[mean_importance["feature"].eq(F_FEATURE)].to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(OUTPUT_DIR / "F_fold_results.csv", index=False, encoding="utf-8-sig")
    mean_importance.to_csv(OUTPUT_DIR / "F_mean_importance.csv", index=False, encoding="utf-8-sig")
    all_importances.to_csv(OUTPUT_DIR / "F_per_fold_importance.csv", index=False, encoding="utf-8-sig")
    print("\nSaved:", OUTPUT_DIR)
    if not RETRAIN_BASELINE:
        print("주의: Baseline은 과거 동일 설정의 기록을 비교에 사용했습니다. 파이프라인이 바뀌었다면 RETRAIN_BASELINE=True로 실행하세요.")
    return fold_results, mean_importance, all_importances


if __name__ == "__main__":
    run_experiment()
