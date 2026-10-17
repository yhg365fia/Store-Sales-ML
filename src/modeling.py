"""
Payday Baseline 29개 피처 단독 재학습 + 전체 Feature Importance 출력.

- 기존 main.py의 main() -> (train, test, holidays) 사용
- 기존 feature_engineering.py의 add_lag_features() 사용
- 기존 3-Fold / 16일 예측 / LightGBM 파라미터 유지
- n_estimators=8000, Early Stopping=150
- Group Statistics A~E는 추가하지 않음
- 실제 Fold별 best_iteration 및 RMSLE 재계산
- 모든 29개 피처의 Gain/Split 중요도 3-Fold 평균과 Fold별 수치 출력 및 CSV 저장

실행: python src/modeling.py  (main.py와 feature_engineering.py도 src/에 있어야 함)
저장: 프로젝트/results/baseline_8000/*.csv
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from main import main
from feature_engineering import LAGS, LAG_FEATURES, add_lag_features


# =========================================================
# 1. 기존 Payday Baseline 구성: 총 29개
# =========================================================

BASE_FEATURES = [
    # Store / Product (6)
    "store_nbr", "family", "city", "state", "type", "cluster",
    # Calendar (5)
    "year", "month", "day", "dayofweek", "is_weekend",
    # Promotion (2)
    "onpromotion", "is_promotion",
    # Holiday (10)
    "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]

PAYDAY_FEATURES = ["is_15th", "is_month_end"]
CATEGORICAL_FEATURES = [
    "store_nbr", "family", "city", "state", "type", "cluster",
]

BASELINE_FEATURES = (
    BASE_FEATURES + list(LAG_FEATURES) + ["dayofyear"] + PAYDAY_FEATURES
)

TARGET_COL = "sales_log"
FORECAST_HORIZON = 16
N_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 150

# 과거 C 실험에서 실제로 재학습한 8,000 상한 Baseline 기록.
# 이번 스크립트는 기록을 재사용하지 않고 3-Fold를 새로 학습해 검증함.
REFERENCE_8000 = {
    "Fold 1": {"best_iteration": 6476, "rmsle": 0.382044},
    "Fold 2": {"best_iteration": 2697, "rmsle": 0.405838},
    "Fold 3": {"best_iteration": 6902, "rmsle": 0.394152},
}

FOLDS = [
    {"fold": "Fold 1", "train_end": "2017-06-28", "valid_start": "2017-06-29", "valid_end": "2017-07-14"},
    {"fold": "Fold 2", "train_end": "2017-07-14", "valid_start": "2017-07-15", "valid_end": "2017-07-30"},
    {"fold": "Fold 3", "train_end": "2017-07-30", "valid_start": "2017-07-31", "valid_end": "2017-08-15"},
]

# modeling.py가 src/ 안에 있을 때 프로젝트 루트의 results/ 사용.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "baseline_8000"


# =========================================================
# 2. 설정 검증 및 기본 피처 준비
# =========================================================

def validate_configuration():
    if tuple(LAGS) != (16, 21, 28):
        raise ValueError("LAGS는 (16, 21, 28)이어야 합니다.")
    if list(LAG_FEATURES) != ["sales_lag_16", "sales_lag_21", "sales_lag_28"]:
        raise ValueError("Lag 피처 구성이 기존과 다릅니다.")
    if len(BASE_FEATURES) != 23:
        raise ValueError(f"BASE_FEATURES는 23개여야 합니다 (실제 {len(BASE_FEATURES)}개).")
    if len(BASELINE_FEATURES) != 29 or len(set(BASELINE_FEATURES)) != 29:
        raise ValueError("Baseline은 중복 없이 정확히 29개 피처여야 합니다.")
    forbidden = [
        f for f in BASELINE_FEATURES
        if (any(x in f.lower() for x in ("rolling", "ema", "seasonal", "_mean"))
            or f in {"sales", TARGET_COL, "dayofyear_sin", "dayofyear_cos", "sales_lag_364"})
    ]
    if forbidden:
        raise ValueError(f"Baseline에 포함되면 안 되는 피처: {forbidden}")


def prepare_features(df):
    validate_configuration()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    required = ["date", "sales", *BASE_FEATURES, "dayofyear", *PAYDAY_FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"필수 컬럼 누락: {missing}\n"
            "main()의 전처리/휴일 매장 매칭/Feature Engineering부터 확인하세요."
        )

    keys = ["store_nbr", "family"]
    if df[keys + ["date"]].isna().any().any():
        raise ValueError("매장·상품군·날짜에 결측치가 있습니다.")
    if df.duplicated(keys + ["date"]).any():
        raise ValueError("매장 × 상품군 × 날짜 조합이 중복됩니다.")

    sales = df["sales"].to_numpy(dtype=float)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Train sales에 결측·무한대·음수가 있습니다.")

    expected_calendar = {
        "dayofyear": df["date"].dt.dayofyear,
        "dayofweek": df["date"].dt.dayofweek,
        "is_15th": (df["date"].dt.day == 15).astype(int),
        "is_month_end": df["date"].dt.is_month_end.astype(int),
    }
    for feat, expected in expected_calendar.items():
        if not np.array_equal(df[feat].to_numpy(), expected.to_numpy()):
            raise ValueError(f"{feat} 계산값이 date와 일치하지 않습니다.")

    df = df.sort_values(keys + ["date"]).reset_index(drop=True)

    # 현재 전처리는 12/25 등의 누락 날짜를 패딩하여 매 그룹이 연속 일별 데이터임.
    # 이 전제가 깨지면 shift(16)은 16일 전 판매량을 가리키지 않으므로 오류 처리.
    date_diff = df.groupby(keys, observed=True)["date"].diff()
    invalid_gap = date_diff.notna() & date_diff.ne(pd.Timedelta(days=1))
    if invalid_gap.any():
        raise ValueError("그룹 내 날짜가 연속적이지 않습니다. main()의 날짜 패딩을 확인하세요.")

    df = add_lag_features(df, lags=LAGS)
    df[TARGET_COL] = np.log1p(df["sales"])

    group_position = df.groupby(keys, observed=True).cumcount()
    for lag in LAGS:
        actual = df[f"sales_lag_{lag}"].isna()
        expected = group_position < lag
        if not actual.equals(expected):
            raise ValueError(f"sales_lag_{lag} 결측 위치가 예상과 다릅니다.")

    print("\n" + "=" * 80)
    print("PAYDAY BASELINE (29 FEATURES) — EXACT 8000-CAP IMPORTANCE")
    print("=" * 80)
    print(f"Baseline features: {len(BASELINE_FEATURES)}")
    print("Group Statistics: NOT INCLUDED")
    print(f"n_estimators: {N_ESTIMATORS} / early stopping: {EARLY_STOPPING_ROUNDS}")
    print("3-Fold Baseline retraining: ENABLED (3 models)")
    print("Lag missing:\n", df[LAG_FEATURES].isna().sum().to_string())
    print("Common QC: OK")
    return df


# =========================================================
# 3. 기존 16일 3-Fold
# =========================================================

def make_fold(df, config):
    train_end = pd.Timestamp(config["train_end"])
    valid_start = pd.Timestamp(config["valid_start"])
    valid_end = pd.Timestamp(config["valid_end"])
    horizon = (valid_end - valid_start).days + 1

    if horizon != FORECAST_HORIZON or valid_start != train_end + pd.Timedelta(days=1):
        raise ValueError("Fold 날짜가 16일 예측 설정과 다릅니다.")
    if min(LAGS) < horizon:
        raise ValueError("16일 일괄 예측보다 짧은 Lag가 있습니다.")

    train_df = df.loc[df["date"].le(train_end)].copy()
    valid_df = df.loc[df["date"].between(valid_start, valid_end)].copy()
    if train_df.empty or valid_df.empty or train_df["date"].max() != train_end:
        raise ValueError("Fold Train/Validation 데이터 또는 종료 날짜가 유효하지 않습니다.")

    valid_counts = valid_df.groupby(["store_nbr", "family"], observed=True).size()
    if not valid_counts.eq(horizon).all():
        raise ValueError("일부 Validation 매장 × 상품군 그룹이 16일을 채우지 못했습니다.")

    print(f"Train: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"Valid: {valid_df['date'].min().date()} ~ {valid_df['date'].max().date()}")
    print("Train shape:", train_df.shape, "| Valid shape:", valid_df.shape)
    return train_df, valid_df


# =========================================================
# 4. 모델 데이터: 정확히 29개만 사용
# =========================================================

def prepare_model_data(train_df, valid_df):
    X_train = train_df[BASELINE_FEATURES].copy()
    X_valid = valid_df[BASELINE_FEATURES].copy()

    # 기존과 동일: Train에서 관측된 범주만으로 카테고리 결정.
    for feature in CATEGORICAL_FEATURES:
        categories = pd.Index(train_df[feature].dropna().unique())
        X_train[feature] = pd.Categorical(X_train[feature], categories=categories)
        X_valid[feature] = pd.Categorical(X_valid[feature], categories=categories)

    required_non_missing = [f for f in BASELINE_FEATURES if f not in LAG_FEATURES]
    for name, X in (("Train", X_train), ("Valid", X_valid)):
        if X[required_non_missing].isna().any().any():
            raise ValueError(f"{name}: Lag 외 일반 피처에 결측치가 있습니다.")
        if len(X.columns) != 29:
            raise ValueError(f"{name}: 입력 피처가 29개가 아닙니다.")
    if X_valid[LAG_FEATURES].isna().any().any():
        raise ValueError("Validation Lag에 결측치가 있습니다.")

    return (
        X_train, X_valid,
        train_df[TARGET_COL].copy(),
        valid_df[TARGET_COL].copy(),
        valid_df["sales"].copy(),
    )


# =========================================================
# 5. 기존 실험과 동일한 LightGBM 설정
# =========================================================

def build_model():
    return LGBMRegressor(
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


def calculate_rmsle(y_true, pred_log):
    y_true = np.asarray(y_true, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    if not np.isfinite(pred_log).all():
        raise ValueError("예측값에 NaN 또는 무한대가 있습니다.")
    pred_log = np.maximum(pred_log, 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - pred_log) ** 2)))


# =========================================================
# 6. Feature Importance: 모든 Fold, 29개, Gain + Split
# =========================================================

def get_feature_importance(model, fold_name):
    booster = model.booster_
    iteration = int(model.best_iteration_)
    names = booster.feature_name()
    if set(names) != set(BASELINE_FEATURES) or len(names) != 29:
        raise ValueError("학습 모델의 피처 구성이 29개 Payday Baseline과 다릅니다.")

    gain = booster.feature_importance(importance_type="gain", iteration=iteration)
    split = booster.feature_importance(importance_type="split", iteration=iteration)
    result = pd.DataFrame({
        "fold": fold_name,
        "best_iteration": iteration,
        "feature": names,
        "gain": gain,
        "split": split,
    })
    total_gain = result["gain"].sum()
    total_split = result["split"].sum()
    result["gain_pct"] = result["gain"] / total_gain * 100 if total_gain > 0 else 0.0
    result["split_pct"] = result["split"] / total_split * 100 if total_split > 0 else 0.0
    return result


# =========================================================
# 7. 한 Fold 학습
# =========================================================

def train_baseline_model(train_df, valid_df, fold_name):
    X_train, X_valid, y_train, y_valid_log, y_valid = prepare_model_data(
        train_df, valid_df
    )
    model = build_model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid_log)],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            log_evaluation(period=100),
        ],
    )
    iteration = int(model.best_iteration_)
    pred_log = model.predict(X_valid, num_iteration=iteration)
    score = calculate_rmsle(y_valid, pred_log)
    importance_df = get_feature_importance(model, fold_name)

    print(f"\nBest iteration: {iteration}")
    print(f"Baseline RMSLE: {score:.6f}")
    if iteration >= N_ESTIMATORS - EARLY_STOPPING_ROUNDS:
        print("CHECK: Best iteration이 학습 상한에 가깝습니다.")
    del X_train, X_valid, y_train, y_valid_log, y_valid, pred_log, model
    gc.collect()
    return iteration, score, importance_df


# =========================================================
# 8. 3-Fold 실행 및 전체 표 생성
# =========================================================

def run_backtesting(save_csv=True):
    validate_configuration()
    train, test, holidays = main()
    del test, holidays
    gc.collect()
    train = prepare_features(train)

    fold_rows, fold_importances = [], []
    for config in FOLDS:
        fold_name = config["fold"]
        print("\n" + "=" * 85)
        print(f"PAYDAY BASELINE — {fold_name}")
        print("=" * 85)
        train_df, valid_df = make_fold(train, config)
        iteration, score, importance = train_baseline_model(
            train_df, valid_df, fold_name
        )
        reference = REFERENCE_8000[fold_name]
        fold_rows.append({
            "fold": fold_name,
            "best_iteration": iteration,
            "rmsle": score,
            "reference_best_iteration": reference["best_iteration"],
            "reference_rmsle": reference["rmsle"],
            "delta_vs_reference": score - reference["rmsle"],
        })
        fold_importances.append(importance)
        del train_df, valid_df, importance
        gc.collect()

    fold_results = pd.DataFrame(fold_rows)
    per_fold_importance = pd.concat(fold_importances, ignore_index=True)

    mean_importance = (
        per_fold_importance
        .groupby("feature", as_index=False)
        .agg(
            mean_gain_pct=("gain_pct", "mean"),
            mean_split_pct=("split_pct", "mean"),
            mean_gain=("gain", "mean"),
            mean_split=("split", "mean"),
        )
        .sort_values("mean_gain_pct", ascending=False)
        .reset_index(drop=True)
    )
    # 평균 중요도 기준 순위. 평균 내림차순으로 정렬한 후 부여.
    mean_importance.insert(0, "gain_rank", np.arange(1, len(mean_importance) + 1))
    split_ranks = mean_importance["mean_split_pct"].rank(method="min", ascending=False)
    mean_importance.insert(4, "split_rank", split_ranks.astype(int))

    if len(mean_importance) != 29 or len(per_fold_importance) != 3 * 29:
        raise ValueError("Feature Importance 출력 행 수가 29 / 87과 다릅니다.")
    if not np.isclose(mean_importance["mean_gain_pct"].sum(), 100.0):
        raise ValueError("평균 Gain Importance 비율의 합이 100%가 아닙니다.")
    if not np.isclose(mean_importance["mean_split_pct"].sum(), 100.0):
        raise ValueError("평균 Split Importance 비율의 합이 100%가 아닙니다.")

    print("\n" + "=" * 95)
    print("PAYDAY BASELINE — 8000-CAP FOLD RESULTS")
    print("=" * 95)
    print(fold_results.round(6).to_string(index=False))
    print("\nMean RMSLE:", f"{fold_results['rmsle'].mean():.6f}")
    print("Std RMSLE :", f"{fold_results['rmsle'].std(ddof=1):.6f}")
    print("\n" + "=" * 95)
    print("ALL 29 FEATURE IMPORTANCE — 3-FOLD MEAN (GAIN SORTED)")
    print("=" * 95)
    print(mean_importance.round(4).to_string(index=False))

    print("\n" + "=" * 95)
    print("PER-FOLD FEATURE IMPORTANCE (29 FEATURES × 3 FOLDS)")
    print("=" * 95)
    for fold_name, group in per_fold_importance.groupby("fold", sort=False):
        print(f"\n{fold_name}")
        print(group.sort_values("gain_pct", ascending=False).round(4).to_string(index=False))

    if save_csv:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fold_results.to_csv(OUTPUT_DIR / "baseline_8000_fold_results.csv", index=False, encoding="utf-8-sig")
        mean_importance.to_csv(OUTPUT_DIR / "baseline_8000_mean_importance_29.csv", index=False, encoding="utf-8-sig")
        per_fold_importance.to_csv(OUTPUT_DIR / "baseline_8000_per_fold_importance_87.csv", index=False, encoding="utf-8-sig")
        print("\nCSV 저장 위치:", OUTPUT_DIR)
        print(" - baseline_8000_fold_results.csv")
        print(" - baseline_8000_mean_importance_29.csv")
        print(" - baseline_8000_per_fold_importance_87.csv")

    del train
    gc.collect()
    return fold_results, mean_importance, per_fold_importance


if __name__ == "__main__":
    fold_results, mean_importance, per_fold_importance = run_backtesting()
