import gc

import numpy as np
import pandas as pd

from lightgbm import (
    LGBMRegressor,
    early_stopping,
    log_evaluation,
)

from main import main

from feature_engineering import (
    LAGS,
    LAG_FEATURES,
    ROLLING_SHIFT,
    ROLLING_WINDOWS,
    ROLLING_MEAN_FEATURES,
    add_lag_features,
    add_rolling_features,
)


# =========================================================
# 1. 피처 및 학습 설정
# =========================================================

BASE_FEATURES = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
    "year",
    "month",
    "day",
    "dayofweek",
    "is_weekend",
    "onpromotion",
    "is_promotion",
    "holiday_count",
    "event_christmas",
    "event_new_year",
    "event_carnival",
    "event_mothers_day",
    "event_independence_day",
    "event_city_foundation",
    "event_good_friday",
    "event_labor_day",
    "event_other",
]

CATEGORICAL_FEATURES = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
]

TIME_SERIES_FEATURES = (
    LAG_FEATURES
    + ROLLING_MEAN_FEATURES
)

FEATURES = BASE_FEATURES + TIME_SERIES_FEATURES

# 이번 실험의 변경 사항: 50 → 150
EARLY_STOPPING_ROUNDS = 150

FORECAST_HORIZON = 16


# =========================================================
# 2. 검증 구간 설정
# =========================================================

FOLDS = [
    {
        "fold": "Fold 1",
        "train_end": "2017-06-28",
        "valid_start": "2017-06-29",
        "valid_end": "2017-07-14",
    },
    {
        "fold": "Fold 2",
        "train_end": "2017-07-14",
        "valid_start": "2017-07-15",
        "valid_end": "2017-07-30",
    },
    {
        "fold": "Fold 3",
        "train_end": "2017-07-30",
        "valid_start": "2017-07-31",
        "valid_end": "2017-08-15",
    },
]


# =========================================================
# 3. 설정 확인
# =========================================================

def validate_configuration():

    if tuple(LAGS) != (16, 21, 28):
        raise ValueError(
            "feature_engineering.py의 LAGS를 "
            "(16, 21, 28)로 설정하세요."
        )

    if tuple(ROLLING_WINDOWS) != (7, 14, 28):
        raise ValueError(
            "feature_engineering.py의 ROLLING_WINDOWS를 "
            "(7, 14, 28)로 설정하세요."
        )

    if ROLLING_SHIFT != 16:
        raise ValueError(
            "ROLLING_SHIFT는 16이어야 합니다."
        )

    expected_features = [
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_rolling_mean_16_7",
        "sales_rolling_mean_16_14",
        "sales_rolling_mean_16_28",
    ]

    if TIME_SERIES_FEATURES != expected_features:
        raise ValueError(
            "시계열 피처 구성이 이번 실험 설정과 다릅니다.\n"
            f"예상: {expected_features}\n"
            f"현재: {TIME_SERIES_FEATURES}"
        )

    if len(FEATURES) != len(set(FEATURES)):
        raise ValueError("피처 목록에 중복이 있습니다.")

    if len(FEATURES) != 29:
        raise ValueError(
            f"입력 피처는 29개여야 합니다: {len(FEATURES)}개"
        )


# =========================================================
# 4. 시계열 피처 생성 및 QC
# =========================================================

def prepare_time_series_features(df):

    validate_configuration()

    df = df.copy()

    required_columns = [
        "date",
        "sales",
        *BASE_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"])

    keys = ["store_nbr", "family"]

    if df[keys + ["date"]].isna().any().any():
        raise ValueError(
            "매장, 상품군 또는 날짜에 결측치가 있습니다."
        )

    if df.duplicated(keys + ["date"]).any():
        raise ValueError(
            "동일한 매장 × 상품군 × 날짜가 중복되어 있습니다."
        )

    sales_values = df["sales"].to_numpy(dtype=float)

    if (
        not np.isfinite(sales_values).all()
        or (sales_values < 0).any()
    ):
        raise ValueError(
            "학습 데이터의 sales는 결측치 없이 "
            "유한한 0 이상의 값이어야 합니다."
        )

    df = (
        df.sort_values(keys + ["date"])
        .reset_index(drop=True)
    )

    # 행 기반 shift가 실제 일 단위를 의미하는지 확인
    date_diff = (
        df.groupby(keys, observed=True)["date"]
        .diff()
    )

    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(pd.Timedelta(days=1))
    )

    if invalid_gap.any():
        raise ValueError(
            "매장 × 상품군별 날짜가 연속적이지 않습니다. "
            "전처리에서 누락 날짜를 확인하세요."
        )

    # 원본 판매량으로 시계열 피처 생성
    df = add_lag_features(
        df,
        lags=LAGS,
    )

    df = add_rolling_features(
        df,
        shift=ROLLING_SHIFT,
        windows=ROLLING_WINDOWS,
    )

    # 모델 입력은 FEATURES에 지정된 29개 컬럼만 사용
    df["sales_log"] = np.log1p(df["sales"])

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"생성되지 않은 피처가 있습니다: {missing_features}"
        )

    # 초기 기록 부족으로 생기는 결측 위치 확인
    group_position = (
        df.groupby(keys, observed=True)
        .cumcount()
    )

    initial_missing_rows = {}

    for lag in LAGS:
        initial_missing_rows[f"sales_lag_{lag}"] = lag

    for window in ROLLING_WINDOWS:
        feature = (
            f"sales_rolling_mean_{ROLLING_SHIFT}_{window}"
        )
        initial_missing_rows[feature] = (
            ROLLING_SHIFT + window - 1
        )

    for feature, initial_rows in initial_missing_rows.items():

        expected_missing = group_position.lt(initial_rows)
        actual_missing = df[feature].isna()

        if not actual_missing.equals(expected_missing):
            raise ValueError(
                f"{feature}: 결측 위치가 예상과 다릅니다."
            )

        valid_values = df[feature].dropna().to_numpy()

        if not np.isfinite(valid_values).all():
            raise ValueError(
                f"{feature}: 무한대 값이 있습니다."
            )

    print("\n===== TIME SERIES FEATURES =====")
    print("Lags:", LAGS)
    print("Rolling shift:", ROLLING_SHIFT)
    print("Rolling windows:", ROLLING_WINDOWS)

    print("\nLag 피처:", LAG_FEATURES)
    print("Rolling mean 피처:", ROLLING_MEAN_FEATURES)

    print("\n시계열 피처 수:", len(TIME_SERIES_FEATURES))
    print("전체 입력 피처 수:", len(FEATURES))
    print("Early stopping rounds:", EARLY_STOPPING_ROUNDS)

    print("\n학습 초기 과거 기록 부족으로 발생한 결측치:")
    print(df[TIME_SERIES_FEATURES].isna().sum())

    print("Lag + Rolling Mean QC: OK")

    return df


# =========================================================
# 5. Fold 데이터 분리
# =========================================================

def make_fold(df, fold_config):

    train_end = pd.Timestamp(fold_config["train_end"])
    valid_start = pd.Timestamp(fold_config["valid_start"])
    valid_end = pd.Timestamp(fold_config["valid_end"])

    horizon = (valid_end - valid_start).days + 1

    if horizon != FORECAST_HORIZON:
        raise ValueError(
            f"검증 구간은 {FORECAST_HORIZON}일이어야 합니다."
        )

    if valid_start != train_end + pd.Timedelta(days=1):
        raise ValueError(
            "학습 종료일 다음 날부터 검증이 시작되어야 합니다."
        )

    if min(*LAGS, ROLLING_SHIFT) < horizon:
        raise ValueError(
            "일괄 예측 기간보다 짧은 lag 또는 shift가 있습니다."
        )

    train_df = df.loc[
        df["date"].le(train_end)
    ].copy()

    valid_df = df.loc[
        df["date"].between(valid_start, valid_end)
    ].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"{fold_config['fold']}: 학습 또는 검증 데이터가 없습니다."
        )

    keys = ["store_nbr", "family"]

    train_groups = (
        train_df.groupby(keys, observed=True)
        .size()
        .index
    )

    valid_counts = (
        valid_df.groupby(keys, observed=True)
        .size()
    )

    if (
        len(train_groups.difference(valid_counts.index)) > 0
        or len(valid_counts.index.difference(train_groups)) > 0
    ):
        raise ValueError(
            "학습과 검증의 매장 × 상품군 구성이 다릅니다."
        )

    if not valid_counts.eq(horizon).all():
        raise ValueError(
            "검증 데이터의 일부 매장 × 상품군이 "
            "16일을 채우지 못했습니다."
        )

    print(
        f"Train: {train_df['date'].min()} "
        f"~ {train_df['date'].max()}"
    )
    print(
        f"Valid: {valid_df['date'].min()} "
        f"~ {valid_df['date'].max()}"
    )

    print("\nTrain shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)

    return train_df, valid_df


# =========================================================
# 6. 모델 입력 준비
# =========================================================

def prepare_model_data(train_df, valid_df):

    X_train = train_df[FEATURES].copy()
    X_valid = valid_df[FEATURES].copy()

    for feature in CATEGORICAL_FEATURES:

        categories = pd.Index(
            train_df[feature].dropna().unique()
        )

        X_train[feature] = pd.Categorical(
            X_train[feature],
            categories=categories,
        )

        X_valid[feature] = pd.Categorical(
            X_valid[feature],
            categories=categories,
        )

    if X_train[BASE_FEATURES].isna().any().any():
        raise ValueError(
            "학습 데이터의 기본 피처에 결측치가 있습니다."
        )

    if X_valid[BASE_FEATURES].isna().any().any():
        raise ValueError(
            "검증 데이터의 기본 피처에 결측치 또는 "
            "학습에 없던 범주가 있습니다."
        )

    # 학습 초기 시계열 결측치는 LightGBM이 처리
    if X_valid[TIME_SERIES_FEATURES].isna().any().any():
        raise ValueError(
            "검증 데이터의 시계열 피처에 결측치가 있습니다."
        )

    y_train = train_df["sales_log"].copy()
    y_valid_log = valid_df["sales_log"].copy()
    y_valid = valid_df["sales"].copy()

    print("\nX_train:", X_train.shape)
    print("X_valid:", X_valid.shape)

    return (
        X_train,
        X_valid,
        y_train,
        y_valid_log,
        y_valid,
    )


# =========================================================
# 7. 모델 생성 — 기존 파라미터 유지
# =========================================================

def build_model():

    return LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        learning_rate=0.05,
        n_estimators=5000,
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


# =========================================================
# 8. RMSLE 평가
# =========================================================

def calculate_rmsle(y_true, pred_log):

    """
    log1p(sales) 예측값으로 RMSLE 계산.

    음수 로그 예측을 0으로 제한하는 것은
    원래 판매량 예측을 0 이상으로 제한하는 것과 같음.
    """

    y_true = np.asarray(y_true, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)

    if not np.isfinite(pred_log).all():
        raise ValueError(
            "예측값에 NaN 또는 무한대가 있습니다."
        )

    pred_log = np.maximum(pred_log, 0.0)

    return float(
        np.sqrt(
            np.mean(
                (np.log1p(y_true) - pred_log) ** 2
            )
        )
    )


# =========================================================
# 9. 피처 중요도
# =========================================================

def get_feature_group(feature):

    if feature in LAG_FEATURES:
        return "Lag"

    if feature in ROLLING_MEAN_FEATURES:
        return "Rolling Mean"

    return "Base"


def get_feature_importance(model, fold_name):

    booster = model.booster_
    iteration = model.best_iteration_

    gain = booster.feature_importance(
        importance_type="gain",
        iteration=iteration,
    )

    split = booster.feature_importance(
        importance_type="split",
        iteration=iteration,
    )

    importance = pd.DataFrame({
        "fold": fold_name,
        "feature": booster.feature_name(),
        "gain": gain,
        "split": split,
    })

    importance["group"] = (
        importance["feature"].map(get_feature_group)
    )

    gain_total = importance["gain"].sum()
    split_total = importance["split"].sum()

    importance["gain_pct"] = (
        importance["gain"] / gain_total * 100
        if gain_total > 0
        else 0.0
    )

    importance["split_pct"] = (
        importance["split"] / split_total * 100
        if split_total > 0
        else 0.0
    )

    return importance


def summarize_feature_importance(all_importance):

    mean_importance = (
        all_importance
        .groupby(
            ["feature", "group"],
            as_index=False,
        )
        .agg(
            mean_gain_pct=("gain_pct", "mean"),
            mean_split_pct=("split_pct", "mean"),
            mean_split=("split", "mean"),
        )
    )

    mean_importance["gain_rank"] = (
        mean_importance["mean_gain_pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    mean_importance["split_rank"] = (
        mean_importance["mean_split"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    mean_importance = (
        mean_importance.sort_values(
            ["gain_rank", "split_rank", "feature"]
        )
        .reset_index(drop=True)
    )

    columns = [
        "feature",
        "group",
        "gain_rank",
        "mean_gain_pct",
        "split_rank",
        "mean_split_pct",
        "mean_split",
    ]

    mean_importance = mean_importance[columns]

    print("\n" + "=" * 60)
    print("MEAN FEATURE IMPORTANCE (gain 기준 정렬)")
    print("=" * 60)

    print(
        mean_importance.round(2).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("TIME SERIES FEATURE IMPORTANCE")
    print("=" * 60)

    time_series_importance = mean_importance.loc[
        mean_importance["group"].ne("Base"),
        [
            "feature",
            "group",
            "gain_rank",
            "mean_gain_pct",
            "split_rank",
            "mean_split_pct",
        ],
    ]

    print(
        time_series_importance.round(2).to_string(index=False)
    )

    group_importance = (
        mean_importance
        .groupby("group", as_index=False)
        .agg(
            n_features=("feature", "size"),
            gain_pct=("mean_gain_pct", "sum"),
            split_pct=("mean_split_pct", "sum"),
        )
        .sort_values("gain_pct", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("IMPORTANCE BY FEATURE GROUP (%)")
    print("=" * 60)

    print(
        group_importance.round(2).to_string(index=False)
    )

    return mean_importance


# =========================================================
# 10. 3개 Fold 백테스트
# =========================================================

def run_backtesting():

    validate_configuration()

    # 기존 전처리 파이프라인
    train, test, holidays = main()

    del test, holidays
    gc.collect()

    train = prepare_time_series_features(train)

    fold_results = []
    fold_importances = []

    for fold_config in FOLDS:

        fold_name = fold_config["fold"]

        print("\n" + "=" * 60)
        print(f"{fold_name} START")
        print("=" * 60)

        train_df, valid_df = make_fold(
            train,
            fold_config,
        )

        (
            X_train,
            X_valid,
            y_train,
            y_valid_log,
            y_valid,
        ) = prepare_model_data(
            train_df,
            valid_df,
        )

        del train_df, valid_df
        gc.collect()

        model = build_model()

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_valid, y_valid_log),
            ],
            eval_metric="rmse",
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[
                early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=True,
                ),
                log_evaluation(period=100),
            ],
        )

        best_iteration = int(model.best_iteration_)

        pred_log = model.predict(
            X_valid,
            num_iteration=best_iteration,
        )

        rmsle = calculate_rmsle(
            y_true=y_valid,
            pred_log=pred_log,
        )

        print("\n===== VALIDATION EVALUATION =====")
        print(f"RMSLE: {rmsle:.6f}")

        print(
            f"\n{fold_name} Best Iteration: {best_iteration}"
        )
        print(f"{fold_name} RMSLE: {rmsle:.6f}")

        fold_results.append({
            "fold": fold_name,
            "train_end": fold_config["train_end"],
            "valid_start": fold_config["valid_start"],
            "valid_end": fold_config["valid_end"],
            "best_iteration": best_iteration,
            "rmsle": rmsle,
        })

        fold_importances.append(
            get_feature_importance(
                model=model,
                fold_name=fold_name,
            )
        )

        del (
            X_train,
            X_valid,
            y_train,
            y_valid_log,
            y_valid,
            pred_log,
            model,
        )
        gc.collect()

    # -----------------------------------------------------
    # Fold 결과 요약
    # -----------------------------------------------------

    results_df = pd.DataFrame(fold_results)

    mean_rmsle = results_df["rmsle"].mean()
    std_rmsle = results_df["rmsle"].std(ddof=1)

    best_fold = results_df.loc[
        results_df["rmsle"].idxmin(),
        "fold",
    ]

    worst_fold = results_df.loc[
        results_df["rmsle"].idxmax(),
        "fold",
    ]

    mean_best_iteration = int(
        round(results_df["best_iteration"].mean())
    )

    median_best_iteration = int(
        results_df["best_iteration"].median()
    )

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "rmsle": lambda value: f"{value:.6f}",
            },
        )
    )

    print(f"\nMean RMSLE : {mean_rmsle:.6f}")
    print(f"Std RMSLE  : {std_rmsle:.6f}")
    print(f"Best Fold  : {best_fold}")
    print(f"Worst Fold : {worst_fold}")

    print(
        f"\nMean Best Iteration: {mean_best_iteration}"
    )
    print(
        f"Median Best Iteration: {median_best_iteration}"
    )

    # -----------------------------------------------------
    # 피처 중요도 요약
    # -----------------------------------------------------

    all_importance = pd.concat(
        fold_importances,
        ignore_index=True,
    )

    mean_importance = summarize_feature_importance(
        all_importance
    )

    # -----------------------------------------------------
    # 최종 상태
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("BACKTEST STATUS")
    print("=" * 60)

    print("Folds:", len(FOLDS))
    print("Features:", len(FEATURES))
    print("Lags:", LAGS)

    print(
        "Rolling shift / windows:",
        ROLLING_SHIFT,
        ROLLING_WINDOWS,
    )

    print(
        "Early stopping rounds:",
        EARLY_STOPPING_ROUNDS,
    )

    print(f"Mean RMSLE: {mean_rmsle:.6f}")
    print(f"Std RMSLE: {std_rmsle:.6f}")

    return (
        results_df,
        mean_importance,
        all_importance,
    )


# =========================================================
# 11. 실행
# =========================================================

if __name__ == "__main__":

    (
        results_df,
        mean_importance,
        all_importance,
    ) = run_backtesting()