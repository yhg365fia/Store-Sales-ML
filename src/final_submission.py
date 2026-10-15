import gc

import numpy as np
import pandas as pd
import lightgbm as lgb

from main import main

from feature_engineering import (
    add_lag_features,
    add_rolling_features,
    LAGS,
    LAG_FEATURES,
    ROLLING_SHIFT,
    ROLLING_WINDOWS,
    ROLLING_MEAN_FEATURES,
)


# =========================================================
# 1. 최종 모델 설정
# =========================================================

# 현재 최고 Local 모델의 Fold 3:
# Best Iteration = 4995
# n_estimators=5000 상한까지 도달했고
# Early Stopping이 발동하지 않았음.
#
# 이번 제출에서는 Full Train을 5000 iteration으로 학습.
BEST_ITERATION = 5000

TARGET_COL = "sales_log"

FORECAST_HORIZON = 16


# =========================================================
# 2. 기본 피처
# =========================================================

BASE_FEATURE_COLS = [
    # 매장 / 상품군
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",

    # 날짜
    "year",
    "month",
    "day",
    "dayofweek",
    "is_weekend",

    # 프로모션
    "onpromotion",
    "is_promotion",

    # Holiday
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


CATEGORICAL_COLS = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
]


# =========================================================
# 3. 현재 최고 시계열 피처
# =========================================================

SELECTED_ROLLING_MEAN_FEATURES = [
    "sales_rolling_mean_16_7",
    "sales_rolling_mean_16_28",
]


TIME_SERIES_FEATURES = (
    list(LAG_FEATURES)
    + SELECTED_ROLLING_MEAN_FEATURES
)


FEATURE_COLS = (
    BASE_FEATURE_COLS
    + TIME_SERIES_FEATURES
)


# =========================================================
# 4. 설정 검증
# =========================================================

def validate_configuration():

    if tuple(LAGS) != (16, 21, 28):
        raise ValueError(
            "LAGS는 (16, 21, 28)이어야 합니다."
        )


    if ROLLING_SHIFT != 16:
        raise ValueError(
            "ROLLING_SHIFT는 16이어야 합니다."
        )


    if tuple(ROLLING_WINDOWS) != (7, 14, 28):
        raise ValueError(
            "ROLLING_WINDOWS는 "
            "(7, 14, 28)이어야 합니다."
        )


    expected_rolling_mean_features = [
        "sales_rolling_mean_16_7",
        "sales_rolling_mean_16_14",
        "sales_rolling_mean_16_28",
    ]


    if (
        list(ROLLING_MEAN_FEATURES)
        != expected_rolling_mean_features
    ):
        raise ValueError(
            "feature_engineering.py의 "
            "Rolling Mean 구성이 예상과 다릅니다.\n"
            f"예상: {expected_rolling_mean_features}\n"
            f"현재: {list(ROLLING_MEAN_FEATURES)}"
        )


    expected_ts_features = [
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_rolling_mean_16_7",
        "sales_rolling_mean_16_28",
    ]


    if TIME_SERIES_FEATURES != expected_ts_features:
        raise ValueError(
            "최종 시계열 피처 구성이 "
            "현재 최고 모델과 다릅니다.\n"
            f"예상: {expected_ts_features}\n"
            f"현재: {TIME_SERIES_FEATURES}"
        )


    if "sales_rolling_mean_16_14" in FEATURE_COLS:
        raise ValueError(
            "Rolling Mean 14가 "
            "최종 모델에 포함되어 있습니다."
        )


    std_features = [
        feature
        for feature in FEATURE_COLS
        if "rolling_std" in feature
    ]


    if std_features:
        raise ValueError(
            "Rolling Std가 최종 모델에 "
            f"포함되어 있습니다: {std_features}"
        )


    ema_features = [
        feature
        for feature in FEATURE_COLS
        if "ema" in feature.lower()
    ]


    if ema_features:
        raise ValueError(
            "EMA가 최종 모델에 "
            f"포함되어 있습니다: {ema_features}"
        )


    seasonal_features = [
        feature
        for feature in FEATURE_COLS
        if (
            "364" in feature
            or "728" in feature
            or "1092" in feature
            or "seasonal" in feature.lower()
        )
    ]


    if seasonal_features:
        raise ValueError(
            "탈락한 연간 계절성 피처가 "
            f"남아 있습니다: {seasonal_features}"
        )


    if len(FEATURE_COLS) != len(set(FEATURE_COLS)):
        raise ValueError(
            "FEATURE_COLS에 중복이 있습니다."
        )


    if len(FEATURE_COLS) != 28:
        raise ValueError(
            "최종 모델은 총 28개 피처여야 합니다. "
            f"현재: {len(FEATURE_COLS)}"
        )


# =========================================================
# 5. Final Submission
# =========================================================

def main_submission():

    validate_configuration()


    # =====================================================
    # 1. 전체 Train / Test 전처리
    # =====================================================

    train_fe, test_fe, holidays = main()

    del holidays
    gc.collect()

    train_fe = train_fe.copy()
    test_fe = test_fe.copy()

    train_count = len(train_fe)
    test_count = len(test_fe)


    print("\n" + "=" * 70)
    print("FINAL SUBMISSION START")
    print("=" * 70)

    print("Original Train:", train_fe.shape)
    print("Original Test :", test_fe.shape)


    # =====================================================
    # 2. Train / Test 표시
    # =====================================================

    train_fe["_is_test"] = False
    test_fe["_is_test"] = True

    train_fe["_test_order"] = -1
    test_fe["_test_order"] = np.arange(test_count)

    test_fe["sales"] = np.nan
    test_fe[TARGET_COL] = np.nan


    # =====================================================
    # 3. Train + Test 연결
    # =====================================================

    combined = pd.concat(
        [
            train_fe,
            test_fe,
        ],
        ignore_index=True,
        sort=False,
    )


    combined["date"] = pd.to_datetime(
        combined["date"]
    )


    group_cols = [
        "store_nbr",
        "family",
    ]


    key_cols = [
        "date",
        "store_nbr",
        "family",
    ]


    # =====================================================
    # 4. 기본 QC
    # =====================================================

    if combined[key_cols].isna().any().any():
        raise ValueError(
            "date / store_nbr / family에 "
            "결측치가 있습니다."
        )


    if combined.duplicated(key_cols).any():
        raise ValueError(
            "date × store_nbr × family가 "
            "중복되어 있습니다."
        )


    train_sales = combined.loc[
        ~combined["_is_test"],
        "sales",
    ]


    if train_sales.isna().any():
        raise ValueError(
            "Train sales에 결측치가 있습니다."
        )


    if (train_sales < 0).any():
        raise ValueError(
            "Train sales에 음수가 있습니다."
        )


    # =====================================================
    # 5. 날짜순 정렬
    # =====================================================

    combined = (
        combined
        .sort_values(
            group_cols + ["date"]
        )
        .reset_index(drop=True)
    )


    # =====================================================
    # 6. 날짜 연속성 확인
    # =====================================================

    date_diff = (
        combined
        .groupby(
            group_cols,
            observed=True
        )["date"]
        .diff()
    )


    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(
            pd.Timedelta(days=1)
        )
    )


    if invalid_gap.any():
        raise ValueError(
            "매장 × 상품군별 날짜가 "
            "연속적이지 않습니다."
        )


    # =====================================================
    # 7. Train / Test 경계 확인
    # =====================================================

    train_end = combined.loc[
        ~combined["_is_test"],
        "date",
    ].max()


    test_start = combined.loc[
        combined["_is_test"],
        "date",
    ].min()


    test_end = combined.loc[
        combined["_is_test"],
        "date",
    ].max()


    if (
        test_start
        != train_end + pd.Timedelta(days=1)
    ):
        raise ValueError(
            "Test가 Train 마지막 날 "
            "다음 날부터 시작하지 않습니다."
        )


    test_horizon = (
        test_end - train_end
    ).days


    if test_horizon != FORECAST_HORIZON:
        raise ValueError(
            "예상 Test 기간은 16일인데 "
            f"현재 {test_horizon}일입니다."
        )


    # =====================================================
    # 8. Test 그룹별 16일 확인
    # =====================================================

    test_group_counts = (
        combined.loc[
            combined["_is_test"]
        ]
        .groupby(
            group_cols,
            observed=True
        )
        .size()
    )


    if not test_group_counts.eq(
        FORECAST_HORIZON
    ).all():
        raise ValueError(
            "일부 Test store × family가 "
            "16일을 채우지 못했습니다."
        )


    # =====================================================
    # 9. Leakage 방지 확인
    # =====================================================

    if min(LAGS) < FORECAST_HORIZON:
        raise ValueError(
            "최소 Lag가 16일보다 짧아서 "
            "Test target leakage가 발생할 수 있습니다."
        )


    if ROLLING_SHIFT < FORECAST_HORIZON:
        raise ValueError(
            "Rolling shift가 16일보다 짧아서 "
            "Test target leakage가 발생할 수 있습니다."
        )


    # =====================================================
    # 10. Lag 생성
    # =====================================================

    combined = add_lag_features(
        combined,
        lags=LAGS,
    )


    # =====================================================
    # 11. Rolling 생성
    # =====================================================

    combined = add_rolling_features(
        combined,
        shift=ROLLING_SHIFT,
        windows=ROLLING_WINDOWS,
    )


    # =====================================================
    # 12. 생성 피처 존재 확인
    # =====================================================

    missing_features = [
        feature
        for feature in TIME_SERIES_FEATURES
        if feature not in combined.columns
    ]


    if missing_features:
        raise ValueError(
            "시계열 피처가 생성되지 않았습니다: "
            f"{missing_features}"
        )


    # =====================================================
    # 13. Lag 결측 QC
    # =====================================================

    group_position = (
        combined
        .groupby(
            group_cols,
            observed=True
        )
        .cumcount()
    )


    for lag in LAGS:

        feature = f"sales_lag_{lag}"

        expected_missing = (
            group_position < lag
        )

        actual_missing = (
            combined[feature].isna()
        )


        if not actual_missing.equals(
            expected_missing
        ):
            raise ValueError(
                f"{feature}: "
                "초기 과거 기록 부족 이외의 "
                "결측치가 발생했습니다."
            )


    # =====================================================
    # 14. Rolling Mean 결측 QC
    # =====================================================

    for window in (7, 28):

        feature = (
            f"sales_rolling_mean_"
            f"{ROLLING_SHIFT}_"
            f"{window}"
        )


        expected_missing = (
            group_position
            < (
                ROLLING_SHIFT
                + window
                - 1
            )
        )


        actual_missing = (
            combined[feature].isna()
        )


        if not actual_missing.equals(
            expected_missing
        ):
            raise ValueError(
                f"{feature}: "
                "초기 과거 기록 부족 이외의 "
                "결측치가 발생했습니다."
            )


    # =====================================================
    # 15. Train / Test 다시 분리
    # =====================================================

    full_train = (
        combined.loc[
            ~combined["_is_test"]
        ]
        .drop(
            columns=[
                "_is_test",
                "_test_order",
            ]
        )
        .copy()
    )


    final_test = (
        combined.loc[
            combined["_is_test"]
        ]
        .sort_values(
            "_test_order"
        )
        .drop(
            columns=[
                "_is_test",
                "_test_order",
                "sales",
                TARGET_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )


    if len(full_train) != train_count:
        raise ValueError(
            "시계열 피처 생성 전후 "
            "Train 행 수가 다릅니다."
        )


    if len(final_test) != test_count:
        raise ValueError(
            "시계열 피처 생성 전후 "
            "Test 행 수가 다릅니다."
        )


    # =====================================================
    # 16. Test 시계열 피처 결측 확인
    # =====================================================

    test_ts_missing = (
        final_test[
            TIME_SERIES_FEATURES
        ]
        .isna()
        .sum()
    )


    if (test_ts_missing > 0).any():
        raise ValueError(
            "Test 시계열 피처에 "
            "결측치가 있습니다.\n"
            f"{test_ts_missing}"
        )


    # =====================================================
    # 17. 상태 출력
    # =====================================================

    print("\n" + "=" * 70)
    print("TIME SERIES FEATURES")
    print("=" * 70)

    print("Lags:", LAGS)
    print("Rolling shift:", ROLLING_SHIFT)
    print(
        "Generated Rolling windows:",
        ROLLING_WINDOWS
    )

    print(
        "Rolling used in model:",
        [7, 28]
    )

    print(
        "Excluded Rolling Mean:",
        14
    )

    print(
        "Rolling Std:",
        "Not used"
    )

    print(
        "EMA:",
        "Not used"
    )

    print(
        "Seasonal Lag:",
        "Not used"
    )


    print(
        "\nModel time-series features:"
    )

    for feature in TIME_SERIES_FEATURES:
        print("-", feature)


    print(
        "\nTrain 시계열 피처 결측치:"
    )

    print(
        full_train[
            TIME_SERIES_FEATURES
        ]
        .isna()
        .sum()
    )


    print(
        "\nTest 시계열 피처 결측치:"
    )

    print(
        final_test[
            TIME_SERIES_FEATURES
        ]
        .isna()
        .sum()
    )


    # =====================================================
    # 18. 모델 입력 준비
    # =====================================================

    X_train = (
        full_train[
            FEATURE_COLS
        ].copy()
    )


    y_train = (
        full_train[
            TARGET_COL
        ].copy()
    )


    X_test = (
        final_test[
            FEATURE_COLS
        ].copy()
    )


    # =====================================================
    # 19. Target QC
    # =====================================================

    if y_train.isna().any():
        raise ValueError(
            "Train Target에 결측치가 있습니다."
        )


    expected_target = np.log1p(
        full_train[
            "sales"
        ].to_numpy(
            dtype=float
        )
    )


    if not np.allclose(
        y_train.to_numpy(
            dtype=float
        ),
        expected_target,
    ):
        raise ValueError(
            "sales_log가 log1p(sales)와 "
            "일치하지 않습니다."
        )


    # =====================================================
    # 20. 범주형 기준 통일
    # =====================================================

    for col in CATEGORICAL_COLS:

        categories = pd.Index(
            full_train[col]
            .dropna()
            .unique()
        )


        X_train[col] = pd.Categorical(
            X_train[col],
            categories=categories,
        )


        X_test[col] = pd.Categorical(
            X_test[col],
            categories=categories,
        )


    # =====================================================
    # 21. 모델 입력 QC
    # =====================================================

    if X_train[
        BASE_FEATURE_COLS
    ].isna().any().any():
        raise ValueError(
            "Train 기본 피처에 "
            "결측치가 있습니다."
        )


    # Train 초반 Lag / Rolling NaN은 정상.
    # LightGBM이 missing value로 처리.


    if X_test.isna().any().any():

        missing_test = (
            X_test
            .isna()
            .sum()
        )

        missing_test = missing_test[
            missing_test > 0
        ]


        raise ValueError(
            "Test 입력 피처에 "
            "결측치가 있습니다.\n"
            f"{missing_test}"
        )


    if len(FEATURE_COLS) != 28:
        raise ValueError(
            "최종 모델 피처 수가 "
            "28개가 아닙니다."
        )


    # =====================================================
    # 22. 최종 데이터 상태
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL TRAIN DATA")
    print("=" * 70)

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "Features:",
        len(FEATURE_COLS)
    )

    print(
        "Train Period:",
        full_train["date"].min(),
        "~",
        full_train["date"].max(),
    )

    print(
        "Test Period:",
        final_test["date"].min(),
        "~",
        final_test["date"].max(),
    )

    print(
        "Final Iterations:",
        BEST_ITERATION
    )


    # =====================================================
    # 23. 최종 LightGBM
    # =====================================================

    model = lgb.LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",

        learning_rate=0.05,

        n_estimators=BEST_ITERATION,

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


    # =====================================================
    # 24. 전체 Train 학습
    #
    # Validation 없음
    # Early Stopping 없음
    # 5000 iteration 고정
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING")
    print("=" * 70)

    print(
        "Iterations:",
        BEST_ITERATION
    )


    model.fit(
        X_train,
        y_train,
        categorical_feature=(
            CATEGORICAL_COLS
        ),
    )


    # =====================================================
    # 25. Test 예측
    # =====================================================

    pred_log = model.predict(
        X_test
    )


    if not np.isfinite(
        pred_log
    ).all():
        raise ValueError(
            "예측 log 값에 "
            "NaN 또는 무한대가 있습니다."
        )


    pred_log = np.maximum(
        pred_log,
        0.0
    )


    pred_sales = np.expm1(
        pred_log
    )


    pred_sales = np.maximum(
        pred_sales,
        0.0
    )


    # =====================================================
    # 26. Prediction 생성
    # =====================================================

    prediction = pd.DataFrame({
        "id":
            final_test[
                "id"
            ].to_numpy(),

        "sales":
            pred_sales,
    })


    # =====================================================
    # 27. Sample Submission 로드
    # =====================================================

    sample = pd.read_csv(
        "data/sample_submission.csv"
    )


    # =====================================================
    # 28. ID QC
    # =====================================================

    if not prediction[
        "id"
    ].is_unique:
        raise ValueError(
            "Test prediction id가 "
            "중복되어 있습니다."
        )


    if not sample[
        "id"
    ].is_unique:
        raise ValueError(
            "sample_submission id가 "
            "중복되어 있습니다."
        )


    if set(
        prediction["id"]
    ) != set(
        sample["id"]
    ):
        raise ValueError(
            "Test id와 "
            "sample_submission id가 "
            "일치하지 않습니다."
        )


    # =====================================================
    # 29. Sample 순서에 맞춰 병합
    # =====================================================

    submission = (
        sample[
            ["id"]
        ]
        .merge(
            prediction,
            on="id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )


    # =====================================================
    # 30. Submission QC
    # =====================================================

    if submission[
        "sales"
    ].isna().any():
        raise ValueError(
            "Submission sales에 "
            "결측치가 있습니다."
        )


    if len(submission) != len(sample):
        raise ValueError(
            "Submission 행 수가 "
            "sample_submission과 다릅니다."
        )


    if (
        submission[
            "sales"
        ] < 0
    ).any():
        raise ValueError(
            "Submission sales에 "
            "음수가 있습니다."
        )


    if not np.isfinite(
        submission[
            "sales"
        ]
    ).all():
        raise ValueError(
            "Submission sales에 "
            "무한대가 있습니다."
        )


    if not np.array_equal(
        submission[
            "id"
        ].to_numpy(),

        sample[
            "id"
        ].to_numpy(),
    ):
        raise ValueError(
            "Submission id 순서가 "
            "sample_submission과 다릅니다."
        )


    # =====================================================
    # 31. 예측값 요약
    # =====================================================

    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)

    print(
        submission[
            "sales"
        ].describe()
    )


    # =====================================================
    # 32. 저장
    # =====================================================

    output_path = (
        "submission_ts_best_iter5000.csv"
    )


    submission.to_csv(
        output_path,
        index=False,
    )


    print("\n" + "=" * 70)
    print("FINAL SUBMISSION")
    print("=" * 70)

    print(
        "Saved:",
        output_path
    )

    print(
        "Shape:",
        submission.shape
    )


    print(
        "\n"
        + submission
        .head(10)
        .to_string(
            index=False
        )
    )


    print(
        "\nFinal Model:"
    )

    print(
        "Base 23 "
        "+ Lag 16/21/28 "
        "+ Rolling Mean 7/28"
    )

    print(
        "Features:",
        len(FEATURE_COLS)
    )

    print(
        "Iterations:",
        BEST_ITERATION
    )


    return (
        model,
        submission,
    )


# =========================================================
# 33. 실행
# =========================================================

if __name__ == "__main__":

    model, submission = (
        main_submission()
    )