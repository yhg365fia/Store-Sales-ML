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
    add_lag_features,
)


# =========================================================
# 1. 기존 Base 23개
# =========================================================

BASE_FEATURES = [

    # Store / Product
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",

    # Calendar
    "year",
    "month",
    "day",
    "dayofweek",
    "is_weekend",

    # Promotion
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


# =========================================================
# 2. 현재 Champion Calendar 피처
# =========================================================

DAYOFYEAR_FEATURES = [
    "dayofyear",
]


# =========================================================
# 3. 이번 실험: Payday
# =========================================================

PAYDAY_FEATURES = [
    "is_15th",
    "is_month_end",
]


# =========================================================
# 4. 범주형
# =========================================================

CATEGORICAL_FEATURES = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
]


# =========================================================
# 5. 피처 구성
# =========================================================

# 기존 최고 성능 모델
# Base23 + Lag3 + dayofyear
# 23 + 3 + 1 = 27
BASELINE_FEATURES = (
    BASE_FEATURES
    + LAG_FEATURES
    + DAYOFYEAR_FEATURES
)


# 이번 실험
# Base23 + Lag3 + dayofyear + Payday2
# 23 + 3 + 1 + 2 = 29
PAYDAY_FEATURE_SET = (
    BASE_FEATURES
    + LAG_FEATURES
    + DAYOFYEAR_FEATURES
    + PAYDAY_FEATURES
)


TARGET_COL = "sales_log"

FORECAST_HORIZON = 16

EARLY_STOPPING_ROUNDS = 150


# =========================================================
# 6. 직전 Champion 결과
#    다시 학습하지 않고 비교용으로만 사용
# =========================================================

BASELINE_RESULTS = {
    "Fold 1": {
        "best_iteration": 5989,
        "rmsle": 0.382917,
    },
    "Fold 2": {
        "best_iteration": 3012,
        "rmsle": 0.405151,
    },
    "Fold 3": {
        "best_iteration": 5509,
        "rmsle": 0.396157,
    },
}


# =========================================================
# 7. Fold
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
# 8. 설정 검증
# =========================================================

def validate_configuration():

    expected_lags = (
        16,
        21,
        28,
    )

    if tuple(LAGS) != expected_lags:

        raise ValueError(
            f"LAGS는 {expected_lags}이어야 합니다. "
            f"현재: {tuple(LAGS)}"
        )


    expected_lag_features = [
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
    ]

    if list(LAG_FEATURES) != expected_lag_features:

        raise ValueError(
            "Lag Feature 구성이 다릅니다.\n"
            f"예상: {expected_lag_features}\n"
            f"현재: {list(LAG_FEATURES)}"
        )


    if len(BASE_FEATURES) != 23:

        raise ValueError(
            "BASE_FEATURES는 23개여야 합니다. "
            f"현재: {len(BASE_FEATURES)}"
        )


    if len(BASELINE_FEATURES) != 27:

        raise ValueError(
            "Champion Baseline은 "
            "27개 피처여야 합니다. "
            f"현재: {len(BASELINE_FEATURES)}"
        )


    if len(PAYDAY_FEATURE_SET) != 29:

        raise ValueError(
            "Payday Experiment는 "
            "29개 피처여야 합니다. "
            f"현재: {len(PAYDAY_FEATURE_SET)}"
        )


    if len(
        PAYDAY_FEATURE_SET
    ) != len(
        set(PAYDAY_FEATURE_SET)
    ):

        raise ValueError(
            "Payday Feature Set에 "
            "중복 피처가 있습니다."
        )


    required = {
        "dayofyear",
        "is_15th",
        "is_month_end",
    }

    if not required.issubset(
        set(PAYDAY_FEATURE_SET)
    ):

        raise ValueError(
            "dayofyear / is_15th / "
            "is_month_end 중 "
            "누락된 피처가 있습니다."
        )


    forbidden_keywords = [
        "rolling",
        "ema",
        "seasonal",
    ]

    forbidden_features = [

        feature

        for feature in PAYDAY_FEATURE_SET

        if (
            any(
                keyword in feature.lower()
                for keyword in forbidden_keywords
            )
            or feature == "sales_lag_364"
        )
    ]

    if forbidden_features:

        raise ValueError(
            "이번 실험에는 "
            "Rolling / EMA / Lag364가 "
            "들어가면 안 됩니다.\n"
            f"{forbidden_features}"
        )


# =========================================================
# 9. 데이터 준비
# =========================================================

def prepare_features(df):

    validate_configuration()

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )


    required_columns = [
        "date",
        "sales",
        *BASE_FEATURES,
        "dayofyear",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "필수 컬럼이 없습니다:\n"
            f"{missing_columns}"
        )


    keys = [
        "store_nbr",
        "family",
    ]


    # -----------------------------------------------------
    # Key QC
    # -----------------------------------------------------

    if df[
        keys + ["date"]
    ].isna().any().any():

        raise ValueError(
            "store_nbr / family / date에 "
            "결측치가 있습니다."
        )


    if df.duplicated(
        keys + ["date"]
    ).any():

        raise ValueError(
            "동일한 "
            "store × family × date가 "
            "중복되어 있습니다."
        )


    # -----------------------------------------------------
    # Sales QC
    # -----------------------------------------------------

    sales_values = (
        df["sales"]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        sales_values
    ).all():

        raise ValueError(
            "sales에 NaN 또는 "
            "무한대가 있습니다."
        )


    if (
        sales_values < 0
    ).any():

        raise ValueError(
            "sales에 음수가 있습니다."
        )


    # -----------------------------------------------------
    # dayofyear QC
    # -----------------------------------------------------

    expected_dayofyear = (
        df["date"]
        .dt.dayofyear
    )

    if not np.array_equal(
        df["dayofyear"].to_numpy(),
        expected_dayofyear.to_numpy(),
    ):

        raise ValueError(
            "dayofyear 계산값이 "
            "date와 일치하지 않습니다."
        )



    # -----------------------------------------------------
    # Payday QC
    # -----------------------------------------------------

    if not df[
        "is_15th"
    ].isin(
        [0, 1]
    ).all():

        raise ValueError(
            "is_15th 값은 "
            "0 또는 1이어야 합니다."
        )


    if not df[
        "is_month_end"
    ].isin(
        [0, 1]
    ).all():

        raise ValueError(
            "is_month_end 값은 "
            "0 또는 1이어야 합니다."
        )


    expected_15th = (
        df["date"].dt.day
        == 15
    ).astype(int)


    if not np.array_equal(
        df["is_15th"].to_numpy(),
        expected_15th.to_numpy(),
    ):

        raise ValueError(
            "is_15th 계산이 잘못되었습니다."
        )


    expected_month_end = (
        df["date"]
        .dt.is_month_end
        .astype(int)
    )


    if not np.array_equal(
        df["is_month_end"].to_numpy(),
        expected_month_end.to_numpy(),
    ):

        raise ValueError(
            "is_month_end 계산이 "
            "잘못되었습니다."
        )


    # -----------------------------------------------------
    # 정렬
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            keys + ["date"]
        )
        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # 날짜 연속성 QC
    # -----------------------------------------------------

    date_diff = (

        df

        .groupby(
            keys,
            observed=True,
        )["date"]

        .diff()
    )


    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(
            pd.Timedelta(
                days=1
            )
        )
    )


    if invalid_gap.any():

        raise ValueError(
            "매장 × 상품군 시계열의 "
            "날짜가 연속적이지 않습니다."
        )


    # -----------------------------------------------------
    # Lag16 / 21 / 28
    # -----------------------------------------------------

    df = add_lag_features(
        df,
        lags=LAGS,
    )


    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    df[
        TARGET_COL
    ] = np.log1p(
        df["sales"]
    )


    # -----------------------------------------------------
    # Lag QC
    # -----------------------------------------------------

    group_position = (

        df

        .groupby(
            keys,
            observed=True,
        )

        .cumcount()
    )


    for lag in LAGS:

        feature = (
            f"sales_lag_{lag}"
        )


        expected_missing = (
            group_position < lag
        )


        actual_missing = (
            df[
                feature
            ].isna()
        )


        if not actual_missing.equals(
            expected_missing
        ):

            raise ValueError(
                f"{feature}의 "
                "결측 위치가 예상과 다릅니다."
            )


    # -----------------------------------------------------
    # 출력
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PAYDAY FEATURE EXPERIMENT"
    )

    print(
        "=" * 75
    )


    print(
        "\nStored Champion Baseline:"
    )

    print(
        "Base23 + Lag16/21/28 "
        "+ dayofyear"
    )

    print(
        "Feature count:",
        len(BASELINE_FEATURES)
    )

    print(
        "Baseline is NOT retrained."
    )


    print(
        "\nExperiment:"
    )

    print(
        "Base23 + Lag16/21/28 "
        "+ dayofyear "
        "+ is_15th "
        "+ is_month_end"
    )

    print(
        "Feature count:",
        len(PAYDAY_FEATURE_SET)
    )


    print(
        "\nEarly stopping:",
        EARLY_STOPPING_ROUNDS
    )


    print(
        "\nLag missing:"
    )

    print(
        df[
            LAG_FEATURES
        ].isna().sum()
    )


    print(
        "\nPayday counts:"
    )

    print(
        df[
            PAYDAY_FEATURES
        ].sum()
    )


    print(
        "\nQC: OK"
    )


    return df


# =========================================================
# 10. Fold 분리
# =========================================================

def make_fold(
    df,
    fold_config,
):

    train_end = pd.Timestamp(
        fold_config[
            "train_end"
        ]
    )

    valid_start = pd.Timestamp(
        fold_config[
            "valid_start"
        ]
    )

    valid_end = pd.Timestamp(
        fold_config[
            "valid_end"
        ]
    )


    horizon = (
        valid_end
        - valid_start
    ).days + 1


    if horizon != FORECAST_HORIZON:

        raise ValueError(
            "Validation 기간은 "
            "16일이어야 합니다."
        )


    if (
        valid_start
        != train_end
        + pd.Timedelta(
            days=1
        )
    ):

        raise ValueError(
            "Validation은 Train 종료일 "
            "다음 날부터 시작해야 합니다."
        )


    if min(LAGS) < horizon:

        raise ValueError(
            "예측 기간보다 짧은 "
            "Lag가 있습니다."
        )


    train_df = df.loc[
        df["date"].le(
            train_end
        )
    ].copy()


    valid_df = df.loc[
        df["date"].between(
            valid_start,
            valid_end,
        )
    ].copy()


    if train_df.empty:

        raise ValueError(
            "Train이 비어 있습니다."
        )


    if valid_df.empty:

        raise ValueError(
            "Validation이 비어 있습니다."
        )


    keys = [
        "store_nbr",
        "family",
    ]


    valid_counts = (

        valid_df

        .groupby(
            keys,
            observed=True,
        )

        .size()
    )


    if not valid_counts.eq(
        horizon
    ).all():

        raise ValueError(
            "일부 Validation 그룹이 "
            "16일을 채우지 못했습니다."
        )


    print(
        "\nTrain:",
        train_df["date"].min(),
        "~",
        train_df["date"].max(),
    )

    print(
        "Valid:",
        valid_df["date"].min(),
        "~",
        valid_df["date"].max(),
    )

    print(
        "Train shape:",
        train_df.shape
    )

    print(
        "Valid shape:",
        valid_df.shape
    )


    return (
        train_df,
        valid_df,
    )


# =========================================================
# 11. 모델 입력 준비
# =========================================================

def prepare_model_data(
    train_df,
    valid_df,
):

    X_train = (
        train_df[
            PAYDAY_FEATURE_SET
        ].copy()
    )

    X_valid = (
        valid_df[
            PAYDAY_FEATURE_SET
        ].copy()
    )


    # -----------------------------------------------------
    # Category 정렬
    # -----------------------------------------------------

    for feature in CATEGORICAL_FEATURES:

        categories = pd.Index(

            train_df[
                feature
            ]

            .dropna()

            .unique()
        )


        X_train[
            feature
        ] = pd.Categorical(
            X_train[
                feature
            ],
            categories=categories,
        )


        X_valid[
            feature
        ] = pd.Categorical(
            X_valid[
                feature
            ],
            categories=categories,
        )


    # -----------------------------------------------------
    # Lag 외 피처 결측 확인
    # -----------------------------------------------------

    non_lag_features = [
        feature
        for feature in PAYDAY_FEATURE_SET
        if feature not in LAG_FEATURES
    ]


    if (
        X_train[
            non_lag_features
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Train의 Lag 외 피처에 "
            "결측치가 있습니다."
        )


    if (
        X_valid[
            non_lag_features
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Validation의 Lag 외 피처에 "
            "결측치가 있습니다."
        )


    # Validation은 충분한 과거가 있으므로
    # Lag 결측이 없어야 함
    if (
        X_valid[
            LAG_FEATURES
        ]
        .isna()
        .any()
        .any()
    ):

        missing_counts = (

            X_valid[
                LAG_FEATURES
            ]

            .isna()

            .sum()
        )


        raise ValueError(
            "Validation Lag에 "
            "결측치가 있습니다.\n"
            f"{missing_counts}"
        )


    y_train = (
        train_df[
            TARGET_COL
        ].copy()
    )


    y_valid_log = (
        valid_df[
            TARGET_COL
        ].copy()
    )


    y_valid = (
        valid_df[
            "sales"
        ].copy()
    )


    return (
        X_train,
        X_valid,
        y_train,
        y_valid_log,
        y_valid,
    )


# =========================================================
# 12. LightGBM
# =========================================================

def build_model():

    return LGBMRegressor(

        boosting_type="gbdt",

        objective="regression",

        learning_rate=0.05,

        n_estimators=6500,

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
# 13. RMSLE
# =========================================================

def calculate_rmsle(
    y_true,
    pred_log,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    pred_log = np.asarray(
        pred_log,
        dtype=float,
    )


    if not np.isfinite(
        pred_log
    ).all():

        raise ValueError(
            "예측값에 NaN 또는 "
            "무한대가 있습니다."
        )


    pred_log = np.maximum(
        pred_log,
        0.0,
    )


    return float(

        np.sqrt(

            np.mean(

                (
                    np.log1p(
                        y_true
                    )
                    - pred_log
                )
                ** 2
            )
        )
    )


# =========================================================
# 14. Feature Importance
# =========================================================

def get_feature_importance(
    model,
    fold_name,
):

    booster = model.booster_

    best_iteration = int(
        model.best_iteration_
    )


    gain = booster.feature_importance(
        importance_type="gain",
        iteration=best_iteration,
    )

    split = booster.feature_importance(
        importance_type="split",
        iteration=best_iteration,
    )


    importance = pd.DataFrame(
        {
            "fold": fold_name,
            "feature": booster.feature_name(),
            "gain": gain,
            "split": split,
        }
    )


    gain_total = (
        importance[
            "gain"
        ].sum()
    )

    split_total = (
        importance[
            "split"
        ].sum()
    )


    importance[
        "gain_pct"
    ] = (
        importance["gain"]
        / gain_total
        * 100
    )


    importance[
        "split_pct"
    ] = (
        importance["split"]
        / split_total
        * 100
    )


    return importance


# =========================================================
# 15. Payday 모델 하나 학습
# =========================================================

def train_payday_model(
    train_df,
    valid_df,
):

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


    print(
        "\n"
        + "-" * 75
    )

    print(
        "PAYDAY EXPERIMENT | "
        "Lag3 + dayofyear "
        "+ is_15th + is_month_end"
    )

    print(
        "-" * 75
    )

    print(
        "Feature count:",
        len(PAYDAY_FEATURE_SET)
    )


    model = build_model()


    model.fit(

        X_train,
        y_train,

        eval_set=[
            (
                X_valid,
                y_valid_log,
            )
        ],

        eval_metric="rmse",

        categorical_feature=(
            CATEGORICAL_FEATURES
        ),

        callbacks=[

            early_stopping(
                stopping_rounds=(
                    EARLY_STOPPING_ROUNDS
                ),
                verbose=True,
            ),

            log_evaluation(
                period=100
            ),
        ],
    )


    best_iteration = int(
        model.best_iteration_
    )


    pred_log = model.predict(
        X_valid,
        num_iteration=best_iteration,
    )


    rmsle = calculate_rmsle(
        y_true=y_valid,
        pred_log=pred_log,
    )


    print(
        f"Best iteration: "
        f"{best_iteration}"
    )

    print(
        f"RMSLE: "
        f"{rmsle:.6f}"
    )


    del X_train
    del X_valid
    del y_train
    del y_valid_log
    del y_valid
    del pred_log

    gc.collect()


    return (
        model,
        best_iteration,
        rmsle,
    )


# =========================================================
# 16. 3-Fold Payday Backtesting
# =========================================================

def run_backtesting():

    validate_configuration()


    # -----------------------------------------------------
    # 기존 Pipeline
    # -----------------------------------------------------

    train, test, holidays = main()

    del test
    del holidays

    gc.collect()


    # -----------------------------------------------------
    # Payday + Lag 생성
    # -----------------------------------------------------

    train = prepare_features(
        train
    )


    fold_results = []

    fold_importances = []


    # =====================================================
    # Fold 반복
    # =====================================================

    for fold_config in FOLDS:

        fold_name = (
            fold_config[
                "fold"
            ]
        )


        print(
            "\n\n"
            + "=" * 75
        )

        print(
            f"{fold_name} START"
        )

        print(
            "=" * 75
        )


        (
            train_df,
            valid_df,
        ) = make_fold(
            train,
            fold_config,
        )


        # -------------------------------------------------
        # 기존 Champion 결과 불러오기
        # -------------------------------------------------

        baseline_iteration = (
            BASELINE_RESULTS[
                fold_name
            ][
                "best_iteration"
            ]
        )

        baseline_rmsle = (
            BASELINE_RESULTS[
                fold_name
            ][
                "rmsle"
            ]
        )


        print(
            "\nStored Champion Baseline"
        )

        print(
            "Base23 + Lag3 + dayofyear"
        )

        print(
            f"Best iteration: "
            f"{baseline_iteration}"
        )

        print(
            f"RMSLE: "
            f"{baseline_rmsle:.6f}"
        )

        print(
            "Baseline retraining: SKIPPED"
        )


        # -------------------------------------------------
        # Payday 모델만 학습
        # -------------------------------------------------

        (
            payday_model,
            payday_iteration,
            payday_rmsle,
        ) = train_payday_model(
            train_df,
            valid_df,
        )


        fold_importances.append(
            get_feature_importance(
                payday_model,
                fold_name,
            )
        )


        delta = (
            payday_rmsle
            - baseline_rmsle
        )


        print(
            "\n"
            + "=" * 75
        )

        print(
            f"{fold_name} COMPARISON"
        )

        print(
            "=" * 75
        )

        print(
            f"Champion baseline : "
            f"{baseline_rmsle:.6f}"
        )

        print(
            f"Payday experiment : "
            f"{payday_rmsle:.6f}"
        )

        print(
            f"Delta             : "
            f"{delta:+.6f}"
        )


        fold_results.append(
            {
                "fold":
                    fold_name,

                "baseline_best_iteration":
                    baseline_iteration,

                "baseline_rmsle":
                    baseline_rmsle,

                "payday_best_iteration":
                    payday_iteration,

                "payday_rmsle":
                    payday_rmsle,

                "delta_vs_baseline":
                    delta,
            }
        )


        del payday_model
        del train_df
        del valid_df

        gc.collect()


    # =====================================================
    # 결과
    # =====================================================

    results_df = pd.DataFrame(
        fold_results
    )


    baseline_mean = (
        results_df[
            "baseline_rmsle"
        ].mean()
    )

    payday_mean = (
        results_df[
            "payday_rmsle"
        ].mean()
    )


    baseline_std = (
        results_df[
            "baseline_rmsle"
        ].std(
            ddof=1
        )
    )

    payday_std = (
        results_df[
            "payday_rmsle"
        ].std(
            ddof=1
        )
    )


    mean_delta = (
        payday_mean
        - baseline_mean
    )


    improved_folds = int(
        (
            results_df[
                "payday_rmsle"
            ]
            <
            results_df[
                "baseline_rmsle"
            ]
        ).sum()
    )


    # -----------------------------------------------------
    # Fold 결과 출력
    # -----------------------------------------------------

    print(
        "\n\n"
        + "=" * 90
    )

    print(
        "PAYDAY BACKTEST RESULTS"
    )

    print(
        "=" * 90
    )


    print(

        results_df.to_string(

            index=False,

            formatters={

                "baseline_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "payday_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "delta_vs_baseline":
                    lambda x:
                    f"{x:+.6f}",
            },
        )
    )


    # -----------------------------------------------------
    # 평균 결과
    # -----------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "model":
                    "Champion baseline",

                "features":
                    len(
                        BASELINE_FEATURES
                    ),

                "mean_rmsle":
                    baseline_mean,

                "std_rmsle":
                    baseline_std,

                "delta":
                    0.0,
            },

            {
                "model":
                    "Payday",

                "features":
                    len(
                        PAYDAY_FEATURE_SET
                    ),

                "mean_rmsle":
                    payday_mean,

                "std_rmsle":
                    payday_std,

                "delta":
                    mean_delta,
            },
        ]
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "MEAN RMSLE COMPARISON"
    )

    print(
        "=" * 90
    )


    print(

        summary_df.to_string(

            index=False,

            formatters={

                "mean_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "std_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "delta":
                    lambda x:
                    f"{x:+.6f}",
            },
        )
    )


    print(
        "\nPayday improved:",
        f"{improved_folds}/3 folds"
    )


    # =====================================================
    # Payday Importance
    # =====================================================

    all_importance = pd.concat(
        fold_importances,
        ignore_index=True,
    )


    payday_importance = (

        all_importance.loc[
            all_importance[
                "feature"
            ].isin(
                PAYDAY_FEATURES
            )
        ]

        .groupby(
            "feature",
            as_index=False,
        )

        .agg(
            mean_gain_pct=(
                "gain_pct",
                "mean",
            ),

            mean_split_pct=(
                "split_pct",
                "mean",
            ),

            mean_split=(
                "split",
                "mean",
            ),
        )

        .sort_values(
            "mean_gain_pct",
            ascending=False,
        )
    )


    print(
        "\n"
        + "=" * 75
    )

    print(
        "PAYDAY FEATURE IMPORTANCE"
    )

    print(
        "=" * 75
    )


    print(
        payday_importance
        .round(4)
        .to_string(
            index=False
        )
    )


    # =====================================================
    # 최종 판정
    # =====================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EXPERIMENT DECISION"
    )

    print(
        "=" * 90
    )


    if mean_delta < 0:

        print(
            "RESULT: PAYDAY IMPROVED"
        )

        print(
            f"Mean improvement: "
            f"{-mean_delta:.6f}"
        )

    elif mean_delta > 0:

        print(
            "RESULT: PAYDAY WORSE"
        )

        print(
            f"Mean degradation: "
            f"{mean_delta:.6f}"
        )

    else:

        print(
            "RESULT: NO DIFFERENCE"
        )


    print(
        "Improved folds:",
        f"{improved_folds}/3"
    )

    print(
        "Baseline retrained:",
        False
    )

    print(
        "n_estimators:",
        6500
    )

    print(
        "Early stopping:",
        EARLY_STOPPING_ROUNDS
    )


    return (
        results_df,
        summary_df,
        payday_importance,
        all_importance,
    )


# =========================================================
# 17. 실행
# =========================================================

if __name__ == "__main__":

    (
        results_df,
        summary_df,
        payday_importance,
        all_importance,
    ) = run_backtesting()