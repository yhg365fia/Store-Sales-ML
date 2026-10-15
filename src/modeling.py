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
# 1. 기본 피처
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


# =========================================================
# 2. 현재 최고 Rolling 구성
# =========================================================

# Mean14는 이전 실험에서 제거했을 때 개선되었으므로
# 모델에서는 Mean7 / Mean28만 사용

SELECTED_ROLLING_MEAN_FEATURES = [
    "sales_rolling_mean_16_7",
    "sales_rolling_mean_16_28",
]


# =========================================================
# 3. 1년 주기 Lag
# =========================================================

# 364일 = 52주
# 약 1년 전의 같은 요일 판매량

YEAR_LAG = 364

YEAR_LAG_FEATURE = "sales_lag_364"


# =========================================================
# 4. 모델 입력 피처
# =========================================================

TIME_SERIES_FEATURES = (
    LAG_FEATURES
    + SELECTED_ROLLING_MEAN_FEATURES
    + [YEAR_LAG_FEATURE]
)


FEATURES = (
    BASE_FEATURES
    + TIME_SERIES_FEATURES
)


TARGET_COL = "sales_log"

FORECAST_HORIZON = 16

EARLY_STOPPING_ROUNDS = 150


# =========================================================
# 5. 현재 최고 모델 결과
# =========================================================

BASELINE_RMSLE = 0.391424


BASELINE_FOLD_RMSLE = {
    "Fold 1": 0.385976,
    "Fold 2": 0.393941,
    "Fold 3": 0.394357,
}


# =========================================================
# 6. Rolling Backtesting Fold
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
# 7. 설정 확인
# =========================================================

def validate_configuration():

    if tuple(LAGS) != (16, 21, 28):

        raise ValueError(
            "feature_engineering.py의 LAGS는 "
            "(16, 21, 28)이어야 합니다."
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


    expected_generated_rolling = [
        "sales_rolling_mean_16_7",
        "sales_rolling_mean_16_14",
        "sales_rolling_mean_16_28",
    ]


    if (
        ROLLING_MEAN_FEATURES
        != expected_generated_rolling
    ):

        raise ValueError(
            "생성되는 Rolling Mean 구성이 "
            "예상과 다릅니다.\n"
            f"예상: {expected_generated_rolling}\n"
            f"현재: {ROLLING_MEAN_FEATURES}"
        )


    expected_time_series_features = [
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_rolling_mean_16_7",
        "sales_rolling_mean_16_28",
        "sales_lag_364",
    ]


    if (
        TIME_SERIES_FEATURES
        != expected_time_series_features
    ):

        raise ValueError(
            "이번 실험의 시계열 피처가 "
            "예상과 다릅니다.\n"
            f"예상: {expected_time_series_features}\n"
            f"현재: {TIME_SERIES_FEATURES}"
        )


    # Mean14는 생성만 하고 모델에는 넣지 않음
    if (
        "sales_rolling_mean_16_14"
        in FEATURES
    ):

        raise ValueError(
            "Rolling Mean 14가 모델 입력에 "
            "포함되어 있습니다."
        )


    # EMA는 이번 실험에서 전부 제외
    ema_features = [
        feature
        for feature in FEATURES
        if "ema" in feature.lower()
    ]


    if ema_features:

        raise ValueError(
            "이번 Lag364 실험에는 "
            "EMA가 들어가면 안 됩니다.\n"
            f"{ema_features}"
        )


    if len(FEATURES) != len(set(FEATURES)):

        raise ValueError(
            "FEATURES에 중복 피처가 있습니다."
        )


    # Base23
    # + Lag3
    # + Rolling2
    # + Lag364 1
    # = 29

    if len(FEATURES) != 29:

        raise ValueError(
            f"이번 실험은 총 29개 피처여야 합니다. "
            f"현재: {len(FEATURES)}개"
        )


# =========================================================
# 8. Lag 364 생성
# =========================================================

def add_year_lag_feature(df):

    """
    매장 × 상품군별 sales_lag_364 생성.

    364일 = 52주이므로
    약 1년 전의 같은 요일 판매량을 가져온다.

    예:
    2017년 화요일
    → 364일 전도 화요일

    Validation/Test의 미래 sales는 사용하지 않는다.
    """

    if YEAR_LAG < FORECAST_HORIZON:

        raise ValueError(
            "YEAR_LAG는 예측 기간보다 "
            "짧을 수 없습니다."
        )


    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )


    keys = [
        "store_nbr",
        "family",
    ]


    df = (
        df
        .sort_values(
            keys + ["date"]
        )
        .reset_index(drop=True)
    )


    df[YEAR_LAG_FEATURE] = (

        df
        .groupby(
            keys,
            observed=True
        )["sales"]

        .shift(
            YEAR_LAG
        )
    )


    return df


# =========================================================
# 9. 시계열 피처 전체 생성
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
            f"필수 컬럼이 없습니다: "
            f"{missing_columns}"
        )


    df["date"] = pd.to_datetime(
        df["date"]
    )


    keys = [
        "store_nbr",
        "family",
    ]


    # -----------------------------------------------------
    # 기본 QC
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
            "동일한 store_nbr × family × date가 "
            "중복되어 있습니다."
        )


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
            "sales에 NaN 또는 무한대가 있습니다."
        )


    if (
        sales_values < 0
    ).any():

        raise ValueError(
            "sales에 음수가 있습니다."
        )


    # -----------------------------------------------------
    # 정렬
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            keys + ["date"]
        )
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # 날짜 연속성 QC
    #
    # Lag는 row 기준이므로
    # 1 row = 1 day여야 함
    # -----------------------------------------------------

    date_diff = (

        df
        .groupby(
            keys,
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
            "매장 × 상품군 시계열의 날짜가 "
            "연속적이지 않습니다."
        )


    # -----------------------------------------------------
    # Lag 16 / 21 / 28
    # -----------------------------------------------------

    df = add_lag_features(
        df,
        lags=LAGS,
    )


    # -----------------------------------------------------
    # Rolling Mean 7 / 14 / 28 생성
    #
    # 모델에서는 Mean7 / Mean28만 사용
    # -----------------------------------------------------

    df = add_rolling_features(
        df,
        shift=ROLLING_SHIFT,
        windows=ROLLING_WINDOWS,
    )


    # -----------------------------------------------------
    # Lag 364 추가
    # -----------------------------------------------------

    df = add_year_lag_feature(
        df
    )


    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    df[TARGET_COL] = np.log1p(
        df["sales"]
    )


    # -----------------------------------------------------
    # 모델 피처 존재 확인
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]


    if missing_features:

        raise ValueError(
            f"생성되지 않은 모델 피처가 있습니다: "
            f"{missing_features}"
        )


    # =====================================================
    # 초기 결측 QC
    # =====================================================

    group_position = (

        df
        .groupby(
            keys,
            observed=True
        )
        .cumcount()
    )


    initial_missing_rows = {}


    # -----------------------------------------------------
    # Lag
    # -----------------------------------------------------

    for lag in LAGS:

        initial_missing_rows[
            f"sales_lag_{lag}"
        ] = lag


    # -----------------------------------------------------
    # Rolling
    # -----------------------------------------------------

    for window in ROLLING_WINDOWS:

        feature = (
            f"sales_rolling_mean_"
            f"{ROLLING_SHIFT}_"
            f"{window}"
        )


        initial_missing_rows[
            feature
        ] = (
            ROLLING_SHIFT
            + window
            - 1
        )


    # -----------------------------------------------------
    # Lag364
    # -----------------------------------------------------

    initial_missing_rows[
        YEAR_LAG_FEATURE
    ] = YEAR_LAG


    # -----------------------------------------------------
    # 실제 결측 위치와 예상 결측 위치 비교
    # -----------------------------------------------------

    for (
        feature,
        initial_rows
    ) in initial_missing_rows.items():

        expected_missing = (
            group_position.lt(
                initial_rows
            )
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
                f"{feature}: "
                f"결측 위치가 예상과 다릅니다."
            )


        valid_values = (
            df[
                feature
            ]
            .dropna()
            .to_numpy()
        )


        if not np.isfinite(
            valid_values
        ).all():

            raise ValueError(
                f"{feature}에 "
                f"무한대 값이 있습니다."
            )


    # =====================================================
    # 상태 출력
    # =====================================================

    print(
        "\n===== TIME SERIES FEATURES ====="
    )


    print(
        "Short Lags:",
        LAGS
    )


    print(
        "Year Lag:",
        YEAR_LAG
    )


    print(
        "Rolling shift:",
        ROLLING_SHIFT
    )


    print(
        "Generated rolling windows:",
        ROLLING_WINDOWS
    )


    print(
        "Rolling used:",
        [7, 28]
    )


    print(
        "\nLag 피처:",
        LAG_FEATURES
    )


    print(
        "Year Lag 피처:",
        YEAR_LAG_FEATURE
    )


    print(
        "생성된 Rolling:",
        ROLLING_MEAN_FEATURES
    )


    print(
        "모델에 사용하는 Rolling:",
        SELECTED_ROLLING_MEAN_FEATURES
    )


    print(
        "\n제외:"
    )

    print(
        "- Rolling Mean 14"
    )

    print(
        "- EMA 전체"
    )


    print(
        "\n시계열 모델 입력 피처 수:",
        len(TIME_SERIES_FEATURES)
    )


    print(
        "전체 입력 피처 수:",
        len(FEATURES)
    )


    print(
        "Early stopping rounds:",
        EARLY_STOPPING_ROUNDS
    )


    qc_features = (
        LAG_FEATURES
        + ROLLING_MEAN_FEATURES
        + [YEAR_LAG_FEATURE]
    )


    print(
        "\n초기 과거 데이터 부족 결측치:"
    )


    print(
        df[
            qc_features
        ].isna().sum()
    )


    print(
        "\nLag + Rolling + Lag364 QC: OK"
    )


    return df


# =========================================================
# 10. Fold 분리
# =========================================================

def make_fold(
    df,
    fold_config
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
            f"Validation은 "
            f"{FORECAST_HORIZON}일이어야 합니다."
        )


    if (
        valid_start
        != train_end
        + pd.Timedelta(days=1)
    ):

        raise ValueError(
            "Validation은 Train 종료일 "
            "다음 날부터 시작해야 합니다."
        )


    # 모든 target 기반 피처는
    # 16일 예측 구간 이전의 값만 사용

    if min(
        *LAGS,
        ROLLING_SHIFT,
        YEAR_LAG,
    ) < horizon:

        raise ValueError(
            "예측 기간보다 짧은 "
            "Lag / Shift가 있습니다."
        )


    train_df = df.loc[
        df["date"].le(
            train_end
        )
    ].copy()


    valid_df = df.loc[
        df["date"].between(
            valid_start,
            valid_end
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


    train_groups = (
        train_df
        .groupby(
            keys,
            observed=True
        )
        .size()
        .index
    )


    valid_counts = (
        valid_df
        .groupby(
            keys,
            observed=True
        )
        .size()
    )


    if len(
        train_groups.difference(
            valid_counts.index
        )
    ) > 0:

        raise ValueError(
            "Train에만 존재하는 "
            "매장 × 상품군이 있습니다."
        )


    if len(
        valid_counts.index.difference(
            train_groups
        )
    ) > 0:

        raise ValueError(
            "Validation에만 존재하는 "
            "매장 × 상품군이 있습니다."
        )


    if not valid_counts.eq(
        horizon
    ).all():

        raise ValueError(
            "Validation의 일부 "
            "매장 × 상품군이 "
            "16일을 채우지 못했습니다."
        )


    print(
        f"Train: "
        f"{train_df['date'].min()} "
        f"~ "
        f"{train_df['date'].max()}"
    )


    print(
        f"Valid: "
        f"{valid_df['date'].min()} "
        f"~ "
        f"{valid_df['date'].max()}"
    )


    print(
        "\nTrain shape:",
        train_df.shape
    )


    print(
        "Valid shape:",
        valid_df.shape
    )


    return (
        train_df,
        valid_df
    )


# =========================================================
# 11. 모델 입력 준비
# =========================================================

def prepare_model_data(
    train_df,
    valid_df
):

    X_train = (
        train_df[
            FEATURES
        ].copy()
    )


    X_valid = (
        valid_df[
            FEATURES
        ].copy()
    )


    # -----------------------------------------------------
    # 범주형 category 통일
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
    # 기본 피처 결측
    # -----------------------------------------------------

    if X_train[
        BASE_FEATURES
    ].isna().any().any():

        raise ValueError(
            "Train 기본 피처에 "
            "결측치가 있습니다."
        )


    if X_valid[
        BASE_FEATURES
    ].isna().any().any():

        raise ValueError(
            "Validation 기본 피처에 "
            "결측치 또는 새로운 범주가 있습니다."
        )


    # -----------------------------------------------------
    # Validation 시계열 피처 결측
    # -----------------------------------------------------

    if X_valid[
        TIME_SERIES_FEATURES
    ].isna().any().any():

        missing_counts = (

            X_valid[
                TIME_SERIES_FEATURES
            ]

            .isna()
            .sum()
        )


        raise ValueError(
            "Validation 시계열 피처에 "
            "결측치가 있습니다.\n"
            f"{missing_counts}"
        )


    # Train 초반의 target-based feature NaN은
    # LightGBM이 그대로 처리


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


    print(
        "\nX_train:",
        X_train.shape
    )


    print(
        "X_valid:",
        X_valid.shape
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

    model = LGBMRegressor(

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


    return model


# =========================================================
# 13. RMSLE
# =========================================================

def calculate_rmsle(
    y_true,
    pred_log
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )


    pred_log = np.asarray(
        pred_log,
        dtype=float
    )


    if not np.isfinite(
        pred_log
    ).all():

        raise ValueError(
            "예측값에 NaN 또는 "
            "무한대가 있습니다."
        )


    # sales >= 0이므로
    # log1p prediction 역시 최소 0으로 제한

    pred_log = np.maximum(
        pred_log,
        0.0
    )


    rmsle = np.sqrt(

        np.mean(

            (
                np.log1p(
                    y_true
                )
                - pred_log
            ) ** 2
        )
    )


    return float(
        rmsle
    )


# =========================================================
# 14. Feature Group
# =========================================================

def get_feature_group(
    feature
):

    if feature == YEAR_LAG_FEATURE:

        return "Year Lag"


    if feature in LAG_FEATURES:

        return "Lag"


    if (
        feature
        in SELECTED_ROLLING_MEAN_FEATURES
    ):

        return "Rolling Mean"


    return "Base"


# =========================================================
# 15. Feature Importance
# =========================================================

def get_feature_importance(
    model,
    fold_name
):

    booster = model.booster_


    iteration = (
        model.best_iteration_
    )


    gain = booster.feature_importance(
        importance_type="gain",
        iteration=iteration,
    )


    split = booster.feature_importance(
        importance_type="split",
        iteration=iteration,
    )


    importance = pd.DataFrame({

        "fold":
            fold_name,

        "feature":
            booster.feature_name(),

        "gain":
            gain,

        "split":
            split,
    })


    importance[
        "group"
    ] = (
        importance[
            "feature"
        ].map(
            get_feature_group
        )
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
        importance[
            "gain"
        ]
        / gain_total
        * 100
        if gain_total > 0
        else 0
    )


    importance[
        "split_pct"
    ] = (
        importance[
            "split"
        ]
        / split_total
        * 100
        if split_total > 0
        else 0
    )


    return importance


# =========================================================
# 16. 평균 Feature Importance
# =========================================================

def summarize_feature_importance(
    all_importance
):

    mean_importance = (

        all_importance

        .groupby(
            [
                "feature",
                "group",
            ],
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
    )


    mean_importance[
        "gain_rank"
    ] = (

        mean_importance[
            "mean_gain_pct"
        ]

        .rank(
            method="min",
            ascending=False,
        )

        .astype(int)
    )


    mean_importance[
        "split_rank"
    ] = (

        mean_importance[
            "mean_split"
        ]

        .rank(
            method="min",
            ascending=False,
        )

        .astype(int)
    )


    mean_importance = (

        mean_importance

        .sort_values(
            [
                "gain_rank",
                "split_rank",
                "feature",
            ]
        )

        .reset_index(
            drop=True
        )
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


    mean_importance = (
        mean_importance[
            columns
        ]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "MEAN FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )


    print(

        mean_importance
        .round(2)
        .to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # 시계열 피처만 출력
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TIME SERIES FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )


    ts_importance = (

        mean_importance.loc[

            mean_importance[
                "group"
            ].ne(
                "Base"
            ),

            [
                "feature",
                "group",
                "gain_rank",
                "mean_gain_pct",
                "split_rank",
                "mean_split_pct",
            ],
        ]
    )


    print(

        ts_importance
        .round(2)
        .to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # 그룹별 중요도
    # -----------------------------------------------------

    group_importance = (

        mean_importance

        .groupby(
            "group",
            as_index=False,
        )

        .agg(

            n_features=(
                "feature",
                "size",
            ),

            gain_pct=(
                "mean_gain_pct",
                "sum",
            ),

            split_pct=(
                "mean_split_pct",
                "sum",
            ),
        )

        .sort_values(
            "gain_pct",
            ascending=False,
        )

        .reset_index(
            drop=True
        )
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "IMPORTANCE BY FEATURE GROUP (%)"
    )

    print(
        "=" * 70
    )


    print(

        group_importance
        .round(2)
        .to_string(
            index=False
        )
    )


    return mean_importance


# =========================================================
# 17. Backtesting
# =========================================================

def run_backtesting():

    validate_configuration()


    # -----------------------------------------------------
    # 전처리 + 기본 Feature Engineering
    # -----------------------------------------------------

    train, test, holidays = main()


    del test
    del holidays

    gc.collect()


    # -----------------------------------------------------
    # Lag + Rolling + Lag364
    # -----------------------------------------------------

    train = prepare_time_series_features(
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
            "\n"
            + "=" * 70
        )

        print(
            f"{fold_name} START"
        )

        print(
            "=" * 70
        )


        (
            train_df,
            valid_df
        ) = make_fold(
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


        del train_df
        del valid_df

        gc.collect()


        # -------------------------------------------------
        # Fold마다 새 모델
        # -------------------------------------------------

        model = build_model()


        model.fit(

            X_train,
            y_train,

            eval_set=[
                (
                    X_valid,
                    y_valid_log
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
            num_iteration=(
                best_iteration
            ),
        )


        rmsle = calculate_rmsle(
            y_true=y_valid,
            pred_log=pred_log,
        )


        baseline_fold = (
            BASELINE_FOLD_RMSLE[
                fold_name
            ]
        )


        delta = (
            rmsle
            - baseline_fold
        )


        print(
            "\n===== VALIDATION EVALUATION ====="
        )


        print(
            f"{fold_name} RMSLE: "
            f"{rmsle:.6f}"
        )


        print(
            f"{fold_name} Baseline: "
            f"{baseline_fold:.6f}"
        )


        print(
            f"{fold_name} Delta: "
            f"{delta:+.6f}"
        )


        print(
            f"{fold_name} "
            f"Best Iteration: "
            f"{best_iteration}"
        )


        fold_results.append({

            "fold":
                fold_name,

            "train_end":
                fold_config[
                    "train_end"
                ],

            "valid_start":
                fold_config[
                    "valid_start"
                ],

            "valid_end":
                fold_config[
                    "valid_end"
                ],

            "baseline_rmsle":
                baseline_fold,

            "rmsle":
                rmsle,

            "delta":
                delta,

            "best_iteration":
                best_iteration,
        })


        fold_importances.append(

            get_feature_importance(
                model=model,
                fold_name=fold_name,
            )
        )


        del X_train
        del X_valid
        del y_train
        del y_valid_log
        del y_valid
        del pred_log
        del model

        gc.collect()


    # =====================================================
    # 18. Backtest 결과
    # =====================================================

    results_df = pd.DataFrame(
        fold_results
    )


    mean_rmsle = (
        results_df[
            "rmsle"
        ].mean()
    )


    std_rmsle = (
        results_df[
            "rmsle"
        ].std(
            ddof=1
        )
    )


    mean_best_iteration = int(
        round(
            results_df[
                "best_iteration"
            ].mean()
        )
    )


    median_best_iteration = int(
        results_df[
            "best_iteration"
        ].median()
    )


    overall_delta = (
        mean_rmsle
        - BASELINE_RMSLE
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKTEST RESULTS"
    )

    print(
        "=" * 70
    )


    print(

        results_df.to_string(

            index=False,

            formatters={

                "baseline_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "rmsle":
                    lambda x:
                    f"{x:.6f}",

                "delta":
                    lambda x:
                    f"{x:+.6f}",
            },
        )
    )


    print(
        f"\nBaseline Mean RMSLE : "
        f"{BASELINE_RMSLE:.6f}"
    )


    print(
        f"Experiment Mean RMSLE: "
        f"{mean_rmsle:.6f}"
    )


    print(
        f"Delta                : "
        f"{overall_delta:+.6f}"
    )


    print(
        f"Std RMSLE            : "
        f"{std_rmsle:.6f}"
    )


    print(
        f"\nMean Best Iteration  : "
        f"{mean_best_iteration}"
    )


    print(
        f"Median Best Iteration: "
        f"{median_best_iteration}"
    )


    # =====================================================
    # 19. 실험 판정
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT COMPARISON"
    )

    print(
        "=" * 70
    )


    print(
        "Baseline:"
    )


    print(
        "Lag 16 / 21 / 28 "
        "+ Rolling Mean 7 / 28"
    )


    print(
        f"RMSLE = "
        f"{BASELINE_RMSLE:.6f}"
    )


    print(
        "\nExperiment:"
    )


    print(
        "Lag 16 / 21 / 28 "
        "+ Lag 364 "
        "+ Rolling Mean 7 / 28"
    )


    print(
        f"RMSLE = "
        f"{mean_rmsle:.6f}"
    )


    print(
        f"\n변화량 = "
        f"{overall_delta:+.6f}"
    )


    if mean_rmsle < BASELINE_RMSLE:

        print(
            "\n판정: 개선"
        )

        print(
            "→ sales_lag_364 채택 후보"
        )


    elif mean_rmsle > BASELINE_RMSLE:

        print(
            "\n판정: 악화"
        )

        print(
            "→ sales_lag_364 제외"
        )


    else:

        print(
            "\n판정: 동일"
        )


    # =====================================================
    # 20. 피처 중요도
    # =====================================================

    all_importance = pd.concat(
        fold_importances,
        ignore_index=True,
    )


    mean_importance = (
        summarize_feature_importance(
            all_importance
        )
    )


    # =====================================================
    # 21. 최종 상태
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKTEST STATUS"
    )

    print(
        "=" * 70
    )


    print(
        "Experiment:",
        "Add sales_lag_364"
    )


    print(
        "Folds:",
        len(FOLDS)
    )


    print(
        "Features:",
        len(FEATURES)
    )


    print(
        "Short Lags:",
        LAGS
    )


    print(
        "Year Lag:",
        YEAR_LAG
    )


    print(
        "Rolling Used:",
        [7, 28]
    )


    print(
        "EMA:",
        "Not used"
    )


    print(
        "Early stopping rounds:",
        EARLY_STOPPING_ROUNDS
    )


    print(
        f"Mean RMSLE: "
        f"{mean_rmsle:.6f}"
    )


    print(
        f"Std RMSLE: "
        f"{std_rmsle:.6f}"
    )


    return (
        results_df,
        mean_importance,
        all_importance,
    )


# =========================================================
# 22. 실행
# =========================================================

if __name__ == "__main__":

    (
        results_df,
        mean_importance,
        all_importance,
    ) = run_backtesting()