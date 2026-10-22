"""
External Feature Experiments — Transactions 2종 + Oil 3종

실험 5개:
1. transactions_lag
   - transactions_lag_16 / 21 / 28
2. transactions_rolling
   - transactions_rolling_mean_16_7 / 16_14 / 16_28
3. oil_raw
   - dcoilwtico
4. oil_rolling_mean
   - oil_rolling_mean_7
5. oil_diff
   - oil_diff_7

각 실험은 동일한 Payday Baseline 29개에 독립적으로 추가합니다.
Baseline 모델은 다시 학습하지 않고 저장된 Fold 점수만 비교에 사용합니다.

5 experiments × 3 folds = 15 models.

Run:
    python src/modeling.py
"""

import gc
import json
from datetime import datetime
from pathlib import Path

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

    TRANSACTION_LAGS,
    TRANSACTION_LAG_FEATURES,
    TRANSACTION_ROLLING_SHIFT,
    TRANSACTION_ROLLING_WINDOWS,
    TRANSACTION_ROLLING_FEATURES,
    add_transaction_lag_features,
    add_transaction_rolling_features,

    OIL_RAW_FEATURES,
    OIL_ROLLING_WINDOW,
    OIL_ROLLING_FEATURES,
    OIL_DIFF_PERIOD,
    OIL_DIFF_FEATURES,
    add_oil_rolling_features,
    add_oil_diff_features,
)


# =========================================================
# 1. 기존 Payday Baseline: 총 29개
# =========================================================

BASE_FEATURES = [
    # Store / Product (6)
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",

    # Calendar (5)
    "year",
    "month",
    "day",
    "dayofweek",
    "is_weekend",

    # Promotion (2)
    "onpromotion",
    "is_promotion",

    # Holiday (10)
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


PAYDAY_FEATURES = [
    "is_15th",
    "is_month_end",
]


CATEGORICAL_FEATURES = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
]


BASELINE_FEATURES = (
    BASE_FEATURES
    + list(LAG_FEATURES)
    + ["dayofyear"]
    + PAYDAY_FEATURES
)


# =========================================================
# 2. 외부 데이터 5개 실험
# =========================================================

# 각각 Baseline 29개에 독립적으로 추가합니다.
# 서로의 외부 피처는 섞지 않습니다.
EXPERIMENTS = {
    "transactions_lag": list(
        TRANSACTION_LAG_FEATURES
    ),

    "transactions_rolling": list(
        TRANSACTION_ROLLING_FEATURES
    ),

    "oil_raw": list(
        OIL_RAW_FEATURES
    ),

    "oil_rolling_mean": list(
        OIL_ROLLING_FEATURES
    ),

    "oil_diff": list(
        OIL_DIFF_FEATURES
    ),
}


# =========================================================
# 3. 학습 / 검증 설정
# =========================================================

TARGET_COL = "sales_log"

FORECAST_HORIZON = 16

N_ESTIMATORS = 8000

EARLY_STOPPING_ROUNDS = 150


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


# Baseline은 다시 학습하지 않고
# 동일한 29개 Payday Baseline의 저장 결과를 사용.
REFERENCE_BASELINE = {
    "Fold 1": 0.382044,
    "Fold 2": 0.405838,
    "Fold 3": 0.394152,
}


REFERENCE_BASELINE_BEST_ITERATION = {
    "Fold 1": 6476,
    "Fold 2": 2697,
    "Fold 3": 6902,
}


OUTPUT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    / "results"
    / "external_features_5exp"
)


# =========================================================
# 4. 설정 검증
# =========================================================

def validate_configuration():

    if tuple(LAGS) != (
        16,
        21,
        28,
    ):
        raise ValueError(
            "Sales LAGS는 (16, 21, 28)이어야 합니다."
        )


    if list(LAG_FEATURES) != [
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
    ]:
        raise ValueError(
            "기존 Sales Lag 구성이 달라졌습니다."
        )


    if len(BASE_FEATURES) != 23:
        raise ValueError(
            f"BASE_FEATURES는 23개여야 합니다. "
            f"현재 {len(BASE_FEATURES)}개"
        )


    if (
        len(BASELINE_FEATURES) != 29
        or len(set(BASELINE_FEATURES)) != 29
    ):
        raise ValueError(
            "Baseline은 중복 없이 29개여야 합니다."
        )


    if tuple(TRANSACTION_LAGS) != (
        16,
        21,
        28,
    ):
        raise ValueError(
            "Transactions Lag는 (16, 21, 28)이어야 합니다."
        )


    if TRANSACTION_ROLLING_SHIFT != 16:
        raise ValueError(
            "Transactions Rolling shift는 16이어야 합니다."
        )


    if tuple(
        TRANSACTION_ROLLING_WINDOWS
    ) != (
        7,
        14,
        28,
    ):
        raise ValueError(
            "Transactions Rolling windows는 "
            "(7, 14, 28)이어야 합니다."
        )


    if OIL_ROLLING_WINDOW != 7:
        raise ValueError(
            "Oil Rolling window는 7이어야 합니다."
        )


    if OIL_DIFF_PERIOD != 7:
        raise ValueError(
            "Oil diff period는 7이어야 합니다."
        )


    if len(EXPERIMENTS) != 5:
        raise ValueError(
            "실험은 정확히 5개여야 합니다."
        )


    all_external = []

    for experiment, features in (
        EXPERIMENTS.items()
    ):

        if not features:
            raise ValueError(
                f"{experiment}: 추가 피처가 없습니다."
            )

        if len(features) != len(
            set(features)
        ):
            raise ValueError(
                f"{experiment}: 추가 피처가 중복됩니다."
            )

        overlap = (
            set(features)
            & set(BASELINE_FEATURES)
        )

        if overlap:
            raise ValueError(
                f"{experiment}: Baseline과 중복되는 피처 "
                f"{sorted(overlap)}"
            )

        all_external.extend(
            features
        )


    # 서로 다른 실험끼리도 같은 이름의 파생 피처를 공유하지 않음.
    if len(all_external) != len(
        set(all_external)
    ):
        raise ValueError(
            "외부 데이터 실험 간 피처 이름이 중복됩니다."
        )


    if len(FOLDS) * len(
        EXPERIMENTS
    ) != 15:
        raise ValueError(
            "5개 실험 × 3 Fold = 15 모델이어야 합니다."
        )


# =========================================================
# 5. 전체 Feature 준비
# =========================================================

def prepare_features(df):

    validate_configuration()

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )


    required = [
        "date",
        "sales",
        "transactions",
        "dcoilwtico",
        *BASE_FEATURES,
        "dayofyear",
        *PAYDAY_FEATURES,
    ]


    missing = sorted(
        set(required)
        - set(df.columns)
    )


    if missing:
        raise ValueError(
            "필수 컬럼 누락:\n"
            f"{missing}\n"
            "main() / preprocessing.py / "
            "feature_engineering.py를 확인하세요."
        )


    keys = [
        "store_nbr",
        "family",
    ]


    # -----------------------------------------------------
    # 기본 키 QC
    # -----------------------------------------------------

    if df[
        [
            "date",
            *keys,
        ]
    ].isna().any().any():

        raise ValueError(
            "date / store_nbr / family에 결측치가 있습니다."
        )


    if df.duplicated(
        [
            "date",
            *keys,
        ]
    ).any():

        raise ValueError(
            "date × store_nbr × family가 중복됩니다."
        )


    # -----------------------------------------------------
    # Target QC
    # -----------------------------------------------------

    sales = df[
        "sales"
    ].to_numpy(
        dtype=float
    )


    if (
        not np.isfinite(sales).all()
        or (sales < 0).any()
    ):
        raise ValueError(
            "sales에 결측/무한대/음수가 있습니다."
        )


    # -----------------------------------------------------
    # 외부 원본 QC
    # -----------------------------------------------------

    for column in [
        "transactions",
        "dcoilwtico",
    ]:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{column}에 결측치가 있습니다."
            )

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{column}에 무한대가 있습니다."
            )

        if (values < 0).any():
            raise ValueError(
                f"{column}에 음수가 있습니다."
            )

        df[column] = values


    # -----------------------------------------------------
    # Calendar QC
    # -----------------------------------------------------

    expected_calendar = {
        "dayofyear":
            df["date"].dt.dayofyear,

        "dayofweek":
            df["date"].dt.dayofweek,

        "is_weekend":
            (
                df["date"]
                .dt.dayofweek
                .ge(5)
                .astype(int)
            ),

        "is_15th":
            (
                df["date"]
                .dt.day
                .eq(15)
                .astype(int)
            ),

        "is_month_end":
            (
                df["date"]
                .dt.is_month_end
                .astype(int)
            ),
    }


    for feature, expected in (
        expected_calendar.items()
    ):

        if not np.array_equal(
            df[feature].to_numpy(),
            expected.to_numpy(),
        ):
            raise ValueError(
                f"{feature} 계산값이 date와 다릅니다."
            )


    # -----------------------------------------------------
    # Sales Lag 생성을 위한 일별 연속성
    # -----------------------------------------------------

    df = (
        df.sort_values(
            [
                *keys,
                "date",
            ]
        )
        .reset_index(drop=True)
    )


    date_diff = (
        df.groupby(
            keys,
            observed=True,
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
            "store × family별 날짜가 연속적이지 않습니다."
        )


    # -----------------------------------------------------
    # 기존 Sales Lag
    # -----------------------------------------------------

    df = add_lag_features(
        df,
        lags=LAGS,
    )


    # -----------------------------------------------------
    # Transactions Lag
    # -----------------------------------------------------

    df = add_transaction_lag_features(
        df,
        lags=TRANSACTION_LAGS,
    )


    # -----------------------------------------------------
    # Transactions Rolling
    # -----------------------------------------------------

    df = add_transaction_rolling_features(
        df,
        shift=TRANSACTION_ROLLING_SHIFT,
        windows=TRANSACTION_ROLLING_WINDOWS,
    )


    # -----------------------------------------------------
    # Oil Rolling Mean
    # -----------------------------------------------------

    df = add_oil_rolling_features(
        df,
        window=OIL_ROLLING_WINDOW,
    )


    # -----------------------------------------------------
    # Oil Diff
    # -----------------------------------------------------

    df = add_oil_diff_features(
        df,
        period=OIL_DIFF_PERIOD,
    )


    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    df[TARGET_COL] = np.log1p(
        df["sales"]
    )


    # -----------------------------------------------------
    # Sales Lag 초기 결측 위치 QC
    # -----------------------------------------------------

    group_position = (
        df.groupby(
            keys,
            observed=True,
        )
        .cumcount()
    )


    for lag in LAGS:

        feature = (
            f"sales_lag_{lag}"
        )

        actual = (
            df[feature]
            .isna()
        )

        expected = (
            group_position
            < lag
        )

        if not actual.equals(
            expected
        ):
            raise ValueError(
                f"{feature} 초기 결측 위치가 예상과 다릅니다."
            )


    # -----------------------------------------------------
    # 생성 피처 존재 여부
    # -----------------------------------------------------

    generated = (
        list(TRANSACTION_LAG_FEATURES)
        + list(TRANSACTION_ROLLING_FEATURES)
        + list(OIL_ROLLING_FEATURES)
        + list(OIL_DIFF_FEATURES)
    )


    missing_generated = [
        feature
        for feature in generated
        if feature not in df.columns
    ]


    if missing_generated:
        raise ValueError(
            "생성되지 않은 외부 피처:\n"
            f"{missing_generated}"
        )


    # -----------------------------------------------------
    # 출력
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EXTERNAL FEATURE EXPERIMENTS"
    )

    print(
        "=" * 90
    )

    print(
        f"Baseline: {len(BASELINE_FEATURES)} features "
        "(NOT RETRAINED)"
    )

    print(
        "Stored Baseline:"
    )

    for fold, score in (
        REFERENCE_BASELINE.items()
    ):
        print(
            f"  {fold}: {score:.6f}"
        )


    print(
        "\nTransactions Lag:",
        list(
            TRANSACTION_LAG_FEATURES
        ),
    )

    print(
        "Transactions Rolling:",
        list(
            TRANSACTION_ROLLING_FEATURES
        ),
    )

    print(
        "Oil Raw:",
        list(
            OIL_RAW_FEATURES
        ),
    )

    print(
        "Oil Rolling:",
        list(
            OIL_ROLLING_FEATURES
        ),
    )

    print(
        "Oil Diff:",
        list(
            OIL_DIFF_FEATURES
        ),
    )

    print(
        f"\nn_estimators={N_ESTIMATORS}, "
        f"early_stopping={EARLY_STOPPING_ROUNDS}"
    )

    print(
        "Total experiments:",
        len(EXPERIMENTS),
    )

    print(
        "Total models:",
        len(EXPERIMENTS)
        * len(FOLDS),
    )

    print(
        "\nGenerated feature missing counts:"
    )

    print(
        df[
            [
                *LAG_FEATURES,
                *TRANSACTION_LAG_FEATURES,
                *TRANSACTION_ROLLING_FEATURES,
                *OIL_ROLLING_FEATURES,
                *OIL_DIFF_FEATURES,
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nCommon QC: OK"
    )


    return df


# =========================================================
# 6. 16일 3-Fold
# =========================================================

def make_fold(
    df,
    config,
):

    train_end = pd.Timestamp(
        config["train_end"]
    )

    valid_start = pd.Timestamp(
        config["valid_start"]
    )

    valid_end = pd.Timestamp(
        config["valid_end"]
    )


    horizon = (
        valid_end
        - valid_start
    ).days + 1


    if (
        horizon
        != FORECAST_HORIZON
    ):
        raise ValueError(
            "Validation은 16일이어야 합니다."
        )


    if (
        valid_start
        != train_end
        + pd.Timedelta(days=1)
    ):
        raise ValueError(
            "Train 종료 다음 날부터 Validation이 시작해야 합니다."
        )


    if min(
        TRANSACTION_LAGS
    ) < horizon:

        raise ValueError(
            "Transactions Lag가 16일 horizon보다 짧습니다."
        )


    if (
        TRANSACTION_ROLLING_SHIFT
        < horizon
    ):
        raise ValueError(
            "Transactions Rolling shift가 "
            "16일 horizon보다 짧습니다."
        )


    train_df = df.loc[
        df["date"] <= train_end
    ].copy()


    valid_df = df.loc[
        df["date"].between(
            valid_start,
            valid_end,
        )
    ].copy()


    if (
        train_df.empty
        or valid_df.empty
    ):
        raise ValueError(
            "Train 또는 Validation이 비었습니다."
        )


    if (
        train_df["date"].max()
        != train_end
    ):
        raise ValueError(
            "Fold Train 종료일이 설정과 다릅니다."
        )


    valid_counts = (
        valid_df
        .groupby(
            [
                "store_nbr",
                "family",
            ],
            observed=True,
        )
        .size()
    )


    if not valid_counts.eq(
        horizon
    ).all():

        raise ValueError(
            "일부 Validation 시계열이 16일이 아닙니다."
        )


    return (
        train_df,
        valid_df,
    )


# =========================================================
# 7. 모델 입력
# =========================================================

def prepare_model_data(
    train_df,
    valid_df,
    features,
):

    X_train = train_df[
        features
    ].copy()

    X_valid = valid_df[
        features
    ].copy()


    active_categorical = [
        feature
        for feature
        in CATEGORICAL_FEATURES
        if feature in features
    ]


    for feature in (
        active_categorical
    ):

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


    # 초기 과거 길이 부족으로 Train 초반에 NaN이 허용되는 피처.
    allowed_train_na = set(
        [
            *LAG_FEATURES,
            *TRANSACTION_LAG_FEATURES,
            *TRANSACTION_ROLLING_FEATURES,
            *OIL_ROLLING_FEATURES,
            *OIL_DIFF_FEATURES,
        ]
    )


    nonnullable = [
        feature
        for feature in features
        if feature
        not in allowed_train_na
    ]


    if X_train[
        nonnullable
    ].isna().any().any():

        raise ValueError(
            "Train의 일반 피처에 결측치가 있습니다."
        )


    if X_valid[
        features
    ].isna().any().any():

        missing_cols = (
            X_valid[
                features
            ]
            .isna()
            .sum()
        )

        missing_cols = (
            missing_cols[
                missing_cols > 0
            ]
        )

        raise ValueError(
            "Validation 피처에 결측치가 있습니다:\n"
            f"{missing_cols.to_string()}"
        )


    if list(
        X_train.columns
    ) != features:

        raise ValueError(
            "Train 피처 순서가 설정과 다릅니다."
        )


    if list(
        X_valid.columns
    ) != features:

        raise ValueError(
            "Validation 피처 순서가 설정과 다릅니다."
        )


    return (
        X_train,
        X_valid,
        train_df[TARGET_COL].copy(),
        valid_df[TARGET_COL].copy(),
        valid_df["sales"].copy(),
        active_categorical,
    )


# =========================================================
# 8. LightGBM
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


# =========================================================
# 9. RMSLE
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
            "예측값에 NaN 또는 무한대가 있습니다."
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
# 10. Feature Importance
# =========================================================

def get_feature_importance(
    model,
    experiment,
    fold_name,
    features,
):

    booster = model.booster_

    iteration = int(
        model.best_iteration_
    )


    names = booster.feature_name()


    if names != features:

        raise ValueError(
            "학습 모델의 피처 구성과 설정이 다릅니다."
        )


    gain = booster.feature_importance(
        importance_type="gain",
        iteration=iteration,
    )


    split = booster.feature_importance(
        importance_type="split",
        iteration=iteration,
    )


    result = pd.DataFrame(
        {
            "experiment":
                experiment,

            "fold":
                fold_name,

            "best_iteration":
                iteration,

            "feature":
                names,

            "gain":
                gain,

            "split":
                split,
        }
    )


    total_gain = (
        result[
            "gain"
        ].sum()
    )


    total_split = (
        result[
            "split"
        ].sum()
    )


    result[
        "gain_pct"
    ] = np.where(
        total_gain > 0,
        result[
            "gain"
        ] / total_gain * 100,
        0.0,
    )


    result[
        "split_pct"
    ] = np.where(
        total_split > 0,
        result[
            "split"
        ] / total_split * 100,
        0.0,
    )


    return result


# =========================================================
# 11. 한 모델 학습
# =========================================================

def train_one(
    train_df,
    valid_df,
    experiment,
    fold_name,
    features,
):

    (
        X_train,
        X_valid,
        y_train,
        y_valid_log,
        y_valid,
        active_categorical,
    ) = prepare_model_data(
        train_df,
        valid_df,
        features,
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
            active_categorical
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


    score = calculate_rmsle(
        y_true=y_valid,
        pred_log=pred_log,
    )


    importance = (
        get_feature_importance(
            model=model,
            experiment=experiment,
            fold_name=fold_name,
            features=features,
        )
    )


    print(
        f"\nBest iteration: "
        f"{best_iteration}"
    )

    print(
        f"RMSLE: {score:.6f}"
    )


    if (
        best_iteration
        >= N_ESTIMATORS
        - EARLY_STOPPING_ROUNDS
    ):

        print(
            "CHECK: Best iteration이 "
            "학습 상한에 가깝습니다."
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


    return (
        best_iteration,
        score,
        importance,
    )


# =========================================================
# 12. Importance 3-Fold 평균
# =========================================================

def summarize_importance(
    per_fold,
):

    result = (
        per_fold
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

            mean_gain=(
                "gain",
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
        .reset_index(
            drop=True
        )
    )


    result.insert(
        0,
        "gain_rank",
        result[
            "mean_gain_pct"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int),
    )


    result[
        "split_rank"
    ] = (
        result[
            "mean_split_pct"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


    return result


# =========================================================
# 13. 5개 실험 × 3 Fold 실행
# =========================================================

def run_backtesting(
    save_csv=True,
):

    validate_configuration()


    run_dir = (
        OUTPUT_DIR
        / datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )


    if save_csv:

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )


        metadata = {
            "baseline_retrained":
                False,

            "baseline_features":
                BASELINE_FEATURES,

            "reference_baseline":
                REFERENCE_BASELINE,

            "experiments":
                EXPERIMENTS,

            "folds":
                FOLDS,

            "model_parameters":
                build_model()
                .get_params(),

            "early_stopping_rounds":
                EARLY_STOPPING_ROUNDS,

            "target":
                "log1p(sales)",
        }


        (
            run_dir
            / "run_config.json"
        ).write_text(

            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),

            encoding="utf-8",
        )


    # -----------------------------------------------------
    # Pipeline
    # -----------------------------------------------------

    train, test, holidays = (
        main()
    )


    del test
    del holidays

    gc.collect()


    train = prepare_features(
        train
    )


    fold_rows = []

    importance_parts = []


    # -----------------------------------------------------
    # 실험 순서:
    # Transactions 2개 먼저 → Oil 3개
    # -----------------------------------------------------

    for (
        experiment,
        added_features,
    ) in EXPERIMENTS.items():


        features = (
            BASELINE_FEATURES
            + added_features
        )


        print(
            "\n\n"
            + "#" * 100
        )

        print(
            f"EXPERIMENT: {experiment}"
        )

        print(
            "Added:",
            added_features,
        )

        print(
            f"Features: "
            f"{len(BASELINE_FEATURES)} "
            f"-> {len(features)}"
        )

        print(
            "#" * 100
        )


        for config in FOLDS:

            fold_name = (
                config[
                    "fold"
                ]
            )


            train_df, valid_df = (
                make_fold(
                    train,
                    config,
                )
            )


            baseline_score = (
                REFERENCE_BASELINE[
                    fold_name
                ]
            )


            baseline_iteration = (
                REFERENCE_BASELINE_BEST_ITERATION[
                    fold_name
                ]
            )


            print(
                "\n"
                + "=" * 90
            )

            print(
                f"{experiment} | "
                f"{fold_name}"
            )

            print(
                "=" * 90
            )

            print(
                "Train:",
                train_df["date"].min().date(),
                "~",
                train_df["date"].max().date(),
            )

            print(
                "Valid:",
                valid_df["date"].min().date(),
                "~",
                valid_df["date"].max().date(),
            )

            print(
                "Train shape:",
                train_df.shape,
                "| Valid shape:",
                valid_df.shape,
            )

            print(
                f"Stored Baseline: "
                f"{baseline_score:.6f}"
            )

            print(
                "Baseline training: SKIPPED"
            )


            (
                best_iteration,
                score,
                importance,
            ) = train_one(

                train_df=train_df,

                valid_df=valid_df,

                experiment=experiment,

                fold_name=fold_name,

                features=features,
            )


            delta = (
                score
                - baseline_score
            )


            print(
                f"Delta vs Baseline: "
                f"{delta:+.6f}"
            )


            fold_rows.append(
                {
                    "experiment":
                        experiment,

                    "fold":
                        fold_name,

                    "added_features":
                        "|".join(
                            added_features
                        ),

                    "feature_count":
                        len(features),

                    "baseline_rmsle":
                        baseline_score,

                    "baseline_best_iteration":
                        baseline_iteration,

                    "best_iteration":
                        best_iteration,

                    "rmsle":
                        score,

                    "delta_rmsle":
                        delta,
                }
            )


            importance_parts.append(
                importance
            )


            # 장시간 실행 중간 결과 보존
            if save_csv:

                pd.DataFrame(
                    fold_rows
                ).to_csv(

                    run_dir
                    / "fold_results_partial.csv",

                    index=False,

                    encoding="utf-8-sig",
                )


                pd.concat(
                    importance_parts,
                    ignore_index=True,
                ).to_csv(

                    run_dir
                    / "per_fold_importance_partial.csv",

                    index=False,

                    encoding="utf-8-sig",
                )


            del train_df
            del valid_df
            del importance

            gc.collect()


    # =====================================================
    # 결과 통합
    # =====================================================

    fold_results = pd.DataFrame(
        fold_rows
    )


    all_importance = pd.concat(
        importance_parts,
        ignore_index=True,
    )


    summary = (
        fold_results
        .groupby(
            "experiment",
            sort=False,
            as_index=False,
        )
        .agg(
            added_features=(
                "added_features",
                "first",
            ),

            feature_count=(
                "feature_count",
                "first",
            ),

            mean_rmsle=(
                "rmsle",
                "mean",
            ),

            std_rmsle=(
                "rmsle",
                "std",
            ),

            mean_delta_rmsle=(
                "delta_rmsle",
                "mean",
            ),

            improved_folds=(
                "delta_rmsle",
                lambda x:
                    int(
                        (
                            x < 0
                        ).sum()
                    ),
            ),

            mean_best_iteration=(
                "best_iteration",
                "mean",
            ),
        )
    )


    baseline_mean = np.mean(
        list(
            REFERENCE_BASELINE.values()
        )
    )


    summary.insert(
        2,
        "baseline_mean_rmsle",
        baseline_mean,
    )


    summary[
        "decision"
    ] = np.select(

        [
            summary[
                "mean_delta_rmsle"
            ] < 0,

            summary[
                "mean_delta_rmsle"
            ] > 0,
        ],

        [
            "IMPROVED",
            "WORSE",
        ],

        default="SAME",
    )


    summary = (
        summary
        .sort_values(
            "mean_rmsle"
        )
        .reset_index(
            drop=True
        )
    )


    # =====================================================
    # Importance 평균
    # =====================================================

    mean_importance_parts = []


    for experiment in (
        EXPERIMENTS
    ):

        per_fold = (
            all_importance.loc[
                all_importance[
                    "experiment"
                ].eq(
                    experiment
                )
            ]
        )


        mean_imp = (
            summarize_importance(
                per_fold
            )
        )


        mean_imp.insert(
            0,
            "experiment",
            experiment,
        )


        mean_importance_parts.append(
            mean_imp
        )


        print(
            "\n"
            + "=" * 90
        )

        print(
            f"FEATURE IMPORTANCE — "
            f"{experiment} — 3-FOLD MEAN"
        )

        print(
            "=" * 90
        )


        print(
            mean_imp
            .round(4)
            .to_string(
                index=False
            )
        )


        added = (
            EXPERIMENTS[
                experiment
            ]
        )


        focus = mean_imp.loc[
            mean_imp[
                "feature"
            ].isin(
                added
            )
        ]


        print(
            "\nAdded feature rows:"
        )

        print(
            focus
            .round(4)
            .to_string(
                index=False
            )
        )


    mean_importance = (
        pd.concat(
            mean_importance_parts,
            ignore_index=True,
        )
    )


    # =====================================================
    # 최종 표
    # =====================================================

    print(
        "\n\n"
        + "=" * 110
    )

    print(
        "FINAL EXTERNAL FEATURE COMPARISON"
    )

    print(
        "=" * 110
    )


    print(
        summary.to_string(

            index=False,

            formatters={
                "baseline_mean_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "mean_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "std_rmsle":
                    lambda x:
                    f"{x:.6f}",

                "mean_delta_rmsle":
                    lambda x:
                    f"{x:+.6f}",

                "mean_best_iteration":
                    lambda x:
                    f"{x:.0f}",
            },
        )
    )


    print(
        "\nStored Baseline mean:",
        f"{baseline_mean:.6f}",
    )


    print(
        "\nInterpretation:"
    )

    print(
        "delta < 0 : Baseline보다 개선"
    )

    print(
        "delta > 0 : Baseline보다 악화"
    )

    print(
        "Baseline 모델은 이번 실행에서 "
        "학습하지 않았습니다."
    )

    print(
        "총 학습 모델 수:",
        len(fold_results),
    )


    # =====================================================
    # 저장
    # =====================================================

    if save_csv:

        for filename, frame in [

            (
                "fold_results.csv",
                fold_results,
            ),

            (
                "score_summary.csv",
                summary,
            ),

            (
                "per_fold_importance.csv",
                all_importance,
            ),

            (
                "mean_importance.csv",
                mean_importance,
            ),
        ]:

            frame.to_csv(

                run_dir
                / filename,

                index=False,

                encoding="utf-8-sig",
            )


        print(
            "\nCSV 저장 위치:",
            run_dir
        )


    del train

    gc.collect()


    return (
        fold_results,
        summary,
        mean_importance,
        all_importance,
    )


# =========================================================
# 14. 실행
# =========================================================

if __name__ == "__main__":

    (
        fold_results,
        score_summary,
        mean_importance,
        per_fold_importance,
    ) = run_backtesting()
