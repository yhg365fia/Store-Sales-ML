"""
Selective Feature Ablation

Baseline:
Base23 + Lag16/21/28 + dayofyear + Payday2 = 29 features

Experiments:
1. remove is_promotion
2. remove is_month_end
3. remove is_15th
4. remove state

Baseline is NOT retrained.

Run:
python src/experiment_ablation.py
"""

import gc
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
)


# =========================================================
# 1. Base Features
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


# =========================================================
# 2. Current Baseline
# =========================================================

BASELINE_FEATURES = (
    BASE_FEATURES
    + list(LAG_FEATURES)
    + ["dayofyear"]
    + PAYDAY_FEATURES
)


# =========================================================
# 3. Ablation Experiments
# =========================================================

ABLATION_EXPERIMENTS = {
    "remove_is_promotion": "is_promotion",
    "remove_is_month_end": "is_month_end",
    "remove_is_15th": "is_15th",
    "remove_state": "state",
}


# 각 실험은 Baseline 29개에서
# 정확히 1개만 제거
EXPERIMENT_FEATURES = {
    experiment_name: [
        feature
        for feature in BASELINE_FEATURES
        if feature != removed_feature
    ]
    for experiment_name, removed_feature
    in ABLATION_EXPERIMENTS.items()
}


# =========================================================
# 4. Model Settings
# =========================================================

N_ESTIMATORS = 8000

EARLY_STOPPING_ROUNDS = 150

FORECAST_HORIZON = 16


# Baseline은 재학습하지 않음
RETRAIN_BASELINE = False


OUTPUT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    / "results"
    / "ablation_4"
)


# =========================================================
# 5. Folds
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
# 6. Stored Baseline
# =========================================================

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


# =========================================================
# 7. Configuration QC
# =========================================================

def validate_configuration():

    if tuple(LAGS) != (
        16,
        21,
        28,
    ):
        raise ValueError(
            "LAGS는 (16, 21, 28)이어야 합니다."
        )


    if len(BASE_FEATURES) != 23:
        raise ValueError(
            f"BASE_FEATURES는 23개여야 합니다. "
            f"현재: {len(BASE_FEATURES)}"
        )


    if len(BASELINE_FEATURES) != 29:
        raise ValueError(
            f"Baseline은 29개여야 합니다. "
            f"현재: {len(BASELINE_FEATURES)}"
        )


    if len(
        set(BASELINE_FEATURES)
    ) != len(
        BASELINE_FEATURES
    ):
        raise ValueError(
            "Baseline Feature에 중복이 있습니다."
        )


    for (
        experiment_name,
        removed_feature,
    ) in ABLATION_EXPERIMENTS.items():

        features = (
            EXPERIMENT_FEATURES[
                experiment_name
            ]
        )

        if len(features) != 28:
            raise ValueError(
                f"{experiment_name}: "
                "28개 피처여야 합니다."
            )


        if removed_feature in features:
            raise ValueError(
                f"{experiment_name}: "
                f"{removed_feature}가 "
                "실제로 제거되지 않았습니다."
            )


        difference = (
            set(BASELINE_FEATURES)
            - set(features)
        )

        if difference != {
            removed_feature
        }:
            raise ValueError(
                f"{experiment_name}: "
                "한 개 피처만 제거되어야 합니다.\n"
                f"차이: {difference}"
            )


# =========================================================
# 8. Feature Preparation
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
        *PAYDAY_FEATURES,
    ]


    missing_columns = sorted(
        set(required_columns)
        - set(df.columns)
    )


    if missing_columns:
        raise ValueError(
            "필수 컬럼 누락:\n"
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
        [
            "date",
            *keys,
        ]
    ].isna().any().any():

        raise ValueError(
            "날짜 / 매장 / 상품군에 "
            "결측치가 있습니다."
        )


    if df.duplicated(
        [
            "date",
            *keys,
        ]
    ).any():

        raise ValueError(
            "date × store_nbr × family "
            "중복이 있습니다."
        )


    # -----------------------------------------------------
    # Sales QC
    # -----------------------------------------------------

    sales_array = df[
        "sales"
    ].to_numpy(
        dtype=float
    )


    if not np.isfinite(
        sales_array
    ).all():

        raise ValueError(
            "sales에 결측치 또는 "
            "무한대가 있습니다."
        )


    if (
        sales_array < 0
    ).any():

        raise ValueError(
            "sales에 음수가 있습니다."
        )


    # -----------------------------------------------------
    # Calendar QC
    # -----------------------------------------------------

    if not df[
        "dayofweek"
    ].eq(
        df["date"].dt.dayofweek
    ).all():

        raise ValueError(
            "dayofweek가 실제 날짜와 다릅니다."
        )


    if not df[
        "dayofyear"
    ].eq(
        df["date"].dt.dayofyear
    ).all():

        raise ValueError(
            "dayofyear가 실제 날짜와 다릅니다."
        )


    expected_weekend = (
        df["date"]
        .dt.dayofweek
        .ge(5)
        .astype(int)
    )


    if not df[
        "is_weekend"
    ].eq(
        expected_weekend
    ).all():

        raise ValueError(
            "is_weekend 계산이 잘못됐습니다."
        )


    # -----------------------------------------------------
    # Promotion QC
    # -----------------------------------------------------

    expected_promotion = (
        df["onpromotion"] > 0
    ).astype(int)


    if not df[
        "is_promotion"
    ].eq(
        expected_promotion
    ).all():

        raise ValueError(
            "is_promotion 계산이 "
            "onpromotion과 일치하지 않습니다."
        )


    # -----------------------------------------------------
    # Payday QC
    # -----------------------------------------------------

    expected_15th = (
        df["date"].dt.day
        == 15
    ).astype(int)


    if not df[
        "is_15th"
    ].eq(
        expected_15th
    ).all():

        raise ValueError(
            "is_15th 계산이 잘못됐습니다."
        )


    expected_month_end = (
        df["date"]
        .dt.is_month_end
        .astype(int)
    )


    if not df[
        "is_month_end"
    ].eq(
        expected_month_end
    ).all():

        raise ValueError(
            "is_month_end 계산이 "
            "잘못됐습니다."
        )


    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            [
                *keys,
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # Daily continuity QC
    # -----------------------------------------------------

    date_diff = (
        df
        .groupby(
            keys,
            observed=True,
        )["date"]
        .diff()
    )


    if not (
        date_diff
        .dropna()
        .eq(
            pd.Timedelta(
                days=1
            )
        )
        .all()
    ):

        raise ValueError(
            "매장 × 상품군별 날짜가 "
            "연속되지 않습니다."
        )


    # -----------------------------------------------------
    # Lag
    # -----------------------------------------------------

    df = add_lag_features(
        df,
        lags=LAGS,
    )


    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    df["sales_log"] = np.log1p(
        df["sales"]
    )


    # -----------------------------------------------------
    # Lag missing QC
    # -----------------------------------------------------

    positions = (
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
            positions < lag
        )

        actual_missing = (
            df[
                feature
            ].isna()
        )


        if not actual_missing.eq(
            expected_missing
        ).all():

            raise ValueError(
                f"{feature} 초기 결측 위치가 "
                "예상과 다릅니다."
            )


    return df


# =========================================================
# 9. Fold Split
# =========================================================

def make_fold(
    df,
    spec,
):

    train_end = pd.Timestamp(
        spec["train_end"]
    )

    valid_start = pd.Timestamp(
        spec["valid_start"]
    )

    valid_end = pd.Timestamp(
        spec["valid_end"]
    )


    if (
        valid_start
        != train_end
        + pd.Timedelta(
            days=1
        )
    ):

        raise ValueError(
            "Fold 경계가 연속되지 않습니다."
        )


    horizon = (
        valid_end
        - valid_start
    ).days + 1


    if horizon != FORECAST_HORIZON:

        raise ValueError(
            "Validation은 16일이어야 합니다."
        )


    if min(
        LAGS
    ) < FORECAST_HORIZON:

        raise ValueError(
            "16일 예측에서 "
            "Lag가 16 미만입니다."
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
            "Train 종료일이 Fold 설정과 다릅니다."
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
        FORECAST_HORIZON
    ).all():

        raise ValueError(
            "Validation의 각 시계열에 "
            "16일 데이터가 있어야 합니다."
        )


    return (
        train_df,
        valid_df,
    )


# =========================================================
# 10. Model Input
# =========================================================

def model_inputs(
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


    # state 제거 실험도 있으므로
    # 실제 사용 Feature에 존재하는
    # categorical만 처리
    active_categorical_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in features
    ]


    for column in (
        active_categorical_features
    ):

        categories = pd.Index(
            train_df[
                column
            ]
            .dropna()
            .unique()
        )


        X_train[
            column
        ] = pd.Categorical(
            X_train[
                column
            ],
            categories=categories,
        )


        X_valid[
            column
        ] = pd.Categorical(
            X_valid[
                column
            ],
            categories=categories,
        )


    # Train에서는 시계열 시작 부분의
    # Lag 결측만 허용
    allowed_na = set(
        LAG_FEATURES
    )


    nonnullable = [
        feature
        for feature in features
        if feature not in allowed_na
    ]


    if (
        X_train[
            nonnullable
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
            nonnullable
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Validation의 Lag 외 피처에 "
            "결측치가 있습니다."
        )


    active_lag_features = [
        feature
        for feature in LAG_FEATURES
        if feature in features
    ]


    if (
        X_valid[
            active_lag_features
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Validation Lag에 "
            "결측치가 있습니다."
        )


    return (
        X_train,
        X_valid,
        active_categorical_features,
    )


# =========================================================
# 11. LightGBM
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
# 12. RMSLE
# =========================================================

def calculate_rmsle(
    y_true,
    prediction_log,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )


    prediction_log = np.asarray(
        prediction_log,
        dtype=float,
    )


    if not np.isfinite(
        prediction_log
    ).all():

        raise ValueError(
            "예측 결과에 결측치 또는 "
            "무한대가 있습니다."
        )


    prediction_log = np.maximum(
        prediction_log,
        0.0,
    )


    return float(

        np.sqrt(

            np.mean(

                (
                    np.log1p(
                        y_true
                    )
                    - prediction_log
                )
                ** 2
            )
        )
    )


# =========================================================
# 13. Train One Model
# =========================================================

def train_one(
    train_df,
    valid_df,
    features,
):

    (
        X_train,
        X_valid,
        active_categorical_features,
    ) = model_inputs(
        train_df,
        valid_df,
        features,
    )


    model = build_model()


    model.fit(

        X_train,

        train_df[
            "sales_log"
        ],

        eval_set=[
            (
                X_valid,

                valid_df[
                    "sales_log"
                ],
            )
        ],

        eval_metric="rmse",

        categorical_feature=(
            active_categorical_features
        ),

        callbacks=[

            early_stopping(
                EARLY_STOPPING_ROUNDS,
                verbose=True,
            ),

            log_evaluation(
                200
            ),
        ],
    )


    best_iteration = int(
        model.best_iteration_
    )


    prediction_log = model.predict(
        X_valid,
        num_iteration=best_iteration,
    )


    score = calculate_rmsle(

        y_true=valid_df[
            "sales"
        ].to_numpy(),

        prediction_log=(
            prediction_log
        ),
    )


    # -----------------------------------------------------
    # Importance
    # -----------------------------------------------------

    booster = model.booster_


    gain = booster.feature_importance(
        importance_type="gain",
        iteration=best_iteration,
    )


    split = booster.feature_importance(
        importance_type="split",
        iteration=best_iteration,
    )


    importance_df = pd.DataFrame(
        {
            "feature":
                booster.feature_name(),

            "gain":
                gain,

            "split":
                split,
        }
    )


    gain_total = max(
        importance_df[
            "gain"
        ].sum(),
        1e-12,
    )


    split_total = max(
        importance_df[
            "split"
        ].sum(),
        1e-12,
    )


    importance_df[
        "gain_pct"
    ] = (
        importance_df[
            "gain"
        ]
        / gain_total
        * 100
    )


    importance_df[
        "split_pct"
    ] = (
        importance_df[
            "split"
        ]
        / split_total
        * 100
    )


    del model
    del X_train
    del X_valid
    del prediction_log

    gc.collect()


    return (
        best_iteration,
        score,
        importance_df,
    )


# =========================================================
# 14. One Ablation Experiment
# =========================================================

def run_one_ablation(
    df,
    experiment_name,
    removed_feature,
):

    features = (
        EXPERIMENT_FEATURES[
            experiment_name
        ]
    )


    print(
        "\n\n"
        + "=" * 90
    )

    print(
        experiment_name.upper()
    )

    print(
        "=" * 90
    )

    print(
        "Removed feature:",
        removed_feature,
    )

    print(
        "Baseline features:",
        len(
            BASELINE_FEATURES
        )
    )

    print(
        "Experiment features:",
        len(
            features
        )
    )

    print(
        "Baseline retraining:",
        RETRAIN_BASELINE,
    )


    results = []

    importances = []


    for spec in FOLDS:

        fold_name = (
            spec[
                "fold"
            ]
        )


        print(
            "\n"
            + "-" * 80
        )

        print(
            f"{experiment_name} | "
            f"{fold_name}"
        )

        print(
            "-" * 80
        )


        train_df, valid_df = (
            make_fold(
                df,
                spec,
            )
        )


        # -----------------------------------------------
        # Stored Baseline
        # -----------------------------------------------

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
            f"Baseline RMSLE: "
            f"{baseline_score:.6f}"
        )

        print(
            f"Baseline Best Iteration: "
            f"{baseline_iteration}"
        )

        print(
            "Baseline training: SKIPPED"
        )


        # -----------------------------------------------
        # Ablation Model
        # -----------------------------------------------

        (
            best_iteration,
            score,
            importance_df,
        ) = train_one(

            train_df=train_df,

            valid_df=valid_df,

            features=features,
        )


        delta = (
            score
            - baseline_score
        )


        print(
            f"\nAblation RMSLE: "
            f"{score:.6f}"
        )

        print(
            f"Ablation Best Iteration: "
            f"{best_iteration}"
        )

        print(
            f"Delta: "
            f"{delta:+.6f}"
        )


        importance_df.insert(
            0,
            "fold",
            fold_name,
        )


        importance_df.insert(
            0,
            "experiment",
            experiment_name,
        )


        importances.append(
            importance_df
        )


        results.append(
            {
                "experiment":
                    experiment_name,

                "removed_feature":
                    removed_feature,

                "fold":
                    fold_name,

                "baseline_rmsle":
                    baseline_score,

                "baseline_best_iteration":
                    baseline_iteration,

                "ablation_rmsle":
                    score,

                "ablation_best_iteration":
                    best_iteration,

                "delta":
                    delta,
            }
        )


        del train_df
        del valid_df
        del importance_df

        gc.collect()


    fold_results = pd.DataFrame(
        results
    )


    all_importance = pd.concat(
        importances,
        ignore_index=True,
    )


    mean_importance = (

        all_importance

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


    mean_importance.insert(
        0,
        "gain_rank",
        np.arange(
            1,
            len(
                mean_importance
            ) + 1,
        ),
    )


    mean_importance[
        "split_rank"
    ] = (
        mean_importance[
            "mean_split_pct"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    baseline_mean = (
        fold_results[
            "baseline_rmsle"
        ].mean()
    )


    experiment_mean = (
        fold_results[
            "ablation_rmsle"
        ].mean()
    )


    experiment_std = (
        fold_results[
            "ablation_rmsle"
        ].std()
    )


    mean_delta = (
        experiment_mean
        - baseline_mean
    )


    improved_folds = int(
        (
            fold_results[
                "delta"
            ] < 0
        ).sum()
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{experiment_name} RESULT"
    )

    print(
        "=" * 80
    )


    print(

        fold_results[
            [
                "fold",
                "baseline_rmsle",
                "ablation_rmsle",
                "delta",
                "ablation_best_iteration",
            ]
        ]

        .to_string(

            index=False,

            float_format=(
                lambda x:
                f"{x:.6f}"
            ),
        )
    )


    print(
        f"\nBaseline mean : "
        f"{baseline_mean:.6f}"
    )

    print(
        f"Ablation mean : "
        f"{experiment_mean:.6f}"
    )

    print(
        f"Std           : "
        f"{experiment_std:.6f}"
    )

    print(
        f"Mean delta    : "
        f"{mean_delta:+.6f}"
    )

    print(
        f"Improved folds: "
        f"{improved_folds}/3"
    )


    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    if mean_delta < 0:

        decision = (
            "IMPROVED"
        )

    elif mean_delta > 0:

        decision = (
            "WORSE"
        )

    else:

        decision = (
            "SAME"
        )


    print(
        "Decision:",
        decision,
    )


    return (
        fold_results,
        mean_importance,
        all_importance,
    )


# =========================================================
# 15. Run All Four Ablations
# =========================================================

def run_experiments():

    validate_configuration()


    # -----------------------------------------------------
    # Existing pipeline
    # -----------------------------------------------------

    raw_train, raw_test, holidays = (
        main()
    )


    del raw_test
    del holidays

    gc.collect()


    # -----------------------------------------------------
    # Lag
    # -----------------------------------------------------

    df = prepare_features(
        raw_train
    )


    del raw_train

    gc.collect()


    print(
        "\n"
        + "=" * 90
    )

    print(
        "SELECTIVE ABLATION EXPERIMENT"
    )

    print(
        "=" * 90
    )

    print(
        "Baseline:",
        "29 features"
    )

    print(
        "Baseline retraining:",
        RETRAIN_BASELINE
    )

    print(
        "n_estimators:",
        N_ESTIMATORS
    )

    print(
        "Early stopping:",
        EARLY_STOPPING_ROUNDS
    )

    print(
        "\nStored baseline:"
    )

    for (
        fold_name,
        score,
    ) in REFERENCE_BASELINE.items():

        print(
            fold_name,
            "=",
            f"{score:.6f}",
        )


    all_fold_results = []

    all_mean_importances = []

    all_per_fold_importances = []


    # =====================================================
    # 4 Experiments
    # =====================================================

    for (
        experiment_name,
        removed_feature,
    ) in ABLATION_EXPERIMENTS.items():

        (
            fold_results,
            mean_importance,
            per_fold_importance,
        ) = run_one_ablation(

            df=df,

            experiment_name=(
                experiment_name
            ),

            removed_feature=(
                removed_feature
            ),
        )


        mean_importance.insert(
            0,
            "removed_feature",
            removed_feature,
        )


        mean_importance.insert(
            0,
            "experiment",
            experiment_name,
        )


        all_fold_results.append(
            fold_results
        )

        all_mean_importances.append(
            mean_importance
        )

        all_per_fold_importances.append(
            per_fold_importance
        )


    # =====================================================
    # Combine
    # =====================================================

    all_fold_results = pd.concat(
        all_fold_results,
        ignore_index=True,
    )


    all_mean_importances = pd.concat(
        all_mean_importances,
        ignore_index=True,
    )


    all_per_fold_importances = pd.concat(
        all_per_fold_importances,
        ignore_index=True,
    )


    # =====================================================
    # Final Summary
    # =====================================================

    experiment_summary = (

        all_fold_results

        .groupby(
            [
                "experiment",
                "removed_feature",
            ],
            as_index=False,
        )

        .agg(

            mean_rmsle=(
                "ablation_rmsle",
                "mean",
            ),

            std_rmsle=(
                "ablation_rmsle",
                "std",
            ),

            mean_delta=(
                "delta",
                "mean",
            ),

            improved_folds=(
                "delta",
                lambda x:
                int(
                    (
                        x < 0
                    ).sum()
                ),
            ),

            mean_best_iteration=(
                "ablation_best_iteration",
                "mean",
            ),
        )
    )


    baseline_mean = np.mean(
        list(
            REFERENCE_BASELINE.values()
        )
    )


    experiment_summary.insert(
        2,
        "baseline_mean_rmsle",
        baseline_mean,
    )


    experiment_summary[
        "decision"
    ] = np.select(

        [
            experiment_summary[
                "mean_delta"
            ] < 0,

            experiment_summary[
                "mean_delta"
            ] > 0,
        ],

        [
            "IMPROVED",
            "WORSE",
        ],

        default="SAME",
    )


    experiment_summary = (
        experiment_summary
        .sort_values(
            "mean_rmsle"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\n\n"
        + "=" * 100
    )

    print(
        "FINAL ABLATION COMPARISON"
    )

    print(
        "=" * 100
    )


    print(

        experiment_summary.to_string(

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

                "mean_delta":
                    lambda x:
                    f"{x:+.6f}",

                "mean_best_iteration":
                    lambda x:
                    f"{x:.0f}",
            },
        )
    )


    print(
        "\nBaseline mean:",
        f"{baseline_mean:.6f}",
    )


    print(
        "\nInterpretation:"
    )

    print(
        "delta < 0  -> 제거 후 성능 개선"
    )

    print(
        "delta = 0  -> 성능 동일"
    )

    print(
        "delta > 0  -> 제거 후 성능 악화"
    )


    # =====================================================
    # Save
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    all_fold_results.to_csv(
        OUTPUT_DIR
        / "ablation_fold_results.csv",

        index=False,

        encoding="utf-8-sig",
    )


    experiment_summary.to_csv(
        OUTPUT_DIR
        / "ablation_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )


    all_mean_importances.to_csv(
        OUTPUT_DIR
        / "ablation_mean_importance.csv",

        index=False,

        encoding="utf-8-sig",
    )


    all_per_fold_importances.to_csv(
        OUTPUT_DIR
        / "ablation_per_fold_importance.csv",

        index=False,

        encoding="utf-8-sig",
    )


    print(
        "\nSaved:",
        OUTPUT_DIR
    )


    print(
        "\nBaseline retraining:",
        RETRAIN_BASELINE
    )


    print(
        "\n주의:"
    )

    print(
        "Baseline은 과거 동일 설정에서 저장된 "
        "점수를 사용합니다."
    )

    print(
        "파이프라인 또는 전처리가 바뀌었다면 "
        "저장 Baseline과 직접 비교하면 안 됩니다."
    )


    return (
        all_fold_results,
        experiment_summary,
        all_mean_importances,
        all_per_fold_importances,
    )


# =========================================================
# 16. Execute
# =========================================================

if __name__ == "__main__":

    (
        all_fold_results,
        experiment_summary,
        all_mean_importances,
        all_per_fold_importances,
    ) = run_experiments()