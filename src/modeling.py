import numpy as np
import pandas as pd
import lightgbm as lgb

from main import main
from evaluation import evaluate_model
from feature_engineering import (
    add_lag_features,
    LAGS,
    LAG_FEATURES,
)


# =========================================================
# 1. Feature 설정
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

FEATURE_COLS = BASE_FEATURE_COLS + LAG_FEATURES

TARGET_COL = "sales_log"

CATEGORICAL_COLS = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
]


# =========================================================
# 2. Backtesting Fold 설정
# =========================================================

BACKTEST_FOLDS = [
    {
        "name": "Fold 1",
        "valid_start": "2017-06-29",
        "valid_end": "2017-07-14",
    },
    {
        "name": "Fold 2",
        "valid_start": "2017-07-15",
        "valid_end": "2017-07-30",
    },
    {
        "name": "Fold 3",
        "valid_start": "2017-07-31",
        "valid_end": "2017-08-15",
    },
]


# =========================================================
# 3. Lag 생성 및 QC
# =========================================================

def prepare_lag_features(train_fe):

    train_fe = train_fe.copy()
    train_fe["date"] = pd.to_datetime(train_fe["date"])

    group_cols = ["store_nbr", "family"]
    key_cols = ["date"] + group_cols

    # 날짜・매장・상품군 및 원본 판매량 확인
    assert not train_fe[key_cols + ["sales"]].isna().any().any(), (
        "Lag 생성에 필요한 컬럼에 결측치가 있습니다."
    )

    assert not train_fe.duplicated(key_cols).any(), (
        "날짜 × 매장 × 상품군 키가 중복됩니다."
    )

    train_fe = train_fe.sort_values(
        group_cols + ["date"]
    ).copy()

    # shift는 행 기준이므로 그룹별 날짜 간격이 1일인지 확인
    date_diff = (
        train_fe
        .groupby(group_cols, observed=True)["date"]
        .diff()
    )

    assert date_diff.dropna().eq(pd.Timedelta(days=1)).all(), (
        "매장 × 상품군별 날짜가 연속적이지 않습니다. "
        "누락 날짜를 확인하세요."
    )

    # 모든 Fold에서 Validation 판매량을 참조하지 않는지 확인
    for fold in BACKTEST_FOLDS:
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])

        horizon = (valid_end - valid_start).days + 1

        assert min(LAGS) >= horizon, (
            f"{fold['name']}: 최소 Lag가 예측 기간보다 짧습니다."
        )

    # 전체 기간에 생성한 뒤 Fold별로 분리
    train_fe = add_lag_features(train_fe, lags=LAGS)

    # 각 그룹의 처음 lag개 행에만 결측치가 있어야 함
    group_position = (
        train_fe
        .groupby(group_cols, observed=True)
        .cumcount()
    )

    for lag in LAGS:
        col = f"sales_lag_{lag}"
        expected_missing = group_position < lag

        assert train_fe[col].isna().eq(expected_missing).all(), (
            f"{col}: 시계열 시작 부분 이외에 결측치가 발생했습니다."
        )

    print()
    print("===== LAG FEATURES =====")
    print("Lags:", LAGS)
    print("추가된 피처:", LAG_FEATURES)
    print("전체 입력 피처 수:", len(FEATURE_COLS))
    print()
    print("학습 초기 과거 기록 부족으로 발생한 결측치:")
    print(train_fe[LAG_FEATURES].isna().sum())
    print("Lag QC: OK")

    return train_fe


# =========================================================
# 4. Fold 생성
# =========================================================

def make_fold(train_fe, valid_start, valid_end):

    valid_start = pd.Timestamp(valid_start)
    valid_end = pd.Timestamp(valid_end)

    # Validation 시작일 이전까지 전부 학습
    train_fold = train_fe[
        train_fe["date"] < valid_start
    ].copy()

    # 다음 16일 Validation
    valid_fold = train_fe[
        (train_fe["date"] >= valid_start)
        & (train_fe["date"] <= valid_end)
    ].copy()

    # QC
    assert len(train_fold) > 0, (
        "Train Fold가 비어 있습니다."
    )

    assert len(valid_fold) > 0, (
        "Validation Fold가 비어 있습니다."
    )

    assert (
        train_fold["date"].max()
        < valid_fold["date"].min()
    ), (
        "Train과 Validation의 시간 순서가 잘못되었습니다."
    )

    assert (
        (valid_end - valid_start).days + 1 == 16
        and valid_fold["date"].nunique() == 16
    ), (
        "Validation 기간이 16일이 아닙니다."
    )

    return train_fold, valid_fold


# =========================================================
# 5. 모델 입력 데이터 준비
# =========================================================

def prepare_model_data(train_fold, valid_fold):

    X_train = train_fold[FEATURE_COLS].copy()
    X_valid = valid_fold[FEATURE_COLS].copy()

    y_train = train_fold[TARGET_COL].copy()
    y_valid = valid_fold[TARGET_COL].copy()

    y_valid_sales = valid_fold["sales"].copy()

    # -----------------------------------------------------
    # 범주형 피처 기준 통일
    # -----------------------------------------------------

    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype("category")

        X_valid[col] = pd.Categorical(
            X_valid[col],
            categories=X_train[col].cat.categories,
        )

    # -----------------------------------------------------
    # 결측치 검사
    # -----------------------------------------------------

    # 기존 피처는 결측치를 허용하지 않음
    assert not X_train[BASE_FEATURE_COLS].isna().any().any(), (
        "Train의 기존 피처에 결측치가 있습니다."
    )

    assert not X_valid[BASE_FEATURE_COLS].isna().any().any(), (
        "Validation의 기존 피처에 결측치가 있습니다."
    )

    # Train의 Lag 초기 결측치는 prepare_lag_features에서 검증함.
    # 그대로 유지하여 LightGBM이 결측값으로 처리하도록 함.

    # 현재 Validation 기간에는 충분한 과거 기록이 있으므로
    # Lag 결측치가 없어야 함
    assert not X_valid[LAG_FEATURES].isna().any().any(), (
        "Validation의 Lag 피처에 결측치가 있습니다."
    )

    assert not y_train.isna().any(), (
        "Train Target에 결측치가 있습니다."
    )

    assert not y_valid.isna().any(), (
        "Validation Target에 결측치가 있습니다."
    )

    assert not y_valid_sales.isna().any(), (
        "Validation 원본 판매량에 결측치가 있습니다."
    )

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
        y_valid_sales,
    )


# =========================================================
# 6. LightGBM 모델 설정
# =========================================================

def create_model():

    model = lgb.LGBMRegressor(
        # 학습 방식
        boosting_type="gbdt",
        objective="regression",

        # 학습 속도
        learning_rate=0.05,
        n_estimators=5000,

        # 트리 구조
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        min_split_gain=0.0,

        # 샘플링
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,

        # 정규화
        reg_alpha=0.0,
        reg_lambda=0.0,

        # 실행 설정
        metric="rmse",
        importance_type="split",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    return model


# =========================================================
# 7. 모델 학습
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_valid, y_valid)
        ],

        eval_metric="rmse",
        categorical_feature=CATEGORICAL_COLS,

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50
            ),
            lgb.log_evaluation(
                period=100
            ),
        ],
    )

    return model


# =========================================================
# 8. Validation 평가
# =========================================================

def validate_model(model, X_valid, y_valid_sales):

    pred_log = model.predict(
        X_valid,
        num_iteration=model.best_iteration_,
    )

    rmsle = evaluate_model(
        y_valid=y_valid_sales,
        pred_log=pred_log,
    )

    return rmsle


# =========================================================
# 9. Feature Importance
# =========================================================

def get_feature_importance(model, fold_name):

    importance_df = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_,
    })

    importance_df["fold"] = fold_name

    return importance_df


# =========================================================
# 10. 3-Fold Backtesting
# =========================================================

def run_backtesting():

    # -----------------------------------------------------
    # 전체 Train 전처리 + 일반 Feature Engineering
    #
    # main() 반환:
    # train_fe, test_fe, holidays
    # -----------------------------------------------------

    train_fe, test_fe, holidays = main()

    # -----------------------------------------------------
    # Lag 생성
    #
    # 전체 Train에서 생성한 뒤 Fold를 분리.
    # Validation 16일, 최소 Lag 16일이므로
    # Validation 행의 Lag는 해당 Fold의 학습 기간만 참조.
    # -----------------------------------------------------

    train_fe = prepare_lag_features(train_fe)

    fold_results = []
    importance_results = []

    # -----------------------------------------------------
    # Fold 반복
    # -----------------------------------------------------

    for fold in BACKTEST_FOLDS:

        fold_name = fold["name"]

        print()
        print("=" * 60)
        print(f"{fold_name} START")
        print("=" * 60)

        # 1. Fold 생성
        train_fold, valid_fold = make_fold(
            train_fe,
            valid_start=fold["valid_start"],
            valid_end=fold["valid_end"],
        )

        print(
            "Train:",
            train_fold["date"].min(),
            "~",
            train_fold["date"].max(),
        )

        print(
            "Valid:",
            valid_fold["date"].min(),
            "~",
            valid_fold["date"].max(),
        )

        print()
        print("Train shape:", train_fold.shape)
        print("Valid shape:", valid_fold.shape)

        # 2. 모델 입력 데이터 준비
        (
            X_train,
            X_valid,
            y_train,
            y_valid,
            y_valid_sales,
        ) = prepare_model_data(
            train_fold,
            valid_fold,
        )

        print()
        print("X_train:", X_train.shape)
        print("X_valid:", X_valid.shape)

        # 3. Fold마다 새 모델 생성
        model = create_model()

        # 4. 학습
        model = train_model(
            model,
            X_train,
            y_train,
            X_valid,
            y_valid,
        )

        # 5. Validation 평가
        rmsle = validate_model(
            model,
            X_valid,
            y_valid_sales,
        )

        print()
        print(
            f"{fold_name} Best Iteration:",
            model.best_iteration_,
        )
        print(
            f"{fold_name} RMSLE:",
            f"{rmsle:.6f}",
        )

        # 6. Fold 결과 저장
        fold_results.append({
            "fold": fold_name,
            "train_end": train_fold["date"].max(),
            "valid_start": valid_fold["date"].min(),
            "valid_end": valid_fold["date"].max(),
            "best_iteration": model.best_iteration_,
            "rmsle": rmsle,
        })

        # 7. Feature Importance 저장
        importance_df = get_feature_importance(
            model,
            fold_name,
        )

        importance_results.append(importance_df)

    # =====================================================
    # 11. Backtesting 결과 정리
    # =====================================================

    results_df = pd.DataFrame(fold_results)

    print()
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # RMSLE 요약
    mean_rmsle = results_df["rmsle"].mean()
    std_rmsle = results_df["rmsle"].std()

    best_fold = results_df.loc[
        results_df["rmsle"].idxmin(),
        "fold",
    ]

    worst_fold = results_df.loc[
        results_df["rmsle"].idxmax(),
        "fold",
    ]

    print()
    print(f"Mean RMSLE : {mean_rmsle:.6f}")
    print(f"Std RMSLE  : {std_rmsle:.6f}")
    print(f"Best Fold  : {best_fold}")
    print(f"Worst Fold : {worst_fold}")

    # Best Iteration 요약
    mean_iteration = results_df["best_iteration"].mean()
    median_iteration = results_df["best_iteration"].median()

    print()
    print("Mean Best Iteration:", round(mean_iteration))
    print("Median Best Iteration:", round(median_iteration))

    # =====================================================
    # 12. 평균 Feature Importance
    # =====================================================

    all_importance = pd.concat(
        importance_results,
        ignore_index=True,
    )

    mean_importance = (
        all_importance
        .groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean")
        )
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 60)
    print("MEAN FEATURE IMPORTANCE")
    print("=" * 60)
    print(mean_importance.to_string(index=False))

    # =====================================================
    # 13. 최종 상태
    # =====================================================

    print()
    print("=" * 60)
    print("BACKTEST STATUS")
    print("=" * 60)

    print("Folds:", len(BACKTEST_FOLDS))
    print("Features:", len(FEATURE_COLS))
    print("Lags:", LAGS)
    print("Mean RMSLE:", f"{mean_rmsle:.6f}")
    print("Std RMSLE:", f"{std_rmsle:.6f}")

    return results_df, mean_importance


# =========================================================
# 14. 프로그램 시작
# =========================================================

if __name__ == "__main__":

    results_df, mean_importance = run_backtesting()