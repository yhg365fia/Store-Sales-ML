import numpy as np
import pandas as pd
import lightgbm as lgb

from main import main
from feature_engineering import (
    add_lag_features,
    LAGS,
    LAG_FEATURES,
)


# =========================================================
# 1. 설정
# =========================================================

# Lag 추가 전 결과에서 가져온 임시 값.
# 최종 학습 전 Lag 추가 후 백테스트 결과를 보고 조정.
BEST_ITERATION = 4044

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

CATEGORICAL_COLS = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
]

TARGET_COL = "sales_log"


# =========================================================
# 2. 최종 제출 모델
# =========================================================

def main_submission():

    # -----------------------------------------------------
    # 1. 전체 Train 전처리 + Feature Engineering
    #
    # 최근 main() 반환 구조:
    # train_fe, test_fe, holidays
    #
    # train_fe에는 Validation으로 사용했던 기간까지 포함됨.
    # -----------------------------------------------------

    train_fe, test_fe, _ = main()

    train_fe = train_fe.copy()
    test_fe = test_fe.copy()

    train_count = len(train_fe)
    test_count = len(test_fe)

    # -----------------------------------------------------
    # 2. Train + Test 연결
    # -----------------------------------------------------

    train_fe["_is_test"] = False
    test_fe["_is_test"] = True

    # Lag 계산 후 Test의 원래 순서를 복원하기 위한 값
    train_fe["_test_order"] = -1
    test_fe["_test_order"] = np.arange(test_count)

    # Test의 정답은 알 수 없으므로 NaN 유지
    test_fe["sales"] = np.nan
    test_fe[TARGET_COL] = np.nan

    combined = pd.concat(
        [train_fe, test_fe],
        ignore_index=True,
    )

    combined["date"] = pd.to_datetime(combined["date"])

    group_cols = ["store_nbr", "family"]
    key_cols = ["date"] + group_cols

    # -----------------------------------------------------
    # 3. Lag 생성 전 QC
    # -----------------------------------------------------

    assert not combined[key_cols].isna().any().any(), (
        "날짜 / 매장 / 상품군에 결측치가 있습니다."
    )

    assert not combined.duplicated(key_cols).any(), (
        "날짜 × 매장 × 상품군 키가 중복됩니다."
    )

    assert not combined.loc[
        ~combined["_is_test"], "sales"
    ].isna().any(), (
        "Train 원본 판매량에 결측치가 있습니다."
    )

    # 매장 × 상품군별 날짜순 정렬
    combined = combined.sort_values(
        group_cols + ["date"]
    ).copy()

    # shift는 행 기준이므로 그룹별 날짜가 연속적인지 확인
    date_diff = (
        combined
        .groupby(group_cols, observed=True)["date"]
        .diff()
    )

    assert date_diff.dropna().eq(
        pd.Timedelta(days=1)
    ).all(), (
        "매장 × 상품군별 누락 날짜가 있습니다."
    )

    # Train / Test 날짜 경계
    train_end = combined.loc[
        ~combined["_is_test"], "date"
    ].max()

    test_start = combined.loc[
        combined["_is_test"], "date"
    ].min()

    test_end = combined.loc[
        combined["_is_test"], "date"
    ].max()

    assert test_start == train_end + pd.Timedelta(days=1), (
        "Test가 Train 마지막 날의 다음 날에 시작하지 않습니다."
    )

    assert min(LAGS) >= (test_end - train_end).days, (
        "최소 Lag가 예측 기간보다 짧아 "
        "Test 기간의 판매량을 참조합니다."
    )

    # -----------------------------------------------------
    # 4. 연결한 데이터에서 Lag 계산
    # -----------------------------------------------------

    combined = add_lag_features(
        combined,
        lags=LAGS,
    )

    # 각 그룹의 처음 lag개 행에만 결측치가 있어야 함.
    # Test Lag 결측치도 이 검사에서 확인됨.
    group_position = (
        combined
        .groupby(group_cols, observed=True)
        .cumcount()
    )

    for lag in LAGS:
        col = f"sales_lag_{lag}"
        expected_missing = group_position < lag

        assert combined[col].isna().eq(
            expected_missing
        ).all(), (
            f"{col}: 초기 과거 기록 부족 이외의 "
            "결측치가 발생했습니다."
        )

    # -----------------------------------------------------
    # 5. 전체 Train / Test 다시 분리
    # -----------------------------------------------------

    full_train = (
        combined.loc[~combined["_is_test"]]
        .drop(columns=["_is_test", "_test_order"])
        .copy()
    )

    test_fe = (
        combined.loc[combined["_is_test"]]
        .sort_values("_test_order")
        .drop(
            columns=[
                "_is_test",
                "_test_order",
                "sales",
                TARGET_COL,
            ]
        )
        .reset_index(drop=True)
    )

    assert len(full_train) == train_count, (
        "Lag 생성 전후 Train 행 수가 다릅니다."
    )

    assert len(test_fe) == test_count, (
        "Lag 생성 전후 Test 행 수가 다릅니다."
    )

    print("\n===== LAG FEATURES =====")
    print("Lags:", LAGS)
    print("Train Lag 결측치:")
    print(full_train[LAG_FEATURES].isna().sum())
    print("Test Lag 결측치:")
    print(test_fe[LAG_FEATURES].isna().sum())

    # -----------------------------------------------------
    # 6. 모델 입력 데이터 준비
    # -----------------------------------------------------

    X_train = full_train[FEATURE_COLS].copy()
    y_train = full_train[TARGET_COL].copy()

    X_test = test_fe[FEATURE_COLS].copy()

    # -----------------------------------------------------
    # 7. 범주형 기준 통일
    # -----------------------------------------------------

    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype("category")

        X_test[col] = pd.Categorical(
            X_test[col],
            categories=X_train[col].cat.categories,
        )

    # -----------------------------------------------------
    # 8. 모델 입력 QC
    # -----------------------------------------------------

    # Train의 기존 피처는 결측치가 없어야 함
    assert not X_train[
        BASE_FEATURE_COLS
    ].isna().any().any(), (
        "Train의 기존 피처에 결측치가 있습니다."
    )

    # Train의 초기 Lag 결측치는 그대로 유지.
    # 위에서 정상적인 위치인지 검증했고,
    # LightGBM이 결측값으로 처리하도록 함.

    assert not y_train.isna().any(), (
        "Train Target에 결측치가 있습니다."
    )

    # Test는 기존 피처와 Lag 모두 결측치가 없어야 함
    assert not X_test.isna().any().any(), (
        "Test 입력 피처에 결측치가 있습니다."
    )

    print("\n===== FINAL TRAIN DATA =====")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("Features:", len(FEATURE_COLS))

    print(
        "Train Period:",
        full_train["date"].min(),
        "~",
        full_train["date"].max(),
    )

    print(
        "Test Period:",
        test_fe["date"].min(),
        "~",
        test_fe["date"].max(),
    )

    # -----------------------------------------------------
    # 9. 최종 모델 생성
    # -----------------------------------------------------

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

        importance_type="split",

        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    # -----------------------------------------------------
    # 10. 전체 Train으로 학습
    # -----------------------------------------------------

    print("\n===== FINAL MODEL TRAINING =====")
    print("Iterations:", BEST_ITERATION)

    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLS,
    )

    # -----------------------------------------------------
    # 11. Test 예측
    # -----------------------------------------------------

    pred_log = model.predict(X_test)

    pred_sales = np.expm1(
        np.maximum(pred_log, 0)
    )

    # -----------------------------------------------------
    # 12. Submission 생성
    # -----------------------------------------------------

    prediction = pd.DataFrame({
        "id": test_fe["id"].to_numpy(),
        "sales": pred_sales,
    })

    sample = pd.read_csv(
        "data/sample_submission.csv"
    )

    assert prediction["id"].is_unique, (
        "Test id가 중복됩니다."
    )

    assert sample["id"].is_unique, (
        "sample_submission의 id가 중복됩니다."
    )

    assert set(prediction["id"]) == set(sample["id"]), (
        "Test와 sample_submission의 id가 일치하지 않습니다."
    )

    # sample_submission의 id 순서에 맞춰 예측값 연결
    submission = sample[["id"]].merge(
        prediction,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # -----------------------------------------------------
    # 13. Submission QC
    # -----------------------------------------------------

    assert submission["sales"].notna().all()
    assert len(submission) == len(sample)
    assert (submission["sales"] >= 0).all()
    assert np.isfinite(submission["sales"]).all()

    assert np.array_equal(
        submission["id"].to_numpy(),
        sample["id"].to_numpy(),
    ), (
        "제출파일의 id 순서가 sample_submission과 다릅니다."
    )

    # -----------------------------------------------------
    # 14. 저장
    # -----------------------------------------------------

    output_path = "submission_full_train.csv"

    submission.to_csv(
        output_path,
        index=False,
    )

    print("\n===== FINAL SUBMISSION =====")
    print("Saved:", output_path)
    print("Shape:", submission.shape)

    print(
        submission
        .head(10)
        .to_string(index=False)
    )

    return model, submission


# =========================================================
# 3. 실행
# =========================================================

if __name__ == "__main__":

    model, submission = main_submission()