
import numpy as np
import pandas as pd
import lightgbm as lgb

from main import main
from evaluation import evaluate_model


# =========================================================
# 1. Feature 설정
# =========================================================

FEATURE_COLS = [

    # 매장 / 상품군
    "store_nbr",
    "family",

    # 날짜
    "year",
    "month",
    "day",
    "dayofweek",
    "is_weekend",

    # 프로모션
    "onpromotion",
    "onpromotion_log",
    "is_promotion",

    # Holiday
    "has_national_event",
    "event_christmas",
    "event_new_year",
    "event_carnival",
    "event_mothers_day",
    "event_independence_day",
    "event_good_friday",
    "event_labor_day",
    "event_other"
]

TARGET_COL = "sales_log"

CATEGORICAL_COLS = [
    "store_nbr",
    "family"
]


# =========================================================
# 2. 데이터 준비
# =========================================================

def prepare_model_data(train_fe, valid_fe):

    X_train = train_fe[FEATURE_COLS].copy()
    X_valid = valid_fe[FEATURE_COLS].copy()

    y_train = train_fe[TARGET_COL].copy()
    y_valid = valid_fe[TARGET_COL].copy()

    y_valid_sales = valid_fe["sales"].copy()

    # 범주형 피처 통일
    for col in CATEGORICAL_COLS:

        X_train[col] = X_train[col].astype("category")

        X_valid[col] = pd.Categorical(
            X_valid[col],
            categories=X_train[col].cat.categories
        )

    # 결측치 검사
    assert not X_train.isna().any().any()
    assert not X_valid.isna().any().any()
    assert not y_train.isna().any()
    assert not y_valid.isna().any()

    print("\n===== MODEL DATA SHAPE =====")
    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("y_train:", y_train.shape)
    print("y_valid:", y_valid.shape)

    print("\n===== FEATURES =====")
    print(FEATURE_COLS)
    print("Total:", len(FEATURE_COLS))

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
        y_valid_sales
    )


# =========================================================
# 3. LightGBM 모델 설정
# =========================================================

def create_model():

    model = lgb.LGBMRegressor(

        # 학습 방식
        boosting_type="gbdt",
        objective="regression",

        # 학습 속도
        learning_rate=0.05,
        n_estimators=1300,

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
        verbosity=-1
    )

    return model


# =========================================================
# 4. 모델 학습
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid
):

    print("\n===== LIGHTGBM TRAINING =====")

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
                period=50
            )
        ]
    )

    print("\nBest Iteration:", model.best_iteration_)

    return model


# =========================================================
# 5. Validation 평가
# =========================================================

def validate_model(model, X_valid, y_valid_sales):

    # 로그 판매량 예측
    pred_log = model.predict(
        X_valid,
        num_iteration=model.best_iteration_
    )

    # RMSLE
    rmsle = evaluate_model(
        y_valid=y_valid_sales,
        pred_log=pred_log
    )

    # 원본 판매량 복원
    pred_sales = np.expm1(
        np.maximum(pred_log, 0)
    )

    # 예측 결과 출력
    result_df = pd.DataFrame({
        "actual_sales": np.asarray(y_valid_sales),
        "predicted_sales": pred_sales
    })

    print("\n===== PREDICTION SAMPLE =====")
    print(result_df.head(10).to_string(index=False))

    return rmsle


# =========================================================
# 6. Feature Importance
# =========================================================

def show_feature_importance(model):

    importance_df = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    print("\n===== FEATURE IMPORTANCE =====")
    print(importance_df.to_string(index=False))

    return importance_df


# =========================================================
# 7. Kaggle Submission 생성
# =========================================================

def create_submission(model, test_fe, train_fe):

    # -----------------------------------------------------
    # 1. Test 입력 피처 준비
    # -----------------------------------------------------

    X_test = test_fe[FEATURE_COLS].copy()

    # Train과 범주형 기준 통일
    for col in CATEGORICAL_COLS:

        X_test[col] = pd.Categorical(
            X_test[col],
            categories=train_fe[col].cat.categories
        )

    assert not X_test.isna().any().any(), (
        "X_test에 결측치가 있습니다."
    )

    # -----------------------------------------------------
    # 2. Test 판매량 예측
    # -----------------------------------------------------

    pred_log = model.predict(
        X_test,
        num_iteration=model.best_iteration_
    )

    # 로그 역변환 및 음수 보정
    pred_sales = np.expm1(
        np.maximum(pred_log, 0)
    )

    # -----------------------------------------------------
    # 3. 제출 데이터 생성
    # -----------------------------------------------------

    submission = pd.DataFrame({
        "id": test_fe["id"].to_numpy(),
        "sales": pred_sales
    })

    # -----------------------------------------------------
    # 4. Sample Submission 형식 맞추기
    # -----------------------------------------------------

    sample = pd.read_csv(
        "data/sample_submission.csv"
    )

    submission = sample[["id"]].merge(
        submission,
        on="id",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # 5. 제출 데이터 검증
    # -----------------------------------------------------

    assert submission["sales"].notna().all(), (
        "제출 파일에 결측치가 있습니다."
    )

    assert len(submission) == len(sample), (
        "제출 파일의 행 수가 다릅니다."
    )

    assert (submission["sales"] >= 0).all(), (
        "제출 파일에 음수 판매량이 있습니다."
    )

    assert np.isfinite(submission["sales"]).all(), (
        "제출 파일에 무한대 값이 있습니다."
    )

    assert list(submission.columns) == ["id", "sales"], (
        "제출 파일의 컬럼 구성이 잘못되었습니다."
    )

    # -----------------------------------------------------
    # 6. CSV 저장
    # -----------------------------------------------------

    submission.to_csv(
        "submission.csv",
        index=False
    )

    print("\n===== KAGGLE SUBMISSION =====")
    print("Saved: submission.csv")
    print("Shape:", submission.shape)

    print(
        submission.head(10).to_string(index=False)
    )

    return submission


# =========================================================
# 8. 전체 모델링 파이프라인
# =========================================================

def run_modeling():

    # -----------------------------------------------------
    # 1. Preprocessing + Feature Engineering
    # -----------------------------------------------------

    train_fe, valid_fe, test_fe, holidays_fe = main()

    # -----------------------------------------------------
    # 2. 모델 입력 데이터
    # -----------------------------------------------------

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        y_valid_sales
    ) = prepare_model_data(
        train_fe,
        valid_fe
    )

    # -----------------------------------------------------
    # 3. 모델 생성
    # -----------------------------------------------------

    model = create_model()

    # -----------------------------------------------------
    # 4. 모델 학습
    # -----------------------------------------------------

    model = train_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    # -----------------------------------------------------
    # 5. Validation 평가
    # -----------------------------------------------------

    rmsle = validate_model(
        model,
        X_valid,
        y_valid_sales
    )

    # -----------------------------------------------------
    # 6. Feature Importance
    # -----------------------------------------------------

    importance_df = show_feature_importance(model)

    # -----------------------------------------------------
    # 7. Kaggle Submission
    # -----------------------------------------------------

    submission = create_submission(
        model,
        test_fe,
        train_fe
    )

    # -----------------------------------------------------
    # 8. 최종 결과
    # -----------------------------------------------------

    print("\n===== BASELINE RESULT =====")
    print("Model: LightGBM")
    print("Target:", TARGET_COL)
    print("Features:", len(FEATURE_COLS))
    print("Best Iteration:", model.best_iteration_)
    print(f"Validation RMSLE: {rmsle:.6f}")
    print("Submission: submission.csv")

    return model, rmsle, importance_df, submission


# =========================================================
# 9. 프로그램 시작
# =========================================================

if __name__ == "__main__":

    model, rmsle, importance_df, submission = run_modeling()
