
import numpy as np
from sklearn.metrics import mean_squared_log_error


# =========================================================
# 1. RMSLE 계산
# =========================================================

def calculate_rmsle(y_true, y_pred):

    # 실제 판매량
    y_true = np.asarray(y_true, dtype=float)

    # 예측 판매량
    y_pred = np.asarray(y_pred, dtype=float)

    # 음수 예측값은 0으로 보정
    y_pred = np.maximum(y_pred, 0)

    # RMSLE 계산
    rmsle = np.sqrt(
        mean_squared_log_error(y_true, y_pred)
    )

    return rmsle


# =========================================================
# 2. 로그 예측값을 원래 판매량으로 복원
# =========================================================

def inverse_log_predictions(pred_log):

    pred_log = np.asarray(pred_log, dtype=float)

    # 로그 예측값을 0 이상으로 제한
    pred_log = np.maximum(pred_log, 0)

    # log1p의 역변환
    pred_sales = np.expm1(pred_log)

    return pred_sales


# =========================================================
# 3. Validation 모델 평가
# =========================================================

def evaluate_model(y_valid, pred_log):

    # -----------------------------------------------------
    # 1. 예측값 역변환
    # -----------------------------------------------------

    pred_sales = inverse_log_predictions(pred_log)

    # -----------------------------------------------------
    # 2. RMSLE 계산
    # -----------------------------------------------------

    rmsle = calculate_rmsle(
        y_true=y_valid,
        y_pred=pred_sales
    )

    # -----------------------------------------------------
    # 3. 평가 결과 출력
    # -----------------------------------------------------

    print("\n===== VALIDATION EVALUATION =====")
    print(f"RMSLE: {rmsle:.6f}")

    return rmsle
