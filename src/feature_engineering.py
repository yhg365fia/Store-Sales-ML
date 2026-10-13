import numpy as np
import pandas as pd


# =========================================================
# 1. 날짜 피처 생성
# =========================================================

def add_date_features(df):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    return df


# =========================================================
# 2. 프로모션 피처 생성
# =========================================================

def add_promotion_features(df):

    df = df.copy()

    df["is_promotion"] = (
        df["onpromotion"] > 0
    ).astype(int)

    return df


# =========================================================
# 3. Lag 피처 생성
# =========================================================

LAGS = (16, 21, 28)

LAG_FEATURES = [
    f"sales_lag_{lag}" for lag in LAGS
]


def add_lag_features(df, lags=LAGS):

    """
    매장 × 상품군별 과거 판매량 피처 생성.

    전제:
    - 각 매장 × 상품군에 날짜별 한 행이 존재해야 함.
    - 날짜 누락이 없어야 shift(lag)가 lag일 전을 의미함.
    - 16일 일괄 예측을 기준으로 lag >= 16 사용.

    시계열 시작 부분의 Lag 결측치는 그대로 유지함.
    """

    if any(lag < 16 for lag in lags):
        raise ValueError(
            "16일 일괄 예측에서는 lag를 16 이상으로 설정하세요."
        )

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["store_nbr", "family", "date"]
    )

    grouped_sales = df.groupby(
        ["store_nbr", "family"],
        observed=True,
    )["sales"]

    for lag in lags:
        df[f"sales_lag_{lag}"] = grouped_sales.shift(lag)

    return df


# =========================================================
# 4. 일반 Feature Engineering 파이프라인
# =========================================================

def apply_feature_engineering(df):

    df = add_date_features(df)
    df = add_promotion_features(df)

    # Lag는 과거 판매량이 준비된 데이터에서 별도로 호출
    # Test만으로는 과거 판매량을 가져올 수 없기 때문

    return df


# =========================================================
# 5. Holiday 피처 실험용 목록
# =========================================================

# A: Holiday 개수만 사용
HOLIDAY_FEATURES_A = [
    "holiday_count"
]


# B: 이벤트 종류만 사용
HOLIDAY_FEATURES_B = [
    "event_christmas",
    "event_new_year",
    "event_carnival",
    "event_mothers_day",
    "event_independence_day",
    "event_city_foundation",
    "event_good_friday",
    "event_labor_day",
    "event_other"
]


# C: Holiday 개수 + 이벤트 종류
HOLIDAY_FEATURES_C = (
    HOLIDAY_FEATURES_A
    + HOLIDAY_FEATURES_B
)