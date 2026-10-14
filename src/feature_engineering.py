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
# 3. Lag 피처 생성 — 3개 모두 사용
# =========================================================

LAGS = (16, 21, 28)

LAG_FEATURES = [
    f"sales_lag_{lag}"
    for lag in LAGS
]


def add_lag_features(df, lags=LAGS):

    """
    매장 × 상품군별 과거 판매량 피처 생성.

    기본 피처:
    - sales_lag_16
    - sales_lag_21
    - sales_lag_28

    전제:
    - 각 매장 × 상품군에 날짜별 한 행이 존재해야 함.
    - 날짜 누락이 없어야 shift(lag)가 lag일 전을 의미함.
    - 16일 일괄 예측을 기준으로 lag >= 16 사용.

    초기 과거 기록 부족으로 발생한 결측치는 유지함.
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
# 4. Rolling mean 피처 생성 — 3개 모두 사용
# =========================================================

ROLLING_SHIFT = 16
ROLLING_WINDOWS = (7, 14, 28)

ROLLING_MEAN_FEATURES = [
    f"sales_rolling_mean_{ROLLING_SHIFT}_{window}"
    for window in ROLLING_WINDOWS
]

ROLLING_FEATURES = ROLLING_MEAN_FEATURES.copy()

# 모델링 / 최종 제출 코드의 QC용 설정
ROLLING_WINDOWS_BY_STAT = {
    "mean": ROLLING_WINDOWS,
}


def add_rolling_features(
    df,
    shift=ROLLING_SHIFT,
    windows=ROLLING_WINDOWS,
):

    """
    매장 × 상품군별 과거 판매량 이동평균 생성.

    기본 피처:
    - sales_rolling_mean_16_7
    - sales_rolling_mean_16_14
    - sales_rolling_mean_16_28

    사용 범위:
    - window=7: 16~22일 전 판매량 평균.
    - window=14: 16~29일 전 판매량 평균.
    - window=28: 16~43일 전 판매량 평균.

    전제:
    - 날짜별 한 행이며 날짜 누락이 없어야 함.
    - 16일 일괄 예측을 기준으로 shift >= 16 사용.

    유효 판매량이 window개보다 적으면 결측치 유지.
    """

    if shift < 16:
        raise ValueError(
            "16일 일괄 예측에서는 shift를 16 이상으로 설정하세요."
        )

    if any(window < 1 for window in windows):
        raise ValueError(
            "평균 구간의 크기는 1 이상으로 설정하세요."
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

    for window in windows:
        df[f"sales_rolling_mean_{shift}_{window}"] = (
            grouped_sales.transform(
                lambda x: (
                    x.shift(shift)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

    return df


# =========================================================
# 5. EMA 피처 생성 — 28 / 364
# =========================================================

EMA_SHIFT = 16
EMA_SPANS = (28, 364)
EMA_MIN_PERIODS = 1

EMA_FEATURES = [
    f"sales_ema_{EMA_SHIFT}_{span}"
    for span in EMA_SPANS
]


def add_ema_features(
    df,
    shift=EMA_SHIFT,
    spans=EMA_SPANS,
):

    """
    매장 × 상품군별 과거 판매량의 지수이동평균 생성.

    기본 피처:
    - sales_ema_16_28
    - sales_ema_16_364

    설정:
    - shift=16: 예측 대상 날짜의 16일 전까지 사용.
    - adjust=False: 재귀식 방식.
    - min_periods=1: 첫 유효 판매량부터 계산.

    주의:
    - span은 가중치 감소 속도를 정하는 설정.
    - span=364는 364일 전 판매량을 가져오는 설정이 아님.
    - 첫 유효 판매량을 초기 EMA로 사용함.

    전제:
    - 날짜별 한 행이며 날짜 누락이 없어야 함.
    - 16일 일괄 예측을 기준으로 shift >= 16 사용.

    원본 판매량에 결측치가 없다면
    각 그룹의 처음 shift개 행은 결측치로 남음.
    """

    if shift < 16:
        raise ValueError(
            "16일 일괄 예측에서는 shift를 16 이상으로 설정하세요."
        )

    if any(span < 1 for span in spans):
        raise ValueError(
            "EMA span은 1 이상으로 설정하세요."
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

    for span in spans:
        df[f"sales_ema_{shift}_{span}"] = (
            grouped_sales.transform(
                lambda x: (
                    x.shift(shift)
                    .ewm(
                        span=span,
                        adjust=False,
                        min_periods=EMA_MIN_PERIODS,
                        ignore_na=False,
                    )
                    .mean()
                )
            )
        )

    return df


# =========================================================
# 6. 전체 시계열 피처 목록 — 총 8개
# =========================================================

TIME_SERIES_FEATURES = (
    LAG_FEATURES
    + ROLLING_FEATURES
    + EMA_FEATURES
)

# 포함되는 피처:
# sales_lag_16
# sales_lag_21
# sales_lag_28
# sales_rolling_mean_16_7
# sales_rolling_mean_16_14
# sales_rolling_mean_16_28
# sales_ema_16_28
# sales_ema_16_364


# =========================================================
# 7. 일반 Feature Engineering 파이프라인
# =========================================================

def apply_feature_engineering(df):

    df = add_date_features(df)
    df = add_promotion_features(df)

    # Lag / Rolling mean / EMA는 과거 판매량이 준비된
    # 데이터에서 모델링 코드가 별도로 호출.
    # Test만으로는 과거 판매량을 가져올 수 없음.

    return df


# =========================================================
# 8. Holiday 피처 실험용 목록
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