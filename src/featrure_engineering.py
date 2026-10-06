import numpy as np
import pandas as pd


# =========================================================
# 1. 날짜 Feature
# =========================================================

def add_date_features(df):
    """
    date 컬럼에서 날짜 관련 feature를 생성합니다.
    """
    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df


# =========================================================
# 2. Promotion Feature
# =========================================================

def add_promotion_features(df):
    """
    onpromotion 컬럼에서 파생 feature를 생성합니다.

    - onpromotion_log:
      긴 오른쪽 꼬리를 완화하기 위한 log1p 변환

    - is_promotion:
      해당 날짜/매장/상품군에 프로모션이 하나라도 있는지 여부
    """
    df = df.copy()

    df["onpromotion_log"] = np.log1p(df["onpromotion"])
    df["is_promotion"] = (df["onpromotion"] > 0).astype(int)

    return df


# =========================================================
# 3. 전체 Feature Engineering
# =========================================================

def apply_feature_engineering(df):
    """
    현재까지 확정한 feature engineering을 한 번에 적용합니다.
    """
    df = add_date_features(df)
    df = add_promotion_features(df)

    return df
