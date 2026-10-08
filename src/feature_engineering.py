
import numpy as np
import pandas as pd


# =========================================================
# 1. 날짜 Feature
# =========================================================

def add_date_features(df):
    """
    date 컬럼에서 날짜 관련 feature 생성.

    - year: 연도
    - month: 월
    - day: 일
    - dayofweek: 요일 (월=0, 일=6)
    - is_weekend: 주말 여부 (0/1)
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

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
    프로모션 관련 파생 피처 생성.

    - onpromotion_log: log1p 변환
    - is_promotion: 프로모션 존재 여부 (0/1)
    """
    df = df.copy()

    df["onpromotion_log"] = np.log1p(df["onpromotion"])
    df["is_promotion"] = (
        df["onpromotion"] > 0
    ).astype(int)

    return df


# =========================================================
# 3. Categorical Feature
# =========================================================

def add_categorical_features(df):
    """
    매장 및 상품군을 범주형으로 변환.
    """
    df = df.copy()

    categorical_cols = [
        "store_nbr",
        "family"
    ]

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df


# =========================================================
# 4. Holiday Event Group 생성
# =========================================================

def add_event_group(holidays):
    """
    description을 기반으로 비슷한 이벤트를 그룹화.

    description 원본은 보존합니다.
    """
    holidays = holidays.copy()

    description = (
        holidays["description"]
        .fillna("")
        .str.lower()
    )

    holidays["event_group"] = "Other"

    event_patterns = {
        "Christmas": r"navidad",
        "New Year": r"primer dia del ano|ano nuevo",
        "Carnival": r"carnaval",
        "Mothers Day": r"dia de la madre",
        "Independence Day": r"independencia",
        "City Foundation": r"fundacion",
    }

    for group, pattern in event_patterns.items():
        mask = description.str.contains(
            pattern,
            regex=True,
            na=False
        )

        holidays.loc[mask, "event_group"] = group

    return holidays


# =========================================================
# 5. Holiday Feature
# =========================================================

def add_holiday_features(holidays):
    """
    Holiday 데이터 자료형 변환.

    - type: 이벤트 종류
    - locale: 적용 범위
    - locale_name: 적용 지역
    - description: 이벤트 원본 이름
    - event_group: 유사 이벤트 그룹
    - transferred: 이동 여부 (bool)
    """
    holidays = holidays.copy()

    holidays["date"] = pd.to_datetime(holidays["date"])

    holidays = add_event_group(holidays)

    categorical_cols = [
        "type",
        "locale",
        "locale_name",
        "description",
        "event_group"
    ]

    for col in categorical_cols:
        holidays[col] = holidays[col].astype("category")

    holidays["transferred"] = (
        holidays["transferred"].astype(bool)
    )

    return holidays


# =========================================================
# 6. Event Group Binary Features (National Only)
# =========================================================

def add_event_binary_features(holidays):
    """
    National 이벤트별 0/1 피처 생성.

    - National만 사용
    - 같은 날짜에 여러 이벤트가 있으면 각각 표시
    - 날짜별로 하나의 행으로 집계
    """
    national = holidays[
        holidays["locale"] == "National"
    ].copy()

    event_names = {
        "Christmas": "event_christmas",
        "New Year": "event_new_year",
        "Carnival": "event_carnival",
        "Mothers Day": "event_mothers_day",
        "Independence Day": "event_independence_day",
        "City Foundation": "event_city_foundation",
        "Other": "event_other"
    }

    for group, col in event_names.items():
        national[col] = (
            national["event_group"] == group
        ).astype(int)

    event_cols = list(event_names.values())

    # 같은 날짜의 복수 이벤트를 0/1로 통합
    event_daily = (
        national
        .groupby("date")[event_cols]
        .max()
        .reset_index()
    )

    return event_daily


# =========================================================
# 7. Holiday Merge (National Only)
# =========================================================

def merge_holiday_features(df, holidays):
    """
    판매 데이터에 National 이벤트 정보를 병합합니다.

    - National 이벤트만 사용
    - 같은 날짜의 복수 이벤트 집계
    - left merge로 판매 데이터 행 수 유지
    - 문자열 집계 피처와 이벤트별 0/1 피처 보존
    """
    df = df.copy()
    original_rows = len(df)

    # -----------------------------------------------------
    # 1. National 이벤트 선택
    # -----------------------------------------------------

    national = holidays[
        holidays["locale"] == "National"
    ].copy()

    # -----------------------------------------------------
    # 2. 복수 이벤트 문자열 및 개수 집계
    # -----------------------------------------------------

    holiday_agg = (
        national
        .groupby("date", observed=True)
        .agg(
            holiday_count=("type", "size"),

            holiday_types=(
                "type",
                lambda x: "|".join(
                    sorted(set(x.astype(str)))
                )
            ),

            event_groups=(
                "event_group",
                lambda x: "|".join(
                    sorted(set(x.astype(str)))
                )
            ),

            holiday_descriptions=(
                "description",
                lambda x: "|".join(
                    sorted(set(x.astype(str)))
                )
            )
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # 3. 이벤트별 0/1 피처 생성
    # -----------------------------------------------------

    event_daily = add_event_binary_features(holidays)

    event_cols = [
        col for col in event_daily.columns
        if col != "date"
    ]

    # -----------------------------------------------------
    # 4. 판매 데이터에 Left Merge
    # -----------------------------------------------------

    df = df.merge(
        holiday_agg,
        on="date",
        how="left",
        validate="many_to_one"
    )

    df = df.merge(
        event_daily,
        on="date",
        how="left",
        validate="many_to_one"
    )

    # -----------------------------------------------------
    # 5. 이벤트가 없는 날짜 처리
    # -----------------------------------------------------

    df["holiday_count"] = (
        df["holiday_count"]
        .fillna(0)
        .astype(int)
    )

    categorical_cols = [
        "holiday_types",
        "event_groups",
        "holiday_descriptions"
    ]

    for col in categorical_cols:
        df[col] = (
            df[col]
            .fillna("None")
            .astype("category")
        )

    # 이벤트별 0/1 피처 결측치 처리
    df[event_cols] = (
        df[event_cols]
        .fillna(0)
        .astype(int)
    )

    # 전국 이벤트 존재 여부
    df["has_national_event"] = (
        df["holiday_count"] > 0
    )

    # -----------------------------------------------------
    # 6. 병합 검증
    # -----------------------------------------------------

    assert len(df) == original_rows, (
        "Holiday 병합 과정에서 행 수가 변경되었습니다."
    )

    return df


# =========================================================
# 8. 전체 Feature Engineering
# =========================================================

def apply_feature_engineering(df):
    """
    판매 데이터 기본 Feature Engineering.

    1) 날짜 피처
    2) 프로모션 피처
    3) 범주형 변환

    Holiday 병합은 main.py에서 별도로 실행합니다.
    """
    df = add_date_features(df)
    df = add_promotion_features(df)
    df = add_categorical_features(df)

    return df


# =========================================================
# 9. Holiday 실험용 피처 조합
# =========================================================

# 실험 A: 문자열 Event Group
HOLIDAY_FEATURES_A = [
    "holiday_types",
    "event_groups",
    "holiday_count",
    "has_national_event"
]

# 실험 B: 이벤트별 0/1 피처
HOLIDAY_FEATURES_B = [
    "holiday_types",
    "holiday_count",
    "has_national_event",
    "event_christmas",
    "event_new_year",
    "event_carnival",
    "event_mothers_day",
    "event_independence_day",
    "event_city_foundation",
    "event_other"
]

# 실험 C: 문자열 + 0/1 피처
HOLIDAY_FEATURES_C = [
    "holiday_types",
    "event_groups",
    "holiday_descriptions",
    "holiday_count",
    "has_national_event",
    "event_christmas",
    "event_new_year",
    "event_carnival",
    "event_mothers_day",
    "event_independence_day",
    "event_city_foundation",
    "event_other"
]
