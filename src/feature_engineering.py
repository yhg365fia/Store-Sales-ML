
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

    df["onpromotion_log"] = np.log1p(df["onpromotion"])

    df["is_promotion"] = (
        df["onpromotion"] > 0
    ).astype(int)

    return df


# =========================================================
# 3. 범주형 피처 변환
# =========================================================

def add_categorical_features(df):

    df = df.copy()

    df["store_nbr"] = df["store_nbr"].astype("category")
    df["family"] = df["family"].astype("category")

    return df


# =========================================================
# 4. Holiday 이벤트 그룹 분류
# =========================================================

def add_event_group(holidays):

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

        # 추가된 이벤트 그룹
        "Good Friday": r"viernes santo",
        "Labor Day": r"dia del trabajo"
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
# 5. Holiday 데이터 피처 엔지니어링
# =========================================================

def add_holiday_features(holidays):

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

    holidays["transferred"] = holidays["transferred"].astype(bool)

    return holidays


# =========================================================
# 6. National 이벤트별 0/1 피처 생성
# =========================================================

def add_event_binary_features(holidays):

    # -----------------------------------------------------
    # 1. National 이벤트만 선택
    # -----------------------------------------------------

    national = holidays[
        holidays["locale"] == "National"
    ].copy()

    # -----------------------------------------------------
    # 2. 이벤트 그룹별 컬럼 이름 지정
    # -----------------------------------------------------

    event_names = {
        "Christmas": "event_christmas",
        "New Year": "event_new_year",
        "Carnival": "event_carnival",
        "Mothers Day": "event_mothers_day",
        "Independence Day": "event_independence_day",
        "City Foundation": "event_city_foundation",

        # 추가된 이벤트별 0/1 피처
        "Good Friday": "event_good_friday",
        "Labor Day": "event_labor_day",

        "Other": "event_other"
    }

    # -----------------------------------------------------
    # 3. 이벤트별 0/1 컬럼 생성
    # -----------------------------------------------------

    for group, col in event_names.items():

        national[col] = (
            national["event_group"] == group
        ).astype(int)

    event_cols = list(event_names.values())

    # -----------------------------------------------------
    # 4. 날짜별 이벤트 존재 여부 집계
    # -----------------------------------------------------

    event_daily = (
        national
        .groupby("date")[event_cols]
        .max()
        .reset_index()
    )

    return event_daily


# =========================================================
# 7. National Holiday 피처 병합
# =========================================================

def merge_holiday_features(df, holidays):

    df = df.copy()

    original_rows = len(df)

    # -----------------------------------------------------
    # 1. National 이벤트만 선택
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
    # 5. Holiday 결측치 처리
    # -----------------------------------------------------

    df["holiday_count"] = (
        df["holiday_count"]
        .fillna(0)
        .astype(int)
    )

    holiday_text_cols = [
        "holiday_types",
        "event_groups",
        "holiday_descriptions"
    ]

    for col in holiday_text_cols:

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

    # -----------------------------------------------------
    # 6. National 이벤트 존재 여부
    # -----------------------------------------------------

    df["has_national_event"] = (
        df["holiday_count"] > 0
    )

    # -----------------------------------------------------
    # 7. 병합 후 행 수 검증
    # -----------------------------------------------------

    assert len(df) == original_rows, (
        "Holiday 병합 과정에서 행 수가 변경되었습니다."
    )

    return df


# =========================================================
# 8. 일반 Feature Engineering 파이프라인
# =========================================================

def apply_feature_engineering(df):

    df = add_date_features(df)

    df = add_promotion_features(df)

    df = add_categorical_features(df)

    return df


# =========================================================
# 9. Holiday 피처 실험용 목록
# =========================================================

# A: 문자열 기반 Holiday 피처
HOLIDAY_FEATURES_A = [
    "holiday_types",
    "event_groups",
    "holiday_descriptions"
]

# B: 이벤트별 0/1 기반 Holiday 피처
HOLIDAY_FEATURES_B = [
    "has_national_event",
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

# C: 문자열 + 이벤트별 0/1 피처
HOLIDAY_FEATURES_C = (
    HOLIDAY_FEATURES_A
    + HOLIDAY_FEATURES_B
)
