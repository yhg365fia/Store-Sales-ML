
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 0. 경로 설정
# =========================================================

def get_data_dir():
    """
    프로젝트 루트 또는 src/, notebook/에서 실행해도
    data/ 폴더를 찾습니다.
    """
    cwd = Path.cwd()

    for path in [cwd, cwd.parent]:
        if (path / "data").is_dir():
            return path / "data"

    raise FileNotFoundError("data 폴더를 찾을 수 없습니다.")


# =========================================================
# 1. 데이터 로드
# =========================================================

def load_data(data_dir=None):
    """
    train.csv, test.csv, holidays_events.csv를 불러옵니다.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    holidays = pd.read_csv(data_dir / "holidays_events.csv")

    return train, test, holidays


# =========================================================
# 2. 날짜형 변환
# =========================================================

def convert_date_columns(train, test, holidays):
    """
    date 컬럼을 datetime으로 변환합니다.
    원본 데이터는 유지합니다.
    """
    train = train.copy()
    test = test.copy()
    holidays = holidays.copy()

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    return train, test, holidays


# =========================================================
# 3. 누락 날짜 탐색
# =========================================================

def find_missing_dates(train):
    """
    최소 날짜와 최대 날짜 사이에서
    전체 데이터가 빠진 날짜를 찾습니다.
    """
    full_dates = pd.date_range(
        train["date"].min(),
        train["date"].max(),
        freq="D"
    )

    missing_dates = full_dates.difference(
        train["date"].drop_duplicates()
    )

    return missing_dates


# =========================================================
# 4. 누락된 크리스마스 날짜 복원
# =========================================================

def add_missing_christmas_rows(train):
    """
    2013~2016년 12월 25일 누락 행을 복원합니다.

    기존 프로젝트 가정:
    - 누락 날짜를 휴점일로 해석
    - 모든 store × family에 sales=0,
      onpromotion=0을 부여
    - 추가 행은 is_added_missing_date=1로 표시

    주의: sales=0은 실제 관측값이 아니라 가정입니다.
    """
    train = train.copy()

    missing_dates = find_missing_dates(train)

    if len(missing_dates) == 0:
        if "is_added_missing_date" not in train.columns:
            train["is_added_missing_date"] = 0
        return train

    non_christmas = [
        date for date in missing_dates
        if not (date.month == 12 and date.day == 25)
    ]

    if non_christmas:
        raise ValueError(
            f"12월 25일 외의 누락 날짜가 발견되었습니다: {non_christmas}"
        )

    stores = train["store_nbr"].unique()
    families = train["family"].unique()

    missing_rows = pd.MultiIndex.from_product(
        [missing_dates, stores, families],
        names=["date", "store_nbr", "family"]
    ).to_frame(index=False)

    missing_rows["sales"] = 0.0
    missing_rows["onpromotion"] = 0
    missing_rows["is_added_missing_date"] = 1

    # Kaggle 원본 id와 겹치지 않도록 음수 ID 생성
    missing_rows["id"] = -np.arange(
        1, len(missing_rows) + 1
    )

    train["is_added_missing_date"] = 0

    missing_rows = missing_rows[
        [
            "id",
            "date",
            "store_nbr",
            "family",
            "sales",
            "onpromotion",
            "is_added_missing_date",
        ]
    ]

    train = pd.concat(
        [train, missing_rows],
        ignore_index=True
    )

    train = (
        train
        .sort_values(["date", "store_nbr", "family"])
        .reset_index(drop=True)
    )

    return train


# =========================================================
# 5. 범주형 변수 처리
# =========================================================

def convert_categories(df):
    """
    매장 번호와 상품군을 범주형으로 변환합니다.
    """
    df = df.copy()

    df["store_nbr"] = df["store_nbr"].astype("category")
    df["family"] = df["family"].astype("category")

    return df


# =========================================================
# 6. Holiday 기본 전처리
# =========================================================

def preprocess_holidays(holidays):
    """
    holidays 기본 전처리:

    - 완전히 중복된 행 제거
    - transferred=True인 원래 휴일 날짜 제외
    - 날짜 및 지역 기준 정렬

    Regional / Local 이벤트는 여기서 삭제하지 않고,
    Feature Engineering 병합 단계에서 제외합니다.
    """
    holidays = holidays.copy()

    holidays = holidays.drop_duplicates()

    # 실제 bool이거나 문자열 True/False인 경우 모두 대응
    holidays["transferred"] = (
        holidays["transferred"]
        .astype("string")
        .str.lower()
        .map({"true": True, "false": False})
        .astype("boolean")
    )

    holidays = holidays[
        holidays["transferred"] == False
    ].copy()

    holidays["transferred"] = (
        holidays["transferred"].astype(bool)
    )

    holidays = (
        holidays
        .sort_values(["date", "locale", "locale_name"])
        .reset_index(drop=True)
    )

    return holidays


# =========================================================
# 7. Train / Validation Split
# =========================================================

def split_train_valid(train, valid_days=16):
    """
    마지막 16일을 Validation으로 분리합니다.
    실제 Kaggle Test 예측 기간과 동일하게 설정합니다.
    """
    valid_start = (
        train["date"].max()
        - pd.Timedelta(days=valid_days - 1)
    )

    train_df = train[
        train["date"] < valid_start
    ].copy()

    valid_df = train[
        train["date"] >= valid_start
    ].copy()

    return train_df, valid_df


# =========================================================
# 8. Target 변환
# =========================================================

def add_target_log(df):
    """
    sales 원본 유지.
    sales_log = log1p(sales) 생성.
    """
    df = df.copy()

    df["sales_log"] = np.log1p(df["sales"])

    return df


# =========================================================
# 9. 전처리 QC
# =========================================================

def print_preprocessing_summary(
    train, train_df, valid_df, test, holidays
):
    """
    전처리 결과를 확인합니다.
    """
    print("===== DATE RANGE =====")
    print(
        "Train:",
        train_df["date"].min(),
        "~",
        train_df["date"].max()
    )
    print(
        "Valid:",
        valid_df["date"].min(),
        "~",
        valid_df["date"].max()
    )
    print(
        "Test :",
        test["date"].min(),
        "~",
        test["date"].max()
    )

    print("\n===== SHAPE =====")
    print("train_df:", train_df.shape)
    print("valid_df:", valid_df.shape)
    print("test    :", test.shape)
    print("holidays:", holidays.shape)

    print("\n===== MISSING DATES =====")
    missing_dates = find_missing_dates(train)
    print("count:", len(missing_dates))
    print(missing_dates)

    print("\n===== ADDED CHRISTMAS ROWS =====")
    print(
        "count:",
        int(train["is_added_missing_date"].sum())
    )

    print("\n===== SALES SKEW =====")
    print("original:", train_df["sales"].skew())
    print("log1p   :", train_df["sales_log"].skew())

    print("\n===== NULL VALUES =====")
    print("train_df:", int(train_df.isna().sum().sum()))
    print("valid_df:", int(valid_df.isna().sum().sum()))
    print("test    :", int(test.isna().sum().sum()))

    print("\n===== DUPLICATES =====")
    print("Train:", int(train_df.duplicated().sum()))
    print("Valid:", int(valid_df.duplicated().sum()))


# =========================================================
# 10. 전체 전처리 함수
# =========================================================

def run_preprocessing(data_dir=None, valid_days=16):
    """
    전체 전처리 순서:

    1) 데이터 로드
    2) 날짜형 변환
    3) 누락된 크리스마스 날짜 복원
    4) 범주형 변환
    5) Holiday 기본 전처리
    6) Train / Validation Split
    7) Target 변환
    8) QC 출력

    반환:
    train_df, valid_df, test, holidays
    """
    train, test, holidays = load_data(data_dir)

    train, test, holidays = convert_date_columns(
        train, test, holidays
    )

    train = add_missing_christmas_rows(train)

    # Test에는 인위적으로 추가한 행 없음
    test["is_added_missing_date"] = 0

    train = convert_categories(train)
    test = convert_categories(test)

    holidays = preprocess_holidays(holidays)

    train_df, valid_df = split_train_valid(
        train,
        valid_days=valid_days
    )

    train_df = add_target_log(train_df)
    valid_df = add_target_log(valid_df)

    print_preprocessing_summary(
        train,
        train_df,
        valid_df,
        test,
        holidays
    )

    return train_df, valid_df, test, holidays
