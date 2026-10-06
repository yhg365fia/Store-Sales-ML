from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 0. 경로 설정
# =========================================================

def get_data_dir():
    """
    프로젝트 루트 또는 notebook/에서 실행해도 data/ 폴더를 찾습니다.
    """
    cwd = Path.cwd()

    if (cwd / "data").exists():
        return cwd / "data"

    if (cwd.parent / "data").exists():
        return cwd.parent / "data"

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

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    holidays = pd.read_csv(data_dir / "holidays_events.csv")

    return train, test, holidays


# =========================================================
# 2. 날짜형 변환
# =========================================================

def convert_date_columns(train, test, holidays):
    """
    각 데이터의 date 컬럼을 datetime으로 변환합니다.
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
    train의 최소 날짜 ~ 최대 날짜 사이에서
    데이터 전체가 통째로 빠진 날짜를 찾습니다.
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
    train에서 통째로 빠진 날짜를 찾고,
    현재 데이터에서 확인된 12월 25일 누락만 복원합니다.

    판단:
    - 2013~2016년 12월 25일이 4년 연속 통째로 빠져 있음
    - holidays에는 Navidad / National / transferred=False로 존재
    - 휴점일로 해석하여 모든 store × family 조합의
      sales=0, onpromotion=0 행을 추가

    원본에 없던 행은 is_added_missing_date=1로 표시합니다.
    """
    train = train.copy()

    missing_dates = find_missing_dates(train)

    if len(missing_dates) == 0:
        train["is_added_missing_date"] = 0
        return train

    # 예상하지 못한 다른 누락 날짜까지 자동으로 0 처리하지 않도록 방어
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

    # Kaggle 원본 id와 겹치지 않도록 음수 unique id 사용
    missing_rows["id"] = -np.arange(1, len(missing_rows) + 1)

    # 기존 행은 실제 관측값
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
    store_nbr, family를 범주형(category)으로 변환합니다.
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
    - 완전히 동일한 행 제거
    - transferred=True인 원래 휴일 날짜 제거
    - 날짜/지역 기준 정렬

    National / Regional / Local을 실제 store에 매칭하는 작업은
    Feature Engineering 단계에서 수행합니다.
    """
    holidays = holidays.copy()

    holidays = holidays.drop_duplicates()

    holidays = holidays[
        holidays["transferred"] == False
    ].copy()

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
    실제 Kaggle test 기간과 동일하게
    마지막 16일을 validation으로 분리합니다.
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
    sales 원본은 유지하고,
    모델링용 target 후보 sales_log를 추가합니다.
    """
    df = df.copy()
    df["sales_log"] = np.log1p(df["sales"])

    return df


# =========================================================
# 9. 최종 QC
# =========================================================

def print_preprocessing_summary(train, train_df, valid_df, test, holidays):
    """
    전처리 결과를 간단히 확인합니다.
    """
    print("===== DATE RANGE =====")
    print("Train:", train_df["date"].min(), "~", train_df["date"].max())
    print("Valid:", valid_df["date"].min(), "~", valid_df["date"].max())
    print("Test :", test["date"].min(), "~", test["date"].max())

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
    added_count = int(train["is_added_missing_date"].sum())
    print("count:", added_count)

    print("\n===== SALES SKEW =====")
    print("original:", train_df["sales"].skew())
    print("log1p   :", train_df["sales_log"].skew())

    print("\n===== NULL VALUES =====")
    print("train_df nulls:", int(train_df.isna().sum().sum()))
    print("valid_df nulls:", int(valid_df.isna().sum().sum()))
    print("test nulls    :", int(test.isna().sum().sum()))


# =========================================================
# 10. 전체 전처리 실행
# =========================================================

def run_preprocessing(data_dir=None, valid_days=16):
    """
    현재까지 확정한 전처리 전체를 순서대로 실행합니다.

    순서:
    1) 데이터 로드
    2) 날짜형 변환
    3) 누락된 12/25 복원
    4) category 변환
    5) holiday 기본 전처리
    6) train/valid split
    7) sales_log 생성
    8) QC 출력
    """
    train, test, holidays = load_data(data_dir)

    train, test, holidays = convert_date_columns(
        train,
        test,
        holidays
    )

    train = add_missing_christmas_rows(train)

    # test에는 인위적으로 추가한 행이 없으므로 0
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


# =========================================================
# 직접 실행
# =========================================================

if __name__ == "__main__":
    train_df, valid_df, test, holidays = run_preprocessing()

# 결측치
print("=== Missing Values ===")
print(train_df.isnull().sum())

# 중복 행
print("\n=== Duplicates ===")
print("Train duplicates:", train_df.duplicated().sum())
print("Valid duplicates:", valid_df.duplicated().sum())

# 날짜 누락 확인
full_dates = pd.date_range(
    train_df["date"].min(),
    train_df["date"].max(),
    freq="D"
)

missing_dates = full_dates.difference(
    train_df["date"].drop_duplicates()
)

print("\n=== Missing Dates ===")
print(missing_dates)

# 추가한 크리스마스 행 확인
print("\n=== Added Christmas Rows ===")
print(
    train_df[
        train_df["is_added_missing_date"] == 1
    ].shape
)

# 기간 확인
print("\n=== Date Range ===")
print("Train:", train_df["date"].min(), "~", train_df["date"].max())
print("Valid:", valid_df["date"].min(), "~", valid_df["date"].max())
print("Test :", test["date"].min(), "~", test["date"].max())

