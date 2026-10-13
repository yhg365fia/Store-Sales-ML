from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_data(data_dir):
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    holidays = pd.read_csv(data_dir / "holidays_events.csv")
    stores = pd.read_csv(data_dir / "stores.csv")

    return train, test, holidays, stores


# ============================================================
# 2. 날짜형 변환
# ============================================================

def convert_dates(train, test, holidays):
    train = train.copy()
    test = test.copy()
    holidays = holidays.copy()

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    return train, test, holidays


# ============================================================
# 3. Train 누락 날짜 복원
# ============================================================

def restore_missing_dates(train):
    train = train.copy()

    # 기존 행 표시
    train["is_added_missing_date"] = 0

    # 전체 날짜 범위
    full_dates = pd.date_range(
        train["date"].min(),
        train["date"].max(),
        freq="D"
    )

    # 실제로 빠진 날짜
    missing_dates = full_dates.difference(
        train["date"].drop_duplicates()
    )

    if len(missing_dates) == 0:
        return train

    # 실제 존재하는 store × family 조합
    store_family = (
        train[
            [
                "store_nbr",
                "family"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 누락 날짜 × store × family
    missing_rows = (
        pd.DataFrame({
            "date": missing_dates
        })
        .merge(
            store_family,
            how="cross"
        )
    )

    # 휴점일로 가정
    missing_rows["sales"] = 0.0
    missing_rows["onpromotion"] = 0
    missing_rows["is_added_missing_date"] = 1

    # 원본 ID가 없는 인위적 행이므로 음수 ID 부여
    missing_rows["id"] = -np.arange(
        1,
        len(missing_rows) + 1
    )

    # 기존 Train과 컬럼 순서 맞추기
    missing_rows = missing_rows[train.columns]

    train = pd.concat(
        [
            train,
            missing_rows
        ],
        ignore_index=True
    )

    train = (
        train
        .sort_values(
            [
                "date",
                "store_nbr",
                "family"
            ]
        )
        .reset_index(drop=True)
    )

    return train


# ============================================================
# 4. Train / Test 기본 범주형 처리
# ============================================================

def preprocess_sales_categories(train, test):
    train = train.copy()
    test = test.copy()

    # Test에는 인위적으로 복원한 날짜 없음
    test["is_added_missing_date"] = 0

    for df in (train, test):
        df["store_nbr"] = df["store_nbr"].astype("category")
        df["family"] = df["family"].astype("category")

    return train, test


# ============================================================
# 5. stores 전처리
# ============================================================

def preprocess_stores(stores):
    stores = stores.copy()

    categorical_cols = [
        "store_nbr",
        "city",
        "state",
        "type",
        "cluster",
    ]

    for col in categorical_cols:
        stores[col] = stores[col].astype("category")

    # store_nbr는 매장별 한 행
    assert stores["store_nbr"].is_unique, (
        "stores의 store_nbr에 중복이 있습니다."
    )

    return stores


# ============================================================
# 6. Holiday event_group 생성
# ============================================================

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
        "Good Friday": r"viernes santo",
        "Labor Day": r"dia del trabajo",
    }

    for group, pattern in event_patterns.items():
        mask = description.str.contains(
            pattern,
            regex=True,
            na=False
        )

        holidays.loc[
            mask,
            "event_group"
        ] = group

    return holidays


# ============================================================
# 7. holidays 기본 전처리
# ============================================================

def preprocess_holidays(holidays):
    holidays = holidays.copy()

    # 완전히 동일한 행만 제거
    holidays = holidays.drop_duplicates()

    # 원래 날짜에서 실제 휴일로 적용되지 않는 행 제거
    holidays = holidays[
        holidays["transferred"] == False
    ].copy()

    # event_group 생성
    holidays = add_event_group(holidays)

    # 범주형 처리
    categorical_cols = [
        "type",
        "locale",
        "locale_name",
        "description",
        "event_group",
    ]

    for col in categorical_cols:
        holidays[col] = holidays[col].astype("category")

    holidays = (
        holidays
        .sort_values(
            [
                "date",
                "locale",
                "locale_name"
            ]
        )
        .reset_index(drop=True)
    )

    return holidays


# ============================================================
# 8. Holiday → 실제 적용 Store 연결
# ============================================================

def build_store_holidays(holidays, stores):

    # --------------------------------------------------------
    # Local Holiday
    # locale_name ↔ city
    # --------------------------------------------------------

    local = (
        holidays[
            holidays["locale"] == "Local"
        ]
        .merge(
            stores[
                [
                    "store_nbr",
                    "city"
                ]
            ],
            left_on="locale_name",
            right_on="city",
            how="inner"
        )
    )

    # --------------------------------------------------------
    # Regional Holiday
    # locale_name ↔ state
    # --------------------------------------------------------

    regional = (
        holidays[
            holidays["locale"] == "Regional"
        ]
        .merge(
            stores[
                [
                    "store_nbr",
                    "state"
                ]
            ],
            left_on="locale_name",
            right_on="state",
            how="inner"
        )
    )

    # --------------------------------------------------------
    # National Holiday
    # 모든 Store에 적용
    # --------------------------------------------------------

    national = (
        holidays[
            holidays["locale"] == "National"
        ]
        .merge(
            stores[["store_nbr"]],
            how="cross"
        )
    )

    # 필요한 컬럼만 유지
    holiday_cols = [
        "date",
        "store_nbr",
        "type",
        "event_group",
    ]

    local = local[holiday_cols]
    regional = regional[holiday_cols]
    national = national[holiday_cols]

    # --------------------------------------------------------
    # National / Regional / Local 합치기
    # --------------------------------------------------------

    store_holidays = pd.concat(
        [
            national,
            regional,
            local,
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # event_group → Binary Feature
    # --------------------------------------------------------

    event_names = {
        "Christmas": "event_christmas",
        "New Year": "event_new_year",
        "Carnival": "event_carnival",
        "Mothers Day": "event_mothers_day",
        "Independence Day": "event_independence_day",
        "City Foundation": "event_city_foundation",
        "Good Friday": "event_good_friday",
        "Labor Day": "event_labor_day",
        "Other": "event_other",
    }

    for group, col in event_names.items():
        store_holidays[col] = (
            store_holidays["event_group"] == group
        ).astype(int)

    event_cols = list(event_names.values())

    # --------------------------------------------------------
    # 같은 date + store의 여러 이벤트 집계
    # --------------------------------------------------------

    store_holidays = (
        store_holidays
        .groupby(
            [
                "date",
                "store_nbr"
            ],
            as_index=False,
            observed=True
        )
        .agg(
            holiday_count=("type", "size"),
            **{
                col: (col, "max")
                for col in event_cols
            }
        )
        .sort_values(
            [
                "date",
                "store_nbr"
            ]
        )
        .reset_index(drop=True)
    )

    assert not store_holidays.duplicated(
        [
            "date",
            "store_nbr"
        ]
    ).any(), (
        "store_holidays에 date + store_nbr 중복이 있습니다."
    )

    return store_holidays


# ============================================================
# 9. date × store_nbr 단위 Feature Table 생성
# ============================================================

def build_store_date_features(
    train,
    test,
    stores,
    store_holidays
):
    # --------------------------------------------------------
    # Train/Test에 실제 존재하는 date × store 추출
    # --------------------------------------------------------

    store_date_features = (
        pd.concat(
            [
                train[
                    [
                        "date",
                        "store_nbr"
                    ]
                ],
                test[
                    [
                        "date",
                        "store_nbr"
                    ]
                ]
            ],
            ignore_index=True
        )
        .drop_duplicates()
        .sort_values(
            [
                "date",
                "store_nbr"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Store 정보 병합
    # --------------------------------------------------------

    store_date_features = store_date_features.merge(
        stores,
        on="store_nbr",
        how="left",
        validate="many_to_one"
    )

    store_cols = [
        "city",
        "state",
        "type",
        "cluster",
    ]

    assert (
        store_date_features[store_cols]
        .isnull()
        .sum()
        .sum()
        == 0
    ), "Store 정보 병합 후 결측치가 있습니다."

    # --------------------------------------------------------
    # Holiday 정보 병합
    # --------------------------------------------------------

    store_date_features = store_date_features.merge(
        store_holidays,
        on=[
            "date",
            "store_nbr"
        ],
        how="left",
        validate="one_to_one"
    )

    event_cols = [
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

    holiday_cols = [
        "holiday_count",
        *event_cols
    ]

    # Holiday 관련 컬럼은 범주형이 아니라
    # count / 0·1 숫자형이므로 NaN → 0
    store_date_features[holiday_cols] = (
        store_date_features[holiday_cols]
        .fillna(0)
        .astype(int)
    )

    # 최종 범주형 보장
    categorical_cols = [
        "store_nbr",
        "city",
        "state",
        "type",
        "cluster",
    ]

    for col in categorical_cols:
        store_date_features[col] = (
            store_date_features[col]
            .astype("category")
        )

    # date × store_nbr는 한 행이어야 함
    assert not store_date_features.duplicated(
        [
            "date",
            "store_nbr"
        ]
    ).any(), (
        "store_date_features에 중복 키가 있습니다."
    )

    return store_date_features


# ============================================================
# 10. Train / Test에 store_date_features 병합
# ============================================================

def merge_store_date_features(
    train,
    test,
    store_date_features
):
    train = train.copy()
    test = test.copy()

    train_rows = len(train)
    test_rows = len(test)

    # Train 병합
    train = train.merge(
        store_date_features,
        on=[
            "date",
            "store_nbr"
        ],
        how="left",
        validate="many_to_one"
    )

    # Test 병합
    test = test.merge(
        store_date_features,
        on=[
            "date",
            "store_nbr"
        ],
        how="left",
        validate="many_to_one"
    )

    # 행 수 유지 확인
    assert len(train) == train_rows, (
        "Train 병합 과정에서 행 수가 변경되었습니다."
    )

    assert len(test) == test_rows, (
        "Test 병합 과정에서 행 수가 변경되었습니다."
    )

    # 병합된 Feature 결측 확인
    feature_cols = [
        col
        for col in store_date_features.columns
        if col not in [
            "date",
            "store_nbr"
        ]
    ]

    assert (
        train[feature_cols]
        .isnull()
        .sum()
        .sum()
        == 0
    ), "Train 병합 후 결측치가 있습니다."

    assert (
        test[feature_cols]
        .isnull()
        .sum()
        .sum()
        == 0
    ), "Test 병합 후 결측치가 있습니다."

    # 범주형 보장
    categorical_cols = [
        "store_nbr",
        "family",
        "city",
        "state",
        "type",
        "cluster",
    ]

    for df in (train, test):
        for col in categorical_cols:
            df[col] = df[col].astype("category")

    return train, test


# ============================================================
# 11. Target 로그 변환
# ============================================================

def add_target_log(train):
    train = train.copy()

    train["sales_log"] = np.log1p(
        train["sales"]
    )

    return train


# ============================================================
# 13. 전체 전처리 실행
# ============================================================

def run_preprocessing(
    data_dir="data"
):
    # --------------------------------------------------------
    # 1. 데이터 로드
    # --------------------------------------------------------

    train, test, holidays, stores = load_data(
        data_dir
    )

    # --------------------------------------------------------
    # 2. 날짜형 변환
    # --------------------------------------------------------

    train, test, holidays = convert_dates(
        train,
        test,
        holidays
    )

    # --------------------------------------------------------
    # 3. 누락 Christmas 날짜 복원
    # --------------------------------------------------------

    train = restore_missing_dates(train)

    # --------------------------------------------------------
    # 4. Train / Test 범주형 처리
    # --------------------------------------------------------

    train, test = preprocess_sales_categories(
        train,
        test
    )

    # --------------------------------------------------------
    # 5. Stores 전처리
    # --------------------------------------------------------

    stores = preprocess_stores(stores)

    # --------------------------------------------------------
    # 6. Holidays 전처리
    # --------------------------------------------------------

    holidays = preprocess_holidays(holidays)

    # --------------------------------------------------------
    # 7. Holiday → 적용 Store 연결
    # --------------------------------------------------------

    store_holidays = build_store_holidays(
        holidays,
        stores
    )

    # --------------------------------------------------------
    # 8. date × store_nbr Feature Table 생성
    # --------------------------------------------------------

    store_date_features = build_store_date_features(
        train,
        test,
        stores,
        store_holidays
    )

    # --------------------------------------------------------
    # 9. Train / Test에 한 번씩 병합
    # --------------------------------------------------------

    train, test = merge_store_date_features(
        train,
        test,
        store_date_features
    )

    # --------------------------------------------------------
    # 10. Target 로그 변환
    # --------------------------------------------------------

    train = add_target_log(train)

    return (
        train,
        test,
        holidays
    )

# ============================================================
# 직접 실행 확인
# ============================================================

if __name__ == "__main__":
    train_df, test, holidays = (
        run_preprocessing()
    )

    print("Train:", train_df.shape)
    print("Test :", test.shape)

    print()
    print(
        "Train:",
        train_df["date"].min(),
        "~",
        train_df["date"].max()
    )

    print(
        "Test :",
        test["date"].min(),
        "~",
        test["date"].max()
    )