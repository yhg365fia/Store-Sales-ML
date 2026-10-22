"""Store Sales 전처리 전체 코드.

기존 Store/Holiday 병합 및 반환 형식을 유지하고 Oil/Transactions를 추가합니다.
- oil_clean: 전체 기간 선형 보간, 첫날 다음 날 값 대체(과거 기록 복원용)
- transactions_filled: Train의 date × store_nbr 기준으로 누락 조합을 복원
  - 해당 매장·날짜의 총 sales가 0이면 transactions=0
  - 총 sales가 양수인데 transactions가 없으면 매장별 시간 선형 보간
  - 선형 보간은 external_cutoff 이전 자료만 사용
- is_imputed/imputation_method는 추적용이며 모델 피처 자동 추가 대상이 아닙니다.
- Oil/Transactions를 store_date_features에 병합한 뒤 Train/Test에 연결합니다.
- external_cutoff 이후 관측은 보간 전에 제외합니다(기본: Train 마지막 날짜).
- 병합 컬럼은 과거 자료 연결용입니다. 미래 거래량은 NaN으로 두며 0 대체하지 않습니다.
- 모델에는 당일 거래량을 직접 넣지 말고 별도 Lag/과거 통계 피처를 사용해야 합니다.
- Fold 실행 시 external_cutoff를 해당 Fold 학습 종료일로 지정해야 합니다.
- 날짜 누락 0개는 모델의 무누수를 의미하지 않습니다.
"""

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
    if "is_added_missing_date" not in train.columns:
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

    unexpected = missing_dates[
        ~((missing_dates.month == 12) & (missing_dates.day == 25))
    ]
    if len(unexpected):
        raise ValueError(f"크리스마스 외 누락 날짜는 별도 확인이 필요합니다: {unexpected.tolist()}")

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
            "store_nbr",
            *EXTERNAL_REFERENCE_COLUMNS,
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
# 12. Oil / Transactions 로드 및 전처리
# ============================================================

def load_external_data(data_dir):
    """기존 load_data()의 4개 반환값은 유지하고 외부 데이터를 별도 로드."""
    data_dir = Path(data_dir)
    oil = pd.read_csv(data_dir / "oil.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv")
    return oil, transactions


def preprocess_oil(oil):
    df = oil[["date", "dcoilwtico"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError("Oil 날짜 결측 또는 중복")
    df = df.sort_values("date").set_index("date").asfreq("D")
    missing = df["dcoilwtico"].isna()
    df["imputation_method"] = np.where(missing, "unresolved", "observed")
    df["dcoilwtico"] = df["dcoilwtico"].interpolate(
        method="time", limit_area="inside"
    )
    df.loc[missing & df["dcoilwtico"].notna(), "imputation_method"] = "time_interpolation"
    if pd.isna(df["dcoilwtico"].iloc[0]) and len(df) > 1:
        next_value = df["dcoilwtico"].iloc[1]
        if pd.notna(next_value):
            df.loc[df.index[0], "dcoilwtico"] = next_value
            df.loc[df.index[0], "imputation_method"] = "next_day_copy"
    df["is_imputed"] = (missing & df["dcoilwtico"].notna()).astype("int8")
    print("날짜 복원 후 보간 전 결측:", int(missing.sum()))
    print("보완 후 결측:", int(df["dcoilwtico"].isna().sum()))
    return df.reset_index()


def preprocess_transactions(
    transactions,
    train,
    store_ids,
    cutoff,
):
    """Train의 date × store_nbr 기준으로 transactions 결측을 보완합니다.

    정책
    ----
    1. cutoff 이후 transactions 관측은 사용하지 않음
    2. Train의 date × store_nbr별 총 sales를 계산
    3. transactions가 없고 해당 매장·날짜의 총 sales가 0이면 0으로 대체
    4. transactions가 없고 총 sales가 양수이면 매장별 시간 선형 보간
    5. 선형 보간 뒤에도 값이 남으면 오류로 중단

    주의
    ----
    transactions 원본값은 모델에 당일 피처로 직접 넣지 않고,
    이후 Lag / 과거 통계용 원천 자료로만 사용합니다.
    """

    cutoff = pd.Timestamp(cutoff)

    # --------------------------------------------------------
    # 1. 원본 Transactions 정리
    # --------------------------------------------------------

    df = transactions[
        [
            "date",
            "store_nbr",
            "transactions",
        ]
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    keys = [
        "date",
        "store_nbr",
    ]

    if df[keys].isna().any().any():
        raise ValueError(
            "Transactions 키에 결측치가 있습니다."
        )

    if df.duplicated(keys).any():
        raise ValueError(
            "Transactions의 date × store_nbr에 중복이 있습니다."
        )

    df["transactions"] = df[
        "transactions"
    ].astype(float)

    if not np.isfinite(
        df["transactions"]
        .dropna()
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Transactions에 비유한 값이 있습니다."
        )

    if (
        df["transactions"]
        .dropna()
        .lt(0)
        .any()
    ):
        raise ValueError(
            "Transactions에 음수 값이 있습니다."
        )

    # cutoff 이후 관측은 보간 전에 제거
    df = df.loc[
        df["date"] <= cutoff
    ].copy()

    if df.empty:
        raise ValueError(
            "cutoff 이전 Transactions 데이터가 없습니다."
        )

    original_count = len(df)

    # --------------------------------------------------------
    # 2. Train의 date × store_nbr별 총 sales 계산
    # --------------------------------------------------------

    train_hist = train.loc[
        train["date"] <= cutoff,
        [
            "date",
            "store_nbr",
            "sales",
        ],
    ].copy()

    if train_hist.empty:
        raise ValueError(
            "cutoff 이전 Train 데이터가 없습니다."
        )

    train_hist["date"] = pd.to_datetime(
        train_hist["date"],
        errors="raise",
    )

    if (
        train_hist["sales"].isna().any()
        or not np.isfinite(
            train_hist["sales"].to_numpy(dtype=float)
        ).all()
        or train_hist["sales"].lt(0).any()
    ):
        raise ValueError(
            "Train sales에 결측/무한대/음수가 있습니다."
        )

    daily_sales = (
        train_hist
        .groupby(
            [
                "date",
                "store_nbr",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            total_sales=(
                "sales",
                "sum",
            )
        )
        .sort_values(
            [
                "store_nbr",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    # store_nbr dtype 차이로 인한 merge 문제 방지
    daily_sales["store_nbr"] = (
        daily_sales["store_nbr"]
        .astype(int)
    )

    df["store_nbr"] = (
        df["store_nbr"]
        .astype(int)
    )

    expected_store_ids = {
        int(x)
        for x in store_ids
    }

    actual_store_ids = set(
        daily_sales[
            "store_nbr"
        ].unique()
    )

    if not actual_store_ids.issubset(
        expected_store_ids
    ):
        raise ValueError(
            "Train에 stores.csv에 없는 store_nbr가 있습니다."
        )

    if daily_sales.duplicated(keys).any():
        raise ValueError(
            "Train의 일별 매장 sales 집계에 중복 키가 있습니다."
        )

    # --------------------------------------------------------
    # 3. Train의 모든 date × store에 Transactions 연결
    # --------------------------------------------------------

    df = daily_sales.merge(
        df,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    df["is_imputed"] = 0

    df["imputation_method"] = np.where(
        df["transactions"].notna(),
        "observed",
        "unresolved",
    )

    missing_before = int(
        df["transactions"]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # 4. sales = 0인 누락 → transactions = 0
    # --------------------------------------------------------

    zero_sales_mask = (
        df["transactions"].isna()
        & df["total_sales"].eq(0)
    )

    df.loc[
        zero_sales_mask,
        "transactions",
    ] = 0.0

    df.loc[
        zero_sales_mask,
        "imputation_method",
    ] = "zero_sales_zero"

    df.loc[
        zero_sales_mask,
        "is_imputed",
    ] = 1

    zero_sales_filled = int(
        zero_sales_mask.sum()
    )

    # --------------------------------------------------------
    # 5. sales > 0인 누락 → 매장별 시간 선형 보간
    # --------------------------------------------------------

    positive_missing_before = (
        df["transactions"].isna()
        & df["total_sales"].gt(0)
    )

    positive_missing_count = int(
        positive_missing_before.sum()
    )

    filled_parts = []

    for store_nbr, store_df in df.groupby(
        "store_nbr",
        observed=True,
        sort=True,
    ):
        store_df = (
            store_df
            .sort_values("date")
            .copy()
        )

        store_df = store_df.set_index(
            "date"
        )

        missing_positive = (
            store_df["transactions"].isna()
            & store_df["total_sales"].gt(0)
        )

        if missing_positive.any():

            interpolated = (
                store_df["transactions"]
                .interpolate(
                    method="time",
                    limit_direction="both",
                )
            )

            fill_mask = (
                missing_positive
                & interpolated.notna()
            )

            store_df.loc[
                fill_mask,
                "transactions",
            ] = interpolated.loc[
                fill_mask
            ]

            store_df.loc[
                fill_mask,
                "imputation_method",
            ] = "time_interpolation"

            store_df.loc[
                fill_mask,
                "is_imputed",
            ] = 1

        store_df = (
            store_df
            .reset_index()
        )

        filled_parts.append(
            store_df
        )

    df = pd.concat(
        filled_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 6. 최종 QC
    # --------------------------------------------------------

    unresolved = df[
        df["transactions"].isna()
    ].copy()

    if not unresolved.empty:
        preview = unresolved[
            [
                "date",
                "store_nbr",
                "total_sales",
            ]
        ].head(20)

        raise ValueError(
            "Transactions 보간 후에도 결측치가 남아 있습니다.\n"
            f"{preview.to_string(index=False)}"
        )

    if (
        df["imputation_method"]
        .eq("unresolved")
        .any()
    ):
        raise ValueError(
            "Transactions에 unresolved 상태가 남아 있습니다."
        )

    if not np.isfinite(
        df["transactions"]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Transactions 보완 후 비유한 값이 있습니다."
        )

    if (
        df["transactions"] < 0
    ).any():
        raise ValueError(
            "Transactions 보완 후 음수 값이 있습니다."
        )

    df["is_imputed"] = (
        df["is_imputed"]
        .astype("int8")
    )

    df = (
        df
        .sort_values(keys)
        .reset_index(drop=True)
    )

    if df.duplicated(keys).any():
        raise ValueError(
            "Transactions 보완 후 date × store_nbr 중복이 있습니다."
        )

    interpolated_count = int(
        df["imputation_method"]
        .eq("time_interpolation")
        .sum()
    )

    remaining_missing = int(
        df["transactions"]
        .isna()
        .sum()
    )

    print(
        "Transactions 원본 행 수:",
        original_count,
    )

    print(
        "Train date × store 조합 수:",
        len(df),
    )

    print(
        "병합 직후 Transactions 결측:",
        missing_before,
    )

    print(
        "sales=0 → 0 대체:",
        zero_sales_filled,
    )

    print(
        "sales>0 결측:",
        positive_missing_count,
    )

    print(
        "시간 선형 보간:",
        interpolated_count,
    )

    print(
        "보완 후 결측:",
        remaining_missing,
    )

    print(
        "\nTransactions 보완 방식:"
    )

    print(
        df[
            "imputation_method"
        ]
        .value_counts()
        .to_string()
    )

    # total_sales는 결측 판단용 보조 컬럼이므로 반환하지 않음
    return df[
        [
            "date",
            "store_nbr",
            "transactions",
            "is_imputed",
            "imputation_method",
        ]
    ].copy()


# 전체 기간 보간본 oil_clean과 별도로 사용하는 예측 시점용 함수.
# 원본 oil을 전달해야 하며, 이미 보간한 oil_clean을 전달하면 안 됩니다.
# 이 함수는 ffill 방식이며 내부 구간 선형 보간과는 다른 대체 정책입니다.
def make_oil_asof(oil, cutoff, end_date):
    cutoff, end_date = pd.Timestamp(cutoff), pd.Timestamp(end_date)
    if end_date < cutoff:
        raise ValueError("end_date는 cutoff 이후여야 합니다.")
    raw = oil[["date", "dcoilwtico"]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    if raw["date"].isna().any() or raw["date"].duplicated().any():
        raise ValueError("Oil 날짜 결측 또는 중복")
    raw = raw.loc[raw["date"] <= cutoff].sort_values("date")
    if raw.empty:
        raise ValueError("cutoff 이전 데이터가 없습니다.")
    raw = raw.set_index("date")
    calendar = pd.date_range(raw.index.min(), end_date, freq="D", name="date")
    daily = raw.reindex(calendar)
    source_dates = pd.Series(daily.index, index=daily.index).where(
        daily["dcoilwtico"].notna()
    )
    daily["source_date"] = source_dates.ffill()
    daily["dcoilwtico"] = daily["dcoilwtico"].ffill()
    assert daily["source_date"].dropna().le(cutoff).all()
    return daily.reset_index()


def check_external_data(oil_clean, transactions_filled):
    """형식/누락 QC. 모델 피처 전체의 누수 검사는 아님."""
    for name, df, keys, value in [
        ("Oil", oil_clean, ["date"], "dcoilwtico"),
        ("Transactions", transactions_filled, ["date", "store_nbr"], "transactions"),
    ]:
        if df.empty:
            raise ValueError(f"{name}: 빈 데이터")
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise ValueError(f"{name}: 날짜형 오류")
        if df[keys].isna().any().any():
            raise ValueError(f"{name}: 키 결측")
        if name == "Transactions" and df[value].isna().any():
            raise ValueError(f"{name}: 대체 후 결측치가 남아 있습니다.")
        if df.duplicated(keys).any():
            raise ValueError(f"{name}: 키 중복")
        if not np.isfinite(df[value].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: 비유한 수치")
        if (df[value] < 0).any():
            raise ValueError(f"{name}: 이 대회 기간에 예상하지 않은 음수 값")
        if not df[keys].reset_index(drop=True).equals(
            df.sort_values(keys)[keys].reset_index(drop=True)
        ):
            raise ValueError(f"{name}: 정렬 오류")
        calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        missing_dates = calendar.difference(pd.DatetimeIndex(df["date"].unique()))
        if len(missing_dates):
            raise ValueError(f"{name}: 전체 기록이 없는 날짜 {missing_dates.tolist()}")
        if name == "Transactions" and df["imputation_method"].eq("unresolved").any():
            raise ValueError(f"{name}: 미해결 대체값")
        print(f"{name}: QC OK | {len(df):,}행 | 보완값 {int(df['is_imputed'].sum()):,}개")
    # 모든 날짜가 존재해도 날짜마다 54개 매장 기록이 있는 것은 아닙니다.


# 과거 관측/보완 자료 컬럼: 모델 입력을 자동 선택할 때 직접 포함하지 마세요.
# 이후 Feature Engineering에서 Lag/과거 통계를 만든 뒤 아래 컬럼은 제외합니다.
EXTERNAL_REFERENCE_COLUMNS = [
    "dcoilwtico", "oil_is_imputed", "oil_imputation_method",
    "transactions", "transactions_is_imputed", "transactions_imputation_method",
]


def merge_external_to_calendar(store_date_features, oil_clean, transactions_filled,
                               external_cutoff):
    """날짜×매장 기준 테이블에 외부 자료를 left merge. 미래/미관측값은 NaN 유지."""
    cutoff = pd.Timestamp(external_cutoff)
    oil_table = oil_clean.rename(columns={
        "is_imputed": "oil_is_imputed",
        "imputation_method": "oil_imputation_method",
    })
    transaction_table = transactions_filled.rename(columns={
        "is_imputed": "transactions_is_imputed",
        "imputation_method": "transactions_imputation_method",
    })
    if set(EXTERNAL_REFERENCE_COLUMNS) & set(store_date_features.columns):
        raise ValueError("캘린더에 외부 데이터 컬럼이 이미 있습니다.")
    if oil_table["date"].gt(cutoff).any() or transaction_table["date"].gt(cutoff).any():
        raise ValueError("외부 데이터에 cutoff 이후 날짜가 포함되어 있습니다.")
    before = len(store_date_features)
    result = store_date_features.merge(
        oil_table[["date", "dcoilwtico", "oil_is_imputed", "oil_imputation_method"]],
        on="date", how="left", validate="many_to_one",
    )
    result = result.merge(
        transaction_table[["date", "store_nbr", "transactions",
                           "transactions_is_imputed", "transactions_imputation_method"]],
        on=["date", "store_nbr"], how="left", validate="one_to_one",
    )
    if len(result) != before or result.duplicated(["date", "store_nbr"]).any():
        raise ValueError("외부 데이터 병합 시 행 수 또는 키 중복 오류")
    if result.loc[result["date"] > cutoff, ["dcoilwtico", "transactions"]].notna().any().any():
        raise ValueError("예측 시작 이후의 외부 관측값이 연결되었습니다.")
    return result.sort_values(["date", "store_nbr"]).reset_index(drop=True)


# ============================================================
# 13. 전체 전처리 실행
# ============================================================

def run_preprocessing(
    data_dir="data",
    return_external=False,
    external_cutoff=None,
):
    """기존 3개 반환값을 유지하며 외부 자료를 캘린더를 통해 연결합니다.

    return_external=True이면 oil_clean, transactions_filled까지 5개 반환.
    external_cutoff=None이면 Train 마지막 날짜까지 사용.
    Fold별 호출 시 해당 학습 종료일을 external_cutoff로 지정해야 합니다.
    cutoff 이후의 행도 유지하지만 Oil/거래량은 NaN입니다.
    여기서는 Lag 생성, 모델 피처 선택, Train/Validation 분할을 하지 않습니다.
    """
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

    # Oil/Transactions: cutoff 이후 관측을 먼저 제거한 후 보완합니다.
    cutoff = train["date"].max() if external_cutoff is None else pd.Timestamp(external_cutoff)
    if pd.isna(cutoff) or cutoff < train["date"].min() or cutoff > train["date"].max():
        raise ValueError("external_cutoff는 Train 기간 안의 날짜여야 합니다.")
    oil, transactions = load_external_data(data_dir)
    oil["date"] = pd.to_datetime(oil["date"], errors="raise")
    transactions["date"] = pd.to_datetime(transactions["date"], errors="raise")
    oil = oil.loc[oil["date"] <= cutoff].copy()
    transactions = transactions.loc[transactions["date"] <= cutoff].copy()
    if oil.empty or transactions.empty:
        raise ValueError("cutoff 이전 외부 데이터가 없습니다.")
    oil_clean = preprocess_oil(oil)
    transactions_filled = preprocess_transactions(
        transactions=transactions,
        train=train,
        store_ids=stores["store_nbr"].dropna().unique(),
        cutoff=cutoff,
    )
    check_external_data(oil_clean, transactions_filled)
    store_date_features = merge_external_to_calendar(
        store_date_features, oil_clean, transactions_filled, cutoff
    )
    print("External cutoff:", cutoff.date())

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

    if return_external:
        return train, test, holidays, oil_clean, transactions_filled

    return (
        train,
        test,
        holidays
    )

# ============================================================
# 직접 실행 확인
# ============================================================

if __name__ == "__main__":
    # src/preprocessing.py 기준 프로젝트의 data 폴더
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_df, test, holidays, oil_clean, transactions_filled = run_preprocessing(
        data_dir=data_dir,
        return_external=True,
    )
    for name, df in [
        ("Train", train_df), ("Test", test), ("Holidays", holidays),
        ("Oil", oil_clean), ("Transactions", transactions_filled),
    ]:
        print(name, df.shape)
        print("기간:", df["date"].min(), "~", df["date"].max())
    print("\n연결된 외부 자료 결측 수 (미래/미관측 구간은 NaN 유지):")
    print("Train:", train_df[["dcoilwtico", "transactions"]].isna().sum())
    print("Test:", test[["dcoilwtico", "transactions"]].isna().sum())
    print("\nTransactions 보완 방식별 행 수:")
    print(
        transactions_filled["imputation_method"]
        .value_counts()
        .to_string()
    )
