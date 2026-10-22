import numpy as np
import pandas as pd


# =========================================================
# 1. 날짜 피처
# =========================================================

def add_date_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofweek"] = df["date"].dt.dayofweek

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    angle = 2 * np.pi * df["dayofyear"] / 365.25
    df["dayofyear_sin"] = np.sin(angle)
    df["dayofyear_cos"] = np.cos(angle)

    df["is_15th"] = (
        df["date"].dt.day == 15
    ).astype(int)

    df["is_month_end"] = (
        df["date"].dt.is_month_end
    ).astype(int)

    return df


# =========================================================
# 2. 프로모션 피처
# =========================================================

def add_promotion_features(df):
    df = df.copy()
    df["is_promotion"] = (
        df["onpromotion"] > 0
    ).astype(int)
    return df


# =========================================================
# 3. Lag
# =========================================================

LAGS = (16, 21, 28)

LAG_FEATURES = [
    f"sales_lag_{lag}" for lag in LAGS
]


def add_lag_features(df, lags=LAGS):
    """
    날짜가 연속적이라는 전제에서 매장 × 상품군별 Lag 생성.
    """
    if any(lag < 16 for lag in lags):
        raise ValueError(
            "16일 일괄 예측에서는 lag를 16 이상으로 설정하세요."
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "family", "date"])

    grouped = df.groupby(
        ["store_nbr", "family"],
        observed=True,
    )["sales"]

    for lag in lags:
        df[f"sales_lag_{lag}"] = grouped.shift(lag)

    return df


# =========================================================
# 4. Rolling mean — 보존, 이번 모델에서는 미사용
# =========================================================

ROLLING_SHIFT = 16
ROLLING_WINDOWS = (7, 14, 28)

ROLLING_MEAN_FEATURES = [
    f"sales_rolling_mean_{ROLLING_SHIFT}_{window}"
    for window in ROLLING_WINDOWS
]

ROLLING_FEATURES = ROLLING_MEAN_FEATURES.copy()

ROLLING_WINDOWS_BY_STAT = {
    "mean": ROLLING_WINDOWS,
}


def add_rolling_features(
    df,
    shift=ROLLING_SHIFT,
    windows=ROLLING_WINDOWS,
):
    if shift < 16:
        raise ValueError("shift는 16 이상이어야 합니다.")

    if any(window < 1 for window in windows):
        raise ValueError("window는 1 이상이어야 합니다.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "family", "date"])

    grouped = df.groupby(
        ["store_nbr", "family"],
        observed=True,
    )["sales"]

    for window in windows:
        df[f"sales_rolling_mean_{shift}_{window}"] = (
            grouped.transform(
                lambda x: (
                    x.shift(shift)
                    .rolling(window=window, min_periods=window)
                    .mean()
                )
            )
        )

    return df


# =========================================================
# 5. EMA — 보존, 이번 모델에서는 미사용
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
    if shift < 16:
        raise ValueError("shift는 16 이상이어야 합니다.")

    if any(span < 1 for span in spans):
        raise ValueError("span은 1 이상이어야 합니다.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "family", "date"])

    grouped = df.groupby(
        ["store_nbr", "family"],
        observed=True,
    )["sales"]

    for span in spans:
        df[f"sales_ema_{shift}_{span}"] = (
            grouped.transform(
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


TIME_SERIES_FEATURES = (
    LAG_FEATURES + ROLLING_FEATURES + EMA_FEATURES
)

# =========================================================
# 6. Transactions 외부 시계열 피처
# =========================================================

TRANSACTION_LAGS = (16, 21, 28)

TRANSACTION_LAG_FEATURES = [
    f"transactions_lag_{lag}"
    for lag in TRANSACTION_LAGS
]

TRANSACTION_ROLLING_SHIFT = 16
TRANSACTION_ROLLING_WINDOWS = (7, 14, 28)

TRANSACTION_ROLLING_FEATURES = [
    f"transactions_rolling_mean_{TRANSACTION_ROLLING_SHIFT}_{window}"
    for window in TRANSACTION_ROLLING_WINDOWS
]


def _store_date_reference(df, value_col):
    """
    date × store_nbr마다 하나의 값만 있어야 하는 외부 변수를
    매장별 일별 시계열로 축약합니다.

    train에는 family별 행이 반복되므로 transactions도 같은 날짜/매장에
    여러 번 반복되어 있습니다. 그대로 shift하면 '행 기준 shift'가 되므로
    반드시 date × store_nbr 한 행으로 축약한 뒤 시계열 피처를 만듭니다.
    """
    required = ["date", "store_nbr", value_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{value_col} 시계열 생성에 필요한 컬럼이 없습니다: {missing}"
        )

    work = df[required].copy()
    work["date"] = pd.to_datetime(work["date"])

    if work[["date", "store_nbr"]].isna().any().any():
        raise ValueError(
            f"{value_col}: date 또는 store_nbr에 결측치가 있습니다."
        )

    values = pd.to_numeric(
        work[value_col],
        errors="coerce",
    )

    if values.isna().any():
        raise ValueError(
            f"{value_col}: 원본 값에 결측치가 있습니다."
        )

    if not np.isfinite(
        values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{value_col}: 무한대 등 유효하지 않은 값이 있습니다."
        )

    if (values < 0).any():
        raise ValueError(
            f"{value_col}: 음수 값이 있습니다."
        )

    work[value_col] = values

    # 같은 date × store에서 family마다 값이 달라지는지 확인
    per_key_nunique = (
        work.groupby(
            ["date", "store_nbr"],
            observed=True,
        )[value_col]
        .nunique(dropna=False)
    )

    if per_key_nunique.gt(1).any():
        raise ValueError(
            f"{value_col}: 같은 date × store_nbr에서 값이 서로 다릅니다."
        )

    daily = (
        work.drop_duplicates(
            ["date", "store_nbr"]
        )
        .sort_values(
            ["store_nbr", "date"]
        )
        .reset_index(drop=True)
    )

    # shift(16)이 실제 16일 전을 뜻하도록 날짜 연속성 확인
    date_diff = (
        daily.groupby(
            "store_nbr",
            observed=True,
        )["date"]
        .diff()
    )

    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(
            pd.Timedelta(days=1)
        )
    )

    if invalid_gap.any():
        raise ValueError(
            f"{value_col}: 매장별 날짜가 연속적이지 않습니다."
        )

    return daily


def _merge_store_date_features(df, daily, feature_cols):
    """
    date × store_nbr 단위로 만든 외부 시계열 피처를
    원래 store × family × date 행에 다시 연결합니다.
    """
    result = df.copy()
    result["_external_row_order"] = np.arange(
        len(result)
    )

    lookup = daily[
        [
            "date",
            "store_nbr",
            *feature_cols,
        ]
    ].copy()

    if lookup.duplicated(
        ["date", "store_nbr"]
    ).any():
        raise ValueError(
            "외부 시계열 lookup에 date × store_nbr 중복이 있습니다."
        )

    before = len(result)

    result = result.merge(
        lookup,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(result) != before:
        raise ValueError(
            "외부 시계열 피처 병합 후 행 수가 변경되었습니다."
        )

    result = (
        result.sort_values(
            "_external_row_order"
        )
        .drop(columns="_external_row_order")
        .reset_index(drop=True)
    )

    return result


def add_transaction_lag_features(
    df,
    lags=TRANSACTION_LAGS,
):
    """
    매장별 transactions Lag.

    16일 일괄 예측이므로 모든 lag는 16 이상이어야 합니다.
    """
    if any(lag < 16 for lag in lags):
        raise ValueError(
            "Transactions Lag는 16 이상이어야 합니다."
        )

    daily = _store_date_reference(
        df,
        "transactions",
    )

    grouped = daily.groupby(
        "store_nbr",
        observed=True,
    )["transactions"]

    feature_cols = []

    for lag in lags:
        feature = f"transactions_lag_{lag}"
        daily[feature] = grouped.shift(lag)
        feature_cols.append(feature)

    return _merge_store_date_features(
        df,
        daily,
        feature_cols,
    )


def add_transaction_rolling_features(
    df,
    shift=TRANSACTION_ROLLING_SHIFT,
    windows=TRANSACTION_ROLLING_WINDOWS,
):
    """
    매장별 transactions 과거 Rolling Mean.

    예측일 t에서 사용할 수 있도록 transactions를 먼저 16일 shift한 뒤,
    7/14/28일 평균을 계산합니다.
    """
    if shift < 16:
        raise ValueError(
            "Transactions Rolling shift는 16 이상이어야 합니다."
        )

    if any(window < 1 for window in windows):
        raise ValueError(
            "Transactions Rolling window는 1 이상이어야 합니다."
        )

    daily = _store_date_reference(
        df,
        "transactions",
    )

    grouped = daily.groupby(
        "store_nbr",
        observed=True,
    )["transactions"]

    feature_cols = []

    for window in windows:
        feature = (
            f"transactions_rolling_mean_{shift}_{window}"
        )

        daily[feature] = grouped.transform(
            lambda x: (
                x.shift(shift)
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
            )
        )

        feature_cols.append(feature)

    return _merge_store_date_features(
        df,
        daily,
        feature_cols,
    )


# =========================================================
# 7. Oil 외부 시계열 피처
# =========================================================

OIL_RAW_FEATURES = [
    "dcoilwtico",
]

OIL_ROLLING_WINDOW = 7

OIL_ROLLING_FEATURES = [
    f"oil_rolling_mean_{OIL_ROLLING_WINDOW}",
]

OIL_DIFF_PERIOD = 7

OIL_DIFF_FEATURES = [
    f"oil_diff_{OIL_DIFF_PERIOD}",
]


def _oil_daily_reference(df):
    """
    dcoilwtico는 날짜별 전국 공통 값이므로 date당 한 행으로 축약합니다.
    같은 날짜의 모든 매장/상품군에서 값이 동일한지도 확인합니다.
    """
    required = [
        "date",
        "dcoilwtico",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Oil 피처 생성에 필요한 컬럼이 없습니다: {missing}"
        )

    work = df[required].copy()
    work["date"] = pd.to_datetime(
        work["date"]
    )

    oil = pd.to_numeric(
        work["dcoilwtico"],
        errors="coerce",
    )

    if oil.isna().any():
        raise ValueError(
            "dcoilwtico에 결측치가 있습니다."
        )

    if not np.isfinite(
        oil.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "dcoilwtico에 무한대 등 유효하지 않은 값이 있습니다."
        )

    if (oil < 0).any():
        raise ValueError(
            "dcoilwtico에 음수 값이 있습니다."
        )

    work["dcoilwtico"] = oil

    per_date_nunique = (
        work.groupby(
            "date",
            observed=True,
        )["dcoilwtico"]
        .nunique(dropna=False)
    )

    if per_date_nunique.gt(1).any():
        raise ValueError(
            "같은 날짜에서 dcoilwtico 값이 서로 다릅니다."
        )

    daily = (
        work.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    date_diff = daily["date"].diff()

    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(
            pd.Timedelta(days=1)
        )
    )

    if invalid_gap.any():
        raise ValueError(
            "Oil 날짜가 연속적이지 않습니다."
        )

    return daily


def _merge_date_features(
    df,
    daily,
    feature_cols,
):
    """
    날짜 단위 Oil 파생 피처를 모든 store × family 행에 연결합니다.
    """
    result = df.copy()
    result["_external_row_order"] = np.arange(
        len(result)
    )

    lookup = daily[
        [
            "date",
            *feature_cols,
        ]
    ].copy()

    if lookup["date"].duplicated().any():
        raise ValueError(
            "Oil lookup 날짜가 중복됩니다."
        )

    before = len(result)

    result = result.merge(
        lookup,
        on="date",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(result) != before:
        raise ValueError(
            "Oil 피처 병합 후 행 수가 변경되었습니다."
        )

    result = (
        result.sort_values(
            "_external_row_order"
        )
        .drop(columns="_external_row_order")
        .reset_index(drop=True)
    )

    return result


def add_oil_rolling_features(
    df,
    window=OIL_ROLLING_WINDOW,
):
    """
    당일을 포함한 최근 7일 dcoilwtico 평균.

    Oil은 예측 기간에도 제공되는 외생 변수로 사용하는 실험이므로,
    Transactions처럼 16일 shift하지 않습니다.
    """
    if window < 1:
        raise ValueError(
            "Oil Rolling window는 1 이상이어야 합니다."
        )

    daily = _oil_daily_reference(
        df
    )

    feature = (
        f"oil_rolling_mean_{window}"
    )

    daily[feature] = (
        daily["dcoilwtico"]
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )

    return _merge_date_features(
        df,
        daily,
        [feature],
    )


def add_oil_diff_features(
    df,
    period=OIL_DIFF_PERIOD,
):
    """
    현재 유가 - 7일 전 유가.
    Oil은 예측 날짜의 값이 제공된다는 전제의 외생 변수 실험입니다.
    """
    if period < 1:
        raise ValueError(
            "Oil diff period는 1 이상이어야 합니다."
        )

    daily = _oil_daily_reference(
        df
    )

    feature = f"oil_diff_{period}"

    daily[feature] = (
        daily["dcoilwtico"]
        - daily["dcoilwtico"].shift(period)
    )

    return _merge_date_features(
        df,
        daily,
        [feature],
    )


EXTERNAL_EXPERIMENT_FEATURES = (
    TRANSACTION_LAG_FEATURES
    + TRANSACTION_ROLLING_FEATURES
    + OIL_RAW_FEATURES
    + OIL_ROLLING_FEATURES
    + OIL_DIFF_FEATURES
)




# =========================================================
# 8. Group Statistics — A / B / C / D / E 목록
# =========================================================

GROUP_STAT_FEATURES_A = [
    "family_dayofweek_mean",
]

GROUP_STAT_FEATURES_B = [
    "store_family_dayofweek_mean",
]

GROUP_STAT_FEATURES_C = [
    "store_family_mean",
]

GROUP_STAT_FEATURES_D = [
    "family_holiday_mean",
]

GROUP_STAT_FEATURES_E = [
    "family_event_group_mean",
]

# 이전 A 코드와의 import 호환용
GROUP_STAT_FEATURES = GROUP_STAT_FEATURES_A.copy()


# =========================================================
# 9. 과거 누적 평균 공통 함수
# =========================================================

def _add_group_past_mean(
    train_df,
    valid_df,
    group_cols,
    feature_name,
):
    """
    Train:
        각 행의 전날까지의 같은 그룹 sales 평균.

    Validation:
        Fold Train 종료일까지 계산한 같은 그룹 평균 고정.

    평균 대상:
        로그 변환 전 원본 sales.

    과거 기록이 없는 그룹:
        NaN 유지. LightGBM이 처리.

    Validation sales는 통계 계산에 사용하지 않음.
    """
    if train_df.empty or valid_df.empty:
        raise ValueError("Train 또는 Validation이 비어 있습니다.")

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    train_df["date"] = pd.to_datetime(train_df["date"])
    valid_df["date"] = pd.to_datetime(valid_df["date"])

    for name, df in [("Train", train_df), ("Valid", valid_df)]:
        if df[["date", *group_cols]].isna().any().any():
            raise ValueError(f"{name} 통계 그룹 키에 결측치가 있습니다.")

        if not df["date"].eq(df["date"].dt.normalize()).all():
            raise ValueError("date는 시간 정보 없는 일별 날짜여야 합니다.")

        if not np.array_equal(
            df["dayofweek"].to_numpy(),
            df["date"].dt.dayofweek.to_numpy(),
        ):
            raise ValueError(f"{name}의 dayofweek가 날짜와 다릅니다.")

    train_end = train_df["date"].max()

    if valid_df["date"].min() <= train_end:
        raise ValueError("Validation은 Train 종료일 이후여야 합니다.")

    def normalize_keys(frame):
        # 선택한 그룹 키만 정규화: D/E에는 dayofweek가 없음.
        if "family" in group_cols:
            frame["family"] = frame["family"].astype(str)
        if "dayofweek" in group_cols:
            frame["dayofweek"] = frame["dayofweek"].astype("int64")
        if "store_nbr" in group_cols:
            frame["store_nbr"] = frame["store_nbr"].astype("int64")
        if "is_holiday" in group_cols:
            frame["is_holiday"] = frame["is_holiday"].astype("int8")
        if "event_group" in group_cols:
            frame["event_group"] = frame["event_group"].astype(str)
        return frame

    # Train sales만 통계 재료로 사용
    history = train_df[
        ["date", *group_cols, "sales"]
    ].copy()

    sales = history["sales"].to_numpy(dtype=float)

    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Train sales에 결측·무한대·음수가 있습니다.")

    history = normalize_keys(history)

    # 날짜 × 그룹별 합계와 행 수
    daily = (
        history.groupby(
            ["date", *group_cols],
            observed=True,
            as_index=False,
        )
        .agg(
            sales_sum=("sales", "sum"),
            sales_count=("sales", "count"),
        )
        .sort_values([*group_cols, "date"])
    )

    grouped = daily.groupby(
        group_cols,
        observed=True,
        sort=False,
    )

    daily["cum_sum"] = grouped["sales_sum"].cumsum()
    daily["cum_count"] = grouped["sales_count"].cumsum()

    daily[feature_name] = (
        daily["cum_sum"] / daily["cum_count"]
    )

    lookup = (
        daily[["date", *group_cols, feature_name]]
        .rename(columns={"date": "history_date"})
        .sort_values("history_date")
    )

    def attach(df, is_validation):
        query = df[["date", *group_cols]].copy()
        query = normalize_keys(query)
        query["_row_order"] = np.arange(len(query))

        if is_validation:
            query["cutoff_date"] = train_end
        else:
            query["cutoff_date"] = (
                query["date"] - pd.Timedelta(days=1)
            )

        matched = pd.merge_asof(
            query.sort_values("cutoff_date"),
            lookup,
            left_on="cutoff_date",
            right_on="history_date",
            by=group_cols,
            direction="backward",
            allow_exact_matches=True,
        )

        invalid = (
            matched["history_date"].notna()
            & matched["history_date"].gt(matched["cutoff_date"])
        )

        if invalid.any():
            raise ValueError("기준일 이후의 통계가 연결되었습니다.")

        matched = matched.sort_values("_row_order")
        df[feature_name] = matched[feature_name].to_numpy()

        return df

    return (
        attach(train_df, is_validation=False),
        attach(valid_df, is_validation=True),
    )


# =========================================================
# 10. A: 상품군 × 요일 — 함수 보존
# =========================================================

def add_family_dayofweek_mean(train_df, valid_df):
    return _add_group_past_mean(
        train_df=train_df,
        valid_df=valid_df,
        group_cols=["family", "dayofweek"],
        feature_name=GROUP_STAT_FEATURES_A[0],
    )


# =========================================================
# 11. B: 매장 × 상품군 × 요일 — 이번 실험
# =========================================================

def add_store_family_dayofweek_mean(train_df, valid_df):
    return _add_group_past_mean(
        train_df=train_df,
        valid_df=valid_df,
        group_cols=["store_nbr", "family", "dayofweek"],
        feature_name=GROUP_STAT_FEATURES_B[0],
    )


# =========================================================
# 12. C: 매장 × 상품군 — 기존 기능 호환
# =========================================================

def add_store_family_mean(train_df, valid_df):
    return _add_group_past_mean(
        train_df=train_df,
        valid_df=valid_df,
        group_cols=["store_nbr", "family"],
        feature_name=GROUP_STAT_FEATURES_C[0],
    )


# =========================================================
# 13. D: 상품군 × 휴일 여부
# =========================================================

def add_family_holiday_mean(train_df, valid_df):
    return _add_group_past_mean(
        train_df=train_df,
        valid_df=valid_df,
        group_cols=["family", "is_holiday"],
        feature_name=GROUP_STAT_FEATURES_D[0],
    )


# =========================================================
# 14. E: 상품군 × 이벤트 종류
# =========================================================

def add_family_event_group_mean(train_df, valid_df):
    return _add_group_past_mean(
        train_df=train_df,
        valid_df=valid_df,
        group_cols=["family", "event_group"],
        feature_name=GROUP_STAT_FEATURES_E[0],
    )


# =========================================================
# 15. 일반 Feature Engineering 파이프라인
# =========================================================

def apply_feature_engineering(df):
    df = add_date_features(df)
    df = add_promotion_features(df)

    # Lag / Rolling / EMA는 과거 판매량 준비 후 별도 호출.
    # Group Statistics는 Fold 분리 후 별도 호출.
    return df


# =========================================================
# 16. Holiday 실험용 목록
# =========================================================

HOLIDAY_FEATURES_A = [
    "holiday_count",
]

HOLIDAY_FEATURES_B = [
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

HOLIDAY_FEATURES_C = (
    HOLIDAY_FEATURES_A + HOLIDAY_FEATURES_B
)

# =========================================================
# 17. 휴일 / 이벤트 그룹 키 생성 — D / E 실험용
# =========================================================

EVENT_FLAG_COLUMNS = HOLIDAY_FEATURES_B.copy()


def add_holiday_group_keys(df):
    """
    main()이 이미 매장에 맞춰 병합한 holiday_count와 event_* 컬럼 활용.

    D: is_holiday = holiday_count > 0
    E: event_group
       - No Event       : 휴일 아님
       - Other Holiday  : 휴일이나 9개 이벤트 플래그 없음
       - christmas 등   : 해당 이벤트 플래그 한 종류
       - Multiple Events: 여러 이벤트 플래그 동시에 참

    새 그룹 키는 미래에 알려진 달력 정보로만 생성; sales를 사용하지 않음.
    """
    df = df.copy()
    required = ["holiday_count", *EVENT_FLAG_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Holiday 그룹화에 필요한 컬럼이 없습니다: {missing}")

    count = pd.to_numeric(df["holiday_count"], errors="coerce")
    if count.isna().any() or not np.isfinite(count.to_numpy(dtype=float)).all():
        raise ValueError("holiday_count에 유효하지 않은 값이 있습니다.")
    if (count < 0).any():
        raise ValueError("holiday_count는 음수일 수 없습니다.")

    flags = df[EVENT_FLAG_COLUMNS]
    if flags.isna().any().any():
        raise ValueError("이벤트 플래그에 결측치가 있습니다.")
    if not flags.isin([0, 1, False, True]).all().all():
        raise ValueError("이벤트 플래그는 0 또는 1이어야 합니다.")

    active = flags.astype("int8")
    num_active = active.sum(axis=1).to_numpy()
    is_holiday = count.gt(0).to_numpy()
    if ((~is_holiday) & (num_active > 0)).any():
        raise ValueError(
            "holiday_count=0인데 이벤트 플래그가 1입니다. "
            "main()의 Holiday Store Matching을 확인하세요."
        )

    df["is_holiday"] = is_holiday.astype("int8")

    # 날짜에 이벤트가 여러 개 중첩돼도 임의로 하나를 선택하지 않음.
    # E의 EDA에서 중첩 빈도도 확인 가능.
    group = np.full(len(df), "No Event", dtype=object)
    group[is_holiday & (num_active == 0)] = "Other Holiday"
    group[num_active > 1] = "Multiple Events"
    for col in EVENT_FLAG_COLUMNS:
        mask = (num_active == 1) & active[col].to_numpy(dtype=bool)
        group[mask] = col.removeprefix("event_")

    df["event_group"] = group
    return df
