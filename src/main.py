
from preprocessing import run_preprocessing

from feature_engineering import (
    apply_feature_engineering,
    add_holiday_features,
    merge_holiday_features
)


# =========================================================
# 1. 전체 파이프라인 실행
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. Preprocessing
    # -----------------------------------------------------

    train_df, valid_df, test, holidays = run_preprocessing()

    # -----------------------------------------------------
    # 2. Feature Engineering
    # -----------------------------------------------------

    train_fe = apply_feature_engineering(train_df)
    valid_fe = apply_feature_engineering(valid_df)
    test_fe = apply_feature_engineering(test)

    holidays_fe = add_holiday_features(holidays)

    # -----------------------------------------------------
    # 3. National Holiday Merge
    # -----------------------------------------------------

    train_fe = merge_holiday_features(train_fe, holidays_fe)
    valid_fe = merge_holiday_features(valid_fe, holidays_fe)
    test_fe = merge_holiday_features(test_fe, holidays_fe)

    # -----------------------------------------------------
    # 4. Feature Engineering QC
    # -----------------------------------------------------

    # print("\n===== FEATURE ENGINEERING SHAPE =====")
    # print("Train   :", train_fe.shape)
    # print("Valid   :", valid_fe.shape)
    # print("Test    :", test_fe.shape)
    # print("Holidays:", holidays_fe.shape)

    # print("\n===== TRAIN FEATURE DTYPES =====")
    # print(train_fe.dtypes)

    # print("\n===== VALID FEATURE DTYPES =====")
    # print(valid_fe.dtypes)

    # print("\n===== TEST FEATURE DTYPES =====")
    # print(test_fe.dtypes)

    # print("\n===== HOLIDAY FEATURE DTYPES =====")
    # print(holidays_fe.dtypes)

    # print("\n===== FEATURE ENGINEERING NULL VALUES =====")
    # print("Train:", int(train_fe.isna().sum().sum()))
    # print("Valid:", int(valid_fe.isna().sum().sum()))
    # print("Test :", int(test_fe.isna().sum().sum()))

    # -----------------------------------------------------
    # 5. National Holiday Merge QC
    # -----------------------------------------------------

    # print("\n===== NATIONAL EVENT COUNTS =====")

    # for name, df in [
    #     ("Train", train_fe),
    #     ("Valid", valid_fe),
    #     ("Test", test_fe)
    # ]:
    #     print(f"\n{name}:")
    #     print(df["has_national_event"].value_counts())

    # -----------------------------------------------------
    # 6. Event Binary Feature QC
    # -----------------------------------------------------

    event_cols = [
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

    # print("\n===== EVENT BINARY FEATURE COUNTS =====")
    # print(train_fe[event_cols].sum())

    # print("\n===== EVENT BINARY UNIQUE VALUES =====")

    # for col in event_cols:
    #     print(col, ":", sorted(train_fe[col].unique()))

    # -----------------------------------------------------
    # 7. Holiday Feature Sample
    # -----------------------------------------------------

    holiday_cols = [
        "date",
        "holiday_count",
        "holiday_types",
        "event_groups",
        "holiday_descriptions",
        "has_national_event"
    ]

    # print("\n===== HOLIDAY FEATURES SAMPLE =====")

    # print(
    #     train_fe[holiday_cols]
    #     .drop_duplicates()
    #     .head(15)
    #     .to_string(index=False)
    # )

    # -----------------------------------------------------
    # 8. 병합된 데이터 미리보기
    # -----------------------------------------------------

    preview_cols = [
        "date",
        "store_nbr",
        "family",
        "onpromotion",
        "dayofweek",
        "is_weekend",
        "holiday_count",
        "holiday_types",
        "event_groups",
        "holiday_descriptions",
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

    # print("\n===== TRAIN MERGED SAMPLE (10 ROWS) =====")
    # print(train_fe[preview_cols].head(10).to_string(index=False))

    # print("\n===== VALID MERGED SAMPLE (10 ROWS) =====")
    # print(valid_fe[preview_cols].head(10).to_string(index=False))

    # print("\n===== TEST MERGED SAMPLE (10 ROWS) =====")
    # print(test_fe[preview_cols].head(10).to_string(index=False))

    # -----------------------------------------------------
    # 9. National Event Sample
    # -----------------------------------------------------

    # national_event_rows = (
    #     train_fe.loc[
    #         train_fe["has_national_event"],
    #         preview_cols
    #     ]
    #     .drop_duplicates(subset=["date"])
    #     .head(10)
    # )

    # print("\n===== NATIONAL EVENT ROWS SAMPLE =====")
    # print(national_event_rows.to_string(index=False))

    # -----------------------------------------------------
    # 10. Good Friday / Labor Day Sample
    # -----------------------------------------------------

    # new_event_rows = (
    #     train_fe.loc[
    #         (train_fe["event_good_friday"] == 1)
    #         | (train_fe["event_labor_day"] == 1),
    #         [
    #             "date",
    #             "holiday_types",
    #             "event_groups",
    #             "holiday_descriptions",
    #             "event_good_friday",
    #             "event_labor_day"
    #         ]
    #     ]
    #     .drop_duplicates()
    #     .head(15)
    # )

    # print("\n===== GOOD FRIDAY / LABOR DAY SAMPLE =====")
    # print(new_event_rows.to_string(index=False))

    # -----------------------------------------------------
    # 11. 병합 전후 행 수 검증 (유지)
    # -----------------------------------------------------

    assert len(train_fe) == len(train_df), (
        "Train 병합 과정에서 행 수가 변경되었습니다."
    )

    assert len(valid_fe) == len(valid_df), (
        "Valid 병합 과정에서 행 수가 변경되었습니다."
    )

    assert len(test_fe) == len(test), (
        "Test 병합 과정에서 행 수가 변경되었습니다."
    )

    # print("\n===== MERGE ROW VALIDATION =====")
    # print("Train:", len(train_df), "->", len(train_fe))
    # print("Valid:", len(valid_df), "->", len(valid_fe))
    # print("Test :", len(test), "->", len(test_fe))

    # -----------------------------------------------------
    # 12. 중복 키 검증 (유지)
    # -----------------------------------------------------

    key_cols = [
        "date",
        "store_nbr",
        "family"
    ]

    for name, df in [
        ("Train", train_fe),
        ("Valid", valid_fe),
        ("Test", test_fe)
    ]:

        duplicate_count = int(
            df.duplicated(subset=key_cols).sum()
        )

        assert duplicate_count == 0, (
            f"{name} 데이터에서 중복 키가 발견되었습니다."
        )

    # -----------------------------------------------------
    # 13. Holiday 결측치 검증 (유지)
    # -----------------------------------------------------

    check_cols = [
        "holiday_count",
        "holiday_types",
        "event_groups",
        "holiday_descriptions",
        "has_national_event"
    ] + event_cols

    for name, df in [
        ("Train", train_fe),
        ("Valid", valid_fe),
        ("Test", test_fe)
    ]:

        null_count = int(
            df[check_cols].isna().sum().sum()
        )

        assert null_count == 0, (
            f"{name}의 Holiday Feature에 결측치가 있습니다."
        )

    # -----------------------------------------------------
    # 14. 이벤트 0/1 값 검증 (유지)
    # -----------------------------------------------------

    for name, df in [
        ("Train", train_fe),
        ("Valid", valid_fe),
        ("Test", test_fe)
    ]:

        is_binary = df[event_cols].isin([0, 1]).all().all()

        assert is_binary, (
            f"{name}의 이벤트 피처에 0과 1 이외의 값이 있습니다."
        )

    # -----------------------------------------------------
    # 15. 최종 상태
    # -----------------------------------------------------

    print("\n===== PIPELINE STATUS =====")
    print("Preprocessing: OK")
    print("Feature Engineering: OK")
    print("National Holiday Merge: OK")
    print("All QC Checks: OK")

    # -----------------------------------------------------
    # 16. 결과 반환
    # -----------------------------------------------------

    return train_fe, valid_fe, test_fe, holidays_fe


# =========================================================
# 2. 프로그램 시작
# =========================================================

if __name__ == "__main__":
    train_fe, valid_fe, test_fe, holidays_fe = main()
