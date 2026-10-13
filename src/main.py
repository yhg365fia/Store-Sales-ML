from preprocessing import run_preprocessing

from feature_engineering import apply_feature_engineering


# =========================================================
# 1. 전체 파이프라인 실행
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. Preprocessing
    #
    # - 기본 전처리
    # - stores 전처리
    # - holidays 전처리
    # - Holiday ↔ Store 지역 매칭
    # - store_date_features 생성
    # - Train / Test 병합
    # - Train / Validation Split
    # -----------------------------------------------------

    train_df, test, holidays = run_preprocessing()

    # -----------------------------------------------------
    # 2. Feature Engineering
    #
    # date / promotion 등 판매 데이터 기반 피처 생성
    # Holiday / Store Feature는 이미 preprocessing에서 병합됨
    # -----------------------------------------------------

    train_fe = apply_feature_engineering(train_df)
    test_fe = apply_feature_engineering(test)

    # -----------------------------------------------------
    # 3. Event Binary Feature 목록
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
        "event_other",
    ]

    # -----------------------------------------------------
    # 4. Feature Engineering QC
    # -----------------------------------------------------

    # print("\n===== FEATURE ENGINEERING SHAPE =====")
    # print("Train:", train_fe.shape)
    # print("Valid:", valid_fe.shape)
    # print("Test :", test_fe.shape)

    # print("\n===== TRAIN FEATURE DTYPES =====")
    # print(train_fe.dtypes)

    # print("\n===== VALID FEATURE DTYPES =====")
    # print(valid_fe.dtypes)

    # print("\n===== TEST FEATURE DTYPES =====")
    # print(test_fe.dtypes)

    # print("\n===== FEATURE ENGINEERING NULL VALUES =====")
    # print("Train:", int(train_fe.isna().sum().sum()))
    # print("Valid:", int(valid_fe.isna().sum().sum()))
    # print("Test :", int(test_fe.isna().sum().sum()))

    # -----------------------------------------------------
    # 5. Store Feature QC
    # -----------------------------------------------------

    store_cols = [
        "city",
        "state",
        "type",
        "cluster",
    ]

    for name, df in [
        ("Train", train_fe),
        ("Test", test_fe),
    ]:

        null_count = int(
            df[store_cols]
            .isna()
            .sum()
            .sum()
        )

        assert null_count == 0, (
            f"{name}의 Store Feature에 결측치가 있습니다."
        )

    # -----------------------------------------------------
    # 6. Holiday Feature QC
    # -----------------------------------------------------

    holiday_cols = [
        "holiday_count",
        *event_cols,
    ]

    for name, df in [
        ("Train", train_fe),
        ("Test", test_fe),
    ]:

        null_count = int(
            df[holiday_cols]
            .isna()
            .sum()
            .sum()
        )

        assert null_count == 0, (
            f"{name}의 Holiday Feature에 결측치가 있습니다."
        )

    # -----------------------------------------------------
    # 7. Event Binary 값 검증
    # -----------------------------------------------------

    for name, df in [
        ("Train", train_fe),
        ("Test", test_fe),
    ]:

        is_binary = (
            df[event_cols]
            .isin([0, 1])
            .all()
            .all()
        )

        assert is_binary, (
            f"{name}의 이벤트 피처에 "
            "0과 1 이외의 값이 있습니다."
        )

    # -----------------------------------------------------
    # 8. Holiday Count 검증
    # -----------------------------------------------------

    for name, df in [
        ("Train", train_fe),
        ("Test", test_fe),
    ]:

        assert (
            df["holiday_count"] >= 0
        ).all(), (
            f"{name}의 holiday_count에 음수가 있습니다."
        )

    # -----------------------------------------------------
    # 9. 병합 전후 행 수 검증
    # -----------------------------------------------------

    assert len(train_fe) == len(train_df), (
        "Feature Engineering 과정에서 "
        "Train 행 수가 변경되었습니다."
    )

    assert len(test_fe) == len(test), (
        "Feature Engineering 과정에서 "
        "Test 행 수가 변경되었습니다."
    )

    # -----------------------------------------------------
    # 10. 중복 키 검증
    # -----------------------------------------------------

    key_cols = [
        "date",
        "store_nbr",
        "family",
    ]

    for name, df in [
        ("Train", train_fe),
        ("Test", test_fe),
    ]:

        duplicate_count = int(
            df
            .duplicated(subset=key_cols)
            .sum()
        )

        assert duplicate_count == 0, (
            f"{name} 데이터에서 "
            "date + store_nbr + family "
            "중복 키가 발견되었습니다."
        )

    # -----------------------------------------------------
    # 11. Event Binary Feature QC
    # -----------------------------------------------------

    # print("\n===== EVENT BINARY FEATURE COUNTS =====")
    # print(train_fe[event_cols].sum())

    # print("\n===== EVENT BINARY UNIQUE VALUES =====")

    # for col in event_cols:
    #     print(
    #         col,
    #         ":",
    #         sorted(train_fe[col].unique())
    #     )

    # -----------------------------------------------------
    # 12. Store + Holiday 병합 결과 미리보기
    # -----------------------------------------------------

    preview_cols = [
        "date",
        "store_nbr",
        "family",

        "city",
        "state",
        "type",
        "cluster",

        "onpromotion",

        "dayofweek",
        "is_weekend",

        "holiday_count",

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

    # print("\n===== TRAIN SAMPLE =====")
    # print(
    #     train_fe[preview_cols]
    #     .head(10)
    #     .to_string(index=False)
    # )

    # print("\n===== VALID SAMPLE =====")
    # print(
    #     valid_fe[preview_cols]
    #     .head(10)
    #     .to_string(index=False)
    # )

    # print("\n===== TEST SAMPLE =====")
    # print(
    #     test_fe[preview_cols]
    #     .head(10)
    #     .to_string(index=False)
    # )

    # -----------------------------------------------------
    # 13. Holiday 적용 샘플
    # -----------------------------------------------------

    # holiday_sample = (
    #     train_fe.loc[
    #         train_fe["holiday_count"] > 0,
    #         preview_cols
    #     ]
    #     .drop_duplicates(
    #         subset=[
    #             "date",
    #             "store_nbr"
    #         ]
    #     )
    #     .head(20)
    # )

    # print("\n===== HOLIDAY SAMPLE =====")
    # print(
    #     holiday_sample.to_string(index=False)
    # )

    # -----------------------------------------------------
    # 14. Test 기간 Holiday 확인
    #
    # 특히 Ambato Local Holiday가
    # Ambato 매장에만 적용됐는지 확인 가능
    # -----------------------------------------------------

    # test_holiday_sample = (
    #     test_fe.loc[
    #         test_fe["holiday_count"] > 0,
    #         [
    #             "date",
    #             "store_nbr",
    #             "city",
    #             "state",
    #             "holiday_count",
    #             *event_cols,
    #         ]
    #     ]
    #     .drop_duplicates(
    #         subset=[
    #             "date",
    #             "store_nbr"
    #         ]
    #     )
    # )

    # print("\n===== TEST HOLIDAY SAMPLE =====")
    # print(
    #     test_holiday_sample.to_string(index=False)
    # )

    # -----------------------------------------------------
    # 15. 최종 상태
    # -----------------------------------------------------

    print("\n===== PIPELINE STATUS =====")
    print("Preprocessing: OK")
    print("Store Date Features: OK")
    print("Holiday Store Matching: OK")
    print("Feature Engineering: OK")
    print("All QC Checks: OK")

    # -----------------------------------------------------
    # 16. 결과 반환
    # -----------------------------------------------------

    return (
        train_fe,
        test_fe,
        holidays,
    )


# =========================================================
# 2. 프로그램 시작
# =========================================================

if __name__ == "__main__":

    train_fe, test_fe, holidays = main()