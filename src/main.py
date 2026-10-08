
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

    print("\n===== FEATURE ENGINEERING SHAPE =====")
    print("Train   :", train_fe.shape)
    print("Valid   :", valid_fe.shape)
    print("Test    :", test_fe.shape)
    print("Holidays:", holidays_fe.shape)

    print("\n===== TRAIN FEATURE DTYPES =====")
    print(train_fe.dtypes)

    print("\n===== VALID FEATURE DTYPES =====")
    print(valid_fe.dtypes)

    print("\n===== TEST FEATURE DTYPES =====")
    print(test_fe.dtypes)

    print("\n===== HOLIDAY FEATURE DTYPES =====")
    print(holidays_fe.dtypes)

    print("\n===== FEATURE ENGINEERING NULL VALUES =====")
    print("Train:", int(train_fe.isna().sum().sum()))
    print("Valid:", int(valid_fe.isna().sum().sum()))
    print("Test :", int(test_fe.isna().sum().sum()))

    # -----------------------------------------------------
    # 5. Holiday Merge QC
    # -----------------------------------------------------

    print("\n===== NATIONAL EVENT COUNTS =====")
    print("Train:")
    print(train_fe["has_national_event"].value_counts())

    print("\nValid:")
    print(valid_fe["has_national_event"].value_counts())

    print("\nTest:")
    print(test_fe["has_national_event"].value_counts())

    print("\n===== HOLIDAY FEATURES SAMPLE =====")
    print(
        train_fe[
            [
                "date",
                "holiday_count",
                "holiday_types",
                "event_groups",
                "holiday_descriptions",
                "has_national_event"
            ]
        ]
        .drop_duplicates()
        .head(15)
    )

    # -----------------------------------------------------
    # 6. 병합 검증
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

    print("\n===== PIPELINE STATUS =====")
    print("Preprocessing: OK")
    print("Feature Engineering: OK")
    print("National Holiday Merge: OK")

    # -----------------------------------------------------
    # 7. 결과 반환
    # -----------------------------------------------------

    return train_fe, valid_fe, test_fe, holidays_fe


# =========================================================
# 2. 프로그램 시작
# =========================================================

if __name__ == "__main__":
    train_fe, valid_fe, test_fe, holidays_fe = main()
