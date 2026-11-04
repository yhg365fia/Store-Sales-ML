import csv
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\projects\Store Sales-ML")

SOURCE = (
    ROOT
    / "submissions"
    / "rolling_and_three_model_20261103_175244_222686"
    / "model_predictions.csv"
)

EMA_FAMILIES = {
    "SCHOOL AND OFFICE SUPPLIES",
    "LIQUOR,WINE,BEER",
    "HOME APPLIANCES",
}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"저장된 예측 파일이 없습니다: {SOURCE}")

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "id", "family", "ema_pred_sales", "baseline_pred_sales"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("원본 파일에 필요한 컬럼이 없습니다.")
        rows = list(reader)

    # 제출 행 수와 ID 중복 확인
    if len(rows) != 28_512:
        raise ValueError(f"제출 행 수가 다릅니다: {len(rows):,}")

    ids = [int(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("중복 ID가 있습니다.")

    # 각 품목: 54개 매장 × 16일 = 864행
    for family in EMA_FAMILIES:
        count = sum(row["family"] == family for row in rows)
        if count != 864:
            raise ValueError(f"{family} 행 수 오류: {count}")

    submission = []
    ema_count = 0

    for row in rows:
        use_ema = row["family"] in EMA_FAMILIES
        column = "ema_pred_sales" if use_ema else "baseline_pred_sales"
        prediction = float(row[column])

        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError(
                f"잘못된 예측값: id={row['id']}, sales={prediction}"
            )

        submission.append({
            "id": int(row["id"]),
            "sales": prediction,
        })
        ema_count += int(use_ema)

    submission.sort(key=lambda row: row["id"])

    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / (
        f"submission_ema7_three_families_trial17_rest_{timestamp}.csv"
    )

    # 기존 파일을 덮어쓰지 않음
    with output_path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sales"])
        writer.writeheader()
        writer.writerows(submission)

    # 저장 후 내용 검증
    with output_path.open("r", encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))

    if len(saved) != len(submission):
        raise ValueError("저장 후 행 수가 일치하지 않습니다.")

    for actual, expected in zip(saved, submission):
        if (
            int(actual["id"]) != expected["id"]
            or float(actual["sales"]) != expected["sales"]
        ):
            raise ValueError("저장 후 예측값이 일치하지 않습니다.")

    print(f"EMA7 적용: {ema_count:,}행 / 3개 품목")
    print(f"Trial17 적용: {len(rows) - ema_count:,}행 / 30개 품목")
    print(f"전체: {len(saved):,}행")
    print(f"\n캐글 제출 파일:\n{output_path}")


if __name__ == "__main__":
    main()