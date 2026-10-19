"""
Payday Baseline vs feature ablation: 3-Fold mean Gain/Split comparison.
Place this file at src/modeling.py with existing main.py and feature_engineering.py.
Run from project root: python src/modeling.py
Default: baseline (29) + drop_is_weekend (28), total 6 models.
Edit ABLATION_EXPERIMENTS below to add independent or joint removals.
Results: results/feature_ablation/<timestamp>/
"""

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from main import main
from feature_engineering import LAGS, LAG_FEATURES, add_lag_features


# =========================================================
# 1. 기존 Payday Baseline 구성: 총 29개
# =========================================================

BASE_FEATURES = [
    # Store / Product (6)
    "store_nbr", "family", "city", "state", "type", "cluster",
    # Calendar (5)
    "year", "month", "day", "dayofweek", "is_weekend",
    # Promotion (2)
    "onpromotion", "is_promotion",
    # Holiday (10)
    "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]

PAYDAY_FEATURES = ["is_15th", "is_month_end"]
CATEGORICAL_FEATURES = [
    "store_nbr", "family", "city", "state", "type", "cluster",
]

BASELINE_FEATURES = (
    BASE_FEATURES + list(LAG_FEATURES) + ["dayofyear"] + PAYDAY_FEATURES
)

TARGET_COL = "sales_log"
FORECAST_HORIZON = 16
N_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 150

# 각 항목은 동일한 29개 Baseline에서 독립적으로 제거합니다.
# 두 피처를 한 목록에 넣으면 두 피처를 동시에 제거하는 실험입니다.
ABLATION_EXPERIMENTS = {
    "drop_is_weekend": ["is_weekend"],
}

FOLDS = [
    {"fold": "Fold 1", "train_end": "2017-06-28", "valid_start": "2017-06-29", "valid_end": "2017-07-14"},
    {"fold": "Fold 2", "train_end": "2017-07-14", "valid_start": "2017-07-15", "valid_end": "2017-07-30"},
    {"fold": "Fold 3", "train_end": "2017-07-30", "valid_start": "2017-07-31", "valid_end": "2017-08-15"},
]

# modeling.py가 src/ 안에 있을 때 프로젝트 루트의 results/ 사용.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "feature_ablation"


# =========================================================
# 2. 설정 검증 및 기본 피처 준비
# =========================================================

def validate_configuration():
    if not ABLATION_EXPERIMENTS:
        raise ValueError("Ablation 실험을 하나 이상 설정하세요.")
    for name, removed in ABLATION_EXPERIMENTS.items():
        if (not isinstance(name, str) or not name or name == "baseline"
                or not all(c.isalnum() or c in "_-" for c in name)):
            raise ValueError("실험명은 baseline 이외의 문자/숫자/_/-로 지정하세요.")
        if not isinstance(removed, list) or not removed:
            raise ValueError(f"{name}: 제거 피처는 비어 있지 않은 리스트여야 합니다.")
        if len(set(removed)) != len(removed):
            raise ValueError(f"{name}: 제거 목록에 중복이 있습니다.")
        unknown = set(removed) - set(BASELINE_FEATURES)
        if unknown:
            raise ValueError(f"{name}: 존재하지 않는 피처 {sorted(unknown)}")
        if len(removed) >= len(BASELINE_FEATURES):
            raise ValueError("모든 피처를 제거할 수 없습니다.")
    if tuple(LAGS) != (16, 21, 28):
        raise ValueError("LAGS는 (16, 21, 28)이어야 합니다.")
    if list(LAG_FEATURES) != ["sales_lag_16", "sales_lag_21", "sales_lag_28"]:
        raise ValueError("Lag 피처 구성이 기존과 다릅니다.")
    if len(BASE_FEATURES) != 23:
        raise ValueError(f"BASE_FEATURES는 23개여야 합니다 (실제 {len(BASE_FEATURES)}개).")
    if len(BASELINE_FEATURES) != 29 or len(set(BASELINE_FEATURES)) != 29:
        raise ValueError("Baseline은 중복 없이 정확히 29개 피처여야 합니다.")
    forbidden = [
        f for f in BASELINE_FEATURES
        if (any(x in f.lower() for x in ("rolling", "ema", "seasonal", "_mean"))
            or f in {"sales", TARGET_COL, "dayofyear_sin", "dayofyear_cos", "sales_lag_364"})
    ]
    if forbidden:
        raise ValueError(f"Baseline에 포함되면 안 되는 피처: {forbidden}")


def prepare_features(df):
    validate_configuration()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    required = ["date", "sales", *BASE_FEATURES, "dayofyear", *PAYDAY_FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"필수 컬럼 누락: {missing}\n"
            "main()의 전처리/휴일 매장 매칭/Feature Engineering부터 확인하세요."
        )

    keys = ["store_nbr", "family"]
    if df[keys + ["date"]].isna().any().any():
        raise ValueError("매장·상품군·날짜에 결측치가 있습니다.")
    if df.duplicated(keys + ["date"]).any():
        raise ValueError("매장 × 상품군 × 날짜 조합이 중복됩니다.")

    sales = df["sales"].to_numpy(dtype=float)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Train sales에 결측·무한대·음수가 있습니다.")

    expected_calendar = {
        "dayofyear": df["date"].dt.dayofyear,
        "dayofweek": df["date"].dt.dayofweek,
        "is_weekend": (df["date"].dt.dayofweek >= 5).astype(int),
        "is_15th": (df["date"].dt.day == 15).astype(int),
        "is_month_end": df["date"].dt.is_month_end.astype(int),
    }
    for feat, expected in expected_calendar.items():
        if not np.array_equal(df[feat].to_numpy(), expected.to_numpy()):
            raise ValueError(f"{feat} 계산값이 date와 일치하지 않습니다.")

    df = df.sort_values(keys + ["date"]).reset_index(drop=True)

    # 현재 전처리는 12/25 등의 누락 날짜를 패딩하여 매 그룹이 연속 일별 데이터임.
    # 이 전제가 깨지면 shift(16)은 16일 전 판매량을 가리키지 않으므로 오류 처리.
    date_diff = df.groupby(keys, observed=True)["date"].diff()
    invalid_gap = date_diff.notna() & date_diff.ne(pd.Timedelta(days=1))
    if invalid_gap.any():
        raise ValueError("그룹 내 날짜가 연속적이지 않습니다. main()의 날짜 패딩을 확인하세요.")

    df = add_lag_features(df, lags=LAGS)
    df[TARGET_COL] = np.log1p(df["sales"])

    group_position = df.groupby(keys, observed=True).cumcount()
    for lag in LAGS:
        actual = df[f"sales_lag_{lag}"].isna()
        expected = group_position < lag
        if not actual.equals(expected):
            raise ValueError(f"sales_lag_{lag} 결측 위치가 예상과 다릅니다.")

    print("\n" + "=" * 80)
    print("FEATURE ABLATION — BASELINE VS REMOVED FEATURES")
    print("=" * 80)
    print(f"Baseline features: {len(BASELINE_FEATURES)}")
    print("Group Statistics: NOT INCLUDED")
    print(f"n_estimators: {N_ESTIMATORS} / early stopping: {EARLY_STOPPING_ROUNDS}")
    print(f"Total models: {len(FOLDS) * (1 + len(ABLATION_EXPERIMENTS))}")
    print("Weekend QC: is_weekend == (dayofweek >= 5): OK")
    print("Lag missing:\n", df[list(LAG_FEATURES)].isna().sum().to_string())
    print("Common QC: OK")
    return df


# =========================================================
# 3. 기존 16일 3-Fold
# =========================================================

def make_fold(df, config):
    train_end = pd.Timestamp(config["train_end"])
    valid_start = pd.Timestamp(config["valid_start"])
    valid_end = pd.Timestamp(config["valid_end"])
    horizon = (valid_end - valid_start).days + 1

    if horizon != FORECAST_HORIZON or valid_start != train_end + pd.Timedelta(days=1):
        raise ValueError("Fold 날짜가 16일 예측 설정과 다릅니다.")
    if min(LAGS) < horizon:
        raise ValueError("16일 일괄 예측보다 짧은 Lag가 있습니다.")

    train_df = df.loc[df["date"].le(train_end)].copy()
    valid_df = df.loc[df["date"].between(valid_start, valid_end)].copy()
    if train_df.empty or valid_df.empty or train_df["date"].max() != train_end:
        raise ValueError("Fold Train/Validation 데이터 또는 종료 날짜가 유효하지 않습니다.")

    valid_counts = valid_df.groupby(["store_nbr", "family"], observed=True).size()
    if not valid_counts.eq(horizon).all():
        raise ValueError("일부 Validation 매장 × 상품군 그룹이 16일을 채우지 못했습니다.")

    print(f"Train: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"Valid: {valid_df['date'].min().date()} ~ {valid_df['date'].max().date()}")
    print("Train shape:", train_df.shape, "| Valid shape:", valid_df.shape)
    return train_df, valid_df


# =========================================================
# 4. 실험별 모델 데이터: 제거 목록에 맞춰 구성
# =========================================================

def prepare_model_data(train_df, valid_df, features):
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()

    # 기존과 동일: Train에서 관측된 범주만으로 카테고리 결정.
    for feature in (f for f in CATEGORICAL_FEATURES if f in features):
        categories = pd.Index(train_df[feature].dropna().unique())
        X_train[feature] = pd.Categorical(X_train[feature], categories=categories)
        X_valid[feature] = pd.Categorical(X_valid[feature], categories=categories)

    required_non_missing = [f for f in features if f not in LAG_FEATURES]
    for name, X in (("Train", X_train), ("Valid", X_valid)):
        if X[required_non_missing].isna().any().any():
            raise ValueError(f"{name}: Lag 외 일반 피처에 결측치가 있습니다.")
        if list(X.columns) != features:
            raise ValueError(f"{name}: 입력 피처 구성 또는 순서가 설정과 다릅니다.")
    if X_valid[[f for f in LAG_FEATURES if f in features]].isna().any().any():
        raise ValueError("Validation Lag에 결측치가 있습니다.")

    return (
        X_train, X_valid,
        train_df[TARGET_COL].copy(),
        valid_df[TARGET_COL].copy(),
        valid_df["sales"].copy(),
    )


# =========================================================
# 5. 기존 실험과 동일한 LightGBM 설정
# =========================================================

def build_model():
    return LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        learning_rate=0.05,
        n_estimators=N_ESTIMATORS,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        min_split_gain=0.0,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        metric="rmse",
        importance_type="split",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def calculate_rmsle(y_true, pred_log):
    y_true = np.asarray(y_true, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    if not np.isfinite(pred_log).all():
        raise ValueError("예측값에 NaN 또는 무한대가 있습니다.")
    pred_log = np.maximum(pred_log, 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - pred_log) ** 2)))


# =========================================================
# 6. Feature Importance: 모든 Fold, 실험별 Gain + Split
# =========================================================

def get_feature_importance(model, fold_name, features):
    booster = model.booster_
    iteration = int(model.best_iteration_)
    names = booster.feature_name()
    if names != features:
        raise ValueError("학습 모델의 피처 구성이 실험 설정과 다릅니다.")

    gain = booster.feature_importance(importance_type="gain", iteration=iteration)
    split = booster.feature_importance(importance_type="split", iteration=iteration)
    result = pd.DataFrame({
        "fold": fold_name,
        "best_iteration": iteration,
        "feature": names,
        "gain": gain,
        "split": split,
    })
    total_gain = result["gain"].sum()
    total_split = result["split"].sum()
    result["gain_pct"] = result["gain"] / total_gain * 100 if total_gain > 0 else 0.0
    result["split_pct"] = result["split"] / total_split * 100 if total_split > 0 else 0.0
    return result


# =========================================================
# 7. 한 Fold 학습
# =========================================================

def train_experiment_model(train_df, valid_df, fold_name, features):
    X_train, X_valid, y_train, y_valid_log, y_valid = prepare_model_data(
        train_df, valid_df, features
    )
    model = build_model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid_log)],
        eval_metric="rmse",
        categorical_feature=[f for f in CATEGORICAL_FEATURES if f in features],
        callbacks=[
            early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            log_evaluation(period=100),
        ],
    )
    iteration = int(model.best_iteration_)
    pred_log = model.predict(X_valid, num_iteration=iteration)
    score = calculate_rmsle(y_valid, pred_log)
    importance_df = get_feature_importance(model, fold_name, features)

    print(f"\nBest iteration: {iteration}")
    print(f"RMSLE: {score:.6f}")
    if iteration >= N_ESTIMATORS - EARLY_STOPPING_ROUNDS:
        print("CHECK: Best iteration이 학습 상한에 가깝습니다.")
    del X_train, X_valid, y_train, y_valid_log, y_valid, pred_log, model
    gc.collect()
    return iteration, score, importance_df


# =========================================================
# 8. 3-Fold 실행, 전체 중요도 및 제거 전후 비교
# =========================================================


def summarize_importance(per_fold):
    result = (
        per_fold.groupby("feature", as_index=False)
        .agg(mean_gain_pct=("gain_pct", "mean"),
             mean_split_pct=("split_pct", "mean"),
             mean_gain=("gain", "mean"), mean_split=("split", "mean"))
        .sort_values("mean_gain_pct", ascending=False)
        .reset_index(drop=True)
    )
    result.insert(0, "gain_rank", result["mean_gain_pct"].rank(
        method="min", ascending=False).astype(int))
    result["split_rank"] = result["mean_split_pct"].rank(
        method="min", ascending=False).astype(int)
    for col in ["mean_gain_pct", "mean_split_pct"]:
        if not np.isclose(result[col].sum(), 100):
            raise ValueError(f"{col} 합계가 100%가 아닙니다.")
    return result


def compare_importance(baseline, experiment, removed):
    columns = ["feature", "gain_rank", "split_rank", "mean_gain_pct",
               "mean_split_pct", "mean_gain", "mean_split"]
    result = baseline[columns].merge(
        experiment[columns], on="feature", how="left",
        suffixes=("_baseline", "_experiment"), validate="one_to_one",
    )
    result["removed"] = result["feature"].isin(removed)
    # 제거 피처의 실험 중요도/순위는 NaN 유지: 실제 측정된 0과 구분.
    for metric in ["gain", "split"]:
        result[f"delta_{metric}_pp"] = (
            result[f"mean_{metric}_pct_experiment"]
            - result[f"mean_{metric}_pct_baseline"]
        )
    return result.sort_values("gain_rank_baseline")


def run_backtesting(save_csv=True):
    validate_configuration()
    experiments = {"baseline": [], **ABLATION_EXPERIMENTS}
    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if save_csv:
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "baseline_features": BASELINE_FEATURES,
            "experiments_removed_features": experiments,
            "folds": FOLDS,
            "model_parameters": build_model().get_params(),
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "target": "log1p(sales)",
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    train, test, holidays = main()
    del test, holidays
    gc.collect()
    train = prepare_features(train)
    fold_rows, importance_parts = [], []

    # Fold 데이터는 한 번 준비하고, 같은 Fold에서 각 모델을 순차 학습.
    for config in FOLDS:
        train_df, valid_df = make_fold(train, config)
        for experiment, removed in experiments.items():
            features = [f for f in BASELINE_FEATURES if f not in removed]
            print("\n" + "=" * 85)
            print(f"{experiment} | {config['fold']} | {len(features)} features")
            print("Removed:", removed)
            iteration, score, importance = train_experiment_model(
                train_df, valid_df, config["fold"], features
            )
            fold_rows.append({
                "experiment": experiment, "fold": config["fold"],
                "feature_count": len(features), "best_iteration": iteration,
                "rmsle": score,
            })
            importance.insert(0, "experiment", experiment)
            importance_parts.append(importance)
            # 장시간 실행 중에도 완료한 결과는 보존.
            if save_csv:
                pd.DataFrame(fold_rows).to_csv(
                    run_dir / "fold_results_partial.csv", index=False, encoding="utf-8-sig")
                pd.concat(importance_parts, ignore_index=True).to_csv(
                    run_dir / "per_fold_importance_partial.csv", index=False, encoding="utf-8-sig")
        del train_df, valid_df
        gc.collect()

    fold_results = pd.DataFrame(fold_rows)
    all_importance = pd.concat(importance_parts, ignore_index=True)
    baseline_scores = fold_results.loc[
        fold_results["experiment"].eq("baseline")].set_index("fold")["rmsle"]
    fold_results["baseline_rmsle"] = fold_results["fold"].map(baseline_scores)
    fold_results["delta_rmsle"] = fold_results["rmsle"] - fold_results["baseline_rmsle"]
    summary = fold_results.groupby("experiment", sort=False, as_index=False).agg(
        feature_count=("feature_count", "first"),
        mean_rmsle=("rmsle", "mean"), std_rmsle=("rmsle", "std"),
        mean_delta_rmsle=("delta_rmsle", "mean"),
    )
    means, comparisons = {}, {}
    for experiment, removed in experiments.items():
        per_fold = all_importance.loc[all_importance["experiment"].eq(experiment)]
        expected_count = len(BASELINE_FEATURES) - len(removed)
        if len(per_fold) != len(FOLDS) * expected_count:
            raise ValueError(f"{experiment}: 중요도 행 수가 예상과 다릅니다.")
        means[experiment] = summarize_importance(per_fold)
        print(f"\nALL FEATURES — {experiment} — 3-FOLD MEAN")
        print(means[experiment].round(4).to_string(index=False))
        if save_csv:
            means[experiment].to_csv(run_dir / f"{experiment}_mean_importance.csv",
                                     index=False, encoding="utf-8-sig")
        if experiment != "baseline":
            comparison = compare_importance(means["baseline"], means[experiment], removed)
            comparisons[experiment] = comparison
            display_cols = ["feature", "removed", "mean_gain_pct_baseline",
                            "mean_gain_pct_experiment", "delta_gain_pp",
                            "mean_split_pct_baseline", "mean_split_pct_experiment",
                            "delta_split_pp"]
            print(f"\nIMPORTANCE CHANGE — {experiment} (delta: percentage points)")
            print(comparison[display_cols].round(4).to_string(index=False))
            print("\nFOCUS: dayofweek / is_weekend")
            print(comparison.loc[comparison["feature"].isin(
                ["dayofweek", "is_weekend"]), display_cols].round(4).to_string(index=False))
            if save_csv:
                comparison.to_csv(run_dir / f"{experiment}_importance_comparison.csv",
                                  index=False, encoding="utf-8-sig")

    print("\nFOLD RESULTS (negative delta_rmsle = improvement)")
    print(fold_results.round(6).to_string(index=False))
    print("\nSCORE SUMMARY")
    print(summary.round(6).to_string(index=False))
    print("\nNOTE: Gain/Split percentages are relative; raw totals also depend on best_iteration.")
    if save_csv:
        for filename, frame in [
            ("fold_results.csv", fold_results), ("score_summary.csv", summary),
            ("per_fold_importance.csv", all_importance),
        ]:
            frame.to_csv(run_dir / filename, index=False, encoding="utf-8-sig")
        print("\nCSV 저장 위치:", run_dir)
    del train
    gc.collect()
    return fold_results, means, comparisons


if __name__ == "__main__":
    fold_results, mean_importances, importance_comparisons = run_backtesting()
