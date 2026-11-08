# Day22

## 오늘 한 일

Kaggle Store Sales - Time Series Forecasting **Trial 17 · Oil Raw Optuna 모델의 33개 품목별 성능 비교, Fold · Backtest 분석, Oil Raw / Trial 17 품목별 조합 제출과 Public 결과 비교, 유가 피처 상관계수 비교 및 Oil EMA7 후속 실험 설계**

- 기존 **Trial 17(Base23 + Lag364·28, 25개 피처)** 과 **Oil Raw Optuna(Base23 + Lag364·28 + Oil Raw, 26개 피처)** 의 전체 Fold 1·2·3 및 2016 Backtest 결과를 비교했다.
- 전체 33개 품목의 Fold별 RMSLE를 확보하고 **Fold 2에서 30개 개선, Fold 3에서 28개 악화**되는 상반된 결과를 확인했다.
- Fold 2·3 전체 평균은 **0.396644 → 0.395944**, Backtest는 **0.747735 → 0.719572**로 개선됐다.
- 기존 Trial 17의 Backtest 품목별 점수를 확보하고, 최신 Oil Raw Optuna와 비교해 **33개 중 26개 품목 개선, 7개 품목 악화**를 확인했다.
- Fold 3와 Backtest에서 동시에 성능이 악화된 5개 품목을 추출했다.
- 기존 Oil Raw · MA28 · EMA16을 선택적으로 사용한 8개 품목과 최신 모델을 비교해 **8개 중 7개 품목에서 개선**을 확인했다.
- 기존 3모델 제출 코드의 피처 구성과 품목별 예측 선택 방식을 재확인했다.
- 두 모델의 예측값을 품목별로 조합해 제출 파일을 만들었다. 기존 Kaggle 제출(**Oil Raw 9개 + Trial 17 24개**)에서 BEAUTY를 Trial 17로 옮겨 **Oil Raw 8개 + Trial 17 25개** 최종 제출 파일을 생성했다. (28,512행, 샘플 제출 파일의 ID 순서와 품목별 원본 예측값 대조)
- **GROCERY II**는 Backtest 개선에도 Fold 3 악화가 컸기 때문에 Fold 3을 일 · 매장 단위로 분해해 분석하고 Trial 17로 확정했다.
- Oil Raw, Rolling 7·16, EMA 7·16 유가 피처의 **전날 대비 판매량 변화와의 상관계수**를 비교했다. (EMA7 실험의 동기)
- Kaggle Public 제출 5개의 **조합(품목 배정)과 점수**를 정리했다. 확인한 최저 점수는 **0.42416**이었다.
- 후속 실험으로 **Oil Raw → Oil EMA7 교체**를 설계했다. 기존 25개 피처와 Oil Raw 전용 Optuna 최적 파라미터를 유지하고 Oil 입력만 변경하도록 했다.
- 실험 코드의 모듈 import 오류를 확인하고, 기존 `main.py`와 `feature_engineering.py`를 직접 사용하는 독립형 코드로 수정했다.
- `--run-dir`에 실제 결과 폴더를 지정해 **Oil Raw Optuna 29번 설정이 로드되고 EMA7이 모델 입력에 포함되는 것**까지 확인했다. 이후 **프로젝트 폴더 접근 권한 오류**가 발생해 EMA7의 Fold 2·3 및 Backtest 성능은 **아직 확인되지 않았다.**

### 오늘의 핵심 결과 한눈에 보기

| 항목 | Trial 17 | Oil Raw Optuna | Δ |
|---|---:|---:|---:|
| Fold 2·3 평균 RMSLE | 0.396644 | **0.395944** | −0.000700 |
| 2016 Backtest RMSLE | 0.747735 | **0.719572** | −0.028162 |
| Backtest 개선 품목 | — | **26/33** (악화 7) | — |

| Kaggle Public | 조합 (Oil Raw / Trial 17 품목 수) | 점수 |
|---|---|---:|
| 최저 (1위) | Oil Raw 9 / Trial 17 24 | **0.42416** |
| 오늘 만든 최종 파일 (2위) | Oil Raw 8 / Trial 17 25 | 0.42431 |

---

## 1. 공통 Base23 및 실험별 피처 구성 정리

현재 비교하는 모델들은 **Base23 + Lag364 + Lag28** 을 공통 기반으로 사용한다.

### Base23 구성

| 분류 | 피처 | 개수 |
|---|---|---:|
| Store / Product | `store_nbr`, `family`, `city`, `state`, `type`, `cluster` | 6 |
| Date | `year`, `month`, `day`, `dayofweek`, `is_weekend` | 5 |
| Promotion | `onpromotion`, `is_promotion` | 2 |
| Holiday | `holiday_count`, `event_christmas`, `event_new_year`, `event_carnival`, `event_mothers_day`, `event_independence_day`, `event_city_foundation`, `event_good_friday`, `event_labor_day`, `event_other` | 10 |
| **합계** | **Base23** | **23** |

### 추가되는 Lag 2개

```python
sales_lag_364
sales_lag_28
```

| Lag | 의미 |
|---|---|
| `Lag364` | 52주(364일) 전 동일 요일의 판매량 (365일이 아님) |
| `Lag28` | 4주 전 동일 요일의 판매량 |

### 비교 대상 / 후속 실험 피처 구성

| 모델 | 피처 구성 | 개수 |
|---|---|---:|
| A. Trial 17 | Base23 + Lag364 + Lag28 | 25 |
| B. Oil Raw | A + `oil_raw` | 26 |
| C. Oil MA28 | A + `oil_ma_28` | 26 |
| D. Oil EMA16 | A + `oil_ema_16` | 26 |
| E. Oil EMA7 | A + `oil_ema_7` | 26 |

### Oil 피처 정의

Oil Raw는 기존 전처리된 `dcoilwtico`를 사용하며, 실험 코드에 따라 입력 컬럼명을 `oil_raw`로 지정한다. 나머지 Oil 피처도 **동일한 전처리 Oil 시계열**에서 파생한다.

| 피처 | 정의 |
|---|---|
| Oil Raw | 해당 날짜의 전처리된 유가 |
| Oil MA28 | 최근 28일의 단순이동평균 |
| Oil EMA16 | 최근 관측값에 높은 가중치를 두는 span 16 지수이동평균 |
| Oil EMA7 | span 7 지수이동평균 |

> 모델 C · D · E는 각 Oil 피처를 **Oil Raw와 동시에 넣는 것이 아니라 대체하는 구성**으로 정의했다.

> 기존 Trial 17과 Oil Raw Optuna는 **파라미터까지 동일한 모델이 아니다.** Trial 17은 기존 베이스라인 Optuna 결과를, 최신 Oil Raw 모델은 별도로 수행한 30-Trial Optuna의 최적 파라미터를 사용한다.

---

## 2. Trial 17과 Oil Raw Optuna의 전체 성능 비교

오늘의 주요 비교 실험은 다음 두 모델이다. 전체 33개 품목을 통합 학습하는 LightGBM 모델이며, Target은 `log1p(sales)`다.

| 구분 | 모델 A — Trial 17 | 모델 B — 최신 Oil Raw Optuna |
|---|---|---|
| 피처 | Base23 + `sales_lag_364` + `sales_lag_28` | Base23 + `sales_lag_364` + `sales_lag_28` + `oil_raw` |
| 피처 수 | 25 | 26 |
| Oil 피처 | 없음 | Oil Raw |
| 파라미터 | 기존 Trial 17 Optuna | Oil Raw 전용 30-Trial Optuna 최적값 |

### 구간별 RMSLE

RMSLE는 낮을수록 좋으며 Δ는 `최신 − 기존`이다. (품목별로 섞기 전, 두 모델을 각각 평가한 값)

| 평가 구간 | Trial 17 | Oil Raw Optuna | Δ | 개선 품목 |
|---|---:|---:|---:|---:|
| Fold 1 | 0.378966 | **0.377529** | −0.001437 | 22/33 |
| Fold 2 | 0.399300 | **0.390602** | −0.008698 | 30/33 |
| Fold 3 | **0.393988** | 0.401286 | +0.007298 | 5/33 |
| Fold 2·3 평균 | 0.396644 | **0.395944** | −0.000700 | 11/33 |
| Backtest | 0.747735 | **0.719572** | −0.028162 | 26/33 |

### 확인한 결과

- Fold 1은 0.001437 개선됐다.
- Fold 2는 0.008698 개선됐고 30개 품목에서 오차가 감소했다.
- Fold 3은 0.007298 악화됐으며 **28개 품목**에서 오차가 증가했다.
- Fold 2·3 전체 평균은 0.000700 개선됐다.
- 2016 Backtest는 약 0.028162 개선됐고 33개 중 26개 품목이 개선됐다. (Δ는 반올림 전 점수로 계산)

특히 **Fold 2와 Fold 3에서 개선 방향이 크게 달랐다.** Fold 2에서는 거의 모든 품목이 개선된 반면, 실제 Test 직전 기간을 평가하는 Fold 3에서는 대부분 악화됐다.

> 다만 **Oil Raw 추가와 Optuna 파라미터 변경이 함께 반영된 비교**이므로, 이 결과를 Oil Raw 피처 하나만의 효과로 해석할 수는 없다.

---

## 3. 33개 품목별 Fold 1·2·3 RMSLE 및 Fold 2·3 평균 변화

Fold별 점수는 **기존(Trial 17) → 최신(Oil Raw Optuna)** 순서다. ★는 기존 Oil 피처별 선택 모델 비교 대상 8개 품목이다. 표는 Fold 2·3 평균 Δ(음수일수록 개선) 기준으로 개선이 큰 순서로 정렬했다.

| 품목 | Fold 1 | Fold 2 | Fold 3 | Fold 2·3 평균 Δ | 최신 Backtest |
|---|---|---|---|---:|---:|
| ★ GROCERY II | 0.5877 → 0.5792 | 0.7627 → 0.6777 | 0.5544 → 0.5795 | **−0.029963** | 0.6999 |
| ★ SCHOOL AND OFFICE SUPPLIES | 0.4512 → 0.4475 | 0.5925 → 0.5859 | 0.7085 → 0.6810 | −0.016994 | 0.5947 |
| ★ HOME APPLIANCES | 0.4151 → 0.4214 | 0.4086 → 0.4058 | 0.3693 → 0.3515 | −0.010285 | 0.3546 |
| ★ GROCERY I | 0.1210 → 0.1191 | 0.1327 → 0.1248 | 0.1497 → 0.1393 | −0.009165 | 1.1636 |
| BEAUTY | 0.4729 → 0.4785 | 0.5422 → 0.5206 | 0.4873 → 0.4907 | −0.009086 | 0.4713 |
| ★ BEVERAGES | 0.1915 → 0.1978 | 0.2202 → 0.2020 | 0.2206 → 0.2228 | −0.007973 | 1.1336 |
| LIQUOR,WINE,BEER | 0.5139 → 0.5020 | 0.4912 → 0.4767 | 0.4354 → 0.4401 | −0.004925 | 0.8405 |
| ★ BABY CARE | 0.3038 → 0.3035 | 0.2746 → 0.2712 | 0.2908 → 0.2873 | −0.003450 | 0.3436 |
| ★ LINGERIE | 0.6111 → 0.6125 | 0.6352 → 0.6309 | 0.6168 → 0.6170 | −0.002108 | 0.6943 |
| BOOKS | 0.1293 → 0.1289 | 0.1198 → 0.1187 | 0.0969 → 0.0960 | −0.001018 | 0.0272 |
| ★ HARDWARE | 0.5152 → 0.5125 | 0.5388 → 0.5360 | 0.5250 → 0.5261 | −0.000867 | 0.5477 |
| PERSONAL CARE | 0.1944 → 0.1932 | 0.2373 → 0.2309 | 0.1799 → 0.1880 | +0.000883 | 0.7506 |
| PLAYERS AND ELECTRONICS | 0.4800 → 0.4718 | 0.4576 → 0.4519 | 0.4682 → 0.4761 | +0.001064 | 0.5385 |
| LADIESWEAR | 0.4700 → 0.4714 | 0.4784 → 0.4757 | 0.5051 → 0.5102 | +0.001186 | 0.5061 |
| HOME CARE | 0.1754 → 0.1703 | 0.2161 → 0.2104 | 0.2004 → 0.2097 | +0.001765 | 0.7802 |
| CLEANING | 0.2955 → 0.2897 | 0.3134 → 0.3053 | 0.2505 → 0.2625 | +0.001950 | 0.9595 |
| AUTOMOTIVE | 0.4987 → 0.5028 | 0.4787 → 0.4776 | 0.5009 → 0.5063 | +0.002171 | 0.5405 |
| HOME AND KITCHEN I | 0.4552 → 0.4418 | 0.4378 → 0.4334 | 0.4875 → 0.4963 | +0.002186 | 0.6181 |
| SEAFOOD | 0.4602 → 0.4592 | 0.4779 → 0.4745 | 0.4918 → 0.5006 | +0.002703 | 0.5135 |
| LAWN AND GARDEN | 0.4014 → 0.4020 | 0.3981 → 0.3976 | 0.4350 → 0.4413 | +0.002906 | 0.5013 |
| HOME AND KITCHEN II | 0.3994 → 0.3948 | 0.4472 → 0.4402 | 0.4414 → 0.4544 | +0.003011 | 0.6013 |
| PET SUPPLIES | 0.4583 → 0.4589 | 0.4749 → 0.4768 | 0.4671 → 0.4721 | +0.003501 | 0.4935 |
| DAIRY | 0.1230 → 0.1160 | 0.1308 → 0.1304 | 0.1279 → 0.1359 | +0.003796 | 0.9737 |
| CELEBRATION | 0.5444 → 0.5409 | 0.5423 → 0.5424 | 0.5332 → 0.5425 | +0.004734 | 0.5951 |
| MAGAZINES | 0.4997 → 0.5150 | 0.4793 → 0.4824 | 0.5067 → 0.5154 | +0.005887 | 0.5443 |
| PREPARED FOODS | 0.2752 → 0.2687 | 0.2728 → 0.2698 | 0.2849 → 0.3014 | +0.006721 | 0.6488 |
| PRODUCE | 0.1295 → 0.1311 | 0.1379 → 0.1347 | 0.1279 → 0.1455 | +0.007186 | 1.1237 |
| BREAD/BAKERY | 0.1518 → 0.1496 | 0.1548 → 0.1500 | 0.1817 → 0.2009 | +0.007209 | 0.9482 |
| DELI | 0.1571 → 0.1527 | 0.1603 → 0.1601 | 0.1733 → 0.1890 | +0.007726 | 0.7903 |
| EGGS | 0.2931 → 0.2953 | 0.2939 → 0.2906 | 0.2995 → 0.3193 | +0.008243 | 0.8067 |
| FROZEN FOODS | 0.2644 → 0.2610 | 0.2574 → 0.2562 | 0.2630 → 0.2834 | +0.009587 | 0.6901 |
| POULTRY | 0.1869 → 0.1850 | 0.1889 → 0.1842 | 0.2294 → 0.2913 | +0.028637 | 0.8079 |
| MEATS | 0.1826 → 0.1775 | 0.1786 → 0.1765 | 0.2383 → 0.3035 | **+0.031498** | 0.7941 |

### 요약

- Fold 2·3 평균에서는 **11개 품목이 개선, 22개가 악화**됐다.
- Fold 3에서는 **28개 품목이 악화**됐다.
- 가장 크게 개선된 품목: GROCERY II → SCHOOL AND OFFICE SUPPLIES → HOME APPLIANCES
- 가장 크게 악화된 품목: MEATS, POULTRY (Fold 3에서 각각 0.2383 → 0.3035, 0.2294 → 0.2913)
- 그런데 **MEATS와 POULTRY는 Backtest에서는 오히려 개선**됐다. (4장 참고)

→ 동일한 모델이라도 검증 시점에 따라 품목별 성능 평가가 달라질 수 있음을 보여준다.

---

## 4. 2016 Backtest 비교 (33개 품목 전체)

### 사용 모델의 피처 구성

- 기존: Base23 + Lag364 + Lag28 (25개)
- 최신: Base23 + Lag364 + Lag28 + Oil Raw (26개)

### Backtest 기간

```text
Train cutoff : 2016-08-15
Validation   : 2016-08-16 ~ 2016-08-31
Horizon      : 16일
```

### 전체 결과

| 모델 | Backtest RMSLE |
|---|---:|
| Trial 17 | 0.747735 |
| Oil Raw Optuna | **0.719572** |
| 변화량 | **−0.028162** |

품목별 결과는 **33개 중 26개 개선, 7개 악화**다. Δ는 최신 − 기존이며 음수는 개선이다. (변화량은 반올림 전 점수로 계산)

### 품목별 Backtest RMSLE

| 품목 | 기존 Trial 17 | 최신 Oil Raw | Δ | 판정 |
|---|---:|---:|---:|---|
| AUTOMOTIVE | 0.537649 | 0.540538 | +0.002889 | 악화 |
| BABY CARE | 0.344101 | 0.343614 | −0.000487 | 개선 |
| BEAUTY | 0.458597 | 0.471326 | +0.012729 | 악화 |
| BEVERAGES | 1.224469 | 1.133592 | **−0.090878** | 개선 |
| BOOKS | 0.050844 | 0.027189 | −0.023654 | 개선 |
| BREAD/BAKERY | 0.988273 | 0.948171 | −0.040102 | 개선 |
| CELEBRATION | 0.601217 | 0.595099 | −0.006118 | 개선 |
| CLEANING | 1.016645 | 0.959498 | −0.057147 | 개선 |
| DAIRY | 1.014463 | 0.973742 | −0.040721 | 개선 |
| DELI | 0.843046 | 0.790318 | −0.052727 | 개선 |
| EGGS | 0.844023 | 0.806734 | −0.037288 | 개선 |
| FROZEN FOODS | 0.702580 | 0.690057 | −0.012523 | 개선 |
| GROCERY I | 1.241111 | 1.163648 | −0.077462 | 개선 |
| GROCERY II | 0.758477 | 0.699860 | −0.058617 | 개선 |
| HARDWARE | 0.549902 | 0.547670 | −0.002231 | 개선 |
| HOME AND KITCHEN I | 0.629332 | 0.618094 | −0.011238 | 개선 |
| HOME AND KITCHEN II | 0.611716 | 0.601251 | −0.010466 | 개선 |
| HOME APPLIANCES | 0.352997 | 0.354581 | +0.001584 | 악화 |
| HOME CARE | 0.810442 | 0.780227 | −0.030215 | 개선 |
| LADIESWEAR | 0.509369 | 0.506115 | −0.003253 | 개선 |
| LAWN AND GARDEN | 0.500516 | 0.501250 | +0.000734 | 악화 |
| LINGERIE | 0.695254 | 0.694289 | −0.000965 | 개선 |
| LIQUOR,WINE,BEER | 0.857722 | 0.840483 | −0.017239 | 개선 |
| MAGAZINES | 0.521464 | 0.544341 | +0.022877 | 악화 |
| MEATS | 0.831868 | 0.794132 | −0.037736 | 개선 |
| PERSONAL CARE | 0.772921 | 0.750566 | −0.022355 | 개선 |
| PET SUPPLIES | 0.490481 | 0.493547 | +0.003066 | 악화 |
| PLAYERS AND ELECTRONICS | 0.551602 | 0.538470 | −0.013131 | 개선 |
| POULTRY | 0.856425 | 0.807858 | −0.048567 | 개선 |
| PREPARED FOODS | 0.661479 | 0.648804 | −0.012675 | 개선 |
| PRODUCE | 1.187887 | 1.123676 | −0.064211 | 개선 |
| SCHOOL AND OFFICE SUPPLIES | 0.594127 | 0.594703 | +0.000576 | 악화 |
| SEAFOOD | 0.522751 | 0.513499 | −0.009253 | 개선 |

### 요약

- 가장 크게 개선된 품목: BEVERAGES(−0.090878), GROCERY I(−0.077462), PRODUCE(−0.064211), GROCERY II(−0.058617), CLEANING(−0.057147), DELI(−0.052727), POULTRY(−0.048567)
- 악화된 7개 품목: MAGAZINES(+0.022877), BEAUTY(+0.012729), PET SUPPLIES(+0.003066), AUTOMOTIVE(+0.002889), HOME APPLIANCES(+0.001584), LAWN AND GARDEN(+0.000734), SCHOOL AND OFFICE SUPPLIES(+0.000576)

> 다만 2016 Backtest는 2017년 Test와 다른 연도의 데이터이고, 최신 파라미터는 2017년 Fold 결과를 이용해 선택됐으므로 **모든 기간에서 일반화 성능이 좋다고 확정할 수는 없다.**

### 결과 출처 (프로젝트 내 파일)

| 구분 | 파일 |
|---|---|
| 기존 Trial 17 Backtest 품목별 | `backtest_comparison.csv` (`baseline_rmsle` 열) |
| 기존 Trial 17 Backtest 전체 · 설정 | `trial17_fixed_backtest.json` |
| 최신 Oil Raw Backtest 품목별 | `backtest_scores.csv` |
| Fold별 품목 오차 | `trial17_family_error_ranking.csv` |

---

## 5. Fold 3와 Backtest에서 동시에 악화된 품목

같은 두 피처 구성을 사용하는 Trial 17과 Oil Raw Optuna를 비교한 결과, Fold 3와 2016 Backtest에서 **공통으로 악화된 품목은 5개**였다.

| 품목 | Fold 3 기존 → 최신 | Backtest 기존 → 최신 | Backtest Δ |
|---|---|---|---:|
| MAGAZINES | 0.5067 → 0.5154 | 0.521464 → 0.544341 | +0.022877 |
| BEAUTY | 0.4873 → 0.4907 | 0.458597 → 0.471326 | +0.012729 |
| PET SUPPLIES | 0.4671 → 0.4721 | 0.490481 → 0.493547 | +0.003066 |
| AUTOMOTIVE | 0.5009 → 0.5063 | 0.537649 → 0.540538 | +0.002889 |
| LAWN AND GARDEN | 0.4350 → 0.4413 | 0.500516 → 0.501250 | +0.000734 |

- 특히 **MAGAZINES와 BEAUTY**에서 상대적으로 큰 악화가 나타났다.
- 이 품목들은 후속 Oil EMA7 실험에서 다시 확인할 대상으로 분류했다.

---

## 6. 기존 Oil 피처별 선택 모델 8개와 최신 Oil Raw Optuna 비교

기존에는 일부 품목에 서로 다른 Oil 표현을 사용하는 모델을 선택했다.

### 각 모델의 피처 구성

| 선택 모델 | 실제 피처 구성 | 개수 |
|---|---|---:|
| Raw | Base23 + Lag364 + Lag28 + Oil Raw | 26 |
| MA28 | Base23 + Lag364 + Lag28 + Oil MA28 | 26 |
| EMA16 | Base23 + Lag364 + Lag28 + Oil EMA16 | 26 |
| 최신 Optuna | Base23 + Lag364 + Lag28 + Oil Raw | 26 |

기존 Raw · MA28 · EMA16 비교 모델은 **Trial 17 기반 파라미터**를 사용했으며, 최신 모델은 **Oil Raw 전용 Optuna 최적 파라미터**를 사용했다.

### 품목별 Fold 2·3 평균 비교

| 품목 | 기존 선택 모델 | 기존 Fold 2·3 평균 | 최신 Fold 2·3 평균 | 최신 − 선택 |
|---|---|---:|---:|---:|
| BABY CARE | Raw | 0.282574 | **0.279254** | −0.003320 |
| BEVERAGES | Raw | 0.215651 | **0.212434** | −0.003217 |
| GROCERY I | Raw | 0.137857 | **0.132040** | −0.005817 |
| GROCERY II | Raw | 0.634710 | **0.628606** | −0.006104 |
| HARDWARE | Raw | **0.530615** | 0.531047 | +0.000432 |
| LINGERIE | MA28 | 0.625076 | **0.623915** | −0.001161 |
| SCHOOL AND OFFICE SUPPLIES | MA28 | 0.638021 | **0.633481** | −0.004540 |
| HOME APPLIANCES | EMA16 | 0.380498 | **0.378640** | −0.001858 |

**최신 Oil Raw Optuna 모델이 8개 중 7개 품목에서 개선됐다.** HARDWARE만 0.000432 악화됐으며 나머지는 모두 기존 선택 모델보다 낮은 Fold 2·3 평균 RMSLE를 기록했다.

> 이 비교는 Oil 입력 표현과 파라미터 구성이 함께 다른 모델들의 결과 비교다. 따라서 Raw · MA28 · EMA16의 **순수한 피처 효과 비교와는 구분**해야 한다.

> 이 장의 "기존 선택 모델"(Raw · MA28 · EMA16)은 Oil 표현별 선택 구성이고, 8장의 Kaggle 제출 조합(Oil Raw / Trial 17 두 모델의 품목별 배정)과는 별개다.

---

## 7. 기존 3모델 학습 및 품목별 예측 선택 구조 (이전 제출 구조)

이전의 3모델 Kaggle 제출 코드는 서로 다른 피처 구성의 모델 3개를 학습하고, 실제 제출 단계에서 품목별로 예측값을 선택하는 구조였다.

### 모델별 학습 피처

```text
Model A — Baseline
Base23 + Lag364 + Lag28
= 25개

Model B — Oil Raw
Base23 + Lag364 + Lag28 + Oil Raw
= 26개

Model C — Oil EMA16
Base23 + Lag364 + Lag28 + Oil EMA16
= 26개
```

3개 모델 모두 전체 33개 품목을 학습하고 전체 Test를 예측한다. 이후 최종 제출 시 품목별로 사용할 모델의 예측값을 선택한다.

### 품목별 라우팅 (기존 3모델 제출 코드 기준)

| 예측 선택 모델 | 적용 품목 | 품목 수 |
|---|---|---:|
| Oil Raw | BEVERAGES, GROCERY I, HARDWARE | 3 |
| Oil EMA16 | HOME APPLIANCES, SCHOOL AND OFFICE SUPPLIES | 2 |
| Baseline | 나머지 전체 품목 | 28 |
| **합계** | **전체 33개 품목** | **33** |

### 전체 흐름

이 구조는 특정 품목만 따로 학습하는 전용 모델과 다르다.

```text
전체 33개 품목 학습 × 3모델
            ↓
각 모델이 전체 Test 예측
            ↓
품목별 모델 선택
            ↓
최종 제출 파일 생성
```

> 6장의 8개 품목 선택 모델 비교는 별도로 기록된 선택 구성이고, 여기서 확인한 3모델 제출 코드의 3·2·28개 라우팅과 **동일한 구성이라고 단정하지 않는다.**
> 오늘의 Kaggle 제출(8장)은 이 3모델 구조가 아니라 **Trial 17과 Oil Raw 두 모델의 예측값을 품목별로 조합**한 파일들이다.

---

## 8. Kaggle Public 제출 결과와 조합

### 5개 제출 결과

Public 점수가 낮은 순서다. 각 조합에서 Oil Raw를 적용하지 않은 품목은 **Trial 17 예측값**을 사용했다.

| 순위 | 제출 파일 | Oil Raw / Trial 17 품목 수 | Public RMSLE | 1위와 차이 |
|---:|---|---|---:|---:|
| 1 | `submission_trial17_oilraw9_20261107_175108_220011.csv` | 9 / 24 | **0.42416** | — |
| 2 | `submission_oilraw8_trial17_25_groceryii_trial17.csv` | 8 / 25 | 0.42431 | +0.00015 |
| 3 | `submission_oilraw7_trial17_26.csv` | 7 / 26 | 0.42591 | +0.00175 |
| 4 | `submission_oilraw16_trial17_17.csv` | 16 / 17 | 0.42670 | +0.00254 |
| 5 | `submission_oilraw12_trial17_21.csv` | 12 / 21 | 0.42710 | +0.00294 |

- 최저 Public 점수: **0.42416** (Oil Raw 9개 + Trial 17 24개)
- 최저와 최고 오차 제출 차이: 0.00294
- **Private 점수는 확인하지 않았다.**

### 조합별 Oil Raw 적용 품목

| 조합 | Oil Raw 적용 품목 |
|---|---|
| 9개 | BABY CARE, BEAUTY, BEVERAGES, BOOKS, GROCERY I, HARDWARE, HOME APPLIANCES, LIQUOR,WINE,BEER, SCHOOL AND OFFICE SUPPLIES |
| 8개 | BABY CARE, BEVERAGES, BOOKS, GROCERY I, HARDWARE, HOME APPLIANCES, LIQUOR,WINE,BEER, SCHOOL AND OFFICE SUPPLIES |
| 7개 | BABY CARE, BEVERAGES, BOOKS, GROCERY I, GROCERY II, HOME APPLIANCES, LIQUOR,WINE,BEER |
| 16개 | BABY CARE, BEVERAGES, BOOKS, BREAD/BAKERY, CLEANING, DELI, EGGS, GROCERY I, GROCERY II, HARDWARE, HOME AND KITCHEN II, HOME APPLIANCES, LINGERIE, LIQUOR,WINE,BEER, PRODUCE, SCHOOL AND OFFICE SUPPLIES |
| 12개 | BABY CARE, BOOKS, BREAD/BAKERY, CLEANING, DELI, EGGS, GROCERY I, GROCERY II, HOME AND KITCHEN II, HOME APPLIANCES, PRODUCE, SCHOOL AND OFFICE SUPPLIES |

### 품목별 Oil Raw 적용 여부 한눈에 보기

○ = Oil Raw 적용, 빈칸 = Trial 17. 5개 조합 중 한 번이라도 Oil Raw가 적용된 17개 품목만 표시했고, 나머지 16개 품목은 모든 조합에서 Trial 17이다.

| 품목 | 9개 (0.42416) | 8개 (0.42431) | 7개 (0.42591) | 16개 (0.42670) | 12개 (0.42710) |
|---|:---:|:---:|:---:|:---:|:---:|
| BABY CARE | ○ | ○ | ○ | ○ | ○ |
| BEAUTY | ○ | | | | |
| BEVERAGES | ○ | ○ | ○ | ○ | |
| BOOKS | ○ | ○ | ○ | ○ | ○ |
| BREAD/BAKERY | | | | ○ | ○ |
| CLEANING | | | | ○ | ○ |
| DELI | | | | ○ | ○ |
| EGGS | | | | ○ | ○ |
| GROCERY I | ○ | ○ | ○ | ○ | ○ |
| GROCERY II | | | ○ | ○ | ○ |
| HARDWARE | ○ | ○ | | ○ | |
| HOME AND KITCHEN II | | | | ○ | ○ |
| HOME APPLIANCES | ○ | ○ | ○ | ○ | ○ |
| LINGERIE | | | | ○ | |
| LIQUOR,WINE,BEER | ○ | ○ | ○ | ○ | |
| PRODUCE | | | | ○ | ○ |
| SCHOOL AND OFFICE SUPPLIES | ○ | ○ | | ○ | ○ |

### 조합 간 비교에서 확인한 점

- **9개 vs 8개:** 품목 배정 차이는 **BEAUTY 한 개**(9개는 Oil Raw, 8개는 Trial 17)다. 다만 9개 파일은 **두 모델을 다시 학습해 만든 기록**이고, 8개 파일은 **저장된 두 제출 CSV를 합쳐 만든 것**이다. 따라서 0.00015 차이를 BEAUTY 배정 효과로만 해석하려면, 두 파일에서 다른 품목의 예측값이 동일한지 먼저 확인해야 한다.
- **7개 vs 8개:** 두 조합은 **같은 원본 예측값**을 사용했다. 7개에서 8개로 바꾸면서 GROCERY II는 Trial 17로, HARDWARE와 SCHOOL AND OFFICE SUPPLIES는 Oil Raw로 옮겼고, 이 **세 변경을 합친 결과** Public 점수가 0.42591 → 0.42431로 개선됐다. (세 변경의 개별 효과는 분리되지 않는다.)
- **16개 · 12개:** Oil Raw 적용 품목이 가장 많거나(16) 중간(12)이면서 Public 점수가 가장 나빴던 두 조합이다. 두 조합에는 BREAD/BAKERY, CLEANING, DELI, EGGS, PRODUCE, HOME AND KITCHEN II처럼 **Fold 2·3 평균에서는 악화(+0.002~+0.008)되고 Backtest에서는 개선**됐던 품목이 포함돼 있다. 이는 표에서 확인되는 사실이며, Public 점수 하락의 원인이라는 해석은 **아직 검증하지 않은 가설**이다.

---

## 9. 최종 제출 파일 구성과 품목별 배정

### 최종 제출 파일

```text
submission_oilraw8_trial17_25_groceryii_trial17.csv
```

| 항목 | 내용 |
|---|---|
| 조합 | **Oil Raw 8개 + Trial 17 25개** |
| 기준 | 기존 Kaggle 제출(Oil Raw 9 + Trial 17 24)에서 **BEAUTY를 Trial 17로 이동** |
| GROCERY II | Fold 3 분석 결과를 반영해 **Trial 17로 확정** |
| 행 수 | 28,512행 |
| 검증 | 샘플 제출 파일의 ID 순서와 품목별 원본 예측값을 대조 |
| Public 점수 | 0.42431 (5개 제출 중 2위) |

Oil Raw를 적용한 8개 품목은 BABY CARE, BEVERAGES, BOOKS, GROCERY I, HARDWARE, HOME APPLIANCES, LIQUOR,WINE,BEER, SCHOOL AND OFFICE SUPPLIES이며, 나머지 25개 품목은 Trial 17 예측값이다.

### 주요 10개 품목의 검증 결과와 최종 배정

Δ는 Oil Raw − Trial 17이며, 음수는 개선이다.

| 품목 | Fold 2·3 평균 Δ | Fold 3 Δ¹ | Backtest Δ | 최종 모델 |
|---|---:|---:|---:|---|
| BABY CARE | −0.003450 | −0.0035 | −0.000487 | Oil Raw |
| BEAUTY | −0.009086 | +0.0034 | +0.012729 | **Trial 17** |
| BEVERAGES | −0.007973 | +0.0022 | −0.090878 | Oil Raw |
| BOOKS | −0.001018 | −0.0009 | −0.023654 | Oil Raw |
| GROCERY I | −0.009165 | −0.0104 | −0.077462 | Oil Raw |
| GROCERY II | −0.029963 | +0.025040 | −0.058617 | **Trial 17** |
| HARDWARE | −0.000867 | +0.0011 | −0.002231 | Oil Raw |
| HOME APPLIANCES | −0.010285 | −0.0178 | +0.001584 | Oil Raw |
| LIQUOR,WINE,BEER | −0.004925 | +0.0047 | −0.017239 | Oil Raw |
| SCHOOL AND OFFICE SUPPLIES | −0.016994 | −0.0275 | +0.000576 | Oil Raw |

¹ GROCERY II는 원본 예측으로 다시 계산한 값이다. 나머지 Fold 3 변화량은 소수점 4자리로 표시된 점수에서 구한 근삿값이다.

### GROCERY II 판단 (Trial 17 확정)

GROCERY II는 Fold 2·3 평균(−0.029963)과 Backtest(−0.058617)에서는 크게 개선됐지만 **Fold 3에서는 0.025040 악화**됐다.

| 확인 항목 | 결과 |
|---|---|
| Fold 3 악화 일수 | 16일 중 **12일** |
| Fold 3 악화 매장 수 | 54개 매장 중 **37곳** |
| 악화가 가장 큰 3개 매장의 비중 | 제곱 로그오차 순증가분의 **63.7%** |
| 이 3개 매장을 제외한 뒤 남는 비중 | **36.3%** |

- 특정 매장 몇 곳의 이상치로만 설명되지 않고, 3개 매장을 빼도 순증가분의 36.3%가 남았다.
- Backtest 개선 폭도 컸지만, 최종 배정에서는 **최근 검증 구간(Fold 3)의 안정성을 우선**해 Trial 17을 적용했다.

---

## 10. 유가 피처별 상관계수 비교 (Oil EMA7 실험의 동기)

### 비교 방법

- 같은 **1,664일**에 대해 **전날 대비 판매량 변화**와 **전날 대비 유가 지표 변화**의 피어슨 상관계수를 계산했다.
- 비교한 유가 지표: Oil Raw, Rolling 7, Rolling 16, EMA 7, EMA 16

### 품목별 상관계수

| 품목 | Oil Raw | Rolling 7 | Rolling 16 | EMA 7 | EMA 16 |
|---|---:|---:|---:|---:|---:|
| BABY CARE | +0.0535 | +0.0181 | +0.0052 | +0.0204 | +0.0127 |
| BEAUTY | −0.0049 | +0.0120 | +0.0035 | +0.0116 | +0.0100 |
| BEVERAGES | +0.0060 | +0.0144 | +0.0071 | +0.0152 | +0.0135 |
| BOOKS | −0.0074 | +0.0046 | +0.0170 | +0.0078 | +0.0124 |
| GROCERY I | −0.0081 | +0.0136 | +0.0040 | +0.0083 | +0.0072 |
| GROCERY II | +0.0186 | +0.0140 | +0.0007 | +0.0254 | +0.0144 |
| HARDWARE | +0.0063 | +0.0217 | +0.0136 | +0.0222 | +0.0180 |
| HOME APPLIANCES | +0.0118 | +0.0089 | +0.0042 | +0.0103 | +0.0049 |
| LIQUOR,WINE,BEER | +0.0225 | +0.0033 | +0.0046 | +0.0179 | +0.0102 |
| SCHOOL AND OFFICE SUPPLIES | +0.0061 | +0.0098 | +0.0070 | +0.0123 | +0.0160 |

### 결과 요약 (기존 Kaggle Oil Raw 9개 품목 기준)

| 비교 | 결과 |
|---|---|
| 상관계수 절댓값이 더 큰 쪽 | **EMA 7: 6개 품목**, Oil Raw: 3개 품목 |
| 절댓값 평균 | EMA 7 **0.0140**, Oil Raw **0.0141** |

- 모든 상관계수가 **0에 가까웠다.** (표의 최댓값은 BABY CARE Oil Raw의 +0.0535)
- EMA 7이 더 큰 품목: BEAUTY, BEVERAGES, BOOKS, GROCERY I, HARDWARE, SCHOOL AND OFFICE SUPPLIES
- Oil Raw가 더 큰 품목: BABY CARE, HOME APPLIANCES, LIQUOR,WINE,BEER
- 절댓값 평균은 사실상 같았다. (차이 0.0001)

> 이 상관계수는 일 변화량끼리의 선형 관계이며, **모델 성능(RMSLE) 개선을 직접 검증한 결과가 아니다.** EMA7을 실험할 동기로는 사용하되, 채택 여부는 Fold · Backtest 결과로 판단한다.

---

## 11. Oil EMA7 후속 실험 설계

Fold별 성능 차이와 위 상관계수 비교를 확인한 뒤, Oil Raw 대신 단기적인 유가 변화의 평활값을 사용하는 실험을 설계했다.

### 모델 비교

**기존 모델**

```text
Base23 + Lag364 + Lag28 + Oil Raw
= 26개 피처

파라미터:
Oil Raw 전용 30-Trial Optuna 최적값 (29번 설정)
```

**새로운 실험 모델**

```text
Base23 + Lag364 + Lag28 + Oil EMA7
= 26개 피처

파라미터:
동일한 Oil Raw Optuna 최적값 (29번 설정)
```

### 변경 / 유지 항목

| 항목 | Oil Raw Optuna | 신규 Oil EMA7 |
|---|---|---|
| Base23 | 유지 | 유지 |
| Lag364 | 유지 | 유지 |
| Lag28 | 유지 | 유지 |
| Oil Raw | 포함 | **제외** |
| Oil EMA7 | 없음 | **추가** |
| 피처 수 | 26 | 26 |
| 파라미터 | Oil Raw Optuna | 동일 |
| Fold 2·3 | 완료 | 예정 (미확인) |
| Backtest | 완료 | 예정 (미확인) |

### Oil EMA7 생성

Oil EMA7은 기존 전처리 파이프라인에서 결측 보완된 `dcoilwtico`를 이용한다.

```python
daily["oil_ema_7"] = (
    daily["dcoilwtico"]
    .ewm(span=7, adjust=False, min_periods=7)
    .mean()
)
```

```text
α = 2 / (7 + 1) = 0.25
```

EMA7은 최근 관측값에 더 높은 가중치를 부여하고 과거 관측값의 영향도 점진적으로 반영한다.

### 실험 조건

- Oil Raw와 EMA7을 동시에 넣는 것이 아니라 **Oil Raw 피처를 EMA7으로 교체**하는 실험이다.
- 검증은 Fold 2, Fold 3, 2016 Backtest를 사용하고, 모든 33개 품목을 하나의 모델로 학습한다.
- Early Stopping 및 Optuna 재실행은 사용하지 않으며, 기존 Oil Raw Optuna 결과에 저장된 학습 횟수를 적용하도록 설계했다.

### 진행 상태

| 단계 | 상태 |
|---|---|
| 실험 코드 작성 (독립형) | 완료 |
| `--run-dir`에 실제 결과 폴더 지정 | 완료 |
| Oil Raw Optuna 29번 설정 로드 | **확인** |
| EMA7이 모델 입력에 포함되는지 | **확인** |
| 학습 진행 중 프로젝트 폴더 접근 | **권한 오류 발생** |
| EMA7 Fold 2·3 · Backtest RMSLE | **미확인** |

---

## 12. 코드 오류 및 파이프라인 수정

### 발생한 오류 ① — 모듈 import 오류

초기 코드에서는 다음 모듈을 불러오도록 작성됐다.

```python
import trial17_oil_raw_optuna_full as original
```

실행 시 다음 오류가 발생했다.

```text
ModuleNotFoundError:
No module named 'trial17_oil_raw_optuna_full'
```

실행 환경에서 해당 모듈을 찾지 못했기 때문이다.

### 수정 방향

기존 프로젝트의 실제 전처리 · 피처 엔지니어링 모듈을 직접 불러오는 구조로 수정했다.

```python
from main import main as original_pipeline
from feature_engineering import add_lag_features
```

### 수정한 코드의 주요 내용

- 기존 전처리 및 Lag 생성 로직 사용
- Base23 + Lag364 + Lag28 유지
- Oil EMA7만 모델 입력에 추가
- 전체 33개 품목 학습
- Fold 2·3, Backtest 수행
- Oil Raw 전용 Optuna 최적 파라미터 재사용
- 기존 결과 파일에서 Fold별 iteration 재사용
- 예측값, 품목별 RMSLE, Feature Importance, 모델 저장
- Optuna 최적화 재실행 없음

파라미터 출처도 기존 Trial 17 베이스라인이 아니라 **11월 7일 Oil Raw 30-Trial Optuna 결과**로 수정했다.

### 발생한 오류 ② — 프로젝트 폴더 접근 권한 오류

- `--run-dir`에 실제 결과 폴더를 지정하자 Oil Raw Optuna 29번 설정이 로드되고 EMA7이 모델 입력에 포함되는 것까지는 확인됐다.
- 그 이후 확인 환경에서 **프로젝트 폴더 접근 권한 오류**가 발생했다.

> 따라서 **신규 EMA7 모델의 성능 수치는 아직 기록하지 않는다.**

---

## 판단 / 배운 점

### 평균 RMSLE와 품목별 성능은 구분해야 한다

- Fold 2·3 전체 RMSLE는 개선됐지만, 품목별 Fold 평균에서는 33개 중 11개만 개선됐다.
- 전체 RMSLE와 품목별 점수의 산술평균은 서로 다른 계산이므로, 전체 점수 개선이 모든 품목의 개선을 뜻하지 않는다.
- 앞으로는 전체 RMSLE뿐 아니라 **품목별 개선 수, 악화 폭, 기간별 변화**를 함께 확인해야 한다.

### 시간적 일반화 성능은 하나의 Fold로 판단할 수 없다

- Oil Raw Optuna는 Fold 2에서 30개 품목을 개선했지만 Fold 3에서는 28개 품목이 악화됐다.
- 같은 모델이더라도 평가 시기에 따라 성능 방향이 크게 달라졌다.
- 가능한 이유로는 계절성, 행사 구성, 판매 패턴 및 Oil 변수의 영향 차이 등이 있지만 **아직 원인 검증은 이루어지지 않았다.**
- Fold 2·3 평균만으로 모델을 선택하면 Fold 3의 악화를 충분히 반영하지 못할 수 있다고 판단했다.

### Backtest와 Fold 3는 서로 다른 정보를 준다

Fold 3는 실제 Test에 시간적으로 가깝고, 2016 Backtest는 실제 Test와 월 · 일 구간이 같다.

```text
Fold 3:     2017-07-31 ~ 2017-08-15
Backtest:   2016-08-16 ~ 2016-08-31
Test:       2017-08-16 ~ 2017-08-31
```

- 두 지표가 서로 다른 결과를 보여준다는 점은 모델의 일반화 성능에 **불확실성이 존재한다는 증거**다.
- 하나의 평가 구간을 절대적인 기준으로 사용하기보다는 평가 목적과 한계를 구분해 활용해야 한다.
- GROCERY II는 이 불일치가 가장 뚜렷하게 나타난 품목이었다. (Backtest −0.058617, Fold 3 +0.025040)

### 구간별 결과가 엇갈릴 때의 배정 기준을 먼저 정해야 한다

- 오늘 최종 배정에서 **GROCERY II만 Fold 3 악화 폭(+0.025040)이 크다는 이유로 Trial 17**이 됐다. BEAUTY도 Trial 17이며, Fold 3와 Backtest가 모두 악화된 품목이다.
- 반면 BEVERAGES(+0.0022), HARDWARE(+0.0011), LIQUOR,WINE,BEER(+0.0047)는 Fold 3에서 악화됐지만 Backtest가 개선돼 Oil Raw로 배정됐다.
- 즉 **"Fold 3 악화 여부"만이 아니라 악화 폭과 안정성(일 · 매장 단위 분포)을 함께 본 판단**이었다. 이 기준이 문서로 정리되어 있지 않으면 다음 실험에서 같은 기준으로 판단했는지 확인하기 어렵다.
- 앞으로는 품목별 배정 전에 **Fold 3 악화 폭, Backtest 변화, 악화의 분포(일 · 매장)** 를 어떻게 반영할지 기준을 먼저 적어 두기로 했다.

### 전체 모델의 성능 개선이 복잡한 품목별 라우팅보다 유리할 수도 있다

- 기존에는 품목마다 Raw · MA28 · EMA16을 선택적으로 적용했다.
- 그러나 오늘 비교한 8개 품목 중 7개에서 최신 Oil Raw Optuna 모델의 성능이 더 좋았다.
- 이는 복잡하게 모델을 나누는 것이 항상 더 좋은 결과를 만드는 것은 아니라는 점을 보여준다.
- 다만 최신 모델은 피처뿐 아니라 파라미터까지 별도로 최적화됐으므로, **성능 개선이 라우팅 제거 때문이라고 확정할 수는 없다.**

### 제출 조합의 Public 점수는 비교 조건이 같을 때만 해석할 수 있다

- 7개 → 8개 조합은 같은 원본 예측값을 쓰지만, **세 품목(GROCERY II, HARDWARE, SCHOOL AND OFFICE SUPPLIES)이 동시에 바뀌어** 개별 효과를 분리할 수 없다.
- 9개 vs 8개(BEAUTY 한 개 차이)는 **재학습 기록 vs 저장된 CSV 병합**이라는 생성 방식이 달라, 0.00015 차이를 BEAUTY 배정 효과로 단정할 수 없다.
- Public 점수 차이가 작을수록(0.00015) 배정 효과와 다른 조건의 차이를 구분하기 어렵다.
- 앞으로는 **한 번에 한 품목(또는 한 가지 요인)만 바꾸고, 같은 방식으로 생성한 파일끼리 비교**해야 한다.

### 상관계수는 실험의 동기일 뿐, 성능 개선의 근거는 아니다

- Oil Raw, Rolling, EMA의 상관계수는 모두 0에 가까웠고, 9개 품목 절댓값 평균은 EMA 7 0.0140, Oil Raw 0.0141로 사실상 같았다.
- EMA 7이 6개 품목에서 더 컸다는 점은 실험해 볼 이유가 되지만, 이 값만으로 EMA7이 더 낫다고 말할 수는 없다.
- 일 변화량끼리의 선형 상관과 LightGBM의 RMSLE 개선은 다른 문제이므로 **Fold · Backtest로 검증**해야 한다.

### 피처 구성과 파라미터 출처를 함께 기록해야 한다

이번 비교에서는 다음 두 실험을 명확히 구분할 필요가 있었다.

```text
실험 A
Base23 + Lag364 + Lag28
기존 Trial 17 Optuna 파라미터

실험 B
Base23 + Lag364 + Lag28 + Oil Raw
Oil Raw 전용 Optuna 파라미터
```

- 피처와 파라미터가 동시에 달라진 비교에서는 어느 요인이 실제 성능을 개선했는지 분리하기 어렵다.
- 앞으로 피처 효과를 검증할 때는 **학습 조건과 파라미터를 가능한 한 동일하게 유지하고 하나의 피처군만 변경**해야겠다고 판단했다.
- 이번 Oil EMA7 실험은 이러한 원칙을 반영해 파라미터를 고정한 상태에서 Oil 표현만 변경하도록 설계했다.

### 같은 원본 데이터를 어떻게 표현하느냐도 실험 변수다

Oil Raw, MA28, EMA16, EMA7은 모두 동일한 Oil 데이터에서 파생된다.

```text
같은 원본 Oil 데이터
    ├─ Raw
    ├─ MA28
    ├─ EMA16
    └─ EMA7
```

- 새로운 외부 데이터를 추가하지 않아도 표현 방식에 따라 모델이 활용하는 정보의 시간적 특성이 달라진다.
- Raw는 특정 날짜의 값을, MA와 EMA는 여러 날짜의 변화 흐름을 반영한다.
- 그러나 평활화가 반드시 좋은 것은 아니다. 단기적인 변화를 완화하는 대신 실제로 유용한 급격한 변화를 약화시킬 수도 있다.
- 따라서 **EMA7도 이론적인 장점만으로 채택하지 않고 Fold · Backtest 결과로 검증**하기로 했다.

### 모델 실험의 재현성과 코드 관리가 중요하다

- 오늘 발생한 모듈 import 오류, 파라미터 경로 문제, 프로젝트 폴더 접근 권한 오류를 통해 모델의 피처 · 파라미터뿐 아니라 **실행 코드의 의존성과 실행 환경 관리**도 중요하다는 점을 확인했다.
- 실험을 반복할수록 코드와 설정 파일이 많아지고, 어떤 실험에서 어떤 파라미터와 피처를 사용했는지 혼동하기 쉬워진다.
- 앞으로는 모델별 설정을 다음과 같이 구분해 기록할 필요가 있다.

```text
Experiment
├─ Base Features
├─ Additional Features
├─ Hyperparameter Source
├─ Fold Configuration
├─ Training Iterations
├─ Prediction Routing
└─ Results / Artifacts
```

특히 같은 Oil Raw 피처를 사용하더라도 기존 Trial 17 파라미터를 사용하는 모델과 Oil Raw Optuna 파라미터를 사용하는 모델은 **별도의 실험으로 관리**해야 한다. 제출 파일도 **생성 방식(재학습 / CSV 병합)과 품목 배정을 함께 기록**해야 한다.

### 느낀 점

- 이전에는 새로운 피처를 추가하거나 모델을 여러 개 학습하는 과정에 집중했다면, 오늘은 **기존에 확보한 결과를 품목별 · 기간별로 다시 분석하는 데 집중**했다.
- Fold 2에서 거의 모든 품목이 개선됐는데 Fold 3에서는 반대로 대부분 악화된 점이 인상적이었다. 평균 RMSLE 하나만으로 모델 성능을 판단하기 어렵다는 것을 실감했다.
- Backtest에서는 다시 개선되는 결과가 나타나면서 어떤 평가 구간을 더 신뢰해야 하는지 판단하기 어려웠다. 하지만 이러한 불일치 자체가 중요한 분석 대상이라는 것을 알게 됐다.
- GROCERY II를 일 · 매장 단위로 나눠 보니, 3개 매장을 빼도 순증가분의 36.3%가 남아 "일부 매장의 문제"로 넘기기 어렵다는 것을 확인했다. 평균 점수 뒤에 있는 분포를 직접 확인하는 것이 판단에 도움이 됐다.
- 품목별로 Raw · MA28 · EMA16을 구분해 사용한 모델보다 하나의 최신 Oil Raw Optuna 모델이 더 좋은 결과를 낸 품목이 많다는 점도 의미 있었다. 복잡한 모델 구성이 반드시 성능을 높여주는 것은 아니었다.
- 기존 Trial 17과 Oil Raw Optuna의 비교에서는 피처와 파라미터가 모두 달라졌다는 점을 다시 확인했다. 앞으로는 실험 목적에 따라 어떤 변수를 고정하고 무엇을 변경하는지 먼저 명확히 해야겠다고 생각했다.
- Public 점수가 5개 제출 사이에서 0.00294 안에서만 움직였고, 어떤 변경이 효과를 냈는지는 조건이 섞여 있어 분리하기 어려웠다. 제출 파일을 만들 때 비교 조건까지 설계해야 한다는 점을 느꼈다.
- 여러 개의 실험 결과가 쌓일수록 코드를 작성하는 능력뿐 아니라 **실험을 재현하고 결과를 올바르게 비교하는 능력**이 중요해지는 것 같다.
- 다음 EMA7 실험은 기존 모델을 무작정 다시 튜닝하지 않고, Oil 표현만 변경해 효과를 확인하도록 설계했다.

**한 줄 정리: 모델 성능을 올리는 것만큼 어떤 피처와 파라미터가 사용됐고, 어느 기간과 품목에서 개선됐는지, 그리고 어떤 조건에서 만든 제출인지 정확히 구분하는 것이 중요하다.**

---

## 다음

1. **Oil EMA7 후속 실험 실행**
   - 프로젝트 폴더 접근 권한 오류 원인 확인 및 해결
   - Base23 + Lag364 + Lag28 + Oil EMA7 (26개)
   - Oil Raw 모델 입력 제외
   - 기존 전처리된 `dcoilwtico` 사용
   - Oil Raw 전용 Optuna 29번 설정(최적 파라미터) 고정
   - Fold 2, Fold 3, 2016 Backtest 학습

2. **Oil Raw vs Oil EMA7 통제 비교**
   - 두 모델 모두 Base23 + Lag364 + Lag28 유지
   - Oil Raw와 EMA7 입력 표현만 변경
   - 파라미터 및 Fold별 iteration 동일하게 유지
   - 전체 RMSLE 및 33개 품목별 RMSLE 비교
   - 상관계수에서 EMA 7이 더 컸던 6개 품목(BEAUTY, BEVERAGES, BOOKS, GROCERY I, HARDWARE, SCHOOL AND OFFICE SUPPLIES)에서 실제 RMSLE도 개선되는지 확인
   - Fold 3 악화 품목과 Backtest 개선 여부 확인

3. **반복 악화 품목 분석**
   - MAGAZINES, BEAUTY, PET SUPPLIES, AUTOMOTIVE, LAWN AND GARDEN
   - EMA7에서도 악화가 반복되는지 확인
   - Fold별 오차 패턴을 비교하고 원인 가설 수립

4. **품목별 모델 배정 기준 정리**
   - Fold 3 악화 폭, Backtest 변화, 악화의 분포(일 · 매장)를 어떻게 반영할지 기준 문서화
   - 현재 최종 배정(Oil Raw 8개 + Trial 17 25개)을 기준으로 한 Oil Raw 전체 모델과의 성능 비교
   - 기존 Baseline · Raw · EMA16 · MA28 모델의 피처 구성 정리
   - 품목별 모델을 유지할 필요가 있는지, 라우팅 방식의 실제 전체 RMSLE 개선 여부 검증

5. **Public 제출 결과와 재현성 점검**
   - 9개 제출 파일과 8개 제출 파일에서 **BEAUTY 외 품목의 예측값이 동일한지** 확인 (0.00015 차이의 해석 전제)
   - 제출 파일별 생성 방식(재학습 / CSV 병합) 기록
   - Private 점수 확인
   - `best_params.json` 경로와 파라미터 출처 확인
   - 학습 코드 · 피처 목록 · 결과 저장 파일 연결

6. **실험 문서 갱신**
   - Trial 17과 Oil Raw Optuna의 정확한 피처 구성
   - MA28 · EMA16 및 품목별 선택 모델 구성
   - 전체 Fold · Backtest 점수와 품목별 개선 수 (Backtest 26개 개선 · 7개 악화)
   - Fold 3 · Backtest 동시 악화 5개 품목
   - Kaggle Public 제출 5개의 조합과 점수, 최종 제출 파일 구성(Oil Raw 8 + Trial 17 25)
   - GROCERY II Fold 3 분석 결과와 Trial 17 확정 근거
   - 유가 피처 상관계수 비교 결과
   - Oil EMA7 실험 설계 및 실제 결과가 확보되면 추가 반영