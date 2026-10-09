# Day05

## 오늘 한 일

Kaggle Store Sales - Time Series Forecasting **LightGBM Baseline 학습 및 첫 제출 파일 생성**

### 1. Feature Engineering 마무리 및 검증

- `merge_holiday_features()` 코드 분석 완료
  - `groupby("date").agg()`로 날짜별 Holiday 집계
  - `lambda`, `set()`, `sorted()`, `"|".join()`으로 복수 이벤트 문자열 결합
  - `holiday_count`는 독립적인 공휴일 수가 아니라 **날짜별 이벤트 기록 행 수**임을 확인
  - `merge(on="date", how="left", validate="many_to_one")`, `fillna(0)`, `astype(int)`, `assert` 검증 이해
- National 이벤트 그룹 추가: `Good Friday`(성금요일), `Labor Day`(노동절)
  - 이진 피처 `event_good_friday`, `event_labor_day` 생성, 이벤트 그룹 총 9개(Other 포함)
  - Regional / Local은 아직 제외

### 2. 전체 파이프라인 QC

`main.py`로 전처리부터 Holiday 병합까지 실행해 검증했다.

| 항목 | 결과 |
| --- | --- |
| Train / Validation / Test | 2,979,504행 × 29열 / 28,512행 × 29열 / 28,512행 × 27열 |
| Holidays | 338행 × 7열 |
| 누락 날짜 · 결측치 · 중복 키 | 모두 0개 |
| 복원된 크리스마스 행 | 7,128개 |
| 병합 전후 행 수 | 동일 |

특이사항:

- `event_city_foundation`은 Train에서 전부 0
- `event_other`에는 여러 종류의 이벤트가 혼합
- **Test 구간에는 National 이벤트가 없어 National Holiday 피처가 전부 0**
- `Work Day`도 설명 문자열에 따라 Christmas 등의 그룹으로 분류될 수 있음

검증 후 `main.py`의 불필요한 출력은 주석 처리하고 핵심 QC 로직은 유지했다.

### 3. 모델 평가 설계

- **지표**: RMSLE (실제값·예측값에 `log1p` 적용 후 RMSE)
- **Target**: `sales_log = log1p(sales)`로 학습하고, 평가 시 `np.expm1(np.maximum(pred_log, 0))`으로 복원 (음수 예측은 0으로 보정)
- **Validation**: Train `2013-01-01 ~ 2017-07-30` / Validation `2017-07-31 ~ 2017-08-15` / Test `2017-08-16 ~ 2017-08-31` (실제와 같은 16일)
- **입력·Target 분리**: `X_train`, `X_valid` / `y_train`, `y_valid`(로그) / `y_valid_sales`(RMSLE 계산용 원본)
  - `id`, `date`, `sales`, `sales_log`, `is_added_missing_date`는 입력에서 제외
- **누수 검토**: 현재는 Lag/Rolling이 없어 문제 없음. 추가할 때 16일 예측 기간의 실제 판매량을 참조하지 않도록 주의

### 4. LightGBM Baseline 구현

파일 구성:

| 파일 | 역할 |
| --- | --- |
| `modeling.py` | 피처 선택, 모델 설정, 학습, 예측, Feature Importance |
| `evaluation.py` | 로그 역변환, 음수 보정, RMSLE 평가 |
| `main.py` | 전처리·피처 엔지니어링 파이프라인 실행 |

입력 피처 19개:

| 구분 | 피처 |
| --- | --- |
| 매장·상품군 | `store_nbr`, `family` (범주형 지정) |
| 날짜 | `year`, `month`, `day`, `dayofweek`, `is_weekend` |
| 프로모션 | `onpromotion`, `onpromotion_log`, `is_promotion` |
| Holiday | `has_national_event` + 이벤트별 0/1 피처 8개 |

- 제외: `holiday_count`, Holiday 문자열 범주형 피처, `event_city_foundation`
- Train과 Validation의 범주 기준을 맞추기 위해 `pd.Categorical(..., categories=X_train[col].cat.categories)` 사용

주요 하이퍼파라미터: `gbdt`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `random_state=42`, `importance_type="split"` (나머지는 기본값)

### 5. 첫 실험 결과

| 항목 | Experiment 01 | Experiment 02 |
| --- | --: | --: |
| `n_estimators` | 1,000 | 1,300 |
| Best Iteration | 1,000 | 1,300 |
| Validation RMSE | 0.446283 | 0.440323 |
| **Validation RMSLE** | **0.446059** | **0.440020** |
| Early Stopping | 미발동 | 미발동 |

반복 횟수를 늘리자 RMSLE가 0.006039 감소했다(약 1.35% 개선).

Feature Importance (Experiment 02, 상위 7개 + 특이점):

| 순위 | Feature | Split |
| --: | --- | --: |
| 1 | `store_nbr` | 13,438 |
| 2 | `family` | 12,430 |
| 3 | `year` | 4,518 |
| 4 | `month` | 3,663 |
| 5 | `day` | 1,507 |
| 6 | `onpromotion` | 1,461 |
| 7 | `dayofweek` | 1,301 |
| 8 | `event_christmas` | 149 |
| 18 | `is_weekend` | 22 |
| 19 | `onpromotion_log` | **0** |

### 6. Kaggle 제출 파일 생성

`create_submission()` 추가: Test 피처 선택 → 범주형 기준 통일 → 로그 예측 → `expm1` 역변환·음수 보정 → `sample_submission.csv` ID 순서로 병합 → 검증 후 `submission.csv` 저장 (`index=False`)

- 28,512행 × 2열(`id`, `sales`), LightGBM 1,300 Iterations, Validation RMSLE 0.440020
- Validation 이전 기간으로 학습한 모델의 결과이며, **Train + Validation 전체 재학습은 아직 진행하지 않았다.** Public Score도 제출 후 확인해야 한다.

### 7. Python 실행 환경 정리

- `.venv`에는 LightGBM이 설치됐지만 VS Code 실행 환경과 달라 Import 오류 발생 (`WindowsApps\python.exe` 실행 별칭 문제 확인)
- 인터프리터와 라이브러리 설치 환경이 일치해야 함을 이해, 로컬 Python 3.13에 LightGBM 4.7.0 설치 후 실행 성공
- 앞으로 이 프로젝트는 **로컬 Python 3.13 사용**, `requirements.txt`로 환경 재현

---

## 판단 / 배운 점

### 모델 학습

- Gradient Boosting은 여러 트리를 **순차적으로** 학습해 손실을 줄인다. LightGBM은 손실 감소가 가장 큰 Leaf를 우선 분할하는 Leaf-wise 방식이다 (XGBoost의 Level-wise와 다름).
- `n_estimators`는 최대 반복 횟수이고 Best Iteration과 항상 같지 않다. Early Stopping(50)은 Validation 성능이 50회 연속 개선되지 않으면 중단한다.
- 하이퍼파라미터 효과는 **동일한 Validation 구간에서** 비교해야 한다.
- 두 실험 모두 Early Stopping이 발동하지 않았으므로 1,300회가 최선이라고 단정할 수 없고, 계속 늘리는 것도 답이 아닐 수 있다.

### Feature Importance 해석

- `store_nbr`, `family`가 가장 많이 쓰였고 날짜·프로모션 피처가 뒤를 이었다. Holiday 피처는 사용 빈도가 낮았다.
- `onpromotion_log`는 중요도 0이다. `onpromotion` 원본과 정보가 겹쳐 트리 모델에서는 분할 기준으로 선택될 필요가 없었던 것으로 보인다.
- **Split Importance는 사용 빈도일 뿐 성능 기여도의 증명이 아니다.** 제거 전후 RMSLE를 직접 비교해야 한다.
- Test 구간에는 National 이벤트가 없으므로, 현재 Holiday 피처는 Train 패턴 설명에는 쓰여도 **Test 예측에는 거의 작동하지 않을 가능성**이 있다. Test의 유일한 휴일인 2017-08-24 Ambato(Local) 매칭이 다음 후보다.

### 시계열 모델링

- 현재 Baseline은 과거 판매량을 정답으로만 쓰고 입력으로는 쓰지 않는다. **날짜 피처만으로는 시간 흐름을 완전히 표현할 수 없다.**
- 시계열에서는 예측 시점에 실제로 알 수 있는 데이터만 사용해야 하고, Validation은 실제 예측 구간과 같게 잡아야 한다.

### 프로젝트 운영

- 전처리·피처 엔지니어링·모델·평가 로직을 파일별로 분리하면 관리가 쉽다.
- 반복 QC 출력은 줄이되 핵심 검증은 유지한다.
- 첫 성능을 확보한 뒤 **피처와 하이퍼파라미터를 하나씩 바꿔 비교**하는 방식이 효과적이다.

### 느낀 점

- Day04까지는 데이터를 정리하고 피처를 설계하는 작업이었다면, Day05에는 그 결과를 **실제 모델 성능(RMSLE 0.440020)과 제출 파일로 연결**했다.
- 피처를 만들 때는 효과가 추상적이었는데, 중요도와 점수를 보니 어떤 피처가 쓰이고 어떤 피처가 안 쓰이는지 구체적으로 보였다.
- 환경 문제로 시간을 썼지만, 인터프리터와 설치 환경이 일치해야 한다는 점을 정리한 것은 이후에도 도움이 될 것 같다.
- 한 줄 정리: **EDA·전처리·피처 엔지니어링을 마무리하고 LightGBM Baseline으로 Validation RMSLE 0.440020과 첫 제출 파일을 확보한 날이다. 이제부터는 과거 판매량을 누수 없이 활용하는 시계열 피처 개선이 핵심이다.**

## 다음

1. Kaggle에 `submission.csv` 제출 후 Public Score 기록
2. 피처 효과 비교
   - `onpromotion_log` 제거 전후 RMSLE
   - Holiday 피처 포함 여부에 따른 RMSLE
3. 시계열 피처 설계
   - 16일 예측 구간에서 누수 없이 쓸 수 있는 Lag / Rolling
   - 매장·상품군별 과거 판매량 통계
4. 모델 개선
   - 반복 횟수와 트리 복잡도 튜닝
   - 여러 구간으로 나누는 Rolling Validation 검토
5. 최종 제출: Train + Validation 전체로 재학습 후 제출
