# Day10

## 오늘 한 일

Kaggle Store Sales - Time Series Forecasting **Calendar Feature 실험 (`dayofyear` vs sin/cos)**

- Day09에서 Public 성능이 가장 좋았던 **`Base23 + Lag 16·21·28`(26개)**을 새로운 메인 Baseline으로 사용
- 과거 판매량을 쓰는 연 단위 피처 대신, 날짜상의 연간 위치를 직접 표현하는 `dayofyear` 추가 효과를 3-Fold Validation으로 확인
- 일부 Fold가 `n_estimators=5000` 상한 근처까지 학습되는 것을 발견해 **6500으로 상향**, Early stopping 150 유지
- 연간 위치 표현 3가지(`dayofyear` / sin·cos / 병용)를 동일 조건으로 비교해 **`dayofyear` 단독 채택**
- 다음 실험(Payday 피처)을 설계하고, 확정된 Baseline은 재학습하지 않도록 실험 구조 수정

### 1. `dayofyear` 첫 실험 (`n_estimators=5000`)

기존 Baseline에 `dayofyear` 하나만 추가해 효과를 확인했다.

| 모델 | 피처 수 | Fold 1 | Fold 2 | Fold 3 | 평균 RMSLE |
|---|---:|---:|---:|---:|---:|
| Lag-only Baseline | 26 | 0.390832 | **0.399856** | 0.400570 | 0.397086 |
| **Lag + dayofyear** | 27 | **0.384091** | 0.405151 | **0.397418** | **0.395553** |

- 평균 0.001533 개선, 3개 Fold 중 2개 개선
- Fold 2는 오히려 악화되어, 이 시점에는 안정적인 피처라고 판단하기 부족했다.

### 2. `n_estimators=5000` 상한 문제

| Fold | Best Iteration (5000 실험) |
|---|---:|
| Fold 1 | 4,995 |
| Fold 2 | 3,012 |
| Fold 3 | 5,000 |

- Fold 1은 상한 직전, Fold 3은 상한까지 학습 → 더 개선될 여지가 있는데 iteration 제한 때문에 멈췄을 가능성
- 변경: `n_estimators` **5000 → 6500**, Early stopping **150 유지**
- 이후 모든 Calendar 표현에 같은 조건을 적용해 공정하게 비교

### 3. 연간 Calendar 표현 3가지 비교

| 모델 | 구성 | 피처 수 |
|---|---|---:|
| **A** | Base23 + Lag 16·21·28 + `dayofyear` | 27 |
| **B** | Base23 + Lag 16·21·28 + `dayofyear_sin` + `dayofyear_cos` | 28 |
| **C** | Base23 + Lag 16·21·28 + `dayofyear` + sin/cos | 29 |

동일한 3-Fold, `n_estimators=6500`, Early stopping 150, 동일한 LightGBM 파라미터로 비교했다.

| Fold | A. `dayofyear` | B. sin/cos | C. 병용 |
|---|---:|---:|---:|
| Fold 1 | 0.382917 | **0.380940** | 0.385058 |
| Fold 2 | **0.405151** | 0.411717 | 0.409489 |
| Fold 3 | 0.396157 | 0.401914 | **0.395554** |
| **평균** | **0.394742** | 0.398190 | 0.396700 |
| 표준편차 | **0.011185** | 0.015723 | 0.012256 |

- **A(`dayofyear` 단독)가 평균 0.394742로 최고**
- B는 A보다 0.003449, C는 0.001958 악화 (둘 다 A보다 나은 Fold는 1/3)

### 4. 6500 상향 효과

| Fold | 5000 상한 | 6500 상한 | 변화 | Best Iteration |
|---|---:|---:|---:|---:|
| Fold 1 | 0.384091 | **0.382917** | -0.001174 | 5,989 |
| Fold 2 | 0.405151 | 0.405151 | 0 | 3,012 |
| Fold 3 | 0.397418 | **0.396157** | -0.001261 | 5,509 |
| **평균** | 0.395553 | **0.394742** | **-0.000811** | |

- 상한에 걸렸던 Fold 1·3이 5000 이후에도 추가 개선 → 5000으로는 부족했음을 확인
- 두 Fold 모두 6500 이전에 Early stopping이 걸려, 현재 채택 모델에는 6500이 충분한 상한

### 5. Calendar Feature Importance

| 모델 | Feature | Mean Gain % | Mean Split % |
|---|---|---:|---:|
| A | `dayofyear` | **0.949** | **11.759** |
| B | `dayofyear_cos` | 0.598 | 6.369 |
| B | `dayofyear_sin` | 0.457 | 6.655 |
| C | `dayofyear` | 0.636 | 6.488 |
| C | `dayofyear_cos` | 0.498 | 4.481 |
| C | `dayofyear_sin` | 0.388 | 4.846 |

- `dayofyear` 단독일 때 Gain 비중이 가장 높았고, 세 피처를 함께 넣으면 중요도가 분산됐다.

### 6. 현재 채택 구성

```text
Base 23 + Lag 16·21·28 + dayofyear  (총 27개)

Local 3-Fold: Fold 1 = 0.382917 / Fold 2 = 0.405151 / Fold 3 = 0.396157
Mean = 0.394742, Std = 0.011185
```

제외: `dayofyear_sin`, `dayofyear_cos`, Rolling mean/std, EMA, Lag364, Seasonal Mean

### 7. 실험 구조 수정과 Payday 설계

- 이후 신규 피처 검증은 **기존 Champion을 다시 학습하지 않고**, 저장된 Fold 결과와 신규 모델만 비교
- Payday 피처 `is_15th`, `is_month_end`를 설계하고, 생성은 `feature_engineering.py`의 `add_date_features()`에서 하기로 결정

```python
df["is_15th"] = (df["date"].dt.day == 15).astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
```

---

## 판단 / 배운 점

### Calendar 위치 정보는 효과가 있었다
- Lag-only 0.397086 → `+ dayofyear` **0.394742**로 개선됐다.
- `dayofyear`는 현재 날짜가 1년 중 어디인지만 알려주므로, 오래된 판매 수준을 가져오는 연간 Lag와 달리 **장기적인 판매 규모 변화의 영향을 덜 받을 수 있다**고 본다. (Lag364와의 직접 비교는 이번 실험에 없었으므로 가설로 둔다.)
- 다만 Fold 2에서는 개선이 없었으므로 Fold별 일관성은 이후에도 계속 확인해야 한다.

### 주기형 변수라고 항상 sin/cos가 더 좋은 것은 아니다
- 12월 31일과 1월 1일이 가깝다는 점에서 sin/cos가 유리할 것이라 예상했지만, 이번 LightGBM에서는 `dayofyear`가 더 좋았다.
- 가능한 이유 (아직 검증하지 않은 **가설**):
  - Tree 모델은 `dayofyear <= 190`처럼 구간을 직접 분할할 수 있다. sin/cos는 같은 구간이 두 좌표로 나뉘어 더 복잡한 분할이 필요할 수 있다.
  - `dayofyear + sin/cos`는 같은 날짜 정보를 세 번 반복해 제공해 분할 선택이 분산됐을 수 있다.
- **검증 구간의 한계**: Fold 1~3은 06/29~08/15(대략 `dayofyear` 180~227)의 연속 구간이라 연말·연초를 포함하지 않는다. sin/cos의 핵심 장점인 **연말↔연초 연결 효과는 이번 검증으로 확인할 수 없다.** 이 결론은 현재 Validation·Test와 같은 여름 구간에 한정해서 받아들인다.
- 모델 종류에 따라 좋은 표현이 다르다. 선형 모델·Neural Network에서는 sin/cos가 유용할 수 있지만, Tree 모델에서는 원본 숫자가 더 쉽게 쓰일 수 있다. **이론적으로 좋아 보이는 표현을 자동으로 채택하지 않고 Validation 결과로 판단한다.**

### 같은 정보를 여러 방식으로 넣는다고 도움이 되지는 않는다
```text
정보량 증가 ≠ 성능 증가
```
- `dayofyear`, `sin`, `cos`는 모두 같은 날짜에서 파생되어, 모델에는 새로운 외부 정보가 추가된 것이 아니다.
- 중복 표현은 중요도를 분산시키고 분할을 복잡하게 만들 수 있다.

### iteration 상한도 실험 조건의 일부다
- Best Iteration이 상한에 붙으면 성능이 확정된 것이 아니라 **학습할 공간이 없었던 것**일 수 있다.
- 6500으로 늘리자 Fold 1·3이 실제로 추가 개선됐다.
- 앞으로 Best Iteration이 반복해서 상한에 붙으면 학습 제한이 충분한지부터 확인한다.

### 확정된 Baseline은 매번 재학습하지 않는다
```text
Champion 결과 저장 → 신규 Feature 모델만 3-Fold 학습 → 저장된 같은 Fold 점수와 비교
```
- 학습 시간과 실험 횟수를 줄일 수 있다.
- 단, **Fold·피처·파라미터·학습 조건이 바뀌면** 기존 결과와 직접 비교하면 안 된다. (이번처럼 `n_estimators`가 바뀐 경우 새 조건으로 다시 확정한 Champion을 기준으로 삼는다.)

### Feature 생성과 선택의 역할 분리
```text
feature_engineering.py → Feature 생성
modeling.py            → 이번 실험에서 사용할 Feature 선택
```
- 날짜에서 파생되는 `is_15th`, `is_month_end`는 `add_date_features()`에서 생성한다.
- DataFrame에 sin/cos 컬럼이 남아 있어도 모델 Feature 목록에서 빼면 학습에 쓰이지 않는다.

### 느낀 점
- Day09에서는 시계열 피처를 늘릴수록 Local 성능은 좋아졌지만 Public에서는 단순한 Lag 모델이 더 좋았다. 그래서 이번에는 Calendar 피처도 **하나씩 통제해서** 비교했다.
- `dayofyear`처럼 매우 단순한 피처가 복잡한 시계열 파생변수보다 안정적인 개선을 만든 점이 흥미로웠다.
- 날짜가 원형이라는 이유만으로 sin/cos가 더 좋을 것이라 생각했지만 실제로는 그렇지 않았다. **수학적으로 세련된 표현보다 모델이 그 정보를 얼마나 쉽게 쓰는지**도 중요하다.
- 처음에는 `n_estimators=5000`이면 충분할 것이라 생각했지만, Best Iteration을 확인하며 학습 상한 자체가 결과에 영향을 준다는 것을 알게 됐다.
- 앞으로는 점수뿐 아니라 Best Iteration, Fold별 일관성, Feature Importance를 함께 보고 판단한다.
- **한 줄 정리: Feature는 많이 만들거나 복잡하게 표현하는 것이 목적이 아니라, 모델이 몰랐던 정보를 가장 효율적으로 전달하는 방식이 무엇인지 검증하는 것이 중요하다.**

---

## 다음

1. **Payday Feature 실험**
   - 현재 Champion: Base23 + Lag 16·21·28 + `dayofyear` (27개, 평균 RMSLE **0.394742**)
   - 실험 모델: 기존 27개 + `is_15th`, `is_month_end` (29개)
   - `feature_engineering.py`의 `add_date_features()`에서 생성, `modeling.py`는 선택만 수행
2. **기존 Champion 재학습 생략**
   - 저장된 Fold 결과로 비교: Fold 1 = 0.382917 (Best Iter 5,989) / Fold 2 = 0.405151 (3,012) / Fold 3 = 0.396157 (5,509)
   - Payday 모델만 Fold당 1회, 총 3회 학습 → Fold별·평균 RMSLE 직접 비교
3. **Payday 효과 분해**
   - 먼저 `is_15th + is_month_end` 묶음 비교
   - 개선되면 `is_15th` 단독, `is_month_end` 단독 Ablation으로 어느 피처가 기여했는지 확인
4. **Group Statistics 검토**
   - 기존 범주형과 중복이 적은 상호작용 통계부터 (우선 후보: `family × dayofweek` 과거 평균 판매량)
   - Target 기반 통계이므로 **Fold Train 데이터만** 사용해 Validation Leakage 방지
5. **External Feature 준비**
   - `oil.csv`, `transactions.csv`: 실제 Test 시점에 이용 가능한 정보인지 먼저 확인
   - 미래 값을 직접 쓰는 Leakage가 없도록 과거 Lag / Rolling 활용 여부 검토
6. **experiments.md 갱신**
   - `dayofyear` 첫 실험, 5000 → 6500 변경 이유, sin/cos 비교 결과, `dayofyear` 채택
   - 현재 메인 Local 기준: **Base23 + Lag16·21·28 + dayofyear, 3-Fold RMSLE 0.394742**