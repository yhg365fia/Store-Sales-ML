**# Day09**

**## 오늘 한 일**

\- Kaggle Store Sales - Time Series Forecasting 시계열 피처 추가 검증 및 Kaggle Public 제출 결과 확인  
  - Day08의 기준 모델인 **Lag 16·21·28 + Rolling mean 7·14·28, Early stopping 150**에서 불필요할 가능성이 높은 피처를 제거하며 추가 실험 진행  
  - 중요도가 매우 낮았던 `Rolling mean 14`를 제거하고 성능 변화 확인  
  - `Rolling mean 7·28`만 유지한 구성이 기존 최고 성능을 갱신하는지 3-Fold 백테스트  
  - 장기 판매 수준을 나타내는 `EMA 364`를 Early stopping 150 조건에서 다시 검증  
  - 기존 Rolling mean을 EMA로 대체하는 구성도 실험해 Rolling과 EMA의 차이 확인  
  - 연 단위 반복 패턴을 직접 반영하기 위해 `Lag 364`와 과거 1·2·3년 값을 평균한 Seasonal Mean 실험  
  - Local Validation에서 최종적으로 **Lag 16·21·28 + Rolling mean 7·28** 조합이 평균 RMSLE **0.391424**로 최고 기록  
  - Local 최고 모델을 전체 Train으로 재학습해 Kaggle Public에 제출  
  - 3-Fold 중앙 Best Iteration인 4,335를 사용한 제출과 Fold 3 상한에 가까운 5,000 iteration 제출을 각각 수행  
  - 두 제출 모두 기존 Lag-only 제출보다 Public 성능이 크게 악화되는 것을 확인  
  - Local 최고 모델과 Public 최고 모델이 서로 다르다는 점을 확인하고 이후 실험 기준을 재정의  
  - 시계열 피처 실험은 여기서 종료하고 다음 단계인 Calendar / Event 피처 실험으로 넘어가기로 결정

**### 추가 시계열 실험 결과**

Day08의 `Lag 16·21·28 + Rolling mean 7·14·28 + ES150` 평균 RMSLE **0.392595**를 기준으로 추가 실험했다.

| 실험 | 시계열 피처 구성 | 평균 RMSLE | 기준 대비 변화 | 판단 |
|---|---|---:|---:|---|
| Day08 기준 | Lag 16·21·28 + Mean 7·14·28 | 0.392595 | — | 기준 |
| **Mean 14 제거** | **Lag 16·21·28 + Mean 7·28** | **0.391424** | **-0.001171** | **Local 최고** |
| EMA 364 추가 | Lag 3개 + Mean 7·28 + EMA 364 | 0.393733 | +0.002309 | 제외 |
| Rolling → EMA | Lag 3개 + EMA 7·28 | 0.395297 | +0.003873 | 제외 |
| Lag 364 추가 | Lag 16·21·28·364 + Mean 7·28 | 0.392009 | +0.000585 | 제외 |
| Seasonal Mean | Lag 3개 + Mean 7·28 + Seasonal Mean | 0.392427 | +0.001003 | 제외 |

**### Rolling mean 14 제거 실험**

\- Day08 최종 모델에서 Gain 비중이 약 0.1%에 불과했던 `sales_rolling_mean_16_14`만 제거  
\- 나머지 Lag, Rolling, 모델 파라미터, Fold, Early stopping 150은 유지  
\- 전체 피처 수: 29개 → **28개**

| Fold | Validation 기간 | RMSLE | Best Iteration |
|---|---|---:|---:|
| Fold 1 | 06/29~07/14 | **0.385976** | 1,136 |
| Fold 2 | 07/15~07/30 | 0.393941 | 4,335 |
| Fold 3 | 07/31~08/15 | 0.394357 | **4,995** |
| **평균** | — | **0.391424** | — |
| 표준편차 | — | 0.004723 | — |

\- 기존 Mean 7·14·28 모델의 평균 0.392595보다 **0.001171 개선**  
\- Mean 14를 제거하자 세 Fold 전체 평균이 개선돼 최종 Local 시계열 모델에서는 제외  
\- Fold 3은 `n_estimators=5000` 상한까지 학습했으며 Early stopping이 발동하지 않고 **4,995 iteration**에서 최고점을 기록  
\- Fold별 Best Iteration 차이가 매우 컸다.
  - Fold 1: 1,136
  - Fold 2: 4,335
  - Fold 3: 4,995

**### 장기 EMA / 연 단위 계절성 실험**

\- EDA에서 판매량에 연 단위 반복 패턴이 보였기 때문에 장기 시계열 정보를 여러 방식으로 표현해 실험했다.

**EMA 364**
\- `shift(16)` 이후 span 364 EMA를 추가  
\- EMA 364는 장기 판매 수준을 부드럽게 표현하지만, **364일 전 판매량 자체나 연 단위 계절 위치를 직접 표현하는 피처는 아니다.**  
\- 평균 RMSLE는 0.393733으로 Local 최고인 0.391424보다 악화  
\- 중요도가 존재하더라도 미래 성능 개선으로 이어지지는 않았다.

**Lag 364**
\- 약 52주 전 같은 요일의 실제 판매량을 직접 사용  
\- 평균 RMSLE 0.392009  
\- Fold 1은 개선됐지만 Fold 2·3은 악화  
\- 오래된 판매 수준이 현재 판매 수준과 달라질 수 있기 때문에 단순히 작년 같은 시점의 값을 가져오는 것이 항상 도움이 되지는 않았다.

**Seasonal Mean**
\- `lag364`, `lag728`, `lag1092`의 사용 가능한 값 평균으로 과거 여러 해의 동일한 계절 위치를 표현  
\- 평균 RMSLE 0.392427  
\- Lag 364 단독과 마찬가지로 Fold 1은 좋아졌지만 이후 Fold는 악화  
\- 여러 해의 과거 값을 평균하더라도 판매 수준 변화까지 해결되지는 않았다.

**### Rolling mean과 EMA 비교**

\- Rolling mean은 정해진 구간의 값을 동일한 가중치로 평균하고, 구간 밖의 과거 값은 완전히 제외한다.  
\- EMA는 최근 값에 더 높은 가중치를 주되 오래된 정보도 점차 작은 가중치로 계속 남긴다.  
\- 같은 단기 판매 수준을 표현하더라도 계산 방식이 다르다.  
\- 이번 데이터에서는 EMA로 Rolling mean을 대체했을 때 평균 RMSLE가 **0.395297**로 악화됐다.  
\- 따라서 단순히 더 복잡하거나 최근값에 가중치를 주는 방식이 항상 더 좋은 것은 아니었다.  
\- 현재 Validation에서는 **Rolling mean 7·28이 EMA보다 더 적합했다.**

---

**## Kaggle Public 제출 결과**

Local 시계열 실험이 끝난 뒤 실제 Test 구간에서 성능이 유지되는지 확인하기 위해 Kaggle에 제출했다.

| 제출 | 구성 | Final Iteration | Local 3-Fold RMSLE | Public RMSLE |
|---|---|---:|---:|---:|
| 이전 `submission_full_train.csv` | **Lag 16·21·28** | 4,044 | 0.403774 | **0.46161** |
| `submission_ts_best.csv` | Lag 16·21·28 + Mean 7·28 | 4,335 | **0.391424** | 0.48334 |
| `submission_ts_best_iter5000.csv` | Lag 16·21·28 + Mean 7·28 | 5,000 | **0.391424** | 0.48431 |

**### Public 결과 해석**

\- Local Validation에서는 Rolling mean을 추가하면서 성능이 크게 개선됐다.
  - Lag only: **0.403774**
  - Lag + Mean 7·28: **0.391424**
  - Local 기준 약 0.01235 개선

\- 그러나 Kaggle Public에서는 결과가 반대로 나타났다.
  - Lag only: **0.46161**
  - Lag + Mean 7·28: **0.48334**
  - Rolling을 추가한 모델이 약 0.02173 악화

\- 최종 iteration 선택 때문인지 확인하기 위해 4,335 → 5,000으로 늘려 다시 제출했지만:
  - 4,335: 0.48334
  - 5,000: 0.48431

\- iteration을 늘려도 개선되지 않았고 오히려 소폭 악화됐다.  
\- 따라서 이번 Public 악화를 단순히 **학습 iteration 부족 때문이라고 보기는 어렵다.**

---

**## 판단 / 배운 점**

**### Local Validation과 실제 Test 성능은 다를 수 있다**

\- 이번 실험에서 가장 중요한 결과는 **Local 최고 모델과 Public 최고 모델이 달랐다는 것**이다.

```text
Local Champion
Base 23 + Lag 16·21·28 + Rolling Mean 7·28
3-Fold RMSLE = 0.391424
Public RMSLE = 0.48334

Public Champion
Base 23 + Lag 16·21·28
3-Fold RMSLE = 0.403774
Public RMSLE = 0.46161
```

\- Rolling mean은 세 Validation Fold에서 매우 강한 피처였지만 실제 2017-08-16~08-31 Test에서는 같은 효과가 나타나지 않았다.  
\- Validation 평균이 좋아졌다는 사실만으로 실제 미래 구간에서도 반드시 좋아진다고 볼 수 없다.  
\- 특히 시계열에서는 **검증 시점과 실제 Test 시점의 패턴 차이**가 모델 간 상대 순위를 바꿀 수 있다.

**### Validation을 믿지 말아야 한다는 의미는 아니다**

\- 이번 결과 때문에 Validation을 버리고 Public 점수로 모델을 선택하는 것은 잘못된 방향이다.  
\- 작은 피처 실험, Lag/window 선택, 파라미터 비교 등은 계속 Local Validation을 기준으로 진행한다.  
\- Kaggle Public은 일정한 실험 단계가 끝났을 때 **외부 검증 용도**로 사용한다.  
\- Public 결과가 예상과 크게 다르면 구현 오류, Validation 구조, 데이터 분포 차이를 점검하고 필요하면 Local 검증 방식을 보완한다.

**### Public Leaderboard를 Validation처럼 사용하지 않기**

\- 여러 번 제출하는 것 자체는 문제가 아니다.  
\- 하지만 다음과 같이 Public 점수만 보고 조합을 반복해서 바꾸는 것은 피해야 한다.

```text
Rolling 7 제출
Rolling 28 제출
Rolling 14·28 제출
iteration 3500 제출
iteration 4000 제출
iteration 4500 제출
→ Public 최고 모델 선택
```

\- 이렇게 하면 Public leaderboard 자체에 과적합될 수 있다.  
\- 대신:

```text
Local에서 가설 수립
→ 백테스트
→ 단계별 후보 선정
→ Public 확인
→ 예상과 다르면 원인 분석
→ Local에서 다시 검증
→ 필요할 때만 재제출
```

의 순서를 유지하기로 했다.

**### 높은 Feature Importance도 일반화를 보장하지 않는다**

\- Rolling mean 7은 Local 모델에서 Gain 비중이 약 72%로 압도적으로 높았다.  
\- 하지만 실제 Public에서는 Rolling이 포함된 모델이 더 나빴다.  
\- 따라서 Feature Importance는 **모델이 해당 피처를 학습 과정에서 많이 사용했다는 사실**을 보여줄 뿐, 보지 않은 미래 데이터에서 성능이 좋아진다는 증거는 아니다.

**### 연 단위 패턴이 보인다고 Lag 364가 자동으로 유용한 것은 아니다**

\- EDA에서 연간 반복 패턴이 보이더라도 `lag364`는 과거의 계절 위치뿐 아니라 당시의 판매 수준까지 그대로 가져온다.  
\- 시간이 지나면서 매장·상품군의 판매 규모가 변했다면 오래된 값이 현재 수준과 맞지 않을 수 있다.  
\- Seasonal Mean도 과거 여러 해를 평균하면서 오래된 판매 수준을 함께 반영하기 때문에 동일한 문제가 남았다.  
\- 연간 패턴은 이후 Calendar 단계에서 `dayofyear`, 연간 주기 표현 등 **시간상의 위치 자체를 표현하는 방식**으로 접근하는 것이 더 적절할 수 있다.

**### 시계열 피처 실험 종료 기준**

\- Lag, Rolling mean, Rolling std, EMA, 장기 EMA, 연간 Lag, Seasonal Mean까지 주요 시계열 기반 후보들을 비교했다.  
\- Local에서는 추가 개선을 얻었지만 실제 Public 결과까지 확인했을 때 현재 가장 안정적인 제출은 Lag-only 모델이었다.  
\- 여기서 Rolling window를 계속 바꾸거나 추가 시계열 파생변수를 무한히 탐색하는 것은 얻는 정보보다 Validation/Public 과적합 위험이 커질 수 있다고 판단했다.  
\- 따라서 **시계열 자체의 파생 피처 실험은 여기서 종료**한다.

---

**### 기준 모델 재정의**

Day08까지는 Local 최고 모델을 이후 실험의 기준으로 사용하려 했지만, Public 결과 확인 후 기준을 다음과 같이 나눠 관리하기로 했다.

**Local 최고 기록**

\- 기본 피처 23개  
\- Lag 16·21·28  
\- Rolling mean 7·28  
\- 총 28개 피처  
\- ES 150 백테스트  
\- 평균 RMSLE: **0.391424**

**앞으로의 메인 실험 Baseline / Public Champion**

\- 기본 피처 23개  
\- Lag 16·21·28  
\- 총 26개 피처  
\- 3-Fold 평균 RMSLE: **0.403774**  
\- Public RMSLE: **0.46161**

\- 이후 Calendar / Event 피처는 우선 **Lag-only 모델을 기준으로 추가 효과를 비교**한다.  
\- Rolling 모델은 삭제하지 않고 Local Champion 및 참고 Benchmark로 보존한다.  
\- 이후 새로운 피처가 Lag-only에서 확실한 개선을 만들 경우 Rolling 모델과의 조합 효과는 필요할 때 별도로 확인할 수 있다.

---

**### 느낀 점**

\- 처음에는 Rolling mean을 넣으면서 Validation 점수가 계속 좋아져 시계열 피처를 더 세밀하게 만들수록 실제 성능도 계속 좋아질 것이라고 생각했다.  
\- 실제로 `0.417374 → 0.403774 → 0.393762 → 0.391424`까지 Local 점수가 단계적으로 내려가면서 실험 방향이 잘 맞고 있다고 판단했다.  
\- 하지만 Kaggle Public에서는 처음 추가했던 Lag 모델이 **0.46161**로 가장 좋았고, Local 최고 Rolling 모델은 **0.48334**로 오히려 크게 악화됐다.  
\- 2~3일 동안 여러 Rolling, EMA, 연간 시계열 피처를 실험한 뒤 결과적으로 가장 단순한 Lag 모델이 실제 Test에서 가장 좋았다는 점은 처음에는 허무하게 느껴졌다.  
\- 그러나 이번 과정에서 **Validation 성능과 실제 일반화 성능은 다를 수 있다는 것을 직접 경험했다.**  
\- 특히 “Local 점수가 좋아졌으니 좋은 모델이다”에서 끝나는 것이 아니라, 검증 구간이 실제 미래를 얼마나 잘 대표하는지도 함께 생각해야 한다는 점을 배웠다.  
\- 실패한 피처 실험도 어떤 정보가 불필요하거나 불안정한지 알려주기 때문에 의미가 있었다.  
\- Rolling, EMA, 연간 Lag를 실제로 비교해보지 않았다면 단순히 이론적으로 유용할 것이라고 생각한 채 다음 단계로 넘어갔을 가능성이 높다.  
\- 또한 Kaggle Public 점수는 많이 볼수록 좋은 것이 아니라, **Local 실험 결과가 실제 외부 구간에서도 유지되는지 확인하는 도구로 사용해야 한다는 태도**를 정리할 수 있었다.  
\- 한 줄 정리: **좋은 Validation 점수는 출발점이지 보장이 아니며, 모델의 가치는 보지 않은 미래 구간에서도 일반화되는지까지 확인해야 한다.**

---

**## 다음**

1. **시계열 파생 피처 실험 종료**
   - Lag 16·21·28을 현재 메인 시계열 피처로 유지
   - Rolling mean / std / EMA / Lag364 / Seasonal Mean 추가 탐색은 일단 중단
   - Local Champion인 Lag + Mean 7·28 모델은 Benchmark로 보존

2. **이후 Local Baseline을 Lag-only 모델로 변경**
   - 기본 피처 23개 + Lag 16·21·28
   - 총 26개 피처
   - 3-Fold 평균 RMSLE **0.403774**
   - Public RMSLE **0.46161**
   - 이후 새로운 피처의 Local 개선 효과를 우선 이 모델과 비교

3. **Calendar 피처 실험 시작**
   - 기존 `year`, `month`, `day`, `dayofweek`, `is_weekend` 외에 추가적인 시간 위치 표현 검토
   - 연간 반복 패턴을 과거 판매량 자체가 아니라 Calendar 위치로 표현하는 방법 우선 검토
   - 후보 예시:
     - `dayofyear`
     - 연간 주기의 sin / cos 표현
     - 월초 / 월말 여부
     - 15일 / 월말 급여일 관련 피처

4. **Event 피처 개선**
   - 기존 Holiday/Event 피처를 유지하면서 이벤트 직전·직후 효과를 표현할 필요가 있는지 검토
   - 실제 Test 구간과 Validation 구간에서 이벤트 구성이 어떻게 다른지도 확인

5. **이후 External Feature 단계**
   - Calendar / Event 실험이 끝난 뒤 `oil.csv`, `transactions.csv` 활용 가능성 검토
   - Test 시점에 실제로 알 수 있는 정보인지 먼저 확인해 데이터 누수를 방지

6. **Public 제출 운영 방식 유지**
   - 작은 실험은 Local Validation으로 판단
   - 큰 피처 단계가 끝났을 때 최종 후보를 Public에 제출
   - Public 결과가 Local과 크게 다르면 원인 분석 후 Local에서 다시 검증
   - Public 점수를 직접 최적화하는 방식의 반복 제출은 피하기

7. **실험 기록 갱신**
   - `experiments.md`에 Mean14 제거, EMA364 재검증, Lag364, Seasonal Mean 결과 추가
   - Public 제출 결과 기록
     - Lag-only: **0.46161**
     - Lag + Mean7·28 / iter4335: **0.48334**
     - Lag + Mean7·28 / iter5000: **0.48431**
   - Local Champion과 Public Champion을 구분해 기록
   - `modeling.md`에도 현재 채택 기준과 Public 결과 반영