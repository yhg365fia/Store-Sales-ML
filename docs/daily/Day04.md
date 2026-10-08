네, 같은 내용이 "오늘 한 일"과 "판단 / 배운 점"에 겹쳐 나오는 부분이 많아서 거기를 중심으로 줄였습니다. 코드 사용 목록, 중복 설명, 느낀 점의 반복을 덜어냈고 핵심 판단은 그대로 뒀습니다.

# Day04

## 오늘 한 일

Kaggle Store Sales - Time Series Forecasting **Feature Engineering 진행**

### 1. 기본 Feature 생성
- **날짜**: `year`, `month`, `day`, `dayofweek`, 주말 여부 `is_weekend` (`dayofweek >= 5`)
- **프로모션**: `onpromotion` 원본 유지, `onpromotion_log`(`log1p`), `is_promotion`(존재 여부)
- **범주형 변환**: `store_nbr`, `family`, `type`, `locale`, `locale_name`, `description`, `event_group`을 `category`로, `transferred`는 `bool` 유지

### 2. Holiday Feature 생성
- `description` 문자열 기준으로 `event_group` 생성 (`Christmas`, `New Year`, `Carnival`, `Mothers Day`, `Independence Day`, `City Foundation`, `Other`), `description` 원본은 보존
- National 이벤트를 대상으로 그룹별 0/1 Feature 7개 생성 (`event_christmas`, `event_new_year` 등)
- 같은 날짜의 복수 이벤트는 `groupby("date")` + `.max()`로 집계

### 3. Holiday 병합 설계
- 문제: 같은 날짜에 이벤트가 여러 개면 단순 병합 시 판매 행이 중복됨
- 해결: 날짜별로 집계한 뒤 `left merge`
- 생성 변수: `holiday_count`, `holiday_types`, `event_groups`, `holiday_descriptions`, `has_national_event`
- Regional / Local 이벤트는 보존만 하고, `stores.csv` 기반 지역 매칭은 추후 진행

### 4. 코드 구조 정리
| 파일 | 역할 |
|---|---|
| `preprocessing.py` | 데이터 로드, 전처리, Train / Validation 분리 |
| `feature_engineering.py` | 날짜·프로모션·이벤트 Feature 생성 및 병합 |
| `main.py` | 전체 파이프라인 실행과 결과 확인 |

- 함수 정의와 실행 코드를 분리하고 `import`와 `if __name__ == "__main__"`의 차이를 확인
- `event_group` 미생성으로 발생한 `KeyError`를 확인하고 생성 함수 추가

### 5. 실행 결과 확인
| 항목 | 결과 |
|---|---|
| Train / Validation / Test | 2,979,504행 / 28,512행 / 28,512행 |
| 기본 Feature 생성 후 결측치 | 0개 |
| 복원한 크리스마스 행 | 7,128개 |

---

## 판단 / 배운 점

### Feature Engineering의 목적
- Feature Engineering은 컬럼을 많이 만드는 작업이 아니라 **모델이 패턴을 더 잘 학습하도록 정보를 표현하는 과정**이다.
- 같은 정보도 표현에 따라 학습 가능한 패턴이 달라진다. (`date` → `month`, `dayofweek`, `is_weekend` / `onpromotion` → 원본, 로그, 존재 여부)
- 다만 **피처를 추가한다고 성능이 반드시 좋아지지는 않는다.** Baseline을 만들어 Validation RMSLE로 비교해야 한다.
- 따라서 **가설 생성 → 피처 추가 → 성능 검증의 반복 과정**으로 이해했다.

### 날짜 / 프로모션 Feature
- EDA에서 요일별 판매량 차이를 확인했기 때문에 `dayofweek`, `is_weekend`를 만들었다. **EDA에서 발견한 패턴이 피처의 근거가 된다.**
- `onpromotion`은 오른쪽 꼬리가 긴 분포라 로그 변환 값을 추가했지만, 원본의 크기 정보를 대체하지는 못하므로 둘 다 보존했다.
- `is_promotion`은 상품 수와 별개로 **프로모션 존재 자체의 영향**을 표현한다. 세 표현의 효과는 모델 성능으로 검증한다.

### 범주형 변수
- `store_nbr`는 정수형이지만 크기에 의미가 없는 식별자이므로 범주형으로 해석해야 한다.
- **숫자로 저장되어 있다는 것과 수치형 변수로 해석해야 한다는 것은 다르다.**
- 모델마다 범주형 처리 방식이 다르므로 LightGBM의 처리 방식을 확인해야 한다.

### Holiday Event Group과 0/1 Feature
- 비슷한 이벤트도 문자열이 다르다. (`Navidad`, `Navidad-1`, `Navidad+1` → Christmas)
- `description`은 개별 이벤트의 **구체적인 차이**를, `event_group`은 유사 이벤트의 **공통 특성**을 담으므로 둘 다 보존했다.
- 같은 날짜에 대표 이벤트 하나만 남기면 정보가 손실되고, `Christmas|Other` 같은 문자열 결합은 `Christmas`와 다른 범주로 취급되는 한계가 있다. 그래서 그룹별 0/1 피처를 추가했다.
- 0/1 피처는 발생 횟수나 구체적 차이를 표현하지 못하므로 `holiday_count`와 문자열 피처로 보완했다.
- 현재 분류 규칙은 완벽하지 않으므로 `Other`에 포함된 이벤트를 추가로 확인해야 한다.

### groupby와 max를 이용한 복수 이벤트 처리
오늘 가장 중요하게 이해한 부분이다.

```python
event_daily = (
    national
    .groupby("date")[event_cols]
    .max()
    .reset_index()
)
```

- `groupby("date")`로 같은 날짜를 묶고, `.max()`로 이벤트가 하나라도 있으면 1, 없으면 0으로 만든 뒤, `.reset_index()`로 날짜를 일반 컬럼으로 복원한다.
- `.sum()`은 같은 그룹 이벤트가 여러 번이면 2 이상이 되지만 `.max()`는 0/1을 유지한다. **존재 여부를 표현할 때는 합계보다 최댓값이 적합하다.**
- 복수 이벤트를 삭제하는 것이 아니라 **정보를 보존하면서 날짜별 한 행으로 변환**하는 과정이다.

### Holiday 병합 방식
- 판매 데이터는 날짜 × 매장 × 상품군 단위이고 Holiday는 한 날짜에 여러 이벤트가 있을 수 있어, **단위가 다르므로 병합 전에 구조를 맞춰야 한다.**
- `inner merge`는 Holiday가 없는 날짜의 판매 데이터를 제거하므로 **`left merge`가 적합**하다. 단, 같은 날짜에 Holiday가 여러 개면 행이 중복되므로 먼저 날짜별로 집계해야 한다.
- 현재는 National 이벤트만 전체 매장에 적용한다. Regional / Local을 모든 매장에 병합하면 잘못된 정보가 들어가므로, 추후 `city`, `state`로 적용 여부를 판단한다.
- 복수 이벤트의 문자열 집계와 최종 병합 부분은 **다음 학습에서 자세히 분석할 예정**이다.

### 코드 구조와 실험 설계
- `if __name__ == "__main__"` 내부에서 만든 변수를 다른 파일에서 import하려던 문제를, 함수 정의와 실행 코드를 분리해 해결했다. 역할이 명확해져 재사용과 디버깅이 쉬워진다.
- `KeyError`를 통해 **피처를 사용하는 함수보다 생성 과정이 먼저 실행되어야 한다**는 점을 확인했다.
- `description`과 `event_group` 중 하나를 고르는 대신, 비교 실험을 위해 둘 다 보존했다. Holiday 표현도 두 방식을 모두 준비했다.

| 실험 | 구성 |
|---|---|
| A | 문자열 `event_groups` |
| B | 이벤트별 0/1 피처 |
| C | 문자열 + 0/1 병용 |

- 다른 조건은 동일하게 두고 Validation RMSLE만 비교한다. **효과를 미리 확신하기보다 가설을 세우고 모델 결과로 검증하는 방식**이 중요하다.

### 느낀 점
- Day03이 데이터를 학습 가능한 상태로 정리하는 작업이었다면, Day04는 **데이터의 정보를 모델이 학습할 수 있는 피처로 표현하는 과정**이었다.
- 처음에는 컬럼을 추가하는 반복 작업처럼 느껴졌고, 추천시스템에 더 관심이 있어 이 작업의 필요성에도 의문이 들었다.
- 하지만 이벤트 데이터 병합을 다루면서 **코드를 작성하는 것과 데이터 구조를 이해하고 설계하는 것은 다르다**는 것을 알게 되었다. `groupby()`, `.max()` 같은 코드도 실제 데이터가 어떻게 변하는지 보며 이해했다.
- 하나의 문제에도 여러 해결 방식이 있고, 적절한 방식은 이론적 장단점보다 **실제 모델 성능으로 검증**해야 한다.
- 복수 이벤트 문자열 집계와 최종 병합은 오늘 완전히 분석하지 못했다. 다시 공부한 뒤 Baseline으로 피처 효과를 확인하고 싶다.
- **한 줄 정리: 새로운 정보를 만드는 것뿐 아니라, 서로 다른 데이터 구조를 정보 손실 없이 연결하고 여러 표현 방식을 실험할 수 있게 설계하는 것이 Feature Engineering의 중요한 역할임을 배운 날이었다.**

---

## 다음

1. **Holiday Merge 코드 분석 마무리**
   - `merge_holiday_features()`의 `# 2. 복수 이벤트 문자열 및 개수 집계` 부분부터 재분석
   - `.agg()`, `lambda`, `set()`, `sorted()`, `"|".join()` 동작 이해
   - `left merge`, `validate="many_to_one"` 이해, 병합 전후 행 수와 결측치 확인
2. **Holiday 피처 최종 검증**
   - `holiday_count`, `event_groups`, `holiday_descriptions`, 이벤트별 0/1 피처 생성 결과 확인
   - `Other`로 분류된 이벤트 확인, National 이벤트만 적용되었는지 확인
3. **모델 평가 설계**
   - RMSLE 이해 및 구현, Train / Validation 기간 재확인
   - 예측값 음수 처리와 `log1p(sales)` 변환 방식 결정
   - 입력 피처와 Target 분리, 데이터 누수(Leakage) 점검
4. **첫 Baseline 모델 학습**
   - LightGBM으로 기본 피처 모델 구축, Validation RMSLE를 이후 실험의 기준점으로 설정
5. **Feature Engineering 실험**
   - Holiday 문자열 vs 0/1 피처, `description` vs `event_group`, 프로모션 피처 효과 비교
   - 필요하면 lag / rolling statistics 추가 (누수 확인)
   - 새로운 문제가 발견되면 EDA / Preprocessing으로 돌아가 원인 분석

---

더 줄이고 싶다면 "판단 / 배운 점"의 "코드 구조와 실험 설계"를 한두 줄로 압축하거나, "다음"을 큰 항목 제목만 남기는 방법이 있습니다.