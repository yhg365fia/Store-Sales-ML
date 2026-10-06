# Day03

## 오늘 한 일
- Kaggle Store Sales - Time Series Forecasting 전처리 시작
  - 데이터 형식 확인
    - `train`, `test`, `holidays`의 `date`를 `datetime`으로 변환
    - `store_nbr`, `family` 등 컬럼의 데이터 타입과 구조 확인
  - `sales`, `onpromotion` 이상치 여부 판단
    - Q3 이후 구간과 log x-scale을 이용해 긴 꼬리 부분의 분포 확인
    - IQR 기준 일반 / 극단 이상치 비율 확인
    - 99 / 99.5 / 99.9 / 99.95 / 99.99 percentile 확인
    - `sales` 상위 0.01%가 어떤 `family`에서 주로 발생하는지 확인
    - `sales`, `onpromotion`의 원본 분포와 `log1p` 적용 후 분포 비교
    - `onpromotion`과 `sales`의 Pearson / Spearman 상관관계 확인, Family별로도 추가 확인
    - 높은 `sales`, `onpromotion` 값을 단순 이상치로 제거하지 않기로 결정
  - 누락 날짜 확인 및 복원
    - train에서 빠져 있는 날짜 확인: 2013~2016년의 `12-25` 총 4개 날짜
    - 해당 날짜가 `holidays`의 `Navidad`와 대응되는 것을 확인
    - 누락된 날짜를 추가하고 `sales`, `onpromotion`을 0으로 복원
  - Holiday 기본 정리
    - `holidays`의 완전 중복 행 제거
    - `transferred=True`인 원래 공휴일 날짜 제거 및 날짜순 정렬
  - 시간 기준 Train / Validation 분리
    - Train: `2013-01-01 ~ 2017-07-30`
    - Validation: `2017-07-31 ~ 2017-08-15`
    - Test: `2017-08-16 ~ 2017-08-31`
    - 실제 test가 16일이므로 validation 역시 마지막 16일로 구성
- 전처리와 Feature Engineering의 범위 구분
  - 데이터의 오류·누락·형식 문제를 정리해 학습 가능한 상태로 만드는 작업은 **Preprocessing**
  - 기존 데이터를 바탕으로 모델이 패턴을 더 잘 학습하도록 새로운 정보를 만드는 작업은 **Feature Engineering**
  - 이번 단계에서는 데이터 정리, 이상치 판단, 누락 날짜 복원, holiday 정리까지 진행
  - 날짜 파생 변수, 프로모션 여부, holiday-store 매칭, lag / rolling 변수 등은 다음 Feature Engineering 단계에서 처리하기로 결정

## 판단 / 배운 점

### 전처리와 Feature Engineering의 차이
- 처음에는 `dayofweek` 생성이나 `sales`의 `log1p` 적용까지 모두 전처리라고 생각하기 쉬웠다.
- 하지만 작업을 해 보니 **두 단계는 목적이 다르다**는 것을 구분하게 되었다.
- **Preprocessing**: 원본의 오류, 누락, 중복, 타입 문제를 정리해 **데이터를 정상적으로 사용할 수 있는 상태로 만드는 과정**
  - 날짜 타입 변환
  - 누락 날짜 복원
  - 중복 데이터 제거
  - 잘못 적용된 holiday 정리
  - 이상치 처리 여부 판단
- **Feature Engineering**: 사용할 수 있는 데이터를 바탕으로 **모델이 패턴을 더 쉽게 학습할 수 있는 새로운 설명 변수를 만드는 과정**
  - `year`, `month`, `dayofweek`
  - 주말 여부
  - 프로모션 여부
  - holiday 종류
  - Local / Regional 이벤트가 실제 매장에 적용되는지 여부
  - lag / rolling statistics
- 같은 작업도 목적에 따라 다르게 볼 수 있다.
  - `date`를 `datetime`으로 바꾸는 것은 데이터를 쓰기 위한 전처리
  - 그 날짜에서 `month`, `dayofweek`를 추출하는 것은 새로운 정보를 만드는 Feature Engineering
- 앞으로는 **"이 작업은 데이터를 정리하는 것인가, 새로운 정보를 만드는 것인가?"**를 기준으로 두 단계를 구분한다.
- 이렇게 나누면 나중에 성능이 변했을 때 원인이 데이터 정리인지 feature 추가인지 추적하기 쉬울 것이라고 생각했다.

### 이상치 처리
- `sales`와 `onpromotion` 모두 오른쪽 꼬리가 긴 strongly right-skewed 분포였다.
- 하지만 **값이 크다는 이유만으로 이상치라고 판단하면 안 된다**고 생각했다.
- 높은 `sales` 값은 `GROCERY I`, `BEVERAGES`처럼 실제 판매 규모가 큰 상품군에서 반복적으로 나타났다.
- percentile을 봐도 극단값은 전체에서는 적지만 실제 관측값으로 존재했다.
- 따라서 IQR 기준을 그대로 적용해 제거하면 **실제 존재하는 대규모 판매량까지 삭제할 가능성**이 있다.
- 결국 높은 값을 제거하지 않고 원본 데이터를 유지하기로 했다.
- 시계열에서는 극단값이 행사, 계절성, 프로모션 같은 실제 사건을 반영할 수 있으므로 **통계적 기준보다 데이터의 의미를 먼저 확인하는 것이 중요하다**고 느꼈다.

### 로그 변환
- `sales` 원본은 낮은 값에 데이터가 몰려 있고, 일부 매우 큰 값 때문에 긴 오른쪽 꼬리가 형성되어 있었다.
- `log1p(sales)`를 적용하면 큰 값의 영향이 줄어 분포가 훨씬 완화되는 것을 확인했다.
- 다만 로그 변환은 데이터 오류를 고치는 작업이 아니라 모델이 값을 다루기 쉽게 표현하는 방법이다. 그래서 단순 전처리라기보다 **feature 또는 target transformation에 가깝다**고 구분했다.
- 원본 `sales`는 보존하고, 필요할 때 모델링 과정에서 별도로 활용한다.
- `onpromotion`도 right-skewed하지만 현재는 원본 값을 유지하고, 변환 여부는 Feature Engineering / 모델링 단계에서 비교한다.

### 프로모션과 판매량
- 전체 데이터에서 `onpromotion`과 `sales` 사이에는 양의 관계가 어느 정도 나타났다.
- Pearson뿐 아니라 순위 관계를 보는 Spearman 상관도 함께 확인했다.
- Family별로 나눠 보면 상품군에 따라 관계의 정도가 달랐다.
- 프로모션이 많을수록 판매량이 증가하는 경향은 있을 수 있지만, **상관관계만으로 프로모션이 판매 증가의 직접 원인이라고 판단해서는 안 된다.**
- Feature Engineering에서는 `onpromotion` 값뿐 아니라 `is_promotion`, 상품군과 프로모션의 관계 등도 실험해볼 수 있다.

### 누락 날짜
- train 전체 기간 중 일부 날짜가 존재하지 않는 것을 발견했다.
- 2013~2016년의 `12월 25일`이 각각 빠져 있어 총 4개 날짜가 누락되어 있었고, `holidays`와 비교하니 모두 `Navidad`(크리스마스)와 대응됐다.
- 무작위 누락이 아니라 **특정 이벤트와 관련된 규칙적인 누락**이라는 점이 중요했다.
- 시계열의 날짜 구조를 연속적으로 유지하기 위해 해당 날짜를 추가하고 `sales`, `onpromotion`을 0으로 복원했다.
- 일반적인 `NaN`뿐 아니라 **행 자체가 없는 날짜도 결측의 한 형태로 확인해야 한다**는 것을 배웠다.

### 공휴일 데이터 전처리
- `holidays`에는 완전히 동일한 중복 행이 있을 수 있어 완전 중복만 제거했다.
- `transferred=True`인 공휴일은 원래 날짜가 실제 휴일로 적용되지 않고 다른 날짜로 이동한 경우라서 원래 날짜를 제거했다.
- 같은 날짜에 National / Regional / Local 이벤트가 동시에 있을 수 있으므로, `date`만 기준으로 중복 제거하면 중요한 정보가 사라진다.
- 그래서 이번 단계에서는 **공휴일 데이터를 잘못 해석하지 않도록 정리하는 수준까지만 전처리**로 두었다.
- 어떤 매장에 공휴일이 적용되는지를 `stores.csv`와 연결해 변수로 만드는 일은 Feature Engineering에서 한다.

### Train / Validation 분리
- 랜덤 분할 대신 **시간 순서를 유지하는 validation**을 사용했다.
- 실제 Kaggle test가 `2017-08-16 ~ 2017-08-31`의 16일이므로, validation도 바로 직전 16일(`2017-07-31 ~ 2017-08-15`)로 구성했다.
- 과거 데이터로 바로 다음 16일을 예측하는 실제 문제 구조를 비슷하게 재현할 수 있다.
- 랜덤 분할은 미래 데이터가 학습 데이터에 섞일 수 있어, 시계열에서는 실제 사용 상황과 맞지 않는 평가가 된다는 점을 이해했다.
- 하나의 validation 구간만으로 판단하면 특정 기간의 영향이 클 수 있으므로, 추후 여러 16일 구간을 이용한 time-series backtesting도 고려한다.

### EDA → Preprocessing → Feature Engineering
- 세 단계는 별개 작업이 아니라 **앞 단계의 판단이 다음 단계로 이어지는 과정**이라는 것을 느꼈다.
  - EDA에서 `sales`에 큰 값이 많다는 것을 발견 → 전처리에서 실제 이상치인지 확인
  - 날짜 누락을 발견 → 원인을 `holidays`와 비교하고 복원
  - 요일·공휴일에 따라 판매량이 달라질 가능성 확인 → `dayofweek`, holiday feature를 만들 근거
- 정리하면 다음과 같다.
  - **EDA:** 어떤 패턴과 문제가 있는지 발견
  - **Preprocessing:** 데이터 자체의 문제를 정리
  - **Feature Engineering:** 발견한 패턴을 모델이 사용할 수 있는 변수로 표현
- 이전에는 코드상 작업 종류 정도로 생각했지만, 이제는 **각 단계가 서로 다른 질문에 답하는 과정**이라고 이해한다.

### 느낀 점
- Day01, Day02에서는 데이터를 관찰하며 구조와 패턴을 이해했다면, Day03에는 **관찰한 내용을 실제 데이터 처리 결정으로 연결하는 작업**을 했다.
- 처음에는 큰 값을 이상치로 처리할 수도 있다고 생각했지만, 분포와 상품군을 직접 확인하니 실제 데이터의 특성일 가능성이 높다는 것을 알게 되었다.
- IQR이나 percentile 같은 기준을 계산하는 것보다 **그 기준을 적용하는 것이 타당한지 판단하는 과정이 더 중요했다.**
- 앞으로는 한 번에 많은 변수를 만들기보다 **전처리를 먼저 확실히 끝내고, baseline을 기준으로 Feature Engineering을 하나씩 추가하며 효과를 확인하는 방식**이 좋다고 생각한다.
- 한 줄 정리: **EDA에서 문제와 패턴을 발견하고, Preprocessing에서 데이터를 바로잡고, Feature Engineering에서 그 패턴을 모델이 이해할 수 있는 정보로 바꾸는 흐름을 처음으로 명확하게 구분해 본 날이었다.**

## 다음
1. Feature Engineering 시작
   - 날짜 feature 생성 (`year`, `month`, `day`, `dayofweek`, 주말 여부 등)
   - 프로모션 관련 feature 생성
   - `stores.csv`의 매장 정보 결합
   - Local / Regional 공휴일을 실제 해당 매장과 연결
2. 기본 Feature 구성
   - 가능한 단순한 변수부터 시작
   - 필요하지 않은 복잡한 feature를 한 번에 추가하지 않기
   - 각 feature가 어떤 의미를 가지는지 기록
3. 시계열 Feature 검토
   - lag feature
   - rolling mean / rolling statistics
   - 미래 정보가 섞이지 않도록 leakage 주의
4. 첫 Baseline 모델 준비
   - 기본 feature만 사용해 모델 학습
   - Validation RMSLE 측정
   - 이후 feature를 하나씩 추가하면서 성능 변화 비교
5. 모델링 과정에서 문제가 발견되면 다시 EDA / Preprocessing으로 돌아가 원인 확인