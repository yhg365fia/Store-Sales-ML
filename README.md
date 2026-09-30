# 📈 Store Sales - Time Series Forecasting

## 1. 프로젝트 개요
- 목표: 에콰도르의 대형 식료품 소매업체 Corporación Favorita의 여러 매장과 제품군별 미래 판매량 예측
- 문제 유형: 시계열 예측 / 회귀
- 평가 지표: RMSLE (Root Mean Squared Logarithmic Error)
- 예측 기간: 학습 데이터 마지막 날짜 이후 15일

## 2. 데이터
- 데이터 출처: Kaggle - Store Sales - Time Series Forecasting
- Target: `sales`

### 주요 데이터
- `train.csv`
  - `date`: 날짜
  - `store_nbr`: 매장 번호
  - `family`: 제품군
  - `onpromotion`: 프로모션 중인 품목 수
  - `sales`: 판매량

- `test.csv`
  - train과 동일한 입력 변수를 사용
  - 마지막 학습 날짜 이후 15일의 `sales` 예측

- `stores.csv`
  - 매장의 `city`, `state`, `type`, `cluster` 정보

- `transactions.csv`
  - 매장별 일일 거래량 정보

- `oil.csv`
  - 일일 유가 정보

- `holidays_events.csv`
  - 공휴일 및 행사 정보
  - 이전된 휴일(`transferred`) 처리에 주의

- `sample_submission.csv`
  - Kaggle 제출 형식 예시

### 추가 도메인 정보
- 공공 부문 임금은 매월 15일과 월말에 지급되며 판매량에 영향을 줄 수 있음
- 2016년 4월 16일 에콰도르 지진 이후 몇 주 동안 생필품 판매량에 큰 변화가 발생

## 3. 프로젝트 구조

```text
Store-Sales-ML/
│
├─ data/
│  ├─ stores.csv
│  ├─ test.csv
│  ├─ train.csv
│  └─ transactions.csv
│
├─ docs/
│  ├─ Day01.md
│  ├─ EDA.md
│  ├─ Preprocessing.md
│  ├─ Feature_Engineering.md
│  ├─ Modeling.md
│  └─ Evaluation.md
│
├─ notebook/
│
├─ main.py
├─ README.md
├─ requirements.txt
└─ .gitignore