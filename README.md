# S&P 500 Market Volatility Regime Classification

> **S&P 500 예측 가능성 탐색: 방향성 예측의 한계와 변동성 레짐 분류를 통한 대안적 접근**  
> 인하대학교 데이터사이언스학과 | 통계적 기계학습 팀 프로젝트 | 2026 Spring

---

## Overview

이 프로젝트는 S&P 500 지수의 일별 OHLCV 데이터(1983~2026)를 활용하여  
**시장 변동성 레짐(Low / Mid / High)을 머신러닝으로 분류**하는 시스템을 구축합니다.

단순 가격 방향성(상승/하락) 예측을 먼저 시도하고 그 한계를 실증적으로 확인한 후,  
문제를 변동성 레짐 분류로 재정의하여 더 예측 가능하고 실용적인 분석 결과를 도출합니다.

---

## Key Results

| Model | Accuracy | F1 (macro) | F1 (High Regime) |
|-------|----------|------------|-----------------|
| **Random Forest** | **59.3%** | **56.3%** | **80.5%** |
| Decision Tree | 53.5% | 50.6% | 78.2% |
| Logistic L1 | 47.1% | 44.4% | 67.6% |
| Logistic L2 | 45.2% | 42.8% | 64.6% |
| SVM | 40.4% | 38.2% | 56.9% |
| KNN (K=5) | 41.9% | 41.5% | 55.3% |
| Naive Bayes | 46.1% | 36.6% | 61.8% |
| MLP Neural Net | 31.8% | 30.3% | 34.4% |
| **Baseline** | 23.8% | **12.8%** | 0.0% |

> Test Set 기준 (2022-01-03 ~ 2026-03-10)

**핵심 발견:**
- 변동성 레짐 분류에서 Random Forest가 베이스라인(F1 12.8%) 대비 **4.4배 향상**
- 특히 불안정 시장(High 레짐) 감지 F1 **80.5%** — 리스크 관리 관점에서 실용적
- 방향성 예측(2클래스)에서는 모든 모델이 베이스라인(53.8%)과 유의미한 차이 없음
  → "문제 정의" 자체가 결과를 바꾼다는 것을 실증

---

## Project Structure

```
sp500_volatility_regime/
│
├── notebook/
│   ├── 01_EDA_Preprocessing.py     # EDA + 전처리 + 피처 엔지니어링
│   └── 02_Modeling.py              # 7개 모델 학습 + 비교 + 시뮬레이션
│
├── data/
│   ├── raw/                        # 원본 데이터 (sap500.csv — gitignore)
│   ├── processed/                  # 전처리 완료 데이터 (자동 생성)
│   └── README.md                   # 데이터 설명
│
├── reports/
│   └── figures/                    # 생성된 시각화 이미지
│
├── app.py                          # Streamlit 용어 대시보드
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Data Quality Analysis

- **원본:** S&P 500 일별 OHLCV (1927-12-30 ~ 2026-03-10, 24,663행)
- **발견:** 1983년 이전 데이터는 OHLC 값이 대부분 동일 → 실질적으로 Close만 존재
- **분석 채택:** 1983-01-03 이후 **10,882 거래일** (약 43년)

### 2. Target Variable

**Primary Target — Volatility Regime (3-class)**

```
향후 20일 실현 변동성(연환산)을 3분위로 분류
  Low  (0): 안정 시장 — 변동성 하위 33%
  Mid  (1): 보통 시장 — 변동성 중간 33%
  High (2): 불안 시장 — 변동성 상위 33%
```

**Secondary Target — Price Direction (2-class, 비교용)**

```
익일 종가 > 당일 종가 → 1 (상승)
익일 종가 ≤ 당일 종가 → 0 (하락/보합)
```

### 3. Feature Engineering (38 Features, 7 Groups)

| Group | Features | Financial Rationale |
|-------|----------|-------------------|
| Returns (5) | log_ret_1d/5d/10d/20d/60d | 정상성 확보, 기간 합산 특성 |
| Moving Average (7) | price_ma20_ratio, ma5_20_spread | 가격 수준 차이 제거, 추세 방향 |
| Volatility (8) | vol_20d, atr_ratio, vol_ratio_5_20 | 변동성 군집 현상 포착 (타겟 상관 최고) |
| Momentum (7) | rsi_14, macd, macd_hist | 과매수/과매도, 추세 전환 신호 |
| Volume (5) | vol_ratio_20d, vol_log | 거래량 급증 = 변동성 확대 선행 신호 |
| Candle Pattern (3) | intraday_ret, body_ratio | 장중 매수/매도 압력 |
| Market State (2) | pos_in_52w_range, above_ma200 | Bull/Bear 시장 구분 |

> **Data Leakage 방지:** 모든 피처에 `shift(1)` 적용 — 모델이 예측 시 미래 정보 사용 불가

### 4. Train / Validation / Test Split (Walk-Forward)

```
Train      : 1983-01-03 ~ 2018-12-31  (8,824일 / 83%)
Validation : 2019-01-02 ~ 2021-12-31  (  757일 /  7%)
Test       : 2022-01-03 ~ 2026-03-10  (1,009일 / 10%)
```

> `shuffle=False` 엄격 적용. Test Set은 최종 평가 전까지 절대 학습에 사용하지 않음.

---

## Quick Start

### 1. 환경 설치

```bash
git clone https://github.com/YOUR_USERNAME/sp500-volatility-regime.git
cd sp500-volatility-regime
pip install -r requirements.txt
```

### 2. 데이터 준비

`data/raw/sap500.csv` 에 S&P 500 OHLCV 데이터를 넣어주세요.  
(Yahoo Finance → `^GSPC` 티커로 다운로드 가능)

### 3. 분석 실행

**Google Colab (권장):**

```python
# 1단계: EDA + 피처 엔지니어링
exec(open('notebooks/01_EDA_Preprocessing.py').read())

# 2단계: 모델링 + 비교
exec(open('notebooks/02_Modeling.py').read())
```

**로컬 실행:**

```bash
# Jupyter Notebook으로 변환 후 실행
pip install jupytext
jupytext --to notebook notebooks/01_EDA_Preprocessing.py
jupytext --to notebook notebooks/02_Modeling.py
```

### 4. 용어 대시보드 실행

```bash
streamlit run app.py
 → [용어 대시보드 실행](https://quantmlglossary.streamlit.app)
```

---

## Story: Why Volatility Regime, Not Direction?

```
1차 시도: 방향성 예측 (상승/하락)
  → 베이스라인: 53.8% ("항상 상승" 예측)
  → 7개 모델 모두 49~54% 범위에 밀집
  → "머신러닝으로도 방향성 예측은 어렵다" 실증

원인 분석:
  → S&P 500 수익률 자기상관: 대부분 95% 신뢰구간 내
  → Signal-to-Noise Ratio가 극단적으로 낮음

문제 재정의: 변동성 레짐 분류
  → |수익률| 자기상관은 유의미 (변동성 군집 현상)
  → 베이스라인: 33.3%
  → Random Forest: 56.3% (4.4배 개선)
  → High 레짐 F1: 80.5% → 실용적 리스크 관리 가능

결론:
  "문제 정의" 자체가 분석 결과를 결정한다.
```

---

## Models Used

수업 범위 내 7가지 방법론을 복잡도 스펙트럼 순으로 배치:

```
단순 ──────────────────────────────────── 복잡
NB → LR(L2) → LR(L1) → KNN → DT → SVM → RF → MLP
```

| Model | Library | Key Parameter |
|-------|---------|---------------|
| Naive Bayes | `GaussianNB` | — |
| Logistic L2 | `LogisticRegression` | `penalty='l2', C=0.1` |
| Logistic L1 | `LogisticRegression` | `penalty='l1', C=0.1` |
| SVM | `SVC` | `kernel='rbf', C=1.0` |
| KNN | `KNeighborsClassifier` | `k=5` (Val F1 기준 최적) |
| Decision Tree | `DecisionTreeClassifier` | `max_depth=3` |
| Random Forest | `RandomForestClassifier` | `n_estimators=200` |
| MLP | `MLPClassifier` | `hidden=(64,32), early_stopping=True` |

---

## Trading Strategy Simulation

변동성 레짐 예측 결과를 활용한 포지션 전략:

```
Low  (안정) 예측 → 100% 투자
Mid  (보통) 예측 →  50% 투자
High (불안) 예측 →   0% (현금 보유)
```

> ⚠️ 본 시뮬레이션은 슬리피지, 세금, 시장충격 비용 미반영.  
> 실제 투자 조언이 아님.

---

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity. *Econometrica*, 50(4), 987–1007.

---

## License

This project is for academic purposes (Inha University, DSC3001).  
Data source: Yahoo Finance (`^GSPC`).
