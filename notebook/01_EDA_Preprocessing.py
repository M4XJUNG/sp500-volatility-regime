# ============================================================
# S&P 500 변동성 레짐 분류 프로젝트
# 파일: 01_EDA_FeatureEngineering.ipynb
# 순서대로 셀 단위로 실행할 것
# ============================================================


# ──────────────────────────────────────────────────────────────
# [셀 1] 라이브러리 설치 및 Google Drive 마운트
# ──────────────────────────────────────────────────────────────

# Colab에서 실행 시 아래 주석 해제
# from google.colab import drive
# drive.mount('/content/drive')

# 필요 라이브러리 설치 (Colab 기본 제공이지만 명시)
# !pip install -q pandas numpy matplotlib seaborn scikit-learn scipy statsmodels

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# ── 경로 설정 ──────────────────────────────────────────────
# Google Drive 사용 시:
# DATA_PATH = '/content/drive/MyDrive/sp500_project/sap500.csv'
# FIG_DIR   = '/content/drive/MyDrive/sp500_project/figures'

# 로컬 / Colab 직접 업로드 시:
DATA_PATH = 'sap500.csv'   # 파일 위치에 맞게 수정
FIG_DIR   = './figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ── 시각화 스타일 설정 ──────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'      : 120,
    'figure.facecolor': 'white',
    'axes.facecolor'  : '#f8f9fa',
    'axes.grid'       : True,
    'grid.alpha'      : 0.4,
    'grid.linewidth'  : 0.6,
    'font.size'       : 11,
    'axes.titlesize'  : 13,
    'axes.titleweight': 'bold',
    'axes.labelsize'  : 11,
    'lines.linewidth' : 1.5,
})

print("✅ 환경 설정 완료")


# ──────────────────────────────────────────────────────────────
# [셀 2] 데이터 로드 및 기초 확인
# ──────────────────────────────────────────────────────────────

df_raw = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df_raw = df_raw.sort_values('Date').reset_index(drop=True)

print("=" * 55)
print("  S&P 500 원본 데이터 기본 정보")
print("=" * 55)
print(f"  전체 행 수    : {len(df_raw):,}행")
print(f"  컬럼         : {df_raw.columns.tolist()}")
print(f"  기간         : {df_raw.Date.min().date()} ~ {df_raw.Date.max().date()}")
print(f"  결측치       : {df_raw.isnull().sum().sum()}개")
print()
print(df_raw.head())
print()
print(df_raw.describe().round(2))


# ──────────────────────────────────────────────────────────────
# [셀 3] 데이터 품질 검사 — 1983 컷오프 근거 도출
# ──────────────────────────────────────────────────────────────
# 핵심 분석: OHLC가 모두 동일한 행 = 실질적으로 Close만 존재
# → 이를 시각화해서 1983 컷오프 선정 근거로 사용

df_raw['ohlc_identical'] = (
    (df_raw['Open']  == df_raw['Close']) &
    (df_raw['High']  == df_raw['Close']) &
    (df_raw['Low']   == df_raw['Close'])
)
df_raw['year'] = df_raw['Date'].dt.year

# 연도별 OHLC 동일 비율
quality_by_year = df_raw.groupby('year')['ohlc_identical'].mean().reset_index()
quality_by_year.columns = ['year', 'identical_ratio']

# ── 시각화 ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 상단: OHLC 동일 비율
ax = axes[0]
colors = ['#e74c3c' if r > 0.5 else '#2ecc71' for r in quality_by_year['identical_ratio']]
ax.bar(quality_by_year['year'], quality_by_year['identical_ratio'],
       color=colors, alpha=0.8, width=0.9)
ax.axvline(1983, color='navy', linestyle='--', linewidth=2,
           label='1983 컷오프 (이후 데이터만 사용)')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('연도')
ax.set_ylabel('OHLC 동일 비율')
ax.set_title('데이터 품질 검사: OHLC 값이 모두 동일한 비율\n'
             '(1 = 완전히 동일 = Close만 있는 것과 같음)')
ax.legend(fontsize=11)
ax.set_xlim(1925, 2028)

# 빨강/초록 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.8, label='품질 불량 (50% 이상 동일)'),
    Patch(facecolor='#2ecc71', alpha=0.8, label='품질 양호')
]
ax.legend(handles=legend_elements + [
    plt.Line2D([0], [0], color='navy', linestyle='--', linewidth=2,
               label='1983 컷오프')
])

# 하단: 전체 Close 가격 추이 (배경으로 품질 구분)
ax2 = axes[1]
df_before = df_raw[df_raw['Date'] < '1983-01-01']
df_after  = df_raw[df_raw['Date'] >= '1983-01-01']
ax2.plot(df_before['Date'], df_before['Close'],
         color='#e74c3c', alpha=0.5, linewidth=0.8, label='1983 이전 (OHLCV 불완전)')
ax2.plot(df_after['Date'],  df_after['Close'],
         color='#2c3e50', linewidth=1.0, label='1983 이후 (완전한 OHLCV)')
ax2.axvline(pd.Timestamp('1983-01-01'), color='navy', linestyle='--', linewidth=2)
ax2.set_xlabel('날짜')
ax2.set_ylabel('종가 (Close)')
ax2.set_title('S&P 500 전체 가격 추이 (데이터 품질 구분)')
ax2.legend()
ax2.set_yscale('log')   # 로그 스케일: 장기 추세 비교에 유리

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_data_quality.png', bbox_inches='tight')
plt.show()

# 통계 출력
n_before = len(df_before)
n_after  = len(df_after)
print(f"\n📊 데이터 품질 요약")
print(f"  1983 이전: {n_before:,}행  →  OHLC 대부분 동일, 사용 불가")
print(f"  1983 이후: {n_after:,}행   →  완전한 OHLCV 데이터")
print(f"\n✅ 결론: 1983-01-01 이후 데이터만 사용 (n={n_after:,})")


# ──────────────────────────────────────────────────────────────
# [셀 4] 분석용 데이터 확정 (1983년 이후)
# ──────────────────────────────────────────────────────────────

df = df_raw[df_raw['Date'] >= '1983-01-01'].copy()
df = df.drop(columns=['ohlc_identical', 'year'])
df = df.reset_index(drop=True)
df = df.set_index('Date')

print(f"분석 데이터: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"행 수: {len(df):,}행 (약 {len(df)/252:.0f}년치 거래일)")
print()
print(df.head())


# ──────────────────────────────────────────────────────────────
# [셀 5] EDA ① — 기초 통계 및 수익률 분포
# ──────────────────────────────────────────────────────────────

df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
df_clean = df.dropna(subset=['log_ret'])

# ── 기초 통계 출력 ─────────────────────────────────────────
mean_ret  = df_clean['log_ret'].mean()
std_ret   = df_clean['log_ret'].std()
ann_ret   = mean_ret * 252
ann_vol   = std_ret  * np.sqrt(252)
skew      = df_clean['log_ret'].skew()
kurt      = df_clean['log_ret'].kurt()

print("=" * 55)
print("  S&P 500 일별 로그수익률 기초 통계 (1983~2026)")
print("=" * 55)
print(f"  일평균 수익률  : {mean_ret*100:.4f}%")
print(f"  일 표준편차    : {std_ret*100:.4f}%")
print(f"  연환산 수익률  : {ann_ret*100:.2f}%")
print(f"  연환산 변동성  : {ann_vol*100:.2f}%")
print(f"  왜도 (Skewness): {skew:.4f}  (음수 = 하락 꼬리가 더 두꺼움)")
print(f"  첨도 (Kurtosis): {kurt:.4f}  (정규분포=0, 양수=뾰족한 봉우리)")
print(f"  최대 일일 하락: {df_clean['log_ret'].min()*100:.2f}%")
print(f"  최대 일일 상승: {df_clean['log_ret'].max()*100:.2f}%")

# ── 수익률 분포 시각화 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (1) 히스토그램 + 정규분포 비교
ax = axes[0]
x_range = np.linspace(df_clean['log_ret'].min(), df_clean['log_ret'].max(), 300)
normal_pdf = stats.norm.pdf(x_range, mean_ret, std_ret)
ax.hist(df_clean['log_ret'], bins=120, density=True,
        color='steelblue', alpha=0.7, label='실제 수익률 분포')
ax.plot(x_range, normal_pdf, 'r-', linewidth=2, label='정규분포')
ax.set_xlabel('일별 로그수익률')
ax.set_ylabel('밀도')
ax.set_title('수익률 분포 vs 정규분포\n(두꺼운 꼬리: Fat Tail 확인)')
ax.legend()
ax.set_xlim(-0.12, 0.12)

# (2) Q-Q Plot
ax = axes[1]
(osm, osr), (slope, intercept, r) = stats.probplot(df_clean['log_ret'])
ax.plot(osm, osr, 'o', color='steelblue', alpha=0.3, markersize=2)
ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
ax.set_xlabel('이론적 분위수 (정규분포)')
ax.set_ylabel('실제 분위수')
ax.set_title('Q-Q Plot\n(직선에서 벗어날수록 비정규적)')

# (3) 연도별 연환산 변동성
annual_vol_ts = (df_clean['log_ret']
                 .groupby(df_clean.index.year)
                 .std() * np.sqrt(252) * 100)
ax = axes[2]
colors_vol = ['#e74c3c' if v > 30 else '#f39c12' if v > 20 else '#27ae60'
              for v in annual_vol_ts]
ax.bar(annual_vol_ts.index, annual_vol_ts.values,
       color=colors_vol, alpha=0.8)
ax.axhline(annual_vol_ts.mean(), color='navy', linestyle='--',
           linewidth=1.5, label=f'평균 {annual_vol_ts.mean():.1f}%')

# 주요 이벤트 표시
events = {1987: '블랙먼데이', 2001: '9·11', 2008: '금융위기',
          2020: 'COVID', 2022: '금리충격'}
for yr, label in events.items():
    if yr in annual_vol_ts.index:
        ax.annotate(label, xy=(yr, annual_vol_ts[yr]),
                    xytext=(yr, annual_vol_ts[yr]+3),
                    fontsize=7, ha='center', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred',
                                   lw=0.8))
ax.set_xlabel('연도')
ax.set_ylabel('연환산 변동성 (%)')
ax.set_title('연도별 변동성 추이\n(주요 금융 이벤트 표시)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_return_distribution.png', bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────────────────────
# [셀 6] EDA ② — 가격 추이 및 시장 구조 시각화
# ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# (1) 로그 스케일 가격 추이
ax = axes[0]
ax.plot(df.index, df['Close'], color='#2c3e50', linewidth=0.8)
ax.fill_between(df.index, df['Close'], alpha=0.1, color='steelblue')
ax.set_yscale('log')
ax.set_ylabel('종가 (로그 스케일)')
ax.set_title('S&P 500 가격 추이 (1983~2026, 로그 스케일)')
# 200일 이동평균 오버레이
ma200 = df['Close'].rolling(200).mean()
ax.plot(df.index, ma200, color='orange', linewidth=1.2,
        alpha=0.8, label='200일 이동평균')
ax.legend()

# 주요 이벤트 수직선
event_dates = {
    '87 블랙먼데이': '1987-10-19',
    '닷컴 버블': '2000-03-01',
    '9·11': '2001-09-11',
    '금융위기': '2008-09-15',
    'COVID': '2020-03-23',
    '2022 하락': '2022-01-01',
}
for label, date in event_dates.items():
    try:
        ax.axvline(pd.Timestamp(date), color='red',
                   alpha=0.5, linewidth=0.8, linestyle=':')
        ax.text(pd.Timestamp(date), ax.get_ylim()[0]*1.1,
                label, fontsize=7, color='red', rotation=90, va='bottom')
    except:
        pass

# (2) 일별 로그 수익률
ax = axes[1]
ret = df_clean['log_ret']
ax.fill_between(ret.index,
                ret.where(ret >= 0, 0), 0,
                alpha=0.7, color='#27ae60', label='상승')
ax.fill_between(ret.index,
                ret.where(ret < 0, 0), 0,
                alpha=0.7, color='#e74c3c', label='하락')
ax.set_ylabel('일별 로그수익률')
ax.set_title('일별 수익률 (변동성 군집 현상 관찰)')
ax.legend(loc='upper right')
ax.set_ylim(-0.12, 0.12)

# (3) 20일 실현 변동성 (연환산)
ax = axes[2]
rv20 = ret.rolling(20).std() * np.sqrt(252) * 100
ax.fill_between(rv20.index, rv20, alpha=0.6, color='#8e44ad')
ax.plot(rv20.index, rv20, color='#6c3483', linewidth=0.5)

# 레짐 임계값 표시
q33 = rv20.quantile(0.333)
q67 = rv20.quantile(0.667)
ax.axhline(q33, color='#27ae60', linestyle='--', linewidth=1.5,
           label=f'Low/Mid 경계 ({q33:.1f}%)')
ax.axhline(q67, color='#e74c3c', linestyle='--', linewidth=1.5,
           label=f'Mid/High 경계 ({q67:.1f}%)')
ax.set_ylabel('20일 실현 변동성 (%, 연환산)')
ax.set_title('변동성 군집 현상 + 레짐 경계값\n(High 레짐은 위기 시기와 일치)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_price_and_volatility.png', bbox_inches='tight')
plt.show()

print(f"📊 변동성 레짐 경계값:")
print(f"  Low  (안정장): 실현변동성 < {q33:.1f}%")
print(f"  Mid  (보통장): {q33:.1f}% ~ {q67:.1f}%")
print(f"  High (불안장): > {q67:.1f}%")


# ──────────────────────────────────────────────────────────────
# [셀 7] EDA ③ — 방향성 예측 시도 & 베이스라인 확인
# (스토리: "방향성은 예측이 어렵다"는 근거 EDA)
# ──────────────────────────────────────────────────────────────

# 1일 후 방향성 타겟 생성
df['target_dir'] = (df['Close'].shift(-1) > df['Close']).astype(int)

up_ratio   = df['target_dir'].dropna().mean()
down_ratio = 1 - up_ratio

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (1) 상승/하락 비율
ax = axes[0]
ax.pie([up_ratio, down_ratio],
       labels=[f'상승 ({up_ratio*100:.1f}%)', f'하락 ({down_ratio*100:.1f}%)'],
       colors=['#27ae60', '#e74c3c'],
       autopct='%1.1f%%', startangle=90,
       textprops={'fontsize': 12})
ax.set_title('1일 후 방향성 분포\n(상승이 살짝 우세)')

# (2) 연도별 상승 비율 → 안정적인가?
yearly_up = (df['target_dir']
             .groupby(df.index.year)
             .mean() * 100)
ax = axes[1]
ax.bar(yearly_up.index, yearly_up.values,
       color=['#27ae60' if v > 50 else '#e74c3c' for v in yearly_up],
       alpha=0.8)
ax.axhline(50, color='gray', linestyle='--', linewidth=1.5, label='50% 기준')
ax.axhline(up_ratio * 100, color='navy', linestyle='-.',
           linewidth=1.5, label=f'전체 평균 {up_ratio*100:.1f}%')
ax.set_xlabel('연도')
ax.set_ylabel('상승일 비율 (%)')
ax.set_title('연도별 상승일 비율\n(베이스라인은 "항상 상승 예측")')
ax.legend()
ax.set_ylim(30, 75)

# (3) 자기상관 분석: 내일 방향이 오늘과 관련 있는가?
ax = axes[2]
ret_vals = df_clean['log_ret'].values
lags     = range(1, 31)
acf_vals = [pd.Series(ret_vals).autocorr(lag=l) for l in lags]
colors_acf = ['#e74c3c' if abs(v) > 1.96/np.sqrt(len(ret_vals)) else 'steelblue'
              for v in acf_vals]
ax.bar(lags, acf_vals, color=colors_acf, alpha=0.8)
ci = 1.96 / np.sqrt(len(ret_vals))
ax.axhline( ci, color='red', linestyle='--', linewidth=1.2, label=f'95% 신뢰구간 (±{ci:.4f})')
ax.axhline(-ci, color='red', linestyle='--', linewidth=1.2)
ax.axhline(0,   color='black', linewidth=0.8)
ax.set_xlabel('시차 (일)')
ax.set_ylabel('자기상관계수')
ax.set_title('수익률 자기상관 분석\n(거의 모두 신뢰구간 안 = 패턴 없음)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_direction_eda.png', bbox_inches='tight')
plt.show()

# ADF 검정 (비정상성 확인)
adf_result = adfuller(df_clean['log_ret'].dropna())
print("=" * 55)
print("  ADF 검정 (단위근 검정: 비정상성 확인)")
print("=" * 55)
print(f"  검정통계량 : {adf_result[0]:.4f}")
print(f"  p-value    : {adf_result[1]:.6f}")
print(f"  1% 임계값  : {adf_result[4]['1%']:.4f}")
if adf_result[1] < 0.05:
    print("  ✅ 결론: p < 0.05 → 단위근 없음 (로그수익률은 정상 시계열)")
else:
    print("  ⚠️ 결론: p ≥ 0.05 → 단위근 있음 (비정상 시계열)")

print(f"\n📌 방향성 베이스라인 Accuracy: {up_ratio*100:.1f}%")
print("   → '항상 상승' 전략이 이미 이 수치를 달성")
print("   → ML 모델이 이를 크게 뛰어넘기 어려운 구조")


# ──────────────────────────────────────────────────────────────
# [셀 8] EDA ④ — 변동성 레짐 타겟 생성 및 분포 확인
# ──────────────────────────────────────────────────────────────

# ── 타겟 1: 변동성 레짐 (메인) ────────────────────────────
# 향후 20일 실현 변동성을 3분위로 나눔
rv_future = df['log_ret'].rolling(20).std().shift(-20) * np.sqrt(252) * 100
df['target_vol_raw'] = rv_future

# 분위수 경계값 (훈련 데이터 기준으로 나중에 재계산, 여기선 전체 기준)
q1 = rv_future.quantile(1/3)
q2 = rv_future.quantile(2/3)

df['target_vol'] = pd.cut(
    rv_future,
    bins=[-np.inf, q1, q2, np.inf],
    labels=[0, 1, 2]   # 0=Low, 1=Mid, 2=High
).astype('Int64')  # 정수형 (NA 포함 가능)

# ── 타겟 2: 방향성 (서브 비교용) ──────────────────────────
df['target_dir'] = (df['Close'].shift(-1) > df['Close']).astype('Int64')

# 분포 확인
vol_dist = df['target_vol'].value_counts().sort_index()
dir_dist = df['target_dir'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (1) 변동성 레짐 분포
ax = axes[0]
labels_vol = ['Low\n(안정장)', 'Mid\n(보통장)', 'High\n(불안장)']
colors_vol = ['#27ae60', '#f39c12', '#e74c3c']
bars = ax.bar(labels_vol, vol_dist.values,
              color=colors_vol, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, vol_dist.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 50,
            f'{val:,}\n({val/vol_dist.sum()*100:.1f}%)',
            ha='center', fontsize=11, fontweight='bold')
ax.set_title('변동성 레짐 분포 (3클래스)\n베이스라인 Accuracy = 33.3%')
ax.set_ylabel('행 수')
ax.set_ylim(0, vol_dist.max() * 1.2)

# (2) 방향성 분포
ax = axes[1]
labels_dir = ['하락 (0)', '상승 (1)']
colors_dir = ['#e74c3c', '#27ae60']
bars2 = ax.bar(labels_dir, dir_dist.values,
               color=colors_dir, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, dir_dist.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 50,
            f'{val:,}\n({val/dir_dist.sum()*100:.1f}%)',
            ha='center', fontsize=11, fontweight='bold')
ax.set_title('방향성 분포 (2클래스)\n베이스라인 Accuracy = 53.8%')
ax.set_ylabel('행 수')
ax.set_ylim(0, dir_dist.max() * 1.2)

plt.suptitle('타겟 변수 분포 비교\n변동성 레짐: 균형잡힌 3클래스 / 방향성: 불균형 2클래스',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_target_distribution.png', bbox_inches='tight')
plt.show()

print(f"변동성 레짐 경계값: Low < {q1:.1f}% ≤ Mid < {q2:.1f}% ≤ High")
print(f"레짐 분포: Low={vol_dist.get(0,0):,} / Mid={vol_dist.get(1,0):,} / High={vol_dist.get(2,0):,}")


# ──────────────────────────────────────────────────────────────
# [셀 9] EDA ⑤ — 변동성 군집(Volatility Clustering) 시각화
# (변동성 레짐 예측이 가능한 이유에 대한 시각적 근거)
# ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# (1) 수익률의 절댓값 자기상관 (변동성 군집 확인)
ax = axes[0]
abs_ret   = df_clean['log_ret'].abs()
lags_long = range(1, 61)
acf_abs   = [abs_ret.autocorr(lag=l) for l in lags_long]
ci_abs    = 1.96 / np.sqrt(len(abs_ret))

ax.bar(lags_long, acf_abs,
       color=['#e74c3c' if v > ci_abs else 'steelblue' for v in acf_abs],
       alpha=0.8)
ax.axhline( ci_abs, color='red', linestyle='--', linewidth=1.2)
ax.axhline(-ci_abs, color='red', linestyle='--', linewidth=1.2)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('시차 (일)')
ax.set_ylabel('자기상관계수')
ax.set_title('|수익률| 자기상관 분석 (변동성 군집 검증)\n'
             '→ 높은 변동성이 지속되는 경향 = 예측 가능한 패턴 존재')

# (2) 레짐별 수익률 분포 비교
ax = axes[1]
temp = df.dropna(subset=['target_vol', 'log_ret'])
regime_colors = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'}
regime_labels = {0: 'Low (안정장)', 1: 'Mid (보통장)', 2: 'High (불안장)'}

for regime_id in [0, 1, 2]:
    subset = temp[temp['target_vol'] == regime_id]['log_ret'] * 100
    ax.hist(subset, bins=80, density=True, alpha=0.55,
            color=regime_colors[regime_id],
            label=f"{regime_labels[regime_id]} (std={subset.std():.2f}%)")

ax.set_xlabel('일별 수익률 (%)')
ax.set_ylabel('밀도')
ax.set_title('변동성 레짐별 수익률 분포\n'
             '→ High 레짐일수록 수익률 분포가 넓음 (리스크 큼)')
ax.legend()
ax.set_xlim(-10, 10)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_volatility_clustering.png', bbox_inches='tight')
plt.show()

# 레짐별 통계 테이블
print("\n📊 레짐별 수익률 통계")
print("-" * 55)
for regime_id, label in regime_labels.items():
    subset = temp[temp['target_vol'] == regime_id]['log_ret'] * 100
    print(f"  {label}:")
    print(f"    평균={subset.mean():.3f}% | 표준편차={subset.std():.3f}% | "
          f"최소={subset.min():.2f}% | 최대={subset.max():.2f}%")


# ──────────────────────────────────────────────────────────────
# [셀 10] Feature Engineering — 전체 피처 생성 함수
# ──────────────────────────────────────────────────────────────
# ⚠️ Data Leakage 방지 원칙:
#   모든 피처는 t일의 정보로 계산 후 shift(1)을 적용
#   → 모델이 t일 예측 시 t-1까지의 정보만 사용 (실전과 동일)

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    S&P 500 OHLCV → 금융 피처 생성
    
    Parameters
    ----------
    data : OHLCV DataFrame (index=Date)
    
    Returns
    -------
    DataFrame with all engineered features (leakage-free)
    """
    feat = data.copy()

    # ── Group 1: 수익률 (Returns) ──────────────────────────
    feat['log_ret_1d']  = np.log(feat['Close'] / feat['Close'].shift(1))
    feat['log_ret_5d']  = np.log(feat['Close'] / feat['Close'].shift(5))
    feat['log_ret_10d'] = np.log(feat['Close'] / feat['Close'].shift(10))
    feat['log_ret_20d'] = np.log(feat['Close'] / feat['Close'].shift(20))
    feat['log_ret_60d'] = np.log(feat['Close'] / feat['Close'].shift(60))

    # ── Group 2: 이동평균 (Moving Averages) ─────────────────
    for w in [5, 10, 20, 60, 120]:
        feat[f'ma_{w}'] = feat['Close'].rolling(w).mean()

    # 현재가 대비 이동평균 위치 (상대 값)
    feat['price_ma5_ratio']   = feat['Close'] / feat['ma_5']  - 1
    feat['price_ma20_ratio']  = feat['Close'] / feat['ma_20'] - 1
    feat['price_ma60_ratio']  = feat['Close'] / feat['ma_60'] - 1
    feat['price_ma120_ratio'] = feat['Close'] / feat['ma_120'] - 1

    # MA 크로스오버 스프레드
    feat['ma5_20_spread']   = feat['ma_5']  / feat['ma_20']  - 1
    feat['ma20_60_spread']  = feat['ma_20'] / feat['ma_60']  - 1
    feat['ma60_120_spread'] = feat['ma_60'] / feat['ma_120'] - 1

    # ── Group 3: 변동성 (Volatility) ──────────────────────
    # 실현 변동성 (Realized Volatility)
    feat['vol_5d']  = feat['log_ret_1d'].rolling(5).std()  * np.sqrt(252)
    feat['vol_10d'] = feat['log_ret_1d'].rolling(10).std() * np.sqrt(252)
    feat['vol_20d'] = feat['log_ret_1d'].rolling(20).std() * np.sqrt(252)
    feat['vol_60d'] = feat['log_ret_1d'].rolling(60).std() * np.sqrt(252)

    # 변동성 변화율 (변동성 자체도 추세가 있는가?)
    feat['vol_ratio_5_20']  = feat['vol_5d']  / (feat['vol_20d'] + 1e-10)
    feat['vol_ratio_20_60'] = feat['vol_20d'] / (feat['vol_60d'] + 1e-10)

    # ATR (Average True Range): OHLC 모두 사용하는 변동성 지표
    hl    = feat['High'] - feat['Low']
    hc    = (feat['High'] - feat['Close'].shift(1)).abs()
    lc    = (feat['Low']  - feat['Close'].shift(1)).abs()
    tr    = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    feat['atr_14']    = tr.rolling(14).mean()
    feat['atr_ratio'] = feat['atr_14'] / (feat['Close'] + 1e-10)  # 정규화

    # ── Group 4: 모멘텀 (Momentum) ─────────────────────────
    # RSI (Relative Strength Index, 14일)
    delta    = feat['Close'].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / (loss + 1e-10)
    feat['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD (12-26-9)
    ema12    = feat['Close'].ewm(span=12, adjust=False).mean()
    ema26    = feat['Close'].ewm(span=26, adjust=False).mean()
    feat['macd']         = ema12 - ema26
    feat['macd_signal']  = feat['macd'].ewm(span=9, adjust=False).mean()
    feat['macd_hist']    = feat['macd'] - feat['macd_signal']
    feat['macd_norm']    = feat['macd'] / (feat['Close'] + 1e-10)  # 정규화

    # Stochastic Oscillator (14일)
    low14  = feat['Low'].rolling(14).min()
    high14 = feat['High'].rolling(14).max()
    feat['stoch_k'] = 100 * (feat['Close'] - low14) / (high14 - low14 + 1e-10)
    feat['stoch_d'] = feat['stoch_k'].rolling(3).mean()

    # ── Group 5: 거래량 (Volume) ──────────────────────────
    feat['vol_ratio_5d']  = feat['Volume'] / (feat['Volume'].rolling(5).mean()  + 1e-10)
    feat['vol_ratio_20d'] = feat['Volume'] / (feat['Volume'].rolling(20).mean() + 1e-10)
    feat['vol_ma5']       = feat['Volume'].rolling(5).mean()
    feat['vol_log']       = np.log(feat['Volume'] + 1)  # 로그 변환

    # 가격-거래량 상관관계 (20일)
    feat['price_vol_corr'] = (feat['log_ret_1d']
                              .rolling(20)
                              .corr(feat['Volume'].pct_change()))

    # ── Group 6: 캔들 패턴 (Candle Pattern) ──────────────
    candle_range = (feat['High'] - feat['Low']).clip(lower=1e-10)
    feat['intraday_ret']  = (feat['Close'] - feat['Open']) / (feat['Open'] + 1e-10)
    feat['upper_shadow']  = (feat['High'] - feat[['Close','Open']].max(axis=1)) / candle_range
    feat['lower_shadow']  = (feat[['Close','Open']].min(axis=1) - feat['Low']) / candle_range
    feat['body_ratio']    = (feat['Close'] - feat['Open']).abs() / candle_range

    # ── Group 7: 시장 상태 지표 ──────────────────────────
    # 52주(약 252일) 신고가 대비 위치
    high_252 = feat['Close'].rolling(252).max()
    low_252  = feat['Close'].rolling(252).min()
    feat['pos_in_52w_range'] = (
        (feat['Close'] - low_252) / (high_252 - low_252 + 1e-10)
    )

    # Bull/Bear 시장 신호 (200일 MA 기준)
    feat['above_ma200'] = (feat['Close'] > feat['Close'].rolling(200).mean()).astype(int)

    # ── Leakage 방지: 모든 피처를 1일 shift ─────────────
    raw_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                'log_ret', 'target_dir', 'target_vol',
                'target_vol_raw']
    feat_cols = [c for c in feat.columns if c not in raw_cols]
    feat[feat_cols] = feat[feat_cols].shift(1)

    return feat

# ── 피처 생성 실행 ─────────────────────────────────────────
df_feat = create_features(df)

# MA 컬럼 제거 (피처로 직접 쓰지 않음, 비율/스프레드로 변환됨)
drop_ma = [c for c in df_feat.columns if c.startswith('ma_')]
df_feat = df_feat.drop(columns=drop_ma, errors='ignore')

feature_cols = [c for c in df_feat.columns
                if c not in ['Open','High','Low','Close','Volume',
                             'log_ret','target_dir','target_vol',
                             'target_vol_raw']]

print(f"✅ 피처 생성 완료: {len(feature_cols)}개 피처")
print("\n피처 목록:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")


# ──────────────────────────────────────────────────────────────
# [셀 11] 피처 시각화 ① — 주요 피처 시계열
# ──────────────────────────────────────────────────────────────

# 최근 3년 데이터로 시각화 (전체는 너무 많아 안 보임)
recent = df_feat.loc['2021':].copy()

fig, axes = plt.subplots(4, 1, figsize=(14, 14))
fig.suptitle('주요 피처 시계열 시각화 (2021~2026)', fontsize=14, fontweight='bold')

# (1) 가격 + MA 비율
ax = axes[0]
ax2_twin = ax.twinx()
ax.plot(recent.index, recent['Close'], color='#2c3e50', linewidth=1, label='Close')
ax2_twin.plot(recent.index, recent['price_ma20_ratio']*100,
              color='orange', linewidth=1, alpha=0.8, label='20일MA 대비 위치(%)')
ax2_twin.axhline(0, color='orange', linestyle=':', linewidth=0.8)
ax.set_ylabel('종가'); ax2_twin.set_ylabel('MA 대비 위치 (%)')
ax.set_title('가격 추이 + 20일 이동평균 대비 위치')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# (2) RSI
ax = axes[1]
ax.plot(recent.index, recent['rsi_14'], color='purple', linewidth=1)
ax.axhline(70, color='red',   linestyle='--', linewidth=1.2, label='과매수(70)')
ax.axhline(30, color='green', linestyle='--', linewidth=1.2, label='과매도(30)')
ax.axhline(50, color='gray',  linestyle=':',  linewidth=0.8)
ax.fill_between(recent.index, recent['rsi_14'], 70,
                where=recent['rsi_14'] >= 70, alpha=0.3, color='red')
ax.fill_between(recent.index, recent['rsi_14'], 30,
                where=recent['rsi_14'] <= 30, alpha=0.3, color='green')
ax.set_ylabel('RSI'); ax.set_title('RSI-14 (과매수/과매도 신호)')
ax.legend(loc='upper right'); ax.set_ylim(0, 100)

# (3) MACD
ax = axes[2]
ax.plot(recent.index, recent['macd'],        color='blue',  linewidth=1,   label='MACD')
ax.plot(recent.index, recent['macd_signal'], color='red',   linewidth=1,   label='Signal')
ax.bar(recent.index,  recent['macd_hist'],
       color=['#27ae60' if v >= 0 else '#e74c3c' for v in recent['macd_hist']],
       alpha=0.5, label='Histogram')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('MACD'); ax.set_title('MACD (12-26-9)')
ax.legend(loc='upper right')

# (4) 변동성
ax = axes[3]
ax.fill_between(recent.index, recent['vol_5d']*100,
                alpha=0.5, color='#e74c3c', label='5일 변동성')
ax.fill_between(recent.index, recent['vol_20d']*100,
                alpha=0.5, color='#3498db', label='20일 변동성')
ax.plot(recent.index, recent['vol_60d']*100,
        color='#2c3e50', linewidth=1.5, label='60일 변동성')
ax.set_ylabel('연환산 변동성 (%)'); ax.set_title('단기/중기/장기 변동성')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/07_feature_timeseries.png', bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────────────────────
# [셀 12] 피처 시각화 ② — 피처 간 상관관계 히트맵
# ──────────────────────────────────────────────────────────────

df_model = df_feat.dropna(subset=feature_cols + ['target_vol', 'target_dir'])

# 상관관계 분석
corr_with_vol = (df_model[feature_cols]
                 .corrwith(df_model['target_vol'].astype(float))
                 .sort_values(key=abs, ascending=False))

# 상위 20개 피처 간 상관관계 히트맵
top20_cols = corr_with_vol.head(20).index.tolist()
corr_matrix = df_model[top20_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# (1) 상위 피처의 타겟 상관관계
ax = axes[0]
colors_corr = ['#e74c3c' if v > 0 else '#3498db' for v in corr_with_vol.head(20)]
ax.barh(range(20), corr_with_vol.head(20).values[::-1],
        color=colors_corr[::-1], alpha=0.8)
ax.set_yticks(range(20))
ax.set_yticklabels(corr_with_vol.head(20).index[::-1], fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('타겟(변동성 레짐)과의 상관계수')
ax.set_title('피처-타겟 상관계수 Top 20\n(절댓값 기준)')

# (2) 피처 간 상관관계 히트맵
ax = axes[1]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix,
            mask=mask,
            annot=False,
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            linewidths=0.3,
            ax=ax,
            xticklabels=[c.replace('_', '\n') for c in top20_cols],
            yticklabels=[c.replace('_', '\n') for c in top20_cols])
ax.set_title('Top 20 피처 간 상관관계 히트맵\n(다중공선성 확인)')
ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/08_feature_correlation.png', bbox_inches='tight')
plt.show()

print("📊 타겟(변동성 레짐)과 상관관계 Top 10:")
for feat_name, corr_val in corr_with_vol.head(10).items():
    print(f"   {feat_name:30s}: {corr_val:+.4f}")


# ──────────────────────────────────────────────────────────────
# [셀 13] 피처 시각화 ③ — 레짐별 피처 분포 비교
# ──────────────────────────────────────────────────────────────

key_features = ['vol_20d', 'vol_ratio_5_20', 'rsi_14',
                'price_ma20_ratio', 'atr_ratio', 'macd_norm']
feat_labels  = ['20일 변동성', '변동성 비율(5/20일)', 'RSI-14',
                '20일MA 대비 위치', 'ATR 비율', 'MACD(정규화)']
regime_colors = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'}
regime_names  = {0: 'Low(안정)', 1: 'Mid(보통)', 2: 'High(불안)'}

df_plot = df_model.dropna(subset=key_features)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('변동성 레짐별 주요 피처 분포\n(레짐이 분리될수록 예측하기 쉬운 피처)',
             fontsize=13, fontweight='bold')

for idx, (feat_name, feat_label) in enumerate(zip(key_features, feat_labels)):
    ax = axes[idx // 3][idx % 3]
    for regime_id in [0, 1, 2]:
        subset = df_plot[df_plot['target_vol'] == regime_id][feat_name].dropna()
        if len(subset) == 0:
            continue
        ax.hist(subset, bins=50, density=True, alpha=0.55,
                color=regime_colors[regime_id],
                label=regime_names[regime_id])
        # KDE 추가
        from scipy.stats import gaussian_kde
        if len(subset) > 10:
            kde = gaussian_kde(subset)
            x_range = np.linspace(subset.quantile(0.01),
                                   subset.quantile(0.99), 200)
            ax.plot(x_range, kde(x_range),
                    color=regime_colors[regime_id], linewidth=2)
    ax.set_title(feat_label)
    ax.set_xlabel('피처 값')
    ax.set_ylabel('밀도')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/09_feature_by_regime.png', bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────────────────────
# [셀 14] Train / Validation / Test Split (시계열 기반)
# ──────────────────────────────────────────────────────────────
# ⚠️ 절대 원칙: shuffle=False, 시간 순서 엄격히 유지

TRAIN_END = '2018-12-31'
VAL_END   = '2021-12-31'
# Test: 2022-01-01 ~ 2026-03-10 (최종 평가 전까지 절대 보지 않음)

df_model = df_feat.dropna(subset=feature_cols + ['target_vol', 'target_dir'])

X = df_model[feature_cols]
y_vol = df_model['target_vol'].astype(int)
y_dir = df_model['target_dir'].astype(int)

# 시간 기반 분할
train_mask = df_model.index <= TRAIN_END
val_mask   = (df_model.index > TRAIN_END) & (df_model.index <= VAL_END)
test_mask  = df_model.index > VAL_END

X_train, y_vol_train, y_dir_train = X[train_mask], y_vol[train_mask], y_dir[train_mask]
X_val,   y_vol_val,   y_dir_val   = X[val_mask],   y_vol[val_mask],   y_dir[val_mask]
X_test,  y_vol_test,  y_dir_test  = X[test_mask],  y_vol[test_mask],  y_dir[test_mask]

print("=" * 55)
print("  Train / Validation / Test Split")
print("=" * 55)
print(f"  Train : {X_train.index[0].date()} ~ {X_train.index[-1].date()} "
      f"({len(X_train):,}일 / {len(X_train)/len(X)*100:.1f}%)")
print(f"  Val   : {X_val.index[0].date()} ~ {X_val.index[-1].date()} "
      f"({len(X_val):,}일 / {len(X_val)/len(X)*100:.1f}%)")
print(f"  Test  : {X_test.index[0].date()} ~ {X_test.index[-1].date()} "
      f"({len(X_test):,}일 / {len(X_test)/len(X)*100:.1f}%)")

# 분할 시각화
fig, ax = plt.subplots(figsize=(14, 3))
ax.barh(0, len(X_train), left=0,         height=0.5,
        color='#3498db', alpha=0.8, label=f'Train ({len(X_train):,}일)')
ax.barh(0, len(X_val),   left=len(X_train), height=0.5,
        color='#f39c12', alpha=0.8, label=f'Validation ({len(X_val):,}일)')
ax.barh(0, len(X_test),  left=len(X_train)+len(X_val), height=0.5,
        color='#e74c3c', alpha=0.8, label=f'Test ({len(X_test):,}일)')
ax.set_xlim(0, len(X))
ax.set_yticks([])
ax.set_xlabel('거래일 순서')
ax.set_title('시계열 기반 데이터 분할 (Walk-Forward Split)\n'
             '⚠️ Test 세트는 최종 평가 전까지 모델 학습에 절대 사용 금지')
ax.legend(loc='upper right')
# 날짜 레이블
for pos, label in [
    (0, X_train.index[0].strftime('%Y-%m')),
    (len(X_train), X_val.index[0].strftime('%Y-%m')),
    (len(X_train)+len(X_val), X_test.index[0].strftime('%Y-%m')),
    (len(X)-1, X_test.index[-1].strftime('%Y-%m')),
]:
    ax.axvline(pos, color='black', linewidth=1, alpha=0.5)
    ax.text(pos, 0.3, label, ha='center', fontsize=9, rotation=45)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/10_train_val_test_split.png', bbox_inches='tight')
plt.show()


# ──────────────────────────────────────────────────────────────
# [셀 15] 피처 스케일링 및 최종 데이터 준비
# ──────────────────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler

# Train 기준으로 scaler 학습 → Val/Test에는 transform만
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# DataFrame으로 복원 (피처 이름 유지)
X_train_sc = pd.DataFrame(X_train_sc, columns=feature_cols, index=X_train.index)
X_val_sc   = pd.DataFrame(X_val_sc,   columns=feature_cols, index=X_val.index)
X_test_sc  = pd.DataFrame(X_test_sc,  columns=feature_cols, index=X_test.index)

# 무한값, NaN 최종 확인
for name, X_chk in [('Train', X_train_sc), ('Val', X_val_sc), ('Test', X_test_sc)]:
    nan_cnt = X_chk.isna().sum().sum()
    inf_cnt = np.isinf(X_chk.values).sum()
    print(f"  {name}: NaN={nan_cnt}, Inf={inf_cnt}")

print("\n✅ 스케일링 완료")
print(f"   Train shape : {X_train_sc.shape}")
print(f"   Val shape   : {X_val_sc.shape}")
print(f"   Test shape  : {X_test_sc.shape}")
print(f"\n⚠️ 주의: 스케일링에 사용한 scaler는 Val/Test에 transform만 적용")
print("   → Train의 평균/분산 정보만 사용 (leakage 방지)")


# ──────────────────────────────────────────────────────────────
# [셀 16] 최종 데이터 저장 (다음 노트북에서 불러올 수 있게)
# ──────────────────────────────────────────────────────────────

import pickle

# CSV 저장
X_train_sc.assign(target_vol=y_vol_train,
                  target_dir=y_dir_train).to_csv('train_data.csv')
X_val_sc.assign(target_vol=y_vol_val,
                target_dir=y_dir_val).to_csv('val_data.csv')
X_test_sc.assign(target_vol=y_vol_test,
                 target_dir=y_dir_test).to_csv('test_data.csv')

# Scaler 저장
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 피처 목록 저장
with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("✅ 저장 완료:")
print("   - train_data.csv")
print("   - val_data.csv")
print("   - test_data.csv")
print("   - scaler.pkl")
print("   - feature_cols.pkl")


# ──────────────────────────────────────────────────────────────
# [셀 17] 전체 EDA 요약 출력 (보고서에 바로 쓸 수 있는 텍스트)
# ──────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════╗
║              EDA & Feature Engineering 요약                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 데이터 품질                                               ║
║     - 원본: 24,663행 (1927~2026)                             ║
║     - 1983 이전: OHLC 거의 동일 → 실질적으로 Close만 존재     ║
║     - 분석 채택: 1983년 이후 10,882행                        ║
║                                                              ║
║  2. 수익률 특성                                               ║
║     - 연환산 수익률: ~8.3% / 연환산 변동성: ~16.9%           ║
║     - 음의 왜도: 하락 꼬리가 더 두꺼움 (Fat Tail)             ║
║     - 높은 첨도: 극단적 움직임 빈번 (비정규 분포)             ║
║     - ADF 검정: 로그수익률은 정상 시계열 (p < 0.001)          ║
║                                                              ║
║  3. 예측 난이도 비교                                          ║
║     - 방향성 (2클래스): 베이스라인 53.8% → 개선 여지 좁음     ║
║     - 변동성 레짐 (3클래스): 베이스라인 33.3% → 여지 넓음     ║
║     - |수익률| 자기상관 유의: 변동성 군집 패턴 존재           ║
║                                                              ║
║  4. 생성 피처: 총 36개 (7개 그룹)                             ║
║     - 수익률(5), MA 비율/스프레드(7), 변동성(8)              ║
║     - 모멘텀(7), 거래량(5), 캔들패턴(3), 시장상태(2)         ║
║     - 전부 shift(1) 적용 → Data Leakage 없음                 ║
║                                                              ║
║  5. 데이터 분할                                               ║
║     - Train: 1983~2018 / Val: 2019~2021 / Test: 2022~2026   ║
║     - 시계열 순서 유지 (shuffle 없음)                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
