import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 데이터를 바탕으로 선형 회귀 모델을 학습하여 미래 및 과거의 연평균 기온을 예측합니다.")

# 데이터 로드 및 전처리 (캐싱을 통해 속도 향상)
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data(url):
    # UTF-8 인코딩으로 데이터 불러오기
    df = pd.read_csv(url, encoding="utf-8")
    
    # 열 이름 앞뒤 공백 제거 (데이터 안정성 확보)
    df.columns = df.columns.str.strip()
    
    # 날짜 데이터 앞뒤 공백 및 탭 문자 제거 후 datetime 변환
    df['날짜'] = df['날짜'].astype(str).str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    
    # 날짜와 평균기온 결측치 제거
    df = df.dropna(subset=['날짜', '평균기온'])
    
    # 연도 추출
    df['연도'] = df['날짜'].dt.year
    
    # 조건 1: 기준 기간인 2025년까지의 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    # 연도별 관측일 수(count)와 연평균 기온(mean) 계산
    yearly_df = df.groupby('연도')['평균기온'].agg(
        관측일수='count',
        연평균기온='mean'
    ).reset_index()
    
    # 조건 2: 관측일이 300일 이상인 해만 남기기
    valid_yearly_df = yearly_df[yearly_df['관측일수'] >= 300].copy()
    
    return valid_yearly_df

# 데이터 불러오기
data = load_and_process_data(DATA_URL)

# 회귀 직선 및 상관계수 계산
years = data['연도'].values
temps = data['연평균기온'].values

# 상관계수 (피어슨 상관계수)
corr_coef = np.corrcoef(years, temps)[0, 1]

# 1차 선형 회귀 (y = slope * x + intercept)
slope, intercept = np.polyfit(years, temps, 1)

# 데이터 요약 정보 계산
num_years = len(data)
start_year = int(data['연도'].min())
end_year = int(data['연도'].max())

# --- 화면 구성: 요약 정보 표시 ---
st.subheader("📊 분석 데이터 요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석에 포함된 해의 개수", f"{num_years}개년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("연도-기온 상관계수 (r)", f"{corr_coef:.4f}")

st.divider()

# --- 화면 구성: 연도 선택 슬라이더 및 예측 결과 크게 표시 ---
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요 (1900년 ~ 2100년)",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)

# 선택한 연도의 예상 기온 계산
predicted_temp = slope * selected_year + intercept

# 예상 기온을 화면에 크게 표시
st.markdown(
    f"""
    <div style="text-align: center; padding: 25px; background-color: #f0f4f8; border-radius: 12px; margin-top: 10px; margin-bottom: 20px;">
        <h3 style="color: #333333; margin-bottom: 5px;">선택한 연도: <b>{selected_year}년</b></h3>
        <p style="font-size: 18px; color: #555555; margin: 0;">회귀 모델 기반 예상 연평균 기온</p>
        <h1 style="color: #e63946; font-size: 64px; margin: 10px 0;">{predicted_temp:.2f} °C</h1>
        <p style="font-size: 14px; color: #777777; margin: 0;">회귀식: 기온 = {slope:.4f} × 연도 + ({intercept:.2f})</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- 화면 구성: Plotly 산점도 및 회귀 직선 ---
st.subheader("📈 서울 연평균 기온 추이 및 회귀 직선")

# 회귀 직선 데이터 생성 (분석 데이터의 시작~끝 연도 구간)
line_x = np.array([start_year, end_year])
line_y = slope * line_x + intercept

fig = go.Figure()

# 1. 실제 연평균 기온 산점도
fig.add_trace(go.Scatter(
    x=data['연도'],
    y=data['연평균기온'],
    mode='markers',
    name='연평균 기온 (관측값)',
    marker=dict(color='#457b9d', size=8, opacity=0.8),
    hovertemplate='<b>%{x}년</b><br>연평균 기온: %{y:.2f}°C<extra></extra>'
))

# 2. 회귀 직선
fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode='lines',
    name=f'회귀 직선 (r = {corr_coef:.2f})',
    line=dict(color='#e63946', width=3),
    hovertemplate='<b>%{x}년 (회귀선)</b><br>예상 기온: %{y:.2f}°C<extra></extra>'
))

# 3. 슬라이더로 선택한 연도의 예측 지점 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode='markers',
    name=f'{selected_year}년 예측 기온',
    marker=dict(color='#f4a261', size=14, symbol='star', line=dict(color='#e76f51', width=2)),
    hovertemplate='<b>%{x}년 (예측값)</b><br>예상 기온: %{y:.2f}°C<extra></extra>'
))

fig.update_layout(
    title=f"서울 연평균 기온 산점도 및 선형 회귀선 ({start_year}년 ~ {end_year}년, 총 {num_years}개년)",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=550,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
