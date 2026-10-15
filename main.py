import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 데이터를 바탕으로 선형 회귀 모델을 학습하여 기온 상승 속도를 비교하고 미래 및 과거의 연평균 기온을 예측합니다.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= 2025]
    
    yearly_df = df.groupby("연도")["평균기온"].agg(관측일수="count", 연평균기온="mean").reset_index()
    valid_yearly_df = yearly_df[yearly_df["관측일수"] >= 300].copy()
    return valid_yearly_df

data = load_and_process_data(DATA_URL)

# 1. 전체 기간 회귀 직선 및 상관계수 계산
years_all = data["연도"].to_numpy()
temps_all = data["연평균기온"].to_numpy()

corr_coef = np.corrcoef(years_all, temps_all)[0, 1]
slope_all, intercept_all = np.polyfit(years_all, temps_all, 1)
rise_per_100y_all = slope_all * 100

num_years = len(data)
start_year = int(data["연도"].min())
end_year = int(data["연도"].max())

# 2. 최근 20년 회귀 직선 계산
recent_start_year = end_year - 19
recent_data = data[data["연도"] >= recent_start_year]

years_recent = recent_data["연도"].to_numpy()
temps_recent = recent_data["연평균기온"].to_numpy()

slope_recent, intercept_recent = np.polyfit(years_recent, temps_recent, 1)
rise_per_100y_recent = slope_recent * 100
corr_recent = np.corrcoef(years_recent, temps_recent)[0, 1]

# 화면 구성 1: 분석 데이터 요약
st.subheader("📊 분석 데이터 요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석에 포함된 해의 개수", f"{num_years}개년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("전체 기간 상관계수 (r)", f"{corr_coef:.4f}")

st.divider()

# 화면 구성 2: 100년당 기온 상승량 나란히 비교
st.subheader("🔥 기온 상승 속도 비교 (100년에 몇 도 오르는가?)")

comp_col1, comp_col2 = st.columns(2)
recent_min_year = int(recent_data["연도"].min())
recent_count = len(recent_data)

with comp_col1:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 25px; background-color: #eef4f8; border-radius: 12px; border: 2px solid #457b9d;">
            <h3 style="color: #1d3557; margin-bottom: 5px;">전체 기간 ({start_year}년 ~ {end_year}년, {num_years}개년)</h3>
            <p style="font-size: 16px; color: #555555; margin: 0;">100년당 예상 기온 변화</p>
            <h1 style="color: #1d3557; font-size: 56px; margin: 15px 0;">{rise_per_100y_all:+.2f} °C</h1>
            <p style="font-size: 14px; color: #666666; margin: 0;">연간 기울기: {slope_all:+.4f} °C/년 (상관계수 r = {corr_coef:.2f})</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with comp_col2:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 25px; background-color: #fdf0ed; border-radius: 12px; border: 2px solid #e63946;">
            <h3 style="color: #e63946; margin-bottom: 5px;">최근 20년 ({recent_min_year}년 ~ {end_year}년, {recent_count}개년)</h3>
            <p style="font-size: 16px; color: #555555; margin: 0;">100년당 예상 기온 변화</p>
            <h1 style="color: #e63946; font-size: 56px; margin: 15px 0;">{rise_per_100y_recent:+.2f} °C</h1>
            <p style="font-size: 14px; color: #666666; margin: 0;">연간 기울기: {slope_recent:+.4f} °C/년 (상관계수 r = {corr_recent:.2f})</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# 화면 구성 3: 연도 선택 슬라이더 및 예측 결과 크게 표시
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요 (1900년 ~ 2100년)",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)

predicted_temp_all = slope_all * selected_year + intercept_all
predicted_temp_recent = slope_recent * selected_year + intercept_recent

st.markdown(
    f"""
    <div style="text-align: center; padding: 25px; background-color: #f8f9fa; border-radius: 12px; margin-top: 10px; margin-bottom: 20px; border: 1px solid #dee2e6;">
        <h3 style="color: #333333; margin-bottom: 5px;">선택한 연도: <b>{selected_year}년</b></h3>
        <p style="font-size: 18px; color: #555555; margin: 0;">전체 기간 회귀 직선 기준 예상 연평균 기온</p>
        <h1 style="color: #2a9d8f; font-size: 64px; margin: 10px 0;">{predicted_temp_all:.2f} °C</h1>
        <p style="font-size: 15px; color: #666666; margin: 0;">
            (참고: 최근 20년 추세가 그대로 이어진다고 가정할 경우 <b>{predicted_temp_recent:.2f} °C</b>)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# 화면 구성 4: Plotly 산점도 및 두 회귀 직선
st.subheader("📈 서울 연평균 기온 추이 및 회귀 직선 비교")

line_x_all = np.array([start_year, end_year])
line_y_all = slope_all * line_x_all + intercept_all

line_x_recent = np.array([recent_min_year, end_year])
line_y_recent = slope_recent * line_x_recent + intercept_recent

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["연평균기온"],
    mode="markers",
    name="연평균 기온 (관측값)",
    marker=dict(color="#8d99ae", size=8, opacity=0.7),
    hovertemplate="<b>%{x}년</b><br>연평균 기온: %{y:.2f}°C<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=line_x_all,
    y=line_y_all,
    mode="lines",
    name=f"전체 기간 회귀선 ({rise_per_100y_all:+.2f}°C / 100년)",
    line=dict(color="#1d3557", width=3),
    hovertemplate="<b>%{x}년 (전체 추세)</b><br>예상 기온: %{y:.2f}°C<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=line_x_recent,
    y=line_y_recent,
    mode="lines",
    name=f"최근 20년 회귀선 ({rise_per_100y_recent:+.2f}°C / 100년)",
    line=dict(color="#e63946", width=4, dash="solid"),
    hovertemplate="<b>%{x}년 (최근 20년 추세)</b><br>예상 기온: %{y:.2f}°C<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp_all],
    mode="markers",
    name=f"{selected_year}년 예측 기온 (전체 추세)",
    marker=dict(color="#2a9d8f", size=14, symbol="star", line=dict(color="#264653", width=2)),
    hovertemplate="<b>%{x}년 (예측값)</b><br>예상 기온: %{y:.2f}°C<extra></extra>"
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
