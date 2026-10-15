import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 데이터를 바탕으로 선형 회귀 모델을 학습하여 기온 상승 속도를 비교하고 미래 및 과거의 연평균 기온을 예측합니다.")

# 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = df.columns.str.strip()
    
    df['날짜'] = df['날짜'].astype(str).str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜', '평균기온'])
    
    df['연도'] = df['날짜'].dt.year
    
    # 조건 1: 기준 기간인 2025년까지의 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    # 연도별 관측일 수와 연평균 기온 계산
    yearly_df = df.groupby('연도')['평균기온'].agg(
        관측일수='count',
        연평균기온='mean'
    ).reset_index()
    
    # 조건 2: 관측일이 300일 이상인 해만 남기기
    valid_yearly_df = yearly_df[yearly_df['관측일수'] >= 300].copy()
    
    return valid_yearly_df

data = load_and_process_data(DATA_URL)

# --- 1. 전체 기간 회귀 직선 및 상관계수 계산 ---
years_all = data['연도'].values
temps_all = data['연평균기온'].values

corr_coef = np.corrcoef(years_all, temps_all)[0, 1]
slope_all, intercept_all = np.polyfit(years_all, temps_all, 1)
rise_per_100y_all = slope_all * 100

num_years = len(data)
start_year = int(data['연도'].min())
end_year = int(data['연도'].max())

# --- 2. 최근 20년 회귀 직선 계산 ---
recent_start_year = end_year - 19
recent_data = data[data['연도'] >= recent_start_year]

years_recent =
