import streamlit as st

from src.data_loader import load_data

st.set_page_config(
    page_title="KBO 타자 WAR 분석 플랫폼",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ KBO 타자 WAR 분석 및 예측 플랫폼")
st.markdown(
    """
**규정타석 이상 KBO 타자**의 WAR(Wins Above Replacement)를 분석하고,
이전 시즌 성적을 기반으로 **다음 시즌 WAR를 예측**하는 웹 서비스입니다.

| 페이지 | 내용 |
|---|---|
| 📊 **1_EDA** | 데이터 탐색 — 분포, 상관관계, 결측치 확인 |
| 📈 **2_시각화** | 선수별 커리어 추이, OPS·HR vs WAR 산점도 |
| 🤖 **3_모델_서비스** | WAR 예측 · 모델 비교 · 유사 선수 추천 |

왼쪽 사이드바에서 페이지를 선택하세요.
"""
)

st.divider()

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 레코드", f"{len(df):,}행")
col2.metric("선수 수 (통산)", f"{df['Id'].nunique():,}명")
col3.metric("수록 시즌", f"{df['Year'].min()} – {df['Year'].max()}")
col4.metric("규정타석(PA≥300) 레코드", f"{(df['PA']>=300).sum():,}행")

st.subheader("데이터 미리보기 (최근 5년)")
recent = df[df["Year"] >= df["Year"].max() - 4].sort_values(["Year", "WAR"], ascending=[False, False])
display_cols = ["Year", "Name", "Team", "Pos.", "Age", "G", "PA", "AVG", "OPS", "HR", "RBI", "SB", "wRC+", "WAR"]
st.dataframe(recent[display_cols].reset_index(drop=True), use_container_width=True, height=320)

st.caption("데이터 출처: KBO Player Dataset (1982-2025) — Kaggle, CC0 Public Domain")
