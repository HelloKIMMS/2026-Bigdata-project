import streamlit as st

with st.sidebar:
    st.markdown("# ⚾ KBO WAR Proeject")
    st.caption("Big Data Analysis & Prediction")
    
    st.divider()

# 3. 용어 사전 (전문성 강화 및 청중 이해 보조)
    with st.expander("📝 용어 사전 (Terminology)", expanded=False):
        st.markdown("#### **[핵심 종합 지표]**")
        st.markdown("""
        - **WAR**: 대체 선수 대비 승리 기여도 (종합)
        - **oWAR**: 공격 기여 WAR (Offensive)
        - **dWAR**: 수비 기여 WAR (Defensive)
        - **wRC+**: 조정 득점 생산력 (100이 리그 평균)
        """)
        
        st.markdown("#### **[타격 비율 지표]**")
        st.markdown("""
        - **AVG**: 타율 (Batting Average)
        - **OBP**: 출루율 (On-Base Percentage)
        - **SLG**: 장타율 (Slugging Percentage)
        - **OPS**: 출루율 + 장타율
        """)
        
        st.markdown("#### **[타격 카운트 지표]**")
        st.markdown("""
        - **H / 2B / 3B**: 안타 / 2루타 / 3루타
        - **HR**: 홈런 (Home Run)
        - **RBI**: 타점 (Runs Batted In)
        - **TB**: 루타수 (Total Bases)
        - **BB**: 볼넷 (Bases on Balls)
        - **SO**: 삼진 (Strikeouts)
        """)
        
        st.markdown("#### **[기본 기록 및 기타]**")
        st.markdown("""
        - **SB**: 도루 (Stolen Bases)
        - **R**: 득점 (Runs Scored)
        - **PA**: 타석 수 (Plate Appearances)
        - **G**: 출장 경기 수 (Games)
        - **Age**: 선수 나이
        """)

    # 4. 개발 및 프로젝트 정보 (최대한 작고 깔끔하게 하단 배치)
    with st.container():
        st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        st.caption("김명서 (20242513)")
        st.caption("Linear Regression (R² 0.292)")
        st.caption("© 2024 KBO Batter WAR Analysis Project")
        
# ======== 프로젝트 기능 중심 사이드바 끝 ========

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
| 📊 **1_탐색적_데이터_분석** | 데이터 탐색 — 분포, 상관관계, 결측치 확인 |
| 📈 **2_다차원_통계_시각화** | 선수별 커리어 추이, OPS·HR vs WAR 산점도 |
| 🎯 **3_WAR_예측_모델_및_서비스** | WAR 예측 · 모델 비교 · 유사 선수 추천 |

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
