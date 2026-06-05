import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from src.data_loader import load_data
from src.features import FEATURE_COLS, add_features, clean

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 탐색적 데이터 분석 (EDA)")

# ── 데이터 로드 ──────────────────────────────────────────────
df_raw = load_data()
df = clean(df_raw)
df = add_features(df)

# ── 사이드바 필터 ─────────────────────────────────────────────
with st.sidebar:
    st.header("필터")
    min_pa = st.slider("최소 PA (타석)", 0, 600, 300, 50)
    year_range = st.slider(
        "시즌 범위",
        int(df["Year"].min()), int(df["Year"].max()),
        (2000, int(df["Year"].max())),
    )

df_f = df[(df["PA"] >= min_pa) & df["Year"].between(*year_range)]

# ── 기본 정보 ─────────────────────────────────────────────────
st.subheader("① 데이터 기본 정보")
c1, c2, c3, c4 = st.columns(4)
c1.metric("필터 후 레코드", f"{len(df_f):,}")
c2.metric("선수 수", f"{df_f['Id'].nunique():,}")
c3.metric("결측치 (WAR)", int(df_f["WAR"].isnull().sum()))
c4.metric("WAR 평균", f"{df_f['WAR'].mean():.2f}")

with st.expander("컬럼별 결측치 현황"):
    null_df = df_f.isnull().sum().reset_index()
    null_df.columns = ["컬럼", "결측치 수"]
    null_df = null_df[null_df["결측치 수"] > 0]
    if null_df.empty:
        st.success("결측치 없음")
    else:
        st.dataframe(null_df, use_container_width=True)

with st.expander("기술통계 (주요 타격 지표)"):
    show_cols = ["WAR", "AVG", "OBP", "SLG", "OPS", "HR", "RBI", "SB", "BB", "SO", "wRC+", "Age", "PA"]
    st.dataframe(df_f[show_cols].describe().T.round(3), use_container_width=True)

st.divider()

# ── WAR 분포 ──────────────────────────────────────────────────
st.subheader("② WAR 분포")
col_a, col_b = st.columns(2)

with col_a:
    fig = px.histogram(
        df_f, x="WAR", nbins=50,
        title=f"WAR 히스토그램 (PA≥{min_pa})",
        labels={"WAR": "WAR"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.add_vline(x=df_f["WAR"].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"평균 {df_f['WAR'].mean():.2f}")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig2 = px.box(
        df_f, x="Year", y="WAR",
        title="연도별 WAR 분포",
        labels={"Year": "시즌", "WAR": "WAR"},
    )
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── 상관관계 히트맵 ───────────────────────────────────────────
st.subheader("③ 주요 지표 상관관계 히트맵")

corr_cols = ["WAR", "AVG", "OBP", "SLG", "OPS", "HR", "RBI", "SB",
             "BB", "SO", "wRC+", "ISO", "BB_rate", "SO_rate", "Age", "PA"]
corr_cols = [c for c in corr_cols if c in df_f.columns]
corr_mat = df_f[corr_cols].corr().round(2)

fig_hm = px.imshow(
    corr_mat,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="피어슨 상관계수 히트맵",
    aspect="auto",
    height=600,
)
st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# ── WAR와 주요 지표 관계 (산점도 매트릭스) ─────────────────────
st.subheader("④ WAR와 주요 지표 상관 막대그래프")

war_corr = df_f[corr_cols].corr()["WAR"].drop("WAR").sort_values()
fig_bar = px.bar(
    x=war_corr.values,
    y=war_corr.index,
    orientation="h",
    title="각 지표와 WAR의 상관계수",
    labels={"x": "상관계수", "y": "지표"},
    color=war_corr.values,
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 포지션별 WAR ──────────────────────────────────────────────
st.subheader("⑤ 포지션별 WAR 분포")
fig_pos = px.box(
    df_f[df_f["Pos."].notna()],
    x="Pos.", y="WAR",
    title="포지션별 WAR 분포",
    color="Pos.",
)
st.plotly_chart(fig_pos, use_container_width=True)

# ── 원시 데이터 탐색 ──────────────────────────────────────────
st.subheader("⑥ 원시 데이터 탐색")
show_cols_raw = ["Year", "Name", "Team", "Pos.", "Age", "G", "PA",
                 "AVG", "OBP", "SLG", "OPS", "HR", "RBI", "SB", "wRC+", "WAR"]
st.dataframe(
    df_f[show_cols_raw].sort_values("WAR", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=400,
)
