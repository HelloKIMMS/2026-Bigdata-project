import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_data
from src.features import add_features, clean

st.set_page_config(page_title="시각화", page_icon="📈", layout="wide")
st.title("📈 KBO 타자 성적 시각화")

df_raw = load_data()
df = add_features(clean(df_raw))

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("필터")
    min_pa = st.slider("최소 PA", 0, 600, 300, 50)
    year_range = st.slider(
        "시즌 범위",
        int(df["Year"].min()), int(df["Year"].max()),
        (1990, int(df["Year"].max())),
    )

df_f = df[(df["PA"] >= min_pa) & df["Year"].between(*year_range)]

# ════════════════════════════════════════════════════
# 탭 구성
# ════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 선수 커리어", "📉 OPS · HR vs WAR", "📅 연도·팀 트렌드", "🏆 시즌 TOP 순위"]
)

# ── 탭 1: 선수 커리어 ──────────────────────────────────────────
with tab1:
    st.subheader("선수별 커리어 기록 조회")

    all_names = sorted(df["Name"].unique())
    selected_name = st.selectbox("선수 선택", all_names)

    player_df = df[df["Name"] == selected_name].sort_values("Year")

    if player_df.empty:
        st.warning("해당 선수의 데이터가 없습니다.")
    else:
        # 기본 정보
        info = player_df.iloc[-1]
        ci1, ci2, ci3, ci4 = st.columns(4)
        ci1.metric("최근 소속팀", info["Team"])
        ci2.metric("포지션", info["Pos."])
        ci3.metric("최근 나이", int(info["Age"]))
        ci4.metric("통산 평균 WAR", f"{player_df['WAR'].mean():.2f}")

        # 커리어 WAR 추이
        fig_war = go.Figure()
        fig_war.add_trace(go.Bar(
            x=player_df["Year"], y=player_df["WAR"],
            name="WAR", marker_color="steelblue",
        ))
        fig_war.add_trace(go.Scatter(
            x=player_df["Year"], y=player_df["WAR"],
            mode="lines+markers", name="추이", line=dict(color="orange", width=2),
        ))
        fig_war.update_layout(title=f"{selected_name} 시즌별 WAR", xaxis_title="시즌", yaxis_title="WAR")
        st.plotly_chart(fig_war, use_container_width=True)

        # 주요 타격 지표 추이
        col_l, col_r = st.columns(2)
        with col_l:
            fig_avg = px.line(
                player_df, x="Year", y=["AVG", "OBP", "SLG"],
                title="타율·출루율·장타율 추이",
                markers=True,
                labels={"value": "비율", "variable": "지표"},
            )
            st.plotly_chart(fig_avg, use_container_width=True)
        with col_r:
            fig_pow = px.bar(
                player_df, x="Year", y=["HR", "RBI", "SB"],
                title="홈런·타점·도루 추이",
                barmode="group",
                labels={"value": "개수", "variable": "지표"},
            )
            st.plotly_chart(fig_pow, use_container_width=True)

        # 상세 테이블
        with st.expander("전체 시즌 기록 보기"):
            show = ["Year", "Team", "Age", "G", "PA", "AVG", "OBP", "SLG", "OPS",
                    "HR", "RBI", "SB", "BB", "SO", "wRC+", "oWAR", "dWAR", "WAR"]
            st.dataframe(player_df[show].reset_index(drop=True), use_container_width=True)

# ── 탭 2: OPS·HR vs WAR ────────────────────────────────────────
with tab2:
    st.subheader("OPS / HR vs WAR 관계 분석")

    col2a, col2b = st.columns(2)
    with col2a:
        fig_ops = px.scatter(
            df_f, x="OPS", y="WAR",
            color="Pos.",
            hover_data=["Name", "Year", "Team", "PA"],
            title=f"OPS vs WAR (PA≥{min_pa})",
            trendline="ols",
            opacity=0.6,
            labels={"OPS": "OPS", "WAR": "WAR"},
        )
        st.plotly_chart(fig_ops, use_container_width=True)

        ops_corr = df_f[["OPS", "WAR"]].corr().iloc[0, 1]
        st.caption(f"OPS ↔ WAR 상관계수: **{ops_corr:.3f}**")

    with col2b:
        fig_hr = px.scatter(
            df_f, x="HR", y="WAR",
            color="Pos.",
            hover_data=["Name", "Year", "Team", "PA"],
            title=f"홈런(HR) vs WAR (PA≥{min_pa})",
            trendline="ols",
            opacity=0.6,
            labels={"HR": "홈런", "WAR": "WAR"},
        )
        st.plotly_chart(fig_hr, use_container_width=True)

        hr_corr = df_f[["HR", "WAR"]].corr().iloc[0, 1]
        st.caption(f"HR ↔ WAR 상관계수: **{hr_corr:.3f}**")

    st.divider()

    # wRC+ vs WAR
    fig_wrc = px.scatter(
        df_f, x="wRC+", y="WAR",
        color="Year",
        hover_data=["Name", "Team", "PA"],
        title="wRC+ vs WAR",
        trendline="ols",
        opacity=0.5,
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_wrc, use_container_width=True)

# ── 탭 3: 연도·팀 트렌드 ──────────────────────────────────────
with tab3:
    st.subheader("연도별 리그 평균 트렌드")

    yearly = (
        df_f.groupby("Year")[["WAR", "AVG", "OPS", "HR", "wRC+"]]
        .mean()
        .reset_index()
    )

    metric = st.selectbox("지표 선택", ["WAR", "AVG", "OPS", "HR", "wRC+"])
    fig_trend = px.line(
        yearly, x="Year", y=metric,
        title=f"연도별 리그 평균 {metric} (PA≥{min_pa})",
        markers=True,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    st.subheader("팀별 평균 WAR (선택 시즌)")
    sel_year = st.selectbox(
        "시즌 선택",
        sorted(df_f["Year"].unique(), reverse=True),
    )
    team_df = (
        df_f[df_f["Year"] == sel_year]
        .groupby("Team")[["WAR", "OPS", "HR", "wRC+"]]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("WAR", ascending=False)
    )
    fig_team = px.bar(
        team_df, x="Team", y="WAR",
        title=f"{sel_year}시즌 팀별 평균 WAR (PA≥{min_pa})",
        color="WAR",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_team, use_container_width=True)
    st.dataframe(team_df, use_container_width=True)

# ── 탭 4: 시즌 TOP ────────────────────────────────────────────
with tab4:
    st.subheader("시즌별 WAR TOP 선수")

    top_year = st.selectbox(
        "시즌 선택",
        sorted(df["Year"].unique(), reverse=True),
        key="top_year",
    )
    top_n = st.slider("상위 N명", 5, 30, 10)
    top_df = (
        df[df["Year"] == top_year]
        .sort_values("WAR", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_df.index += 1

    show_top = ["Name", "Team", "Pos.", "Age", "G", "PA",
                "AVG", "OPS", "HR", "RBI", "SB", "wRC+", "WAR"]
    st.dataframe(top_df[show_top], use_container_width=True)

    fig_top = px.bar(
        top_df, x="WAR", y="Name",
        orientation="h",
        color="WAR",
        color_continuous_scale="Blues",
        title=f"{top_year}시즌 WAR TOP {top_n}",
        hover_data=["Team", "OPS", "HR", "wRC+"],
        text="WAR",
    )
    fig_top.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_top.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig_top, use_container_width=True)
