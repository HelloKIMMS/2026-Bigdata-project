import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data_loader import load_data
from src.features import (
    FEATURE_COLS,
    KNN_FEATURE_COLS,
    add_features,
    build_next_war_dataset,
    clean,
)

st.set_page_config(page_title="모델·서비스", page_icon="🤖", layout="wide")
st.title("🤖 WAR 예측 모델 및 서비스")

# ── 데이터 준비 ──────────────────────────────────────────────
df_raw = load_data()
df = add_features(clean(df_raw))

MIN_PA = 300  # 규정타석 기준

# 학습 데이터: 이전 시즌 → 다음 시즌 WAR
dataset = build_next_war_dataset(df, min_pa=MIN_PA)
valid_features = [c for c in FEATURE_COLS if c in dataset.columns]

# 2023시즌을 테스트셋, 2000–2022를 학습셋으로 분리
TEST_YEAR = 2023
train = dataset[dataset["Year"] < TEST_YEAR].dropna(subset=valid_features + ["next_WAR"])
test = dataset[dataset["Year"] == TEST_YEAR].dropna(subset=valid_features + ["next_WAR"])

X_train = train[valid_features].values
y_train = train["next_WAR"].values
X_test = test[valid_features].values
y_test = test["next_WAR"].values


@st.cache_resource
def train_models(X_tr, y_tr):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    lr = LinearRegression()
    lr.fit(X_tr_s, y_tr)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                       random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)

    return scaler, lr, rf, xgb


@st.cache_resource
def train_knn(df_qualified, feature_cols, min_pa=300):
    knn_df = df_qualified[df_qualified["PA"] >= min_pa].dropna(subset=feature_cols).copy()
    scaler = StandardScaler()
    X_knn = scaler.fit_transform(knn_df[feature_cols].values)
    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn.fit(X_knn)
    return knn, scaler, knn_df.reset_index(drop=True)


def eval_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R²": r2_score(y_true, y_pred),
    }


scaler_lr, model_lr, model_rf, model_xgb = train_models(X_train, y_train)
knn_model, knn_scaler, knn_base = train_knn(df, KNN_FEATURE_COLS, MIN_PA)

# ════════════════════════════════════════════════════
# 탭 구성
# ════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 모델 성능 비교", "🔮 WAR 예측", "🔍 유사 선수 추천"])

# ── 탭 1: 모델 성능 비교 ──────────────────────────────────────
with tab1:
    st.subheader("모델 성능 비교 (2024시즌 테스트셋)")
    st.caption(
        f"학습: 규정타석(PA≥{MIN_PA}) 연속 시즌 쌍 (–{TEST_YEAR-1}시즌) | "
        f"테스트: {TEST_YEAR}시즌 성적 → {TEST_YEAR+1}시즌 WAR 예측 | "
        f"n_test={len(test)}"
    )

    if len(X_test) == 0:
        st.warning("테스트 데이터가 부족합니다.")
    else:
        pred_lr = model_lr.predict(scaler_lr.transform(X_test))
        pred_rf = model_rf.predict(X_test)
        pred_xgb = model_xgb.predict(X_test)

        metrics = {
            "Linear Regression": eval_metrics(y_test, pred_lr),
            "Random Forest": eval_metrics(y_test, pred_rf),
            "XGBoost": eval_metrics(y_test, pred_xgb),
        }
        metrics_df = pd.DataFrame(metrics).T.round(4)
        st.dataframe(metrics_df, use_container_width=True)

        # 실제 vs 예측 산점도
        model_choice = st.radio("산점도 모델 선택", list(metrics.keys()), horizontal=True)
        pred_map = {
            "Linear Regression": pred_lr,
            "Random Forest": pred_rf,
            "XGBoost": pred_xgb,
        }
        preds = pred_map[model_choice]

        scatter_df = pd.DataFrame({
            "실제 WAR": y_test,
            "예측 WAR": preds,
            "Name": test["Name"].values,
            "Year": test["Year"].values,
            "Team": test["Team"].values,
        })
        fig_sc = px.scatter(
            scatter_df, x="실제 WAR", y="예측 WAR",
            hover_data=["Name", "Year", "Team"],
            title=f"{model_choice} — 실제 vs 예측 WAR",
            color_discrete_sequence=["steelblue"],
        )
        lim = max(abs(scatter_df[["실제 WAR", "예측 WAR"]].values.min()) + 0.5,
                  abs(scatter_df[["실제 WAR", "예측 WAR"]].values.max()) + 0.5)
        fig_sc.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                         line=dict(dash="dash", color="red"))
        st.plotly_chart(fig_sc, use_container_width=True)

        # Feature Importance (RF 기준)
        st.subheader("Feature Importance (Random Forest 기준)")
        imp_df = (
            pd.DataFrame({"피처": valid_features, "중요도": model_rf.feature_importances_})
            .sort_values("중요도", ascending=True)
        )
        fig_imp = px.bar(
            imp_df, x="중요도", y="피처", orientation="h",
            title="WAR 예측에 영향을 미치는 주요 지표",
            color="중요도", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── 탭 2: WAR 예측 ────────────────────────────────────────────
with tab2:
    st.subheader("🔮 성적 입력 → 다음 시즌 예상 WAR 예측")
    st.caption("현재 시즌 성적을 입력하면 다음 시즌 WAR를 예측합니다.")

    use_model = st.radio("사용 모델", ["Random Forest (권장)", "XGBoost", "Linear Regression"], horizontal=True)

    # 참고값: 2024시즌 평균
    ref = df[df["Year"] == df["Year"].max()].mean(numeric_only=True)

    with st.form("war_predict_form"):
        st.markdown("**현재 시즌 WAR (가장 중요한 예측 변수)**")
        cw1, cw2, cw3 = st.columns(3)
        war_cur = cw1.number_input("WAR (종합)", -3.0, 12.0, float(round(ref.get("WAR", 1.5), 2)), 0.1, format="%.2f")
        owar = cw2.number_input("oWAR (공격 WAR)", -3.0, 12.0, float(round(ref.get("oWAR", 1.2), 2)), 0.1, format="%.2f")
        dwar = cw3.number_input("dWAR (수비 WAR)", -3.0, 5.0, float(round(ref.get("dWAR", 0.3), 2)), 0.1, format="%.2f")

        st.markdown("**타격 지표 입력**")
        c1, c2, c3, c4 = st.columns(4)
        avg = c1.number_input("타율 (AVG)", 0.0, 0.500, float(round(ref.get("AVG", 0.270), 3)), 0.001, format="%.3f")
        obp = c2.number_input("출루율 (OBP)", 0.0, 0.600, float(round(ref.get("OBP", 0.340), 3)), 0.001, format="%.3f")
        slg = c3.number_input("장타율 (SLG)", 0.0, 0.900, float(round(ref.get("SLG", 0.400), 3)), 0.001, format="%.3f")
        ops = c4.number_input("OPS", 0.0, 1.500, float(round(ref.get("OPS", 0.740), 3)), 0.001, format="%.3f")

        c5, c6, c7, c8 = st.columns(4)
        hr = c5.number_input("홈런 (HR)", 0, 60, int(ref.get("HR", 10)))
        rbi = c6.number_input("타점 (RBI)", 0, 200, int(ref.get("RBI", 45)))
        sb = c7.number_input("도루 (SB)", 0, 80, int(ref.get("SB", 8)))
        bb = c8.number_input("볼넷 (BB)", 0, 150, int(ref.get("BB", 35)))

        c9, c10, c11, c12 = st.columns(4)
        so = c9.number_input("삼진 (SO)", 0, 200, int(ref.get("SO", 65)))
        wrc = c10.number_input("wRC+", 0, 250, int(ref.get("wRC+", 100)))
        age = c11.number_input("나이 (Age)", 17, 45, int(ref.get("Age", 27)))
        g = c12.number_input("경기 수 (G)", 1, 144, int(ref.get("G", 120)))

        c13, c14, c15, c16 = st.columns(4)
        pa = c13.number_input("타석 (PA)", 1, 700, int(ref.get("PA", 450)))
        r_val = c14.number_input("득점 (R)", 0, 150, int(ref.get("R", 55)))
        h_val = c15.number_input("안타 (H)", 0, 250, int(ref.get("H", 120)))
        b2 = c16.number_input("2루타 (2B)", 0, 60, int(ref.get("2B", 20)))

        c17, c18 = st.columns([1, 3])
        b3 = c17.number_input("3루타 (3B)", 0, 20, int(ref.get("3B", 2)))
        tb = c18.number_input("루타 (TB)", 0, 400, int(ref.get("TB", 180)))

        submitted = st.form_submit_button("WAR 예측하기", type="primary")

    if submitted:
        # FEATURE_COLS 순서: WAR, oWAR, dWAR, AVG, OBP, SLG, OPS, HR, RBI, SB,
        #                    BB, SO, wRC+, Age, G, PA, R, H, 2B, 3B, TB
        input_vals = np.array([[war_cur, owar, dwar,
                                avg, obp, slg, ops, hr, rbi, sb,
                                bb, so, wrc, age, g, pa, r_val, h_val, b2, b3, tb]])
        if use_model.startswith("Random Forest"):
            pred = model_rf.predict(input_vals)[0]
        elif use_model.startswith("XGBoost"):
            pred = model_xgb.predict(input_vals)[0]
        else:
            pred = model_lr.predict(scaler_lr.transform(input_vals))[0]

        st.success(f"### 예상 다음 시즌 WAR: **{pred:.2f}**")

        grade_map = [
            (6.0, "🌟 MVP급 (6.0 이상)"),
            (4.0, "⭐ 올스타급 (4.0–6.0)"),
            (2.0, "✅ 주전급 (2.0–4.0)"),
            (0.5, "📌 백업 (0.5–2.0)"),
            (-99, "⚠️ 대체선수 수준 (0.5 미만)"),
        ]
        for threshold, label in grade_map:
            if pred >= threshold:
                st.info(f"평가: {label}")
                break

        # 비슷한 WAR 선수들 참고
        similar_war = df[
            (df["PA"] >= MIN_PA) &
            (df["WAR"].between(pred - 0.5, pred + 0.5))
        ].sort_values("Year", ascending=False).head(5)
        if not similar_war.empty:
            with st.expander("비슷한 WAR 선수 사례 참고"):
                st.dataframe(
                    similar_war[["Year", "Name", "Team", "AVG", "OPS", "HR", "wRC+", "WAR"]].reset_index(drop=True),
                    use_container_width=True,
                )

# ── 탭 3: 유사 선수 추천 ──────────────────────────────────────
with tab3:
    st.subheader("🔍 유사 선수 추천 (KNN 기반)")
    st.caption(f"타격 지표 유사도 기반 추천 | 사용 피처: {', '.join(KNN_FEATURE_COLS)}")

    # 선수 선택 or 직접 입력
    search_mode = st.radio("검색 방법", ["선수 선택", "직접 입력"], horizontal=True)

    if search_mode == "선수 선택":
        years_avail = sorted(df["Year"].unique(), reverse=True)
        sel_year_knn = st.selectbox("시즌", years_avail, key="knn_year")
        cands = df[(df["Year"] == sel_year_knn) & (df["PA"] >= MIN_PA)].sort_values("WAR", ascending=False)
        sel_player_knn = st.selectbox("선수", cands["Name"].tolist(), key="knn_player")

        query_row = df[
            (df["Name"] == sel_player_knn) & (df["Year"] == sel_year_knn)
        ].iloc[0]
        query_vals = query_row[KNN_FEATURE_COLS].values.reshape(1, -1)

        st.markdown("**선택 선수 기록**")
        display_info = query_row[["Name", "Year", "Team", "Pos.", "Age", "PA"] + KNN_FEATURE_COLS + ["WAR"]]
        st.dataframe(
            pd.DataFrame(display_info).T.rename(columns={display_info.name: "값"}),
            use_container_width=True,
        )
    else:
        st.markdown("**타격 지표 직접 입력**")
        knn_cols2 = st.columns(len(KNN_FEATURE_COLS))
        knn_defaults = knn_base[KNN_FEATURE_COLS].mean()
        knn_inputs = {}
        for col_obj, feat in zip(knn_cols2, KNN_FEATURE_COLS):
            default_val = float(round(knn_defaults[feat], 3))
            step = 0.001 if feat in ["AVG", "OBP", "SLG", "OPS"] else 1.0
            fmt = "%.3f" if step < 1 else "%.0f"
            knn_inputs[feat] = col_obj.number_input(feat, value=default_val, step=step, format=fmt)
        query_vals = np.array([[knn_inputs[f] for f in KNN_FEATURE_COLS]])
        sel_player_knn = None

    if st.button("유사 선수 찾기", type="primary"):
        query_scaled = knn_scaler.transform(query_vals)
        distances, indices = knn_model.kneighbors(query_scaled)

        neighbors = knn_base.iloc[indices[0]].copy()
        neighbors["유사도 거리"] = distances[0].round(3)

        # 검색 선수 자신 제외
        if sel_player_knn:
            neighbors = neighbors[neighbors["Name"] != sel_player_knn]

        result_cols = ["Name", "Year", "Team", "Pos.", "Age", "PA"] + KNN_FEATURE_COLS + ["WAR", "유사도 거리"]
        result_cols = [c for c in result_cols if c in neighbors.columns]

        st.success(f"유사 선수 {len(neighbors)}명 추천")
        st.dataframe(neighbors[result_cols].reset_index(drop=True), use_container_width=True)

        # 레이더 차트 비교
        if sel_player_knn and len(neighbors) > 0:
            st.subheader("레이더 차트 비교")
            radar_features = ["AVG", "OBP", "SLG", "OPS", "wRC+"]
            radar_features = [f for f in radar_features if f in neighbors.columns]

            # 정규화 (0~1)
            all_vals = df[radar_features].dropna()
            norm_min = all_vals.min()
            norm_max = all_vals.max()

            def normalize(row):
                return ((row[radar_features] - norm_min) / (norm_max - norm_min + 1e-9)).tolist()

            fig_radar = go.Figure()
            query_df_radar = df[(df["Name"] == sel_player_knn)].sort_values("Year").iloc[-1]
            fig_radar.add_trace(go.Scatterpolar(
                r=normalize(query_df_radar) + [normalize(query_df_radar)[0]],
                theta=radar_features + [radar_features[0]],
                fill="toself",
                name=sel_player_knn,
            ))
            for _, nb_row in neighbors.head(3).iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalize(nb_row) + [normalize(nb_row)[0]],
                    theta=radar_features + [radar_features[0]],
                    fill="toself",
                    name=f"{nb_row['Name']} ({int(nb_row['Year'])})",
                    opacity=0.6,
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                    title="타격 지표 레이더 차트 비교")
            st.plotly_chart(fig_radar, use_container_width=True)
