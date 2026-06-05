import numpy as np
import pandas as pd

# 모델 학습에 사용할 기본 피처 (현재 시즌 WAR 포함 — 가장 강력한 예측 변수)
FEATURE_COLS = [
    "WAR", "oWAR", "dWAR",
    "AVG", "OBP", "SLG", "OPS", "HR", "RBI", "SB",
    "BB", "SO", "wRC+", "Age", "G", "PA", "R", "H",
    "2B", "3B", "TB",
]

# KNN 유사 선수 추천용 피처 (비율 중심)
KNN_FEATURE_COLS = ["AVG", "OBP", "SLG", "OPS", "HR", "RBI", "SB", "BB", "SO", "wRC+"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["WAR"]).copy()
    df = df[df["PA"] > 0]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 타율 기반 파생 지표
    df["BB_rate"] = df["BB"] / df["PA"].replace(0, np.nan)
    df["SO_rate"] = df["SO"] / df["PA"].replace(0, np.nan)
    df["ISO"] = df["SLG"] - df["AVG"]  # Isolated Power
    denominator = (df["AB"] - df["SO"] - df["HR"] + df["SF"]).replace(0, np.nan)
    df["BABIP"] = (df["H"] - df["HR"]) / denominator
    return df


def build_next_war_dataset(df: pd.DataFrame, min_pa: int = 300) -> pd.DataFrame:
    """이전 시즌 성적 → 다음 시즌 WAR 예측 데이터셋.

    규정타석 이상 선수만 사용하고, 연속된 시즌 쌍만 유효 샘플로 취급한다.
    """
    qualified = df[df["PA"] >= min_pa].sort_values(["Id", "Year"]).copy()
    qualified["next_WAR"] = qualified.groupby("Id")["WAR"].shift(-1)
    qualified["next_Year"] = qualified.groupby("Id")["Year"].shift(-1)
    dataset = qualified.dropna(subset=["next_WAR"])
    # 연속 시즌만 유효 (예: 2020→2021, 공백 없는 경우)
    dataset = dataset[dataset["next_Year"] == dataset["Year"] + 1].copy()
    return dataset
