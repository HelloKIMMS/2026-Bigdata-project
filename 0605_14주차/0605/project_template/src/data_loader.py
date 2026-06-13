import os
import pandas as pd
import streamlit as st

_DATA_FILE = "kbo_batting_stats_by_season_1982-2025.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", _DATA_FILE)
    df = pd.read_csv(path, encoding="utf-8")
    return df


@st.cache_data
def load_qualified(min_pa: int = 300) -> pd.DataFrame:
    df = load_data()
    return df[df["PA"] >= min_pa].copy()
