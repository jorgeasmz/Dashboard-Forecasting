import pandas as pd
import streamlit as st

from src.config import DATASET_URL, DATE_COLUMN_RAW


def fetch_data(source: str = DATASET_URL) -> pd.DataFrame:
    """
    Reads the time-series CSV and normalises the date column.

    Accepts anything pandas can read: a remote URL or a local path.
    Raises on failure so the caller decides how to report it.
    """
    df = pd.read_csv(source)

    # Ensure the date column is actually datetime
    df[DATE_COLUMN_RAW] = pd.to_datetime(df[DATE_COLUMN_RAW])

    return df


@st.cache_data
def load_data(source: str = DATASET_URL) -> pd.DataFrame:
    """
    Streamlit-facing wrapper around fetch_data.

    Reports the failure in the UI and degrades to an empty DataFrame, so the
    app shows a warning instead of a traceback.
    Using @st.cache_data ensures we don't re-download the data on every interaction.
    """
    try:
        return fetch_data(source)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
