import pandas as pd
import streamlit as st
from src.config import DATASET_URL, DATE_COLUMN_RAW, SALES_COLUMN_RAW

@st.cache_data
def load_data(url: str = DATASET_URL) -> pd.DataFrame:
    """
    Loads the dataset from a CSV URL and performs initial type conversion.
    Using @st.cache_data ensures we don't re-download the data on every interaction.
    """
    try:
        df = pd.read_csv(url)
        
        # Ensure the date column is actually datetime
        df[DATE_COLUMN_RAW] = pd.to_datetime(df[DATE_COLUMN_RAW])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
