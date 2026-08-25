import pandas as pd
import pytest

from src.config import DATE_COLUMN_RAW, SALES_COLUMN_RAW


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Three months of sales, chronological, shaped like loader output."""
    return pd.DataFrame({
        DATE_COLUMN_RAW: pd.to_datetime(["1960-01-01", "1960-02-01", "1960-03-01"]),
        SALES_COLUMN_RAW: [100, 120, 140],
    })


@pytest.fixture
def unsorted_raw_df() -> pd.DataFrame:
    """The same three months, deliberately out of order."""
    return pd.DataFrame({
        DATE_COLUMN_RAW: pd.to_datetime(["1960-03-01", "1960-01-01", "1960-02-01"]),
        SALES_COLUMN_RAW: [140, 100, 120],
    })


@pytest.fixture
def monthly_series() -> pd.DataFrame:
    """A longer synthetic series already in Prophet's schema, for fitting."""
    periods = 30
    return pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=periods, freq="MS"),
        "y": [100 + 5 * i for i in range(periods)],
    })
