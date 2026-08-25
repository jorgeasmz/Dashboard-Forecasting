import pandas as pd
import pytest

from src.config import DATE_COLUMN_RAW, SALES_COLUMN_RAW
from src.loader import fetch_data, load_data


@pytest.fixture
def csv_path(tmp_path):
    """A local CSV with dates as plain strings, like the real dataset."""
    path = tmp_path / "sales.csv"
    path.write_text(
        f"{DATE_COLUMN_RAW},{SALES_COLUMN_RAW}\n"
        "1960-01,100\n"
        "1960-02,120\n"
        "1960-03,140\n"
    )
    return path


def test_reads_csv_and_parses_the_date_column(csv_path):
    df = fetch_data(str(csv_path))

    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN_RAW])
    assert df[SALES_COLUMN_RAW].tolist() == [100, 120, 140]


def test_fetch_data_raises_when_the_source_is_missing(tmp_path):
    """The pure loader surfaces failures instead of hiding them."""
    with pytest.raises(FileNotFoundError):
        fetch_data(str(tmp_path / "does-not-exist.csv"))


def test_load_data_degrades_to_an_empty_frame(tmp_path):
    """app.py branches on df.empty, so the wrapper must never raise."""
    df = load_data(str(tmp_path / "does-not-exist.csv"))

    assert df.empty
