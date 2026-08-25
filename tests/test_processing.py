import pandas as pd
import pytest

from src.config import DATE_COLUMN_RAW, SALES_COLUMN_RAW
from src.processing import prepare_for_prophet


def test_renames_columns_to_prophet_schema(raw_df):
    result = prepare_for_prophet(raw_df)

    assert list(result.columns) == ["ds", "y"]


def test_sorts_by_date(unsorted_raw_df):
    result = prepare_for_prophet(unsorted_raw_df)

    assert result["ds"].is_monotonic_increasing
    assert result["y"].tolist() == [100, 120, 140]


def test_does_not_mutate_the_caller_dataframe(raw_df):
    before = raw_df.copy(deep=True)

    prepare_for_prophet(raw_df)

    pd.testing.assert_frame_equal(raw_df, before)


def test_raises_a_clear_error_when_columns_are_missing():
    df = pd.DataFrame({"date": ["1960-01"], "value": [100]})

    with pytest.raises(ValueError) as excinfo:
        prepare_for_prophet(df)

    message = str(excinfo.value)
    assert DATE_COLUMN_RAW in message
    assert SALES_COLUMN_RAW in message
