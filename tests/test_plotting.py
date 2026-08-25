import pandas as pd
import pytest
from plotly.graph_objs import Figure

from src.config import DATE_COLUMN_RAW, SALES_COLUMN_RAW
from src.forecasting import Forecaster
from src.plotting import plot_components, plot_forecast, plot_raw_data


def test_plot_raw_data_builds_a_single_series_figure(raw_df):
    fig = plot_raw_data(raw_df, DATE_COLUMN_RAW, SALES_COLUMN_RAW)

    assert isinstance(fig, Figure)
    assert len(fig.data) == 1


def test_plot_components_plots_the_trend():
    forecast = pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=6, freq="MS"),
        "trend": [1, 2, 3, 4, 5, 6],
    })

    fig = plot_components(forecast)

    assert isinstance(fig, Figure)
    assert fig.data[0].name == "Trend"


@pytest.mark.slow
def test_plot_forecast_draws_history_prediction_and_interval(monthly_series):
    """Regression guard: prophet.plot.plot_plotly raises on a fitted model."""
    forecaster = Forecaster()
    forecaster.train(monthly_series)
    forecast = forecaster.predict(periods=3)

    fig = plot_forecast(forecaster.model, forecast)

    assert isinstance(fig, Figure)
    names = [trace.name for trace in fig.data]
    assert names == ["Upper bound", "Uncertainty interval", "Forecast", "Observed"]
