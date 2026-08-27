import pytest

from src.config import INTERVAL_WIDTH
from src.forecasters import (
    LightGBMForecaster,
    ProphetForecaster,
    SarimaForecaster,
    SeasonalNaive,
    build_all,
    future_index,
)

FAMILIES = [SeasonalNaive, ProphetForecaster, SarimaForecaster, LightGBMForecaster]


def test_future_index_continues_the_history(history):
    index = future_index(history, periods=4, frequency="MS")

    assert len(index) == 4
    assert index[0] > history["ds"].iloc[-1]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_returns_the_same_columns(family, history):
    forecast = family(12).fit(history).predict(6, "MS")

    assert list(forecast.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(forecast) == 6


@pytest.mark.parametrize("family", FAMILIES)
def test_intervals_bracket_the_point_forecast(family, history):
    forecast = family(12).fit(history).predict(6, "MS")

    assert (forecast["yhat_lower"] <= forecast["yhat"]).all()
    assert (forecast["yhat"] <= forecast["yhat_upper"]).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_forecast_starts_after_the_history(family, history):
    forecast = family(12).fit(history).predict(6, "MS")

    assert forecast["ds"].iloc[0] > history["ds"].iloc[-1]


def test_seasonal_naive_repeats_the_last_cycle(history):
    forecast = SeasonalNaive(12).fit(history).predict(12, "MS")

    expected = history["y"].tail(12).tolist()
    assert forecast["yhat"].tolist() == pytest.approx(expected)


def test_sarima_drops_the_seasonal_term_for_long_cycles():
    """A 365-day cycle is out of reach for the seasonal specification."""
    assert SarimaForecaster(365).seasonal_period == 0
    assert SarimaForecaster(12).seasonal_period == 12


def test_lightgbm_lags_follow_the_seasonal_period():
    assert LightGBMForecaster(12).lags == [1, 2, 3, 12, 24]
    assert LightGBMForecaster(1).lags == [1, 2, 3]


def test_lightgbm_fits_one_model_per_interval_bound(history):
    model = LightGBMForecaster(12).fit(history)

    assert set(model.models) == {(1 - INTERVAL_WIDTH) / 2, 0.5, 1 - (1 - INTERVAL_WIDTH) / 2}


def test_build_all_returns_every_family():
    names = {model.name for model in build_all(12)}

    assert names == {"seasonal_naive", "prophet", "sarima", "lightgbm"}
