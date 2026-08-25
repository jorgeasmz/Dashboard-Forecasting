import pytest

from src.forecasting import Forecaster


def test_predicting_before_training_raises():
    with pytest.raises(ValueError, match="not been trained"):
        Forecaster().predict(periods=3)


def test_get_forecast_is_none_before_predicting():
    assert Forecaster().get_forecast() is None


@pytest.mark.slow
def test_forecast_has_the_expected_shape_and_columns(monthly_series):
    """Contract test: shape and schema, never numerical accuracy."""
    forecaster = Forecaster()
    forecaster.train(monthly_series)

    horizon = 6
    forecast = forecaster.predict(periods=horizon)

    assert len(forecast) == len(monthly_series) + horizon
    for column in ("ds", "yhat", "yhat_lower", "yhat_upper"):
        assert column in forecast.columns
    assert forecaster.get_forecast() is forecast
