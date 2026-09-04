from dataclasses import replace

from src.ingest import replace_observations, upsert_series
from src.serving import forecast


def test_falls_back_to_the_baseline_when_nothing_has_been_scored(
    session, source, history
):
    series = upsert_series(session, source)
    replace_observations(session, series, history)

    name, prediction = forecast(session, source.slug, horizon=3)

    assert name == "seasonal_naive"
    assert len(prediction) == 3


def test_the_served_fit_uses_the_window_the_backtest_scored(
    monkeypatch, recorder, session, source, history
):
    """Fitting on more than was measured would serve a model nothing scored."""
    series = upsert_series(session, replace(source, max_train=20))
    replace_observations(session, series, history)
    monkeypatch.setattr("src.serving.build_all", lambda period: [recorder(period)])

    forecast(session, source.slug, horizon=3)

    assert recorder.sizes == [20]


def test_an_unregistered_series_is_reported(session):
    import pytest

    with pytest.raises(LookupError):
        forecast(session, "no-such-series")
