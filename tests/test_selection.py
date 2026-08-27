from src.selection import aggregate, best_model


def test_aggregate_averages_across_folds(session, populated):
    summary = aggregate(session, populated.slug)

    assert set(summary["model"]) == {"prophet", "sarima", "seasonal_naive"}
    assert (summary["folds"] == 2).all()


def test_aggregate_orders_by_scaled_error(session, populated):
    summary = aggregate(session, populated.slug)

    assert summary["mase"].is_monotonic_increasing


def test_best_model_is_the_lowest_mase(session, populated):
    assert best_model(session, populated.slug) == "prophet"


def test_an_unscored_series_has_no_selection(session):
    assert best_model(session, "nothing-here") is None


def test_aggregating_an_unscored_series_returns_an_empty_frame(session):
    assert aggregate(session, "nothing-here").empty
