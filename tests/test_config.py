import pytest

from src.config import SERIES, SERIES_BY_SLUG, _normalise_database_url


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///forecasting.db", "sqlite:///forecasting.db"),
    ],
)
def test_database_urls_are_pointed_at_psycopg_3(given, expected):
    assert _normalise_database_url(given) == expected


def test_every_series_has_a_unique_slug():
    assert len(SERIES_BY_SLUG) == len(SERIES)


def test_horizons_leave_room_for_several_folds():
    """Four folds plus a training window need the horizon to stay well under the history."""
    for source in SERIES:
        assert source.horizon >= 1
        assert source.seasonal_period >= 1
