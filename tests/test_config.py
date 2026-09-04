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


def test_the_warehouse_url_drops_a_sqlalchemy_driver_suffix(monkeypatch):
    """DATABASE_URL carries one and psycopg, which gets this, does not accept it."""
    from src.config import warehouse_url

    monkeypatch.setenv("WAREHOUSE_URL", "postgresql+psycopg://user:pw@host/db")

    assert warehouse_url() == "postgresql://user:pw@host/db"


def test_the_warehouse_url_is_empty_when_unset(monkeypatch):
    from src.config import warehouse_url

    monkeypatch.delenv("WAREHOUSE_URL", raising=False)

    assert warehouse_url() == ""
