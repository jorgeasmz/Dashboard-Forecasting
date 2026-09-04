import pandas as pd
import pytest

from src import warehouse
from src.config import WarehouseColumn
from src.ingest import fetch_series


def test_hourly_reads_the_mart_at_its_own_grain():
    statement = warehouse.query(WarehouseColumn("spot_price"))

    assert "spot_price::double precision as y" in statement
    assert "group by" not in statement


def test_daily_averages_in_the_warehouse():
    """Aggregating there transfers 3,896 rows rather than 93,504."""
    statement = warehouse.query(WarehouseColumn("spot_price", grain="day"))

    assert "avg(spot_price)" in statement
    assert "group by market_date" in statement


def test_daily_drops_incomplete_days():
    """A day missing an hour would average over fewer values than the rest."""
    statement = warehouse.query(WarehouseColumn("spot_price", grain="day"))

    assert "having count(spot_price) = 24" in statement


def test_a_column_outside_the_contract_is_refused():
    """The name reaches the statement as text, so it is checked rather than trusted."""
    with pytest.raises(ValueError, match="not a measurement column"):
        warehouse.query(WarehouseColumn("spot_price; drop table series"))


def test_an_unsupported_grain_is_refused():
    with pytest.raises(ValueError, match="Unsupported grain"):
        warehouse.query(WarehouseColumn("spot_price", grain="week"))


def test_reading_without_a_connection_says_which_series_need_one():
    with pytest.raises(RuntimeError, match="WAREHOUSE_URL"):
        warehouse.read(WarehouseColumn("spot_price"), url="")


def test_fetch_series_routes_a_warehouse_series_away_from_http(monkeypatch):
    """A warehouse-backed series must not fall through to the CSV reader."""
    from src.config import SeriesSource

    rows = pd.DataFrame({
        "ds": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "y": [180.0, 191.5],
    })
    monkeypatch.setattr(warehouse, "read", lambda origin, url: rows.copy())
    monkeypatch.setattr("src.ingest.warehouse_url", lambda: "postgresql://ignored")

    source = SeriesSource(
        slug="spot-price-hourly",
        name="Colombian spot price, hourly",
        origin=WarehouseColumn("spot_price"),
        frequency="h",
        seasonal_period=24,
        horizon=24,
        unit="COP/kWh",
    )

    frame = fetch_series(source)

    assert list(frame.columns) == ["ds", "y"]
    assert frame["y"].tolist() == [180.0, 191.5]


def test_a_warehouse_series_is_skipped_when_none_is_configured(session, monkeypatch):
    """The published series still ingest with no infrastructure beyond this project."""
    from src import ingest

    monkeypatch.setattr("src.ingest.warehouse_url", lambda: "")
    monkeypatch.setattr(ingest, "fetch_series", lambda source, location=None: pd.DataFrame(
        {"ds": pd.to_datetime(["2020-01-01"]), "y": [1.0]}
    ))

    counts = ingest.ingest_all(session)

    assert counts["spot-price-hourly"] is None
    assert counts["spot-price-daily"] is None
    assert counts["airline-passengers"] == 1
