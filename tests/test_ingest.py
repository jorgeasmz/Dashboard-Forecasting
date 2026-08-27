import pandas as pd

from src.ingest import fetch_series, load_observations, replace_observations, upsert_series


def test_reads_and_renames_to_the_prophet_schema(csv_file, source):
    frame = fetch_series(source, location=str(csv_file))

    assert list(frame.columns) == ["ds", "y"]
    assert pd.api.types.is_datetime64_any_dtype(frame["ds"])


def test_orders_by_time(csv_file, source):
    frame = fetch_series(source, location=str(csv_file))

    assert frame["ds"].is_monotonic_increasing


def test_coerces_sentinel_values_and_drops_them(tmp_path, source):
    """Some published series carry stray characters on a few values."""
    path = tmp_path / "dirty.csv"
    path.write_text("Month,Value\n2020-01,10\n2020-02,?3\n2020-03,12\n")

    frame = fetch_series(source, location=str(path))

    assert len(frame) == 2
    assert frame["y"].tolist() == [10.0, 12.0]


def test_truncates_to_the_configured_window(csv_file, source):
    from dataclasses import replace

    frame = fetch_series(replace(source, max_observations=10), location=str(csv_file))

    assert len(frame) == 10


def test_upsert_is_idempotent(session, source):
    first = upsert_series(session, source)
    second = upsert_series(session, source)

    assert first.id == second.id


def test_observations_round_trip(session, source, history):
    series = upsert_series(session, source)

    replace_observations(session, series, history)
    stored = load_observations(session, source.slug)

    assert len(stored) == len(history)
    assert stored["ds"].is_monotonic_increasing


def test_replacing_observations_does_not_accumulate(session, source, history):
    series = upsert_series(session, source)

    replace_observations(session, series, history)
    replace_observations(session, series, history)

    assert len(load_observations(session, source.slug)) == len(history)
