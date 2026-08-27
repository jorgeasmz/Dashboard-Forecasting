import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.main import app
from src.config import SeriesSource
from src.database import Base, get_session
from src.schema import Evaluation, Observation, Series


@pytest.fixture
def source() -> SeriesSource:
    """A monthly series definition pointing at no remote file."""
    return SeriesSource(
        slug="test-series",
        name="Test series",
        filename="test.csv",
        date_column="Month",
        value_column="Value",
        frequency="MS",
        seasonal_period=12,
        horizon=6,
        unit="units",
    )


@pytest.fixture
def history() -> pd.DataFrame:
    """Sixty monthly points with a trend and a yearly cycle."""
    index = pd.date_range("2015-01-01", periods=60, freq="MS")
    trend = np.linspace(100, 160, 60)
    season = 12 * np.sin(2 * np.pi * np.arange(60) / 12)
    return pd.DataFrame({"ds": index, "y": trend + season})


@pytest.fixture
def csv_file(tmp_path, history, source):
    """The same series written out in its published layout."""
    path = tmp_path / "test.csv"
    frame = history.rename(columns={"ds": source.date_column, "y": source.value_column})
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def populated(session, source, history):
    """A stored series with observations and one fold of results per model."""
    series = Series(
        slug=source.slug, name=source.name, frequency=source.frequency,
        seasonal_period=source.seasonal_period, horizon=source.horizon, unit=source.unit,
    )
    session.add(series)
    session.commit()

    session.add_all(
        Observation(series_id=series.id, ts=row.ds.to_pydatetime(), value=float(row.y))
        for row in history.itertuples()
    )
    session.add_all(
        Evaluation(
            series_id=series.id, model=model, fold=fold,
            cutoff=history["ds"].iloc[-1].to_pydatetime(), horizon=source.horizon,
            mae=mae, rmse=mae * 1.2, mase=mase, coverage=coverage,
        )
        for model, mase, mae, coverage in [
            ("prophet", 0.8, 8.0, 0.75),
            ("sarima", 1.1, 11.0, 0.80),
            ("seasonal_naive", 1.0, 10.0, 0.95),
        ]
        for fold in range(2)
    )
    session.commit()
    return series


@pytest.fixture
def client(engine, populated):
    def override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override
    yield TestClient(app)
    app.dependency_overrides.clear()
