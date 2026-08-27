from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SeriesSummary(BaseModel):
    """One registered series and the model selected for it."""

    model_config = ConfigDict(protected_namespaces=())

    slug: str
    name: str
    frequency: str
    seasonal_period: int
    horizon: int
    unit: str
    observations: int
    best_model: str | None = Field(None, description="Lowest mean MASE across folds")


class ObservationOut(BaseModel):
    ts: datetime
    value: float


class ModelEvaluation(BaseModel):
    """Mean metrics for one model across the rolling-origin folds."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    folds: int
    mae: float
    rmse: float
    mase: float = Field(..., description="Scaled against the seasonal naive; 1.0 matches it")
    coverage: float = Field(..., description="Share of actuals inside the 80% interval")


class ForecastPoint(BaseModel):
    ds: datetime
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    slug: str
    model: str
    horizon: int
    points: list[ForecastPoint]
