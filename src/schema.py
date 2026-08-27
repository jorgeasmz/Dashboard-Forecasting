from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class Series(Base):
    """One time series and the parameters used to model it."""

    __tablename__ = "series"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    frequency: Mapped[str] = mapped_column(String(8), nullable=False)
    seasonal_period: Mapped[int] = mapped_column(Integer, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)

    observations: Mapped[list["Observation"]] = relationship(
        back_populates="series", cascade="all, delete-orphan"
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        back_populates="series", cascade="all, delete-orphan"
    )


class Observation(Base):
    """A single measurement."""

    __tablename__ = "observations"
    __table_args__ = (UniqueConstraint("series_id", "ts", name="uq_observation"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("series.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    series: Mapped[Series] = relationship(back_populates="observations")


class Evaluation(Base):
    """One model's result on one rolling-origin fold."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("series.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    fold: Mapped[int] = mapped_column(Integer, nullable=False)
    cutoff: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)

    mae: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    # Scaled against the seasonal naive, so results compare across series.
    mase: Mapped[float] = mapped_column(Float, nullable=False)
    # Share of actual points that fell inside the prediction interval.
    coverage: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    series: Mapped[Series] = relationship(back_populates="evaluations")
