"""
Model selection from stored backtest results.
"""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.schema import Evaluation, Series


def aggregate(session: Session, slug: str) -> pd.DataFrame:
    """Mean metrics per model across folds, best MASE first."""
    statement = (
        select(
            Evaluation.model,
            Evaluation.fold,
            Evaluation.mae,
            Evaluation.rmse,
            Evaluation.mase,
            Evaluation.coverage,
        )
        .join(Series)
        .where(Series.slug == slug)
    )
    rows = session.execute(statement).all()
    if not rows:
        return pd.DataFrame(columns=["model", "folds", "mae", "rmse", "mase", "coverage"])

    frame = pd.DataFrame(rows, columns=["model", "fold", "mae", "rmse", "mase", "coverage"])
    summary = (
        frame.groupby("model")
        .agg(folds=("fold", "count"), mae=("mae", "mean"), rmse=("rmse", "mean"),
             mase=("mase", "mean"), coverage=("coverage", "mean"))
        .reset_index()
        .sort_values("mase")
    )
    return summary.reset_index(drop=True)


def best_model(session: Session, slug: str) -> str | None:
    """The model with the lowest mean MASE, or None if the series is unscored."""
    summary = aggregate(session, slug)
    return None if summary.empty else str(summary.iloc[0]["model"])
