"""
Ingests every registered series and backtests every model family against it.

Usage: python -m src.pipeline
"""

from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.backtest import backtest
from src.config import BACKTEST_FOLDS, SERIES
from src.database import SessionLocal
from src.ingest import ingest_all, load_observations
from src.schema import Evaluation, Series
from src.selection import aggregate


def store_evaluations(session: Session, series: Series, results: list[dict]) -> None:
    """Replaces the stored results for one series."""
    session.execute(delete(Evaluation).where(Evaluation.series_id == series.id))
    session.add_all(Evaluation(series_id=series.id, **row) for row in results)
    session.commit()


def run(session: Session, folds: int = BACKTEST_FOLDS) -> None:
    print("Ingesting series...")
    counts = ingest_all(session)
    for slug, rows in counts.items():
        print(f"  {slug:<22} {rows:>5} observations")

    for source in SERIES:
        print(f"\nBacktesting {source.slug} (horizon {source.horizon}, {folds} folds)...")
        history = load_observations(session, source.slug)
        results = backtest(history, source, folds)

        series = session.query(Series).filter_by(slug=source.slug).one()
        store_evaluations(session, series, results)

        summary = aggregate(session, source.slug)
        for row in summary.itertuples():
            print(
                f"  {row.model:<16} MASE={row.mase:>6.3f}  MAE={row.mae:>9.2f}"
                f"  coverage={row.coverage:>5.0%}"
            )


def main() -> None:
    with SessionLocal() as session:
        run(session)


if __name__ == "__main__":
    main()
