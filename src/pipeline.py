"""
Ingests every registered series and backtests every model family against it.

Usage: python -m src.pipeline [--series SLUG ...]
"""

import argparse

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


def run(
    session: Session,
    folds: int = BACKTEST_FOLDS,
    slugs: list[str] | None = None,
) -> None:
    selected = [s for s in SERIES if slugs is None or s.slug in slugs]

    print("Ingesting series...")
    counts = ingest_all(session, slugs)
    for slug, rows in counts.items():
        if rows is None:
            print(f"  {slug:<22} skipped, WAREHOUSE_URL is unset")
        else:
            print(f"  {slug:<22} {rows:>6} observations")

    for source in selected:
        # The whole history: the folds are cut from the end of it and each one
        # applies its own training window.
        history = load_observations(session, source.slug)
        if history.empty:
            continue

        window = f", {source.max_train} training window" if source.max_train else ""
        print(
            f"\nBacktesting {source.slug} "
            f"(horizon {source.horizon}, {folds} folds{window})..."
        )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        metavar="SLUG",
        help="Restrict the run to one series. Repeatable. Defaults to all of them.",
    )
    arguments = parser.parse_args()

    known = {source.slug for source in SERIES}
    unknown = set(arguments.series or []) - known
    if unknown:
        parser.error(f"unknown series: {', '.join(sorted(unknown))}")

    with SessionLocal() as session:
        run(session, slugs=arguments.series)


if __name__ == "__main__":
    main()
