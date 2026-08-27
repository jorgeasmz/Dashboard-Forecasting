from src import pipeline
from src.ingest import replace_observations, upsert_series
from src.schema import Evaluation
from src.selection import aggregate, best_model


def test_pipeline_ingests_and_scores_every_family(session, source, history, monkeypatch):
    """Runs the whole batch offline, with the download replaced by fixtures."""
    def fake_ingest(active_session):
        series = upsert_series(active_session, source)
        replace_observations(active_session, series, history)
        return {source.slug: len(history)}

    monkeypatch.setattr(pipeline, "ingest_all", fake_ingest)
    monkeypatch.setattr(pipeline, "SERIES", [source])

    pipeline.run(session, folds=2)

    summary = aggregate(session, source.slug)
    assert len(summary) == 4
    assert (summary["folds"] == 2).all()
    assert best_model(session, source.slug) in set(summary["model"])


def test_rerunning_replaces_results_instead_of_appending(
    session, source, history, monkeypatch
):
    def fake_ingest(active_session):
        series = upsert_series(active_session, source)
        replace_observations(active_session, series, history)
        return {source.slug: len(history)}

    monkeypatch.setattr(pipeline, "ingest_all", fake_ingest)
    monkeypatch.setattr(pipeline, "SERIES", [source])

    pipeline.run(session, folds=2)
    first = session.query(Evaluation).count()
    pipeline.run(session, folds=2)

    assert session.query(Evaluation).count() == first
