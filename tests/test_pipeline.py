from src import pipeline
from src.ingest import replace_observations, upsert_series
from src.schema import Evaluation
from src.selection import aggregate, best_model


def test_pipeline_ingests_and_scores_every_family(session, source, history, monkeypatch):
    """Runs the whole batch offline, with the download replaced by fixtures."""
    def fake_ingest(active_session, slugs=None):
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
    def fake_ingest(active_session, slugs=None):
        series = upsert_series(active_session, source)
        replace_observations(active_session, series, history)
        return {source.slug: len(history)}

    monkeypatch.setattr(pipeline, "ingest_all", fake_ingest)
    monkeypatch.setattr(pipeline, "SERIES", [source])

    pipeline.run(session, folds=2)
    first = session.query(Evaluation).count()
    pipeline.run(session, folds=2)

    assert session.query(Evaluation).count() == first


def test_a_restricted_run_touches_only_the_series_asked_for(
    session, source, history, monkeypatch
):
    """Re-scoring one series must not refetch and rescore the other six."""
    from dataclasses import replace as replace_fields

    other = replace_fields(source, slug="other-series", name="Other series")
    ingested: list[list[str] | None] = []

    def fake_ingest(active_session, slugs=None):
        ingested.append(slugs)
        for definition in (source, other):
            if slugs is None or definition.slug in slugs:
                series = upsert_series(active_session, definition)
                replace_observations(active_session, series, history)
        return {source.slug: len(history)}

    monkeypatch.setattr(pipeline, "ingest_all", fake_ingest)
    monkeypatch.setattr(pipeline, "SERIES", [source, other])

    pipeline.run(session, folds=2, slugs=[other.slug])

    assert ingested == [[other.slug]]
    assert aggregate(session, other.slug).shape[0] == 4
    assert aggregate(session, source.slug).empty
