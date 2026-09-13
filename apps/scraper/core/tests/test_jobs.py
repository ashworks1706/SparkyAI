"""The scraper job queue: what each kind does, what it queues next, and the order of claims."""

from __future__ import annotations

import uuid

import pytest
from scraper import jobs
from scraper.core.types import Job, QueryError, QueryResult, RunResult


class Recorder:
    """Stands in for a connection; enqueue_job is patched to record on it."""

    def __init__(self) -> None:
        self.queued: list[tuple[str, dict, int]] = []


def job(kind: str, **input: object) -> Job:
    return Job(id=uuid.uuid4(), kind=kind, input=dict(input))


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(
        jobs.postgres,
        "enqueue_job",
        lambda _conn, kind, input, priority: rec.queued.append((kind, input, priority)) or True,
    )
    return rec


def answered(monkeypatch, source: str, text: str) -> None:
    monkeypatch.setattr(
        jobs,
        "run_job",
        lambda _job: QueryResult(source=source, url=f"https://x.test/{source}", text=text),
    )


def test_live_queries_outrank_indexing_which_outranks_scheduled_runs():
    assert 0 > jobs.INDEX_PRIORITY > jobs.RUN_PRIORITY, "the engine queues at the default of 0"
    assert jobs.LIVE_KINDS == (jobs.QUERY,)
    assert set(jobs.BACKGROUND_KINDS) == {jobs.INDEX, jobs.RUN}


def test_a_live_answer_queues_the_indexing_of_its_whole_page(monkeypatch, recorder):
    answered(monkeypatch, "news", "x" * 9_000)
    result = jobs.handle(recorder, job(jobs.QUERY, source="news", params={}))
    assert result["url"] == "https://x.test/news"
    assert result["text"].endswith("[truncated]"), "the caller gets the held text"
    assert recorder.queued == [
        (
            jobs.INDEX,
            {"source": "news", "url": "https://x.test/news", "text": "x" * 9_000},
            jobs.INDEX_PRIORITY,
        )
    ], "the index gets all of it"


def test_an_answer_from_a_source_that_is_not_indexed_queues_nothing(monkeypatch, recorder):
    answered(monkeypatch, "shuttles", "Lot 37: next 6:54 PM")
    jobs.handle(recorder, job(jobs.QUERY, source="shuttles", params={}))
    assert recorder.queued == []


def test_indexing_can_be_switched_off(monkeypatch, recorder):
    answered(monkeypatch, "news", "stories")
    monkeypatch.setattr(jobs.settings().scraper, "index_live_results", False)
    jobs.handle(recorder, job(jobs.QUERY, source="news", params={}))
    assert recorder.queued == []


def test_an_index_job_indexes_the_page_it_carries(monkeypatch, recorder):
    seen = []

    def index_result(query, url, text):
        seen.append((query.key, url, text))
        return RunResult("news-abc", changed=True, chunks=3, content_hash="h")

    monkeypatch.setattr(jobs.index, "index_result", index_result)
    result = jobs.handle(recorder, job(jobs.INDEX, source="news", url="https://n", text="t"))
    assert seen == [("news", "https://n", "t")]
    assert result == {"source": "news-abc", "changed": True, "chunks": 3}


def test_a_run_job_runs_its_registered_source(monkeypatch, recorder):
    ran = []
    monkeypatch.setattr(
        jobs.pipeline,
        "run_source",
        lambda src: ran.append(src.key) or RunResult(src.key, False, 0, "h"),
    )
    assert jobs.handle(recorder, job(jobs.RUN, source="news"))["source"] == "news"
    assert ran == ["news"]
    with pytest.raises(QueryError, match="unknown source"):
        jobs.handle(recorder, job(jobs.RUN, source="nope"))


def test_a_kind_with_no_handler_is_rejected(recorder):
    with pytest.raises(QueryError, match="no handler"):
        jobs.handle(recorder, job("mystery"))
