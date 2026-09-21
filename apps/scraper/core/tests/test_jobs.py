"""The scraper job queue: what each kind does, what it queues next, and the order of claims."""

from __future__ import annotations

import uuid

import pytest
from scraper import jobs
from scraper.core.types import Job, QueryError, QueryResult, RunResult
from scraper.store import postgres


class Recorder:
    """Stands in for a connection; enqueue_job is patched to record on it."""

    def __init__(self) -> None:
        self.queued: list[tuple[str, dict, int]] = []
        self.backlog_full = False


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
    monkeypatch.setattr(
        jobs.postgres, "backlog_at_least", lambda _conn, _kind, _limit: rec.backlog_full
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
    answered(monkeypatch, "news", "x" * 40_000)
    result = jobs.handle(recorder, job(jobs.QUERY, source="news", params={}))
    assert result["url"] == "https://x.test/news"
    assert result["text"].endswith("[truncated]"), "the caller gets the held text"
    assert recorder.queued == [
        (
            jobs.INDEX,
            {"source": "news", "url": "https://x.test/news", "text": "x" * 40_000},
            jobs.INDEX_PRIORITY,
        )
    ], "the index gets all of it"


def test_an_answer_from_a_source_that_is_not_indexed_queues_nothing(monkeypatch, recorder):
    answered(monkeypatch, "shuttles", "Lot 37: next 6:54 PM")
    jobs.handle(recorder, job(jobs.QUERY, source="shuttles", params={}))
    assert recorder.queued == []


def test_a_deep_index_backlog_answers_the_caller_and_drops_the_indexing(monkeypatch, recorder):
    # A student is waiting on the answer; the source is refetched on its schedule anyway.
    answered(monkeypatch, "news", "stories")
    recorder.backlog_full = True
    result = jobs.handle(recorder, job(jobs.QUERY, source="news", params={}))
    assert result["text"] == "stories", "the answer is unaffected"
    assert recorder.queued == [], "nothing is added to a queue that is already deep"


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
        lambda src, pacer=None: ran.append(src.key) or RunResult(src.key, False, 0, "h"),
    )
    assert jobs.handle(recorder, job(jobs.RUN, source="news"))["source"] == "news"
    assert ran == ["news"]
    with pytest.raises(QueryError, match="unknown source"):
        jobs.handle(recorder, job(jobs.RUN, source="nope"))


def test_a_kind_with_no_handler_is_rejected(recorder):
    with pytest.raises(QueryError, match="no handler"):
        jobs.handle(recorder, job("mystery"))


class FakeCursor:
    """Records the SQL a pruning or backlog call issues and answers it."""

    def __init__(self, row: tuple | None = None, rowcount: int = 0) -> None:
        self.row = row
        self.rowcount = rowcount
        self.sql: list[str] = []
        self.args: list[tuple] = []

    def execute(self, sql: str, args: tuple = ()) -> FakeCursor:
        self.sql.append(" ".join(sql.split()))
        self.args.append(args)
        return self

    def fetchone(self) -> tuple | None:
        return self.row


def test_a_backlog_check_reads_no_further_than_the_limit():
    conn = FakeCursor(row=(1,))
    assert postgres.backlog_at_least(conn, "live_index", 500)
    assert conn.args[-1] == ("live_index", 499), "it looks for one row past the limit"
    assert "limit 1" in conn.sql[-1], "and never reads the whole backlog"

    empty = FakeCursor(row=None)
    assert not postgres.backlog_at_least(empty, "live_index", 500)

    # A limit of zero is no limit at all, so nothing is read.
    unread = FakeCursor(row=(1,))
    assert not postgres.backlog_at_least(unread, "live_index", 0)
    assert unread.sql == []


def test_pruning_finished_jobs_is_batched_and_leaves_running_ones_alone():
    conn = FakeCursor(rowcount=12)
    assert postgres.prune_jobs(conn, 3600.0, 5_000) == 12
    sql = conn.sql[-1]
    assert "delete from jobs" in sql
    assert "status in ('done', 'failed', 'cancelled')" in sql, "queued and running are kept"
    assert "limit %s" in sql, "one cycle never deletes unbounded rows"
    assert conn.args[-1] == (3600.0, 5_000)

    # Retention off keeps everything and issues nothing.
    off = FakeCursor(rowcount=9)
    assert postgres.prune_jobs(off, 0, 5_000) == 0
    assert off.sql == []


def test_a_migration_that_cannot_run_in_a_transaction_is_split_into_its_statements():
    # Several statements in one execute put Postgres in an implicit transaction, which
    # create index concurrently refuses, so such a file is sent one statement at a time.
    sql = """
    -- concurrent: a comment; with a semicolon in it
    set lock_timeout = '5s';

    create index concurrently if not exists a_idx on jobs (kind) where status = 'queued';
    drop index if exists old_idx;
    """
    split = postgres.statements(sql)
    assert len(split) == 3, split
    assert split[0].endswith("'5s'"), "the comment above it rides along, its semicolon ignored"
    assert split[1].startswith("create index concurrently")
    assert split[2] == "drop index if exists old_idx"

    # A semicolon inside a quoted value does not end a statement.
    quoted = postgres.statements("select ';' as a; select 2")
    assert quoted == ["select ';' as a", "select 2"]

    # Trailing whitespace and comment-only tails produce no empty statement.
    assert postgres.statements("select 1;\n-- done\n") == ["select 1"]


def test_every_migration_using_concurrently_says_it_runs_outside_a_transaction():
    from pathlib import Path

    for path in sorted(postgres.MIGRATIONS_DIR.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        if "concurrently" in sql.lower():
            assert sql.lstrip().startswith(postgres.CONCURRENT_MARKER), (
                f"{Path(path).name} would fail inside a transaction"
            )
