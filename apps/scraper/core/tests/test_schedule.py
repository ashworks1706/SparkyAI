from datetime import UTC, datetime, timedelta

from scraper.jobs import is_due

NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def row(**over: object) -> dict:
    base = {"enabled": True, "fetch_every": timedelta(hours=6), "last_attempt": None}
    base.update(over)
    return base


def test_a_source_with_no_row_yet_is_due() -> None:
    assert is_due(None, NOW)


def test_a_source_that_has_never_been_attempted_is_due() -> None:
    assert is_due(row(), NOW)


def test_a_disabled_source_is_never_due() -> None:
    assert not is_due(row(enabled=False, last_attempt=NOW - timedelta(days=30)), NOW)


def test_an_attempt_inside_the_interval_is_not_due() -> None:
    assert not is_due(row(last_attempt=NOW - timedelta(hours=5)), NOW)


def test_an_attempt_past_the_interval_is_due() -> None:
    assert is_due(row(last_attempt=NOW - timedelta(hours=6)), NOW)


def test_unchanged_content_still_holds_the_source_off_until_the_interval_elapses() -> None:
    # last_attempt advances even when no new version was written.
    unchanged = row(last_attempt=NOW - timedelta(minutes=5))
    assert not is_due(unchanged, NOW)
    assert is_due(row(last_attempt=NOW - timedelta(hours=7)), NOW)


def test_the_interval_comes_from_the_row_not_a_constant() -> None:
    slow = row(fetch_every=timedelta(days=7), last_attempt=NOW - timedelta(days=1))
    assert not is_due(slow, NOW)
    fast = row(fetch_every=timedelta(hours=1), last_attempt=NOW - timedelta(days=1))
    assert is_due(fast, NOW)
