import pytest
from scraper.core.settings import Scraper
from scraper.core.types import PipelineError
from scraper.ingest.pipeline import check_quality_floor

RATIO = Scraper().quality_floor_ratio
MIN_CHARS = Scraper().quality_floor_min_chars


def check(text_chars: int, previous_chars: int | None) -> None:
    check_quality_floor("news", text_chars, previous_chars, ratio=RATIO, min_chars=MIN_CHARS)


def test_an_unknown_previous_size_skips_the_floor() -> None:
    # A first run, or a version written before text_chars was recorded.
    check(10, None)


def test_a_short_previous_version_is_below_the_comparison_floor() -> None:
    check(1, MIN_CHARS - 1)


def test_a_run_that_grows_is_accepted() -> None:
    check(40_000, 20_000)


def test_ordinary_churn_is_accepted() -> None:
    check(19_000, 20_000)
    check(11_000, 20_000)


def test_a_run_at_the_floor_is_accepted() -> None:
    check(10_000, 20_000)


def test_a_run_under_the_floor_is_refused() -> None:
    with pytest.raises(PipelineError) as e:
        check(9_999, 20_000)
    assert "9999" in str(e.value)
    assert "20000" in str(e.value)
    assert "10000" in str(e.value)


def test_a_near_empty_run_over_a_large_version_is_refused() -> None:
    with pytest.raises(PipelineError, match="refusing to replace the index"):
        check(12, 40_000)
