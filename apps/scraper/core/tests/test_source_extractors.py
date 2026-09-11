from pathlib import Path

import pytest
from scraper.chunk import chunk_text
from scraper.core.types import Fetched
from scraper.sources import SOURCES
from scraper.sources.courses import extract_courses
from scraper.sources.events import extract_events
from scraper.sources.library_hours import extract_hours

FIXTURES = Path(__file__).parent / "fixtures"


def page(name: str) -> Fetched:
    markdown = (FIXTURES / name).read_text(encoding="utf-8")
    return Fetched(
        url=f"https://example.invalid/{name}",
        status=200,
        body=markdown.encode("utf-8"),
        content_type="text/markdown",
        text=markdown,
    )


def line_with(text: str, needle: str) -> str:
    matches = [line for line in text.splitlines() if needle in line]
    assert matches, f"no line carries {needle!r}"
    return matches[0]


def test_library_hours_keeps_a_whole_week_on_one_line() -> None:
    text = extract_hours(page("library_hours.md"))
    hayden = line_with(text, "Hayden Library")
    assert "Monday 7 a.m. - 2 a.m." in hayden
    assert "Friday 7 a.m. - 8 p.m." in hayden
    assert "Sunday 10 a.m. - 2 a.m." in hayden
    assert "Saturday Closed" in line_with(text, "Noble Library")
    assert "Monday 7:30 a.m. - 10 p.m." in line_with(text, "Fletcher Library")
    assert "September 8 - September 14, 2025" in text
    assert "|" not in text
    assert "---" not in text
    assert "https://" not in text


def test_events_keep_name_date_and_location_together() -> None:
    text = extract_events(page("events.md"))
    welcome = line_with(text, "Sun Devil Welcome")
    assert "Thursday, August 21, 2025 5:00 p.m. - 7:00 p.m." in welcome
    assert "Old Main Lawn, Tempe campus" in welcome
    fair = line_with(text, "Career Fair")
    assert "September 10, 2025 10:00 a.m. - 3:00 p.m." in fair
    assert "Memorial Union, Ventana Ballroom, Tempe campus" in fair
    showcase = line_with(text, "Graduate Research Showcase")
    assert "Friday, October 3, 2025 1:00 p.m." in showcase
    assert "Online via Zoom" in showcase
    assert "Meet student organizations" in text
    assert "https://" not in text


def test_courses_keep_number_title_and_units_together() -> None:
    text = extract_courses(page("courses.md"))
    assert "CSE 110 | Principles of Programming | 3 units" in text
    assert "CSE 310 | Data Structures and Algorithms | 3 units" in text
    assert "ACC 231 | Uses of Accounting Information I | 3 units" in text
    assert "ACC 494 | Special Topics in Accounting | 1 unit" in text
    assert "Fall 2025" in text
    assert "https://" not in text


@pytest.mark.parametrize("key", ["library_hours", "events", "courses"])
def test_registered_source_carries_its_extractor(key: str) -> None:
    assert callable(SOURCES[key].extractor)


@pytest.mark.parametrize(
    "extractor",
    [extract_hours, extract_events, extract_courses],
    ids=["hours", "events", "courses"],
)
def test_extractors_fall_back_to_the_raw_body_when_the_fetcher_gave_no_text(extractor) -> None:
    html = b"<html><body><main><p>CSE 110: Principles of Programming (3)</p></main></body></html>"
    fetched = Fetched(
        url="https://example.invalid/page", status=200, body=html, content_type="text/html"
    )
    assert "Principles of Programming" in extractor(fetched)


def test_an_extractor_that_returns_nothing_yields_no_chunks_for_the_pipeline_guard() -> None:
    empty = Fetched(
        url="https://example.invalid/empty",
        status=200,
        body=b"",
        content_type="text/markdown",
        text="   \n\n",
    )
    assert chunk_text(extract_hours(empty)) == []
