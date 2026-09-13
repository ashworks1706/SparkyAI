"""What a live query result becomes in the retrieval index once its caller has the answer."""

from __future__ import annotations

import pytest
from scraper.query import index
from scraper.query.registry import QUERY_SOURCES
from scraper.query.run import for_caller, should_index
from scraper.sources import SOURCES


def test_a_page_that_is_a_scheduled_source_refreshes_that_source():
    hours = QUERY_SOURCES["library_hours"]
    source = index.page_source(hours, SOURCES["library_hours"].url)
    assert source.key == "library_hours"
    assert source.category == SOURCES["library_hours"].category
    assert source.extractor is None, "the live text is already extracted"


def test_any_other_page_is_its_own_source_keyed_by_query_and_url():
    courses = QUERY_SOURCES["courses"]
    one = index.page_source(
        courses, "https://catalog.apps.asu.edu/catalog/classes/classlist?term=2267"
    )
    again = index.page_source(
        courses, "https://catalog.apps.asu.edu/catalog/classes/classlist?term=2267"
    )
    other = index.page_source(
        courses, "https://catalog.apps.asu.edu/catalog/classes/classlist?term=2271"
    )
    assert one.key == again.key, "the same search refreshes the same source"
    assert one.key != other.key
    assert one.key.startswith("courses-")
    assert one.category == "courses"
    assert one.key not in SOURCES, "a live page never takes a scheduled key"


def test_a_live_page_carries_its_text_as_already_extracted():
    news = QUERY_SOURCES["news"]
    page = index.live_page(news, "https://news.asu.edu/", "ASU ranks No. 1")
    assert page.text == "ASU ranks No. 1"
    assert page.body == b"ASU ranks No. 1"
    assert page.url == "https://news.asu.edu/"
    assert page.title == "news: https://news.asu.edu/", "the title names the source and the page"


def test_minute_to_minute_answers_are_not_indexed():
    assert {k for k, q in QUERY_SOURCES.items() if not q.index} == {
        "shuttles",
        "study_rooms",
        "web",
    }
    assert should_index(QUERY_SOURCES["news"], enabled=True)
    assert not should_index(QUERY_SOURCES["shuttles"], enabled=True)
    assert not should_index(QUERY_SOURCES["news"], enabled=False)


def test_every_query_source_names_the_category_it_is_indexed_under():
    for key, query in QUERY_SOURCES.items():
        assert query.category and query.category != "live", key


def test_a_failed_index_raises_so_its_job_records_the_failure(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("embedding server is down")

    monkeypatch.setattr(index.pipeline, "index_page", refuse)
    with pytest.raises(RuntimeError, match="embedding"):
        index.index_result(QUERY_SOURCES["news"], "https://news.asu.edu/", "text")


def test_the_caller_gets_a_held_text_while_the_index_gets_all_of_it():
    assert for_caller("short", 10) == "short"
    assert for_caller("x" * 20, 10) == "x" * 10 + "\n[truncated]"
