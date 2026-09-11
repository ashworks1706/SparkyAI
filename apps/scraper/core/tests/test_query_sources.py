"""Query sources: term codes, URL building, and what the worker rejects."""

from __future__ import annotations

import uuid

import pytest
from scraper.core.types import Job, QueryError
from scraper.query.registry import QUERY_SOURCES, term_code, url_for
from scraper.query.worker import run_job


def test_term_codes_are_derived_rather_than_tabulated():
    assert term_code("Fall 2024") == "2247"
    assert term_code("Spring 2025") == "2251"
    assert term_code("Summer 2024") == "2244"
    assert term_code("fall 2026") == "2267"


@pytest.mark.parametrize("bad", ["Fall", "Autumn 2026", "Fall 26", ""])
def test_an_unusable_term_is_rejected_with_the_shape_it_wanted(bad):
    with pytest.raises(QueryError, match="term"):
        term_code(bad)


def test_class_search_url_carries_only_the_filters_that_were_given():
    url = url_for(
        QUERY_SOURCES["class_search"],
        {"term": "Fall 2026", "keywords": "CSE 310", "level": "undergraduate"},
    )
    assert url.startswith("https://catalog.apps.asu.edu/catalog/classes/classlist?")
    assert "term=2267" in url
    assert "keywords=CSE+310" in url
    assert "level=undergrad" in url
    # Empty filters are omitted from the URL.
    assert "daysOfWeek" not in url


def test_a_missing_required_parameter_names_itself():
    with pytest.raises(QueryError, match="needs: term"):
        url_for(QUERY_SOURCES["class_search"], {"keywords": "CSE 310"})


def test_an_unknown_parameter_lists_the_ones_that_exist():
    with pytest.raises(QueryError, match="no parameter"):
        url_for(QUERY_SOURCES["class_search"], {"term": "Fall 2026", "subject": "CSE"})


def test_an_unmapped_enum_value_says_what_is_allowed():
    with pytest.raises(QueryError, match="undergraduate"):
        url_for(QUERY_SOURCES["class_search"], {"term": "Fall 2026", "level": "phd"})


def test_scholarship_search_needs_nothing_and_narrows_with_what_it_gets():
    bare = url_for(QUERY_SOURCES["scholarship_search"], {})
    assert bare == "https://goglobal.asu.edu/scholarship-search"
    narrowed = url_for(QUERY_SOURCES["scholarship_search"], {"gpa": "3.0"})
    assert "gpa=3.0" in narrowed


def test_the_worker_rejects_an_unknown_source_without_fetching():
    job = Job(id=uuid.uuid4(), kind="source_query", input={"source": "nope", "params": {}})
    with pytest.raises(QueryError, match="unknown source"):
        run_job(job)


def test_the_worker_rejects_params_that_are_not_an_object():
    job = Job(id=uuid.uuid4(), kind="source_query", input={"source": "class_search", "params": []})
    with pytest.raises(QueryError, match="params must be an object"):
        run_job(job)
