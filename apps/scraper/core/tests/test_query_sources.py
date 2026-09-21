"""Query sources: term codes, URL building, and what the worker rejects."""

from __future__ import annotations

import uuid

import pytest
from scraper.core.types import Job, QueryError
from scraper.ingest.extract import extract_text
from scraper.query.params import term_code
from scraper.query.registry import QUERY_SOURCES, url_for
from scraper.query.run import run_job


def test_term_codes_are_derived_rather_than_tabulated():
    assert term_code("Fall 2024") == "2247"
    assert term_code("Spring 2025") == "2251"
    assert term_code("Summer 2024") == "2244"
    assert term_code("fall 2026") == "2267"


@pytest.mark.parametrize("bad", ["Fall", "Autumn 2026", "Fall 26", ""])
def test_an_unusable_term_is_rejected_with_the_shape_it_wanted(bad):
    with pytest.raises(QueryError, match="term"):
        term_code(bad)


def test_every_live_source_is_registered_under_its_own_key():
    assert set(QUERY_SOURCES) == {
        "courses",
        "course_catalog",
        "scholarships",
        "events",
        "clubs",
        "news",
        "library_catalog",
        "library_hours",
        "study_rooms",
        "sports",
        "sports_news",
        "shuttles",
        "campus_map",
        "social_media",
        "dining",
        "jobs",
        "web",
    }


def test_the_catalog_is_a_different_page_from_the_class_search():
    """Sections and seats come from classlist; descriptions and prerequisites from courselist."""
    catalog = url_for(QUERY_SOURCES["course_catalog"], {"keywords": "CSE 485"})
    assert catalog.startswith("https://catalog.apps.asu.edu/catalog/courses/courselist?")
    assert "keywords=CSE+485" in catalog
    assert "term" not in catalog, "a catalog entry is read for no term unless one was given"
    termed = url_for(QUERY_SOURCES["course_catalog"], {"keywords": "CSE 485", "term": "Fall 2026"})
    assert "term=2267" in termed


def test_dining_hours_come_from_the_pdf_of_the_campus_named():
    url = url_for(QUERY_SOURCES["dining"], {"campus": "tempe"})
    assert url.startswith("https://sundevilhospitality.asu.edu/")
    assert "Tempe" in url
    with pytest.raises(QueryError, match="campus"):
        url_for(QUERY_SOURCES["dining"], {})
    with pytest.raises(QueryError, match="campus"):
        url_for(QUERY_SOURCES["dining"], {"campus": "polytechnic"})


def test_course_search_url_carries_only_the_filters_that_were_given():
    url = url_for(
        QUERY_SOURCES["courses"],
        {"term": "Fall 2026", "keywords": "CSE 310", "level": "undergraduate"},
    )
    assert url.startswith("https://catalog.apps.asu.edu/catalog/classes/classlist?")
    assert "term=2267" in url
    assert "keywords=CSE+310" in url
    assert "level=undergrad" in url
    # Empty filters are omitted from the URL.
    assert "daysOfWeek" not in url
    assert "session" not in url


def test_a_missing_required_parameter_names_itself():
    with pytest.raises(QueryError, match="needs: term"):
        url_for(QUERY_SOURCES["courses"], {"keywords": "CSE 310"})


def test_an_unknown_parameter_lists_the_ones_that_exist():
    with pytest.raises(QueryError, match="no parameter"):
        url_for(QUERY_SOURCES["courses"], {"term": "Fall 2026", "subject": "CSE"})


def test_a_source_with_no_parameters_says_it_takes_nothing():
    with pytest.raises(QueryError, match="takes: nothing"):
        url_for(QUERY_SOURCES["library_hours"], {"library": "hayden"})
    assert url_for(QUERY_SOURCES["library_hours"], {}) == "https://lib.asu.edu/hours"


def test_an_unmapped_value_says_what_is_allowed():
    with pytest.raises(QueryError, match="undergraduate"):
        url_for(QUERY_SOURCES["courses"], {"term": "Fall 2026", "level": "phd"})
    with pytest.raises(QueryError, match="takes one value"):
        url_for(QUERY_SOURCES["scholarships"], {"focus": "stem, humanities"})


def test_scholarship_filters_become_the_option_ids_of_the_search_form():
    bare = url_for(QUERY_SOURCES["scholarships"], {})
    assert bare == "https://onsa.asu.edu/scholarships"
    narrowed = url_for(
        QUERY_SOURCES["scholarships"],
        {"keywords": "women", "citizenship": "US Citizen", "focus": "STEM"},
    )
    assert "combine=women" in narrowed
    assert "field_citizenship_status=75" in narrowed
    assert "field_focus=55" in narrowed


def test_study_rooms_need_a_known_library_and_an_iso_date():
    url = url_for(QUERY_SOURCES["study_rooms"], {"library": "Hayden", "date": "2026-09-14"})
    assert "lid=13858" in url and "gid=28619" in url and "date=2026-09-14" in url
    with pytest.raises(QueryError, match="date must look like"):
        url_for(QUERY_SOURCES["study_rooms"], {"library": "hayden", "date": "Sep 14"})
    with pytest.raises(QueryError, match="hayden"):
        url_for(QUERY_SOURCES["study_rooms"], {"library": "mars", "date": "2026-09-14"})


def test_a_sport_maps_to_its_schedule_page():
    url = url_for(QUERY_SOURCES["sports"], {"sport": "Men's Basketball"})
    assert url == "https://thesundevils.com/sports/mens-basketball/schedule"


def test_keyword_searches_encode_the_keyword():
    assert url_for(QUERY_SOURCES["clubs"], {"keywords": "machine learning"}) == (
        "https://sundevilcentral.eoss.asu.edu/club_signup?view=all&search=machine+learning"
    )
    assert "search=robotics" in url_for(QUERY_SOURCES["news"], {"keywords": "robotics"})
    assert url_for(QUERY_SOURCES["news"], {}) == "https://news.asu.edu/", (
        "no topic is the front page"
    )
    assert (
        url_for(QUERY_SOURCES["news"], {"keywords": "Latest ASU news"}) == "https://news.asu.edu/"
    )
    catalog = url_for(
        QUERY_SOURCES["library_catalog"], {"keywords": "deep learning", "type": "books"}
    )
    assert "any%2Ccontains%2Cdeep+learning%2CAND" in catalog
    assert "rtype%2Cexact%2Cbooks" in catalog


def test_a_source_that_shares_a_scheduled_page_reuses_its_extractor():
    from scraper.sources import SOURCES

    assert QUERY_SOURCES["library_hours"].extractor is SOURCES["library_hours"].extractor


def test_the_worker_rejects_an_unknown_source_without_fetching():
    job = Job(id=uuid.uuid4(), kind="source_query", input={"source": "nope", "params": {}})
    with pytest.raises(QueryError, match="unknown source"):
        run_job(job)


def test_the_worker_rejects_params_that_are_not_an_object():
    job = Job(id=uuid.uuid4(), kind="source_query", input={"source": "courses", "params": []})
    with pytest.raises(QueryError, match="params must be an object"):
        run_job(job)


def test_a_page_that_lists_results_inside_a_form_keeps_them():
    from scraper.core.types import Fetched

    html = b"<main><form><ul><li>Robotics Club</li></ul></form></main>"
    page = Fetched(url="https://x.test", status=200, body=html, content_type="text/html")
    assert "Robotics Club" in QUERY_SOURCES["clubs"].extractor(page)
    assert "Robotics Club" not in extract_text(html), "other pages still drop forms"


def test_a_source_sets_exactly_one_way_to_be_fetched():
    from scraper.core.types import QuerySource

    with pytest.raises(ValueError, match="exactly one"):
        QuerySource(key="x", description="x", params=())
    with pytest.raises(ValueError, match="exactly one"):
        QuerySource(
            key="x", description="x", params=(), to_url=lambda p: "", answer=lambda p: ("", "")
        )


def test_choices_are_checked_without_case_before_anything_is_fetched():
    from scraper.query.registry import check

    check(QUERY_SOURCES["courses"], {"term": "Fall 2026", "days": "Monday, WEDNESDAY"})
    with pytest.raises(QueryError, match="is not one of"):
        check(QUERY_SOURCES["courses"], {"term": "Fall 2026", "days": "Funday"})
    with pytest.raises(QueryError, match="takes one value"):
        check(QUERY_SOURCES["shuttles"], {"route": "mercado, polytechnic-tempe"})


def test_shuttle_times_are_grouped_by_route_soonest_first_in_arizona_time():
    import datetime

    from scraper.query.http import ARIZONA
    from scraper.query.sources.shuttles import render

    now = datetime.datetime(2026, 9, 12, 16, 0, tzinfo=ARIZONA)
    at = lambda h, m: int(datetime.datetime(2026, 9, 12, h, m, tzinfo=ARIZONA).timestamp())  # noqa: E731
    routes = [
        {"routeID": 1, "longName": "Mercado"},
        {"routeID": 2, "longName": "Polytechnic-Tempe"},
        {"routeID": 3, "longName": "Hidden", "hidden": True},
    ]
    stops = [{"stopID": 10, "longName": "Lot 37"}, {"stopID": 11, "longName": "SIM Building"}]
    etas = [
        {"stopID": 10, "routeID": 2, "ETA1": at(16, 54), "ETA2": at(18, 54)},
        {"stopID": 11, "routeID": 2, "ETA1": at(16, 44), "ETA2": 0},
    ]
    out = render(routes, stops, etas, None, now)
    assert "as of 4:00 PM" in out
    assert "Route: Mercado\nNo bus is scheduled on this route now." in out
    assert out.index("SIM Building: next 4:44 PM") < out.index("Lot 37: next 4:54 PM, then 6:54 PM")
    assert "Hidden" not in out
    assert "Mercado" not in render(routes, stops, etas, "Polytechnic-Tempe", now)


def test_campus_places_are_distinct_plain_and_linked_to_a_map():
    from scraper.query.sources.campus_map import render

    hayden = {
        "attributes": {
            "Name": "Hayden Library",
            "Type": "Library",
            "Description": "<p>Books &amp; more</p>",
        },
        "geometry": {"x": -111.935242, "y": 33.418938},
    }
    out = render("hayden", [hayden, hayden])
    assert out.count("Hayden Library") == 1
    assert "Hayden Library (Library): Books & more" in out
    assert "query=33.418938,-111.935242" in out
    assert "No place" in render("nowhere", [])


def test_official_sports_news_is_narrowed_to_the_sport_tag():
    import xml.etree.ElementTree as ET

    from scraper.query.sources.sports_news import official_lines

    feed = ET.fromstring(
        "<rss><channel>"
        "<item><title>Football wins</title><link>https://t/1</link>"
        "<pubDate>Sat, 12 Sep 2026 03:55:21 GMT</pubDate><sports>Football</sports></item>"
        "<item><title>Volleyball wins</title><link>https://t/2</link><sports>Volleyball</sports></item>"
        "</channel></rss>"
    )
    lines = official_lines(feed, "football", "")
    assert lines == ["Fri Sep 11, 2026 | Football wins | https://t/1"]
    assert len(official_lines(feed, "", "")) == 2
    assert official_lines(feed, "", "volleyball") == ["undated | Volleyball wins | https://t/2"]


def test_channel_posts_carry_date_title_link_and_are_narrowed_by_keyword():
    import xml.etree.ElementTree as ET

    from scraper.query.sources.social_media import posts_of

    feed = ET.fromstring(
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:media="http://search.yahoo.com/mrss/">'
        '<entry><title>Football intro</title><link href="https://y/1"/>'
        "<published>2026-09-12T15:45:23+00:00</published>"
        "<media:group><media:description>Season opener</media:description></media:group></entry>"
        '<entry><title>Scholarships</title><link href="https://y/2"/></entry>'
        "</feed>"
    )
    assert posts_of(feed, "football") == [
        "Sat Sep 12, 2026 | Football intro | https://y/1 | Season opener"
    ]
    assert len(posts_of(feed, "")) == 2


def test_web_results_are_listed_once_each_with_title_link_date_and_snippet():
    from scraper.query.sources.web import render

    found = {
        "answers": [{"answer": "ASU beat Morgan State 70-7."}],
        "results": [
            {
                "url": "https://thesundevils.com/sports/football/schedule",
                "title": "2026 Football Schedule",
                "content": "<b>Sun Devil</b> football &amp; more",
                "publishedDate": "2026-09-05T00:00:00",
            },
            {"url": "https://thesundevils.com/sports/football/schedule", "title": "duplicate"},
            {"url": "https://espn.com/asu", "title": "ESPN", "content": ""},
        ],
    }
    out = render("asu football", found, limit=8, snippet_chars=300)
    assert out.splitlines()[1] == "Answer: ASU beat Morgan State 70-7."
    assert (
        "1. 2026 Football Schedule | https://thesundevils.com/sports/football/schedule | 2026-09-05"
        in out
    )
    assert "Sun Devil football & more" in out
    assert "duplicate" not in out
    assert "2. ESPN | https://espn.com/asu" in out


def test_a_web_search_with_no_results_says_which_engines_did_not_answer():
    from scraper.query.sources.web import render

    with pytest.raises(QueryError, match="duckduckgo"):
        render("x", {"results": [], "unresponsive_engines": [["duckduckgo", "CAPTCHA"]]}, 8, 300)
