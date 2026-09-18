from urllib.parse import urlparse

import pytest
import typer
from scraper.cli import selected
from scraper.sources import SOURCES
from scraper.sources.pages import OTHER_HOSTS, PAGES


def test_every_source_is_registered_once_with_a_url_and_category() -> None:
    assert SOURCES
    for key, src in SOURCES.items():
        assert src.key == key
        assert src.url.startswith("https://")
        assert src.category
        assert src.fetch_every_hours > 0


def test_every_page_is_an_asu_page_indexed_as_fetched() -> None:
    assert PAGES
    urls = [page.url for page in PAGES]
    assert len(urls) == len(set(urls))
    for page in PAGES:
        assert SOURCES[page.key] is page
        assert page.extractor is None
        host = urlparse(page.url).hostname or ""
        assert host == "asu.edu" or host.endswith(".asu.edu") or host in OTHER_HOSTS, page.url


def test_a_run_selects_one_source_a_category_or_all() -> None:
    assert selected("news", all_sources=False, category=None) == ["news"]
    assert selected(None, all_sources=True, category=None) == list(SOURCES)
    assert selected(None, all_sources=False, category="library") == [
        key for key, src in SOURCES.items() if src.category == "library"
    ]
    with pytest.raises(typer.BadParameter):
        selected(None, all_sources=False, category="no_such_category")
    with pytest.raises(typer.BadParameter):
        selected("news", all_sources=True, category=None)
    with pytest.raises(typer.BadParameter):
        selected(None, all_sources=False, category=None)
