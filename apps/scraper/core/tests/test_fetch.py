import pytest
from scraper.core.types import FetchError, FetchRejected
from scraper.fetch import parse_firecrawl

URL = "https://lib.asu.edu/hours"


def _payload(**data):
    return {"success": True, "data": data}


def test_markdown_and_metadata_become_a_page() -> None:
    page = parse_firecrawl(
        URL,
        _payload(
            markdown="# Hours\nOpen 7am",
            metadata={"statusCode": 200, "title": "Hours", "url": URL + "/"},
        ),
    )
    assert page.text == "# Hours\nOpen 7am"
    assert page.title == "Hours"
    assert page.url == URL + "/"
    assert page.content_type == "text/markdown"


def test_missing_markdown_or_status_is_a_fetch_error() -> None:
    with pytest.raises(FetchError):
        parse_firecrawl(URL, _payload(markdown="", metadata={"statusCode": 200}))
    with pytest.raises(FetchError):
        parse_firecrawl(URL, _payload(markdown="x", metadata={}))
    with pytest.raises(FetchError):
        parse_firecrawl(URL, {"success": False, "error": "timeout"})


def test_origin_4xx_is_rejected_not_retried() -> None:
    with pytest.raises(FetchRejected):
        parse_firecrawl(URL, _payload(markdown="Not found", metadata={"statusCode": 404}))
