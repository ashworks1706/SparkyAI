from scraper.sources import SOURCES


def test_every_source_is_registered_once_with_a_url_and_category() -> None:
    assert SOURCES
    for key, src in SOURCES.items():
        assert src.key == key
        assert src.url.startswith("https://")
        assert src.category
        assert src.fetch_every_hours > 0
