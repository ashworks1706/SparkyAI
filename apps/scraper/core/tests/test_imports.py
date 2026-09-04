import scraper
import scraper.sources


def test_package_imports() -> None:
    assert scraper.__doc__
    assert scraper.sources.__doc__
