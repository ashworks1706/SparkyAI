import knowledge
import knowledge.scraper


def test_package_imports() -> None:
    assert knowledge.__doc__
    assert knowledge.scraper.__doc__
