"""Tests for the admin authenticated driver: routing, session detection, and the events merge."""

from __future__ import annotations

from pathlib import Path

import pytest
from scraper.core.settings import settings
from scraper.core.types import AuthError, Fetched, QuerySource
from scraper.ingest import fetch
from scraper.ingest.drivers import admin, public
from scraper.ingest.drivers.asu_sso import Credentials, on_host
from scraper.ingest.drivers.page import Loaded, to_fetched
from scraper.query import sundevil_central
from scraper.query.registry import QUERY_SOURCES, url_for
from scraper.query.sources import clubs, events


def _fetched(url: str) -> Fetched:
    return Fetched(url=url, status=200, body=b"<html></html>", content_type="text/html")


def test_sign_in_hosts_and_service_login_pages_are_recognised() -> None:
    assert admin.is_login_url("https://weblogin.asu.edu/cas/login?service=x")
    assert admin.is_login_url("https://idp.asu.edu/idp/profile/SAML2")
    assert admin.is_login_url(
        "https://sundevilcentral.eoss.asu.edu/webapp/auth/login?msg=LOGIN_REQUIRED&redirect=events"
    )
    assert not admin.is_login_url("https://sundevilcentral.eoss.asu.edu/events")
    assert not admin.is_login_url("https://sundevilcentral.eoss.asu.edu/club_signup")


def test_on_host_matches_subdomains_only() -> None:
    assert on_host("https://api-1234.duosecurity.com/frame", ("duosecurity.com",))
    assert not on_host("https://notduosecurity.com/", ("duosecurity.com",))


def test_credentials_never_print_the_password() -> None:
    creds = Credentials(username="sparky", password="hunter2")
    assert "hunter2" not in repr(creds)
    assert "hunter2" not in str(creds)


def test_missing_session_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "state_path", lambda: Path("/nonexistent/admin_state.json"))
    with pytest.raises(AuthError):
        admin.fetch_authenticated("https://sundevilcentral.eoss.asu.edu/events")
    assert admin.check() is False


def test_fetch_routes_authenticated_sources_to_the_admin_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake(url: str) -> Fetched:
        seen.append(url)
        return _fetched(url)

    def public_forbidden(url: str) -> Fetched:
        raise AssertionError("an authenticated fetch reached the public driver")

    monkeypatch.setattr(admin, "fetch_authenticated", fake)
    monkeypatch.setattr(public, "fetch_rendered", public_forbidden)
    fetch.fetch("https://sundevilcentral.eoss.asu.edu/club_signup", needs_js=True, auth=True)
    assert seen == ["https://sundevilcentral.eoss.asu.edu/club_signup"]


def test_a_loaded_page_maps_status_to_fetch_errors() -> None:
    page = to_fetched("https://x.invalid", Loaded(status=200, url="https://x.invalid/", html="<p>"))
    assert page.body == b"<p>"
    with pytest.raises(Exception, match="404"):
        to_fetched("https://x.invalid", Loaded(status=404, url="https://x.invalid", html=""))
    with pytest.raises(Exception, match="no response"):
        to_fetched("https://x.invalid", Loaded(status=0, url="https://x.invalid", html=""))


def test_clubs_is_live_only_and_authenticated() -> None:
    assert clubs.QUERY.auth is True
    assert clubs.QUERY.index is False
    assert clubs.QUERY.needs_js is True


def test_an_authenticated_source_may_not_be_indexed() -> None:
    with pytest.raises(ValueError, match="index=False"):
        QuerySource(
            key="x",
            description="d",
            params=(),
            to_url=lambda _p: "https://example.invalid",
            auth=True,
            index=True,
        )


def test_events_merges_public_and_sun_devil_central(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(events, "_public", lambda _k: ("https://asuevents.asu.edu/home", "Fair"))
    monkeypatch.setattr(events, "_sundevil", lambda _k: ("https://sdc.invalid/events", "Mixer"))
    url, text = events.answer({"keywords": "career"})
    assert url == "https://sdc.invalid/events"
    assert "ASU Events" in text and "Fair" in text
    assert "Sun Devil Central" in text and "Mixer" in text
    assert text.index("Sun Devil Central") < text.index("ASU Events")


def test_events_gives_each_section_a_share_of_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    long = "\n".join(f"public event {i}" for i in range(2000))
    monkeypatch.setattr(events, "_public", lambda _k: ("https://asuevents.asu.edu/home", long))
    monkeypatch.setattr(events, "_sundevil", lambda _k: ("https://sdc.invalid/events", long))
    _, text = events.answer({})
    assert len(text) <= settings().scraper.query_max_chars + 200
    assert "Sun Devil Central" in text and "ASU Events" in text


def test_events_falls_back_to_public_when_the_session_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(events, "_public", lambda _k: ("https://asuevents.asu.edu/home", "Fair"))

    def gone(_k: str) -> tuple[str, str]:
        raise AuthError("expired")

    monkeypatch.setattr(events, "_sundevil", gone)
    url, text = events.answer({"keywords": "career"})
    assert "ASU Events" in text and "Fair" in text
    assert "Sun Devil Central" not in text


def test_events_raises_when_no_section_can_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_k: str) -> tuple[str, str]:
        raise RuntimeError("down")

    monkeypatch.setattr(events, "_public", fail)
    monkeypatch.setattr(events, "_sundevil", fail)
    with pytest.raises(Exception, match="could not be read"):
        events.answer({"keywords": "career"})


def _fixture(name: str) -> Fetched:
    body = (Path(__file__).parent / "fixtures" / name).read_bytes()
    return Fetched(
        url="https://sundevilcentral.eoss.asu.edu/", status=200, body=body, content_type="text/html"
    )


def test_sun_devil_central_clubs_become_one_entry_each() -> None:
    text = sundevil_central.extract_clubs(_fixture("sundevil_clubs.html"))
    assert text.startswith("1 matching groups of 786")
    assert (
        "Robotics Club | Tempe - Academic, Special Interest | "
        "https://sundevilcentral.eoss.asu.edu/student_community?club_id=101" in text
    )
    assert "contact Pat Example" in text
    assert "Mission: We build robots together." in text
    assert "Membership Benefits: Lab access." in text
    assert "Register new" not in text


def test_sun_devil_central_events_become_one_line_each() -> None:
    text = sundevil_central.extract_events(_fixture("sundevil_events.html"))
    lines = text.splitlines()
    assert lines[0] == "1 upcoming events shown of 3025"
    assert lines[1].startswith("Career Soda Social | Mon, Sep 21, 2026 11 AM")
    assert "Flex Space, Fusion on First | 3 going | FREE | tags Social, In-Person Event" in lines[1]
    assert lines[1].endswith("https://sundevilcentral.eoss.asu.edu/rsvp_boot?id=555")
    assert "filter list" not in text


def test_changed_sun_devil_central_markup_falls_back_to_page_text() -> None:
    page = Fetched(
        url="x", status=200, body=b"<main><p>New layout event</p></main>", content_type="text/html"
    )
    assert "New layout event" in sundevil_central.extract_events(page)
    assert "New layout event" in sundevil_central.extract_clubs(page)


def test_sun_devil_central_events_search_by_search_word(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(url: str, **_kw: object) -> Fetched:
        seen.append(url)
        return _fixture("sundevil_events.html")

    monkeypatch.setattr(fetch, "fetch", fake)
    events._sundevil("career fair")
    assert seen == ["https://sundevilcentral.eoss.asu.edu/events?search_word=career+fair"]


def test_filler_words_are_dropped_from_sun_devil_central_searches() -> None:
    assert sundevil_central.keywords("robotics clubs ASU") == "robotics"
    assert sundevil_central.keywords("career events this week") == "career"
    assert sundevil_central.keywords("upcoming events Sun Devil Central") == ""
    assert sundevil_central.keywords("Clubs at ASU?") == ""
    assert url_for(QUERY_SOURCES["clubs"], {"keywords": "ASU robotics club"}).endswith(
        "search=robotics"
    )


def test_an_empty_sun_devil_central_listing_says_so() -> None:
    groups = Fetched(
        url="x", status=200, body=b"<h1>Groups (786)</h1><ul></ul>", content_type="text/html"
    )
    assert sundevil_central.extract_clubs(groups).startswith("No groups match")
    none = Fetched(
        url="x",
        status=200,
        body=b"<h1>Events (0)</h1><p>No result found</p>",
        content_type="text/html",
    )
    assert sundevil_central.extract_events(none).startswith("No upcoming")


def _clubs_page(search: str, *names_and_missions: tuple[str, str]) -> Fetched:
    items = "".join(
        f'<li class="list-group-item"><input id="cb_club_{i}"/><h2><a>{name}</a></h2>'
        f'<p id="club_{i}"><strong>Mission</strong> {mission}</p></li>'
        for i, (name, mission) in enumerate(names_and_missions)
    )
    body = f"<h1>Groups (786)</h1><ul>{items}</ul>".encode()
    url = f"https://sundevilcentral.eoss.asu.edu/club_signup?view=all&search={search}"
    return Fetched(url=url, status=200, body=body, content_type="text/html")


def test_clubs_keep_whole_word_matches_with_name_matches_first() -> None:
    page = _clubs_page(
        "AI",
        ("First Draft: Creative Writing Club", "We write fiction and poetry."),
        ("Robotics Club", "We apply AI to robots."),
        ("The AI Society at ASU", "Learn machine learning."),
        ("Chair Yoga", "Stretch."),
    )
    text = sundevil_central.extract_clubs(page)
    assert text.startswith("2 matching groups of 786 for AI (2 partial-word matches left out)")
    assert text.index("The AI Society") < text.index("Robotics Club")
    assert "Creative Writing" not in text and "Chair Yoga" not in text


def test_clubs_keep_everything_when_nothing_matches_whole() -> None:
    page = _clubs_page("rob", ("Robotics Club", "Robots."))
    text = sundevil_central.extract_clubs(page)
    assert "Robotics Club" in text and "partial matches follow" in text


def test_events_keeps_one_section_when_the_other_cannot_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    def refused(_k: str) -> tuple[str, str]:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(events, "_public", refused)
    monkeypatch.setattr(events, "_sundevil", lambda _k: ("https://sdc.invalid/events", "Mixer"))
    _, text = events.answer({"keywords": "career"})
    assert "Sun Devil Central" in text and "Mixer" in text


def test_no_more_browsers_than_the_cap_are_open_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    from scraper.ingest.drivers import page

    page._browsers.cache_clear()
    monkeypatch.setattr(settings().scraper, "max_browsers", 2)
    open_now = 0
    most = 0
    lock = threading.Lock()

    def browse() -> None:
        nonlocal open_now, most
        with page.browser_slot():
            with lock:
                open_now += 1
                most = max(most, open_now)
            time.sleep(0.05)
            with lock:
                open_now -= 1

    threads = [threading.Thread(target=browse) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    page._browsers.cache_clear()
    assert most == 2


def test_a_page_past_the_size_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    from scraper.core.types import FetchRejected

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5_000)

    real = httpx.Client

    def client(**kw):  # noqa: ANN003, ANN202
        return real(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(fetch.httpx, "Client", client)
    monkeypatch.setattr(settings().scraper, "max_page_bytes", 1_000)
    with pytest.raises(FetchRejected, match="larger than"):
        fetch.fetch_http.__wrapped__("https://x.test/big")
    monkeypatch.setattr(settings().scraper, "max_page_bytes", 10_000)
    assert len(fetch.fetch_http.__wrapped__("https://x.test/big").body) == 5_000
