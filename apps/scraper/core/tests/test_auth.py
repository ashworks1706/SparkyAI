"""Tests for the admin authenticated driver: routing, session detection, and the events merge."""

from __future__ import annotations

from pathlib import Path

import pytest
from scraper.core.types import AuthError, Fetched, QuerySource
from scraper.ingest import auth, fetch
from scraper.query.sources import clubs, events


def _fetched(url: str) -> Fetched:
    return Fetched(url=url, status=200, body=b"<html></html>", content_type="text/html")


def test_login_hosts_are_recognised() -> None:
    assert auth.is_login_url("https://weblogin.asu.edu/cas/login?service=x")
    assert auth.is_login_url("https://idp.asu.edu/idp/profile/SAML2")
    assert not auth.is_login_url("https://sundevilcentral.eoss.asu.edu/events")


def test_missing_session_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "state_path", lambda: Path("/nonexistent/admin_state.json"))
    with pytest.raises(AuthError):
        auth.fetch_authenticated("https://sundevilcentral.eoss.asu.edu/events")


def test_fetch_routes_authenticated_sources_to_the_admin_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake(url: str) -> Fetched:
        seen.append(url)
        return _fetched(url)

    monkeypatch.setattr(auth, "fetch_authenticated", fake)
    fetch.fetch("https://sundevilcentral.eoss.asu.edu/club_signup", needs_js=True, auth=True)
    assert seen == ["https://sundevilcentral.eoss.asu.edu/club_signup"]


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
    assert url == "https://asuevents.asu.edu/home"
    assert "ASU Events" in text and "Fair" in text
    assert "Sun Devil Central" in text and "Mixer" in text


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
