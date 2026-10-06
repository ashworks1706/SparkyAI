"""Shared fixtures: no test reads a live robots.txt."""

from __future__ import annotations

import pytest
from scraper.ingest import robots


@pytest.fixture(autouse=True)
def robots_allow_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robots, "allowed", lambda _url: True)
