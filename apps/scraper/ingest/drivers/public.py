"""The public browser driver: headless Chromium with no session, for JS pages anyone can read."""

from __future__ import annotations

from scraper.core.settings import settings
from scraper.core.types import Fetched
from scraper.ingest.drivers.page import browser_slot, load, skip_heavy, to_fetched


def fetch_rendered(url: str) -> Fetched:
    """Loads the page in a fresh headless Chromium with no cookies and returns the rendered DOM."""
    from playwright.sync_api import sync_playwright

    s = settings().scraper
    with browser_slot(), sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=s.user_agent)
            skip_heavy(context)
            loaded = load(context, url, s.request_timeout_secs)
        finally:
            browser.close()
    return to_fetched(url, loaded)
