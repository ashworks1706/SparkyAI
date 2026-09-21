"""The admin authenticated driver: one operator-captured browser session for login-gated sources.

An operator runs scraper login once, signs in and completes any MFA in a real browser, and the
browser storage state (cookies and local storage) is saved to disk. Login-gated sources then fetch
through a headless browser loaded with that state. No password is asked for or stored. This session
is admin scoped and shared by the scraper across every guild; it is not the per-user MyASU session.
Authenticated content is never written to the retrieval index.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from scraper.core.settings import settings
from scraper.core.types import AuthError, Fetched, FetchError, FetchRejected


def state_path() -> Path:
    """Where the captured storage state is read from and written to."""
    return Path(settings().auth.storage_state_path)


def _login_hosts() -> tuple[str, ...]:
    """Hosts that mean the browser landed on a sign-in page rather than the content."""
    raw = settings().auth.login_hosts
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip())


def is_login_url(url: str) -> bool:
    """Whether url is on a sign-in host, so a fetch that ended there has no live session."""
    host = (urlsplit(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in _login_hosts())


def capture_login() -> Path:
    """Opens a real browser for the operator to sign in, then saves the session. Returns its path.

    The operator completes login and any MFA themselves. Only cookies and local storage are saved.
    """
    from playwright.sync_api import sync_playwright

    cfg = settings().auth
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        try:
            context = browser.new_context(user_agent=settings().scraper.user_agent)
            page = context.new_page()
            page.goto(cfg.login_url, wait_until="domcontentloaded")
            print(
                "A browser window opened. Sign in to ASU and complete any MFA.\n"
                "When you are fully signed in, return here and press Enter to save the session."
            )
            input()
            context.storage_state(path=str(path))
        finally:
            browser.close()
    return path


def fetch_authenticated(url: str) -> Fetched:
    """Loads url in a headless browser carrying the admin session and returns the rendered DOM.

    Raises AuthError when the session is missing or has expired, so the operator knows to re-run
    scraper login. Never falls back to an unauthenticated fetch.
    """
    from playwright.sync_api import sync_playwright

    cfg = settings().auth
    path = state_path()
    if not path.exists():
        raise AuthError(
            f"no admin session at {path}; an operator must run `just scraper login` to sign in"
        )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                storage_state=str(path), user_agent=settings().scraper.user_agent
            )
            page = context.new_page()
            response = page.goto(
                url, wait_until="networkidle", timeout=int(cfg.nav_timeout_secs * 1000)
            )
            status = response.status if response else 0
            final_url = page.url
            html = page.content()
        finally:
            browser.close()
    if is_login_url(final_url):
        raise AuthError(
            "the admin session has expired; an operator must run `just scraper login` to sign in"
        )
    if status == 0:
        raise FetchError(f"{url}: navigation returned no response")
    if 400 <= status < 500:
        raise FetchRejected(f"{url}: {status}")
    if status >= 500:
        raise FetchError(f"{url}: {status}")
    return Fetched(
        url=final_url, status=status, body=html.encode("utf-8"), content_type="text/html"
    )
