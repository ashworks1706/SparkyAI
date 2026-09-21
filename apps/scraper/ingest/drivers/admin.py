"""The admin authenticated driver: one operator's ASU session for login-gated shared sources.

scraper login opens the MyASU sign-in page in a browser, asks the operator for a username and
password through a callback, submits them, and waits for the operator to approve Duo. It then signs
in to each configured service through ASU single sign-on and saves the browser storage state
(cookies and local storage) to disk. The password is used for that one form and never stored.

Login-gated sources then fetch through a headless browser loaded with that state. A service that
dropped its own session is re-entered through single sign-on while the ASU session lasts, and the
refreshed state is saved. This session is admin scoped and shared by the scraper across every
guild; it is not the per-user MyASU session. Authenticated content is never written to the index.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import structlog
from playwright.sync_api import BrowserContext, Page
from playwright.sync_api import Error as PlaywrightError

from scraper.core.settings import Auth, settings
from scraper.core.types import AuthError, Fetched, FetchError
from scraper.ingest.drivers import asu_sso
from scraper.ingest.drivers.page import Loaded, browser_slot, load, skip_heavy, to_fetched

log = structlog.get_logger()

#: Asks the operator for one sign-in attempt. None skips signing in.
Ask = Callable[[], asu_sso.Credentials | None]
#: Relays progress to the operator.
Notify = Callable[[str], None]


def state_path() -> Path:
    """Where the captured storage state is read from and written to."""
    return Path(settings().auth.storage_state_path)


def _split(raw: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def is_login_url(url: str) -> bool:
    """Whether url is a sign-in page, so a fetch that ended there has no live session."""
    cfg = settings().auth
    if asu_sso.on_host(url, _split(cfg.login_hosts)):
        return True
    path = urlsplit(url).path.lower().rstrip("/")
    return any(path == p.rstrip("/") for p in _split(cfg.login_paths))


def has_session() -> bool:
    """Whether a captured session exists on disk. It may still have expired."""
    return state_path().exists()


def check() -> bool:
    """Whether the saved session still reaches auth.check_url, through single sign-on if needed.

    Raises FetchError when the page cannot be reached, which says nothing about the session.
    """
    if not has_session():
        return False
    try:
        fetch_authenticated(settings().auth.check_url)
    except AuthError as e:
        log.info("admin session not usable", reason=str(e))
        return False
    return True


def capture_login(ask: Ask, notify: Notify) -> Path | None:
    """Signs in through MyASU and Duo, enters each service, and saves the session.

    Returns the saved path, or None when ask skips. Raises AuthError when sign-in fails.
    """
    from playwright.sync_api import sync_playwright

    cfg = settings().auth
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg.login_headless)
        try:
            context = browser.new_context(user_agent=settings().scraper.user_agent)
            page = context.new_page()
            if not _sign_in(page, cfg, ask, notify):
                return None
            notify("Signed in to MyASU.")
            _enter_service(page, cfg)
            loaded = load(context, cfg.check_url, cfg.nav_timeout_secs)
            if is_login_url(loaded.url):
                raise AuthError(f"signed in to MyASU but {cfg.check_url} still asks for a login")
            notify(f"Signed in to {asu_sso.host_of(cfg.check_url)}.")
            if not _save(context):
                raise AuthError(f"could not write the admin session to {state_path()}")
        finally:
            browser.close()
    return state_path()


def _sign_in(page: Page, cfg: Auth, ask: Ask, notify: Notify) -> bool:
    """Runs the MyASU form until it is accepted. False when the operator skips."""
    for attempt in range(1, cfg.login_attempts + 1):
        page.goto(cfg.login_url, wait_until="networkidle", timeout=_ms(cfg.nav_timeout_secs))
        if not asu_sso.on_cas_form(page):
            raise AuthError(f"{cfg.login_url} did not show the ASU sign-in form; at {page.url}")
        creds = ask()
        if creds is None:
            return False
        try:
            asu_sso.sign_in(page, creds, _split(cfg.sso_hosts), cfg.duo_timeout_secs, notify)
        except asu_sso.BadCredentials as e:
            notify(f"Sign-in refused ({e}). Attempt {attempt} of {cfg.login_attempts}.")
            continue
        return True
    raise AuthError(f"sign-in refused {cfg.login_attempts} times")


def _enter_service(page: Page, cfg: Auth) -> None:
    """Opens the service sign-in page and follows its single sign-on link on the ASU session."""
    page.goto(cfg.service_login_url, wait_until="networkidle", timeout=_ms(cfg.nav_timeout_secs))
    if not is_login_url(page.url):
        return
    link = page.get_by_text(cfg.service_sso_text, exact=False).first
    if link.count() == 0:
        raise AuthError(f"{cfg.service_login_url} has no link named {cfg.service_sso_text}")
    link.click()
    asu_sso.settle(page)
    if asu_sso.on_cas_form(page):
        raise AuthError(
            "the ASU session has expired; an operator must run `just scraper login` to sign in"
        )
    try:
        page.wait_for_url(
            lambda u: not asu_sso.on_host(u, _split(cfg.sso_hosts)) and not is_login_url(u),
            timeout=_ms(cfg.nav_timeout_secs),
        )
    except PlaywrightError as e:
        raise AuthError(
            "the ASU session has expired; an operator must run `just scraper login` to sign in"
        ) from e


def fetch_authenticated(url: str) -> Fetched:
    """Loads url in a headless browser carrying the admin session and returns the rendered DOM.

    Raises AuthError when the session is missing or expired, so the operator knows to run scraper
    login again, and FetchError when the page cannot be reached after auth.fetch_attempts tries.
    Never falls back to an unauthenticated fetch.
    """
    cfg = settings().auth
    path = state_path()
    if not path.exists():
        raise AuthError(
            f"no admin session at {path}; an operator must run `just scraper login` to sign in"
        )
    for attempt in range(1, cfg.fetch_attempts + 1):
        try:
            return to_fetched(url, _load_signed_in(url, path, cfg))
        except PlaywrightError as e:
            log.warning("authenticated fetch failed", url=url, attempt=attempt, error=str(e))
            last = e
    raise FetchError(f"{url}: {str(last).splitlines()[0]}") from last


def _load_signed_in(url: str, path: Path, cfg: Auth) -> Loaded:
    """One headless load of url with the saved session, re-entering single sign-on if needed."""
    from playwright.sync_api import sync_playwright

    with browser_slot(), sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                storage_state=str(path), user_agent=settings().scraper.user_agent
            )
            skip_heavy(context)
            loaded = load(context, url, cfg.nav_timeout_secs)
            if is_login_url(loaded.url):
                loaded = _resume(context, url, cfg)
        finally:
            browser.close()
    return loaded


def _resume(context: BrowserContext, url: str, cfg: Auth) -> Loaded:
    """Re-enters the service through single sign-on, saves the refreshed session, reloads url."""
    page = context.new_page()
    try:
        _enter_service(page, cfg)
    finally:
        page.close()
    loaded = load(context, url, cfg.nav_timeout_secs)
    if is_login_url(loaded.url):
        raise AuthError(
            "the admin session has expired; an operator must run `just scraper login` to sign in"
        )
    _save(context)
    log.info("admin session refreshed through single sign-on")
    return loaded


def _save(context: BrowserContext) -> bool:
    """Writes the storage state beside its path and swaps it in, readable by the owner only."""
    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        context.storage_state(path=str(tmp))
        tmp.chmod(0o600)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("admin session not saved", path=str(path), error=str(e))
        return False
    return True


def _ms(secs: float) -> int:
    return int(secs * 1000)
