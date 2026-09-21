"""The ASU single sign-on flow on a browser page: the MyASU CAS form, then Duo. Stores nothing."""

from __future__ import annotations

import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from scraper.core.types import AuthError

_USERNAME = "#username"
_PASSWORD = "#password"
_SUBMIT = "button[name=submitBtn]"
_CAS_ERROR = ".alert-danger, .banner-danger, #loginErrorsPanel, .login-error"
_DUO_HOST = "duosecurity.com"
_DUO_CODE = ".verification-code, [class*=verification-code], [data-testid*=code]"
_CODE_DIGITS = re.compile(r"(?<!\d)\d{3,8}(?!\d)")
_DUO_TRUST = "#trust-browser-button"
_DUO_TRUST_LABEL = re.compile(r"^\s*yes\b", re.IGNORECASE)
_DUO_HEADING = "h1, h2"
_POLL_SECS = 0.5


@dataclass(frozen=True)
class Credentials:
    """One sign-in attempt. Never logged, never written anywhere."""

    username: str
    password: str

    def __repr__(self) -> str:
        return f"Credentials(username={self.username!r}, password=***)"


class BadCredentials(AuthError):
    """CAS refused the username or password."""


def host_of(url: str) -> str:
    """The lowercase host of url."""
    return (urlsplit(url).hostname or "").lower()


def on_host(url: str, hosts: tuple[str, ...]) -> bool:
    """Whether url is on one of hosts or a subdomain of one."""
    host = host_of(url)
    return any(host == h or host.endswith("." + h) for h in hosts)


def on_cas_form(page: Page) -> bool:
    """Whether the page shows the CAS username and password form."""
    if not on_host(page.url, ("weblogin.asu.edu",)):
        return False
    try:
        field = page.locator(_PASSWORD).first
        return field.count() > 0 and field.is_visible()
    except PlaywrightError:
        return False


def sign_in(
    page: Page,
    creds: Credentials,
    sso_hosts: tuple[str, ...],
    timeout_secs: float,
    notify: Callable[[str], None],
) -> None:
    """Fills the CAS form on page, submits it, and waits out Duo until the page leaves sso_hosts.

    Raises BadCredentials when CAS shows its form again, and AuthError when Duo is not approved
    within timeout_secs.
    """
    page.fill(_USERNAME, creds.username)
    page.fill(_PASSWORD, creds.password)
    page.click(_SUBMIT)
    settle(page)
    if on_cas_form(page):
        message = _text(page, _CAS_ERROR) or "the username or password was not accepted"
        raise BadCredentials(message)
    notify("Password accepted. Waiting for Duo: approve the push on your device.")
    _await_duo(page, sso_hosts, timeout_secs, notify)


def _await_duo(
    page: Page, sso_hosts: tuple[str, ...], timeout_secs: float, notify: Callable[[str], None]
) -> None:
    """Polls until the page leaves the sign-in hosts, relaying Duo's prompts as they change."""
    deadline = time.monotonic() + timeout_secs
    said: set[str] = set()
    trusted = False
    while on_host(page.url, sso_hosts):
        if time.monotonic() > deadline:
            raise AuthError(f"Duo was not approved within {int(timeout_secs)} seconds")
        if on_host(page.url, (_DUO_HOST,)):
            code = _duo_code(page)
            if code and code not in said:
                said.add(code)
                notify(f"Duo verification code: {code}. Enter it in the Duo app.")
            heading = _text(page, _DUO_HEADING)
            if heading and heading not in said:
                said.add(heading)
                notify(f"Duo: {heading}")
                if not code and "code" in heading.lower():
                    shot = Path(tempfile.gettempdir()) / "sparky-duo.png"
                    try:
                        page.screenshot(path=str(shot))
                        notify(f"Duo code not found in the page; see the screenshot at {shot}")
                    except PlaywrightError:
                        pass
            if not trusted and _trust(page):
                trusted = True
                notify("Duo approved. Trusting this browser.")
        page.wait_for_timeout(int(_POLL_SECS * 1000))
    settle(page)


def _duo_code(page: Page) -> str:
    """The Verified Duo Push code on the page: its own element, else the digits in the page text."""
    code = _CODE_DIGITS.search(_text(page, _DUO_CODE))
    if code:
        return code.group(0)
    body = _text(page, "body")
    if "code" not in body.lower():
        return ""
    code = _CODE_DIGITS.search(body)
    return code.group(0) if code else ""


def settle(page: Page) -> None:
    """Waits for the page to stop loading; a page that keeps polling is left as it is."""
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except PlaywrightTimeout:
        pass


def _text(page: Page, selector: str) -> str:
    """The trimmed text of the first visible match of selector, or empty mid-navigation."""
    try:
        for el in page.locator(selector).all():
            if el.is_visible():
                text = " ".join(el.inner_text(timeout=1_000).split())
                if text:
                    return text
    except PlaywrightError:
        return ""
    return ""


def _trust(page: Page) -> bool:
    """Answers yes to Duo asking whether this is the operator's device, by id or by label."""
    if _click(page, _DUO_TRUST):
        return True
    try:
        button = page.get_by_role("button", name=_DUO_TRUST_LABEL).first
        if button.count() == 0 or not button.is_visible():
            return False
        button.click(timeout=5_000)
    except PlaywrightError:
        return False
    return True


def _click(page: Page, selector: str) -> bool:
    """Clicks the first visible match of selector. False when there is none or the page moved."""
    try:
        el = page.locator(selector).first
        if el.count() == 0 or not el.is_visible():
            return False
        el.click(timeout=5_000)
    except PlaywrightError:
        return False
    return True
