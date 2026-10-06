"""Per-host spacing of fetches, so a batch of runs does not trip a site's rate limit."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from urllib.parse import urlparse


class HostPacer:
    """Holds each fetch until gap_secs have passed since the last fetch to the same host.

    One pacer may be shared by several threads; each caller reserves its slot before it sleeps.
    """

    def __init__(
        self,
        gap_secs: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._gap = gap_secs
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        """Reserves the next free slot for url's host, then sleeps until it comes."""
        host = urlparse(url).hostname or ""
        with self._lock:
            now = self._clock()
            last = self._last.get(host)
            slot = now if self._gap <= 0 or last is None else max(now, last + self._gap)
            self._last[host] = slot
        if slot > now:
            self._sleep(slot - now)
