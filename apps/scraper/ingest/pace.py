"""Per-host spacing of fetches, so a batch of runs does not trip a site's rate limit."""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlparse


class HostPacer:
    """Holds each fetch until gap_secs have passed since the last fetch to the same host."""

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

    def wait(self, url: str) -> None:
        """Sleeps until url's host may be fetched again, then records the fetch."""
        host = urlparse(url).hostname or ""
        last = self._last.get(host)
        if self._gap > 0 and last is not None:
            remaining = self._gap - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last[host] = self._clock()
