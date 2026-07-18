"""Fixed-window rate limiter with credit accounting (spec 2.4 step 4).

In-process and therefore per-worker: with N uvicorn workers the effective
limit is N x the configured value. That is fine for the MVP and for the
notebook demo, but a multi-worker deployment should back this with Redis.
The interface below is what a Redis implementation would need to satisfy.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class RateLimitDecision:
    allowed: bool
    consumed: int
    remaining: int
    reset_after: int


class FixedWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self._limit = limit
        self._window = window_seconds
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, int]] = {}

    def check(self, identity: str, cost: int = 1) -> RateLimitDecision:
        now = time.monotonic()
        with self._lock:
            started, used = self._state.get(identity, (now, 0))
            if now - started >= self._window:
                started, used = now, 0

            reset_after = max(0, int(self._window - (now - started)))

            if used + cost > self._limit:
                self._state[identity] = (started, used)
                return RateLimitDecision(
                    False, 0, max(0, self._limit - used), reset_after
                )

            used += cost
            self._state[identity] = (started, used)
            return RateLimitDecision(True, cost, self._limit - used, reset_after)
