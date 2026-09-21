"""Pause-aware match clock shared by the UI thread and the computer runner."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class ActiveClock:
    """Seconds of active (unpaused) match time. Starts stopped; `now()` is frozen while stopped."""

    def __init__(self, time_fn: Callable[[], float] = time.monotonic):
        self._time = time_fn
        self._lock = threading.Lock()
        self._elapsed = 0.0
        self._started_at: float | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._started_at is not None

    def now(self) -> float:
        with self._lock:
            if self._started_at is None:
                return self._elapsed
            return self._elapsed + (self._time() - self._started_at)

    def start(self) -> None:
        with self._lock:
            if self._started_at is None:
                self._started_at = self._time()

    resume = start

    def pause(self) -> None:
        with self._lock:
            if self._started_at is not None:
                self._elapsed += self._time() - self._started_at
                self._started_at = None

    def reset(self) -> None:
        """Zero and stop."""
        with self._lock:
            self._elapsed = 0.0
            self._started_at = None
