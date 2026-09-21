"""Decision and player statistics plus winner ranking. Thread-safe: runner writes, UI reads views."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .config import Mode
from .game import BoardView

LATENCY_WINDOW = 200
RATE_WINDOW_S = 1.0


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(q / 100.0 * len(ordered)) - 1))
    return ordered[k]


def pct(part: int, whole: int) -> float | None:
    return 100.0 * part / whole if whole else None


class RateWindow:
    """Events within the last `window_s` (exclusive of events exactly `window_s` old)."""

    def __init__(self, window_s: float = RATE_WINDOW_S):
        self.window_s = window_s
        self._times: deque[float] = deque()

    def add(self, t: float) -> None:
        self._times.append(t)

    def rate(self, now: float) -> float:
        while self._times and self._times[0] <= now - self.window_s:
            self._times.popleft()
        return len(self._times) / self.window_s

    def clear(self) -> None:
        self._times.clear()


@dataclass(frozen=True)
class StatsView:
    last_ms: float | None
    p50: float | None
    p95: float | None
    predictions: int  # completed predictions incl. stale
    decisions: int  # predictions applied to a step
    dps: float  # applied decisions/s over the last second of active time
    mean_dps: float | None  # applied decisions / active seconds
    steps: int
    stale: int
    late: int
    late_pct: float | None
    overrides: int
    override_pct: float | None
    agree_pct: float | None
    sharpness: float | None  # last applied decision
    mean_sharpness: float | None


class DecisionStats:
    def __init__(self, now: Callable[[], float], window: int = LATENCY_WINDOW):
        self._now = now
        self._lock = threading.Lock()
        self._latency: deque[float] = deque(maxlen=window)
        self._rate = RateWindow()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._latency.clear()
            self._rate.clear()
            self._last_ms: float | None = None
            self._predictions = 0
            self._decisions = 0
            self._steps = 0
            self._stale = 0
            self._late = 0
            self._overrides = 0
            self._agree = 0
            self._agree_n = 0
            self._sharpness: float | None = None
            self._sharp_sum = 0.0
            self._sharp_n = 0

    def record_prediction(self, latency_ms: float) -> None:
        with self._lock:
            self._latency.append(latency_ms)
            self._last_ms = latency_ms
            self._predictions += 1

    def record_stale(self) -> None:
        with self._lock:
            self._stale += 1

    def record_step(
        self,
        *,
        applied: bool,
        late: bool,
        override: bool,
        agree: bool | None = None,
        sharpness: float | None = None,
    ) -> None:
        with self._lock:
            self._steps += 1
            self._late += late
            self._overrides += override
            if applied:
                self._decisions += 1
                self._rate.add(self._now())
            if agree is not None:
                self._agree_n += 1
                self._agree += agree
            if sharpness is not None:
                self._sharpness = sharpness
                self._sharp_sum += sharpness
                self._sharp_n += 1

    def view(self) -> StatsView:
        now = self._now()
        with self._lock:
            lat = list(self._latency)
            return StatsView(
                last_ms=self._last_ms,
                p50=percentile(lat, 50),
                p95=percentile(lat, 95),
                predictions=self._predictions,
                decisions=self._decisions,
                dps=self._rate.rate(now),
                mean_dps=self._decisions / now if now > 0 else None,
                steps=self._steps,
                stale=self._stale,
                late=self._late,
                late_pct=pct(self._late, self._steps),
                overrides=self._overrides,
                override_pct=pct(self._overrides, self._steps),
                agree_pct=pct(self._agree, self._agree_n),
                sharpness=self._sharpness,
                mean_sharpness=self._sharp_sum / self._sharp_n if self._sharp_n else None,
            )


@dataclass(frozen=True)
class HumanStatsView:
    moves: int
    turns: int
    turns_per_s: float


class HumanStats:
    """Main-thread only."""

    def __init__(self, now: Callable[[], float]):
        self._now = now
        self._rate = RateWindow()
        self.moves = 0
        self.turns = 0

    def record_step(self, turned: bool) -> None:
        self.moves += 1
        if turned:
            self.turns += 1
            self._rate.add(self._now())

    def view(self) -> HumanStatsView:
        return HumanStatsView(self.moves, self.turns, self._rate.rate(self._now()))


def rank(human: BoardView, computer: BoardView, mode: Mode) -> str | None:
    """'human' / 'computer' / 'draw'; None in unranked MAX mode.

    Order: total food ↓, best life ↓, deaths ↑.
    """
    if mode is Mode.MAX:
        return None
    h = (human.total_food, human.best, -human.deaths)
    c = (computer.total_food, computer.best, -computer.deaths)
    if h == c:
        return "draw"
    return "human" if h > c else "computer"
