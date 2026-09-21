"""Test doubles and board builders shared across test modules."""

from __future__ import annotations

import queue
from collections import deque
from collections.abc import Callable, Sequence

from snake_laya.brain import Decision
from snake_laya.game import Board, BoardView, Dir, Pos
from snake_laya.inference import Request, Result


class FakeTime:
    """Manual wall clock."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_view(
    body: Sequence[Pos],
    heading: Dir = Dir.RIGHT,
    food: Pos | None = None,
    w: int = 12,
    h: int = 8,
    alive: bool = True,
) -> BoardView:
    return BoardView(w, h, tuple(body), heading, food, 0, 0, 0, 0, 0, alive, False)


def make_board(body: Sequence[Pos], heading: Dir, food: Pos | None, w: int = 12, h: int = 8, seed: int = 0) -> Board:
    board = Board(w, h, seed)
    board.body = deque(body)
    board.heading = heading
    board.food = food
    return board


def serpentine(w: int, h: int) -> list[Pos]:
    """Every cell, adjacent consecutive cells (boustrophedon)."""
    cells: list[Pos] = []
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        cells.extend((x, y) for x in xs)
    return cells


def straight_decision(request: Request) -> Decision:
    """Policy that always continues straight (probability 1 on heading)."""
    heading = request.view.heading
    probs = {d: 1.0 if d is heading else 0.0 for d in request.moves}
    return Decision(probs, heading, 0.9, 5.0)


class FakeWorker:
    """Deterministic serial inference on FakeTime; also serves as the results queue.

    Mirrors InferenceWorker: one predict in flight, newer submits replace the unstarted request.
    """

    def __init__(
        self,
        time: FakeTime,
        latency: float | Callable[[Request], float] = 0.03,
        decide: Callable[[Request], Decision] = straight_decision,
    ):
        self.time = time
        self.latency = latency
        self.decide = decide
        self.submitted: list[Request] = []
        self.cleared = 0
        self._inflight: tuple[Request, float] | None = None
        self._pending: Request | None = None
        self._ready: deque[tuple[float, Result]] = deque()
        self.results = self

    def _lat(self, request: Request) -> float:
        return self.latency(request) if callable(self.latency) else self.latency

    def submit(self, request: Request) -> None:
        self.submitted.append(request)
        if self._inflight is None:
            self._inflight = (request, self.time.t + self._lat(request))
        else:
            self._pending = request

    def inject(self, arrival: float, result: Result) -> None:
        self._ready.append((arrival, result))
        self._ready = deque(sorted(self._ready, key=lambda item: item[0]))

    def _run_until(self, t: float) -> None:
        while self._inflight is not None and self._inflight[1] <= t:
            request, done = self._inflight
            self.inject(done, Result(request.gen, self.decide(request)))
            self._inflight = None
            if self._pending is not None:
                nxt, self._pending = self._pending, None
                self._inflight = (nxt, done + self._lat(nxt))

    def get(self, timeout: float) -> Result:
        target = self.time.t + timeout
        self._run_until(target)
        if self._ready and self._ready[0][0] <= target:
            arrival, result = self._ready.popleft()
            self.time.t = max(self.time.t, arrival)
            return result
        self.time.t = target
        raise queue.Empty

    def get_nowait(self) -> Result:
        self._run_until(self.time.t)
        if self._ready and self._ready[0][0] <= self.time.t:
            return self._ready.popleft()[1]
        raise queue.Empty

    def clear(self) -> None:
        self.cleared += 1
        self._pending = None
        self._ready.clear()


class ListLog:
    enabled = True

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.closed = False

    def write(self, record: dict) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.closed = True
