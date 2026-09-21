"""Single inference thread: one predict at a time, latest-request-only mailbox."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from .brain import Brain, Decision
from .features import Move
from .game import BoardView, Dir


@dataclass(frozen=True)
class Request:
    gen: int  # board generation the prediction is for
    view: BoardView
    moves: dict[Dir, Move]


@dataclass(frozen=True)
class Result:
    gen: int
    decision: Decision | None
    error: BaseException | None = None


class InferenceWorker:
    def __init__(self, brain: Brain):
        self.brain = brain
        self.results: queue.Queue[Result] = queue.Queue()
        self._cond = threading.Condition()
        self._pending: Request | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="laya-inference", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, request: Request) -> None:
        """Queue `request`, replacing any request not yet started. An in-flight predict is never cancelled."""
        with self._cond:
            self._pending = request
            self._cond.notify()

    def clear(self) -> None:
        """Drop the unstarted request and undelivered results (consumer thread only)."""
        with self._cond:
            self._pending = None
        while True:
            try:
                self.results.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify()

    def join(self, timeout: float | None = None) -> bool:
        """True if the thread finished (or never started)."""
        if self._thread.is_alive():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                request, self._pending = self._pending, None
            try:
                decision = self.brain.decide(request.view, request.moves)
                self.results.put(Result(request.gen, decision))
            except Exception as exc:  # surfaced to the runner → error overlay
                self.results.put(Result(request.gen, None, exc))
