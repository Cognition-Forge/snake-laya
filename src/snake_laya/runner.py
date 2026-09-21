"""Computer side: owns the computer board, schedules steps, applies generation-matched predictions.

Only this thread mutates the computer board; only the inference worker touches the model. The UI
reads immutable `RunnerView` snapshots.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .brain import Decision, apply_safety, heuristic_pick, late_fallback
from .clock import ActiveClock
from .config import RESPAWN_S, GameConfig, Mode
from .decision_log import NullLog
from .features import Move, analyse, build_question, build_state
from .game import Board, BoardView, Dir, StepResult
from .inference import Request, Result
from .stats import DecisionStats

POLL_S = 0.05  # max latency to notice stop/pause/reset


class Worker(Protocol):
    results: queue.Queue[Result]

    def submit(self, request: Request) -> None: ...

    def clear(self) -> None: ...


@dataclass(frozen=True)
class RunnerView:
    board: BoardView
    mode: Mode
    gen: int
    probs: dict[Dir, float] | None  # last applied decision; None on late steps
    executed: Dir | None
    last_late: bool
    error: str | None


class ComputerRunner:
    def __init__(
        self,
        cfg: GameConfig,
        *,
        worker: Worker,
        clock: ActiveClock,
        stats: DecisionStats,
        log: Any = None,
        meta: dict[str, Any] | None = None,
    ):
        self._cfg = cfg
        self._worker = worker
        self._clock = clock
        self._stats = stats
        self._log = log or NullLog()
        self._meta = meta or {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending_reset: tuple[int, Mode] | None = None
        self._thread = threading.Thread(target=self._run, name="computer-runner", daemon=True)
        self._gen = 0
        self._error: str | None = None
        self._setup(cfg.seed, cfg.mode)

    # ---- control (any thread) -------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> bool:
        if self._thread.is_alive():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    def request_reset(self, seed: int, mode: Mode) -> None:
        """Applied by the runner thread within POLL_S."""
        with self._lock:
            self._pending_reset = (seed, mode)

    def view(self) -> RunnerView:
        with self._lock:
            return self._view

    # ---- runner thread ----------------------------------------------------------------------

    def _setup(self, seed: int, mode: Mode) -> None:
        self._board = Board(self._cfg.width, self._cfg.height, seed)
        self._seed = seed
        self._mode = mode
        self._match_id = uuid.uuid4().hex[:12]
        self._held: Decision | None = None
        self._deadline: float | None = None
        self._respawn_at: float | None = None
        self._probs: dict[Dir, float] | None = None
        self._executed: Dir | None = None
        self._last_late = False
        self._stats.reset()
        self._worker.clear()
        self._advance()
        # Results for generations below this belong to a previous match: ignored, not counted.
        self._gen_floor = self._gen

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.iterate()
        except Exception as exc:  # never die silently: surface in the UI
            self._fail(exc)

    def iterate(self) -> None:
        """One scheduling step. Public for deterministic tests."""
        with self._lock:
            reset, self._pending_reset = self._pending_reset, None
        if reset is not None:
            self._setup(*reset)
            return
        if self._error is not None or not self._clock.running or self._board.cleared:
            self._stop.wait(POLL_S)
            return
        if not self._board.alive:
            self._try_respawn()
            return
        if self._await_input():
            self._step()

    def _interrupted(self) -> bool:
        return (
            self._stop.is_set() or self._error is not None or self._pending_reset is not None or not self._clock.running
        )

    def _await_input(self) -> bool:
        """True → step now. SYNC: wait until the tick deadline, collecting results; MAX: until a current result."""
        if self._mode is Mode.SYNC and self._deadline is None:
            self._deadline = self._clock.now() + self._cfg.tick_s
        while not self._interrupted():
            if self._mode is Mode.MAX:
                if self._held is not None:
                    return True
                self._poll(POLL_S)
                continue
            remaining = self._deadline - self._clock.now()
            if remaining <= 0:
                # A current result may sit behind stale ones: drain before declaring LATE.
                self._drain()
                return self._error is None
            self._poll(min(remaining, POLL_S))
        return False

    def _poll(self, timeout: float) -> bool:
        try:
            result = self._worker.results.get(timeout=timeout) if timeout > 0 else self._worker.results.get_nowait()
        except queue.Empty:
            return False
        self._accept(result)
        return True

    def _drain(self) -> None:
        while self._poll(0):
            pass

    def _accept(self, result: Result) -> None:
        if result.gen < self._gen_floor:
            return
        if result.error is not None:
            self._fail(result.error)
            return
        self._stats.record_prediction(result.decision.latency_ms)
        if result.gen == self._gen:
            self._held = result.decision
        else:
            self._stats.record_stale()

    def _step(self) -> None:
        board, moves, view = self._board, self._moves, self._req_view
        decision, self._held = self._held, None
        if decision is not None:
            pick, overridden = apply_safety(decision.raw, moves, decision.probs, self._cfg.safety)
            baseline = heuristic_pick(moves, board.heading)
            agree = decision.raw is baseline
        else:
            pick, overridden = late_fallback(moves, board.heading, self._cfg.safety)
            baseline, agree = None, None
        score_before = board.score
        result = board.step(pick)
        now = self._clock.now()

        self._stats.record_step(
            applied=decision is not None,
            late=decision is None,
            override=overridden,
            agree=agree,
            sharpness=decision.sharpness if decision else None,
        )
        self._probs = decision.probs if decision else None
        self._executed = pick
        self._last_late = decision is None
        if self._log.enabled:
            self._log.write(self._record(view, moves, decision, pick, overridden, baseline, result, score_before, now))

        if self._mode is Mode.SYNC and self._deadline is not None:
            self._deadline += self._cfg.tick_s
            if self._deadline <= now:  # fell behind (e.g. OS stall): resync instead of bursting
                self._deadline = now + self._cfg.tick_s
        if result.died:
            self._respawn_at = now + RESPAWN_S
            self._deadline = None
        self._advance()

    def _try_respawn(self) -> None:
        remaining = self._respawn_at - self._clock.now()
        if remaining > 0:
            self._stop.wait(min(remaining, POLL_S))
            return
        self._board.respawn()
        self._respawn_at = None
        self._advance()

    def _advance(self) -> None:
        """New board generation; request a prediction for it when the board is playable."""
        self._gen += 1
        self._held = None
        if self._board.active:
            self._req_view = self._board.view()
            self._moves = analyse(self._req_view)
            self._worker.submit(Request(self._gen, self._req_view, self._moves))
        self._publish()

    def _fail(self, exc: BaseException) -> None:
        self._error = f"{type(exc).__name__}: {exc}"
        self._publish()

    def _publish(self) -> None:
        view = RunnerView(
            board=self._board.view(),
            mode=self._mode,
            gen=self._gen,
            probs=self._probs,
            executed=self._executed,
            last_late=self._last_late,
            error=self._error,
        )
        with self._lock:
            self._view = view

    def _record(
        self,
        view: BoardView,
        moves: dict[Dir, Move],
        decision: Decision | None,
        pick: Dir,
        overridden: bool,
        baseline: Dir | None,
        result: StepResult,
        score_before: int,
        now: float,
    ) -> dict[str, Any]:
        return {
            "match_id": self._match_id,
            "seed": self._seed,
            "mode": str(self._mode),
            **self._meta,
            "gen": self._gen,
            "t_active": round(now, 4),
            "state": build_state(view),
            "criteria": build_question(moves)["move"]["criteria"],
            "moves": {
                d.label: {
                    "status": m.status,
                    "reason": m.reason,
                    "dist_after": m.dist_after,
                    "reachable": m.reachable,
                    "trap": m.trap,
                }
                for d, m in moves.items()
            },
            "probabilities": {d.label: p for d, p in decision.probs.items()} if decision else None,
            "raw": decision.raw.label if decision else None,
            "pick": pick.label,
            "override": overridden,
            "late": decision is None,
            "baseline_pick": baseline.label if baseline else None,
            "latency_ms": round(decision.latency_ms, 3) if decision else None,
            "ate": result.ate,
            "died": result.died,
            "score_delta": self._board.score - score_before,
            "length_after": len(self._board.body),
        }
