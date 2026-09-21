"""Match lifecycle and the human side. Pure logic, no pygame: driven once per frame by `main`."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from enum import Enum, auto

from .clock import ActiveClock
from .config import COUNTDOWN_S, RESPAWN_S, GameConfig, Mode
from .game import Board, BoardView, Dir, StepResult
from .stats import HumanStats, rank

MAX_QUEUED_TURNS = 2


class InputQueue:
    """Buffered human turns: at most MAX_QUEUED_TURNS, one consumed per tick.

    Each key is validated against the last *queued* direction (else the heading), so a fast
    up→left within one tick works while up→down (reverse via the queue) is rejected.
    """

    def __init__(self) -> None:
        self._turns: deque[Dir] = deque()

    def push(self, d: Dir, heading: Dir) -> bool:
        last = self._turns[-1] if self._turns else heading
        if d is last or d is last.opposite or len(self._turns) >= MAX_QUEUED_TURNS:
            return False
        self._turns.append(d)
        return True

    def pop(self) -> Dir | None:
        return self._turns.popleft() if self._turns else None

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def pending(self) -> tuple[Dir, ...]:
        return tuple(self._turns)


class HumanController:
    """Steps the human board on the active clock; 1 s active-time respawn after death."""

    def __init__(self, board: Board, tick_s: float, clock_now: Callable[[], float]):
        self.board = board
        self.tick_s = tick_s
        self.queue = InputQueue()
        self.stats = HumanStats(clock_now)
        self._next_tick: float | None = None
        self.respawn_at: float | None = None

    def key(self, d: Dir) -> bool:
        if not self.board.active:
            return False
        return self.queue.push(d, self.board.heading)

    def tick(self, now: float) -> StepResult | None:
        board = self.board
        if board.cleared:
            return None
        if not board.alive:
            if self.respawn_at is not None and now >= self.respawn_at:
                board.respawn()
                self.queue.clear()
                self.respawn_at = None
                self._next_tick = now + self.tick_s
            return None
        if self._next_tick is None:
            self._next_tick = now + self.tick_s
            return None
        if now < self._next_tick:
            return None
        self._next_tick += self.tick_s
        if self._next_tick <= now:  # long frame stall: resync, never burst several steps
            self._next_tick = now + self.tick_s
        d = self.queue.pop() or board.heading
        turned = d is not board.heading
        result = board.step(d)
        self.stats.record_step(turned)
        if result.died:
            self.respawn_at = now + RESPAWN_S
            self.queue.clear()
        return result


class Phase(Enum):
    LOADING = auto()
    ERROR = auto()
    COUNTDOWN = auto()
    RUNNING = auto()
    PAUSED = auto()
    CONFIRM_MODE = auto()
    OVER = auto()


class Match:
    """Phase machine: LOADING → COUNTDOWN → RUNNING ⇄ PAUSED → OVER; CONFIRM_MODE overlays any live phase.

    `on_reset(seed, mode)` lets the computer side reset in lockstep with the human side.
    """

    def __init__(
        self,
        cfg: GameConfig,
        clock: ActiveClock,
        *,
        on_reset: Callable[[int, Mode], None] = lambda seed, mode: None,
        wall: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.clock = clock
        self.mode = cfg.mode
        self.on_reset = on_reset
        self._wall = wall
        self.phase = Phase.LOADING
        self.error: str | None = None
        self._confirm_return: Phase | None = None
        self._countdown_end = 0.0
        self.human = self._new_human()

    def _new_human(self) -> HumanController:
        board = Board(self.cfg.width, self.cfg.height, self.cfg.seed)
        return HumanController(board, self.cfg.tick_s, self.clock.now)

    # ---- transitions ----------------------------------------------------------------------

    def ready(self) -> None:
        """Model loaded and warmed up."""
        if self.phase is Phase.LOADING:
            self.reset()

    def fail(self, message: str) -> None:
        self.error = message
        self.clock.pause()
        self.phase = Phase.ERROR

    def reset(self) -> None:
        """Full restart in the current mode: boards, RNGs (same seed), clock, stats, input."""
        if self.phase is Phase.ERROR:
            return
        self.clock.reset()
        self.human = self._new_human()
        self._confirm_return = None
        self._start_countdown()
        self.on_reset(self.cfg.seed, self.mode)

    def restart(self) -> None:
        """User `R`: ignored while loading or after a fatal error."""
        if self.phase not in (Phase.LOADING, Phase.ERROR):
            self.reset()

    def toggle_pause(self) -> None:
        if self.phase is Phase.RUNNING:
            self.clock.pause()
            self.phase = Phase.PAUSED
        elif self.phase is Phase.PAUSED:
            self.clock.resume()
            self.phase = Phase.RUNNING

    def request_mode_switch(self) -> None:
        if self.phase in (Phase.COUNTDOWN, Phase.RUNNING, Phase.PAUSED, Phase.OVER):
            self._confirm_return = self.phase
            self.clock.pause()
            self.phase = Phase.CONFIRM_MODE

    def confirm_mode_switch(self) -> None:
        if self.phase is Phase.CONFIRM_MODE:
            self.mode = self.mode.other
            self.reset()

    def cancel_mode_switch(self) -> None:
        if self.phase is not Phase.CONFIRM_MODE:
            return
        self.phase, self._confirm_return = self._confirm_return, None
        if self.phase is Phase.RUNNING:
            self.clock.resume()
        elif self.phase is Phase.COUNTDOWN:
            self._start_countdown()

    def _start_countdown(self) -> None:
        self._countdown_end = self._wall() + COUNTDOWN_S
        self.phase = Phase.COUNTDOWN

    # ---- per frame -----------------------------------------------------------------------

    def key(self, d: Dir) -> None:
        if self.phase is Phase.RUNNING:
            self.human.key(d)

    def update(self) -> None:
        if self.phase is Phase.COUNTDOWN and self._wall() >= self._countdown_end:
            self.clock.start()
            self.phase = Phase.RUNNING
        if self.phase is not Phase.RUNNING:
            return
        now = self.clock.now()
        if now >= self.cfg.duration_s:
            self.clock.pause()
            self.phase = Phase.OVER
            return
        self.human.tick(now)

    # ---- queries -------------------------------------------------------------------------

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.cfg.duration_s - self.clock.now())

    @property
    def countdown_left(self) -> int:
        return max(0, math.ceil(self._countdown_end - self._wall()))

    @property
    def ranked(self) -> bool:
        return self.mode is Mode.SYNC

    def standing(self, computer: BoardView) -> str | None:
        """Current leader / final winner: 'human' | 'computer' | 'draw' | None (unranked)."""
        return rank(self.human.board.view(), computer, self.mode)
