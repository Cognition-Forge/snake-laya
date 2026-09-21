"""Pure snake rules: board state, movement, collision, seeded food. No I/O, no threads."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

Pos = tuple[int, int]

MIN_W, MIN_H = 12, 8
START_LEN = 6


class Dir(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    @property
    def opposite(self) -> Dir:
        return Dir((-self.dx, -self.dy))

    @property
    def label(self) -> str:
        return self.name.lower()


# Canonical order for Laya options, UI bars and deterministic tie-breaks.
DIR_ORDER: tuple[Dir, ...] = (Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT)


def shift(pos: Pos, d: Dir) -> Pos:
    return (pos[0] + d.dx, pos[1] + d.dy)


def legal_dirs(body: Sequence[Pos], heading: Dir) -> tuple[Dir, ...]:
    """All directions except reversing into the neck (only possible once length >= 2)."""
    if len(body) < 2:
        return DIR_ORDER
    return tuple(d for d in DIR_ORDER if d is not heading.opposite)


def collision(width: int, height: int, body: Sequence[Pos], food: Pos | None, d: Dir) -> str | None:
    """'wall' / 'body' if moving `d` kills the snake, else None.

    Tail rule: the tail cell is free because it moves away this step, unless the snake eats
    (then the tail stays).
    """
    x, y = shift(body[0], d)
    if not (0 <= x < width and 0 <= y < height):
        return "wall"
    nxt = (x, y)
    if nxt in body and not (nxt == body[-1] and nxt != food):
        return "body"
    return None


@dataclass(frozen=True)
class StepResult:
    ate: bool
    died: bool


@dataclass(frozen=True)
class BoardView:
    """Immutable snapshot, safe to hand across threads."""

    width: int
    height: int
    body: tuple[Pos, ...]  # head first
    heading: Dir
    food: Pos | None
    score: int
    total_food: int
    best: int
    deaths: int
    steps: int
    alive: bool
    cleared: bool

    @property
    def head(self) -> Pos:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


class Board:
    """One snake on one board. `score` is the current life; `total_food`/`best`/`deaths` span the match."""

    def __init__(self, width: int, height: int, seed: int):
        if width < MIN_W or height < MIN_H:
            raise ValueError(f"grid {width}x{height} is below minimum {MIN_W}x{MIN_H}")
        self.width = width
        self.height = height
        self.seed = seed
        self._rng = random.Random(seed)
        self.total_food = 0
        self.best = 0
        self.deaths = 0
        self.steps = 0
        self.cleared = False
        self._place_snake()
        self.food = self._spawn_food()

    def _place_snake(self) -> None:
        y = self.height // 2
        head_x = max(START_LEN - 1, self.width // 3)
        self.body: deque[Pos] = deque((head_x - i, y) for i in range(START_LEN))
        self.heading = Dir.RIGHT
        self.score = 0
        self.alive = True

    def _spawn_food(self) -> Pos | None:
        # Row-major scan keeps spawns reproducible for a given seed and body.
        occupied = set(self.body)
        free = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in occupied]
        return self._rng.choice(free) if free else None

    @property
    def head(self) -> Pos:
        return self.body[0]

    @property
    def active(self) -> bool:
        return self.alive and not self.cleared

    def legal_dirs(self) -> tuple[Dir, ...]:
        return legal_dirs(self.body, self.heading)

    def collision(self, d: Dir) -> str | None:
        return collision(self.width, self.height, self.body, self.food, d)

    def step(self, d: Dir) -> StepResult:
        if not self.active:
            raise RuntimeError("board is not active (dead or cleared)")
        if d not in self.legal_dirs():
            raise ValueError(f"illegal move {d.label}: reverses into the neck")
        self.steps += 1
        self.heading = d
        if self.collision(d) is not None:
            self.alive = False
            self.deaths += 1
            return StepResult(ate=False, died=True)
        nxt = shift(self.head, d)
        ate = nxt == self.food
        self.body.appendleft(nxt)
        if not ate:
            self.body.pop()
            return StepResult(ate=False, died=False)
        self.score += 1
        self.total_food += 1
        self.best = max(self.best, self.score)
        self.food = self._spawn_food()
        self.cleared = self.food is None
        return StepResult(ate=True, died=False)

    def respawn(self) -> None:
        """New life: reset body and score; match totals and the RNG stream carry on."""
        self._place_snake()
        if self.food is None or self.food in self.body:
            self.food = self._spawn_food()

    def view(self) -> BoardView:
        return BoardView(
            width=self.width,
            height=self.height,
            body=tuple(self.body),
            heading=self.heading,
            food=self.food,
            score=self.score,
            total_food=self.total_food,
            best=self.best,
            deaths=self.deaths,
            steps=self.steps,
            alive=self.alive,
            cleared=self.cleared,
        )
