"""Board geometry → per-direction facts and the Laya state/question text.

Laya base checkpoints are weak zero-shot at spatial reasoning, so all geometry is computed here
and Laya chooses among described options. Wording was tuned in-game (english, 800 steps × 3 seeds):
- no heading in the state: "heading right" lexically pulls the model to option "right" (momentum bias)
- plain words beat numbers ("moves toward the food" vs "(6 -> 5)")
- goal-phrased instruction lifted toward-food picks from ~80% to ~99%
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .game import BoardView, Dir, Pos, collision, legal_dirs, shift

INSTRUCTIONS = "Which direction moves the snake toward the food without crashing?"
QUESTION_ID = "move"


@dataclass(frozen=True)
class Move:
    dir: Dir
    status: str  # "safe" | "fatal"
    reason: str | None  # "wall" | "body" when fatal
    dist_before: int | None  # Manhattan head→food; None when no food
    dist_after: int | None  # None when fatal or no food
    reachable: int  # free cells reachable from the new head; 0 when fatal
    trap: bool  # reachable < length after the move

    @property
    def safe(self) -> bool:
        return self.status == "safe"


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def flood_fill(width: int, height: int, blocked: set[Pos], start: Pos) -> int:
    """Count free cells reachable from `start` (excluded from the count)."""
    seen = {start}
    queue = deque([start])
    count = 0
    while queue:
        pos = queue.popleft()
        for d in Dir:
            x, y = shift(pos, d)
            nxt = (x, y)
            if 0 <= x < width and 0 <= y < height and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                count += 1
                queue.append(nxt)
    return count


def analyse(view: BoardView) -> dict[Dir, Move]:
    """Facts for each legal direction, in DIR_ORDER. Reverse is never included."""
    head, food = view.head, view.food
    before = manhattan(head, food) if food is not None else None
    moves: dict[Dir, Move] = {}
    for d in legal_dirs(view.body, view.heading):
        reason = collision(view.width, view.height, view.body, food, d)
        if reason is not None:
            moves[d] = Move(d, "fatal", reason, before, None, 0, False)
            continue
        nxt = shift(head, d)
        ate = nxt == food
        new_body = (nxt,) + (view.body if ate else view.body[:-1])
        reachable = flood_fill(view.width, view.height, set(new_body), nxt)
        after = manhattan(nxt, food) if food is not None else None
        moves[d] = Move(d, "safe", None, before, after, reachable, reachable < len(new_body))
    return moves


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def food_text(head: Pos, food: Pos | None) -> str:
    if food is None:
        return "none (board full)"
    dx, dy = food[0] - head[0], food[1] - head[1]
    parts = []
    if dx:
        parts.append(f"{plural(abs(dx), 'cell')} {'right' if dx > 0 else 'left'}")
    if dy:
        parts.append(f"{plural(abs(dy), 'cell')} {'down' if dy > 0 else 'up'}")
    return ", ".join(parts)


def criterion(move: Move) -> str:
    if not move.safe:
        return "crashes into the wall" if move.reason == "wall" else "crashes into its own body"
    if move.dist_before is None or move.dist_after is None:
        text = "moves safely"
    elif move.dist_after == 0:
        text = "eats the food"
    elif move.dist_after < move.dist_before:
        text = "moves toward the food"
    else:
        text = "moves away from the food"
    return text + ("; leads into a dead end" if move.trap else "")


def build_state(view: BoardView) -> dict[str, str]:
    return {
        "game": "snake: eat the food, never crash",
        "food": food_text(view.head, view.food),
    }


def build_question(moves: dict[Dir, Move]) -> dict[str, dict]:
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {d.label: criterion(m) for d, m in moves.items()},
        }
    }
