"""Computer move selection: Laya policy, heuristic baseline, safety and late-tick fallback."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .features import QUESTION_ID, Move, analyse, build_question, build_state
from .game import DIR_ORDER, Board, BoardView, Dir

LAYA_REPO = "convaiinnovations/laya"
# --model name → bundle subfolder (None = repo root = English checkpoint)
LAYA_MODELS: dict[str, str | None] = {
    "english": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}
EXPERIMENTAL_MODELS = {"typed-decisions"}


@dataclass(frozen=True)
class Decision:
    probs: dict[Dir, float]  # offered (legal) directions only
    raw: Dir  # policy's own pick before safety
    sharpness: float | None  # Laya confidence = 1 - H(p)/log k; not a success probability
    latency_ms: float


class Brain(Protocol):
    label: str
    device: str

    def decide(self, view: BoardView, moves: dict[Dir, Move]) -> Decision: ...

    def warmup(self, n: int = 3) -> None: ...


def _order(d: Dir) -> int:
    return DIR_ORDER.index(d)


def argmax_dir(dirs: list[Dir] | tuple[Dir, ...], probs: dict[Dir, float]) -> Dir:
    """Highest probability; ties → earliest in DIR_ORDER."""
    return max(dirs, key=lambda d: (probs.get(d, 0.0), -_order(d)))


def parse_probs(raw: Any, offered: list[Dir] | tuple[Dir, ...]) -> dict[Dir, float]:
    """Laya `probabilities` → {Dir: p}; missing, non-numeric, NaN or negative values become 0."""
    raw = raw if isinstance(raw, dict) else {}
    out: dict[Dir, float] = {}
    for d in offered:
        try:
            p = float(raw.get(d.label, 0.0))
        except (TypeError, ValueError):
            p = 0.0
        out[d] = p if math.isfinite(p) and p > 0.0 else 0.0
    return out


def heuristic_pick(moves: dict[Dir, Move], heading: Dir) -> Dir:
    """Baseline: safe move ranked by (no trap, closer to food, more room); no safe move → heading.

    A static one-step heuristic, not an oracle: it can still walk into future traps.
    """
    safe = [d for d, m in moves.items() if m.safe]
    if not safe:
        return heading if heading in moves else next(iter(moves))

    def rank(d: Dir) -> tuple:
        m = moves[d]
        dist = m.dist_after if m.dist_after is not None else 0
        return (not m.trap, -dist, m.reachable, -_order(d))

    return max(safe, key=rank)


def apply_safety(raw: Dir, moves: dict[Dir, Move], probs: dict[Dir, float], enabled: bool) -> tuple[Dir, bool]:
    """(pick, overridden). Fatal raw pick → most probable safe move; no safe move → raw (death)."""
    if not enabled or moves[raw].safe:
        return raw, False
    safe = [d for d, m in moves.items() if m.safe]
    if not safe:
        return raw, False
    return argmax_dir(safe, probs), True


def late_fallback(moves: dict[Dir, Move], heading: Dir, safety: bool) -> tuple[Dir, bool]:
    """(pick, overridden) for a tick with no current prediction: continue straight unless fatal."""
    if not safety or moves[heading].safe:
        return heading, False
    safe = [d for d, m in moves.items() if m.safe]
    if not safe:
        return heading, False

    def rank(d: Dir) -> tuple:
        m = moves[d]
        dist = m.dist_after if m.dist_after is not None else 0
        return (m.reachable, -dist, -_order(d))

    return max(safe, key=rank), True


def _warmup_sample() -> tuple[BoardView, dict[Dir, Move]]:
    view = Board(30, 20, seed=0).view()
    return view, analyse(view)


class HeuristicBrain:
    label = "HEURISTIC"
    device = "cpu"

    def decide(self, view: BoardView, moves: dict[Dir, Move]) -> Decision:
        start = time.perf_counter()
        pick = heuristic_pick(moves, view.heading)
        probs = {d: 1.0 if d is pick else 0.0 for d in moves}
        return Decision(probs, pick, 1.0, (time.perf_counter() - start) * 1000)

    def warmup(self, n: int = 3) -> None:
        pass


class LayaBrain:
    """One Laya `choice` question per step over the legal directions."""

    def __init__(self, model: str = "english", device: str | None = None, loader: Callable[..., Any] | None = None):
        if model not in LAYA_MODELS:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(LAYA_MODELS)}")
        if loader is None:
            import laya  # deferred: pulls in torch/transformers

            loader = laya.load
        self.model = model
        self.agent = loader(LAYA_REPO, device=device, subfolder=LAYA_MODELS[model])
        self.device = str(getattr(self.agent, "device", device or "auto"))
        self.label = f"LAYA {model}"

    def decide(self, view: BoardView, moves: dict[Dir, Move]) -> Decision:
        offered = tuple(moves)
        state, question = build_state(view), build_question(moves)
        start = time.perf_counter()
        result = self.agent.predict(state, question)
        latency_ms = (time.perf_counter() - start) * 1000
        answer = result["answers"][QUESTION_ID]
        probs = parse_probs(answer.get("probabilities"), offered)
        conf = answer.get("confidence")
        sharpness = float(conf) if isinstance(conf, (int, float)) and math.isfinite(conf) else None
        return Decision(probs, argmax_dir(offered, probs), sharpness, latency_ms)

    def warmup(self, n: int = 3) -> None:
        """First calls pay kernel/graph setup; run them before stats start."""
        view, moves = _warmup_sample()
        for _ in range(n):
            self.decide(view, moves)
