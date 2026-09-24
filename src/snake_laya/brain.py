"""Computer move selection: Laya policy, heuristic baseline, safety and late-tick fallback."""

from __future__ import annotations

import importlib
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .features import QUESTION_ID, Move, analyse, build_question, build_state
from .game import DIR_ORDER, Board, BoardView, Dir


@dataclass(frozen=True)
class Checkpoint:
    """One pinned snapshot: hub repo, commit, and the subfolder holding it (None = repo root)."""

    repo: str
    revision: str
    subfolder: str | None = None


LAYA_REPO = "convaiinnovations/laya"
# Pinned: upstream commits can change model behaviour. Bump deliberately; a new revision downloads once.
LAYA_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
# Every file an agent reads from a checkpoint dir. Missing tokenizer/encoder makes the backend fetch the
# base encoder from the hub, so all are required for an offline start.
CHECKPOINT_FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
)
# --brain laya: upstream PyTorch weights; one repo bundles all three checkpoints as subfolders
LAYA_MODELS: dict[str, Checkpoint] = {
    "english": Checkpoint(LAYA_REPO, LAYA_REVISION),
    "multilingual": Checkpoint(LAYA_REPO, LAYA_REVISION, "multilingual"),
    "typed-decisions": Checkpoint(LAYA_REPO, LAYA_REVISION, "typed-decisions"),
}
# --brain laya-mlx: independent Apple-silicon FP16 ports; one repo per checkpoint, none nested
LAYA_MLX_MODELS: dict[str, Checkpoint] = {
    "english": Checkpoint("aac6fef/laya-mlx", "20aed815fc6acde75733882e7ec0e3f28aeb9717"),
    "multilingual": Checkpoint("aac6fef/laya-multilingual-mlx", "f2b4faf51023039425946074e2cf1361d2db11d5"),
    "typed-decisions": Checkpoint("aac6fef/laya-typed-decisions-mlx", "f9e501c2080cc57c13d6887820329758f5351125"),
}
BACKEND_MODELS: dict[str, dict[str, Checkpoint]] = {"laya": LAYA_MODELS, "laya-mlx": LAYA_MLX_MODELS}
BACKEND_PACKAGES = {"laya": "laya", "laya-mlx": "laya_mlx"}
# torch names the accelerator; MLX only splits cpu from the Metal GPU ("gpu" and "metal" are aliases)
BACKEND_DEVICES: dict[str, tuple[str, ...]] = {"laya": ("cpu", "cuda", "mps"), "laya-mlx": ("cpu", "gpu", "metal")}
MODELS = tuple(LAYA_MODELS)
DEVICES = tuple(dict.fromkeys(d for devices in BACKEND_DEVICES.values() for d in devices))
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


def resolve_checkpoint(ckpt: Checkpoint, download: Callable[..., str] | None = None) -> str:
    """Local snapshot root holding `ckpt` at its pinned revision: cache first, hub only when files are missing.

    Fetches only this checkpoint's files; a bundled repo's other subfolders are left alone.
    """
    if download is None:
        from huggingface_hub import snapshot_download as download  # deferred: import cost

    files = [f"{ckpt.subfolder}/{f}" if ckpt.subfolder else f for f in CHECKPOINT_FILES]
    kw = {"revision": ckpt.revision, "allow_patterns": files}
    try:
        root = download(ckpt.repo, local_files_only=True, **kw)
        # an existing snapshot dir is returned even when these files were never fetched
        if all(os.path.isfile(os.path.join(root, f)) for f in files):
            return root
    except FileNotFoundError:  # huggingface_hub LocalEntryNotFoundError: revision not cached
        pass
    return download(ckpt.repo, **kw)


_MLX_DEVICE = re.compile(r"Device\((\w+),\s*\d+\)")


def device_label(agent: Any, requested: str | None) -> str:
    """Device the agent actually chose, for display: torch reports `mps`, MLX reports `Device(gpu, 0)`."""
    device = getattr(agent, "device", None)
    if device is None:
        return requested or "auto"
    match = _MLX_DEVICE.fullmatch(str(device))
    return match[1] if match else str(device)


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
    """One Laya `choice` question per step over the legal directions.

    `backend` selects the runtime: `laya` (PyTorch, any device) or `laya-mlx` (Apple silicon, MLX).
    Both expose the same `load(path, device=, subfolder=)` and answer schema, so only the checkpoint
    repos and the accepted device names differ.
    """

    def __init__(
        self,
        model: str = "english",
        device: str | None = None,
        backend: str = "laya",
        loader: Callable[..., Any] | None = None,
        resolve: Callable[[Checkpoint], str] = resolve_checkpoint,
    ):
        if backend not in BACKEND_MODELS:
            raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKEND_MODELS)}")
        models = BACKEND_MODELS[backend]
        if model not in models:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(models)}")
        if device is not None and device not in BACKEND_DEVICES[backend]:
            raise ValueError(
                f"backend {backend!r} has no device {device!r}; choose from {list(BACKEND_DEVICES[backend])}"
            )
        if loader is None:
            loader = importlib.import_module(BACKEND_PACKAGES[backend]).load  # deferred: pulls in torch/mlx

        self.model = model
        self.backend = backend
        ckpt = models[model]
        # local path → the backend skips its own hub lookup
        self.agent = loader(resolve(ckpt), device=device, subfolder=ckpt.subfolder)
        self.device = device_label(self.agent, device)
        self.label = f"{backend.upper()} {model}"

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
