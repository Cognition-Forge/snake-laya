"""CLI entry: human vs Laya snake (GUI) or headless decision benchmark.

Usage:
    uv run snake-laya                         # sync versus, english checkpoint, auto device
    uv run snake-laya --mode max --model multilingual
    uv run snake-laya --brain laya-mlx        # Apple-silicon MLX runtime
    uv run snake-laya --bench 200 --device mps
"""

from __future__ import annotations

import argparse
import platform
import random
import re
import sys
import threading
from importlib.util import find_spec
from typing import TextIO

from .brain import (
    BACKEND_DEVICES,
    BACKEND_MODELS,
    BACKEND_PACKAGES,
    DEVICES,
    EXPERIMENTAL_MODELS,
    MODELS,
    Brain,
    HeuristicBrain,
    LayaBrain,
    apply_safety,
    heuristic_pick,
)
from .clock import ActiveClock
from .config import GameConfig, Mode
from .features import analyse
from .game import MIN_H, MIN_W, Board

JOIN_TIMEOUT_S = 5.0


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not value > 0:  # also rejects NaN
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def _grid(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX](\d+)", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"expected WxH (e.g. 30x20), got {text!r}")
    w, h = int(match[1]), int(match[2])
    if w < MIN_W or h < MIN_H:
        raise argparse.ArgumentTypeError(f"grid must be at least {MIN_W}x{MIN_H}, got {w}x{h}")
    return w, h


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="snake-laya", description="Human vs Laya snake with live decision telemetry")
    p.add_argument(
        "--mode", type=Mode, choices=list(Mode), default=Mode.SYNC, help="sync: fair versus; max: unranked showcase"
    )
    p.add_argument("--tick-ms", type=_positive_int, default=120, help="snake step interval (default 120)")
    p.add_argument(
        "--computer-tick-ms",
        type=_positive_int,
        default=None,
        help="computer step interval in sync mode (default: --tick-ms); differing ticks make the match unranked",
    )
    p.add_argument("--grid", type=_grid, default=(30, 20), help=f"WxH, min {MIN_W}x{MIN_H} (default 30x20)")
    p.add_argument("--duration", type=_positive_float, default=180.0, help="active match seconds (default 180)")
    p.add_argument("--seed", type=int, default=None, help="board seed (default random)")
    p.add_argument("--model", choices=list(MODELS), default="english", help="Laya checkpoint (default english)")
    p.add_argument(
        "--device",
        choices=list(DEVICES),
        default=None,
        help="default: auto; laya takes cpu/cuda/mps, laya-mlx takes cpu/gpu/metal",
    )
    p.add_argument(
        "--brain",
        choices=[*BACKEND_MODELS, "heuristic"],
        default="laya",
        help="laya: PyTorch runtime; laya-mlx: Apple-silicon MLX runtime; heuristic: no model",
    )
    p.add_argument("--no-safety", dest="safety", action="store_false", help="execute Laya's raw pick even if fatal")
    p.add_argument("--log", metavar="PATH", help="append one JSONL record per computer step")
    p.add_argument("--bench", type=_positive_int, metavar="N", help="headless: N computer decisions, print stats, exit")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    # device names are backend-specific, so argparse choices (their union) cannot reject a mismatch
    if args.brain in BACKEND_DEVICES and args.device is not None and args.device not in BACKEND_DEVICES[args.brain]:
        allowed = ", ".join(BACKEND_DEVICES[args.brain])
        parser.error(f"--device {args.device} is not available for --brain {args.brain} (choose from {allowed})")
    if args.seed is None:
        args.seed = random.randrange(2**31)
    return args


def config_from(args: argparse.Namespace) -> GameConfig:
    w, h = args.grid
    return GameConfig(
        width=w,
        height=h,
        tick_ms=args.tick_ms,
        computer_tick_ms=args.computer_tick_ms,
        duration_s=args.duration,
        seed=args.seed,
        mode=args.mode,
        safety=args.safety,
    )


def make_brain(args: argparse.Namespace) -> Brain:
    if args.brain == "heuristic":
        return HeuristicBrain()
    return LayaBrain(args.model, args.device, args.brain)


def missing_backend(brain: str) -> str:
    """Pre-flight: error text if the backend package is absent, else "". find_spec does not import it."""
    package = BACKEND_PACKAGES.get(brain)
    if package is None or find_spec(package) is not None:
        return ""
    msg = f"--brain {brain} needs the {package} package, which is not installed; run `uv sync`"
    if brain == "laya-mlx":
        msg += f" (installed on Apple silicon / arm64 macOS only; this Python is {sys.platform}/{platform.machine()})"
    return msg


def failure_hint(args: argparse.Namespace, exc: BaseException | None = None) -> str:
    if args.brain not in BACKEND_MODELS:
        return ""
    # package present but a transitive import (e.g. mlx) failed: the device is never reached
    if isinstance(exc, ImportError):
        return f"Hint: the {args.brain} runtime is incomplete; run `uv sync`."
    if args.device != "cpu":
        return "Hint: retry with --device cpu (some accelerator ops are unsupported)."
    return ""


def _fmt(value: float | None, spec: str = ".1f", suffix: str = "") -> str:
    return "—" if value is None else f"{value:{spec}}{suffix}"


def run_bench(args: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    """Headless max-speed run: no pygame, no threads. Measures raw decision throughput."""
    from .stats import DecisionStats

    cfg = config_from(args)
    try:
        brain = make_brain(args)
        brain.warmup()
    except Exception as exc:
        print(f"error: model load/warm-up failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if hint := failure_hint(args, exc):
            print(hint, file=sys.stderr)
        return 1

    board = Board(cfg.width, cfg.height, cfg.seed)
    clock = ActiveClock()
    stats = DecisionStats(clock.now)
    clock.start()
    while stats.view().decisions < args.bench and not board.cleared:
        if not board.alive:
            board.respawn()
        view = board.view()
        moves = analyse(view)
        decision = brain.decide(view, moves)
        stats.record_prediction(decision.latency_ms)
        pick, overridden = apply_safety(decision.raw, moves, decision.probs, cfg.safety)
        agree = decision.raw is heuristic_pick(moves, view.heading)
        board.step(pick)
        stats.record_step(applied=True, late=False, override=overridden, agree=agree, sharpness=decision.sharpness)
    clock.pause()

    s = stats.view()
    rows = [
        ("brain", f"{brain.label} on {brain.device}"),
        ("decisions", f"{s.decisions} in {clock.now():.2f} s"),
        ("predict", f"last {_fmt(s.last_ms)} ms · P50 {_fmt(s.p50)} ms · P95 {_fmt(s.p95)} ms"),
        ("rate", f"{_fmt(s.mean_dps)} decisions/s"),
        ("overrides", f"{s.overrides} ({_fmt(s.override_pct, suffix='%')})"),
        ("baseline agree", _fmt(s.agree_pct, suffix="%")),
        ("sharpness", f"{_fmt(s.mean_sharpness, '.3f')} mean"),
        ("game", f"total food {board.total_food} · best {board.best} · deaths {board.deaths}"),
    ]
    for key, value in rows:
        print(f"{key:<15} {value}", file=out)
    return 0


def run_gui(args: argparse.Namespace) -> int:
    import pygame  # deferred: --bench must work without a display

    from .decision_log import DecisionLog, NullLog
    from .inference import InferenceWorker
    from .match import Match, Phase
    from .runner import ComputerRunner
    from .stats import DecisionStats
    from .ui import KEY_DIRS, Renderer

    cfg = config_from(args)
    clock = ActiveClock()
    stats = DecisionStats(clock.now)
    log = DecisionLog(args.log) if args.log else NullLog()
    runner: ComputerRunner | None = None
    worker: InferenceWorker | None = None

    def on_reset(seed: int, mode: Mode) -> None:
        if runner is not None:
            runner.request_reset(seed, mode)

    match = Match(cfg, clock, on_reset=on_reset)

    loaded: dict[str, object] = {}

    def load() -> None:
        try:
            brain = make_brain(args)
            brain.warmup()
            loaded["brain"] = brain
        except Exception as exc:
            # one dict write: the render loop polls "error" and must never see it without its hint
            loaded["error"] = (
                f"Model load/warm-up failed: {type(exc).__name__}: {exc}\n{failure_hint(args, exc)}".strip()
            )

    loader = threading.Thread(target=load, name="model-loader", daemon=True)
    loader.start()

    pygame.init()
    renderer = Renderer(cfg.width, cfg.height)
    screen = pygame.display.set_mode(renderer.size)
    pygame.display.set_caption("snake-laya · human vs Laya")
    frame_clock = pygame.time.Clock()
    labels = {"model": args.model if args.brain in BACKEND_MODELS else "heuristic", "device": args.device or "auto"}

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = _handle_key(event.key, match, pygame, KEY_DIRS)

            if runner is None and match.phase is Phase.LOADING:
                if "error" in loaded:
                    match.fail(loaded["error"])
                elif "brain" in loaded:
                    brain = loaded["brain"]
                    labels = {"model": brain.label, "device": brain.device}
                    match.ready()  # runner does not exist yet: its own setup covers this reset
                    worker = InferenceWorker(brain)
                    runner = ComputerRunner(
                        cfg,
                        worker=worker,
                        clock=clock,
                        stats=stats,
                        log=log,
                        meta={"model": brain.label, "device": brain.device},
                    )
                    worker.start()
                    runner.start()

            runner_view = runner.view() if runner is not None else None
            if runner_view is not None and runner_view.error and match.phase is not Phase.ERROR:
                match.fail(f"Inference failed: {runner_view.error}\n{failure_hint(args)}".strip())
            match.update()

            renderer.draw(screen, match, runner_view, stats.view(), labels)
            pygame.display.flip()
            frame_clock.tick(60)
    finally:
        _shutdown(runner, worker, log)
        pygame.quit()
    return 0


def _handle_key(key: int, match, pygame, key_dirs: dict[int, object]) -> bool:
    """Returns False to quit."""
    from .match import Phase

    if match.phase is Phase.CONFIRM_MODE:
        if key == pygame.K_y:
            match.confirm_mode_switch()
        elif key in (pygame.K_n, pygame.K_ESCAPE):
            match.cancel_mode_switch()
        return True
    if key == pygame.K_ESCAPE:
        return False
    if key in key_dirs:
        match.key(key_dirs[key])
    elif key == pygame.K_SPACE:
        match.toggle_pause()
    elif key == pygame.K_r:
        match.restart()
    elif key == pygame.K_m:
        match.request_mode_switch()
    return True


def _shutdown(runner, worker, log) -> None:
    """Stop → join → close log. Daemon threads are only the fallback if a join times out."""
    if runner is not None:
        runner.stop()
        if not runner.join(JOIN_TIMEOUT_S):
            print("warning: computer runner did not stop in time", file=sys.stderr)
    if worker is not None:
        worker.stop()
        if not worker.join(JOIN_TIMEOUT_S):
            print("warning: inference worker still busy; exiting anyway", file=sys.stderr)
    log.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # before any window/thread: a missing runtime otherwise surfaces only after the GUI opens
    if msg := missing_backend(args.brain):
        print(f"error: {msg}", file=sys.stderr)
        return 1
    if args.brain in BACKEND_MODELS and args.model in EXPERIMENTAL_MODELS:
        print(
            f"warning: --model {args.model} is experimental: fine-tuned on four unrelated synthetic workflows",
            file=sys.stderr,
        )
    if args.bench:
        return run_bench(args)
    return run_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
