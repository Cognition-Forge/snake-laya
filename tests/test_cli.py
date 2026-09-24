import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

from snake_laya import main as cli
from snake_laya.config import Mode


@pytest.mark.parametrize(
    "argv",
    [
        ["--grid", "5x5"],
        ["--grid", "11x8"],
        ["--grid", "12x7"],
        ["--grid", "abc"],
        ["--grid", "30x"],
        ["--tick-ms", "0"],
        ["--tick-ms", "-5"],
        ["--tick-ms", "1.5"],
        ["--computer-tick-ms", "0"],
        ["--computer-tick-ms", "-5"],
        ["--computer-tick-ms", "1.5"],
        ["--computer-tick-ms", "fast"],
        ["--duration", "0"],
        ["--duration", "nan"],
        ["--duration", "x"],
        ["--bench", "0"],
        ["--mode", "fast"],
        ["--model", "klingon"],
        ["--device", "tpu"],
        ["--brain", "oracle"],
        ["--brain", "laya", "--device", "gpu"],  # MLX device on the torch backend
        ["--brain", "laya", "--device", "metal"],
        ["--brain", "laya-mlx", "--device", "cuda"],  # torch devices on the MLX backend
        ["--brain", "laya-mlx", "--device", "mps"],
        ["--brain", "heuristic", "--device", "tpu"],
    ],
)
def test_invalid_args_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(argv)
    assert exc.value.code == 2


def test_defaults():
    args = cli.parse_args([])
    assert args.grid == (30, 20) and args.mode is Mode.SYNC and args.model == "english"
    assert args.tick_ms == 120 and args.computer_tick_ms is None and args.duration == 180.0 and args.safety is True
    assert isinstance(args.seed, int) and args.bench is None and args.log is None
    assert args.brain == "laya" and args.device is None


@pytest.mark.parametrize(
    "argv, brain, device",
    [
        (["--brain", "laya-mlx"], "laya-mlx", None),
        (["--brain", "laya-mlx", "--device", "gpu"], "laya-mlx", "gpu"),
        (["--brain", "laya-mlx", "--device", "metal"], "laya-mlx", "metal"),
        (["--brain", "laya-mlx", "--device", "cpu"], "laya-mlx", "cpu"),
        (["--brain", "laya", "--device", "mps"], "laya", "mps"),
        (["--brain", "laya", "--device", "cuda"], "laya", "cuda"),
        (["--brain", "heuristic"], "heuristic", None),
        (["--brain", "heuristic", "--device", "cpu"], "heuristic", "cpu"),  # unused but accepted
    ],
)
def test_backend_device_combinations_accepted(argv, brain, device):
    args = cli.parse_args(argv)
    assert args.brain == brain and args.device == device


@pytest.mark.parametrize("backend", ["laya", "laya-mlx"])
@pytest.mark.parametrize("model", ["english", "multilingual", "typed-decisions"])
def test_make_brain_passes_backend(backend, model, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "LayaBrain", lambda m, d, b: seen.update(model=m, device=d, backend=b))
    cli.make_brain(SimpleNamespace(brain=backend, model=model, device="cpu"))
    assert seen == {"model": model, "device": "cpu", "backend": backend}


def test_make_brain_heuristic_ignores_model():
    brain = cli.make_brain(SimpleNamespace(brain="heuristic", model="multilingual", device=None))
    assert brain.label == "HEURISTIC"


def test_config_from_args():
    args = cli.parse_args(["--grid", "12X8", "--seed", "9", "--mode", "max", "--no-safety", "--tick-ms", "80"])
    cfg = cli.config_from(args)
    assert (cfg.width, cfg.height, cfg.seed, cfg.mode, cfg.safety, cfg.tick_ms) == (12, 8, 9, Mode.MAX, False, 80)
    assert cfg.computer_tick_ms is None


def test_config_from_args_computer_tick():
    cfg = cli.config_from(cli.parse_args(["--tick-ms", "100", "--computer-tick-ms", "40"]))
    assert (cfg.tick_ms, cfg.computer_tick_ms, cfg.computer_tick_s) == (100, 40, 0.04)


@pytest.mark.parametrize(
    "argv, warns",
    [
        (["--model", "typed-decisions"], True),
        (["--model", "typed-decisions", "--brain", "laya-mlx"], True),
        (["--model", "typed-decisions", "--brain", "heuristic"], False),  # model unused
        (["--model", "multilingual"], False),
        (["--model", "multilingual", "--brain", "laya-mlx"], False),
    ],
)
def test_experimental_model_warning(argv, warns, monkeypatch, capsys):
    monkeypatch.setattr(cli, "find_spec", lambda name: object())  # env-independent: laya_mlx is arm64-macOS only
    monkeypatch.setattr(cli, "run_gui", lambda args: 0)
    assert cli.main(argv) == 0
    assert ("experimental" in capsys.readouterr().err) is warns


def test_bench_heuristic_prints_table():
    args = cli.parse_args(["--bench", "30", "--brain", "heuristic", "--seed", "3"])
    out = io.StringIO()
    assert cli.run_bench(args, out) == 0
    text = out.getvalue()
    for key in ("brain", "decisions", "predict", "rate", "overrides", "baseline agree", "sharpness", "game"):
        assert key in text
    assert "HEURISTIC" in text and "30 in" in text


@pytest.mark.parametrize(
    "argv, exc, want, not_want",
    [
        (["--bench", "5"], RuntimeError("no MPS kernel"), "--device cpu", "uv sync"),
        (
            ["--bench", "5", "--brain", "laya-mlx"],
            ModuleNotFoundError("No module named 'mlx'"),
            "uv sync",
            "--device cpu",
        ),
        (["--bench", "5", "--device", "cpu"], RuntimeError("bad weights"), None, "Hint"),
    ],
)
def test_bench_load_failure_returns_1_with_hint(argv, exc, want, not_want, monkeypatch, capsys):
    def boom(args):
        raise exc

    monkeypatch.setattr(cli, "make_brain", boom)
    assert cli.run_bench(cli.parse_args(argv), io.StringIO()) == 1
    err = capsys.readouterr().err
    assert str(exc) in err and not_want not in err
    assert want is None or want in err


@pytest.mark.parametrize(
    "brain, device, exc, want",
    [
        ("laya", None, None, "--device cpu"),
        ("laya", "mps", RuntimeError("op"), "--device cpu"),
        ("laya", "cpu", RuntimeError("op"), ""),
        ("laya", "mps", ImportError("torch"), "uv sync"),
        ("laya-mlx", None, ModuleNotFoundError("laya_mlx"), "uv sync"),
        ("laya-mlx", "cpu", ImportError("mlx"), "uv sync"),  # import fails before the device is used
        ("laya-mlx", "gpu", None, "--device cpu"),
        ("laya-mlx", "cpu", None, ""),
        ("heuristic", "mps", ImportError("x"), ""),
        ("heuristic", None, None, ""),
    ],
)
def test_failure_hint(brain, device, exc, want):
    hint = cli.failure_hint(SimpleNamespace(brain=brain, device=device), exc)
    if want:
        assert want in hint
        assert ("--device cpu" in hint) is (want == "--device cpu")
    else:
        assert hint == ""


def _no_find_spec(name):
    raise AssertionError(f"find_spec({name!r}) must not be called")


@pytest.mark.parametrize(
    "brain, spec, find, want",
    [
        ("laya-mlx", None, None, ("laya_mlx", "uv sync", "arm64", "linux/x86_64")),
        ("laya", None, None, ("--brain laya ", "laya package", "uv sync")),
        ("laya-mlx", object(), None, ()),
        ("laya", object(), None, ()),
        ("heuristic", None, _no_find_spec, ()),
    ],
)
def test_missing_backend(brain, spec, find, want, monkeypatch):
    monkeypatch.setattr(cli, "find_spec", find or (lambda name: spec))
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
    msg = cli.missing_backend(brain)
    if not want:
        assert msg == ""
    for part in want:
        assert part in msg
    if brain == "laya":
        assert "arm64" not in msg


@pytest.mark.parametrize("extra", [[], ["--bench", "5"]])
def test_main_missing_backend_exits_1_before_run(extra, monkeypatch, capsys):
    def never(*a, **k):
        raise AssertionError("must fail before running")

    monkeypatch.setattr(cli, "find_spec", lambda name: None)
    monkeypatch.setattr(cli, "run_gui", never)
    monkeypatch.setattr(cli, "run_bench", never)
    assert cli.main(["--brain", "laya-mlx", "--model", "typed-decisions", *extra]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: --brain laya-mlx needs the laya_mlx package")
    assert "experimental" not in err


def test_main_heuristic_skips_backend_check(monkeypatch):
    monkeypatch.setattr(cli, "find_spec", _no_find_spec)
    monkeypatch.setattr(cli, "run_gui", lambda args: 0)
    assert cli.main(["--brain", "heuristic"]) == 0


def test_bench_is_headless():
    """--bench must never import pygame (runs without a display)."""
    code = (
        "import sys; from snake_laya.main import main; "
        "rc = main(['--bench', '10', '--brain', 'heuristic', '--seed', '1']); "
        "assert 'pygame' not in sys.modules, 'pygame imported'; sys.exit(rc)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


class Keys:
    K_y, K_n, K_ESCAPE, K_SPACE, K_r, K_m, K_UP = range(7)


class MatchSpy:
    def __init__(self, phase):
        from snake_laya.match import Phase

        self.phase = getattr(Phase, phase)
        self.calls = []

    def __getattr__(self, name):
        return lambda *a: self.calls.append((name, *a))


@pytest.mark.parametrize(
    "phase, key, calls, keep_running",
    [
        ("CONFIRM_MODE", Keys.K_y, [("confirm_mode_switch",)], True),
        ("CONFIRM_MODE", Keys.K_n, [("cancel_mode_switch",)], True),
        ("CONFIRM_MODE", Keys.K_ESCAPE, [("cancel_mode_switch",)], True),  # Esc cancels, doesn't quit
        ("CONFIRM_MODE", Keys.K_UP, [], True),
        ("RUNNING", Keys.K_ESCAPE, [], False),
        ("RUNNING", Keys.K_SPACE, [("toggle_pause",)], True),
        ("RUNNING", Keys.K_r, [("restart",)], True),
        ("RUNNING", Keys.K_m, [("request_mode_switch",)], True),
        ("RUNNING", Keys.K_UP, [("key", "UP")], True),
        ("RUNNING", Keys.K_y, [], True),
    ],
)
def test_handle_key(phase, key, calls, keep_running):
    spy = MatchSpy(phase)
    assert cli._handle_key(key, spy, Keys, {Keys.K_UP: "UP"}) is keep_running
    assert spy.calls == calls
