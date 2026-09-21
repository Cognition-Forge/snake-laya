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
        ["--duration", "0"],
        ["--duration", "nan"],
        ["--duration", "x"],
        ["--bench", "0"],
        ["--mode", "fast"],
        ["--model", "klingon"],
        ["--device", "tpu"],
        ["--brain", "oracle"],
    ],
)
def test_invalid_args_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(argv)
    assert exc.value.code == 2


def test_defaults():
    args = cli.parse_args([])
    assert args.grid == (30, 20) and args.mode is Mode.SYNC and args.model == "english"
    assert args.tick_ms == 120 and args.duration == 180.0 and args.safety is True
    assert isinstance(args.seed, int) and args.bench is None and args.log is None


def test_config_from_args():
    args = cli.parse_args(["--grid", "12X8", "--seed", "9", "--mode", "max", "--no-safety", "--tick-ms", "80"])
    cfg = cli.config_from(args)
    assert (cfg.width, cfg.height, cfg.seed, cfg.mode, cfg.safety, cfg.tick_ms) == (12, 8, 9, Mode.MAX, False, 80)


@pytest.mark.parametrize(
    "argv, warns",
    [
        (["--model", "typed-decisions"], True),
        (["--model", "typed-decisions", "--brain", "heuristic"], False),  # model unused
        (["--model", "multilingual"], False),
    ],
)
def test_experimental_model_warning(argv, warns, monkeypatch, capsys):
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


def test_bench_load_failure_returns_1_with_hint(monkeypatch, capsys):
    def boom(args):
        raise RuntimeError("no MPS kernel")

    monkeypatch.setattr(cli, "make_brain", boom)
    args = cli.parse_args(["--bench", "5"])
    assert cli.run_bench(args, io.StringIO()) == 1
    err = capsys.readouterr().err
    assert "no MPS kernel" in err and "--device cpu" in err


@pytest.mark.parametrize(
    "brain, device, hint",
    [("laya", None, True), ("laya", "mps", True), ("laya", "cpu", False), ("heuristic", None, False)],
)
def test_device_hint(brain, device, hint):
    assert bool(cli.device_hint(SimpleNamespace(brain=brain, device=device))) is hint


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
