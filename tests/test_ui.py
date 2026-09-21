"""Render smoke tests: every phase/panel combination draws without error (offscreen)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest
from helpers import FakeTime

from snake_laya.clock import ActiveClock
from snake_laya.config import COUNTDOWN_S, GameConfig, Mode
from snake_laya.game import Board, Dir
from snake_laya.match import Match, Phase
from snake_laya.runner import RunnerView
from snake_laya.stats import DecisionStats
from snake_laya.ui import Renderer, fmt_clock, fmt_num, fmt_pct


@pytest.fixture(scope="module", autouse=True)
def pygame_font():
    pygame.font.init()
    yield
    pygame.font.quit()


def runner_view(alive=True, probs=True, late=False, mode=Mode.SYNC):
    board = Board(30, 20, seed=2)
    if not alive:
        board.alive = False
    return RunnerView(
        board=board.view(),
        mode=mode,
        gen=3,
        probs={Dir.UP: 0.1, Dir.DOWN: 0.2, Dir.RIGHT: 0.7} if probs else None,
        executed=Dir.RIGHT,
        last_late=late,
        error=None,
    )


def match_in(phase, mode=Mode.SYNC):
    t = FakeTime()
    m = Match(GameConfig(mode=mode, seed=4, duration_s=5), ActiveClock(t), wall=t)
    if phase is Phase.LOADING:
        return m
    if phase is Phase.ERROR:
        m.fail("Inference failed: RuntimeError: unsupported op " + "x" * 120 + "\nHint: retry with --device cpu")
        return m
    m.ready()
    if phase is Phase.COUNTDOWN:
        return m
    t.advance(COUNTDOWN_S)
    m.update()
    if phase is Phase.PAUSED:
        m.toggle_pause()
    elif phase is Phase.CONFIRM_MODE:
        m.request_mode_switch()
    elif phase is Phase.OVER:
        t.advance(5)
        m.update()
    assert m.phase is phase
    return m


@pytest.mark.parametrize("phase", list(Phase))
@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("with_runner", [True, False])
def test_draw_every_phase(phase, mode, with_runner):
    renderer = Renderer(30, 20)
    surf = pygame.Surface(renderer.size)
    stats = DecisionStats(lambda: 1.0)
    stats.record_prediction(9.1)
    stats.record_step(applied=True, late=False, override=False, agree=True, sharpness=0.4)
    rv = runner_view(mode=mode) if with_runner else None
    renderer.draw(surf, match_in(phase, mode), rv, stats.view(), {"model": "LAYA english", "device": "mps"})


@pytest.mark.parametrize(
    "rv", [runner_view(alive=False), runner_view(probs=False, late=True)], ids=["dead", "late-no-probs"]
)
def test_draw_runner_edge_states(rv):
    renderer = Renderer(30, 20)
    surf = pygame.Surface(renderer.size)
    renderer.draw(surf, match_in(Phase.RUNNING), rv, DecisionStats(lambda: 0.0).view(), {})


@pytest.mark.parametrize("grid", [(12, 8), (30, 20), (64, 48)])
def test_layout_scales_with_grid(grid):
    r = Renderer(*grid)
    assert 8 <= r.cell <= 28
    assert r.size[0] > 2 * r.board_w and r.size[1] > r.board_h


@pytest.mark.parametrize(
    "fn, value, text",
    [
        (fmt_num, None, "—"),
        (fmt_num, 9.14, "9.1"),
        (fmt_pct, None, "—"),
        (fmt_pct, 12.345, "12.3%"),
        (fmt_clock, 180.0, "3:00"),
        (fmt_clock, 0.2, "0:01"),
        (fmt_clock, 0.0, "0:00"),
    ],
)
def test_formatters(fn, value, text):
    assert fn(value) == text
