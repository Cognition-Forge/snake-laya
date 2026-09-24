"""Real-model check. Run: uv run pytest --run-slow tests/test_laya_smoke.py"""

import importlib.util

import pytest

from snake_laya.brain import BACKEND_PACKAGES, LayaBrain
from snake_laya.features import analyse
from snake_laya.game import Board


@pytest.mark.slow
@pytest.mark.parametrize("backend", list(BACKEND_PACKAGES))
@pytest.mark.parametrize("model", ["english", "multilingual"])
def test_real_laya_returns_distribution_over_offered_moves(backend, model):
    package = BACKEND_PACKAGES[backend]
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed")
    brain = LayaBrain(model, backend=backend)
    view = Board(30, 20, seed=0).view()
    moves = analyse(view)
    decision = brain.decide(view, moves)
    assert set(decision.probs) == set(moves)
    assert sum(decision.probs.values()) == pytest.approx(1.0, abs=0.01)
    assert decision.raw in moves
    assert decision.sharpness is not None and 0.0 <= decision.sharpness <= 1.0
    assert decision.latency_ms > 0
