import math

import pytest
from helpers import make_view

from snake_laya.brain import (
    LAYA_REPO,
    HeuristicBrain,
    LayaBrain,
    apply_safety,
    argmax_dir,
    heuristic_pick,
    late_fallback,
    parse_probs,
)
from snake_laya.features import Move, analyse
from snake_laya.game import Dir

U, D, L, R = Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT


def safe(d, reach=50, after=5, trap=False):
    return Move(d, "safe", None, 6, after, reach, trap)


def fatal(d, reason="wall"):
    return Move(d, "fatal", reason, 6, None, 0, False)


@pytest.mark.parametrize(
    "raw, offered, expected",
    [
        ({"up": 0.2, "down": 0.3, "right": 0.5}, (U, D, R), {U: 0.2, D: 0.3, R: 0.5}),
        ({"up": 0.2}, (U, D), {U: 0.2, D: 0.0}),  # missing key
        ({"up": float("nan"), "down": 0.4}, (U, D), {U: 0.0, D: 0.4}),
        ({"up": -0.1, "down": math.inf}, (U, D), {U: 0.0, D: 0.0}),
        ({"up": "0.7", "down": "x"}, (U, D), {U: 0.7, D: 0.0}),  # numeric string ok, junk → 0
        ({"up": None}, (U,), {U: 0.0}),
        (None, (U, D), {U: 0.0, D: 0.0}),  # wrong type
        ({}, (), {}),
    ],
)
def test_parse_probs(raw, offered, expected):
    assert parse_probs(raw, offered) == expected


@pytest.mark.parametrize(
    "dirs, probs, expected",
    [
        ((U, D, R), {U: 0.1, D: 0.7, R: 0.2}, D),
        ((U, D, R), {U: 0.4, D: 0.4, R: 0.2}, U),  # tie → DIR_ORDER
        ((D, R), {}, D),  # all zero → first in order
    ],
)
def test_argmax_dir(dirs, probs, expected):
    assert argmax_dir(dirs, probs) is expected


@pytest.mark.parametrize(
    "raw, moves, probs, enabled, expected",
    [
        (R, {U: safe(U), D: safe(D), R: safe(R)}, {R: 0.8}, True, (R, False)),
        (R, {U: safe(U), D: safe(D), R: fatal(R)}, {U: 0.1, D: 0.2, R: 0.7}, True, (D, True)),
        (R, {U: safe(U), D: safe(D), R: fatal(R)}, {U: 0.15, D: 0.15, R: 0.7}, True, (U, True)),  # tie
        (R, {U: fatal(U), D: fatal(D), R: fatal(R)}, {R: 0.9}, True, (R, False)),  # no safe move
        (R, {U: safe(U), D: safe(D), R: fatal(R)}, {R: 0.9}, False, (R, False)),  # --no-safety
    ],
)
def test_apply_safety(raw, moves, probs, enabled, expected):
    assert apply_safety(raw, moves, probs, enabled) == expected


@pytest.mark.parametrize(
    "moves, heading, safety, expected",
    [
        ({U: safe(U), D: safe(D), R: safe(R)}, R, True, (R, False)),  # straight is safe
        ({U: safe(U, reach=10), D: safe(D, reach=40), R: fatal(R)}, R, True, (D, True)),  # most room
        ({U: safe(U, reach=40, after=7), D: safe(D, reach=40, after=5), R: fatal(R)}, R, True, (D, True)),  # closer
        ({U: safe(U, reach=40, after=5), D: safe(D, reach=40, after=5), R: fatal(R)}, R, True, (U, True)),  # order
        ({U: fatal(U), D: fatal(D), R: fatal(R)}, R, True, (R, False)),  # doomed
        ({U: safe(U), D: safe(D), R: fatal(R)}, R, False, (R, False)),  # safety off
    ],
)
def test_late_fallback(moves, heading, safety, expected):
    assert late_fallback(moves, heading, safety) == expected


@pytest.mark.parametrize(
    "moves, heading, expected",
    [
        ({U: safe(U, after=3, trap=True), D: safe(D, after=9), R: fatal(R)}, R, D),  # avoid trap over closer
        ({U: safe(U, after=3), D: safe(D, after=9), R: safe(R, after=4)}, R, U),  # closer
        ({U: safe(U, after=4, reach=10), R: safe(R, after=4, reach=30)}, R, R),  # more room
        ({U: fatal(U), D: fatal(D), R: fatal(R)}, R, R),  # no safe → heading
        ({U: fatal(U), D: fatal(D)}, R, U),  # heading not offered → first offered
    ],
)
def test_heuristic_pick(moves, heading, expected):
    assert heuristic_pick(moves, heading) is expected


def test_heuristic_brain_one_hot():
    view = make_view([(5, 3), (4, 3), (3, 3)], R, food=(9, 3))
    decision = HeuristicBrain().decide(view, analyse(view))
    assert decision.raw is R
    assert decision.probs == {U: 0.0, D: 0.0, R: 1.0}
    assert decision.latency_ms >= 0


class StubAgent:
    device = "mps"

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {"move": self.answer}}


@pytest.mark.parametrize(
    "model, subfolder", [("english", None), ("multilingual", "multilingual"), ("typed-decisions", "typed-decisions")]
)
def test_laya_brain_loads_checkpoint(model, subfolder):
    seen = {}

    def loader(repo, device=None, subfolder=None):
        seen.update(repo=repo, device=device, subfolder=subfolder)
        return StubAgent({})

    brain = LayaBrain(model, "cpu", loader=loader)
    assert seen == {"repo": LAYA_REPO, "device": "cpu", "subfolder": subfolder}
    assert brain.device == "mps" and brain.label == f"LAYA {model}"


def test_laya_brain_unknown_model():
    with pytest.raises(ValueError):
        LayaBrain("klingon", loader=lambda *a, **k: StubAgent({}))


@pytest.mark.parametrize(
    "answer, raw, sharpness",
    [
        ({"probabilities": {"up": 0.1, "down": 0.6, "right": 0.3}, "confidence": 0.42}, D, 0.42),
        ({"probabilities": {"up": 0.5, "down": 0.5}, "confidence": float("nan")}, U, None),  # tie, NaN conf
        ({"probabilities": {}}, U, None),  # empty → first offered
        ({"choice": "down"}, U, None),  # no probabilities key
    ],
)
def test_laya_brain_decide(answer, raw, sharpness):
    agent = StubAgent(answer)
    brain = LayaBrain("english", loader=lambda *a, **k: agent)
    view = make_view([(5, 3), (4, 3), (3, 3)], R, food=(9, 3))
    decision = brain.decide(view, analyse(view))
    assert decision.raw is raw
    assert decision.sharpness == sharpness
    assert set(decision.probs) == {U, D, R}
    state, questions = agent.calls[0]
    assert set(questions["move"]["criteria"]) == {"up", "down", "right"}  # reverse never offered
    assert state["food"] == "4 cells right"


def test_laya_brain_warmup_calls_predict():
    agent = StubAgent({"probabilities": {"right": 1.0}})
    LayaBrain("english", loader=lambda *a, **k: agent).warmup(3)
    assert len(agent.calls) == 3


def test_laya_brain_propagates_predict_errors():
    class Broken(StubAgent):
        def predict(self, state, questions):
            raise RuntimeError("MPS op not supported")

    brain = LayaBrain("english", loader=lambda *a, **k: Broken({}))
    view = make_view([(5, 3), (4, 3)], R, food=(9, 3))
    with pytest.raises(RuntimeError):
        brain.decide(view, analyse(view))
