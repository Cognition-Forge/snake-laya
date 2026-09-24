import math
import os
from types import SimpleNamespace

import pytest
from helpers import make_view
from huggingface_hub.errors import LocalEntryNotFoundError

from snake_laya.brain import (
    BACKEND_DEVICES,
    BACKEND_MODELS,
    CHECKPOINT_FILES,
    LAYA_MLX_MODELS,
    LAYA_MODELS,
    LAYA_REPO,
    LAYA_REVISION,
    MODELS,
    Checkpoint,
    HeuristicBrain,
    LayaBrain,
    apply_safety,
    argmax_dir,
    device_label,
    heuristic_pick,
    late_fallback,
    parse_probs,
    resolve_checkpoint,
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


@pytest.mark.parametrize("backend", list(BACKEND_MODELS))
@pytest.mark.parametrize("model", MODELS)
def test_laya_brain_loads_checkpoint(backend, model):
    seen = {}
    ckpt = BACKEND_MODELS[backend][model]

    def loader(repo, device=None, subfolder=None):
        seen.update(repo=repo, device=device, subfolder=subfolder)
        return StubAgent({})

    brain = LayaBrain(model, "cpu", backend, loader=loader, resolve=lambda c: f"/snap/{c.repo}")
    assert seen == {"repo": f"/snap/{ckpt.repo}", "device": "cpu", "subfolder": ckpt.subfolder}
    assert brain.device == "mps" and brain.label == f"{backend.upper()} {model}"
    assert brain.backend == backend and brain.model == model


def test_laya_brain_defaults_to_torch_backend():
    brain = LayaBrain(loader=lambda *a, **k: StubAgent({}), resolve=NO_HUB)
    assert brain.backend == "laya" and brain.label == "LAYA english"


@pytest.mark.parametrize(
    "model, backend, device",
    [
        ("klingon", "laya", None),  # unknown model
        ("klingon", "laya-mlx", None),
        ("english", "laya-onnx", None),  # unknown backend
        ("english", "laya", "gpu"),  # MLX device name on the torch backend
        ("english", "laya", "metal"),
        ("english", "laya-mlx", "cuda"),  # torch device names on the MLX backend
        ("english", "laya-mlx", "mps"),
    ],
)
def test_laya_brain_rejects_bad_combinations(model, backend, device):
    with pytest.raises(ValueError):
        LayaBrain(model, device, backend, loader=lambda *a, **k: StubAgent({}), resolve=NO_HUB)


def test_laya_brain_rejects_bad_combination_before_loading():
    """Validation must not pay for a model load first."""

    def loader(*a, **k):
        raise AssertionError("loader must not run")

    with pytest.raises(ValueError):
        LayaBrain("english", "cuda", "laya-mlx", loader=loader, resolve=NO_HUB)


@pytest.mark.parametrize("backend", list(BACKEND_MODELS))
@pytest.mark.parametrize("device", [None, "cpu"])
def test_laya_brain_accepts_shared_and_absent_devices(backend, device):
    brain = LayaBrain("english", device, backend, loader=lambda *a, **k: StubAgent({}), resolve=NO_HUB)
    assert brain.backend == backend


@pytest.mark.parametrize(
    "agent_device, requested, expected",
    [
        ("mps", None, "mps"),  # torch reports a plain name
        ("cuda:0", None, "cuda:0"),
        ("Device(gpu, 0)", None, "gpu"),  # MLX repr
        ("Device(cpu, 0)", "cpu", "cpu"),
        ("Device(gpu, 12)", None, "gpu"),
        (None, "cpu", "cpu"),  # agent exposes device=None → fall back to the request
        (None, None, "auto"),
        ("MISSING", "mps", "mps"),  # no device attribute at all
    ],
)
def test_device_label(agent_device, requested, expected):
    agent = object() if agent_device == "MISSING" else SimpleNamespace(device=agent_device)
    assert device_label(agent, requested) == expected


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
    brain = LayaBrain("english", loader=lambda *a, **k: agent, resolve=NO_HUB)
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
    LayaBrain("english", loader=lambda *a, **k: agent, resolve=NO_HUB).warmup(3)
    assert len(agent.calls) == 3


def test_laya_brain_propagates_predict_errors():
    class Broken(StubAgent):
        def predict(self, state, questions):
            raise RuntimeError("MPS op not supported")

    brain = LayaBrain("english", loader=lambda *a, **k: Broken({}), resolve=NO_HUB)
    view = make_view([(5, 3), (4, 3)], R, food=(9, 3))
    with pytest.raises(RuntimeError):
        brain.decide(view, analyse(view))


def NO_HUB(ckpt):
    return "/snap"


class FakeHub:
    """snapshot_download stand-in over a tmp snapshot dir; records calls."""

    def __init__(self, root, cached: bool, online_error: Exception | None = None):
        self.root, self.cached, self.online_error = str(root), cached, online_error
        self.calls = []

    def __call__(self, repo, *, revision, allow_patterns, local_files_only=False):
        self.calls.append({"repo": repo, "revision": revision, "files": allow_patterns, "offline": local_files_only})
        if local_files_only:
            if not self.cached:
                raise LocalEntryNotFoundError("not cached")
            return self.root
        if self.online_error:
            raise self.online_error
        for f in allow_patterns:
            write(self.root, f)
        return self.root


def write(root, rel):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("x")


def prefixed(sub):
    return [f"{sub}/{f}" if sub else f for f in CHECKPOINT_FILES]


ALL_CHECKPOINTS = [
    pytest.param(ckpt, id=f"{backend}-{model}")
    for backend, models in BACKEND_MODELS.items()
    for model, ckpt in models.items()
]


@pytest.mark.parametrize("ckpt", ALL_CHECKPOINTS)
@pytest.mark.parametrize(
    "cached, present, online",
    [
        (True, "all", False),  # complete cache → no hub call
        (False, "none", True),  # revision never cached
        (True, "none", True),  # snapshot dir exists (other checkpoint fetched) but not this one
        (True, "partial", True),  # tokenizer/encoder missing → would send the backend to the hub
    ],
    ids=["hit", "miss", "other-checkpoint-only", "partial"],
)
def test_resolve_checkpoint(tmp_path, ckpt, cached, present, online):
    files = prefixed(ckpt.subfolder)
    for f in {"all": files, "none": [], "partial": files[:2]}[present]:
        write(tmp_path, f)
    hub = FakeHub(tmp_path, cached)
    assert resolve_checkpoint(ckpt, download=hub) == str(tmp_path)
    assert [c["offline"] for c in hub.calls] == ([True, False] if online else [True])
    for call in hub.calls:
        assert call == {"repo": ckpt.repo, "revision": ckpt.revision, "files": files, "offline": call["offline"]}
    assert all(os.path.isfile(tmp_path / f) for f in files)


def test_resolve_checkpoint_english_excludes_bundled_subfolders():
    subs = {c.subfolder for c in LAYA_MODELS.values() if c.subfolder}
    assert not any(f.split("/")[0] in subs for f in prefixed(None))


def test_resolve_checkpoint_online_failure_propagates(tmp_path):
    hub = FakeHub(tmp_path, cached=False, online_error=ConnectionError("offline"))
    with pytest.raises(ConnectionError):
        resolve_checkpoint(LAYA_MODELS["english"], download=hub)


@pytest.mark.parametrize("ckpt", ALL_CHECKPOINTS)
def test_revisions_are_commit_hashes(ckpt):
    assert len(ckpt.revision) == 40 and all(c in "0123456789abcdef" for c in ckpt.revision)


def test_torch_checkpoints_share_one_pinned_repo():
    assert {(c.repo, c.revision) for c in LAYA_MODELS.values()} == {(LAYA_REPO, LAYA_REVISION)}


def test_mlx_checkpoints_use_distinct_repo_roots():
    """MLX ports are published one repo per checkpoint, so none may carry a subfolder."""
    assert len({c.repo for c in LAYA_MLX_MODELS.values()}) == len(LAYA_MLX_MODELS)
    assert all(c.subfolder is None for c in LAYA_MLX_MODELS.values())


def test_backends_offer_the_same_model_names():
    assert all(tuple(models) == MODELS for models in BACKEND_MODELS.values())


def test_every_backend_declares_devices():
    assert set(BACKEND_DEVICES) == set(BACKEND_MODELS)
    assert all("cpu" in devices for devices in BACKEND_DEVICES.values())


def test_checkpoint_defaults_to_repo_root():
    assert Checkpoint("repo", "abc").subfolder is None
