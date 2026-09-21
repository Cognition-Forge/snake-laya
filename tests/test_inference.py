import threading

import pytest
from helpers import make_view

from snake_laya.brain import Decision
from snake_laya.features import analyse
from snake_laya.game import Dir
from snake_laya.inference import InferenceWorker, Request

VIEW = make_view([(5, 3), (4, 3), (3, 3)], Dir.RIGHT, food=(9, 3))
MOVES = analyse(VIEW)


class GatedBrain:
    """Blocks inside decide() until released, recording the generations it saw."""

    label, device = "GATED", "cpu"

    def __init__(self):
        self.seen: list[int] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def decide(self, view, moves):
        self.seen.append(view.steps)
        self.entered.set()
        assert self.release.wait(5)
        return Decision({Dir.RIGHT: 1.0}, Dir.RIGHT, 1.0, 1.0)

    def warmup(self, n=3):
        pass


def req(gen):
    from dataclasses import replace

    return Request(gen, replace(VIEW, steps=gen), MOVES)  # steps tags the request for GatedBrain


@pytest.fixture
def gated():
    brain = GatedBrain()
    worker = InferenceWorker(brain)
    worker.start()
    yield brain, worker
    brain.release.set()
    worker.stop()
    assert worker.join(2)


def test_newer_request_replaces_unstarted_one(gated):
    brain, worker = gated
    worker.submit(req(1))
    assert brain.entered.wait(2)  # gen 1 in flight
    worker.submit(req(2))
    worker.submit(req(3))  # replaces 2 before it starts
    brain.release.set()
    results = [worker.results.get(timeout=2) for _ in range(2)]
    assert [r.gen for r in results] == [1, 3]
    assert brain.seen == [1, 3]


def test_clear_drops_pending_and_results(gated):
    brain, worker = gated
    worker.submit(req(1))
    assert brain.entered.wait(2)
    worker.submit(req(2))
    worker.clear()  # drops unstarted gen 2
    brain.release.set()
    assert worker.results.get(timeout=2).gen == 1  # in-flight is never cancelled
    worker.clear()
    assert worker.results.empty()
    assert brain.seen == [1]


def test_errors_are_returned_not_raised():
    class Broken:
        label, device = "BROKEN", "cpu"

        def decide(self, view, moves):
            raise RuntimeError("unsupported op")

    worker = InferenceWorker(Broken())
    worker.start()
    try:
        worker.submit(req(4))
        result = worker.results.get(timeout=2)
        assert result.gen == 4 and result.decision is None
        assert isinstance(result.error, RuntimeError)
    finally:
        worker.stop()
        assert worker.join(2)


def test_join_without_start_is_true():
    assert InferenceWorker(GatedBrain()).join(0.1)
