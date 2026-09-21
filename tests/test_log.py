import json

from helpers import FakeTime

from snake_laya.clock import ActiveClock
from snake_laya.config import GameConfig
from snake_laya.decision_log import DecisionLog, NullLog
from snake_laya.inference import InferenceWorker
from snake_laya.main import _shutdown
from snake_laya.runner import ComputerRunner
from snake_laya.stats import DecisionStats


def test_writes_jsonl_and_appends_across_instances(tmp_path):
    path = tmp_path / "d.jsonl"
    log = DecisionLog(path)
    log.write({"a": 1, "text": "héllo"})
    # line-buffered: visible before close
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == {"a": 1, "text": "héllo"}
    log.close()
    DecisionLog(path).close()
    log2 = DecisionLog(path)
    log2.write({"a": 2})
    log2.close()
    assert [json.loads(line)["a"] for line in path.read_text().splitlines()] == [1, 2]


def test_close_is_idempotent_and_write_after_close_is_noop(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    log.close()
    log.close()
    log.write({"x": 1})
    assert log.closed and (tmp_path / "d.jsonl").read_text() == ""


def test_null_log():
    log = NullLog()
    assert log.enabled is False
    log.write({"x": 1})
    log.close()


def test_shutdown_joins_threads_and_closes_log(tmp_path):
    class Instant:
        label, device = "I", "cpu"

        def decide(self, view, moves):
            from snake_laya.brain import HeuristicBrain

            return HeuristicBrain().decide(view, moves)

    clock = ActiveClock()
    clock.start()
    stats = DecisionStats(clock.now)
    log = DecisionLog(tmp_path / "d.jsonl")
    worker = InferenceWorker(Instant())
    cfg = GameConfig(width=12, height=8, tick_ms=10)
    runner = ComputerRunner(cfg, worker=worker, clock=clock, stats=stats, log=log)
    worker.start()
    runner.start()
    import time

    time.sleep(0.2)
    _shutdown(runner, worker, log)
    assert runner.join(0) and worker.join(0)
    assert log.closed
    lines = (tmp_path / "d.jsonl").read_text().splitlines()
    assert lines and all(json.loads(line)["pick"] for line in lines)


def test_shutdown_tolerates_missing_components():
    _shutdown(None, None, NullLog())


def test_fake_time_helper_sanity():
    t = FakeTime(1.0)
    t.advance(0.5)
    assert t() == 1.5
