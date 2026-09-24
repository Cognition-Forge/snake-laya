import pytest
from helpers import FakeTime, FakeWorker, ListLog, straight_decision

from snake_laya.brain import Decision
from snake_laya.clock import ActiveClock
from snake_laya.config import GameConfig, Mode
from snake_laya.game import START_LEN, Dir
from snake_laya.inference import Result
from snake_laya.runner import ComputerRunner
from snake_laya.stats import DecisionStats

U, D, L, R = Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT
NEVER = 1e9  # inference latency that never completes within a test


def build(
    t,
    *,
    mode=Mode.SYNC,
    latency=0.03,
    decide=straight_decision,
    safety=True,
    tick_ms=100,
    computer_tick_ms=None,
    log=None,
    meta=None,
):
    clock = ActiveClock(t)
    stats = DecisionStats(clock.now)
    worker = FakeWorker(t, latency, decide)
    cfg = GameConfig(
        width=12, height=8, tick_ms=tick_ms, computer_tick_ms=computer_tick_ms, seed=1, mode=mode, safety=safety
    )
    runner = ComputerRunner(cfg, worker=worker, clock=clock, stats=stats, log=log, meta=meta)
    clock.start()
    return runner, worker, clock, stats


def place(runner, worker, body, heading, food):
    """Replace the computer board layout and re-request as a fresh match generation."""
    board = runner._board
    board.body.clear()
    board.body.extend(body)
    board.heading, board.food = heading, food
    worker._inflight = None
    worker.clear()
    runner._advance()
    runner._gen_floor = runner._gen


def decision(raw, probs=None, sharpness=0.5):
    return Decision(probs or {raw: 1.0}, raw, sharpness, 5.0)


def test_sync_on_time_prediction_applied_at_deadline():
    t = FakeTime()
    runner, worker, _, stats = build(t, latency=0.03)
    head = runner.view().board.head
    runner.iterate()
    s = stats.view()
    assert (s.steps, s.decisions, s.late, s.stale) == (1, 1, 0, 0)
    assert t.t == pytest.approx(0.1)  # stepped on the tick, not when the result arrived
    assert runner.view().board.head == (head[0] + 1, head[1])
    assert runner.view().executed is R and not runner.view().last_late
    assert len(worker.submitted) == 2  # next generation requested


@pytest.mark.parametrize(
    "computer_tick_ms, latency, step_times, late",
    [
        (None, 0.01, [0.1, 0.2, 0.3], 0),  # unset → human tick
        (40, 0.01, [0.04, 0.08, 0.12], 0),  # faster than human tick
        (250, 0.01, [0.25, 0.5, 0.75], 0),  # slower than human tick
        (40, 0.06, [0.04, 0.08, 0.12], 3),  # inference slower than computer tick → LATE
    ],
)
def test_sync_paced_by_computer_tick(computer_tick_ms, latency, step_times, late):
    t = FakeTime()
    runner, _, _, stats = build(t, tick_ms=100, computer_tick_ms=computer_tick_ms, latency=latency)
    times = []
    for _ in step_times:
        runner.iterate()
        times.append(t.t)
    assert times == pytest.approx(step_times)
    s = stats.view()
    assert (s.steps, s.late) == (len(step_times), late)


def test_max_mode_ignores_computer_tick():
    t = FakeTime()
    runner, _, _, stats = build(t, mode=Mode.MAX, computer_tick_ms=500, latency=0.02)
    runner.iterate()
    assert t.t == pytest.approx(0.02) and stats.view().steps == 1


def test_sync_slow_inference_misses_several_ticks():
    t = FakeTime()
    runner, _, _, stats = build(t, latency=0.25)
    for _ in range(3):
        runner.iterate()
    s = stats.view()
    assert (s.steps, s.decisions, s.late) == (3, 0, 3)
    assert s.late_pct == 100.0
    assert s.stale == 1 and s.predictions == 1  # gen-1 result landed at 0.25, after its tick
    assert runner.view().last_late and runner.view().probs is None


def test_stale_then_current_result_applied_not_late():
    """R1: a stale result ahead of the current one must not trigger LATE."""
    t = FakeTime()
    runner, worker, _, stats = build(t, latency=NEVER)
    runner.iterate()  # one late step: generation advances inside this match
    gen = runner.view().gen
    worker.inject(t.t + 0.1, Result(gen - 1, decision(D)))  # stale, same arrival as current
    worker.inject(t.t + 0.1, Result(gen, decision(U, {U: 0.9, D: 0.05, R: 0.05})))
    runner.iterate()
    s = stats.view()
    assert (s.steps, s.late, s.decisions, s.stale) == (2, 1, 1, 1)
    assert runner.view().executed is U


def test_results_from_previous_match_are_ignored():
    t = FakeTime()
    runner, worker, _, stats = build(t, latency=NEVER)
    worker.inject(0.01, Result(runner._gen_floor - 1, decision(U)))
    runner.iterate()
    s = stats.view()
    assert (s.stale, s.predictions, s.late) == (0, 0, 1)


WALL_AHEAD = ([(11, 4), (10, 4), (9, 4)], R, (0, 0))


@pytest.mark.parametrize(
    "safety, alive, overrides",
    [(True, True, 1), (False, False, 0)],
)
def test_late_step_with_fatal_heading(safety, alive, overrides):
    t = FakeTime()
    runner, worker, _, stats = build(t, latency=NEVER, safety=safety)
    place(runner, worker, *WALL_AHEAD)
    runner.iterate()
    s = stats.view()
    assert (s.late, s.overrides) == (1, overrides)
    assert runner.view().board.alive is alive


@pytest.mark.parametrize(
    "safety, alive, executed",
    [(True, True, D), (False, False, R)],
)
def test_fatal_raw_pick(safety, alive, executed):
    t = FakeTime()
    into_wall = lambda req: decision(R, {U: 0.1, D: 0.3, R: 0.6})  # noqa: E731
    runner, worker, _, stats = build(t, decide=into_wall, safety=safety)
    place(runner, worker, *WALL_AHEAD)
    runner.iterate()
    assert runner.view().board.alive is alive
    assert runner.view().executed is executed
    assert stats.view().overrides == (1 if safety else 0)


def test_max_mode_steps_on_every_result():
    t = FakeTime()
    runner, _, _, stats = build(t, mode=Mode.MAX, latency=0.01)
    for _ in range(4):
        runner.iterate()
    s = stats.view()
    assert (s.steps, s.decisions, s.late) == (4, 4, 0)
    assert t.t == pytest.approx(0.04)


def test_respawn_waits_one_second_of_active_time():
    t = FakeTime()
    runner, worker, clock, _ = build(t, latency=0.01, safety=False)
    place(runner, worker, *WALL_AHEAD)
    runner.iterate()  # dies at active 0.1
    assert not runner.view().board.alive
    submitted = len(worker.submitted)

    clock.pause()
    t.advance(5.0)
    runner.iterate()
    assert not runner.view().board.alive  # paused time does not count

    clock.resume()
    t.advance(0.5)
    runner.iterate()
    assert not runner.view().board.alive  # active 0.6 < 1.1
    assert len(worker.submitted) == submitted  # no requests while dead

    t.advance(0.5)
    runner.iterate()
    view = runner.view().board
    assert view.alive and view.length == START_LEN and view.deaths == 1
    assert len(worker.submitted) == submitted + 1


def test_paused_runner_neither_steps_nor_requests():
    t = FakeTime()
    runner, worker, clock, stats = build(t)
    clock.pause()
    before = len(worker.submitted)
    runner.iterate()
    assert stats.view().steps == 0 and len(worker.submitted) == before


def test_reset_applies_in_runner_thread():
    t = FakeTime()
    runner, worker, _, stats = build(t)
    runner.iterate()
    gen = runner.view().gen
    cleared = worker.cleared
    runner.request_reset(7, Mode.MAX)
    runner.iterate()
    view = runner.view()
    assert view.mode is Mode.MAX and view.board.steps == 0 and view.gen > gen
    assert stats.view().steps == 0
    assert worker.cleared == cleared + 1


def test_inference_error_halts_runner():
    t = FakeTime()
    runner, worker, _, stats = build(t, latency=NEVER)
    worker.inject(0.01, Result(runner.view().gen, None, RuntimeError("boom")))
    runner.iterate()
    runner.iterate()
    assert runner.view().error == "RuntimeError: boom"
    assert stats.view().steps == 0


def test_log_record_fields():
    t = FakeTime()
    log = ListLog()
    runner, _, _, _ = build(t, log=log, meta={"model": "LAYA english", "device": "mps"})
    runner.iterate()
    (rec,) = log.records
    assert set(rec) == {
        "match_id", "seed", "mode", "model", "device", "gen", "t_active", "state", "criteria", "moves",
        "probabilities", "raw", "pick", "override", "late", "baseline_pick", "latency_ms",
        "ate", "died", "score_delta", "length_after",
    }  # fmt: skip
    assert rec["mode"] == "sync" and rec["model"] == "LAYA english"
    assert rec["raw"] == rec["pick"] == "right" and rec["late"] is False
    assert set(rec["criteria"]) == set(rec["moves"]) == {"up", "down", "right"}
    assert rec["moves"]["right"]["status"] == "safe" and rec["moves"]["right"]["reachable"] > 0
    assert rec["died"] is False and rec["score_delta"] in (0, 1)


def test_log_late_record_has_no_decision():
    t = FakeTime()
    log = ListLog()
    runner, _, _, _ = build(t, latency=NEVER, log=log)
    runner.iterate()
    rec = log.records[0]
    assert rec["late"] is True and rec["raw"] is None and rec["probabilities"] is None and rec["latency_ms"] is None


def test_thread_stops_promptly():
    clock = ActiveClock()  # never started → runner idles
    stats = DecisionStats(clock.now)
    runner = ComputerRunner(GameConfig(width=12, height=8), worker=FakeWorker(FakeTime()), clock=clock, stats=stats)
    runner.start()
    runner.stop()
    assert runner.join(1.0)
