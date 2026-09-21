import pytest
from helpers import FakeTime, make_board

from snake_laya.clock import ActiveClock
from snake_laya.config import COUNTDOWN_S, RESPAWN_S, GameConfig, Mode
from snake_laya.game import START_LEN, Board, Dir
from snake_laya.match import HumanController, InputQueue, Match, Phase

U, D, L, R = Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT


@pytest.mark.parametrize(
    "heading, keys, accepted, pending",
    [
        (R, [U], [True], (U,)),
        (R, [L], [False], ()),  # reverse of heading
        (R, [R], [False], ()),  # same as heading
        (R, [U, L], [True, True], (U, L)),  # fast double turn within a tick
        (R, [U, D], [True, False], (U,)),  # reverse of last *queued*
        (R, [U, U], [True, False], (U,)),  # duplicate of last queued
        (R, [U, L, D], [True, True, False], (U, L)),  # capacity 2
        (U, [L, D], [True, True], (L, D)),  # D is valid after L even though heading is U
    ],
)
def test_input_queue(heading, keys, accepted, pending):
    q = InputQueue()
    assert [q.push(k, heading) for k in keys] == accepted
    assert q.pending == pending


def controller(t, body=None, heading=R, food=(0, 0), tick_s=0.1):
    clock = ActiveClock(t)
    clock.start()
    board = make_board(body, heading, food) if body else Board(12, 8, seed=1)
    return HumanController(board, tick_s, clock.now), clock


def test_human_steps_once_per_tick_and_consumes_one_turn():
    t = FakeTime()
    ctl, clock = controller(t, [(5, 4), (4, 4), (3, 4)], food=(0, 0))
    assert ctl.tick(clock.now()) is None  # schedules first tick
    ctl.key(U)
    ctl.key(L)
    t.advance(0.1)
    ctl.tick(clock.now())
    assert ctl.board.head == (5, 3) and ctl.queue.pending == (L,)
    t.advance(0.05)
    assert ctl.tick(clock.now()) is None  # mid-tick
    t.advance(0.05)
    ctl.tick(clock.now())
    assert ctl.board.head == (4, 3)
    assert (ctl.stats.moves, ctl.stats.turns) == (2, 2)


def test_long_stall_does_not_burst_steps():
    t = FakeTime()
    ctl, clock = controller(t, [(5, 4), (4, 4), (3, 4)])
    ctl.tick(clock.now())
    t.advance(1.0)  # ten ticks late
    ctl.tick(clock.now())
    ctl.tick(clock.now())
    assert ctl.board.steps == 1


def test_human_death_respawns_after_active_second():
    t = FakeTime()
    ctl, clock = controller(t, [(11, 4), (10, 4), (9, 4)], R)
    ctl.tick(clock.now())
    ctl.key(U)
    ctl.queue.clear()  # ensure straight into the wall
    t.advance(0.1)
    assert ctl.tick(clock.now()).died
    assert not ctl.key(U)  # dead: input ignored
    clock.pause()
    t.advance(5)
    ctl.tick(clock.now())
    assert not ctl.board.alive  # pause freezes the respawn timer
    clock.resume()
    t.advance(RESPAWN_S)
    ctl.tick(clock.now())
    assert ctl.board.alive and len(ctl.board.body) == START_LEN and ctl.board.deaths == 1


def build_match(t, *, mode=Mode.SYNC, duration=10.0):
    clock = ActiveClock(t)
    resets = []
    cfg = GameConfig(width=12, height=8, tick_ms=100, duration_s=duration, seed=5, mode=mode)
    m = Match(cfg, clock, on_reset=lambda seed, mode: resets.append((seed, mode)), wall=t)
    return m, clock, resets


def start(m, t):
    m.ready()
    t.advance(COUNTDOWN_S)
    m.update()
    assert m.phase is Phase.RUNNING


def test_lifecycle_loading_countdown_running_over():
    t = FakeTime()
    m, clock, resets = build_match(t, duration=1.0)
    assert m.phase is Phase.LOADING
    m.ready()
    assert m.phase is Phase.COUNTDOWN and m.countdown_left == 3 and resets == [(5, Mode.SYNC)]
    m.key(U)
    assert len(m.human.queue) == 0  # ignored in countdown
    t.advance(COUNTDOWN_S)
    m.update()
    assert m.phase is Phase.RUNNING and clock.running
    t.advance(1.0)
    m.update()
    assert m.phase is Phase.OVER and not clock.running and m.remaining_s == 0.0


def test_pause_freezes_clock_and_human():
    t = FakeTime()
    m, clock, _ = build_match(t)
    start(m, t)
    m.update()
    m.toggle_pause()
    assert m.phase is Phase.PAUSED
    steps = m.human.board.steps
    t.advance(2.0)
    m.update()
    m.key(U)
    assert m.human.board.steps == steps and clock.now() == 0.0 and len(m.human.queue) == 0
    m.toggle_pause()
    assert m.phase is Phase.RUNNING and clock.running


def test_restart_resets_everything():
    t = FakeTime()
    m, clock, resets = build_match(t)
    start(m, t)
    for _ in range(5):
        t.advance(0.1)
        m.update()
    m.key(U)
    old_human = m.human
    m.restart()
    assert m.phase is Phase.COUNTDOWN
    assert m.human is not old_human and m.human.board.steps == 0 and len(m.human.queue) == 0
    assert clock.now() == 0.0 and not clock.running
    assert m.human.board.food == Board(12, 8, seed=5).food  # re-seeded identically
    assert resets[-1] == (5, Mode.SYNC)


@pytest.mark.parametrize("phase_setup", ["loading", "error"])
def test_restart_ignored_when_not_playable(phase_setup):
    t = FakeTime()
    m, _, resets = build_match(t)
    if phase_setup == "error":
        m.fail("boom")
    m.restart()
    assert m.phase is (Phase.LOADING if phase_setup == "loading" else Phase.ERROR)
    assert resets == []


def test_mode_switch_confirm():
    t = FakeTime()
    m, clock, resets = build_match(t)
    start(m, t)
    m.request_mode_switch()
    assert m.phase is Phase.CONFIRM_MODE and not clock.running
    m.confirm_mode_switch()
    assert m.mode is Mode.MAX and m.phase is Phase.COUNTDOWN and not m.ranked
    assert resets[-1] == (5, Mode.MAX)


@pytest.mark.parametrize("before", [Phase.RUNNING, Phase.PAUSED, Phase.OVER, Phase.COUNTDOWN])
def test_mode_switch_cancel_restores_phase(before):
    t = FakeTime()
    m, clock, resets = build_match(t, duration=1.0)
    if before is Phase.COUNTDOWN:
        m.ready()
        t.advance(2.9)  # nearly done
    else:
        start(m, t)
        if before is Phase.PAUSED:
            m.toggle_pause()
        elif before is Phase.OVER:
            t.advance(1.0)
            m.update()
    n = len(resets)
    m.request_mode_switch()
    t.advance(5.0)
    m.cancel_mode_switch()
    assert m.phase is before and m.mode is Mode.SYNC and len(resets) == n
    assert clock.running is (before is Phase.RUNNING)
    if before is Phase.COUNTDOWN:
        assert m.countdown_left == 3  # countdown restarted, not skipped


def test_mode_switch_ignored_while_loading():
    t = FakeTime()
    m, _, _ = build_match(t)
    m.request_mode_switch()
    assert m.phase is Phase.LOADING
    m.confirm_mode_switch()
    m.cancel_mode_switch()
    assert m.phase is Phase.LOADING and m.mode is Mode.SYNC


def test_fail_pauses_clock():
    t = FakeTime()
    m, clock, _ = build_match(t)
    start(m, t)
    m.fail("Inference failed")
    assert m.phase is Phase.ERROR and m.error == "Inference failed" and not clock.running
    m.toggle_pause()
    assert m.phase is Phase.ERROR


@pytest.mark.parametrize("mode, expected", [(Mode.SYNC, "computer"), (Mode.MAX, None)])
def test_standing(mode, expected):
    t = FakeTime()
    m, _, _ = build_match(t, mode=mode)
    computer = Board(12, 8, seed=5)
    computer.total_food = 3
    assert m.standing(computer.view()) == expected
