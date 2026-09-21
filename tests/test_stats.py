from dataclasses import replace

import pytest
from helpers import FakeTime, make_view

from snake_laya.config import Mode
from snake_laya.stats import DecisionStats, HumanStats, RateWindow, percentile, rank


@pytest.mark.parametrize(
    "values, q, expected",
    [
        ([], 50, None),
        ([5.0], 50, 5.0),
        ([5.0], 95, 5.0),
        ([1.0, 2.0], 50, 1.0),
        ([3.0, 1.0, 2.0], 50, 2.0),  # unsorted input
        (list(map(float, range(1, 101))), 95, 95.0),
        (list(map(float, range(1, 101))), 100, 100.0),
    ],
)
def test_percentile(values, q, expected):
    assert percentile(values, q) == expected


@pytest.mark.parametrize(
    "events, now, rate",
    [
        ([], 5.0, 0.0),
        ([0.0], 0.999, 1.0),
        ([0.0], 1.0, 0.0),  # exactly window_s old → excluded
        ([0.1, 0.5, 0.9], 1.2, 2.0),  # 0.1 aged out
    ],
)
def test_rate_window(events, now, rate):
    window = RateWindow(1.0)
    for t in events:
        window.add(t)
    assert window.rate(now) == rate


def test_empty_view_has_no_division_errors():
    t = FakeTime()
    v = DecisionStats(t).view()
    assert (v.last_ms, v.p50, v.p95, v.mean_dps, v.late_pct, v.override_pct, v.agree_pct, v.mean_sharpness) == (
        None,
    ) * 8
    assert (v.decisions, v.steps, v.stale, v.dps) == (0, 0, 0, 0.0)


def test_counts_and_percentages():
    t = FakeTime()
    s = DecisionStats(t)
    for ms in (10.0, 20.0, 30.0):
        s.record_prediction(ms)
    s.record_stale()
    t.advance(0.5)
    s.record_step(applied=True, late=False, override=False, agree=True, sharpness=0.4)
    s.record_step(applied=True, late=False, override=True, agree=False, sharpness=0.6)
    s.record_step(applied=False, late=True, override=True)
    t.advance(0.5)
    v = s.view()
    assert (v.predictions, v.decisions, v.steps, v.stale, v.late, v.overrides) == (3, 2, 3, 1, 1, 2)
    assert v.last_ms == 30.0 and v.p50 == 20.0 and v.p95 == 30.0
    assert v.late_pct == pytest.approx(100 / 3)
    assert v.override_pct == pytest.approx(200 / 3)
    assert v.agree_pct == 50.0  # late step (agree=None) not counted
    assert v.sharpness == 0.6 and v.mean_sharpness == pytest.approx(0.5)
    assert v.mean_dps == pytest.approx(2.0)  # 2 applied / 1.0 active s
    assert v.dps == 2.0


def test_latency_window_rolls_over():
    s = DecisionStats(FakeTime(), window=200)
    for ms in range(1, 251):
        s.record_prediction(float(ms))
    v = s.view()
    assert v.predictions == 250
    assert v.p50 == 150.0  # window holds 51..250


def test_dps_decays_and_reset_clears():
    t = FakeTime()
    s = DecisionStats(t)
    s.record_step(applied=True, late=False, override=False)
    t.advance(2.0)
    assert s.view().dps == 0.0
    s.reset()
    v = s.view()
    assert (v.decisions, v.steps, v.p50) == (0, 0, None)


def test_human_stats_turn_rate():
    t = FakeTime()
    h = HumanStats(t)
    h.record_step(turned=True)
    h.record_step(turned=False)
    t.advance(0.5)
    h.record_step(turned=True)
    v = h.view()
    assert (v.moves, v.turns, v.turns_per_s) == (3, 2, 2.0)


def board(total, best, deaths):
    return replace(make_view([(5, 3), (4, 3)]), total_food=total, best=best, deaths=deaths)


@pytest.mark.parametrize(
    "human, computer, mode, winner",
    [
        (board(10, 3, 5), board(8, 8, 0), Mode.SYNC, "human"),  # total food decides first
        (board(8, 3, 0), board(8, 5, 4), Mode.SYNC, "computer"),  # then best life
        (board(8, 5, 1), board(8, 5, 2), Mode.SYNC, "human"),  # then fewer deaths
        (board(8, 5, 2), board(8, 5, 2), Mode.SYNC, "draw"),
        (board(0, 0, 0), board(0, 0, 0), Mode.SYNC, "draw"),
        (board(10, 3, 5), board(8, 8, 0), Mode.MAX, None),  # unranked
    ],
)
def test_rank(human, computer, mode, winner):
    assert rank(human, computer, mode) == winner
