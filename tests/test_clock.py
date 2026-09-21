import threading

import pytest
from helpers import FakeTime

from snake_laya.clock import ActiveClock


@pytest.mark.parametrize(
    "script, expected_now, expected_running",
    [
        ([], 0.0, False),  # starts stopped
        ([("advance", 5)], 0.0, False),  # time does not count before start
        ([("start",), ("advance", 2)], 2.0, True),
        ([("start",), ("advance", 2), ("pause",), ("advance", 10)], 2.0, False),  # frozen while paused
        ([("start",), ("advance", 2), ("pause",), ("advance", 10), ("resume",), ("advance", 1)], 3.0, True),
        ([("start",), ("advance", 1), ("start",), ("advance", 1)], 2.0, True),  # double start is a no-op
        ([("pause",), ("pause",), ("advance", 3)], 0.0, False),  # pause while stopped is a no-op
        ([("start",), ("advance", 4), ("reset",)], 0.0, False),  # reset zeroes and stops
        ([("start",), ("advance", 4), ("reset",), ("start",), ("advance", 1)], 1.0, True),
    ],
)
def test_active_clock(script, expected_now, expected_running):
    t = FakeTime()
    clock = ActiveClock(t)
    for op, *arg in script:
        if op == "advance":
            t.advance(arg[0])
        else:
            getattr(clock, op)()
    assert clock.now() == pytest.approx(expected_now)
    assert clock.running is expected_running


def test_active_clock_concurrent_access_is_consistent():
    clock = ActiveClock()
    stop = threading.Event()

    def toggler():
        while not stop.is_set():
            clock.start()
            clock.pause()

    thread = threading.Thread(target=toggler)
    thread.start()
    try:
        last = 0.0
        for _ in range(2000):
            now = clock.now()
            assert now >= last  # monotonic despite concurrent pause/resume
            last = now
    finally:
        stop.set()
        thread.join()
