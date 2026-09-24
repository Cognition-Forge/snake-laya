import pytest

from snake_laya.config import GameConfig


@pytest.mark.parametrize(
    "tick_ms, computer_tick_ms, computer_tick_s, equal",
    [
        (120, None, 0.12, True),  # unset → human tick
        (120, 120, 0.12, True),  # explicit but equal
        (120, 60, 0.06, False),  # faster computer
        (120, 500, 0.5, False),  # slower computer
        (1, 2, 0.002, False),  # minimum ticks
    ],
)
def test_computer_tick(tick_ms, computer_tick_ms, computer_tick_s, equal):
    cfg = GameConfig(tick_ms=tick_ms, computer_tick_ms=computer_tick_ms)
    assert cfg.tick_s == tick_ms / 1000
    assert cfg.computer_tick_s == pytest.approx(computer_tick_s)
    assert cfg.equal_ticks is equal
