"""Shared constants and match configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

RESPAWN_S = 1.0
COUNTDOWN_S = 3.0


class Mode(StrEnum):
    SYNC = "sync"  # computer steps on the human tick: competitive
    MAX = "max"  # computer steps as fast as inference allows: telemetry showcase, unranked

    @property
    def other(self) -> Mode:
        return Mode.MAX if self is Mode.SYNC else Mode.SYNC


@dataclass(frozen=True)
class GameConfig:
    width: int = 30
    height: int = 20
    tick_ms: int = 120
    computer_tick_ms: int | None = None  # SYNC computer step interval; None → tick_ms
    duration_s: float = 180.0
    seed: int = 0
    mode: Mode = Mode.SYNC
    safety: bool = True

    @property
    def tick_s(self) -> float:
        return self.tick_ms / 1000.0

    @property
    def computer_tick_s(self) -> float:
        return (self.computer_tick_ms or self.tick_ms) / 1000.0

    @property
    def equal_ticks(self) -> bool:
        """Both snakes step at the same rate: required for a ranked SYNC match."""
        return self.computer_tick_ms in (None, self.tick_ms)
