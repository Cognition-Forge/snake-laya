"""JSONL decision log: one line per computer step (input, decision, outcome)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class DecisionLog:
    enabled = True

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        # Line-buffered: every record reaches the OS even if the process is killed.
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    @property
    def closed(self) -> bool:
        return self._file is None


class NullLog:
    enabled = False

    def write(self, record: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass
