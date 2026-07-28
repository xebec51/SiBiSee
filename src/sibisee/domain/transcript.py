from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


@dataclass
class TranscriptBuilder:
    tokens: list[str] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def append(self, token: str | None) -> None:
        if token:
            with self._lock:
                self.tokens.append(token)

    def append_once(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            if self.tokens and self.tokens[-1] == token:
                return False
            self.tokens.append(token)
            return True

    def undo(self) -> str | None:
        with self._lock:
            if not self.tokens:
                return None
            return self.tokens.pop()

    def clear(self) -> None:
        with self._lock:
            self.tokens.clear()

    def snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self.tokens)

    @property
    def text(self) -> str:
        return " ".join(self.snapshot()).strip()

    @property
    def last_token(self) -> str | None:
        snapshot = self.snapshot()
        return snapshot[-1] if snapshot else None
