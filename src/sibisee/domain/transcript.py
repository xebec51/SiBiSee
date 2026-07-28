from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranscriptBuilder:
    tokens: list[str] = field(default_factory=list)

    def append(self, token: str | None) -> None:
        if token:
            self.tokens.append(token)

    def undo(self) -> str | None:
        if not self.tokens:
            return None
        return self.tokens.pop()

    def clear(self) -> None:
        self.tokens.clear()

    @property
    def text(self) -> str:
        return " ".join(self.tokens).strip()

    @property
    def last_token(self) -> str | None:
        return self.tokens[-1] if self.tokens else None
