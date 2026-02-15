"""Long-term memory persisted as a standalone markdown file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LongTermMemory:
    """Markdown-backed long-term memory store."""

    file_path: Path
    max_summary_chars: int = 2000
    max_entries: int = 500

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Create the markdown file with a minimal schema if missing."""
        if self.file_path.exists():
            return

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        initial = (
            "# Kalacode Long-Term Memory\n\n"
            "This file stores persistent memory across sessions.\n\n"
            "## Notes\n"
        )
        self.file_path.write_text(initial, encoding="utf-8")

    def read(self) -> str:
        """Read full markdown memory."""
        try:
            return self.file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._ensure_initialized()
            return self.file_path.read_text(encoding="utf-8")

    def get_summary(self) -> str:
        """
        Return a bounded summary string for prompt injection.

        Keeps the most recent part of the file because new notes are appended.
        """
        text = self.read().strip()
        if len(text) <= self.max_summary_chars:
            return text
        return text[-self.max_summary_chars :]

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a concise conversation note as markdown bullet entries."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        user_line = self._one_line(user_text, 200)
        assistant_line = self._one_line(assistant_text, 300)
        entry = (
            f"\n### {ts}\n"
            f"- User: {user_line}\n"
            f"- Assistant: {assistant_line}\n"
        )
        self.file_path.write_text(self.read() + entry, encoding="utf-8")
        self._trim_entries()

    def _trim_entries(self) -> None:
        """Trim oldest note blocks when entry count exceeds configured limit."""
        text = self.read()
        marker = "\n### "
        parts = text.split(marker)
        if len(parts) - 1 <= self.max_entries:
            return

        header = parts[0]
        notes = parts[1:]
        kept = notes[-self.max_entries :]
        trimmed = header + "".join(f"{marker}{item}" for item in kept)
        self.file_path.write_text(trimmed, encoding="utf-8")

    @staticmethod
    def _one_line(text: str, max_chars: int) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3] + "..."
