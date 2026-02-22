"""Long-term memory persisted as a standalone markdown file."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


@dataclass
class LongTermMemory:
    """Markdown-backed long-term memory store."""

    file_path: Path
    max_summary_chars: int = 2000
    max_entries: int = 500
    dedup_threshold: float = 0.82

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Create the markdown file with a minimal schema if missing."""
        if self.file_path.exists():
            return

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(self._initial_template(), encoding="utf-8")

    @staticmethod
    def _initial_template() -> str:
        """Base markdown schema used for initialization and clear."""
        return (
            "# Kalacode Long-Term Memory\n\n"
            "This file stores persistent memory across sessions.\n"
            "Only durable facts, preferences, and decisions are kept.\n\n"
            "## Notes\n"
        )

    def clear(self) -> None:
        """Reset memory file to its initial template."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(self._initial_template(), encoding="utf-8")

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

    def store_items(self, items: list[str]) -> None:
        """Persist a list of pre-extracted memory strings to the markdown file.

        Performs fuzzy deduplication against existing entries before writing.
        Items are stored as plain bullets (no [KIND] prefix).
        Called by Agent after LLM-based extraction.
        """
        if not items:
            return

        existing_texts = self._existing_item_texts()
        unique_items = [
            item for item in items
            if not self._is_fuzzy_duplicate(item, existing_texts)
        ]
        if not unique_items:
            return

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        lines = [f"\n### {ts}"]
        for item in unique_items:
            lines.append(f"- {item}")
        entry = "\n".join(lines) + "\n"
        self.file_path.write_text(self.read() + entry, encoding="utf-8")
        self._trim_entries()

    def _existing_item_texts(self) -> list[str]:
        """Return normalized text of all stored items for fuzzy comparison.

        Handles both legacy tagged format (- [KIND] text) and plain format (- text).
        """
        text = self.read()
        items: list[str] = []

        # Legacy tagged format: - [FACT] text, - [PREFERENCE] text, - [DECISION] text
        for _, item_text in re.findall(
            r"^- \[(FACT|PREFERENCE|DECISION)\] (.+)$", text, re.MULTILINE
        ):
            items.append(" ".join(item_text.lower().split()))

        # New plain format: lines starting with "- " that are not tagged
        for item_text in re.findall(
            r"^- (?!\[(?:FACT|PREFERENCE|DECISION)\])(.+)$", text, re.MULTILINE
        ):
            items.append(" ".join(item_text.lower().split()))

        return items

    def _is_fuzzy_duplicate(self, candidate: str, existing_texts: list[str]) -> bool:
        """Return True if candidate is similar enough to any existing item.

        Uses difflib.SequenceMatcher with normalized strings.
        """
        normalized = " ".join(candidate.lower().split())
        for existing in existing_texts:
            ratio = difflib.SequenceMatcher(None, normalized, existing).ratio()
            if ratio >= self.dedup_threshold:
                return True
        return False

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        """Append only durable memory items extracted from a conversation turn."""
        items = self._extract_durable_items(user_text=user_text, assistant_text=assistant_text)
        if not items:
            return

        existing = self._existing_item_set()
        unique_items = [(kind, text) for kind, text in items if self._item_key(kind, text) not in existing]
        if not unique_items:
            return

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        lines = [f"\n### {ts}"]
        for kind, text in unique_items:
            lines.append(f"- [{kind}] {text}")
        entry = "\n".join(lines) + "\n"
        self.file_path.write_text(self.read() + entry, encoding="utf-8")
        self._trim_entries()

    def _extract_durable_items(self, user_text: str, assistant_text: str) -> list[tuple[str, str]]:
        """
        Extract durable memory from turn content.

        Returns list of (KIND, text) where KIND is FACT/PREFERENCE/DECISION.
        """
        items: list[tuple[str, str]] = []
        for sentence in self._split_sentences(user_text):
            kind = self._classify_sentence(sentence, source="user")
            if kind:
                items.append((kind, self._one_line(sentence, 220)))

        # Assistant content is noisier; keep only explicit decision-like statements.
        for sentence in self._split_sentences(assistant_text):
            kind = self._classify_sentence(sentence, source="assistant")
            if kind == "DECISION":
                items.append((kind, self._one_line(sentence, 220)))

        return items[:8]

    def _classify_sentence(self, text: str, source: str) -> str | None:
        """Classify sentence into durable memory category or None."""
        normalized = " ".join(text.split())
        lowered = normalized.lower()

        if self._is_transient(normalized):
            return None

        decision_patterns = (
            "first work on ",
            "we will ",
            "let's ",
            "decided ",
            "decision ",
            "selected ",
            "choose ",
            "chosen ",
        )
        if any(pat in lowered for pat in decision_patterns):
            return "DECISION"

        # Assistant output is noisy for facts/preferences; keep only explicit decisions.
        if source == "assistant":
            return None

        preference_patterns = (
            "i prefer ",
            "i want ",
            "i'd like ",
            "please ",
            "don't use ",
            "do not use ",
            "always ",
            "never ",
            "use python ",
            "use ",
        )
        if any(pat in lowered for pat in preference_patterns):
            return "PREFERENCE"

        fact_patterns = (
            "my name is ",
            "i am ",
            "i'm ",
            "repo is ",
            "project is ",
            "python 3.",
            "python 3.1",
        )
        if source == "user" and any(pat in lowered for pat in fact_patterns):
            return "FACT"

        return None

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentence-like chunks."""
        cleaned = (text or "").replace("\r", "\n")
        pieces = re.split(r"[\n]+|(?<=[.!?])\s+", cleaned)
        out = []
        for piece in pieces:
            s = piece.strip().strip("-*")
            if s:
                out.append(s)
        return out

    @staticmethod
    def _is_transient(text: str) -> bool:
        """Heuristic filter for non-durable content."""
        lowered = text.lower()
        transient_markers = (
            "?",
            "error:",
            "traceback",
            "http://",
            "https://",
            "`",
            "pip install",
            "running ",
            "done",
            "thanks",
        )
        if len(lowered) < 12:
            return True
        return any(marker in lowered for marker in transient_markers)

    def _existing_item_set(self) -> set[str]:
        """Load normalized keys of already-stored items to avoid duplicates."""
        text = self.read()
        matches = re.findall(r"^- \[(FACT|PREFERENCE|DECISION)\] (.+)$", text, re.MULTILINE)
        return {self._item_key(kind, item) for kind, item in matches}

    @staticmethod
    def _item_key(kind: str, text: str) -> str:
        return f"{kind}:{' '.join(text.lower().split())}"

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
