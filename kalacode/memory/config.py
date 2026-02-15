"""Memory system configuration."""

import os
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for memory system."""

    # Short-term memory settings
    max_context_tokens: int = 100_000
    max_recent_messages: int = 20
    enable_stm: bool = True
    enable_ltm: bool = True
    ltm_file_path: str = ".kalacode_memory.md"
    ltm_max_summary_chars: int = 2000
    ltm_max_entries: int = 500

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables."""
        return cls(
            max_context_tokens=int(
                os.environ.get("KALACODE_MAX_CONTEXT_TOKENS", "100000")
            ),
            max_recent_messages=int(
                os.environ.get("KALACODE_MAX_RECENT_MESSAGES", "20")
            ),
            enable_stm=os.environ.get("KALACODE_ENABLE_STM", "true").lower()
            in ("true", "1", "yes"),
            enable_ltm=os.environ.get("KALACODE_ENABLE_LTM", "true").lower()
            in ("true", "1", "yes"),
            ltm_file_path=os.environ.get("KALACODE_LTM_FILE", ".kalacode_memory.md"),
            ltm_max_summary_chars=int(
                os.environ.get("KALACODE_LTM_MAX_SUMMARY_CHARS", "2000")
            ),
            ltm_max_entries=int(os.environ.get("KALACODE_LTM_MAX_ENTRIES", "500")),
        )
