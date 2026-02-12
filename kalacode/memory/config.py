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
        )
