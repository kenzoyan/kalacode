"""Short-term memory management with token tracking and sliding window."""

import json
from typing import Any, Dict, List, Optional


class TokenCounter:
    """Simple token counter using rough estimation."""

    @staticmethod
    def count_message(message: Dict[str, Any]) -> int:
        """
        Estimate token count for a single message.

        Uses rough heuristic: ~4 characters = 1 token
        This is approximate but avoids external dependencies.
        """
        text = message.get("content", "")

        # Count content tokens
        if text:
            content_tokens = len(text) // 4
        else:
            content_tokens = 0

        # Count tool call tokens if present
        tool_tokens = 0
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                # Function name
                tool_tokens += len(tool_call.get("function", {}).get("name", "")) // 4
                # Function arguments (JSON string)
                args = tool_call.get("function", {}).get("arguments", "")
                tool_tokens += len(args) // 4

        # Role overhead (small, ~3-5 tokens per message)
        overhead = 5

        return content_tokens + tool_tokens + overhead

    @classmethod
    def count_messages(cls, messages: List[Dict[str, Any]]) -> int:
        """Count total tokens across multiple messages."""
        return sum(cls.count_message(msg) for msg in messages)


class ShortTermMemory:
    """
    Manages short-term conversation memory with sliding window.

    Features:
    - Token counting to prevent context overflow
    - Sliding window to keep recent messages
    - Truncation when limits are exceeded
    """

    def __init__(self, max_tokens: int = 100_000, max_messages: int = 20):
        """
        Initialize short-term memory.

        Args:
            max_tokens: Maximum tokens to keep in context
            max_messages: Maximum number of recent messages to keep
        """
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []
        self._token_counter = TokenCounter()

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to memory."""
        self.messages.append(message)
        self._maybe_truncate()

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Add multiple messages to memory."""
        self.messages.extend(messages)
        self._maybe_truncate()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in current context window."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages = []

    def count_tokens(self) -> int:
        """Count total tokens in current context."""
        return self._token_counter.count_messages(self.messages)

    def count_messages(self) -> int:
        """Count number of messages in memory."""
        return len(self.messages)

    def _maybe_truncate(self) -> None:
        """Truncate old messages if limits exceeded."""
        # Check message count limit
        if len(self.messages) > self.max_messages:
            messages_to_remove = len(self.messages) - self.max_messages
            self.messages = self.messages[messages_to_remove:]

        # Check token limit (but keep at least the most recent message)
        current_tokens = self.count_tokens()
        while current_tokens > self.max_tokens and len(self.messages) > 1:
            # Remove oldest message (but keep at least 1)
            self.messages.pop(0)
            current_tokens = self.count_tokens()

        # Note: If a single message exceeds max_tokens, we still keep it
        # to avoid empty context. This is by design.
        self._sanitize_tool_message_sequence()

    def _sanitize_tool_message_sequence(self) -> None:
        """
        Remove invalid/orphan tool messages.

        OpenAI chat requires each `role=tool` message to come after an
        assistant message that contains a matching `tool_calls` entry.
        Truncation can break this linkage, so we drop orphaned tool messages.
        """
        sanitized: List[Dict[str, Any]] = []
        open_tool_call_ids: set[str] = set()

        for message in self.messages:
            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                sanitized.append(message)
                for tool_call in message.get("tool_calls", []):
                    tool_call_id = tool_call.get("id")
                    if tool_call_id:
                        open_tool_call_ids.add(tool_call_id)
                continue

            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id in open_tool_call_ids:
                    sanitized.append(message)
                    open_tool_call_ids.remove(tool_call_id)
                # Drop orphan tool message silently.
                continue

            sanitized.append(message)

        self.messages = sanitized

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "message_count": len(self.messages),
            "token_count": self.count_tokens(),
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens,
            "utilization": {
                "messages": f"{len(self.messages)}/{self.max_messages}",
                "tokens": f"{self.count_tokens()}/{self.max_tokens}",
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"ShortTermMemory("
            f"messages={stats['message_count']}/{self.max_messages}, "
            f"tokens={stats['token_count']}/{self.max_tokens})"
        )
