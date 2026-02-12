"""Terminal UI utilities."""

import os
import re


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


class Display:
    """Handles terminal display and formatting."""

    def __init__(self, use_colors: bool = True):
        self.colors = Colors() if use_colors else self._no_colors()

    @staticmethod
    def _no_colors():
        """Return a Colors class with empty strings (no colors)."""

        class NoColors:
            RESET = BOLD = DIM = BLUE = CYAN = GREEN = YELLOW = RED = ""

        return NoColors()

    def separator(self) -> str:
        """Get a terminal-width separator line."""
        width = min(os.get_terminal_size().columns, 80)
        return f"{self.colors.DIM}{'─' * width}{self.colors.RESET}"

    def render_markdown(self, text: str) -> str:
        """Render basic markdown formatting."""
        # Bold text
        text = re.sub(
            r"\*\*(.+?)\*\*", f"{self.colors.BOLD}\\1{self.colors.RESET}", text
        )
        return text

    def header(self, title: str, subtitle: str = "") -> None:
        """Print a header."""
        print(f"{self.colors.BOLD}{title}{self.colors.RESET}", end="")
        if subtitle:
            print(f" | {self.colors.DIM}{subtitle}{self.colors.RESET}")
        else:
            print()

    def user_prompt(self, symbol: str = "❯") -> None:
        """Print the user input prompt."""
        print(self.separator())
        print(
            f"{self.colors.BOLD}{self.colors.BLUE}{symbol}{self.colors.RESET} ",
            end="",
            flush=True,
        )

    def message(self, text: str, prefix: str = "⏺", color: str = None) -> None:
        """Print a message with optional color."""
        color_code = (
            getattr(self.colors, color.upper(), "") if color else self.colors.CYAN
        )
        formatted_text = self.render_markdown(text)
        print(f"\n{color_code}{prefix}{self.colors.RESET} {formatted_text}")

    def tool_call(self, tool_name: str, arg_preview: str) -> None:
        """Print a tool call notification."""
        print(
            f"\n{self.colors.GREEN}⏺ {tool_name.capitalize()}{self.colors.RESET}"
            f"({self.colors.DIM}{arg_preview}{self.colors.RESET})"
        )

    def tool_result(self, result_preview: str) -> None:
        """Print a tool result preview."""
        print(f"  {self.colors.DIM}⎿  {result_preview}{self.colors.RESET}")

    def tool_output_line(self, line: str) -> None:
        """Print a line of tool output."""
        print(f"  {self.colors.DIM}│ {line}{self.colors.RESET}", flush=True)

    def error(self, message: str) -> None:
        """Print an error message."""
        print(f"{self.colors.RED}⏺ Error: {message}{self.colors.RESET}")

    def info(self, message: str, color: str = "green") -> None:
        """Print an info message."""
        color_code = getattr(self.colors, color.upper(), self.colors.GREEN)
        print(f"{color_code}⏺ {message}{self.colors.RESET}")
