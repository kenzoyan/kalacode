"""File manipulation tools."""

from typing import Any, Dict
from .base import Tool


class ReadTool(Tool):
    """Read file contents with line numbers."""

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "Read file with line numbers (file path, not directory)"

    @property
    def parameters(self) -> Dict[str, str]:
        return {"path": "string", "offset": "number?", "limit": "number?"}

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            with open(args["path"]) as f:
                lines = f.readlines()

            offset = args.get("offset", 0)
            limit = args.get("limit", len(lines))
            selected = lines[offset : offset + limit]

            return "".join(
                f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected)
            )
        except Exception as err:
            return f"error: {err}"


class WriteTool(Tool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "Write content to file"

    @property
    def parameters(self) -> Dict[str, str]:
        return {"path": "string", "content": "string"}

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            with open(args["path"], "w") as f:
                f.write(args["content"])
            return "ok"
        except Exception as err:
            return f"error: {err}"


class EditTool(Tool):
    """Edit file by replacing old string with new string."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return "Replace old with new in file (old must be unique unless all=true)"

    @property
    def parameters(self) -> Dict[str, str]:
        return {"path": "string", "old": "string", "new": "string", "all": "boolean?"}

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            with open(args["path"]) as f:
                text = f.read()

            old, new = args["old"], args["new"]

            if old not in text:
                return "error: old_string not found"

            count = text.count(old)
            if not args.get("all") and count > 1:
                return f"error: old_string appears {count} times, must be unique (use all=true)"

            replacement = (
                text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
            )

            with open(args["path"], "w") as f:
                f.write(replacement)

            return "ok"
        except Exception as err:
            return f"error: {err}"
