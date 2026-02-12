"""Search and file discovery tools."""

import glob as globlib
import os
import re
from typing import Any, Dict
from .base import Tool


class GlobTool(Tool):
    """Find files by pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "Find files by pattern, sorted by modification time"

    @property
    def parameters(self) -> Dict[str, str]:
        return {"pat": "string", "path": "string?"}

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
            files = globlib.glob(pattern, recursive=True)
            files = sorted(
                files,
                key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
                reverse=True,
            )
            return "\n".join(files) or "none"
        except Exception as err:
            return f"error: {err}"


class GrepTool(Tool):
    """Search files for regex pattern."""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search files for regex pattern"

    @property
    def parameters(self) -> Dict[str, str]:
        return {"pat": "string", "path": "string?"}

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            pattern = re.compile(args["pat"])
            hits = []
            search_path = args.get("path", ".") + "/**"

            for filepath in globlib.glob(search_path, recursive=True):
                try:
                    with open(filepath) as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.search(line):
                                hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
                except Exception:
                    pass

            return "\n".join(hits[:50]) or "none"
        except Exception as err:
            return f"error: {err}"
