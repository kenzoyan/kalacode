"""Tools module for kalacode agent."""

from .base import Tool, ToolRegistry
from .file_tools import ReadTool, WriteTool, EditTool
from .search_tools import GlobTool, GrepTool
from .shell_tools import BashTool


def get_default_tools() -> list[Tool]:
    """Get the default set of tools for the agent."""
    return [
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
        BashTool(),
    ]


__all__ = [
    "Tool",
    "ToolRegistry",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "BashTool",
    "get_default_tools",
]
