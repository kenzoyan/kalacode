"""Base classes for tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, str]:
        """
        Tool parameters schema.
        Format: {"param_name": "type"} where type can be "string", "number", "boolean"
        Add "?" suffix for optional parameters (e.g., "string?")
        """
        pass

    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> str:
        """Execute the tool with given arguments."""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function calling schema."""
        properties = {}
        required = []

        for param_name, param_type in self.parameters.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")

            # Map to JSON Schema types
            json_type = base_type
            if base_type == "number":
                json_type = "integer"

            properties[param_name] = {"type": json_type}

            if not is_optional:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self, tools: list[Tool] = None):
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return f"error: tool '{name}' not found"
        try:
            return tool.execute(args)
        except Exception as err:
            return f"error: {err}"

    def to_openai_schemas(self) -> list[Dict[str, Any]]:
        """Get OpenAI function schemas for all tools."""
        return [tool.to_openai_schema() for tool in self.get_all()]
