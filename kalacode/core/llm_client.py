"""API client for interacting with LLMs."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env in current directory or parent directories
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try to find .env in project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars


class LLMClient:
    """Client for LLM API interactions supporting both OpenAI and Azure OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key (if None, will use env var)
            base_url: Base URL for API endpoint (for Azure, use full endpoint URL)
            model: Model/deployment name
        """
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,  # None for standard OpenAI, custom URL for Azure
        )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4")
        self.base_url = base_url

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_completion_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """
        Make a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool schemas in OpenAI format
            max_completion_tokens: Maximum tokens in response
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            Response dict from the API (or generator if streaming)
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = tools

        if stream:
            # Return streaming generator
            return self.client.chat.completions.create(**kwargs)
        else:
            # Return parsed response
            response = self.client.chat.completions.create(**kwargs)
            return self._parse_response(response)

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse the API response into a consistent format."""
        choice = response.choices[0]
        message = choice.message

        result = {
            "role": message.role,
            "content": message.content or "",
            "tool_calls": [],
        }

        # Parse tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                result["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )

        return result


def create_client_from_env() -> LLMClient:
    """Create an LLM client based on environment variables."""
    return LLMClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),  # Set this for Azure endpoints
        model=os.environ.get("OPENAI_MODEL", "gpt-4"),
    )
