"""Agent orchestration and conversation management."""

import json
import os
from typing import Any, Dict, List, Optional

from ..core import LLMClient
from ..tools import ToolRegistry
from ..ui import Display


class Agent:
    """Main agent that orchestrates LLM interactions and tool use."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        display: Display,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.display = display
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return f"You are a helpful coding assistant. Current working directory: {os.getcwd()}"

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def process_user_input(self, user_input: str) -> None:
        """Process user input and run agentic loop."""
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep calling API until no more tool calls
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get response from LLM
            response = self.llm.chat_completion(
                messages=self.messages,
                tools=self.tools.to_openai_schemas(),
            )

            # Handle text content
            if response["content"]:
                self.display.message(response["content"])

            # Handle tool calls
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # No more tool calls, conversation turn complete
                self.messages.append(
                    {"role": "assistant", "content": response["content"]}
                )
                break

            # Add assistant message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": response["content"],
                "tool_calls": [],
            }

            # Execute tools and collect results
            tool_results = []
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                # Preview for display
                arg_values = list(function_args.values())
                arg_preview = str(arg_values[0])[:50] if arg_values else ""
                self.display.tool_call(function_name, arg_preview)

                # Execute tool
                result = self.tools.execute(function_name, function_args)

                # Preview result
                result_lines = result.split("\n")
                preview = result_lines[0][:60]
                if len(result_lines) > 1:
                    preview += f" ... +{len(result_lines) - 1} lines"
                elif len(result_lines[0]) > 60:
                    preview += "..."
                self.display.tool_result(preview)

                # Format for OpenAI API
                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(function_args),
                        },
                    }
                )

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

            # Add assistant message and tool results to conversation
            self.messages.append(assistant_message)
            self.messages.extend(tool_results)

        if iteration >= max_iterations:
            self.display.error("Max iterations reached")


class AgentRunner:
    """Runs the agent REPL loop."""

    def __init__(self, agent: Agent, display: Display):
        self.agent = agent
        self.display = display

    def run(self) -> None:
        """Run the main REPL loop."""
        # Print header
        provider_info = "azure" if self.agent.llm.base_url else "openai"
        self.display.header(
            "kalacode",
            f"{self.agent.llm.model} ({provider_info}) | {os.getcwd()}",
        )
        print()

        while True:
            try:
                self.display.user_prompt()
                user_input = input().strip()
                print(self.display.separator())

                if not user_input:
                    continue

                # Handle commands
                if user_input in ("/q", "/quit", "exit"):
                    break

                if user_input == "/c":
                    self.agent.reset_conversation()
                    self.display.info("Cleared conversation")
                    continue

                # Process user input
                self.agent.process_user_input(user_input)
                print()

            except (KeyboardInterrupt, EOFError):
                print()
                break
            except Exception as err:
                self.display.error(str(err))
