"""Agent orchestration and conversation management."""

import json
import os
from typing import Any, Dict, List, Optional

from ..core import LLMClient
from ..memory import LongTermMemory, MemoryConfig, ShortTermMemory
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
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.display = display
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Initialize short-term memory
        self.memory_config = memory_config or MemoryConfig.from_env()
        if self.memory_config.enable_stm:
            self.stm = ShortTermMemory(
                max_tokens=self.memory_config.max_context_tokens,
                max_messages=self.memory_config.max_recent_messages,
            )
        else:
            self.stm = None
        if self.memory_config.enable_ltm:
            self.ltm = LongTermMemory(
                file_path=self.memory_config.ltm_file_path,
                max_summary_chars=self.memory_config.ltm_max_summary_chars,
                max_entries=self.memory_config.ltm_max_entries,
            )
        else:
            self.ltm = None

        # Keep messages list for backward compatibility
        self.messages: List[Dict[str, Any]] = []

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return (
            "You are a helpful coding assistant.\n"
            f"Current working directory: {os.getcwd()}\n\n"
            "Operate in ReAct style:\n"
            "1) Think briefly about the next best step.\n"
            "2) Act by calling a tool when needed.\n"
            "3) Observe tool output and iterate.\n"
            "4) Respond only when you have enough evidence.\n\n"
            "Rules:\n"
            "- Prefer tool calls over guessing for file/system facts.\n"
            "- Keep user-facing reasoning concise; do not dump long internal deliberation.\n"
            "- If a tool fails, adjust and retry with a better action.\n"
            "- For code changes, verify with a relevant command before final response.\n"
            "- When done, provide a direct, actionable answer."
        )

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.messages = []
        if self.stm:
            self.stm.clear()

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get short-term memory statistics."""
        if self.stm:
            return self.stm.get_stats()
        return None

    def _get_context_messages(self) -> List[Dict[str, Any]]:
        """Get messages for API call (uses STM if enabled, otherwise full history)."""
        if self.stm:
            return self.stm.get_messages()
        return self.messages

    def _build_system_prompt(self) -> str:
        """Build dynamic system prompt including long-term memory summary."""
        prompt = self.system_prompt
        if self.ltm:
            ltm_summary = self.ltm.get_summary()
            if ltm_summary:
                prompt += (
                    "\n\nLong-term memory (markdown summary, may be partial):\n"
                    f"{ltm_summary}"
                )
        return prompt

    def _build_api_messages(self) -> List[Dict[str, Any]]:
        """Compose messages sent to the LLM."""
        return [{"role": "system", "content": self._build_system_prompt()}] + self._get_context_messages()

    def _add_to_memory(self, message: Dict[str, Any]) -> None:
        """Add message to memory (both STM and messages list)."""
        self.messages.append(message)
        if self.stm:
            self.stm.add_message(message)

    def _append_to_ltm(self, user_input: str, assistant_output: str) -> None:
        """Persist completed turn to long-term markdown memory."""
        if not self.ltm:
            return
        if not assistant_output.strip():
            return
        self.ltm.append_turn(user_text=user_input, assistant_text=assistant_output)

    def process_user_input(self, user_input: str) -> None:
        """Process user input and run agentic loop."""
        # Add user message
        self._add_to_memory({"role": "user", "content": user_input})

        # Agentic loop: keep calling API until no more tool calls
        max_iterations = 10
        iteration = 0
        final_assistant_output = ""

        while iteration < max_iterations:
            iteration += 1

            # Get response from LLM with streaming
            # Use STM context if available, otherwise use full messages
            context_messages = self._build_api_messages()
            stream = self.llm.chat_completion(
                messages=context_messages,
                tools=self.tools.to_openai_schemas(),
                stream=True,
            )

            # Process streaming response
            full_content = ""
            tool_calls = []
            current_tool_call = None

            # Show streaming prefix
            print(
                f"\n{self.display.colors.CYAN}⏺{self.display.colors.RESET} ",
                end="",
                flush=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Stream text content
                if delta.content:
                    self.display.stream_text(delta.content)
                    full_content += delta.content

                # Collect tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        if tc_chunk.index is not None:
                            # New tool call
                            if (
                                current_tool_call is None
                                or tc_chunk.index != current_tool_call.get("index")
                            ):
                                if current_tool_call is not None:
                                    tool_calls.append(current_tool_call)
                                current_tool_call = {
                                    "index": tc_chunk.index,
                                    "id": tc_chunk.id or "",
                                    "type": tc_chunk.type or "function",
                                    "function": {
                                        "name": tc_chunk.function.name or "",
                                        "arguments": tc_chunk.function.arguments or "",
                                    },
                                }
                            else:
                                # Continue existing tool call
                                if tc_chunk.function.name:
                                    current_tool_call["function"]["name"] += (
                                        tc_chunk.function.name
                                    )
                                if tc_chunk.function.arguments:
                                    current_tool_call["function"]["arguments"] += (
                                        tc_chunk.function.arguments
                                    )

            # Add last tool call if exists
            if current_tool_call is not None:
                tool_calls.append(current_tool_call)

            print()  # Newline after streaming

            # Check for tool calls
            if not tool_calls:
                # No more tool calls, conversation turn complete
                self._add_to_memory({"role": "assistant", "content": full_content})
                final_assistant_output = full_content
                break

            # Add assistant message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": full_content,
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
            self._add_to_memory(assistant_message)
            for tool_result in tool_results:
                self._add_to_memory(tool_result)

        if iteration >= max_iterations:
            self.display.error("Max iterations reached")
        else:
            self._append_to_ltm(user_input=user_input, assistant_output=final_assistant_output)


class AgentRunner:
    """Runs the agent REPL loop."""

    def __init__(self, agent: Agent, display: Display):
        self.agent = agent
        self.display = display

    def run(self) -> None:
        """Run the main REPL loop."""
        # Show landing page with ASCII logo
        provider_info = "azure" if self.agent.llm.base_url else "openai"
        self.display.show_landing_page(self.agent.llm.model, provider_info)

        # Show memory info if STM is enabled
        if self.agent.stm:
            stats = self.agent.get_memory_stats()
            if stats:
                self.display.info(
                    f"Short-term memory: {stats['utilization']['messages']}, "
                    f"{stats['utilization']['tokens']}",
                    color="cyan",
                )
        if self.agent.ltm:
            self.display.info(
                f"Long-term memory file: {self.agent.ltm.file_path}",
                color="cyan",
            )

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

                if user_input == "/stats":
                    stats = self.agent.get_memory_stats()
                    if stats:
                        self.display.info(
                            f"Memory: {stats['message_count']} messages, "
                            f"{stats['token_count']} tokens "
                            f"(max: {stats['max_messages']} msgs, {stats['max_tokens']} tokens)"
                        )
                    else:
                        self.display.info("Short-term memory disabled")
                    continue

                if user_input == "/memory show":
                    if not self.agent.ltm:
                        self.display.info("Long-term memory disabled", color="yellow")
                        continue
                    self.display.info(
                        f"Long-term memory file: {self.agent.ltm.file_path}",
                        color="cyan",
                    )
                    print(self.agent.ltm.read())
                    continue

                if user_input == "/memory clear":
                    if not self.agent.ltm:
                        self.display.info("Long-term memory disabled", color="yellow")
                        continue
                    self.agent.ltm.clear()
                    self.display.info("Long-term memory cleared", color="cyan")
                    continue

                # Process user input
                self.agent.process_user_input(user_input)
                print()

            except (KeyboardInterrupt, EOFError):
                print()
                break
            except Exception as err:
                self.display.error(str(err))
