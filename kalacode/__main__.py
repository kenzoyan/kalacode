#!/usr/bin/env python3
"""Kalacode - A minimal coding agent."""

import argparse
import sys
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    # Try current directory first
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
    else:
        # Try script directory
        script_dir = Path(__file__).parent.parent
        env_path = script_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment from: {env_path}")
except ImportError:
    print(
        "Warning: python-dotenv not installed. Please set environment variables manually."
    )
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

from kalacode.core import create_client_from_env
from kalacode.core.agent import Agent, AgentRunner
from kalacode.tools import get_default_tools, ToolRegistry
from kalacode.ui import Display


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Kalacode - A minimal coding agent")
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "--provider",
        choices=["azure", "openai"],
        help="LLM provider (overrides LLM_PROVIDER env var)",
    )

    args = parser.parse_args()

    # Initialize components
    display = Display(use_colors=not args.no_color)

    try:
        # Create LLM client
        llm_client = create_client_from_env()

        # Create tool registry
        tools = get_default_tools()
        tool_registry = ToolRegistry(tools)

        # Create agent
        agent = Agent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            display=display,
        )

        # Run agent
        runner = AgentRunner(agent, display)
        runner.run()

    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        display.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
