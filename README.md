# Kalacode

A minimal, well-organized coding agent with Azure OpenAI support - inspired by nanocode.

## Features

- Clean, modular architecture
- OpenAI SDK with Azure OpenAI support
- ReAct-style tool-using loop (Think -> Act -> Observe -> Respond)
- Short-term memory with truncation safeguards for tool-call sequencing
- Long-term memory persisted in a standalone markdown file
- Durable memory extraction (facts/preferences/decisions only)
- File manipulation tools (read, write, edit)
- Search capabilities (glob, grep)
- Shell command execution
- Interactive REPL interface
- Friendly slash commands with help and Tab completion
- Colored terminal output

## Project Structure

```
kalacode/
├── kalacode/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm_client.py    # OpenAI/Azure client
│   │   └── agent.py         # Agent orchestration
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── config.py        # Memory configuration
│   │   ├── short_term.py    # Sliding-window short-term memory
│   │   └── long_term.py     # Markdown long-term memory
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py          # Tool base classes
│   │   ├── file_tools.py    # Read, write, edit
│   │   ├── search_tools.py  # Glob, grep
│   │   └── shell_tools.py   # Bash execution
│   └── ui/
│       ├── __init__.py
│       └── display.py       # Terminal UI
├── .kalacode_memory.md
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### 1. Clone and setup

```bash
git clone <your-repo-url>
cd kalacode
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

**For Azure OpenAI:**
```env
OPENAI_API_KEY=your-azure-api-key
OPENAI_BASE_URL=https://your-resource.cognitiveservices.azure.com/openai/v1/
OPENAI_MODEL=gpt-4
```

Example with real Azure endpoint:
```env
OPENAI_API_KEY=abc123def456
OPENAI_BASE_URL=https://your-resource.cognitiveservices.azure.com/openai/v1/
OPENAI_MODEL=gpt-5.1
```

**For Standard OpenAI:**
```env
OPENAI_API_KEY=sk-your-openai-api-key
# OPENAI_BASE_URL is not needed for standard OpenAI (leave unset)
OPENAI_MODEL=gpt-4
```

## Usage

### Run directly:

```bash
python -m kalacode
```

### Available commands:

- `/help` or `/commands` - Show available commands
- `/q` or `/quit` or `exit` - Exit the application
- `/c` - Clear conversation history
- `/stats` - Show short-term memory stats
- `/memory` - Show memory subcommands
- `/memory show` - Display long-term memory markdown file
- `/memory clear` - Reset long-term memory markdown file

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key (Azure or OpenAI) | - |
| `OPENAI_BASE_URL` | Base URL (set for Azure, unset for OpenAI) | None |
| `OPENAI_MODEL` | Model/deployment name | `gpt-4` |
| `KALACODE_ENABLE_STM` | Enable short-term memory | `true` |
| `KALACODE_MAX_CONTEXT_TOKENS` | STM token budget | `100000` |
| `KALACODE_MAX_RECENT_MESSAGES` | STM message window size | `20` |
| `KALACODE_ENABLE_LTM` | Enable long-term markdown memory | `true` |
| `KALACODE_LTM_FILE` | LTM markdown file path | `.kalacode_memory.md` |
| `KALACODE_LTM_MAX_SUMMARY_CHARS` | Max LTM chars injected in prompt | `2000` |
| `KALACODE_LTM_MAX_ENTRIES` | Max timestamped LTM entries retained | `500` |

## Memory Behavior

- Short-term memory keeps recent context and sanitizes invalid tool-message sequences after truncation.
- Long-term memory is stored in markdown and injected as bounded context.
- Only durable items are saved to LTM: facts, preferences, and decisions.
- Use `/memory show` and `/memory clear` to inspect/reset LTM.

## Available Tools

The agent has access to these tools:

- **read** - Read file contents with line numbers
- **write** - Write content to a file
- **edit** - Replace text in a file
- **glob** - Find files by pattern
- **grep** - Search files for regex patterns
- **bash** - Execute shell commands (30s timeout)

## Example Session

```
kalacode | gpt-4 (azure) | /home/user/project

❯ Create a hello world Python script

⏺ Creating a simple hello world script...

⏺ Write(hello.py)
  ⎿  ok

⏺ I've created hello.py with a simple print statement.

❯ /q
```

## Architecture

### Modular Design

- **core/llm_client.py** - Abstraction for OpenAI/Azure API interactions
- **core/agent.py** - Agent orchestration and conversation management
- **tools/** - Modular tool system with base classes
- **ui/display.py** - Terminal UI with color support

### Extending with New Tools

Create a new tool by extending the `Tool` base class:

```python
from kalacode.tools.base import Tool

class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description of what my tool does"

    @property
    def parameters(self) -> Dict[str, str]:
        return {"param1": "string", "param2": "number?"}

    def execute(self, args: Dict[str, Any]) -> str:
        # Your implementation
        return "result"
```

Register it in `kalacode/tools/__init__.py`.

## License

MIT
