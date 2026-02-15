# Kalacode

A minimal, well-organized coding agent with Azure OpenAI support - inspired by nanocode.

## Features

- Clean, modular architecture
- OpenAI SDK with Azure OpenAI support
- File manipulation tools (read, write, edit)
- Search capabilities (glob, grep)
- Shell command execution
- Interactive REPL interface
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
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py          # Tool base classes
│   │   ├── file_tools.py    # Read, write, edit
│   │   ├── search_tools.py  # Glob, grep
│   │   └── shell_tools.py   # Bash execution
│   └── ui/
│       ├── __init__.py
│       └── display.py       # Terminal UI
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## Installation

### 1. Clone and setup

```bash
git clone <your-repo-url>
cd kalacode
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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

### 3. Install the package (optional)

```bash
pip install -e .
```

## Usage

### Run directly:

```bash
python -m kalacode
```

### Or if installed:

```bash
kalacode
```

### Available commands:

- `/q` or `/quit` or `exit` - Exit the application
- `/c` - Clear conversation history
- `/stats` - Show short-term memory stats
- `/memory show` - Display long-term memory markdown file
- `/memory clear` - Reset long-term memory markdown file

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key (Azure or OpenAI) | - |
| `OPENAI_BASE_URL` | Base URL (set for Azure, unset for OpenAI) | None |
| `OPENAI_MODEL` | Model/deployment name | `gpt-4` |

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
