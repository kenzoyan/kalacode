"""Shell execution tools."""

import subprocess
from typing import Any, Dict
from .base import Tool


class BashTool(Tool):
    """Execute shell commands."""

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Run shell command (timeout: 30s)"

    @property
    def parameters(self) -> Dict[str, str]:
        return {"cmd": "string"}

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            proc = subprocess.Popen(
                args["cmd"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            output_lines = []
            try:
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        output_lines.append(line)

                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                output_lines.append("\n(timed out after 30s)")

            return "".join(output_lines).strip() or "(empty)"
        except Exception as err:
            return f"error: {err}"
