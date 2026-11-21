"""
Coco's personal MCP server: typed operations on the long-term memory file.

Unlike the generic filesystem server, this one never takes a file path from the model.
The memory file is fixed when the server starts (by mcp_host.py, from mcp_servers.json):

    python personal_server.py --memory-file path/to/memory.md

Tools:
    list_memories  list saved memories with their categories
    remember       save one preference, fact, or decision
    forget         remove one specifically matched memory (ambiguity is an error, never a guess)
    current_time   the local date and time (no file access)

Failures raise ToolError, so the client receives an MCP error result (is_error=True),
never a "successful" result that only says something went wrong.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

## The script's own folder is on sys.path when run directly, so this finds memory_store.py
import memory_store


def build_server(memory_file: Path) -> MCPServer:
    memory_file = memory_file.resolve()
    ## log_level WARNING: the SDK logs failed tool calls (including their text) at INFO
    server = MCPServer("coco-personal", version="1.0.0", log_level="WARNING")

    def checked(result: str) -> str:
        if result.startswith("Error:"):
            raise ToolError(result.removeprefix("Error:").strip())
        return result

    def storage(action):
        ## Turn disk problems into visible tool errors instead of crashes or silent failures
        try:
            return action()
        except OSError as e:
            raise ToolError(f"Could not access the memory file: {e.strerror or e}") from None

    @server.tool(description="List everything saved in long-term memory, grouped by category.")
    def list_memories() -> str:
        items = storage(lambda: memory_store.by_section(memory_file))
        if not items:
            return "No memories saved yet."
        return "\n".join(f"{category}: {memory}" for category, memory in items)

    @server.tool(description="Save a lasting preference, fact, or decision about the user to long-term memory.")
    def remember(fact: str, category: Literal["Preferences", "Facts", "Decisions"] = "Facts") -> str:
        return checked(storage(lambda: memory_store.remember(memory_file, fact, category)))

    @server.tool(description="Remove a wrong or outdated memory from long-term memory.")
    def forget(text: str) -> str:
        return checked(storage(lambda: memory_store.forget(memory_file, text)))

    @server.tool(description="Return the current local date and time.")
    def current_time() -> str:
        return datetime.now().isoformat()

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--memory-file", required=True, type=Path)
    args = parser.parse_args()
    if not args.memory_file.parent.is_dir():
        ## Fail at startup, where the host reports it, rather than on the first tool call
        sys.exit(f"Memory folder does not exist: {args.memory_file.parent}")
    build_server(args.memory_file).run("stdio")


if __name__ == "__main__":
    main()
