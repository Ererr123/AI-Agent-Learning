# MCP setup and operation

How Coco gets its tools from MCP servers: how to run it, configure it, add or turn off a server, handle secrets, and fix problems. The plan and the reasons behind it are in [phase-3-mcp.md](phase-3-mcp.md).

## How it fits together

```text
agent.py  (conversation loop, unchanged)
   |  OpenAI-style tool schemas / tool calls
mcp_host.py  (MCP client: starts servers, discovers tools, allowlist, routing, errors)
   |  MCP over stdio, one subprocess per server
   +-- notes:    @modelcontextprotocol/server-filesystem@2.2.0, limited to notes/
   +-- personal: personal_server.py, limited to memory.md
```

| Model-facing tool | Server | Server's own tool |
|---|---|---|
| `list_notes` | notes | `list_directory` with `path` fixed to `.` |
| `read_file` | notes | `read_text_file` (`head`/`tail` hidden) |
| `write_file` | notes | `write_file` |
| `remember`, `forget`, `list_memories`, `current_time` | personal | same names |

The model sees only these 7 tools. The filesystem server offers 16 tools, including move, edit and copy. Coco does not expose the other 13, and the host refuses calls to them.

## Requirements

- Python 3.11+ (tested on 3.14) with `pip install -r src/loop/requirements.txt` (`openai`, `mcp==2.3.0`).
- Node.js with `npx` (tested with v24), for the filesystem server. The first start downloads the pinned package. Later starts use the npm cache.
- The local model server, as before (`llama-server ... --jinja`).

## Running

**Double-click [`coco.bat`](../coco.bat)** in the repo root. It:

1. Checks for Python and Node.js, and installs the Python requirements if they're missing.
2. Starts `llama-server` from the `llama-*` folder next to it, in a minimized window. If a server is already answering on port 8080, it uses that one.
3. Waits for the model to load, then runs Coco from the repo root (`notes\`, `memory.md`, `logs\`).
4. Stops the llama-server it started when you type `quit`. A server you started yourself is left running.

The model, port and context size are set at the top of `coco.bat`. If you close the window with the X instead of typing `quit`, llama-server keeps running, and the next launch reuses it.

By hand, from `src/loop`:

```text
python agent.py                              # chat; startup lists each server, its state, and the tools
python -m unittest discover -s tests -v      # host/server tests: no model or accounts needed (~25s)
python evals.py --repeat 3                   # agent evals against the real model
```

At startup Coco prints `MCP server: notes: running (3 tools)` and the same for `personal`. Any problem is printed as a `WARNING:`. If no tools start, Coco stops and does not chat without tools.

## Configuration

Servers are defined in [`src/loop/mcp_servers.json`](../src/loop/mcp_servers.json). That file is checked in and must never contain secrets. Personal additions go in `src/loop/mcp_servers.local.json`, which is git-ignored. A server in the local file with the same name replaces the checked-in one. A new name adds a server.

```json
{
  "servers": {
    "<name>": {
      "command": "npx",
      "args": ["-y", "some-server@1.2.3", "{notes_dir}"],
      "cwd": "{notes_dir}",
      "env": {"API_TOKEN": "${env:MY_API_TOKEN}"},
      "enabled": true,
      "timeout": 30,
      "log_payloads": false,
      "tools": {
        "server_tool_name": {
          "name": "model_facing_name",
          "description": "Shorter text for a small model.",
          "fixed_args": {"path": "."},
          "hide_args": ["optional_arg"],
          "arg_descriptions": {"arg": "Clearer text."}
        }
      }
    }
  }
}
```

- **Placeholders** in `command`, `args`, `cwd` and `env`: `{notes_dir}`, `{memory_file}`, `{python}` (the running interpreter) and `{here}` (`src/loop`).
- **`${env:NAME}`** reads an environment variable when the server starts. If the variable is not set, that server does not start and the warning names the variable. The value is never printed.
- **`tools` is an allowlist.** A tool that is not listed never reaches the model, whatever the server offers. A listed tool the server does not offer produces a warning.
- **`fixed_args`** are set by the host on every call. They are removed from the model's schema, and the model cannot override them.
- **`hide_args`** removes optional arguments from the schema. Hiding a required argument is refused.
- **Duplicate names.** If two servers expose the same model-facing name, both are renamed `<server>_<tool>` and a warning is shown.
- **`log_payloads`** defaults to `false`: tool arguments and results are then written to `logs/` as `[not logged]`. Only `notes` and `personal` set it to `true`, which matches what Coco logged before MCP.
- **`timeout`** is in seconds per call. The default is 30.

Changing the config or a server version changes the `prompt_version` in logs and evals, because the tool schemas are part of it.

### Add a server

1. Review the server's code, authentication and data handling. Record its command, exact version and permissions in its `description`.
2. Add it to `mcp_servers.local.json`, pinning an exact version. Start with only read-only tools in `tools`.
3. Put any token in an environment variable and reference it with `${env:NAME}`.
4. Start Coco and check the startup lines: the server should be running with the tools you expect.
5. Add a fake-server test or an eval case for the new behaviour before relying on it.

### Disable a server

Set `"enabled": false`, or delete its entry from the local file. Its subprocess is not started and none of its tools reach the model. To remove a single tool, delete it from `tools`.

## Security model

- Each server is a separate subprocess with its own trust boundary. The model never sees commands, environment variables or tokens. It sees only tool names, descriptions and argument schemas.
- **Filesystem.** The official server is started with `notes/` as its only allowed directory, and with its working directory set to `notes/`, so plain file names resolve there. `../` paths and absolute paths outside `notes/` are denied by the server. `tests/test_mcp_host.py` and the `safety/*` evals check this.
- **Memory.** `personal_server.py` takes the memory file path from its command line. No tool accepts a path. `forget` refuses to choose between several matching memories. Disk errors are returned as tool errors.
- **Failures are never successes.** Startup failures, unknown or non-allowlisted tools, bad JSON, server errors, timeouts and crashes all return `Error: ...` with `ok=False`. Results over 40,000 characters are cut off and marked.
- **Server stderr** goes to `logs/mcp-servers.log`, which is git-ignored, not to the chat terminal.
- **External content is data.** Text returned by tools (files now; email, documents and events later) is not instructions. The system prompt rules still apply.

## GitHub and Google services (not connected yet)

GitHub, Drive, Calendar and Gmail need accounts, consent screens and tokens, so they are not set up in this repository. Connect them one at a time, each in `mcp_servers.local.json`, following the gates in [phase-3-mcp.md](phase-3-mcp.md) (steps 2 and 3):

1. Pick a maintained server. Pin its version and record its authentication method and scopes.
2. For GitHub, use a fine-grained token limited to the selected repository, with read-only permissions. If the server has a read-only mode, turn it on. Allowlist only search and read tools.
3. For Google, use the provider's consent flow with read-only scopes (Drive files you select, Calendar read-only, Gmail read-only). Keep token files outside the repository.
4. Leave `log_payloads` set to `false` for these servers.
5. Verify, then record in this document: a real read works; write tools are not exposed; access can be revoked.

**Revoking access.** Disable the server in the config, then revoke the token at the provider: GitHub → Settings → Developer settings → Personal access tokens; Google → Account → Security → Third-party access. Delete any local token file.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Server 'notes' failed to start: ... not found` | Node/npx is not on PATH. Install Node.js or fix PATH. |
| `did not start within 60s` | First npx download is slow or offline. Run `npx -y @modelcontextprotocol/server-filesystem@2.2.0 .` once by hand. |
| `Server 'personal' failed to start` | Check `logs/mcp-servers.log`. Usually the `mcp` package is missing (`pip install -r requirements.txt`). |
| `environment variable X is not set` | Set it in your shell or secret manager before starting Coco. |
| `does not offer configured tool 'x'` | The server version renamed the tool. Check its tool list and update `tools`. |
| `No current_time tool is available` | The personal server did not start. See the line above. |
| Model stops calling tools | Run `python evals.py`. The preflight checks that tool schemas reach the model. |

For a closer look, `logs/mcp-servers.log` holds each server's stderr plus a start line from the host.
