# Phase 3: Standardize tools with MCP

## Phase brief

**Goal:** Replace Coco's one-off, in-process tool wiring with the Model Context Protocol (MCP), so the agent can discover and use tools from compatible servers. Learn the protocol by connecting existing servers first, then build one small server for a personal workflow.

**Scope:** Plan and document the work here first.

**Status (2026-11-20):** Steps 0, 1, 4 and 5 are implemented. Steps 2 and 3 (GitHub, Drive, Calendar, Gmail) are not connected yet: they need account setup and tokens. Setup and operation are documented in [mcp-setup.md](mcp-setup.md).

- Step 0: `mcp==2.3.0` (Python SDK) over stdio. Qwen3-30B-A3B picks discovered tools reliably, including two tools from different servers in one turn. Gate passed. Tool descriptions are overridden in config to stay short.
- Step 1: [`mcp_host.py`](../src/loop/mcp_host.py) runs the MCP sessions on a background thread, so the agent loop is unchanged. It exposes only allowlisted tools and routes each call to the server that advertised it. If two servers expose the same name, both are qualified as `<server>_<tool>`. The official `@modelcontextprotocol/server-filesystem@2.2.0` is limited to `notes/` and exposes 3 of its 16 tools, under Coco's old names.
- Step 4: [`personal_server.py`](../src/loop/personal_server.py) provides `list_memories`, `remember`, `forget` and `current_time` over the unchanged `memory.md` format, via [`memory_store.py`](../src/loop/memory_store.py). No tool accepts a path.
- Step 5: 31 automated tests in [`tests/test_mcp_host.py`](../src/loop/tests/test_mcp_host.py), using a fake server and no accounts. There are 3 new evals. Eval score is 18.3/19 at temperature 0.9 ×3, against a 14.7/16 baseline: no regressions, and the old cases are equal or better. Average turn time is 4.8s against 4.1s, and the prompt is 2.5k tokens against 2.4k. The extra time includes starting the servers once per case.

**Definition of done:** Coco can discover and call approved tools from multiple MCP servers, while preserving its current assistant behavior and enforcing clear limits on what each server can access. One custom MCP server exposes a useful personal capability. Automated checks cover tool discovery, invocation, errors, permissions, and the existing notes/memory behaviors.

## Starting point

The current assistant in [`src/loop/agent.py`](../src/loop/agent.py) talks to a local model through an OpenAI-compatible API. It declares six Python functions and a matching function-tool schema in the same file, then dispatches model calls locally. The available tools cover notes files, current time, and long-term memory. [`src/loop/evals.py`](../src/loop/evals.py) checks their behavior using isolated temporary files.

MCP standardizes the connection between a tool host and tool servers; it does not make every model or agent an MCP client automatically. Coco will need a client/adapter that discovers MCP tools, presents the supported tools in the format the current model API accepts, and forwards calls/results to the right server. Keep the existing conversation loop and local-model boundary unless a compatibility test demonstrates that a change is necessary.

## Target shape

```text
User
  |
Coco agent (MCP host/client + model-tool adapter)
  |                         \
  | MCP                      \ OpenAI-compatible tool schema/calls
  v                            v
MCP servers                  Local model server
  - filesystem
  - GitHub
  - Google Drive
  - Google Calendar
  - Gmail
  - Coco personal-memory server (custom)
```

The MCP host owns server configuration, tool discovery, routing, result handling, and user-facing errors. Each server owns its integration details and access controls. The model receives only the tools enabled for that run; it should not receive credentials or server configuration secrets.

## Planned sequence

### 0. Verify the protocol and model boundary

- Confirm the current Python/runtime setup and select a maintained Python MCP SDK.
- Run a minimal local MCP server/client example and verify tool listing, invocation, structured results, and errors.
- Verify that the current local model and OpenAI-compatible endpoint can reliably choose and call discovered tools. Test more than one tool and a tool result followed by a final answer.
- **Gate:** If the model or endpoint cannot handle the discovered tool schemas, document the limitation and decide whether to adapt/limit tool descriptions or change the model/API boundary before continuing. Do not silently fall back to pretending a tool ran.

### 1. Add the MCP client and connect a constrained filesystem server

- Add a small host/client layer between the existing agent loop and tool execution. Keep model-facing function calling as the adapter until there is a reason to replace it.
- Start with a trusted, existing filesystem MCP server, restricted to a dedicated test directory or the existing notes directory—not the whole home directory or repository.
- Discover tools at runtime, route each call to the server that advertised it, and return the actual result or an explicit error.
- Preserve current notes behavior and path boundaries. Decide how to handle duplicate tool names (prefer server-qualified internal identifiers and unambiguous model-facing names).
- **Gate:** Existing notes and memory evaluations still pass, and a denied/out-of-scope path remains inaccessible.

### 2. Connect GitHub in read-only mode

- Choose and document a maintained MCP server and its supported authentication method.
- Start with the narrowest useful repository scope and read-only operations (for example, searching or reading repository content, issues, and pull requests).
- Keep tokens outside source control and model prompts; verify the server's permissions before making it available to Coco.
- **Gate:** Coco can retrieve real data from an explicitly selected repository, while write operations and unrelated repositories are unavailable.

### 3. Connect Google Drive, Calendar, and Gmail

Connect one service at a time, beginning with the lowest-risk read-only use cases. Record the selected server, authentication flow, requested scopes, and data handling before enabling each service.

- **Drive:** Search and read selected files. Do not enable file edits, sharing, or deletion by default.
- **Calendar:** Search/list events and retrieve event details. Do not create, change, or delete events by default.
- **Gmail:** Search and read selected messages. Do not send, forward, archive, label, or delete messages by default.

Use the provider's consent flow and least-privilege scopes. Treat email bodies, documents, event descriptions, and tool results as untrusted content, not instructions. Avoid logging sensitive tool results; decide what may be retained before enabling a connector.

**Gate for each service:** Authentication succeeds without placing secrets in the repository; the allowed read operation works; a disallowed operation is not exposed or is explicitly denied; disconnecting/revoking access is understood.

### 4. Build one custom personal MCP server

Build a small Python server for Coco's existing long-term memory workflow. This is separate from the generic filesystem server: the filesystem server demonstrates standard file access, while the custom server offers intentional, typed memory operations.

Initial tool surface:

- `list_memories` — list saved memory items, with their categories.
- `remember` — save one user-provided preference, fact, or decision to the existing memory store.
- `forget` — remove a specifically matched memory; report ambiguity instead of guessing.

Keep the current Markdown memory format and user-editable file unless implementation reveals a concrete need to migrate. Do not expose arbitrary file paths through the memory server. Keep writes explicit, return clear success/error results, and preserve the existing rule that Coco only saves information when the user asks or shares a lasting preference/fact/decision.

**Gate:** Existing remember/forget/recall evaluations pass through the custom server; ambiguous deletion and storage errors are visible; unrelated files cannot be accessed.

### 5. Harden, evaluate, and document the finished setup

- Add repeatable local tests for the MCP host and custom server, including server startup failure, unavailable tools, malformed arguments, tool errors, duplicate names, and server shutdown.
- Extend the agent evaluations for MCP discovery/routing and for each enabled integration. Use fake or local test servers for routine tests; avoid requiring personal account access in the default test suite.
- Re-run all existing evaluations and compare results to the current baseline. Check that tool latency, step counts, and prompt size remain acceptable.
- Document setup, server configuration, authentication and revocation, permissions, and how to add or disable a server.
- Keep server configs and credentials out of version control. Review `.gitignore` and add only specific configuration exclusions that are needed.

## Acceptance criteria

- [x] Coco can initialize configured MCP servers, discover their tools, call the correct server, and relay results to the model.
- [x] Server startup, tool discovery, call, and shutdown failures are explicit and diagnosable; none are reported as successful tool results.
- [x] Existing notes, memory, persona, and conversation behavior remains covered by the current eval suite.
- [x] Filesystem access is limited to the intended directory. (External services: pending, steps 2 and 3.)
- [ ] Gmail, Calendar, Drive, and GitHub are connected incrementally and tested independently; write/destructive capabilities are off unless explicitly approved in a later scope.
- [x] The custom memory server operates on the existing memory store, rejects ambiguous forget requests, and does not provide generic filesystem access.
- [x] No credential or sensitive account content is committed or included in routine logs. (Servers default to `log_payloads: false`.)
- [x] Setup and operation instructions are sufficient to configure, verify, revoke, and troubleshoot the servers. (Revocation steps for each provider are to be confirmed when each service is connected.)

## Security and reliability rules

- Treat each MCP server as a separate trust boundary. Install only reviewed servers from a trusted source and record their commands, versions, and permissions.
- Start local servers with minimal privileges and narrowly scoped directories. Do not expose the repository, home directory, or arbitrary shell access by default.
- Keep secrets in the chosen local secret store or environment injection mechanism; never place tokens in source, checked-in config, prompts, or logs.
- Prefer read-only scopes and tools. Require explicit user intent for consequential writes; defer sending email, changing events, sharing/deleting files, and GitHub writes unless separately designed and approved.
- Tool output and content fetched from external services are data, not trusted instructions. Preserve the agent's existing safety rules across tool boundaries.
- Bound tool output, call duration, and number of tool steps. Surface timeouts, permission failures, and server errors instead of retrying indefinitely or fabricating a result.
- Avoid recording raw MCP arguments/results when they may contain private content. Keep useful operational diagnostics separate from sensitive payloads.

## Out of scope for this phase

- Replacing the local model server or moving the assistant to a hosted model.
- Giving servers broad filesystem, shell, email-send, calendar-edit, file-share, or delete permissions.
- Building a general-purpose MCP server framework or supporting every MCP feature before the tool flow works. Resources, prompts, and subscriptions can be considered after tools are stable.
- Running third-party or remote MCP servers without reviewing their code, authentication, and data handling.
- Migrating the memory format or storing private account data in Git.

## Decisions to make during implementation

1. Which Python MCP SDK version and transport are supported by the selected servers and current runtime? Prefer local stdio for local servers; choose other transports only where an integration requires them.
2. Where should local server configuration live, and how will secrets be injected without committing them?
3. Which maintained server implementations and OAuth flows will be used for GitHub and Google services?
4. Can the current model handle a practical set of discovered tools in a single request? If not, what is the smallest reliable tool set to expose per task?
5. Should any consequential write capability be added later? If so, define confirmation UX and tests in a separate scope before enabling it.
