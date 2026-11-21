# Week 9: MCP, the Model Context Protocol

In week 8 you gave Claude a tool by writing a Python function **inside your own program**. But the same "search the FastAPI docs" tool could be useful in many apps: Claude Code, Claude Desktop, an IDE, a teammate's agent. Without a standard, each app would need its own copy of the tool. **MCP is the standard plug**: write the tool once as an **MCP server**, and any app that speaks MCP can use it.

**Built in:** `rag-docs/scratch/mcpdemo/` (`order_server.py`, a toy server, and `talk_to_server.py`, which talks to it with raw JSON), and the project: `rag-docs/src/mcp_server.py`, `tests/test_mcp_server.py` and `src/mcp_host.py`.

**Versions used:** `mcp` Python SDK **2.3.0**, protocol version **2026-07-28**. Most tutorials online use SDK 1.x (`FastMCP`) and the older handshake; see 1.5.

---

## 1. Concepts

### 1.1 The problem MCP solves

Without a standard, every app × every tool is a separate integration:

```
WITHOUT MCP: N apps × M tools = N×M integrations     WITH MCP: N + M
  Claude Code ──┬── docs search                        Claude Code ──┐          ┌── docs search server
  Claude Desktop├── GitHub                             Claude Desktop├── MCP ───┼── GitHub server
  your agent ───┴── Postgres                           your agent ───┘          └── Postgres server
  (each app writes its own glue for each tool)         (each side implements the protocol once)
```

**Java analogy: JDBC.** Before JDBC, every app needed vendor-specific code for every database. JDBC defined one interface; each database ships a **driver**, and any Java app can use any database. An **MCP server is a driver for a capability**: search, files, a database, an API. A closer analogy is **LSP** (Language Server Protocol): one Java language server works in VS Code, IntelliJ and Vim, because they all speak LSP. MCP was designed with LSP as its model.

### 1.2 The architecture: host, client, server

```
┌──────────────────── HOST (e.g. Claude Code) ────────────────────┐
│                                                                 │
│  the LLM + the agent loop (week 8)                              │
│        │ tool_use: search_docs(...)                             │
│        ▼                                                        │
│  MCP client #1 ──────────┐   MCP client #2 ──────────┐          │
└──────────────────────────│───────────────────────────│──────────┘
                           │ JSON-RPC                  │ JSON-RPC
                           ▼                           ▼
               MCP server: rag-docs          MCP server: GitHub
               (your Python process)         (someone else's)
```

| Role | What it is | Example |
| --- | --- | --- |
| **Host** | The app the user talks to. It owns the LLM and the agent loop | Claude Code, Claude Desktop, your own agent |
| **Client** | A connection *inside* the host, **one per server** | created by the host for each configured server |
| **Server** | A program that exposes capabilities over MCP | your `rag-docs` search, a GitHub server |

**The model still never runs code** (week 8, 1.2). The chain just has one more hop:

```
Claude ─► tool_use ─► host's agent loop ─► MCP client ─► tools/call ─► MCP server runs the function
Claude ◄─ tool_result ◄─ host ◄─ MCP client ◄─ result ◄─────────────────┘
```

From Claude's point of view, nothing changes: it sees tool definitions and sends `tool_use` blocks. MCP standardises the part **between the host and the tool's code**.

### 1.3 The three things a server can offer

| Primitive | What it is | Who decides to use it | Example |
| --- | --- | --- | --- |
| **Tools** | Functions with a JSON Schema input | **The model** (like week 8) | `search_docs(query)`, `create_issue(title)` |
| **Resources** | Read-only data, addressed by a URI | **The app / user** (attach as context) | `docs://path-params.md`, `file:///…/README.md` |
| **Prompts** | Reusable prompt templates with arguments | **The user** (e.g. a slash command) | `/explain-fastapi-topic topic=Enum` |

Tools are the most used. **The difference is who is in control:** the model calls tools on its own; resources and prompts are things the user or app pulls in deliberately.

### 1.4 The protocol: JSON-RPC 2.0 (real captured traffic)

MCP messages are **JSON-RPC 2.0**: a `method`, `params`, and an `id` to match responses to requests (the same correlation idea as `tool_use_id`). This is real traffic between a client and our toy `orders` server, captured with a logging proxy (trimmed):

```jsonc
// 1. What can you do? (modern protocol: server/discover)
CLIENT→SERVER {"jsonrpc":"2.0","id":1,"method":"server/discover",
               "params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28", ...}}}
SERVER→CLIENT {"jsonrpc":"2.0","id":1,"result":{"capabilities":{"tools":{...},"resources":{...},"prompts":{...}},
               "supportedVersions":["2026-07-28"], ...}}

// 2. Which tools do you have?
CLIENT→SERVER {"jsonrpc":"2.0","id":2,"method":"tools/list", ...}
SERVER→CLIENT {"jsonrpc":"2.0","id":2,"result":{"tools":[{
                 "name":"get_order_status",
                 "description":"Look up the current status of one order. ...",
                 "inputSchema":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}}]}}

// 3. Run one
CLIENT→SERVER {"jsonrpc":"2.0","id":3,"method":"tools/call",
               "params":{"name":"get_order_status","arguments":{"order_id":"A123"}, ...}}
SERVER→CLIENT {"jsonrpc":"2.0","id":3,"result":{
                 "content":[{"type":"text","text":"shipped on 2026-11-06, arriving 2026-11-10"}],
                 "isError":false}}
```

**This maps almost one to one onto week 8:**

| MCP | Claude API (week 8) |
| --- | --- |
| `tools/list` → `name`, `description`, `inputSchema` | the `tools` parameter: `name`, `description`, `input_schema` |
| `tools/call` with `name` + `arguments` | a `tool_use` block's `name` + `input` |
| result `content` + `isError` | a `tool_result`'s `content` + `is_error` |

So a host's job is mostly translation: it converts `tools/list` into the `tools` parameter, and each `tool_use` into a `tools/call`.

### 1.5 Protocol versions: discover vs the legacy handshake

The spec changed in **2026-07-28**, and most material online (including what you'll find in older tutorials) describes the earlier version:

| | Legacy (up to `2025-11-25`) | Modern (`2026-07-28`) |
| --- | --- | --- |
| How a connection starts | `initialize` request → response → `notifications/initialized` | `server/discover`, then plain requests |
| Where the version lives | negotiated once, in the handshake | in **every request's** `_meta` |
| Style | a stateful session | closer to stateless (like REST): each request stands alone |

The SDK server accepts **both** (tested: a legacy client got `"protocolVersion":"2025-11-25"`), so older hosts still work. The SDK client tries `server/discover` first and falls back to `initialize` on older servers.

**SDK 2.x renames:** `FastMCP` → **`MCPServer`** (`from mcp.server.mcpserver import MCPServer`). Code written for 1.x fails with an import error that points at the migration guide.

### 1.6 Transports: how the bytes travel

| Transport | How it works | Use it for |
| --- | --- | --- |
| **stdio** | The host **starts your server as a subprocess** and talks over its stdin/stdout, one JSON message per line | Local tools: your machine, your files. No network, no auth |
| **Streamable HTTP** | The server is a web service at a URL (e.g. `https://…/mcp`) | Remote/shared servers. Needs authentication (OAuth) |

**The stdio golden rule: never write to stdout in a stdio server.** stdout *is* the protocol channel, so a stray `print("debug")` lands in the middle of the JSON stream and breaks the connection. Log to **stderr** instead.

### 1.7 Errors: anticipated vs unexpected

The captured traffic showed a surprise. The tool raised `ValueError("No order with id A999. Order ids look like A123.")`, but the client received only:

```json
{"content":[{"type":"text","text":"Error executing tool get_order_status"}],"isError":true}
```

The `MCPServer` SDK splits errors in two:
- **`ToolError`** (`from mcp.server.mcpserver.exceptions import ToolError`): "a failure you anticipated". The message **goes to the model**, and it's logged without a traceback.
- **Any other exception** is treated as a crash. The model sees only `Error executing tool <name>`, and the details stay in the server log.

Why: an arbitrary exception message can leak internals (file paths, SQL, connection strings) to the model, and through it to the user. That's the **opposite default** from the Anthropic tool runner (week 8), which sends `repr(e)` for every exception. **Raise `ToolError` for every error the model should read**, and keep the message helpful (what went wrong + what valid input looks like).

### 1.8 Security

- **A stdio server runs with your permissions.** It's a program on your machine, so it can read your files and your `.env`. Install servers like you install dependencies: from sources you trust.
- **Tool descriptions are prompt input.** A third-party server's descriptions go straight into the model's context. A malicious one could hide instructions there ("tool poisoning"), and the same goes for tool *results* (prompt injection, week 8 1.10).
- **Least privilege:** expose only what's needed. A read-only `search_docs` is low risk; `delete_file` is not. Hosts like Claude Code ask before running tools, and keep that on for risky servers.
- **Remote servers need authentication** (the SDK supports OAuth). A public, unauthenticated tool endpoint is a public API.

### 1.9 MCP or a plain tool?

| Use a plain tool (week 8) | Use an MCP server |
| --- | --- |
| Only **your** app needs it | **Several hosts** should use it (Claude Code, Desktop, IDE, other agents) |
| It needs your app's in-process state | You want to plug into apps you don't control |
| Simplest possible setup | Another team, or the public, provides it |

`rag-docs/src/agent.py` is a plain tool, and that's right for a single app. This week turns the same search into an MCP server, so **Claude Code itself** can search your FastAPI docs.

---

## 2. How we built it

### 2.1 Talking to a server by hand (`scratch/mcpdemo/talk_to_server.py`)

No MCP client library: start `order_server.py` as a child process, write one JSON line to its stdin, read one line from its stdout. Four steps, and the real output:

```
STEP 1  server/discover        → it supports: ['prompts', 'resources', 'tools']
STEP 2  tools/list             → get_order_status, its description, its input schema
STEP 3  tools/call A123        → 'shipped on 2026-11-06, arriving 2026-11-10'   isError: False
STEP 4  tools/call A999        → 'Error executing tool get_order_status: No order with id A999. ...'   isError: True
```

**MCP is just an agreed set of JSON messages.** Step 2 gives the host the same three things as week 8's `TOOLS` list (name, description, input schema), and steps 3–4 are a `tool_use` → `tool_result` round trip, over a pipe instead of a function call.

### 2.2 The rag-docs MCP server (`src/mcp_server.py`)

```python
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from answer import format_documents
from retrieve import retrieve

server = MCPServer("rag-docs")


@server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def search_docs(
    query: Annotated[str, Field(description="What to search for, in FastAPI terms, e.g. 'dependency injection Depends'.")],
    top_k: Annotated[int, Field(description="How many passages to return.", ge=1, le=20)] = 8,
) -> str:
    """Search the FastAPI docs. Call this for any FastAPI question, using FastAPI terms."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        raise ToolError("The document index is empty. Run `uv run python -m src.ingest` first.")
    return format_documents(chunks)


if __name__ == "__main__":
    server.run()          # stdio: the host starts this file and talks over stdin/stdout
```

- **It reuses week 5's `retrieve()` and `format_documents()` unchanged.** The MCP server is a thin adapter: the protocol on one side, your existing code on the other. **Java analogy:** a REST controller that only translates HTTP into a call to an existing service.
- **Two differences from `@beta_tool`** (both found by testing):
  - `MCPServer` sends the **whole docstring, raw**: no `Args:` parsing, and indentation included (`"...rephrase\n        the question..."`). Keep it to one line.
  - Parameter descriptions and validation come from **`Annotated[..., Field(description=..., ge=1, le=20)]`**. `top_k=500` is rejected **before** the function runs, with a clear error for the model. **Java analogy:** Bean Validation, `@Min(1) @Max(20)` on a DTO field.
- **`ToolError` for the anticipated failure** (an empty index), so the model reads *our* message.
- **Annotations:** `read_only_hint=True` (it changes nothing) and `open_world_hint=False` (a fixed local index, not the web). They're **hints**: a host must not trust them from an unknown server.
- **Checked before writing it:** it needs **no API key** (the host has the LLM; creating the Anthropic client in `answer.py` doesn't fail without a key), **stdout stays clean** (nothing in Chroma prints to it), and it starts in ~1.5 s, with ~0.5 s per search.

### 2.3 Testing a server in-process (`tests/test_mcp_server.py`)

```python
from mcp import Client

async def call(arguments: dict):
    async with Client(mcp_server.server) as client:       # a server OBJECT: no subprocess, no stdio
        return await client.call_tool("search_docs", arguments)

def test_top_k_out_of_range_is_rejected_before_searching(fake_retrieve, top_k):
    result = asyncio.run(call({"query": "x", "top_k": top_k}))
    assert result.is_error
    assert fake_retrieve == []          # validation stopped it; retrieve never ran
```

- `Client(server_object)` connects **in-process**, but every call still goes through MCP: listing, validation, errors. It's like Spring's `MockMvc`: the full request pipeline, with no real network.
- `monkeypatch.setattr(mcp_server, "retrieve", fake)` replaces the search, so the tests run offline in milliseconds.
- **7 tests:** the tool is listed with its descriptions and limits; a search returns tagged documents; `top_k` is passed through; `top_k` 0 and 21 are rejected before searching; a missing `query` is rejected; an empty index gives *our* message.
- The MCP client is `async`; `asyncio.run(...)` runs one coroutine from a normal test.

### 2.4 Connecting it to Claude Code

```bash
claude mcp add --scope user rag-docs -- uv run --directory /Users/shady/Documents/rag-docs python src/mcp_server.py
claude mcp list        # rag-docs: ... ✔ Connected
```

| Part | Meaning |
| --- | --- |
| `--scope user` | available in **all** your projects (e.g. chat-service, which is a FastAPI app). `local` = this project only; `project` = a `.mcp.json` committed for the team |
| `--` | everything after it is the command that **starts** the server |
| `uv run --directory …` | use the project's environment, whatever folder Claude Code is in (tested from `/tmp`) |

Then, in a **new** Claude Code session (servers load at startup), `/mcp` shows it connected, and a FastAPI question gets answered with citations like `[query-params.md]` from **your** 135 chunks.

### 2.5 Your own MCP host (`src/mcp_host.py`)

What Claude Code does with a server, in about 50 lines:

```python
SERVER = StdioServerParameters(command=sys.executable, args=["src/mcp_server.py"])
claude = anthropic.AsyncAnthropic()

def to_claude_tool(tool: Tool) -> ToolParam:                  # MCP → Claude API
    return {"name": tool.name, "description": tool.description or "",
            "input_schema": tool.input_schema}

def to_tool_result(tool_use_id: str, result: CallToolResult) -> ToolResultBlockParam:
    text = "".join(c.text for c in result.content if isinstance(c, TextContent))
    return {"type": "tool_result", "tool_use_id": tool_use_id,
            "content": text, "is_error": result.is_error}

async def run_host(question: str) -> str:
    async with Client(SERVER) as mcp:                         # starts the server; stops it on exit
        tools = [to_claude_tool(t) for t in (await mcp.list_tools()).tools]    # tools/list
        messages = [{"role": "user", "content": question}]
        for step in range(1, MAX_STEPS + 1):
            response = await claude.messages.create(model=MODEL, max_tokens=2048, tools=tools,
                                                    messages=messages, system=mcp.instructions or "")
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "end_turn":
                return "".join(b.text for b in response.content if b.type == "text")
            results = [to_tool_result(b.id, await mcp.call_tool(b.name, b.input))  # tools/call
                       for b in response.content if b.type == "tool_use"]
            messages.append({"role": "user", "content": results})
```

- **The host never mentions `search_docs`, Chroma or `retrieve()`.** It *discovers* the tools with `tools/list`. Point `SERVER` at another MCP server and the same file uses *its* tools. That's the "N + M" of 1.1.
- **The bridge is all snake_case renames:** `inputSchema` → `input_schema`, `isError` → `is_error`, `tools/list` → `list_tools()`, `tools/call` → `call_tool()`.
- **`async`/`await`:** MCP is I/O (pipes, HTTP), so its client is async. `await` ≈ `CompletableFuture.get()` without blocking the thread; `asyncio.run(...)` is the entry point; `async with` ≈ try-with-resources (it starts and stops the server). Calling an `async def` without `await` does nothing but create a coroutine object. The host therefore uses `anthropic.AsyncAnthropic()`.
- **The SDK shortcut,** once you understand the bridge (verified with SDK 1.15):
  ```python
  from anthropic.lib.tools.mcp import async_mcp_tool
  tools = [async_mcp_tool(t, mcp.session) for t in (await mcp.list_tools()).tools]
  runner = claude.beta.messages.tool_runner(model=MODEL, max_tokens=2048, max_iterations=6,
                                            tools=tools, messages=[...])
  final = await runner.until_done()
  ```
  It's the same relationship as week 8's manual loop vs the tool runner.

### 2.6 The server offers; the host decides

The same question, the same server and the same model, with one change in the host:

| Host's system prompt | "How do I connect FastAPI to MongoDB?" |
| --- | --- |
| none | answered from Claude's **general knowledge** (`pip install motor`, lifespan, `Depends`…), with an honest note that it wasn't from the docs |
| the server's `instructions` | **"I don't know. The FastAPI docs passages I searched didn't cover MongoDB…"** |

```python
server = MCPServer("rag-docs", instructions=(
    "Answer FastAPI questions using only the passages returned by search_docs. "
    "Cite the source file after each claim, like [body.md]. "
    "If the passages don't contain the answer, say you don't know."))
```

- **Grounding rules are behaviour, and behaviour belongs to the host.** In week 8 they lived in `agent.py`'s `SYSTEM_PROMPT`; an MCP server has no system prompt of its own.
- **`instructions` are how a server *suggests* behaviour.** They're sent with `server/discover` (or `initialize`), and the client exposes them as `client.instructions`. Verified in all three connection modes: in-process, modern stdio and legacy stdio.
- **A host should only adopt instructions from servers it trusts.** They go into the system prompt, the most trusted part of the conversation, so a malicious server's instructions are prompt injection.

| | Decides | In our code |
| --- | --- | --- |
| **Server** | *what* can be done, and *suggests how* | `search_docs` + `instructions` |
| **Host** | *what actually happens*: system prompt, permissions, limits | `mcp_host.py`, or Claude Code |

**Permissions:** no prompt appeared before the search, although no settings rule allowed it. The likely cause is the session's permission mode (shown at the bottom of the prompt; Shift+Tab cycles it). Auto mode approves actions it judges low-risk. `/permissions` shows the rules in effect; MCP tools appear as `mcp__rag-docs__search_docs`, and you can add an `ask` rule to always be asked.

## 3. Key points

- **MCP is a standard plug for tools:** write a capability once as a server, and any MCP host can use it (N + M instead of N × M). Analogies: JDBC drivers, LSP.
- **Host** = the app with the LLM and the agent loop; **client** = one connection inside the host per server; **server** = your program exposing capabilities.
- **The model still never runs code.** `tool_use` → host → MCP `tools/call` → server runs it → result → `tool_result`.
- **Three primitives:** tools (the model decides), resources (the app or user attaches data), prompts (the user picks a template).
- **The wire format is JSON-RPC 2.0**, one message per line over stdio. `tools/list` ≈ the API's `tools` parameter; `tools/call` ≈ `tool_use`; `content` + `isError` ≈ `tool_result` + `is_error`.
- **Protocol 2026-07-28:** `server/discover`, and the version + client capabilities in **every** request's `_meta`. The legacy `initialize` handshake still works.
- **SDK 2.x:** `MCPServer` (formerly `FastMCP`), `Client` (connects to a URL, a subprocess, or an in-process server object).
- **stdio servers must never write to stdout:** it's the protocol channel. Log to stderr.
- **`ToolError` for anticipated failures**, so the model sees your message. Any other exception shows only "Error executing tool <name>": safe by default, unlike the Anthropic tool runner.
- `MCPServer` sends the docstring **raw** (no `Args:` parsing); parameter descriptions and validation come from `Annotated[..., Field(description=..., ge=, le=)]`.
- **Annotations** (`read_only_hint`, `destructive_hint`, `idempotent_hint`, `open_world_hint`) are hints. Never trust them from an unknown server.
- **Test servers in-process** with `Client(server_object)`: the full MCP pipeline, no subprocess. It's like `MockMvc`.
- **Server offers, host decides:** grounding rules belong in the host's system prompt. A server can suggest them through `instructions`, and a host should only adopt them from trusted servers.
- **Everything a third-party server sends is untrusted input:** descriptions, annotations, instructions and results. A stdio server also runs with your permissions.
- **Use MCP when several hosts should share a capability.** For a tool only your app uses, a plain function tool is simpler.

## 4. Gotchas we hit

1. **`No module named 'mcp.server.fastmcp'`.** SDK 2.x renamed `FastMCP` to `MCPServer` (`from mcp.server.mcpserver import MCPServer`). Most tutorials still show 1.x code.
2. **The helpful error message never reached the client.** A plain `ValueError` is treated as a crash, and the model only sees `Error executing tool <name>`. Raise `ToolError` for anticipated failures.
3. **`params._meta is missing the required envelope key(s): io.modelcontextprotocol/clientCapabilities`.** Sending raw JSON by hand, every request needs both `protocolVersion` and `clientCapabilities` (an empty `{}` is fine) in `_meta`. The SDK client adds them automatically, which is why you never see this when using it.
4. **The first demo was too complex** (client + logging proxy + server + async). A script that writes one JSON line and reads one line back showed the same protocol more clearly.
5. **`MCPServer` doesn't parse `Args:` and doesn't clean up the docstring.** The raw text, indentation included, is the description. Keep it to one line, and put parameter descriptions in `Field(description=...)`.
6. **A tool description example that contradicts its own instruction.** "Rephrase into FastAPI terms", then `e.g. 'How do I install the software?'`, a full English question and not FastAPI vocabulary. Models copy examples (week 8, 2.7), so the example wins.
7. **Pylance: `Cannot access attribute "text" for class "ImageContent"`** (in the tests, written by me). A tool result's `content` can be text, image, audio or a resource. Narrow it first: `assert isinstance(block, TextContent)`. It's the same fix as `b.type == "tool_use"` in week 8.
8. **Ruff format wrapped `return format_documents(chunks)` over three lines.** A long end-of-line comment pushed it past the line length. The formatter moves code, never comments, so put long comments on their own line above.

9. **The host answered an unanswerable question from general knowledge.** Its system prompt was empty, and the server's grounding rules didn't exist yet. Behaviour belongs to the host: add a system prompt, or adopt the server's `instructions`.
10. **No permission prompt in Claude Code** before an MCP tool ran, although no settings rule allowed it. Check the session's permission mode (bottom of the prompt; Shift+Tab) and `/permissions`.

## 5. Revision questions

1. What problem does MCP solve? Explain it with the N × M picture and one Java analogy.
2. Name the three roles in MCP and say which one owns the LLM.
3. Claude Code uses your `search_docs` MCP tool. Trace every hop from Claude's decision to your `retrieve()` call and back.
4. What are the three server primitives, and who decides when each is used?
5. Which MCP messages correspond to the Claude API's `tools` parameter, a `tool_use` block and a `tool_result`?
6. What changed in the 2026-07-28 protocol compared with the legacy handshake, and can old clients still use a new server?
7. Why does a single `print("debug")` break a stdio MCP server, and what should you do instead?
8. Your tool raises `ValueError("No order with id A999")`. What does the model see, and how do you make it see your message? Why is that the default?
9. How do you give an `MCPServer` tool parameter a description and a range limit? What happens to an `Args:` section in its docstring?
10. What do tool annotations like `read_only_hint` mean, and why can't a host rely on them?
11. How do the tests run the MCP server without starting a subprocess, and what is still being tested?
12. What does `claude mcp add --scope user rag-docs -- uv run --directory … python src/mcp_server.py` do, part by part? What do the other scopes mean?
13. Your MCP host answered the MongoDB question from general knowledge. Why, and what fixed it?
14. Why is it risky for a host to put any server's `instructions` straight into its system prompt?
15. When would you *not* build an MCP server, and just use a plain function tool?

## Answers

1. Without a standard, every app needs its own glue for every tool: N apps × M tools integrations. With MCP, each app implements the client side once and each tool implements the server side once: N + M. It's like **JDBC**: one interface, a driver per database, and any app can use any database. (Or **LSP**: one language server works in every editor.)
2. **Host** (the app the user talks to; it owns the LLM and the agent loop, e.g. Claude Code), **client** (one connection inside the host per server), **server** (a program exposing capabilities, e.g. `rag-docs`).
3. Claude emits a `tool_use` block for `search_docs` → Claude Code's agent loop receives it → its MCP client sends `tools/call` (JSON-RPC over the server's stdin) → `mcp_server.py` validates the arguments and runs `search_docs` → `retrieve()` searches Chroma → `format_documents()` → the result goes back as JSON on stdout → the client hands it to the loop → it becomes a `tool_result` in the next API call → Claude answers with citations.
4. **Tools** (functions; the **model** decides to call them), **resources** (read-only data by URI; the **app or user** attaches them), **prompts** (templates; the **user** picks them, e.g. as slash commands).
5. `tools/list` (name, description, `inputSchema`) → the API's `tools` parameter; `tools/call` (name, arguments) ↔ a `tool_use` block; the result's `content` + `isError` → a `tool_result`'s `content` + `is_error`.
6. Legacy: a stateful handshake (`initialize` → response → `notifications/initialized`) that negotiates the version once. 2026-07-28: `server/discover`, then each request carries the protocol version and client capabilities in `_meta`, so it's closer to stateless. Yes: the SDK server accepts both (tested: a legacy client got version `2025-11-25`).
7. In a stdio server, stdout **is** the protocol channel: the host reads one JSON message per line. A stray print injects a non-JSON line, and the host can't parse it. Log to **stderr** (or a file) instead.
8. Only `Error executing tool <name>`: a plain exception is treated as an unexpected crash, and its text is hidden. Raise **`ToolError`** for anticipated failures, so the message reaches the model. It's the safe default because an arbitrary exception message can leak internals (paths, SQL, credentials) to the model and the user.
9. `Annotated[int, Field(description="How many passages to return.", ge=1, le=20)]`. The description and the limits appear in the schema, and an out-of-range value is rejected before the function runs. An `Args:` section is **not** parsed: it's sent as part of the raw description, so keep the docstring to one line.
10. They describe the tool's behaviour: `read_only_hint` (changes nothing), `destructive_hint` (may delete or overwrite), `idempotent_hint` (repeat calls are harmless), `open_world_hint` (reaches the outside world). They're claims made by the server, so a malicious one could label a destructive tool read-only. Use them for display, or for servers you already trust, never as a security guarantee.
11. `Client(mcp_server.server)` connects to the server **object** in-process, with no subprocess or stdio, and `monkeypatch` replaces `retrieve`. Still tested: tool listing and schemas, argument validation (`top_k` limits, missing `query`), the call itself, error results and `ToolError` messages, and annotations. That's the full MCP request pipeline, like Spring's `MockMvc`.
12. It registers an MCP server named `rag-docs` with Claude Code. `--scope user` makes it available in all your projects; `--` separates Claude's options from the command that **starts** the server; `uv run --directory …` runs it with the project's environment whatever the current folder is. Other scopes: `local` (this project, only you, the default) and `project` (a `.mcp.json` committed to the repo for the team).
13. The host had **no system prompt**, so nothing told Claude to answer only from the documents. The grounding rules had lived in `agent.py`'s `SYSTEM_PROMPT`, and an MCP server has no system prompt. Fix: the server sends `instructions`, and the host passes `mcp.instructions` as `system`. After that, it refused.
14. The system prompt is the most trusted part of the conversation. A third-party server's instructions (like its descriptions and results) are untrusted input. A malicious server could send "ignore your previous rules…", which is prompt injection with system-level authority. Only adopt instructions from servers you trust, and keep the host's own rules in charge.
15. When only **your own app** uses the tool, when it needs your app's in-process state, or when you want the simplest setup. Week 8's `agent.py` is a plain tool for exactly that reason. MCP pays off when several hosts should share the capability.
