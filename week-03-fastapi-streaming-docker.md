# Week 3: Serving LLMs: FastAPI, streaming, Docker

In weeks 1 and 2 Claude was called from a script on your own laptop. This week it moves behind an HTTP API that other programs can use, the way an LLM feature ships in a real product. FastAPI handles the routes and validation, Pydantic defines the request and response shapes, and server-sent events stream the answer as it is generated. Claude's failures become meaningful HTTP status codes, a mocked test suite runs in under a second for free, and Docker packages the service so it runs the same way on any machine.

**Built in:** [github.com/Vsaraf334/chat-service](https://github.com/Vsaraf334/chat-service)

---

## 1. Concepts

### 1.1 FastAPI compared with Spring Boot

FastAPI is a Python web framework. If you know Spring Boot, almost every idea has a direct equivalent:

| Spring Boot | FastAPI |
| --- | --- |
| `@RestController` class | A `FastAPI()` app object |
| `@GetMapping("/health")` annotation | `@app.get("/health")` decorator |
| `@PathVariable String id` | A function parameter whose name appears in the path: `{conversation_id}` |
| `@RequestBody @Valid ChatRequest req` | A parameter typed as a Pydantic model: `request: ChatRequest` |
| Bean Validation (`@Size`, `@Min`, `@Max`) | Pydantic `Field(min_length=..., ge=..., le=...)` |
| `ResponseStatusException(HttpStatus.NOT_FOUND)` | `raise HTTPException(status_code=404, ...)` |
| springdoc-openapi / Swagger UI | Built in: `/docs` is generated automatically |
| Embedded Tomcat | uvicorn (an ASGI server) |
| `@ResponseStatus(HttpStatus.CREATED)` | `@app.post(..., status_code=201)` |

**Decorators vs annotations.** A Java annotation is metadata that Spring reads with reflection. A Python decorator is a **function that runs when the module loads**. `@app.get("/health")` calls `app.get("/health")`, which returns a function that registers `health` in the app's routing table. The result is the same, a URL mapped to a handler, but in Python it's plain code running rather than reflection.

**ASGI and uvicorn.** FastAPI doesn't listen on a port itself. A server called **uvicorn** accepts HTTP connections and hands each request to the app through a standard interface called **ASGI** (Asynchronous Server Gateway Interface). It's the same split as Tomcat (the server) and your Spring app (the logic):

```
client ──HTTP──► uvicorn (server, port 8000) ──ASGI──► FastAPI app ──► your function
```

`fastapi dev app/main.py` starts uvicorn with auto-reload for development. `fastapi run` starts it without reload, for production (the Dockerfile uses that).

**Automatic OpenAPI.** FastAPI reads your type hints and Pydantic models and produces an OpenAPI schema at `/openapi.json`, plus an interactive UI at `/docs`. You write no extra annotations: **the types are the documentation**.

### 1.2 How a request becomes a function call

```
POST /conversations/abc-123/messages
Content-Type: application/json
{"message": "hi", "max_tokens": 200}
        │
        ▼  FastAPI matches the route pattern /conversations/{conversation_id}/messages
conversation_id = "abc-123"                ← from the URL path
request = ChatRequest(message="hi", max_tokens=200)   ← body parsed AND validated by Pydantic
        │
        ▼
send_message(conversation_id, request)     ← your function runs only if everything is valid
        │
        ▼  the returned ChatResponse is converted to JSON
200 {"reply": "...", "model": "...", "input_tokens": ..., "output_tokens": ...}
```

FastAPI decides where each parameter comes from by looking at it:
- its **name appears in the path** → a path parameter
- its **type is a Pydantic model** → the JSON body
- (other simple types would be query parameters, e.g. `?limit=5`)

### 1.3 Pydantic models: the contract at the boundary

```python
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
```

- `ge` = greater than or equal, `le` = less than or equal (`gt`/`lt` are the strict versions).
- `default=1024` makes the field optional. A field with no default is required.
- If the body breaks a rule, FastAPI responds **422 Unprocessable Entity** with field-level error details, and **your function never runs**.

That last point matters for an LLM service: **a bad request never costs an API call**. Validation is also cost control: `max_length=4000` and `le=4096` cap how much one request can spend.

The return type `-> ChatResponse` does the same job on the way out. FastAPI validates and serialises the returned object, so the response always has the documented shape.

### 1.4 Status codes this service uses

| Code | Meaning | Where it's used |
| --- | --- | --- |
| 200 OK | Success | Most endpoints |
| 201 Created | A new resource was created | `POST /conversations` |
| 404 Not Found | That resource doesn't exist | Unknown `conversation_id` |
| 422 Unprocessable Entity | The body is well-formed JSON but breaks the rules | Empty message, `max_tokens` of 0 |
| 500 Internal Server Error | Our server is broken | Invalid API key (our configuration problem) |
| 502 Bad Gateway | The service we depend on failed | Claude unreachable or returned an error |
| 503 Service Unavailable | Temporarily overloaded, try again | Claude rate limit |

**4xx means the caller did something wrong; 5xx means the server side did.** Choosing between 502, 503 and 500 tells the caller whether retrying makes sense: 503 says "retry shortly", 500 says "retrying won't help, it's broken".

### 1.5 Endpoint design

```
GET  /health                                  liveness check, no Claude call
POST /chat                                    one question → full answer + token usage
POST /chat/stream                             one question → answer streamed as it's generated
POST /conversations                           start a conversation → 201 + conversation_id
POST /conversations/{conversation_id}/messages   one turn; Claude sees the whole history
GET  /conversations/{conversation_id}            read the history
```

Design choices worth being able to explain:
- **A health endpoint that doesn't call Claude.** Load balancers and container platforms poll it constantly; it should be cheap and test only "is the process alive".
- **The server picks the model.** Callers can't choose an expensive model, so cost stays under the service's control.
- **Conversations are a resource with an ID.** `POST` creates one (201), then messages are posted to it. It's standard REST nesting.

### 1.6 Conversation memory: the API is stateless

The Claude API remembers nothing between calls. To hold a conversation, **you send the full history every turn**:

```
turn 1 sends: [user: "My name is Shady"]
turn 2 sends: [user: "My name is Shady", assistant: "Hi Shady!", user: "What is my name?"]
turn 3 sends: ... all previous messages again + the new one
```

Consequences:
- **Input tokens grow every turn**, so long conversations get more expensive and slower. Real systems summarise old turns or use prompt caching.
- **Every turn adds two messages**: the user's and the assistant's reply. If you forget to store the reply, Claude never sees its own previous answers.
- The history must **alternate user/assistant**, so a failed call must not leave a lonely user message behind (see section 2.5).

This service keeps conversations in a Python dict, `conversations: dict[str, list[dict]] = {}`. That's simple and fast, but:
- **everything is lost on restart**, and
- **it can't be shared** between several server processes or machines (each process has its own dict).

A production version would store conversations in Postgres or Redis.

### 1.7 Streaming with server-sent events (SSE)

Without streaming, the user stares at a spinner until the whole answer exists, which can take many seconds. With streaming, text appears as Claude generates it. The total time is the same, but the **perceived** wait drops to the time until the first words appear.

**Server-sent events** is a simple, standard format for a server pushing text to a client over one long HTTP response:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"text": "RAG retrieves"}

data: {"text": " relevant documents first..."}

data: [DONE]

```

Rules:
- each event is a line starting with `data: `
- **a blank line ends the event**, so every event ends with `\n\n`
- `[DONE]` is our own convention (OpenAI uses it too) to say "finished normally"
- each `data:` carries JSON, so text containing quotes or newlines is safely escaped

SSE is one-directional (server → client) over plain HTTP. WebSockets are two-directional; chat streaming doesn't need that.

**How FastAPI streams: a generator.** A function containing `yield` is a **generator**: calling it doesn't run it, it returns an object that produces one value each time it's asked. In Java terms, it's like a lazy `Iterator` whose `next()` runs the function body up to the next `yield`.

`StreamingResponse(generator)` sends the status and headers immediately, then pulls values from the generator one at a time and writes each to the network as it arrives.

```
Claude ──token──► stream.text_stream ──► stream_reply yields "data: ...\n\n" ──► StreamingResponse ──► client
```

### 1.8 Why errors in a stream can't be HTTP errors

With a normal endpoint, an error becomes a status code such as 503. With streaming, **the `200 OK` status line has already been sent** before the first chunk, and HTTP has no way to change it afterwards. So the error has to travel **inside the stream**, as a final event:

```
data: {"text": "Hel"}

data: {"error": "Claude is busy. Try again shortly."}

```

There's no `[DONE]` after it, so the client can tell "finished" from "failed". Clients must check every event for an `error` key.

### 1.9 Mapping upstream errors

The Anthropic SDK raises typed exceptions, all subclasses of `anthropic.APIError`. The service converts each one into the HTTP response that means the most to *its* caller:

| SDK error | Status | Why |
| --- | --- | --- |
| `RateLimitError` | 503 | Temporary; tell the caller to retry |
| `APIConnectionError` | 502 | The upstream couldn't be reached |
| `AuthenticationError` | 500 + log | **Our** misconfiguration. The caller gets a generic message, and the real cause goes to the server log, so internal details never leak. |
| anything else (`APIError`) | 502 | The upstream misbehaved |

The order of the `isinstance` checks matters: the specific subclasses come first and the base-class fallback comes last, exactly like `catch` blocks in Java.

**The SDK already retries transient failures twice** (with backoff) before raising, so an error that reaches your code has already survived the retries.

`raise to_http_error(e) from e` keeps the original exception attached as the cause, the same as `new ServiceException(msg, cause)` in Java, so logs show the full chain.

### 1.10 Testing without calling Claude

Tests that call the real API are slow, cost money, need a key, and give different answers every run. Instead, **replace the client with a fake**.

**`unittest.mock.MagicMock`** is an object that accepts any attribute access or call and records everything. It's like Mockito's `mock()`, but you don't declare the type first:

```python
fake = MagicMock()
fake.messages.create.return_value = <fake response>    # when(...).thenReturn(...)
fake.messages.create.side_effect = rate_limit_error()   # when(...).thenThrow(...)
fake.messages.create.call_args.kwargs                   # ArgumentCaptor: what was it called with?
fake.messages.create.assert_not_called()                # verify(mock, never())
fake.messages.create.assert_called_once()               # verify(mock, times(1))
```

- `return_value`: what the call returns.
- `side_effect`: set it to an exception and the call **raises** it.
- `call_args.kwargs`: the keyword arguments of the last call.

**MagicMock stores a reference to the arguments, not a copy.** If code calls `create(messages=history)` and later appends to `history`, then `call_args.kwargs["messages"]` shows the list **as it is now**, including the later items, not as it was at the time of the call. This explains a confusing result we saw when inspecting the conversation endpoint's mock: the recorded `messages` already contained the assistant's reply. Always check history **after** the code has run completely, and remember the mock holds the same list object.

**`monkeypatch`** (a pytest fixture) temporarily replaces an attribute and **restores it automatically after the test**:

```python
monkeypatch.setattr(main, "client", fake)   # main.client is the fake during this test only
```

This works because the endpoints look up the module-level name `client` each time they run. There's no constructor injection here; monkeypatch swaps the global for the duration of the test.

**`TestClient`** sends HTTP requests to the app **in memory**, with no real server or port. It's like Spring's `MockMvc`:

```python
api = TestClient(main.app)
response = api.post("/chat", json={"message": "hi"})
```

**Fixtures** are pytest's setup mechanism. A test requests one by naming it as a parameter. `autouse=True` makes a fixture run for **every** test in the file without being named, which is how the in-memory `conversations` dict is emptied before each test (like a JUnit `@BeforeEach`). Without it, a conversation created by one test would leak into the next.

### 1.11 Docker

**The problem:** "it works on my machine". Your Mac has Python 3.12, uv and the right packages; a server might not. Docker packages the OS files, Python, the dependencies and your code into one **image** that runs the same everywhere. It's like a fat JAR that also bundles the JVM and a Linux system.

**What Docker Desktop installs on a Mac.** Containers are a Linux feature, so Docker Desktop runs a small **Linux virtual machine** in the background:

```
Your Mac (macOS)
 ├── docker            ← the CLI you type commands into
 └── Docker Desktop
      └── Linux VM
           └── Docker engine   ← builds images, runs containers
                ├── images      (stored on the VM's virtual disk)
                └── containers
```

That's why the Docker app must be running for `docker` commands to work, and why the VM's virtual disk depends on free space on the Mac's real disk.

**Image vs container:**

| | Image | Container |
| --- | --- | --- |
| What it is | A read-only snapshot of files plus a start command | A running instance of an image |
| Java analogy | A class, or a JAR on disk | An object, or a running JVM process |
| How many | One `chat-service` image (380 MB) | Any number can run from the same image |

**Layers and caching.** Each Dockerfile instruction creates a **layer**, and Docker caches each layer. If an instruction and everything before it are unchanged, Docker reuses the cached layer instead of re-running it. **Once one layer changes, every layer after it is rebuilt.**

The real layers of the `chat-service` image (from `docker history`):

| Layer | Size | Dockerfile line |
| --- | --- | --- |
| Debian + Python 3.12 | base image | `FROM python:3.12-slim` |
| uv | 49.3 MB | `COPY --from=ghcr.io/astral-sh/uv:0.11.12 ...` |
| `pyproject.toml`, `uv.lock` | 373 kB | `COPY pyproject.toml uv.lock ./` |
| **dependencies** | **86.6 MB** | `RUN uv sync --locked --no-dev --no-install-project` |
| **your code** | **45.1 kB** | `COPY app ./app` |
| `appuser` account | 69.6 kB | `RUN useradd --create-home appuser` |

The code is 45 kB and the dependencies are 86.6 MB. **That's why dependencies are copied and installed before the code**: changing a line of `main.py` only rebuilds the 45 kB layer, and the 86.6 MB layer comes from the cache. If the code were copied first, every code change would reinstall every package. It's the same trick as copying `pom.xml` and running `mvn dependency:go-offline` before copying `src/`.

**`.dockerignore`** works like `.gitignore` for the build. Files listed there are never sent to Docker, so they can't end up in the image. The most important entry is `.env`.

**Secrets at run time, not build time.** The API key is **never** inside the image; anyone with the image could extract it. It's passed when a container starts:

```bash
docker run --env-file .env -p 8000:8000 chat-service
```

**Port mapping.** `-p 8001:8000` means "port 8001 on the Mac forwards to port 8000 inside the container". The app inside must listen on `0.0.0.0` (all interfaces). Inside a container, `127.0.0.1` would mean the container's own loopback, which the forwarded traffic doesn't reach.

**A non-root user.** By default, processes in a container run as root. If an attacker broke into the app, they'd be root inside the container. `useradd` plus `USER appuser` runs the app with ordinary permissions. It's least privilege, a common security review item.

**Docker needs disk space.** Images, layers and containers live on the Linux VM's virtual disk, which is a file on your Mac. When the Mac's disk was full, that virtual disk couldn't grow, and the build failed with input/output and read-only errors. Free space and restart Docker.

---

## 2. How we built it

### 2.1 The app, the client and the store

```python
MODEL = "claude-haiku-5-5"
SYSTEM_PROMPT = "You are a concise, helpful assistant."

app = FastAPI(title="Chat Service")
client = anthropic.Anthropic()
conversations: dict[str, list[dict]] = {}
logger = logging.getLogger(__name__)
```

- `anthropic.Anthropic()` reads `ANTHROPIC_API_KEY` from the environment, which `--env-file .env` provides.
- `client` and `conversations` are module-level globals. That's simple for a small service, and it's what makes `monkeypatch.setattr(main, "client", fake)` and `main.conversations.clear()` work in tests.

### 2.2 One place builds every Claude request

```python
def claude_args(messages: list[dict], max_tokens: int) -> dict:
    """The arguments every Claude call in this service uses."""
    return {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
```

It's used as `client.messages.create(**claude_args(messages, request.max_tokens))`. The `**` unpacks the dict into keyword arguments, so `f(**{"a": 1})` is the same as `f(a=1)`.

Every endpoint goes through this one function, so the model, system prompt and limits **can't drift apart** between `/chat`, `/chat/stream` and conversations. Change the model in one place and every endpoint follows.

### 2.3 Health and creating a conversation

```python
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/conversations", status_code=201)
def create_conversation() -> dict:
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = []
    return {"conversation_id": conversation_id}
```

`uuid4()` makes a random ID that callers can't guess. Each new conversation starts as an empty list.

### 2.4 A single question: `/chat`

```python
@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    messages = [{"role": "user", "content": request.message}]
    try:
        response = client.messages.create(**claude_args(messages, request.max_tokens))
    except anthropic.APIError as e:
        raise to_http_error(e) from e
    reply = "".join(block.text for block in response.content if block.type == "text")
    return ChatResponse(
        reply=reply,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
```

- Claude's `response.content` is a **list of blocks**. Only `text` blocks are joined into the reply; other block types are skipped. A test checks this.
- Token usage goes back to the caller so cost is visible.

### 2.5 A conversation turn

```python
@app.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: str, request: ChatRequest) -> ChatResponse:
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    history = conversations[conversation_id]
    history.append({"role": "user", "content": request.message})
    try:
        response = client.messages.create(**claude_args(history, request.max_tokens))
    except anthropic.APIError as e:
        history.pop()
        raise to_http_error(e) from e
    reply = "".join(block.text for block in response.content if block.type == "text")
    history.append({"role": "assistant", "content": reply})
    return ChatResponse(...)
```

The order of operations is the important part:
1. **404 first**, before any work.
2. **Append the user message**, then send the **whole history**.
3. **On failure, `history.pop()`** removes the user message again. Otherwise the history would hold a message that was never answered, and the next turn would send two user messages in a row.
4. **On success, append the assistant's reply**, so the next turn includes it.

`history` is the same list object as the one stored in `conversations`, so appending to it updates the stored conversation directly. There's no "save" step.

### 2.6 Streaming: the generator and the endpoint

```python
def stream_reply(request: ChatRequest):
    messages = [{"role": "user", "content": request.message}]
    try:
        with client.messages.stream(
            **claude_args(messages, request.max_tokens)
        ) as stream:
            for text in stream.text_stream:
                payload = json.dumps({"text": text})
                yield f"data: {payload}\n\n"
    except anthropic.APIError as e:
        detail = to_http_error(e).detail
        error_detail = json.dumps({"error": detail})
        yield f"data: {error_detail}\n\n"
        return

    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(stream_reply(request), media_type="text/event-stream")
```

- `client.messages.stream(...)` is used with `with`, so the connection to Claude is closed properly even if the client disconnects.
- `stream.text_stream` yields pieces of text as Claude generates them.
- `json.dumps` builds the JSON **before** it goes into the f-string (see Gotchas).
- On an error, the generator yields an error event and then `return`s, so **no `[DONE]`** follows.
- It reuses `to_http_error(e).detail`, so streamed errors use the same messages as normal ones.
- Validation still happens first: `request: ChatRequest` is checked **before** `chat_stream` runs, so an empty message gets a normal 422, not a stream.

### 2.7 Error mapping

```python
def to_http_error(error: anthropic.APIError) -> HTTPException:
    """Map a Claude SDK error to the HTTP error this service returns."""
    if isinstance(error, anthropic.RateLimitError):
        return HTTPException(
            status_code=503, detail="Claude is busy. Try again shortly."
        )
    if isinstance(error, anthropic.APIConnectionError):
        return HTTPException(status_code=502, detail="API connection timeout.")
    if isinstance(error, anthropic.AuthenticationError):
        logger.error("Claude rejected the API key. Check ANTHROPIC_API_KEY.")
        return HTTPException(status_code=500, detail="Server error")
    return HTTPException(status_code=502, detail="Claude returned an unexpected error")
```

It **returns** an exception rather than raising it, so the same mapping serves both cases: the normal endpoints `raise` it, and the stream uses its `.detail`.

### 2.8 Tests

The fake client fixture:

```python
@pytest.fixture
def fake_client(monkeypatch):
    """Replace the real Claude client with a fake that returns a fixed reply."""
    fake = MagicMock()
    fake.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello from a fake Claude")],
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    monkeypatch.setattr(main, "client", fake)
    return fake
```

`SimpleNamespace(a=1)` creates a throwaway object with attribute `.a`, which is enough to look like the SDK's response object.

Building real SDK exceptions for `side_effect`. The SDK's error classes need an HTTP request or response object, which the tests build with the HTTP library:

```python
FAKE_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

def rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=FAKE_REQUEST), body=None
    )
```

Checking exactly what was sent to Claude:

```python
def test_chat_sends_message_to_claude(api, fake_client):
    api.post("/chat", json={"message": "What is RAG?", "max_tokens": 200})
    sent = fake_client.messages.create.call_args.kwargs
    assert sent["max_tokens"] == 200
    assert sent["messages"] == [{"role": "user", "content": "What is RAG?"}]
```

Validation boundaries, with `parametrize` (one test function, four cases):

```python
@pytest.mark.parametrize(
    ("max_tokens", "expected_status"), [(1, 200), (4096, 200), (0, 422), (4097, 422)]
)
def test_max_tokens_boundaries(api, fake_client, max_tokens, expected_status):
    response = api.post("/chat", json={"message": "hi", "max_tokens": max_tokens})
    assert response.status_code == expected_status
    if expected_status == 200:
        fake_client.messages.create.assert_called_once()
    else:
        fake_client.messages.create.assert_not_called()
```

Testing on both sides of each limit (0 and 1, 4096 and 4097) is what caught the `gt`/`lt` bug.

Faking a stream. `messages.stream(...)` is used as a context manager, so the fake has to follow the path `stream(...)` → `__enter__()` → `text_stream`:

```python
fake_stream = fake_client.messages.stream.return_value.__enter__.return_value
fake_stream.text_stream = ["Hel", "lo"]
```

Then the test asserts the **exact bytes**: `'data: {"text": "Hel"}\n\ndata: {"text": "lo"}\n\ndata: [DONE]\n\n'`.

A clean state before every test:

```python
@pytest.fixture(autouse=True)
def empty_conversations():
    """Start every test with no stored conversations."""
    main.conversations.clear()
```

And the test that a failed message isn't kept: make `create` raise a rate-limit error, send a message, then assert a 503 **and** an empty history.

The suite has 17 tests, needs no API key or network, and runs in under a second.

### 2.9 The Dockerfile

```dockerfile
FROM python:3.12-slim

# uv, copied from its official image (pinned to the version used in development)
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first, in their own layer, so code changes don't reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Then the application code
COPY app ./app

EXPOSE 8000

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
```

| Line | Why |
| --- | --- |
| `python:3.12-slim` | A small Debian image with Python, without build tools |
| `COPY --from=...uv` | Takes just the uv binary from uv's official image, pinned to a version |
| `--locked` | Fail if `uv.lock` doesn't match `pyproject.toml`, so the image gets exactly the tested versions |
| `--no-dev` | Skip pytest, ruff and pyright; production doesn't need them |
| `--no-install-project` | Install only dependencies, because the code isn't copied yet |
| `EXPOSE 8000` | Documentation only; it doesn't publish the port (`-p` does) |
| `--no-sync` | Don't let `uv run` re-check or re-install packages at startup |
| `--host 0.0.0.0` | Listen on all interfaces so port mapping works |

`.dockerignore`:

```
.env
.venv
.git
.vscode
__pycache__
.pytest_cache
.ruff_cache
tests
```

Build and run:

```bash
docker build -t chat-service .
docker run --env-file .env -p 8000:8000 chat-service
```

When we checked the running container, health and `/chat` worked, the process ran as `appuser`, and there was no `.env` inside the image.

---

## 3. Key points

- FastAPI maps decorators to routes; Pydantic models are both the validation and the documentation, and `/docs` comes for free.
- Path parameters come from `{name}` in the route, and the body comes from a Pydantic-typed parameter.
- Invalid input returns **422 before your function runs**, so bad requests cost nothing.
- 201 for create, 404 for an unknown resource, 503 for "retry later", 502 for "upstream failed", 500 for "we're misconfigured".
- Log the real cause server-side; send callers a generic message for internal errors.
- **Build every Claude request in one function** so endpoints can't drift apart.
- The Claude API is stateless: send the full history every turn. Input tokens grow per turn, and the history must alternate user/assistant.
- On a failed turn, remove the user message (`history.pop()`) so the history stays consistent.
- An in-memory store is lost on restart and can't be shared across processes; production uses a database.
- SSE: `data: <json>\n\n` per event, `[DONE]` at the end; a generator plus `StreamingResponse` sends it.
- Mid-stream errors become an error event, because the 200 status has already been sent.
- MagicMock: `return_value`, `side_effect`, `call_args.kwargs`, `assert_not_called`. It stores **references**, not copies.
- `monkeypatch` swaps a global for one test and restores it; an `autouse` fixture resets shared state.
- Image = the package, container = a running instance. Put dependencies before code to keep the layer cache useful.
- Secrets go in at **run time** (`--env-file`), never in the image; `.dockerignore` keeps `.env` out. Run as a non-root user, and listen on `0.0.0.0`.

---

## 4. Gotchas we hit

| What happened | Cause | Fix / lesson |
| --- | --- | --- |
| JSON in the SSE line broke | Writing JSON by hand inside an f-string: `{` and `}` mean "placeholder" in an f-string | Build the JSON with `json.dumps(...)` into a variable first, then put the variable in the f-string |
| Endpoints disappeared or were commented out while editing | Unfinished edits | The tests caught it. Don't leave commented-out code; delete it or finish it. |
| Calls broke after `claude_args` got a new signature | Old call sites had an extra `)` and were missing `**` | Search for every caller when a signature changes; `**` is needed to unpack the dict into keyword arguments |
| `max_tokens=1` was rejected and 4096 failed | `gt`/`lt` (strict) were used instead of `ge`/`le` | Boundary tests (0, 1, 4096, 4097) exposed it. Test both sides of every limit. |
| Claude didn't remember its own answers | The assistant reply wasn't appended to the history | Each turn adds **two** messages: user and assistant |
| A GET route didn't work | A missing `}` in `{conversation_id`, invalid `GET /...` syntax, and a second function also named `chat` | Paths need balanced braces, routes are declared with `@app.get(...)`, and handler names must be unique in the module (a second `def chat` silently replaces the first) |
| The recorded mock arguments showed "too many" messages | MagicMock keeps a reference to the `history` list, which was appended to after the call | Inspect after the code finishes, and remember that `call_args` shows the list's current contents |
| Pylance couldn't resolve `fastapi` or `anthropic` imports | VS Code was using a global Python, not the project's `.venv` | Set `python.defaultInterpreterPath` to `${workspaceFolder}/.venv/bin/python` in `.vscode/settings.json` (or choose "Enter interpreter path") |
| `docker build` failed with I/O and read-only errors | The Mac disk was full (about 117 MB free), so Docker's VM disk couldn't grow | Free disk space (to about 24 GB) and restart Docker Desktop |

---

## 5. Revision questions

1. How does a Python decorator like `@app.get("/health")` differ from a Spring annotation like `@GetMapping`?
2. What do uvicorn and ASGI do? What are their Spring equivalents?
3. A request to `POST /chat` has `"max_tokens": 0`. What status comes back, and does Claude get called? Why does that matter for cost?
4. When would you return 502, 503 and 500? Which tells the caller to retry?
5. Why does the service log the real cause of an authentication error but send the caller "Server error"?
6. Why is there a `claude_args()` helper instead of building the arguments in each endpoint?
7. The Claude API is stateless. How does the service hold a conversation, and what does that do to cost as a conversation grows?
8. Why does `send_message` call `history.pop()` when Claude fails?
9. Give two limitations of storing conversations in a Python dict.
10. Write out one SSE event carrying the text `Hi`. Why must each event end with `\n\n`?
11. Why can't a streaming endpoint return a 503 when Claude fails halfway through? What does it do instead, and how can the client tell?
12. What's the difference between `return_value` and `side_effect` on a MagicMock?
13. A test checks `call_args.kwargs["messages"]` after a conversation turn and sees the assistant's reply in it, even though the reply didn't exist when `create` was called. Why?
14. In the Dockerfile, why is `COPY pyproject.toml uv.lock` plus `uv sync` placed before `COPY app ./app`? Use the layer sizes in your answer.
15. Why is the API key passed with `--env-file` at `docker run`, and what stops `.env` from being copied into the image anyway?

---

## Answers

1. A decorator is a function that runs when the module loads and registers the handler in the app's routing table. An annotation is passive metadata that Spring reads through reflection at startup. Both end with a URL mapped to a function.
2. uvicorn is the HTTP server that listens on the port. ASGI is the standard interface uvicorn uses to pass requests to the FastAPI app. The equivalents are embedded Tomcat and the Servlet API.
3. 422 Unprocessable Entity, because `ge=1` is violated. Claude is **not** called: validation runs before the handler. Invalid requests can never spend API tokens.
4. 503: temporary overload such as a rate limit, so retry shortly. 502: the upstream failed or couldn't be reached. 500: our own server is broken (for example, a bad API key), so retrying won't help. 503 is the "retry" signal.
5. The real cause is an internal configuration detail that callers can't fix and shouldn't see. The log gives the operator what they need, and the response leaks nothing.
6. So the model, system prompt and limits are defined once and can't drift apart between `/chat`, `/chat/stream` and conversations. One change updates every endpoint.
7. It stores the history per `conversation_id` and sends the **whole** history on every turn. Input tokens, and therefore cost and latency, grow with each turn.
8. The user message was appended before the call. If the call fails and it stays, the history would end with an unanswered user message, and the next turn would send two user messages in a row.
9. Everything is lost when the server restarts, and it can't be shared between several processes or machines. (Memory use also grows without limit.)
10. `data: {"text": "Hi"}` followed by `\n\n`. The blank line is what marks the end of an event in the SSE format; without it the client doesn't know the event is complete.
11. The `200 OK` status and headers were sent before the first chunk, and HTTP can't change them afterwards. The service sends a final `data: {"error": "..."}` event and **no** `[DONE]`. The client checks for an `error` key or a missing `[DONE]`.
12. `return_value` is what the mocked call returns. `side_effect` set to an exception makes the call raise it (it can also be a function or a list of values).
13. MagicMock stores a **reference** to the `history` list, not a copy. The code appended the assistant reply to the same list after the call, so `call_args` shows the list's current contents.
14. Docker rebuilds every layer after the first one that changed. Dependencies (86.6 MB) change rarely and the code (45.1 kB) changes often. With dependencies first, a code change rebuilds only the 45 kB layer and reuses the cached 86.6 MB one. The other way round, every code change would reinstall all packages.
15. Anything inside an image can be extracted by whoever has the image, so secrets must only enter at run time. `.dockerignore` lists `.env`, so it's never sent to the build and can't be copied in by `COPY`.
