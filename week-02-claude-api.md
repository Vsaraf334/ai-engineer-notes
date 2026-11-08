# Week 2: Calling LLMs with the Claude API

This week went from "an LLM is a website I type into" to "an LLM is an HTTP API my code calls". It covered a single API call, then a multi-turn chat in the terminal that streams its replies, remembers the conversation, tracks the cost of every turn, and survives network and API failures. It also introduced Pydantic, the validation library that FastAPI (week 3) uses for every request and response.

**Built in:** `rag-docs/scratch/first_call.py` (one call), `scratch/chat.py` (streaming chat CLI), `scratch/classes.py` (`Conversation`, `Usage`), `scratch/pydantic_drill.py`.

---

## 1. Concepts

### 1.1 What an LLM API call is

A large language model (LLM) such as Claude runs on Anthropic's servers. Your program doesn't run the model; it sends an HTTP request and receives the model's reply as JSON.

```
your Python program                        Anthropic's servers
┌───────────────────┐   HTTPS POST /v1/messages   ┌──────────────────────┐
│ client.messages   │ ──────────────────────────► │  Claude model         │
│   .create(...)    │   model, max_tokens,        │  reads the input,     │
│                   │   system, messages, API key │  generates a reply    │
│                   │ ◄────────────────────────── │                       │
│ response object   │   JSON: content, usage,     └──────────────────────┘
└───────────────────┘   stop_reason
```

The `anthropic` Python package (the "SDK", software development kit) wraps that HTTP call. It builds the request, adds your API key, retries temporary failures, and turns the JSON reply into Python objects. In Java terms it plays the role of a typed HTTP client such as a Feign or Retrofit client: you call a method, and the SDK handles the HTTP.

### 1.2 Tokens

Models don't read characters or words. They read **tokens**: chunks of text from a fixed vocabulary. A token is often a whole common word, part of a longer word, or a piece of punctuation.

```
"Explain RAG in two sentences"  →  [Explain] [ R] [AG] [ in] [ two] [ sentences]   (illustration)
```

The exact split depends on the model's tokenizer, so you don't predict it by hand; the API reports the count back to you.

There are two kinds of token on every call:

| | Input tokens | Output tokens |
| --- | --- | --- |
| What | Everything you send: system prompt + all messages | Everything the model writes back |
| Who controls the size | You (what you send) | The model, capped by `max_tokens` |
| Price | Lower per token | Higher per token (generation is the expensive part) |

**Cost is per token** because the work the servers do grows with the number of tokens processed and generated. Our first call (`first_call.py`) asked a ten-word question and used **26 input tokens and 110 output tokens**. The input count also includes some formatting the API adds around your message.

### 1.3 The context window

The **context window** is the maximum number of tokens the model can consider in one call: input and output together. Everything the model "knows" about your conversation must fit inside it. Nothing outside the request exists for the model.

```
┌──────────────────────── context window ────────────────────────┐
│ system prompt │ msg 1 │ msg 2 │ ... │ latest msg │ reply ...    │
└────────────────────────────────────────────────────────────────┘
        input tokens (you send)                  output tokens
```

This is why long conversations eventually need trimming or summarising, and why RAG (weeks 4–5) sends only the *relevant* chunks of a document rather than the whole document.

### 1.4 The Messages API

One call to `client.messages.create(...)` takes these main parameters:

| Parameter | Meaning | Our value |
| --- | --- | --- |
| `model` | Which Claude model answers | `"claude-haiku-5-5"` (small, fast, cheap: good for development) |
| `max_tokens` | Hard cap on **output** tokens for this reply | `1024` |
| `system` | The system prompt: standing instructions and persona | `"You are a concise assistant for a Java developer learning Python."` |
| `messages` | The conversation, as a list of `{"role", "content"}` dicts | user and assistant turns |

**Roles.** Each message has a role:

- `"user"`: what the person (or your program) says.
- `"assistant"`: what Claude said earlier.

Messages alternate, and the list normally ends with a `user` message: the one Claude should answer now.

**System prompt vs messages.** The system prompt is separate from the message list. It sets *how* Claude should behave for the whole conversation ("be concise", "answer only from the documents"), while messages are the conversation itself. It's like the configuration of a service versus the requests it handles.

### 1.5 The API is stateless

This is the most important idea of the week.

**The API remembers nothing between calls.** Each request is independent, like a stateless REST endpoint with no session. If you want Claude to remember that your name is Shady, **you** must send the earlier messages again on every turn.

```
Turn 1 sends:  [user: "my name is Shady"]
Turn 2 sends:  [user: "my name is Shady", assistant: "Hi Shady...", user: "what is my name?"]
Turn 3 sends:  [... all 4 previous messages ..., user: "..."]
```

Consequence: **input tokens grow every turn**, because the whole history is re-sent. Illustration with made-up round numbers (each user message ~10 tokens, each reply ~40, system prompt ~20):

| Turn | Sent this turn | Input tokens (approx.) |
| --- | --- | --- |
| 1 | system + 1 message | 30 |
| 2 | system + 3 messages | 80 |
| 3 | system + 5 messages | 130 |
| 10 | system + 19 messages | 480 |

Real turns from our chat runs showed the same pattern: early turns around **31–36** input tokens, later turns **91**, **96**, then **177** input tokens as the history grew.

So a long conversation gets slower and more expensive per turn, and eventually hits the context window. Real apps handle this by trimming old turns, summarising them, or using prompt caching.

In Java terms: there is no `HttpSession`. Your client holds the state, like a client that resends its whole shopping cart with every request.

### 1.6 Anatomy of a response

```python
response.content       # list of content blocks
response.usage         # input_tokens, output_tokens
response.stop_reason   # why generation stopped
```

**Content blocks.** `response.content` is a **list**, not a string, because a reply can contain different kinds of block (text, and from week 8, tool calls; some models also return thinking blocks). You keep only the text blocks:

```python
reply = "".join(b.text for b in response.content if b.type == "text")
```

This filter matters: code that assumes `response.content[0].text` breaks as soon as the first block isn't text.

**`usage`** tells you exactly how many input and output tokens the call used. It's the basis for cost tracking.

**`stop_reason`** tells you why Claude stopped writing:

| `stop_reason` | Meaning | What to do |
| --- | --- | --- |
| `"end_turn"` | Claude finished its answer naturally | Nothing: this is normal |
| `"max_tokens"` | It hit your `max_tokens` cap mid-answer | The reply is **cut off**: warn, or raise `max_tokens` |

Always check for `"max_tokens"`. A truncated reply looks normal at a glance but stops mid-sentence. In week 6 this exact problem broke the LLM judge: its JSON was cut off halfway.

### 1.7 Streaming

Without streaming, you wait for the **whole** reply, then it appears at once. For a long answer that's several seconds of blank screen.

With streaming, the server sends the text **in small pieces as it's generated**. The user sees the first words almost immediately.

```
Non-streaming:  [ ......... waiting ......... ] → whole reply appears
Streaming:      [Py][thon][ lists][ are][ like][ Java's][ ArrayList]...  → appears word by word
```

The important measure is **time to first token**: how long until the user sees *something*. Streaming doesn't make the total generation faster, but it makes the app *feel* much faster.

The SDK pattern:

```python
with client.messages.stream(...) as stream:     # opens the streaming connection
    for text in stream.text_stream:              # yields text pieces as they arrive
        print(text, end="", flush=True)
    response = stream.get_final_message()        # the complete message, with usage and stop_reason
```

- `with ... as stream` is Python's try-with-resources: the connection is closed when the block ends, even after an error.
- `stream.text_stream` is an iterator of text pieces (it uses `yield` internally, like a Java `Iterator` that produces values lazily).
- `get_final_message()` gives you the same kind of object `create()` returns, so `usage` and `stop_reason` still work after streaming.
- `print(..., end="", flush=True)`: `end=""` stops `print` adding a newline after each piece, and `flush=True` pushes each piece to the screen immediately instead of buffering it.

### 1.8 Tracking cost

Cost per call = input tokens × input price + output tokens × output price, with prices quoted **per million tokens**.

In `chat.py` the prices are constants next to the model name:

```python
INPUT_PRICE = 0.10  # $ per million input tokens
OUTPUT_PRICE = 0.50  # $ per million output tokens
```

And the `Usage` class does the maths:

```python
def cost(self, input_price_per_m: float, output_price_per_m: float) -> float:
    return (
        self.input_tokens * input_price_per_m
        + self.output_tokens * output_price_per_m
    ) / 1_000_000
```

Real turns cost fractions of a cent: `[36 in, 45 out, $0.00003]`, `[177 in, 23 out, $0.00003]`. The point of tracking it isn't this week's bill; it's knowing your cost **per request** before you serve thousands of users. Always check current prices on Anthropic's pricing page before relying on these numbers.

### 1.9 Errors and retries

Network calls fail. The SDK raises a different exception class for each kind of failure, organised as a hierarchy (verified against the installed SDK):

```
Exception
└── AnthropicError
    └── APIError
        ├── APIConnectionError            couldn't reach the server (no internet, DNS, ...)
        │   └── APITimeoutError           the request took too long
        └── APIStatusError                the server answered with an error status code
            ├── BadRequestError           400  the request itself is invalid
            ├── AuthenticationError       401  missing or wrong API key
            ├── NotFoundError             404  e.g. a model name that doesn't exist
            ├── RateLimitError            429  too many requests right now
            └── InternalServerError       5xx  a problem on Anthropic's side
```

**Automatic retries.** The SDK already retries temporary failures (connection errors, rate limits, server errors) **2 times by default** (`anthropic.DEFAULT_MAX_RETRIES == 2`), waiting a little longer each time. Your `except` only runs after those retries have failed.

**Permanent vs temporary errors.** The key design question is "will trying again help?"

| Error | Kind | What `chat.py` does |
| --- | --- | --- |
| `AuthenticationError` | Permanent: the key is wrong | Print a message and **stop** (`break`) |
| `NotFoundError` | Permanent: the model name is wrong | Print a message and **stop** |
| `RateLimitError` | Temporary | Remove the unanswered message, let the user try again (`continue`) |
| `APIConnectionError` | Temporary | Same |
| `APIStatusError` (anything else) | Probably temporary | Same |

**Order matters.** Python checks `except` clauses top to bottom and uses the **first** match, exactly like Java's `catch` blocks. Specific classes (`RateLimitError`) must come before their parents (`APIStatusError`). Otherwise the general clause catches everything and the specific ones never run.

**Pop the failed message.** Before the call, the user's message was added to the history. If the call fails, that message has no reply. Leaving it there would break the alternating user/assistant pattern on the next turn, so the code removes it with `conversation.messages.pop()`.

### 1.10 Keeping the API key safe

The SDK reads the key from the `ANTHROPIC_API_KEY` environment variable:

```python
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
```

The key lives in a `.env` file in the project folder, which is listed in `.gitignore` so it's never committed:

```
ANTHROPIC_API_KEY=<your key>
```

`uv run --env-file .env python scratch/chat.py` loads that file into the environment for that one command. Without `--env-file`, the variable isn't set and the SDK can't authenticate.

Rules: never paste the key into code, chat, commits or screenshots; set a spend limit in the Claude Console; if a key ever leaks, revoke it and create a new one.

### 1.11 Pydantic: validating data at the boundary

Pydantic turns a class with type hints into a **validated data model**. It's like a Java record plus Bean Validation (`@NotNull`, `@Min`, `@Max`) plus Jackson, in one.

```python
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    model: str = "claude-haiku-5-5"
    max_tokens: int = Field(default=1024, ge=1, le=4096)
```

| Feature | Meaning | Java equivalent |
| --- | --- | --- |
| `message: str` | Required field of type `str` | `@NotNull String message` |
| `= "claude-haiku-5-5"` | Default value | Field initialiser |
| `Field(ge=1, le=4096)` | 1 ≤ value ≤ 4096 (`ge` = greater or equal, `le` = less or equal) | `@Min(1) @Max(4096)` |
| `Field(min_length=1)` | At least 1 character | `@Size(min = 1)` |
| `Literal["user", "assistant"]` | Only these exact values | An `enum` |
| `@field_validator("question")` | Custom check or cleanup for one field | A custom `ConstraintValidator` |

**Type conversion.** `ChatRequest(message="hi", max_tokens="512")` gives `max_tokens == 512`, an `int`. Pydantic converts compatible values instead of failing.

**Reading and writing:**

| Method | Does |
| --- | --- |
| `Model(field=...)` | Build and validate from Python values |
| `Model.model_validate(dict)` | Build and validate from a dict (including nested dicts → nested models) |
| `Model.model_validate_json(str)` | Parse a JSON **string** and validate it |
| `obj.model_dump()` | Turn the model back into a plain dict |

Invalid data raises `ValidationError`, with a message saying which field failed and why.

**Why "validate at the boundary".** Data entering your program from outside (an HTTP request, a file, **an LLM's output**) can be anything. Validate it once, where it enters, and the rest of your code can trust it. You did this in chat-service (`Field(ge=1, le=4096)` on requests) and in week 6 (the judge's JSON parsed into a `Literal` verdict).

### 1.12 The `Conversation` class

The chat keeps its history in a small class from `classes.py`:

```python
class Conversation:
    system: str
    messages: list[Message]

    def __init__(self, system: str = ""):
        self.system = system
        self.messages = []

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))

    def last(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def to_api(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)
```

- **`system`** is stored separately because the API takes it as its own parameter, not as a message.
- **`to_api()`** converts the `Message` objects into the exact list-of-dicts format the API expects. It's an adapter between your domain model and the external API, like a DTO mapper in Java.
- **`last()`** returns the latest message, or `None` for an empty conversation. `messages[-1]` means "the last item".
- **`__len__`** makes `len(conversation)` work. Methods named `__like_this__` ("dunder" methods) hook your class into Python's built-in operations, the way implementing `Iterable` or overriding `equals` hooks into Java's.
- **`self.messages = []` inside `__init__`**, not as a class-level default, so every conversation gets its own list. A shared mutable default would leak messages between objects.

---

## 2. How we built it

### 2.1 The first call: `first_call.py`

```python
import anthropic

MODEL = "claude-haiku-5-5"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain RAG in two sentences for a Java developer."}
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)

print("---")
print("stop reason:", response.stop_reason)
print("input tokens:", response.usage.input_tokens)
print("output tokens:", response.usage.output_tokens)
```

Run with `uv run --env-file .env python scratch/first_call.py`. Output ended with `stop reason: end_turn`, 26 input tokens and 110 output tokens.

### 2.2 The chat loop: `chat.py`

The structure of each turn:

```
read input ─► add user message ─► stream reply (whole history sent) ─► add assistant reply ─► print cost
                     ▲                        │ on error
                     └──── pop the message ◄──┘ (temporary) or stop (permanent)
```

The streaming call and error handling:

```python
    conversation.add("user", user_input)

    print("claude> ", end="", flush=True)
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=conversation.system,
            messages=conversation.to_api(),
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()
        print()
    # Permanent errors: retrying won't help, so stop the chat.
    except anthropic.AuthenticationError:
        print("\nAuthentication failed. Check ANTHROPIC_API_KEY in .env.")
        break
    except anthropic.NotFoundError:
        print(f"\nModel '{MODEL}' not found. Check the MODEL setting.")
        break
    # Temporary errors: drop the unanswered message so the history stays valid,
    # then let the user try again.
    except anthropic.RateLimitError:
        print("\nClaude is busy right now. Wait a moment and try again.")
        conversation.messages.pop()
        continue
    except anthropic.APIConnectionError:
        print("\nConnection failed. Check your internet and try again.")
        conversation.messages.pop()
        continue
    except anthropic.APIStatusError as e:
        print(f"\nThe API returned an error ({e.status_code}). Try again.")
        conversation.messages.pop()
        continue
```

- `break` leaves the `while True` loop (end the chat). `continue` skips the rest of this turn and asks for the next input.
- `APIStatusError` comes **last** among the status errors because it's the parent of `AuthenticationError`, `NotFoundError` and `RateLimitError`.

After a successful call: save the reply, then report cost and truncation:

```python
    reply = "".join(b.text for b in response.content if b.type == "text")
    conversation.add("assistant", reply)

    usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
    cost = usage.cost(INPUT_PRICE, OUTPUT_PRICE)
    total_cost += cost
    print(f"[{usage.input_tokens} in, {usage.output_tokens} out, ${cost:.5f}]")
    if response.stop_reason == "max_tokens":
        print("Warning: response was cut off due to max_tokens limit.")
```

`conversation.add("assistant", reply)` is what gives the chat memory. Remove that line and Claude sees your old questions but not its own old answers.

A real session:

```
you> my name is Shady
claude> Hi Shady, nice to meet you! I'm here to help you as you learn Python, ...
you> what is my name ?
claude> Your name is Shady, as you told me.
you> quit
```

### 2.3 Pydantic drill highlights: `pydantic_drill.py`

A closed set of values and nested models:

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatHistory(BaseModel):
    messages: list[ChatMessage] = []

    def to_api(self) -> list[dict]:
        return [msg.model_dump() for msg in self.messages]
```

`ChatHistory.model_validate({"messages": [{"role": "bot", ...}]})` raises `ValidationError`: the nested dict is validated too. Unlike a plain class, `= []` is safe here, because Pydantic copies the default for each object.

A custom validator that cleans and rejects:

```python
class Question(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if value.strip():
            return value.strip()
        raise ValueError("question must not be blank")
```

The validator **returns** the value to store (here, stripped), or raises to reject it.

Validating JSON produced by an LLM:

```python
class Answer(BaseModel):
    answer: str
    sources: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


def parse_answer(raw_json: str) -> Answer:
    return Answer.model_validate_json(raw_json)
```

`parse_answer("Sure! Here is the answer: ...")` raises `ValidationError`: LLMs don't always return clean JSON, so parse defensively.

---

## 3. Key points

- An LLM API call is an HTTPS request; the SDK wraps it like a typed HTTP client.
- Cost and limits are measured in **tokens**. Input tokens (what you send) are cheaper than output tokens (what the model writes).
- The **context window** caps input + output; anything not in the request doesn't exist for the model.
- A call takes `model`, `max_tokens`, `system` and `messages` (roles `user` / `assistant`).
- **The API is stateless.** You re-send the whole history every turn, so input tokens grow every turn.
- `response.content` is a **list of blocks**; keep only `type == "text"`.
- `usage` gives exact token counts; `stop_reason == "max_tokens"` means the reply was **cut off**.
- Streaming improves **time to first token**; use `messages.stream`, `text_stream` and `get_final_message()`.
- Prices are per million tokens: cost = in × in_price + out × out_price, divided by 1,000,000.
- The SDK retries temporary failures **2 times** by default before raising.
- Stop on permanent errors (401 auth, 404 model); let the user retry on temporary ones (429, connection, 5xx).
- Order `except` clauses from most specific to most general.
- After a failed call, **pop** the unanswered user message so the history stays valid.
- The API key goes in `.env` (git-ignored), loaded with `uv run --env-file .env`.
- Pydantic = typed fields + validation + JSON parsing. **Validate at the boundary**, including LLM output.

---

## 4. Gotchas we hit

| What happened | Cause | Fix |
| --- | --- | --- |
| Running `chat.py` printed 20 `PASS` lines before the chat started | `from classes import Conversation` runs every top-level line in `classes.py`, including its tests | Put the tests under `if __name__ == "__main__":` so they run only when the file is run directly |
| Authentication error on a run that "should work" | The script was run without `--env-file .env`, so `ANTHROPIC_API_KEY` wasn't set | Always `uv run --env-file .env python ...` |
| Testing the error handling with `MODEL = "claude-haiku-44"` gave a 404 | A model name that doesn't exist → `NotFoundError` | Caught as a permanent error (stop the chat); then the real model name was put back |
| Cost tracking needed fixing in review | The cost wasn't computed exactly once per turn and accumulated, and the prices weren't named | Compute `cost` once per turn, `total_cost += cost`, and keep `INPUT_PRICE` / `OUTPUT_PRICE` as constants next to `MODEL` |
| A general `except` hid the specific ones | Parent class listed before its subclasses | Most specific `except` first, `APIStatusError` last |
| `(base)` in every terminal prompt; a stray pytest from Anaconda on day 1 | Anaconda's base environment auto-activates | `uv run` ignores it; optionally `conda config --set auto_activate_base false` |

---

## 5. Revision questions

1. What does the SDK do for you that a raw HTTP call wouldn't?
2. Why are output tokens priced higher than input tokens?
3. What is the context window, and what happens to information that isn't in the request?
4. What's the difference between the `system` parameter and a `user` message?
5. The API is stateless. How does a chat app still "remember" earlier turns, and what does that cost?
6. Why is `response.content` a list, and why filter on `type == "text"`?
7. What does `stop_reason == "max_tokens"` mean, and what should your code do about it?
8. What does streaming improve, and what doesn't it improve?
9. What do `end=""` and `flush=True` do in the streaming `print`?
10. A turn used 2,000 input and 500 output tokens. Using `chat.py`'s prices, what did it cost?
11. Which errors should end the chat, and which should let the user try again? Why?
12. Why must `except anthropic.APIStatusError` come after `except anthropic.RateLimitError`?
13. Why does `chat.py` call `conversation.messages.pop()` after a temporary error?
14. Where does the API key live, how does it reach the SDK, and what keeps it out of git?
15. What does "validate at the boundary" mean, and why does it apply to LLM output too?

---

## Answers

1. It builds the request and headers, adds the API key, retries temporary failures (2 times by default), raises typed exceptions, and converts the JSON reply into Python objects (`response.content`, `usage`, `stop_reason`).
2. Generating each output token is the expensive part of the model's work; input tokens are processed together, while output tokens are produced one at a time.
3. The maximum number of tokens (input + output) the model can consider in one call. Anything not in the request doesn't exist for the model; it has no memory outside it.
4. `system` holds standing instructions for the whole conversation (how to behave); `user` messages are the conversation turns themselves. The API takes `system` as a separate parameter.
5. The app keeps the history itself and sends all of it on every call. Input tokens grow each turn, so each turn costs more and is slower, until the history must be trimmed or summarised.
6. A reply can contain several kinds of block (text, tool calls, thinking). Only text blocks have the answer text; assuming `content[0]` is text breaks when it isn't.
7. The reply hit your output cap and was cut off mid-answer. Warn the user, raise `max_tokens`, or treat the output as incomplete (e.g. don't parse cut-off JSON).
8. It improves time to first token (perceived speed). It doesn't make the total generation time shorter or cheaper.
9. `end=""` stops `print` adding a newline after each piece; `flush=True` writes each piece to the screen immediately instead of buffering it.
10. (2,000 × 0.10 + 500 × 0.50) / 1,000,000 = (200 + 250) / 1,000,000 = **$0.00045**.
11. End on permanent errors: `AuthenticationError` (bad key) and `NotFoundError` (bad model name), because retrying gives the same result. Let the user retry on temporary ones: `RateLimitError`, `APIConnectionError`, other `APIStatusError`s.
12. `APIStatusError` is the parent class. Python uses the first matching `except`, so if the parent came first it would catch rate limits too and the specific handler would never run.
13. The user's message was added before the call. With no reply, leaving it would break the user/assistant alternation and confuse the next turn, so it's removed.
14. In `.env` as `ANTHROPIC_API_KEY=...`; `uv run --env-file .env` loads it into the environment, and `anthropic.Anthropic()` reads it from there. `.env` is listed in `.gitignore`.
15. Check data once, where it enters the program (requests, files, API responses), so the rest of the code can trust it. LLM output is external, unpredictable data: it can be invalid JSON or out-of-range values, so parse it with a schema such as a Pydantic model.
