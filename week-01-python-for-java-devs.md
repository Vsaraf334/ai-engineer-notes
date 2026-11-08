# Week 1: Python for Java developers

This week was about learning to write Python the way Python developers do, starting from a Java background. Most ideas map directly: a `dict` is a `HashMap` and a `list` is an `ArrayList`. The differences are in typing (types are checked by tools, not by the compiler), syntax (comprehensions, unpacking, slicing), and tooling (uv instead of Maven, pytest instead of JUnit). Every topic was practised in a drill file with PASS/FAIL tests, and then used for real in `chunk_text`, the first function of the RAG project.

**Built in:** drills in `rag-docs/scratch/` (`drills.py`, `strings.py`, `functions.py`, `classes.py`, `leetcode_1.py`–`leetcode_4.py`), and `chunk_text` with its tests in `rag-docs/src/ingest.py` and `rag-docs/tests/`.

---

## 1. Concepts

### 1.1 Project tooling: uv is Maven for Python

| Java / Maven | Python / uv |
| --- | --- |
| `pom.xml` | `pyproject.toml`: project name, Python version, dependencies |
| Resolved versions in the local repo | `uv.lock`: the exact version of every package, including indirect ones. Commit it. |
| `~/.m2/repository` + classpath | `.venv/`: a private Python install for this project only. Never commit it. |
| `mvn dependency:add` (by hand in the POM) | `uv add chromadb` edits `pyproject.toml`, updates `uv.lock`, installs |
| `<scope>test</scope>` | `uv add --dev pytest` puts it in the `dev` group |
| `mvn install` | `uv sync`: makes `.venv` match `uv.lock` exactly |
| `mvn exec:java` | `uv run python file.py` runs with the project's `.venv` |

The project file from rag-docs:

```toml
[project]
name = "rag-docs"
requires-python = ">=3.11"
dependencies = [
  "anthropic",
  "chromadb",
  "pydantic>=2.14.1",
]

[dependency-groups]
dev = [
    "pytest>=9.1.1",
    "ruff>=0.17.1",
]
```

**Why a virtual environment?** Without one, every project shares the same global Python and its packages, the way a single shared classpath would. Two projects needing different versions of a library would then conflict. `.venv` gives each project its own isolated set of packages.

**`uv run` vs plain `python`:** `uv run` always uses the project's `.venv`. Plain `python` uses whatever is first on your `PATH`. On this machine that was **Anaconda's Python**, which has hundreds of packages preinstalled. Code could then "work" even though a dependency was missing from `pyproject.toml`, which is a hidden bug. Always use `uv run`, and add dependencies with `uv add`.

### 1.2 Typing: dynamic at run time, checked by tools

Java checks types when it **compiles**. Python checks **nothing** before running: a variable can hold any object, and a wrong type only fails when the line runs.

```python
x = 5        # x refers to an int
x = "five"   # now a str: perfectly legal
```

**Duck typing:** Python cares what an object can *do*, not what class it is. If it has `.read()`, you can read from it. Java needs an interface for that; Python just calls the method.

**Type hints** add the safety back, but they're checked by a separate tool, not by Python itself:

```python
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
```

- **Pylance** (VS Code) and **pyright** (command line) read the hints and underline mistakes, like IntelliJ's inspections.
- At run time the hints are ignored. `chunk_text(42)` still runs and fails inside.
- `X | None` means "X or None", like `@Nullable X` or `Optional<X>`.

**Type narrowing:** after `if x is None: return` or `assert x`, the checker knows `x` isn't `None` any more. This pattern comes up constantly with libraries that return optional values.

### 1.3 Collections

| Java | Python | Literal | Notes |
| --- | --- | --- | --- |
| `ArrayList<T>` | `list` | `[1, 2, 3]` | Ordered, mutable |
| `HashMap<K,V>` | `dict` | `{"a": 1}` | Keeps insertion order |
| `HashSet<T>` | `set` | `{1, 2}` (an empty set is `set()`, since `{}` is a dict) | Unique items, fast `in` |
| record / fixed array | `tuple` | `(1, "a")` | Immutable; used for fixed groups and multiple return values |
| `Map<T, Integer>` counting loop | `collections.Counter` | `Counter(words)` | Counts items in one call |

**Membership** is one operator for everything: `x in my_list`, `key in my_dict`, `x in my_set`. It's O(1) for dicts and sets and O(n) for lists.

**Indexing and slicing:**

```
text = "abcdef"
index:  0 1 2 3 4 5
       -6-5-4-3-2-1     ← negative indexes count from the end

text[1:4]   → "bcd"     start included, end excluded (like substring(1, 4))
text[:3]    → "abc"     from the start
text[3:]    → "def"     to the end
text[-2:]   → "ef"      the last two
text[::-1]  → "fedcba"  step -1 reverses
```

Slicing **never throws** for out-of-range ends. `"abc"[0:800]` is just `"abc"`, and `chunk_text` relies on this.

**`enumerate` and `zip`** replace index loops:

```python
for i, num in enumerate(nums):      # index + value, like for (int i...) with nums.get(i)
for a, b in zip(list1, list2):      # walk two lists in step; stops at the shorter one
```

**Unpacking** assigns several names at once: `a, b = b, a` swaps values, and `for text, topic in pairs:` unpacks each tuple.

**Comprehensions** are Python's version of `stream().filter().map().collect()`:

```python
[len(w) for w in words if w[0].isupper()]     # list
{v: k for k, v in d.items()}                  # dict: swap keys and values
{x % 3 for x in nums}                         # set
```

Read them left to right as "**what to produce**, for each item, **if** this condition holds".

### 1.4 Strings and f-strings

Strings are immutable, like Java's. The common methods: `split()` (with no argument it splits on any run of whitespace), `" ".join(parts)`, `strip()`, `lower()`, `startswith()`, `replace()`, `isalnum()`.

**f-strings** put expressions inside `{}`:

```python
print(f"{name} scored {score:.2%} across {n_docs:,} documents")
# Asha scored 87.65% across 12,345 documents
```

The part after `:` is a format spec, like `String.format`: `.2%` means percentage with 2 decimals, `,` adds thousands separators, `!r` shows the `repr` (with quotes).

**Brace escaping:** inside an f-string (or a `.format()` template), a literal `{` or `}` must be **doubled**: `f"{{"` produces `{`. This matters whenever a string contains JSON. Building JSON is better done with `json.dumps(...)` than by hand.

**Triple-quoted strings** (`"""..."""`) can span lines, which is ideal for prompts. `r"..."` raw strings keep backslashes as-is, which is ideal for regex.

### 1.5 Truthiness

In an `if`, Python treats these as **false**: `None`, `False`, `0`, `0.0`, `""`, `[]`, `{}`, `set()`. Everything else is true.

```python
if not text:          # empty string or None
if stack:             # non-empty list
```

That's convenient, but it merges different cases into one:

| Check | True for | Use when |
| --- | --- | --- |
| `if nums[j]:` | any non-zero number | "not zero" is what you mean (move-zeroes) |
| `if nums[j] != 0:` | the same here | when you want to be explicit |
| `if items is None:` | **only** `None` | "was this argument passed at all?" |
| `if not items:` | `None` **and** `[]` | "is there nothing to work with?" |

The difference in the last two rows matters. In `add_item`, `if not items: items = []` would *replace* a caller's empty list with a new one, so the caller would never see the appended item. `is None` only replaces the missing default.

Use `is` / `is not` only for `None` (identity). Use `==` for values. It's the same distinction as Java's `==` vs `.equals()`.

### 1.6 Functions

```python
def greet(name: str, greeting: str = "Hello", punctuation: str = "!") -> str:
```

- **Default arguments** replace Java overloading: `greet("Asha")`.
- **Keyword arguments** can be passed by name, in any order: `greet("Ben", punctuation="?")`. SDK calls such as `client.messages.create(model=..., max_tokens=...)` use this everywhere.
- **`*args`** collects any number of positional arguments into a tuple, like Java varargs: `def average(*nums)`.
- **`**kwargs`** collects any number of keyword arguments into a dict. The same `**` also *spreads* a dict: `{"model": model, "max_tokens": 1024, **options}`. Later keys override earlier ones.

**The mutable default gotcha:** a default value is created **once**, when the `def` runs, not on every call.

```python
def add_item(item, items=[]):     # ONE list, shared by every call
    items.append(item)
    return items

add_item("a")  # ['a']
add_item("b")  # ['a', 'b']   ← surprise
```

The fix is to default to `None` and create the list inside the function (see section 2). The same rule is why dataclasses need `field(default_factory=list)`.

**Functions are values.** You can store them in lists and dicts and pass them around. Without `()` you have the function itself; with `()` you call it.

```python
apply_all("  Hello WORLD ", [str.strip, str.lower])   # passes two functions
lambda s: s.replace("-", " ")                          # a small anonymous function
```

`str.strip` without brackets is like a Java method reference, `String::strip`.

**Typing a tuple of exception classes:**

```python
exceptions: tuple[type[Exception], ...] = (Exception,)
```

- `type[Exception]` means "the class `Exception` or a subclass", not an instance. Java's equivalent is `Class<? extends Exception>`.
- `tuple[X, ...]` means "a tuple of any length, all X".
- `(Exception,)` needs the trailing comma: `(Exception)` is just `Exception` in brackets.
- `except exceptions as e:` accepts a tuple and catches any of its types.

### 1.7 Classes

```python
class Chunk:
    def __init__(self, source: str, text: str, index: int):
        self.source = source
```

- **`__init__`** is the constructor. **`self`** is `this`, but explicit: it's the first parameter of every method.
- Fields are created by assigning to `self.x`. They aren't declared up front.
- There's no `new`: you write `Chunk("a.md", "text", 0)`.
- There are no access modifiers. A leading underscore (`_name`) means "internal, please don't touch" by convention only.

**Dunder methods** ("double underscore") hook objects into Python's syntax:

| Method | Enables | Java equivalent |
| --- | --- | --- |
| `__repr__` | `repr(x)`, debugger display | `toString()` |
| `__eq__` | `a == b` | `equals()` |
| `__len__` | `len(x)` | `size()` |
| `__iter__` | `for item in x` | `Iterable` |

**`@dataclass`** writes `__init__`, `__repr__` and `__eq__` for you, like a Java `record` or Lombok's `@Data`. **`@dataclass(frozen=True)`** makes instances immutable: assigning to a field raises `FrozenInstanceError`. That's a good fit for values like chat messages.

**`@property`** lets a method be read like a field (`usage.total_tokens`, no brackets). It's for computed values.

**`@classmethod`** receives the **class** (`cls`) instead of an instance. It's used for alternative constructors, like Java static factories: `Message.user("hi")`, `Message.from_dict(d)`. Using `cls(...)` rather than `Message(...)` means subclasses get the right type.

**Inheritance:** `class UpperTool(Tool):` is `extends Tool`. A subclass overrides methods by defining them again. A base class can force overrides in two ways:
- **Raise `NotImplementedError`** in the base method (what the drill's `Tool` does). It fails only when called.
- **`abc.ABC` + `@abstractmethod`**, the stricter option. Python refuses to create an instance of a class that hasn't implemented the method, like a Java `abstract` class.

Class attributes (`name = "upper"`, written directly in the class body) are shared by all instances, like `static` fields, but each subclass can redefine them.

### 1.8 Generators and `yield`

A function containing `yield` returns a **generator**: a lazy sequence that produces one value each time the caller asks, then pauses.

```python
def read_lines(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield stripped
```

- Nothing runs until something loops over it. It's like a Java `Iterator` whose `next()` resumes where the method left off.
- Memory stays flat: a 10 GB file is read one line at a time.
- Later this is exactly how **streaming LLM responses** work: text arrives in pieces, and a generator yields each piece as it comes.

### 1.9 Exceptions and `with`

- `try` / `except SomeError as e` / `finally` is `try` / `catch` / `finally`.
- `raise ValueError("...")` is `throw new IllegalArgumentException("...")`. There are no checked exceptions.
- `raise X from e` keeps the original cause, like `new X(msg, cause)`.
- **`with open(...) as f:`** is try-with-resources: the file is closed when the block ends, even after an exception.

### 1.10 Modules and `if __name__ == "__main__":`

Every `.py` file is a module. Importing it **runs it top to bottom**. Python sets `__name__` to `"__main__"` only for the file you executed directly, so:

```python
if __name__ == "__main__":
    for name, text in load_documents():
        print(f"{name}: {len(chunk_text(text))} chunks")
```

This block acts like `public static void main`: it runs when you execute the file and is skipped when a test imports it. Without the guard, importing `ingest` in a test would print every document.

### 1.11 `pathlib`

```python
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
```

- `__file__` is the path of the current source file. `.resolve()` makes it absolute, and `.parent` goes up one folder.
- **`/` joins paths**, like `Path.resolve()` in Java NIO.
- `path.suffix` (`".md"`), `path.name` (`"body.md"`), `path.stem` (`"body"`), `path.read_text(encoding="utf-8")`, `dir.glob("*")`.

Building paths from `__file__` makes them independent of the directory you run the command from.

### 1.12 pytest

A test is any function named `test_*` in a file named `test_*.py`. It uses a plain `assert`, and pytest shows both sides when it fails. There's no `assertEquals`.

| JUnit | pytest |
| --- | --- |
| `@Test` | a function named `test_...` |
| `assertEquals(a, b)` | `assert a == b` |
| `assertThrows(...)` | `with pytest.raises(ValueError):` |
| `@BeforeEach` setup | a **fixture**: a function decorated `@pytest.fixture`, requested by naming it as a parameter |
| `@ParameterizedTest` | `@pytest.mark.parametrize("x", [...])` |
| `@TempDir` | the built-in `tmp_path` fixture |
| Mockito `when(...)` | `monkeypatch.setattr(module, "name", fake)`, undone after the test |

- `autouse=True` makes a fixture run for every test without being named.
- `tests/conftest.py` is loaded automatically, and is the place for shared fixtures and setup.

### 1.13 Code quality tools

- **ruff check**: the linter (like Checkstyle/SpotBugs). `--fix` applies safe fixes. `--select I --fix` sorts imports.
- **ruff format**: the formatter (like google-java-format), 88-character lines.
- **pyright / Pylance**: the type checker.

Import order convention: standard library, then third-party, then your own modules, with a blank line between groups.

---

## 2. How we built it

### 2.1 Drill style

Each drill file had stub functions and checks underneath that print PASS/FAIL. For example, from `functions.py`:

```python
def check(name, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"{status}  {name}: got {got!r}, expected {expected!r}")
```

`"PASS" if got == expected else "FAIL"` is Python's version of the ternary `?:`.

### 2.2 Collections in practice (LeetCode)

```python
def contains_duplicate(nums: list[int]) -> bool:
    return len(set(nums)) != len(nums)          # a set drops duplicates

def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)              # compare letter counts

def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        remaining_amount = target - num
        if remaining_amount in seen:             # NOT seen[remaining_amount] (see Gotchas)
            return [seen[remaining_amount], i]
        seen[num] = i
    return []
```

And the in-place swap using tuple unpacking (move zeroes):

```python
def move_zeroes(nums: list[int]) -> None:
    i = j = 0
    while j < len(nums):
        if nums[j]:                               # truthy = non-zero
            nums[i], nums[j] = nums[j], nums[i]   # swap without a temp variable
            i += 1
        j += 1
```

### 2.3 The fixed `add_item` and `retry`

```python
def add_item(item: str, items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []          # a NEW list on every call that didn't pass one
    items.append(item)
    return items
```

```python
def retry(
    fn, attempts: int = 3, exceptions: tuple[type[Exception], ...] = (Exception,)
):
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except exceptions as e:
            last_error = e

    assert last_error is not None  # loop ran at least once (attempts >= 1)
    raise last_error
```

- `_` is the conventional name for a loop variable you don't use.
- The `assert` narrows `Exception | None` to `Exception` for the type checker. Without it, Pylance flags `raise last_error` because `None` can't be raised.

### 2.4 Classes that came back in later weeks

```python
@dataclass(frozen=True)
class Message:
    role: str
    content: str

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)
```

`"Message"` in quotes is a forward reference: the class isn't fully defined yet at that point.

```python
@dataclass
class Doc:
    name: str
    text: str
    tags: list[str] = field(default_factory=list)   # a new list per object
```

```python
class Conversation:
    def __init__(self, system: str = ""):
        self.system = system
        self.messages = []

    def last(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def to_api(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)
```

`Conversation` became the chat history in week 2, and `to_api()` is exactly the format the Claude API expects. `Tool` / `UpperTool` / `run_tool` are a preview of agent tools in week 8: look up a tool by name and call its `run`.

### 2.5 `chunk_text`: the first real function

RAG splits long documents into overlapping windows, so each piece is small enough to embed and so a sentence cut at a boundary still appears whole in the neighbouring chunk.

```
size = 800, overlap = 100, step = size - overlap = 700

text:   0 ──────────────────── 800
                        700 ──────────────────── 1500
                                           1400 ──────── ...
        └ chunk 0 ┘    └ shared 100 chars ┘
```

The final version in `src/ingest.py`:

```python
def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping character windows."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if not text:
        return []

    chunks = []
    start = 0
    while True:
        chunks.append(text[start : start + size])
        if start + size >= len(text):  # this chunk reached the end
            return chunks
        start += size - overlap
```

Why each line exists:
- `overlap >= size` would make the step 0 or negative, which is an infinite loop. Fail fast instead.
- Empty text gives no chunks, not one empty chunk.
- **Stop as soon as a chunk reaches the end of the text.** That one rule fixes the edge-case bugs (section 4).
- `text[start : start + size]` is safe past the end because slicing never throws.

The tests in `tests/test_chunk_text.py` check the boundaries:

```python
def test_no_duplicate_tail_chunk():
    # 1450 chars: chunk 2 (700..1450) reaches the end, so there must not be a third.
    assert len(chunk_text("a" * 1450)) == 2


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_chunks_cover_whole_text_and_respect_size():
    text = "".join(chr(97 + i % 26) for i in range(2500))
    chunks = chunk_text(text)

    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    # Rebuild the text by dropping the overlap from every chunk after the first.
    rebuilt = chunks[0] + "".join(c[CHUNK_OVERLAP:] for c in chunks[1:])
    assert rebuilt == text
```

The last test is a **property** test: rather than checking specific chunk contents, it checks that dropping the overlaps rebuilds the original text exactly, so nothing is lost or duplicated.

`tests/conftest.py` makes `from ingest import ...` work by putting `src/` on the import path:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

`sys.path` is Python's classpath: the list of folders searched on `import`.

`load_documents` is tested with `tmp_path`, so the tests create their own files and never depend on `data/`:

```python
def test_skips_readme(tmp_path):
    (tmp_path / "README.md").write_text("Put your documents here.", encoding="utf-8")
    (tmp_path / "notes.md").write_text("real content", encoding="utf-8")

    assert load_documents(tmp_path) == [("notes.md", "real content")]
```

---

## 3. Key points

- `uv add` / `uv sync` / `uv run` replace Maven. Commit `uv.lock`; never commit `.venv`.
- Always run through `uv run`. A global Python (here, Anaconda) can hide missing dependencies.
- Types are checked by Pylance/pyright, not by Python. Hints document intent and catch bugs before run time.
- `list` / `dict` / `set` / `tuple` = `ArrayList` / `HashMap` / `HashSet` / immutable record. `in` checks membership for all of them.
- Comprehensions replace most `stream().filter().map()` chains: `[f(x) for x in xs if cond]`.
- Slices are `[start:end]`, end excluded, negative indexes from the end, and never out-of-bounds errors.
- Empty and zero values are falsy. Use `is None` when "missing" and "empty" must be treated differently.
- Never use a mutable default argument. Use `None` + create inside, or `field(default_factory=list)`.
- Functions are values: no brackets means the function itself, brackets call it.
- `@dataclass` = record. `frozen=True` = immutable. `@classmethod` = static factory. `@property` = computed field.
- Dunder methods (`__repr__`, `__eq__`, `__len__`) plug objects into built-in syntax.
- `yield` makes a lazy generator. It's the basis of streaming.
- `if __name__ == "__main__":` is `main()`. It keeps imports free of side effects.
- In f-strings and `.format()` templates, double the braces to get a literal `{`.
- pytest: plain `assert`, fixtures by parameter name, `tmp_path`, `parametrize`, `monkeypatch`.

---

## 4. Gotchas we hit

1. **pytest came from Anaconda, not the project.** `import pytest` worked in the terminal but resolved to Anaconda's copy, and the editor showed errors. Fix: `uv add --dev pytest`, so it's a declared dependency of the project.
2. **Pylance showed red imports** even though the code ran. VS Code was using a global interpreter, not `.venv`. Fix: select `.venv/bin/python` as the interpreter (pinned in `.vscode/settings.json`).
3. **`chunk_text` edge cases.** The first version (not kept in git) had three boundary bugs:
   - a **duplicate tail chunk**: when a chunk already reached the end, the loop still produced one more, overlapping-only chunk
   - **short texts** (shorter than the chunk size) produced the wrong number of chunks
   - an **off-by-one at the boundary** when the text length equalled the overlap (100)

   Fix: the `while True` loop that **returns as soon as a chunk reaches the end**, plus guards for empty text and `overlap >= size`. Each bug got its own regression test.
4. **`if map[remaining_amount]:` in Two Sum.** Indexing a dict with a missing key raises `KeyError`, and a stored index of `0` is falsy. Fix: `if remaining_amount in seen:`.
5. **`if not items` vs `if items is None`** in `add_item`. `not items` also replaces a caller's empty list, so the caller loses the result.
6. **`raise last_error` flagged by Pylance** as possibly `None`. Fix: `assert last_error is not None` to narrow the type.
7. **`match.group()` underlined red.** `re.search(...)` returns `re.Match | None`: `None` when nothing matched. Pylance flags `.group()` on something that might be `None`. Fix: check first, `if match: print(match.group())`, which is type narrowing again. (Separately, `uvx ruff check --select I --fix .` sorts imports into standard library, third-party and local blocks.)
8. **f-string braces breaking JSON.** Writing JSON inside an f-string means every literal brace needs doubling. Fix: build a dict and call `json.dumps(...)`, then put that string into the f-string.

---

## 5. Revision questions

1. What do `pyproject.toml`, `uv.lock` and `.venv` correspond to in a Maven project, and which of them go into git?
2. Why can running `python` instead of `uv run python` hide a missing dependency?
3. A function has the hint `-> list[str]` but returns an `int`. What happens at run time, and who reports the problem?
4. What do `text[-3:]`, `text[:-1]` and `text[::-1]` return for `"abcdef"`?
5. Rewrite this loop as a comprehension: `result = []; for w in words: if len(w) > 3: result.append(w.upper())`.
6. When do `if not x:` and `if x is None:` behave differently? Give an example where using the wrong one is a bug.
7. Why does `def f(items=[])` misbehave, and what are the two standard fixes?
8. What is the difference between `str.strip` and `str.strip()` when passed to a function?
9. What does `tuple[type[Exception], ...]` mean, and why does `(Exception,)` need a comma?
10. What does `@dataclass(frozen=True)` generate, and what happens if you assign to a field?
11. Why use `cls(...)` instead of `Message(...)` inside a `@classmethod`?
12. Two ways to make a base class force subclasses to implement `run`. Which fails earlier?
13. What does a function containing `yield` return, and when does its body actually run?
14. Why does every runnable module in the project end with `if __name__ == "__main__":`?
15. In `chunk_text`, why must the loop stop when `start + size >= len(text)` rather than when `start >= len(text)`?

---

## Answers

1. `pyproject.toml` ≈ `pom.xml`; `uv.lock` ≈ the exact resolved dependency versions; `.venv` ≈ the local repository + classpath for this project. Commit `pyproject.toml` and `uv.lock`; git-ignore `.venv`.
2. Plain `python` may be a global install (Anaconda) with many packages preinstalled. Your code imports a package that isn't declared in `pyproject.toml` and still works locally, then fails for anyone else (or in Docker).
3. Nothing happens at run time: hints aren't enforced. Pylance/pyright report the mismatch when checking the code.
4. `"def"`, `"abcde"`, `"fedcba"`.
5. `result = [w.upper() for w in words if len(w) > 3]`.
6. They differ for empty or zero values (`[]`, `""`, `0`): `not x` is true, `x is None` is false. Example: `add_item("a", my_list)` with an empty `my_list`. `if not items: items = []` replaces the caller's list, so the caller never sees `"a"`.
7. The default list is created once, when `def` runs, and shared by all calls, so items pile up. Fixes: default to `None` and create the list inside the function; in a dataclass, use `field(default_factory=list)`.
8. `str.strip` is the function itself (a value to call later); `str.strip()` tries to call it right away, which fails here because it has no string to work on.
9. A tuple of any length whose items are exception *classes* (`Exception` or subclasses), not instances. Without the comma, `(Exception)` is just `Exception` in brackets, not a tuple.
10. `__init__`, `__repr__` and `__eq__` (plus `__hash__`, so it can go in sets and be a dict key). Assigning raises `dataclasses.FrozenInstanceError`.
11. `cls` is whatever class the method was called on, so a subclass calling the factory gets an instance of the subclass, not of the base class.
12. Raising `NotImplementedError` in the base method fails only when `run` is called. `abc.ABC` + `@abstractmethod` fails earlier: Python refuses to create an instance of a subclass that hasn't implemented it.
13. A generator object. The body runs only when something iterates over it, and pauses at each `yield` until the next value is requested.
14. Importing a module runs all its top-level code. The guard ensures the "script" part runs only when the file is executed directly, not when tests or other modules import its functions.
15. Because the chunk starting at `start` already covers the text up to `start + size`. If that reaches the end, the whole text is covered. Waiting until `start >= len(text)` produces an extra chunk made only of overlap: the duplicate tail bug.
