# Week 6: Evaluating RAG — evals and LLM judges

A RAG system that "seems to work" on three questions you tried by hand tells you almost nothing. This week was about **measuring** it: a fixed set of 24 questions, a runner that scores retrieval and answers separately, and an LLM judge that grades answers by meaning instead of by exact words. With that in place, every change became an experiment with a number attached: the judge exposed a dishonest grader, the judge's reasons pointed to missing code in the documents, fixing the data moved the bottleneck to retrieval, and raising `top_k` from 4 to 8 took answer accuracy from 17/20 to 19/20.

**Built in:** [github.com/Vsaraf334/rag-docs](https://github.com/Vsaraf334/rag-docs) — `evals/`, plus `src/prepare_docs.py`

---

## 1. Concepts

### 1.1 Why evals exist

In Java you test a method by asserting its exact output:

```java
assertEquals(42, calculator.add(40, 2));
```

That doesn't work for an LLM system:

- **There's no single right output.** "Raise `HTTPException(status_code=404)`" and "Use `HTTPException` with a 404 status code" are both correct.
- **The output varies between runs.** The same question with the same chunks can produce different wording, and a borderline answer can tip either way.
- **Quality is a distribution, not a yes/no.** A system is "good" when it answers most questions well, not when one test passes.

So instead of unit tests, you run an **eval**: a fixed set of questions, run through the whole system, scored, and summarised as numbers. It's closer to a benchmark or a load test than a JUnit test.

```
          unit test                              eval
  one input → exact expected output    24 inputs → scored outputs → 3 numbers
  pass / fail                          19/20, 18/20, 4/4 … compared between versions
```

Evals turn opinions into evidence. Compare these two interview answers:

- "I used top_k = 8 because I read that more context is better." (I heard it's better.)
- "I measured top_k 4 against 8: answer accuracy went from 17/20 to 19/20, with no new hallucinations, for about twice the input tokens." (I measured it.)

The second one is what gets hired.

### 1.2 Designing an eval set

Our eval set is `evals/questions.jsonl`: **JSON Lines**, one JSON object per line.

```json
{"question": "How do I return a 404 error?", "source": "handling-errors.md", "expected": "HTTPException"}
{"question": "How do I connect FastAPI to MongoDB?", "source": null, "expected": "don't know"}
```

| Field | Meaning |
| --- | --- |
| `question` | What a real user would type |
| `source` | The file that contains the answer; `null` means the docs don't cover it |
| `expected` | A short key fact a correct answer must contain |

JSONL is popular for evals because each line is independent: easy to append, diff in git, and stream.

Rules that made our 24 questions useful:

1. **Cover every document.** About 2 questions per file, so no part of the system goes untested.
2. **Include unanswerable questions.** 4 of the 24 (MongoDB, JWT, Kubernetes, CORS) aren't in the docs. They test the most important RAG behaviour: saying "I don't know" instead of hallucinating. Some were chosen to be *tempting*: Kubernetes sounds close to the "deploy" section, and CORS sounds like something FastAPI docs should cover.
3. **Paraphrase; don't copy the docs.** "Why does /users/me return the wrong thing?" never says "order matters". "Heavy jobs on multiple servers" never says "Celery". This tests meaning-based search, not word matching.
4. **Verify every expected fact.** Before writing the file, each `expected` phrase was grep'd in its source document. A failure should mean the system missed something that's really there, not that the question was unfair.
5. **Keep known borderline cases.** "How do I test my API?" was included on purpose because retrieval had already ranked the wrong file first for it.
6. **Keep the set stable.** If the questions change, old and new scores aren't comparable.

### 1.3 Measure retrieval and generation separately

RAG has two halves, and they fail for different reasons:

```
question ──► [ retrieval ] ──► chunks ──► [ generation ] ──► answer
                  ▲                             ▲
       "Is the right file in            "Is the answer correct, and
        the retrieved chunks?"           does it refuse when it should?"
```

If you only score the final answer, a wrong answer doesn't tell you **which half to fix**. With both scores:

| Retrieval | Answer | Diagnosis | Fix area |
| --- | --- | --- | --- |
| Miss | Wrong / "don't know" | Claude never saw the right text | Chunking, top_k, embeddings |
| Hit | Wrong | Claude had the text but answered badly | Prompt, model, data quality |
| Hit | Correct | Working | — |
| Miss | Correct | Lucky, or answered from general knowledge | Check grounding |

Our first run showed exactly this: "How do I accept a JSON object in a POST request?" had a **retrieval miss**, and Claude correctly said "I don't know". Generation did its job; retrieval failed.

One important design rule follows: **the eval retrieves once and passes the same chunks to Claude.** That's why `answer()` was split into `answer(question)` and `generate(question, chunks)`. If `answer()` searched again internally, the retrieval score and the answer score could describe two different sets of chunks.

### 1.4 The three metrics

| Metric | Over | Counts |
| --- | --- | --- |
| **Retrieval hit rate** | 20 answerable questions | `source` is among the files of the retrieved chunks |
| **Answer accuracy** | 20 answerable questions | The judge says `correct` (partials are shown separately) |
| **Correct refusals** | 4 unanswerable questions | The judge says `correct`, meaning the answer declined |

Refusals are scored separately because they reward the opposite behaviour: for these questions, answering *is* the failure.

A known blind spot: **retrieval is scored per file, not per section.** A chunk from the right file but the wrong section counts as a hit. This later hid problems in the hybrid search experiment (week 7).

### 1.5 Substring grading, and why it lies

The first grader was a containment check:

```python
def is_correct(answer: str, expected: str) -> bool:
    normalized = answer.lower().replace("’", "'")  # curly apostrophe -> straight
    return expected.lower() in normalized
```

It's cheap and deterministic, but it judges **words, not meaning**. It was wrong in both directions:

- **False fail (UserOut):** "How do I stop the user's password from being sent back?" Claude correctly explained creating an output model without the password and setting `response_model`. It just didn't use the exact name `UserOut`, so it failed.
- **False pass (Enum):** "How do I restrict a URL parameter to a fixed set of allowed values?" Claude said it didn't know the full answer, and only mentioned `Enum` because of a one-line hint in `query-params.md`. It contained the word, so it passed.

The baseline said 18/20; the honest number was about 17/20, with errors in both directions.

### 1.6 LLM-as-judge

An **LLM judge** asks a model: "Does this answer *mean* the right thing?"

```
              ┌──────────── judge prompt ─────────────┐
question  ──► │ <question>…</question>                │
expected  ──► │ <key_fact>UserOut</key_fact>          │ ──► Sonnet ──► {"verdict": "correct",
answerable──► │ <answerable>True</answerable>         │                "reason": "Describes the
answer    ──► │ <answer>Create an output model…</answer>                output-model approach…"}
              └───────────────────────────────────────┘
```

Design choices, and why:

1. **A stronger model judges a smaller one.** Haiku answers and Sonnet grades. A model grading its own output tends to be lenient; a stronger, different model is a more independent reviewer.
2. **Three verdicts, not pass/fail.** The Enum case showed a middle ground:

   | Verdict | Answerable question | Unanswerable question |
   | --- | --- | --- |
   | `correct` | Answers it and agrees with the key fact (wording may differ) | Says it doesn't know |
   | `partial` | Only partly answers, or says "don't know" when the docs *had* the answer | — |
   | `incorrect` | Wrong or contradicts the key fact | Gives a solution anyway (hallucination) |

3. **A reason for every verdict.** You read the reasons to check the judge. Judges make mistakes too: in one run the judge accepted the Enum answer as `correct` even though the real instructions were never retrieved. That's too lenient, and only reading the reason revealed it.
4. **Structured output, validated.** The judge replies with JSON, parsed into a Pydantic model whose `verdict` is a `Literal`. A confused judge then fails loudly instead of being silently miscounted.

The judge's **reasons** turned out to be its most valuable output. Three of the four partials in the first judged run said the same thing: *"the documents don't include the code"*. That pointed straight at the root cause (section 1.9).

### 1.7 `Literal`: an enum for strings

```python
class Verdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str
```

`Literal[...]` means "only these exact values". The Java equivalent is an `enum`: Jackson throws on an unknown value, and Pydantic does the same.

```python
Verdict.model_validate_json('{"verdict": "correct", "reason": "ok"}')       # OK
Verdict.model_validate_json('{"verdict": "mostly right", "reason": "ok"}')  # ValidationError
```

With a plain `str`, `"Correct"` or `"mostly correct"` would be accepted, and `r["verdict"] == "correct"` would quietly miscount it. The principle is the same as `Field(ge=1, le=4096)` in chat-service: **validate data at the boundary where it enters your program.**

### 1.8 LLM output is untrusted input

Two real failures showed that a prompt instruction is a strong request, not a guarantee.

**Failure 1: truncated JSON.** With `max_tokens=200`, Pydantic failed with `EOF while parsing a string`. Inspecting the response showed:

```
stop_reason: end_turn
usage: 190 output tokens
thinking ->            ← Sonnet reasoned first (hidden)
text -> {"verdict": "correct", "reason": "...
```

Sonnet 5.5 **thinks before answering, and thinking tokens count toward `max_tokens`**. On the failing run, the thinking was a little longer and the budget ran out mid-JSON. The fixes:

- `max_tokens=1024`. You only pay for tokens actually used, so a larger cap costs nothing extra.
- Check `stop_reason`. `"max_tokens"` means the output was cut off; `"end_turn"` means the model finished. Fail with a clear message instead of a confusing parse error.

**Failure 2: prose before the JSON.** Despite "Reply with only JSON", one of 24 calls answered `The key fact is "integer"... {"verdict": ...}`. Pydantic failed at line 1, column 1. The fix is to parse only the part from the first `{` to the last `}`.

The general rule: **treat LLM output like user input.** Validate its shape, handle the malformed case, and fail clearly. (Stricter options exist, such as forcing JSON through tool use or a structured-output feature, but extracting `{...}` was enough here.)

### 1.9 The improvement loop: measure → diagnose → fix → re-measure

```
   ┌──────────► measure (run the eval)
   │                 │
   │                 ▼
re-measure      diagnose (read the misses and the judge's reasons)
   ▲                 │
   │                 ▼
   └────────────  fix ONE thing
```

Our real run history (`top_k = 4` unless stated):

| # | Change | Retrieval | Answers | Refusals |
| --- | --- | --- | --- | --- |
| 1 | Baseline, substring grading | 18/20 | 18/20 | 4/4 |
| 2 | LLM judge replaces substring | 18/20 | 16/20 (4 partial) | 4/4 |
| 3 | Inline code examples into the docs | 17/20 | 17/20 (3 partial) | 4/4 |
| 4 | `top_k` 4 → 8 | 19/20 | 19/20 (1 partial) | 4/4 |

What each step taught:

- **Step 2: the score went down, but the measurement got honest.** The judge fixed the UserOut false fail and caught three false passes ("headers", "RequestValidationError", "httpx") where the right word appeared but Claude said it didn't know the syntax.
- **Step 3 fixed exactly what it targeted.** The three "missing code" partials became `correct`. But retrieval dropped from 18 to 17: the code made chunks larger and noisier, and `path-params.md` started losing to the similar `query-params.md`. **A fix in one place can cost something somewhere else, and only measuring shows it.**
- **Step 4:** all remaining failures were retrieval misses. The right chunk was often ranked 5th to 8th, just outside the top 4.

### 1.10 Data preparation and regular expressions

The FastAPI docs reference their code examples with placeholders:

```
{* ../../docs_src/handling_errors/tutorial001_py310.py hl[1] *}
```

Neither the embedding model nor Claude can follow that, so answers kept saying "the excerpts don't include the code". The fix was a **data preparation step that runs before ingest**, as real pipelines do: clean the data → index it → evaluate.

The regex, piece by piece:

```
\{\* \.\./\.\./(docs_src/\S+\.py)[^*]*\*\}
```

| Piece | Matches |
| --- | --- |
| `\{\*` | `{*` (escaped: `{` and `*` are special in regex) |
| ` \.\./\.\./` | ` ../../` (`.` alone means "any character") |
| `(docs_src/\S+\.py)` | **Group 1**: the file path. `\S+` = one or more non-space characters |
| `[^*]*` | Anything except `*`, which skips ` hl[1] ` |
| `\*\}` | `*}` |

Java ↔ Python:

| Java | Python |
| --- | --- |
| `Pattern.compile("...")` | `re.compile(r"...")` |
| `matcher.group(0)` / `group(1)` | `match.group(0)` / `match.group(1)` |
| `matcher.replaceAll(m -> ...)` | `pattern.sub(function, text)` |

- **Raw strings** `r"..."` don't treat `\` as an escape, so you write `\{` instead of Java's `"\\{"`.
- **Groups:** `group(0)` is always the whole match. `group(1)`, `group(2)`… are the parts in round brackets, counted from the opening brackets, left to right. Asking for a group that doesn't exist raises `IndexError: no such group`.
- **`sub` with a function:** you pass the function itself (no `()`), and Python calls it once per match, putting its return value where the match was.

**Write from the source of truth.** The script always reads the original pages from `fastapi-src` and **overwrites** the files in `data/`. Running it once or ten times gives the same result (it's idempotent). If it edited `data/` in place, a second run would have nothing left to replace. After changing `data/`, delete the index and rebuild it, because `upsert` never removes chunks whose ids no longer exist.

### 1.11 `top_k`: the trade-off

| | top_k = 4 | top_k = 8 |
| --- | --- | --- |
| Answer accuracy | 17/20 | 19/20 |
| Refusals | 4/4 | 4/4 |
| Input tokens per answer | 1× | about 2× |

Costs of more chunks:

- **Tokens:** about twice the input cost per question. That's tiny on Haiku, but it scales with traffic.
- **Latency:** more text for the model to read.
- **Noise:** irrelevant chunks can distract the model, and could even tempt it to answer unanswerable questions from loosely related text. That's why the refusal score must be checked after this change; here it stayed at 4/4.

### 1.12 Noise, and small eval sets

- With 20 answerable questions, **one question is 5%**.
- LLM answers vary between runs. In week 7, Chroma and Postgres retrieved **identical** chunks for all 24 questions, yet answer accuracy came out 19/20 once and 18/20 another time. The difference was generation and judging, not retrieval.
- So: **treat a one-question difference as noise**. Compare deterministic parts directly (identical chunks is far stronger evidence than two similar scores). To measure small effects, run the eval several times and average, or add more questions.

### 1.13 Retrieval misses: what actually helps

The stubborn miss was "How do I restrict a URL parameter to a fixed set of allowed values?". The answer is in a section titled "Predefined values" that uses `Enum`. That's a **vocabulary mismatch**: the user's words aren't the docs' words.

**BM25 (keyword search)** was investigated before building anything:

- On 10 identifier-only queries (`add_task`, `TestClient`, `BaseModel`…), vector search put the right file first 9/10 times and BM25 10/10. Vector search already handles identifiers well on this data.
- For the Enum question, the real Enum chunk ranked 17th with vector search and 9th with BM25. Neither got it into the top 8.
- Most importantly, **BM25 can only match words that are in the question**. The question doesn't contain "Enum", so keyword search can't bridge the gap.

The recommendation was corrected: skip BM25. (Week 7 built hybrid search anyway, measured it, and confirmed it made things worse.)

Tools that do target vocabulary mismatch:

| Technique | How it helps |
| --- | --- |
| **Query rewriting** | Ask an LLM to rephrase the question into the docs' likely terms before searching |
| **A stronger embedding model** | Better at linking "fixed set of allowed values" with "Enum" |
| **A reranker** | Retrieve ~30 candidates, then a model that reads question and chunk *together* picks the best few |

### 1.14 Experiment hygiene

- **Change one thing at a time**, so you know which change caused the difference.
- **Make settings flags, not code edits:** `--top-k 4`, `--backend pg`.
- **Commit the baseline before changing anything**, so you can compare.
- **Put the result in the commit message**, so `git log` reads like an engineering journal: `Retrieve 8 chunks: answer accuracy 17/20 -> 19/20`.

---

## 2. How we built it

### 2.1 Splitting `answer()` so the eval can reuse chunks (`src/answer.py`)

```python
def answer(question: str) -> str:
    return generate(question, retrieve(question))


def generate(question: str, chunks: list[dict]) -> str:
    """Ask Claude to answer the question using only the given chunks."""
    response = client.messages.create(...)
```

Normal use calls `answer()`. The eval calls `retrieve` once, records the sources, then calls `generate` with the same chunks.

### 2.2 The judge (`evals/judge.py`)

```python
JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_PROMPT = """You grade answers from a documentation Q&A system.

<question>{question}</question>
<key_fact>{expected}</key_fact>
<answerable>{answerable}</answerable>
<answer>{answer}</answer>

If answerable is true, the answer should contain the key fact (wording may differ):
- "correct": answers the question and agrees with the key fact
- "partial": only partly answers it, or says it doesn't know
- "incorrect": wrong or contradicts the key fact
If answerable is false, the docs don't cover this question:
- "correct": the answer says it doesn't know
- "incorrect": the answer gives a solution anyway

Reply with only JSON: {{"verdict": "...", "reason": "one sentence"}}"""
```

- XML tags separate the data (question, answer) from the instructions.
- `{question}` and the others are `.format()` placeholders. **`{{` and `}}` are literal braces.** Single braces in the JSON example would make `.format()` look for a placeholder called `"verdict"` and crash with `KeyError`. Doubling a brace is like writing `\\` for a real backslash in a Java string. f-strings follow the same rule.
- `answerable=True` becomes the text `True`, which Claude understands fine.

```python
def judge(question: str, expected: str, answerable: bool, answer: str) -> Verdict:
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, answerable=answerable, answer=answer
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Judge reply was cut off; increase max_tokens")
    text = "".join(b.text for b in response.content if b.type == "text")
    # The model sometimes writes a sentence before the JSON, so parse only the {...} part
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Judge reply has no JSON: {text!r}")
    return Verdict.model_validate_json(text[start : end + 1])
```

Three layers of defence: a generous token budget, a `stop_reason` check, and JSON extraction, all before Pydantic validates the shape. The `if b.type == "text"` filter skips the `thinking` block.

### 2.3 The runner (`evals/run_eval.py`)

```python
def evaluate(q: dict, retriever: Callable[..., list[dict]], top_k: int) -> dict:
    """Run one question through retrieval and generation, and score both."""
    chunks = retriever(q["question"], top_k=top_k)
    sources = [c["source"] for c in chunks]
    answer = generate(q["question"], chunks)
    result = judge(q["question"], q["expected"], q["source"] is not None, answer)
    return {
        **q,
        "retrieved": sources,
        "hit": q["source"] in sources,
        "answer": answer,
        "verdict": result.verdict,
        "reason": result.reason,
    }
```

- `q["source"] is not None` is the `answerable` flag.
- `**q` copies the question's fields into the result dict, like spreading an object in JavaScript.
- For unanswerable questions, `q["source"]` is `None`, so `hit` is meaningless; the summary only counts hits over answerable questions.

Reporting and counting:

```python
        if r["verdict"] != "correct":
            print(f"ANSWER {r['verdict'].upper():<9}{r['question']}")
            print(f"    {r['reason']}")

    hits = sum(r["hit"] for r in answerable)
    correct = sum(r["verdict"] == "correct" for r in answerable)
    partial = sum(r["verdict"] == "partial" for r in answerable)
    refused = sum(r["verdict"] == "correct" for r in unanswerable)
```

- `sum(condition for r in ...)` counts matches, because `True` counts as 1. In Java: `stream().filter(...).count()`.
- `{...:<9}` pads to 9 characters, left-aligned, so `PARTIAL` and `INCORRECT` line up (`String.format("%-9s", ...)`).

Settings are command-line flags (the `--backend` flag was added in week 7):

```python
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--backend", choices=RETRIEVERS, default="chroma")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    main(args.backend, args.top_k)
```

```bash
uv run --env-file .env python evals/run_eval.py              # Chroma, top_k 8
uv run --env-file .env python evals/run_eval.py --top-k 4    # compare with 4
```

### 2.4 Data preparation (`src/prepare_docs.py`)

```python
PLACEHOLDER = re.compile(r"\{\* \.\./\.\./(docs_src/\S+\.py)[^*]*\*\}")


def inline_code(markdown: str, repo: Path = FASTAPI_REPO) -> str:
    """Replace every code placeholder with the file it points to."""

    def replace(match: re.Match) -> str:
        code = (repo / match.group(1)).read_text(encoding="utf-8")
        return f"```python\n{code.rstrip()}\n```"

    return PLACEHOLDER.sub(replace, markdown)


if __name__ == "__main__":
    for page in PAGES:
        text = (TUTORIAL_DIR / f"{page}.md").read_text(encoding="utf-8")
        (DATA_DIR / f"{page}.md").write_text(inline_code(text), encoding="utf-8")
        print(f"{page}.md: {len(PLACEHOLDER.findall(text))} code blocks inlined")
```

- `replace` is a function defined inside another function (a named lambda); `sub` calls it once per match.
- `write_text` replaces the whole file. Strings are immutable, so `inline_code` builds a new string, and the full new content is written out.
- Result: all 62 placeholders replaced (0 left), and the index grew from 102 to 135 chunks.

The rebuild sequence after changing data:

```bash
uv run python src/prepare_docs.py
grep -c '{\*' data/*.md          # 0 for every file
rm -rf chroma_db                 # avoid stale chunks
uv run python -m src.ingest
uv run --env-file .env python evals/run_eval.py
```

---

## 3. Key points

- LLM systems can't be tested by exact output; an **eval** runs a fixed question set and reports numbers you compare between versions.
- A good eval set covers every document, includes **unanswerable** questions, uses **paraphrased** wording, has **verified** expected facts, and stays **stable**.
- Score **retrieval** and **generation** separately; that's how you know which half to fix.
- Our metrics: **retrieval hit rate**, **answer accuracy**, **correct refusals**.
- Substring grading is cheap but judges words, not meaning: it gave both a **false fail** (UserOut) and **false passes** (Enum, headers).
- **LLM-as-judge:** a stronger, different model grades with a rubric, returns a **reason**, and its output is **validated** (`Literal` + Pydantic).
- **Judges make mistakes.** Spot-check their reasons.
- The judge's **reasons** are diagnostic gold: three partials pointed to the same root cause, missing code.
- **Thinking tokens count toward `max_tokens`.** Check `stop_reason == "max_tokens"` for truncation.
- **Treat LLM output as untrusted input:** expect prose around the JSON, extract `{...}`, validate, fail clearly.
- The loop: **measure → diagnose → fix one thing → re-measure**. A fix can cost something elsewhere (inlining code lowered retrieval by one).
- Data preparation is a pipeline step: always regenerate from the **source of truth**, then rebuild the index.
- `top_k` 4 → 8: 17/20 → 19/20 answers, refusals still 4/4, about 2× input tokens.
- **One question is 5%** of 20, and LLM output varies, so treat one-question differences as noise.
- Vocabulary mismatch isn't fixed by keyword search; it needs query rewriting, better embeddings or a reranker.

---

## 4. Gotchas we hit

| Problem | Symptom | Fix |
| --- | --- | --- |
| Substring grader judged words | Correct UserOut answer failed; "I don't know + Enum" passed | LLM judge with three verdicts |
| `max_tokens=200` on the judge | `EOF while parsing a string`, JSON cut mid-sentence | `max_tokens=1024`; check `stop_reason` |
| Judge wrote a sentence before the JSON | `Invalid JSON: expected value at line 1 column 1` | Parse only from the first `{` to the last `}` |
| Judge was lenient once | Enum answer marked `correct` without the real instructions | Read reasons; don't trust the judge blindly |
| Code examples missing from chunks | Answers said "the docs don't include the code" | `prepare_docs.py` inlines the 62 placeholders |
| Wrong mental arithmetic in a test hint | Expected 3 chunks for 1,500 characters; it's 2 (0–800, 700–1500) | Trust the code and the test, not the hint |
| Curly apostrophes | `don’t know` wouldn't match `don't know` | Normalise `’` to `'` before comparing |
| Unused `# noqa: E402` comments | Ruff flagged `RUF100` | Remove them; the rule wasn't enabled |
| Ruff import order | `I001` after adding `from judge import judge` | Let `ruff check --fix` sort imports |
| Recommending BM25 before measuring | Would have added complexity for no gain | Check the evidence first; vector already 9/10 on identifiers |

---

## 5. Revision questions

1. Why can't you test an LLM system with `assertEquals`, and what replaces it?
2. Name four properties of a good eval question set.
3. Why does the eval measure retrieval and generation separately? Give a concrete example.
4. Why was `answer()` split into `answer()` and `generate()`?
5. What were the false pass and the false fail of the substring grader?
6. Why does a *stronger, different* model judge the answers?
7. What does `Literal["correct", "partial", "incorrect"]` protect against?
8. The judge's JSON was cut off mid-sentence. What caused it, and what two changes fixed it?
9. Why does the judge parse `text[start : end + 1]` instead of the whole text?
10. In `JUDGE_PROMPT`, why are the braces doubled in `{{"verdict": ...}}`?
11. Step 2 (adding the judge) lowered answer accuracy from 18/20 to 16/20. Why was that an improvement?
12. Inlining code fixed three answers but lowered retrieval by one. Why?
13. In the placeholder regex, what are `group(0)` and `group(1)`?
14. What are the costs of raising `top_k` from 4 to 8, and which metric must you re-check?
15. Why can't BM25 fix the "fixed set of allowed values" question, and what could?

---

## Answers

1. Correct answers can be worded many ways, output varies between runs, and quality is a rate rather than yes/no. An **eval** replaces it: a fixed question set run through the whole system and summarised as scores, compared across versions.
2. Covers every document; includes unanswerable questions; paraphrases instead of copying the docs; verified expected facts; stable over time (also: includes known borderline cases).
3. They fail for different reasons and need different fixes. Example: "accept a JSON object in a POST request" was a retrieval miss (`body.md` never retrieved) while Claude correctly said "I don't know". Only scoring both revealed that retrieval, not generation, was the problem.
4. So the eval can retrieve **once**, record which sources came back, and pass the **same** chunks to Claude. Otherwise the retrieval score and the answer score could describe different chunks.
5. False fail: the password answer was correct but didn't contain the exact word `UserOut`. False pass: the Enum answer said it didn't really know but contained the word `Enum`. (Also false passes for headers, RequestValidationError and httpx, where the word appeared alongside "I don't know the syntax".)
6. A model grading its own output tends to be lenient. A stronger, different model gives a more independent and more capable judgement. Here, Haiku answers and Sonnet judges.
7. Any verdict outside those three exact strings, such as `"Correct"` or `"mostly right"`, raises a `ValidationError` instead of being silently miscounted by `== "correct"`.
8. Sonnet thinks before answering, and **thinking tokens count toward `max_tokens`**; 200 wasn't enough. Fixes: raise `max_tokens` to 1024, and check `stop_reason == "max_tokens"` to fail with a clear message.
9. The model sometimes writes a sentence before (or after) the JSON despite being told not to. Extracting from the first `{` to the last `}` keeps just the JSON. LLM output is untrusted input.
10. `.format()` treats `{name}` as a placeholder. Single braces would make it look for a placeholder named `"verdict"` and raise `KeyError`. Doubled braces produce literal `{` and `}`.
11. The measurement became honest: it fixed a false fail (UserOut) and exposed false passes where answers contained the word but didn't really answer. A lower, true score beats a higher, misleading one.
12. Code made chunks larger and noisier, so the embeddings of `path-params.md` chunks moved, and they started losing to similar `query-params.md` chunks. A fix in one place can cost something elsewhere; only measurement shows it.
13. `group(0)` is the whole placeholder, `{* ../../docs_src/... hl[1] *}`. `group(1)` is the part in round brackets: the file path, `docs_src/handling_errors/tutorial001_py310.py`.
14. About twice the input tokens, more latency, and more irrelevant context that could distract the model. Re-check **correct refusals**, because extra loosely related context could tempt Claude to answer unanswerable questions.
15. BM25 only matches words that appear in the question, and the question doesn't contain "Enum" or "predefined". It's a vocabulary mismatch. Query rewriting, a stronger embedding model, or a reranker could help.
