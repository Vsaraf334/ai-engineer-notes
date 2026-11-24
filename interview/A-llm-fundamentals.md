# A. LLM fundamentals

Interviewers use these to check you understand what's under the API. You don't need the maths of transformers; you need to explain behaviour you'll see in production (cost, limits, randomness, made-up facts) and what you do about it.

---

## A1. What is a token, and why does it matter?

**Short answer:** A token is the unit a model reads and writes: a word, part of a word, a punctuation mark or a space. Text is split into tokens before the model sees it. Tokens matter because they set **cost** (you pay per input and output token, and output tokens cost more), **limits** (the context window and `max_tokens` are counted in tokens) and **latency** (the model generates output one token at a time).

**Deeper:**
- Common English words are often one token; rare words, code, other languages and long numbers take more. A rough rule for English is 3–4 characters per token, but it varies by model, so measure with the API's token counts instead of guessing.
- Input and output are billed separately. Output is usually several times more expensive per token, and it's also what makes a response slow.
- Everything you send counts as input: the system prompt, tool definitions, the whole conversation history and retrieved documents, on **every** call.

**From my projects:**
- In rag-docs, one MCP host run went from 570 input tokens (system prompt, tool definition, question) to 3,084 on the next call. The difference was the 8 retrieved passages.
- In incident-agent, an investigation averages about 7,600 tokens with Sonnet, but the worst run took 17,400, because every step re-sends the whole history.

**Follow-ups:** How do you reduce token usage? (Shorter prompts, fewer or smaller retrieved chunks, prompt caching, a cheaper model for easy steps; see area G.) How do you count tokens before sending? (The API's token-counting endpoint, or the `usage` field after each call.)

---

## A2. What is a context window, and what happens when you exceed it?

**Short answer:** The context window is the maximum number of tokens a model can handle in one request: everything you send plus what it generates. If the request is too long, the API rejects it with an error. The model has no memory between requests: anything it should "remember" must be inside the window every time.

**Deeper:**
- **Bigger isn't free:** a long context costs more on every call and is slower, and quality can drop when the relevant part is buried among a lot of irrelevant text. Sending only what's relevant usually beats sending everything.
- **Strategies when content is too big:** retrieve only the relevant parts (RAG), summarise older conversation turns, drop old tool results, or split the task.
- **Agents grow their context:** every tool call and result is appended, so long investigations get expensive. Cap the steps and keep tool outputs compact.

**From my projects:** In incident-agent the tools return compact text (one log line per line, capped at 50 lines) instead of raw JSON, because every tool result stays in the context for the rest of the run.

**Follow-ups:** Why not put all your documents into a long context instead of using RAG? (Cost and latency per call, the quality drop with irrelevant text, and documents that don't fit at all; see C2.)

---

## A3. How does an LLM generate text?

**Short answer:** It predicts the next token, one at a time. Given everything so far, the model produces a probability for every possible next token; one is chosen, appended, and the process repeats until it emits a stop signal or hits `max_tokens`. Everything, including JSON and tool calls, is produced this way.

**Deeper:**
- This is why output is slow and costs more than input: output tokens are generated one after another, while the input is processed in one go.
- Because structure is generated token by token, nothing guarantees valid JSON unless the API constrains the generation (structured outputs, strict tools).
- The model was trained to predict likely text, then further trained to follow instructions and be helpful. It has no separate fact database; what it "knows" is patterns in its weights, up to its training cut-off.

**From my projects:** Haiku sometimes leaked its own tool-call formatting (`</cause><parameter name="evidence">`) into a long string argument. That only makes sense once you know tool calls are generated as text: partway through a long string, it lost track of being inside it. Strict mode didn't stop it, because any text is a valid string.

**Follow-ups:** What's a transformer, briefly? (The neural network architecture behind LLMs; its attention mechanism lets each token weigh every earlier token when predicting the next one.)

---

## A4. What do temperature and other sampling settings do?

**Short answer:** They control how the next token is chosen from the model's probabilities. Low temperature makes the most likely tokens even more likely, so output is more focused and repeatable; high temperature flattens the probabilities, so output is more varied. Use low values for extraction, classification and tool use; higher values only when you want variety, like brainstorming.

**Deeper:**
- `top_p` and `top_k` limit the choice to the most likely tokens. Usually you tune one setting, not all of them.
- Low temperature doesn't make the model correct, only more consistent. A confident wrong answer stays wrong.
- Some model features restrict sampling settings (for example when extended thinking is on), so check the model's documentation.

**Follow-ups:** If you set temperature to 0, is the output deterministic? (No; see A5.)

---

## A5. Why isn't the output deterministic, and how do you deal with it?

**Short answer:** Sampling picks tokens with some randomness, and even at temperature 0, small numerical differences in how the servers compute and batch requests can change a token, after which everything downstream differs. So the same input can give different outputs. You deal with it by designing for variation: validate outputs, retry on failure, and evaluate with repeated runs rather than one.

**Deeper:**
- One different token early on can change a whole answer, or an agent's whole path.
- For evaluation, run each test several times and report rates ("passed in 2 of 3 runs"), and know your noise level before claiming an improvement.
- For production, make the important decisions checkable: schemas, validation, human approval for actions with side effects.

**From my projects:**
- The same checkout alert, with identical tools and data: one run filed an incident, another concluded "two candidates, can't tell which".
- Two evals of identical code gave 42/45 and 37/45 runs. So I only trust differences bigger than that, and each scenario runs 3 times and passes on a majority.

**Follow-ups:** How many runs is enough? (Enough that the difference you care about is bigger than the run-to-run spread; more runs for small differences.)

---

## A6. Why do LLMs hallucinate, and how do you reduce it?

**Short answer:** The model generates plausible text, not verified facts. When it doesn't know something, the most likely continuation can still sound confident. It has no built-in way to check facts, and its knowledge stops at its training cut-off. You reduce it by **grounding** (giving it the source material and telling it to answer only from that), letting it say "I don't know", asking for citations, and measuring hallucinations with evals.

**Deeper:**
- Grounding helps most: the answer is supported by text in the prompt rather than recalled from memory.
- Explicit permission to say "I don't know" matters; without it, models tend to answer anyway.
- Citations make answers checkable, by people and by an eval.
- Include questions the system *can't* answer in your eval set, and check that it declines them.

**From my projects:**
- rag-docs answers only from retrieved FastAPI docs, cites `[source.md]` after each claim, and says it doesn't know otherwise: it declined 4 of 4 unanswerable questions.
- When my MCP host had no system prompt, it answered a MongoDB question (not in the docs) from general knowledge. Adding the grounding instructions fixed it. Grounding is an instruction plus data, and the instruction matters.

**Follow-ups:** How do you detect hallucinations automatically? (An LLM judge that checks the answer against the retrieved sources; checks that cited sources exist and support the claim.)

---

## A7. What's the difference between the system prompt and the user message?

**Short answer:** The system prompt sets the role, rules and context for the whole conversation (who the model is, what it may do, the output format). User messages are the turn-by-turn input. Models are trained to give the system prompt more weight, so stable rules belong there and the varying request belongs in the user message.

**Deeper:**
- The system prompt is still just text in the context. It's **not a security boundary**: user input or retrieved documents can try to override it (prompt injection). Never rely on it alone for safety.
- Keep it stable across requests: that also makes prompt caching effective (G2).

**From my projects:** incident-agent's system prompt holds the on-call rules ("only blame a deploy if the errors started after it", "file nothing if the evidence is weak"). Over MCP, the same text is offered as the server's `instructions`, and the host decides whether to use it: the server can't force it.

**Follow-ups:** Where would you put few-shot examples? (Usually in the system prompt or as example turns; see B2.)

---

## A8. What are stop reasons, and what is `max_tokens`?

**Short answer:** Every response says why generation stopped. The main ones: `end_turn` (finished normally), `tool_use` (it wants a tool to run), `max_tokens` (it hit your output limit and was cut off) and `stop_sequence` (it produced a string you told it to stop at). `max_tokens` caps the length of **one** response. Your code must check the stop reason: a cut-off response is incomplete.

**Deeper:**
- You pay for the tokens generated, not for the cap, so a generous cap costs nothing extra; a tight one causes cut-off answers.
- A response cut off by `max_tokens` can end in the middle of a tool call. Never execute it: treat it as an error.
- An agent loop should handle each stop reason explicitly and fail loudly on anything unexpected.

**From my projects:** My first agent loop used `max_tokens=1024`. The investigation steps were fine, but the step that filed the incident (long quoted evidence) was cut off, and my loop raised `Unexpected stop_reason: max_tokens` instead of running a half-written tool call. I raised the cap to 4096 at no extra cost.

**Follow-ups:** What should an agent loop do on each stop reason? (`end_turn`: return the answer; `tool_use`: run the tools and continue; anything else: stop with an error, keeping the trace.)

---

## A9. What's the difference between an embedding model and a generative model?

**Short answer:** A generative model (like Claude) reads text and writes text. An embedding model reads text and outputs a fixed-length list of numbers (a vector) that represents its meaning, so that texts with similar meanings get vectors that are close together. Embeddings are what make semantic search possible; they can't write answers.

**Deeper:**
- Embedding models are small, fast and cheap, often run locally, and every vector has the same size whatever the input length.
- Closeness is measured with cosine similarity or a related distance.
- You must use the **same** embedding model for documents and queries; vectors from different models aren't comparable.
- In RAG, the embedding model finds the passages; the generative model writes the answer from them.

**From my projects:** rag-docs embeds its 135 chunks with all-MiniLM-L6-v2, a small model that runs locally and produces 384-number vectors, stored in Chroma or Postgres with pgvector. Claude never sees the vectors; it gets the retrieved text. My MCP server does no LLM work at all: just embedding plus search.

**Follow-ups:** How would you choose an embedding model? (Measure retrieval quality on your own questions; consider language, domain, vector size and cost.)

---

## A10. How do you choose a model for a task?

**Short answer:** Trade off quality, cost and latency, and decide with an eval on your own task, not a general benchmark. Start with a capable model to see what good looks like, then try cheaper and faster ones and keep the cheapest that still passes. Different steps can use different models.

**Deeper:**
- Compare more than the headline score: how often each model fails, *how* it fails, its consistency across runs, tokens used, and cost per correct result.
- A cheaper model that fails more can cost more overall (retries, wasted work, people's time).
- Routing: a small model for easy, high-volume steps, a larger one for hard or high-stakes steps.

**From my projects:** In incident-agent I compared Haiku 5.5 and Sonnet 5.5 on the same 15 scenarios × 3 runs:

| | Haiku | Sonnet |
| --- | --- | --- |
| Runs succeeded | 37/45 | 45/45 |
| Cause correct | 45/45 | 45/45 |
| Failed `file_incident` calls | 15 | 0 |
| Avg tokens per run | 11,239 | 7,609 |

Both found the right cause every time; Haiku's failures were formatting its long write-tool call. I chose Sonnet because incidents are low volume and the agent's job is filing a correct record. A routing design would let Haiku investigate and Sonnet write the incident.

**Follow-ups:** Sonnet used fewer tokens; is it cheaper? (Not necessarily: it costs more per token. Compare price × tokens per correct outcome.) What about open-source models? (See J4.)

---

## Quick revision: the short answers in one line each

1. **Token:** the unit of text; sets cost, limits and latency.
2. **Context window:** the max tokens per request, input + output; no memory between calls.
3. **Generation:** next-token prediction, one at a time; structure isn't guaranteed.
4. **Temperature:** low = consistent, high = varied; low isn't the same as correct.
5. **Non-determinism:** sampling and server numerics; evaluate with repeated runs.
6. **Hallucination:** plausible, not verified; ground, allow "I don't know", cite, evaluate.
7. **System prompt:** stable rules with more weight, but not a security boundary.
8. **Stop reasons:** check them; `max_tokens` cut-offs are incomplete; caps are free.
9. **Embeddings:** text → vector for similarity search; can't generate.
10. **Model choice:** an eval on your task; quality, cost, latency; route by step.
