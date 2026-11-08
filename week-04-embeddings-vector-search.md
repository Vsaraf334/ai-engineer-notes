# Week 4: Embeddings and vector search

This week built the **indexing half** of a RAG system. Documents are split into chunks, each chunk becomes an **embedding** (a list of 384 numbers that captures its meaning), and the embeddings are stored in **Chroma**, a vector database. Later, a question is embedded the same way, and the database returns the chunks whose numbers are closest. That is "search by meaning": it finds the right text even when the question and the document share no words.

**Built in:** [github.com/Vsaraf334/rag-docs](https://github.com/Vsaraf334/rag-docs) (`src/ingest.py`, `tests/test_store_chunks.py`) and the practice file `scratch/chroma_drill.py`.

---

## 1. Concepts

### 1.1 The RAG big picture

RAG (retrieval-augmented generation) means: before asking the LLM, **find the relevant pieces of your own documents** and give them to it. It has two halves:

```
INDEXING (done once, when documents change)          ← week 4
  documents ─► split into chunks ─► embed each chunk ─► store vectors + text

ANSWERING (done for every question)                  ← week 5
  question ─► embed it ─► find the closest chunks ─► give them to the LLM ─► cited answer
```

Why not send all the documents to the LLM every time?

- **Cost:** you pay per input token, so sending 100 pages per question is expensive.
- **Quality:** a model given a lot of irrelevant text answers worse than one given just the right paragraphs.
- **Limits:** a context window is large but not unlimited, and a company's documentation can be far bigger.

So the core problem of RAG is **search**: given a question, find the few chunks that answer it.

### 1.2 Why keyword search isn't enough

The obvious search is keywords, like SQL `WHERE text LIKE '%404%'`. It fails when the user's words differ from the document's words:

```
Question: "How do I tell the client a resource doesn't exist?"
Document: "Raise HTTPException with status_code=404 when an item is not found."
```

These share no important words, yet the document is exactly the answer. Real users rarely use the same words as the documentation. We need search by **meaning**, not by **spelling**.

### 1.3 What an embedding is

An **embedding** is a fixed-length list of numbers (a vector) that represents the meaning of a piece of text. The model we use always outputs **384 numbers**:

```
"Raise HTTPException with status_code=404..."  → [-0.001, 0.071, -0.038, 0.008, ... 384 numbers]
"How do I tell the client it doesn't exist?"   → numbers that are CLOSE to the line above
"Preheat the oven to 180 degrees..."           → numbers that are FAR from both
```

Picture every text as a point on a map. Texts about errors cluster together and recipes cluster somewhere else. "Search by meaning" then becomes a geometry problem: **find the stored points nearest to the question's point**.

Two important facts:

- **Individual numbers mean nothing on their own.** Number #17 is not "the API dimension". Nobody designed the dimensions; they came out of training. Only **distances between vectors** carry meaning.
- **The length is always the same** (384), whether the input is three words or a whole paragraph. That's what makes any two texts comparable.

Java analogy: an embedding is like a `hashCode()` that, instead of being random, puts **similar objects close together**. You never read a hash code's digits; you only compare them. The same goes here.

### 1.4 How an embedding model works inside

Our model is **all-MiniLM-L6-v2**, which Chroma downloads (about 80 MB) the first time you use it. It belongs to the same family as Claude (a **transformer** neural network) but does a different job:

```
Claude (generative):   text in → writes new text, one token at a time
MiniLM (embedding):    text in → one fixed vector of 384 numbers. It never writes anything.
```

Turning `"Raise HTTPException when an item is not found"` into a vector takes 4 steps:

**Step 1: tokenize.** The text is split into known word pieces from a fixed vocabulary of about 30,000 pieces. Rare words are split into smaller pieces:

```
["raise", "http", "##ex", "##ception", "when", "an", "item", "is", "not", "found"]
```

Each piece maps to an ID, like a lookup in a `Map<String, Integer>`.

**Step 2: look up a starting vector for each token.** A learned table gives every token ID its own vector. At this point "found" has the same vector in every sentence.

**Step 3: six layers of attention add context.** This is the key idea behind every modern LLM. In each layer, every token looks at all the other tokens in the sentence and adjusts its own numbers based on them:

- "not" next to "found" shifts toward the meaning *missing*.
- "raise" next to "exception" shifts toward *throwing an error*, not *lifting something* or *a pay raise*.

After 6 layers (the "L6" in the name), each token's vector reflects what it means **in this sentence**.

**Step 4: pool and normalise.** The token vectors are averaged into one vector (**mean pooling**), then scaled to length 1 (**normalised**). The result is the 384 numbers stored in the database.

```
text ─► word pieces ─► token vectors ─► 6 attention layers ─► average ─► scale to length 1 ─► [384 numbers]
```

### 1.5 How it learned: contrastive training

The model's internal weights started out random. It was trained on roughly **a billion pairs of related texts**:

- a question and its accepted answer (from Q&A sites)
- a search query and the page people clicked
- a title and its article
- two sentences that mean the same thing

The training rule is simple:

```
repeat over and over:
    take a batch of pairs
    embed both sides
    pull each pair's vectors CLOSER together
    push them AWAY from the other texts in the batch
    nudge the weights slightly so the next batch scores better (gradient descent)
```

This is called **contrastive learning**. After enough repetitions, the weights settle into a state where texts with similar meanings land near each other. Nobody wrote a rule saying "404 ≈ not found". The model saw enough questions about missing resources paired with answers about 404 errors.

### 1.6 all-MiniLM-L6-v2 in numbers

| Property | Value |
| --- | --- |
| Output | 384 numbers per text |
| Size | About 22 million parameters (Claude has hundreds of billions) |
| Where it runs | Locally on your CPU. Free, no API key, no internet after the first download |
| Input limit | About 256 tokens. Anything longer is **silently cut off** |
| Download | About 80 MB, cached in `~/.cache/chroma/onnx_models/` |

The input limit is one of the main reasons to chunk documents (section 1.10).

### 1.7 Measuring "close": distance metrics

There are several ways to measure how far apart two vectors are:

| Metric | Formula idea | Range for normalised vectors | Similar texts |
| --- | --- | --- | --- |
| **Cosine similarity** | Angle between the arrows: cos(angle) | −1 to 1 | **High** (near 1) |
| **Cosine distance** | 1 − cosine similarity | 0 to 2 | **Low** (near 0) |
| **Squared L2 distance** | Sum of squared differences between the arrow tips | 0 to 4 | **Low** (near 0) |

Picture each embedding as an arrow from the centre of space. Similar meanings point in similar directions.

Chroma's default is **squared L2 distance**. Because our vectors are normalised to length 1, it has a simple relationship to cosine similarity:

```
squared L2 distance = 2 − 2 × cosine similarity
```

So the metrics always agree on **which text is closest**. Only the scale of the numbers differs. (In week 7, Postgres's cosine distance came out at exactly half of Chroma's numbers for the same chunks, in the same order.)

The real numbers from the drill, for the question *"How do I tell the client a resource doesn't exist?"*:

```
1.316  Raise HTTPException with status_code=404 when an item is not found.   ← related
1.578  Use a Pydantic model to validate the JSON body of a request.          ← same topic area
1.990  Preheat the oven to 180 degrees before baking banana bread.           ← unrelated
2.059  The Eiffel Tower is in Paris and was finished in 1889.                ← unrelated
2.121  Mash three ripe bananas with melted butter and sugar.                 ← unrelated
```

Around **2.0** means unrelated (cosine similarity about 0, perpendicular arrows). Lower means more related. You'll hear "cosine similarity" constantly in interviews; it's the same comparison seen from the other side.

### 1.8 The golden rule: one model for everything

The question and the chunks **must be embedded with the same model**. Vectors from different models are like coordinates on different maps, so comparing them is meaningless. If you ever change the embedding model, you must **re-embed every chunk**.

### 1.9 Vector databases and Chroma

A **vector database** stores vectors next to their text and answers one main question fast: *"which stored vectors are nearest to this one?"* In Java terms, think of:

```java
class Row { String id; String document; Map<String, Object> metadata; float[] embedding; }
List<Row> findNearest(float[] query, int k);
```

You could build that yourself with a list and a loop over every row. A vector database does it efficiently and also persists the data.

**Chroma** is a simple vector database that runs inside your Python program. Its main concepts:

| Chroma | Java / SQL equivalent |
| --- | --- |
| `chromadb.EphemeralClient()` | An in-memory database (like H2 in-memory): gone when the program ends |
| `chromadb.PersistentClient(path=...)` | A database stored in files on disk: survives restarts |
| `client.get_or_create_collection(name=...)` | A table (created if missing) |
| `col.add(ids=..., documents=..., metadatas=...)` | `INSERT` (fails on a duplicate id) |
| `col.upsert(...)` | `INSERT ... ON CONFLICT UPDATE`: insert or replace |
| `col.query(query_texts=[...], n_results=k)` | "the k nearest rows" |
| `where={"topic": "food"}` | `WHERE topic = 'food'` |
| `col.count()` | `SELECT count(*)` |

**Chroma embeds for you.** When you pass `documents=[...]` to `add` or `upsert`, Chroma runs MiniLM on each text. When you pass `query_texts=[...]` to `query`, it embeds the question with the same model. That convenience is why the golden rule (section 1.8) is automatic with Chroma. In week 7, with Postgres, our code had to call the model itself.

**Parallel lists:** `add` takes three lists, and position matters. `ids[0]`, `documents[0]` and `metadatas[0]` together form **one row**.

**Why `query()` returns lists of lists:** `query_texts` is a **list**, so you can ask several questions in one call. The outer list has one entry per question, and the inner list holds that question's results:

```python
{'ids':       [['s0', 's1']],
 'documents': [['Raise HTTPException...', 'Use a Pydantic model...']],
 'metadatas': [[{'topic': 'api'}, {'topic': 'api'}]],
 'distances': [[1.295, 1.583]]}
```

With one question, you always take index `[0]`: `res["documents"][0]` is the list of documents for your question. Note the plural key names: `metadatas`, `distances`.

**Metadata filters run before ranking.** In the drill, searching "Eiffel Tower" with `where={"topic": "food"}` returned banana bread. The travel sentence was excluded before any distances were compared, so it was never a candidate. In the real project, metadata is `{"source": "handling-errors.md"}`, which is what lets the answer cite its file.

### 1.10 Chunking

Documents are split into **chunks** before embedding, for three reasons:

1. **The embedding model's input limit.** MiniLM reads about 256 tokens. A 20-page document would be cut off after the first ~256 tokens, and the rest would be invisible to search.
2. **Precision.** One vector for a whole document averages many topics into a blur. A chunk about one topic gives a sharp vector that matches questions about that topic.
3. **Cost of the answer step.** Only the matching chunks are sent to the LLM, not whole documents.

Our chunking is simple: **fixed-size character windows with overlap**.

```
CHUNK_SIZE = 800, CHUNK_OVERLAP = 100

text:    [0 ............ 800]
                   [700 ............ 1500]
                               [1400 ............ 2200]
          ◄─ 100 ─►  shared between neighbours
```

Each window starts `size − overlap` = 700 characters after the previous one. The **overlap** means a sentence that falls on a boundary appears whole in at least one chunk.

The trade-offs:

| | Smaller chunks | Larger chunks |
| --- | --- | --- |
| Vector | Sharper, about one topic | Blurrier, several topics |
| Context per chunk | Less, and may cut an explanation in half | More complete |
| Number of chunks | More (more storage, more results to manage) | Fewer |

800 characters is roughly 200 tokens, which fits inside MiniLM's limit. The right size is something to **measure** with evals (week 6), not guess. Character-based chunking can split a code block or a sentence; heading-aware chunking (one section per chunk) is a common improvement.

### 1.11 Deterministic ids and upsert

Every chunk gets an id built from its file and position: `body.md:0`, `body.md:1`, … The same chunk always gets the same id, so it is **deterministic**.

Combined with `upsert`, this makes ingestion **safe to re-run**:

- With random ids, a second run would insert every chunk again, giving 204 rows, half of them duplicates.
- With `add` and fixed ids, a second run would fail on the duplicate ids.
- With **`upsert` and fixed ids**, a second run replaces each row with itself: still 102.

**One remaining trap: stale chunks.** `upsert` only touches the ids you send. If a document gets **shorter** (from 12 chunks to 10), its old chunks `:10` and `:11` stay in the index with outdated text. The simple fix after changing documents is to rebuild from scratch:

```bash
rm -rf chroma_db
uv run python -m src.ingest
```

### 1.12 Data quality matters as much as the model

The FastAPI docs we indexed don't contain their code examples inline. They contain placeholders:

```
{* ../../docs_src/handling_errors/tutorial001_py310.py hl[1] *}
```

Neither the embedding model nor Claude can follow such a reference, so every chunk was **missing its code**. In week 6, evals proved this was the biggest cause of weak answers. A preparation step that inlines the real code fixed three answers at once. The lesson: **look at what is actually in your chunks** before tuning models.

### 1.13 Choosing an embedding model

MiniLM is small, old and free. It's a good starting point, not necessarily the best choice. Paid API models are usually more accurate, for example **Voyage AI** (which Anthropic recommends for use with Claude) and **OpenAI's text-embedding-3** models. Trade-offs:

| | Local model (MiniLM) | API model (Voyage, OpenAI) |
| --- | --- | --- |
| Cost | Free | Per token |
| Privacy | Text never leaves the machine | Text is sent to the provider |
| Quality | Decent | Usually better |
| Speed | CPU-bound | Network-bound |

The right way to choose: build an eval set (week 6), then **measure** retrieval with each model. "I measured it" beats "I heard it's better". And remember the golden rule: switching models means re-embedding everything.

---

## 2. How we built it

### 2.1 The drill: see meaning-based search work

`scratch/chroma_drill.py` stored five sentences (two about APIs, two about food, one about travel) and searched them.

```python
def make_collection():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name="drill")
```

An in-memory client, then a collection (table) named `drill`.

```python
def add_sentences(col, sentences: list[tuple[str, str]]) -> None:
    ids = [f"s{i}" for i in range(len(sentences))]
    documents = [text for text, _ in sentences]
    metadatas = [{"topic": topic} for _, topic in sentences]
    col.add(ids=ids, documents=documents, metadatas=metadatas)
```

The input is a list of `(text, topic)` **tuples**, so unpacking `for text, _ in sentences` works. Three parallel lists, then one `add` call. Inside `add`, Chroma embeds each document.

```python
def search(col, question: str, k: int = 2) -> list[str]:
    res = col.query(query_texts=[question], n_results=k)
    return res["documents"][0]
```

`[0]` selects the results for the first (and only) question.

```python
def search_with_scores(col, question: str, k: int = 2) -> list[tuple[str, float]]:
    res = col.query(query_texts=[question], n_results=k)
    return list(zip(res["documents"][0], res["distances"][0]))
```

`zip` pairs the two lists position by position, like a zipper. `list(...)` is needed because `zip` is lazy, like a Java `Stream` before `.collect()`.

```python
def search_topic(col, question: str, topic: str, k: int = 2) -> list[str]:
    res = col.query(query_texts=[question], n_results=k, where={"topic": topic})
    return res["documents"][0]
```

The `where` filter is applied before ranking.

The test that matters most: *"How do I tell the client a resource doesn't exist?"* found the HTTPException sentence first, with **no shared keywords**.

### 2.2 The real project: `src/ingest.py`

Settings at the top of the file:

```python
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "docs"
CHUNK_SIZE = 800  # characters per chunk; tune this during evals
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks
```

`chroma_db/` is generated data, so it's in `.gitignore`, like Maven's `target/`.

**Chunking** (written in week 1, used here):

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

The guard stops an infinite loop: if `overlap >= size`, `start` would never move forward.

**Building chunk records** with an id, the text and the source file:

```python
def build_chunks(docs: list[tuple[str, str]]) -> list[dict]:
    """Turn (filename, text) pairs into chunk dicts with id, text and source."""
    chunks = []
    for filename, text in docs:
        for i, piece in enumerate(chunk_text(text)):
            chunks.append({"id": f"{filename}:{i}", "text": piece, "source": filename})
    return chunks
```

`enumerate` gives `(0, piece), (1, piece), …`, which produces the deterministic ids.

**Storing in Chroma on disk:**

```python
def get_collection(path: Path = CHROMA_DIR):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(chunks: list[dict], path: Path = CHROMA_DIR) -> None:
    """Embed and store chunks. Re-running replaces chunks with the same id."""
    col = get_collection(path)
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
```

The chunks are **dicts**, so each value is read by key (`c["text"]`), not by unpacking. The `path` parameter exists so that tests can point at a temporary folder.

```python
if __name__ == "__main__":
    chunks = build_chunks(load_documents())
    store_chunks(chunks)
    print(f"Stored {len(chunks)} chunks in {CHROMA_DIR}")
```

Run with `uv run python -m src.ingest`. With the 8 FastAPI tutorial pages this first printed **Stored 102 chunks**, and after the code examples were inlined in week 6, **135**.

### 2.3 Tests: `tests/test_store_chunks.py`

```python
def test_build_chunks_ids_and_source():
    docs = [("a.md", "short text"), ("b.md", "x" * 1500)]
    chunks = build_chunks(docs)
    assert [c["id"] for c in chunks] == ["a.md:0", "b.md:0", "b.md:1"]
    assert all(c["source"] == c["id"].split(":")[0] for c in chunks)
```

1,500 characters gives **2** chunks: 0–800, then 700–1500, which reaches the end.

```python
def test_store_chunks_twice_does_not_duplicate(tmp_path):
    chunks = build_chunks([("a.md", "Raise HTTPException for a 404 error.")])
    store_chunks(chunks, path=tmp_path)
    store_chunks(chunks, path=tmp_path)
    assert get_collection(tmp_path).count() == 1
```

This proves upsert plus deterministic ids. `tmp_path` is a pytest fixture that gives each test its own empty folder, so tests never touch the real `chroma_db/`.

```python
def test_stored_chunk_is_searchable_with_source(tmp_path):
    docs = [
        ("errors.md", "Raise HTTPException for a 404 error."),
        ("food.md", "Bake banana bread at 180 degrees."),
    ]
    store_chunks(build_chunks(docs), path=tmp_path)
    res = get_collection(tmp_path).query(query_texts=["item not found"], n_results=1)
    assert res["metadatas"][0][0] == {"source": "errors.md"}
```

`[0][0]` is the first question's first result.

---

## 3. Key points

- RAG has two halves: **indexing** (chunk, embed, store; done once) and **answering** (embed the question, find nearest chunks, ask the LLM; done per question).
- Keyword search fails on paraphrases. Embeddings match **meaning**.
- An embedding is a fixed-length vector (384 numbers for MiniLM). Individual numbers mean nothing; only **distances** do.
- An embedding model is a small transformer: word pieces → token vectors → attention layers add context → mean pooling → normalise.
- It is trained with **contrastive learning**: related pairs are pulled together and unrelated texts pushed apart, over about a billion pairs.
- all-MiniLM-L6-v2: 384 dimensions, about 22M parameters, local and free, about **256-token input limit** (longer text is cut off silently).
- For normalised vectors, `squared L2 = 2 − 2·cosine similarity`, so all the common metrics rank results in the same order.
- Always embed questions and chunks with the **same model**. Changing models means re-embedding everything.
- A vector database stores vectors next to text and metadata and finds nearest neighbours. Chroma also embeds for you.
- `query()` returns **lists of lists** (one inner list per question), so use `[0]`. Keys are plural: `metadatas`, `distances`.
- Metadata `where` filters are applied **before** ranking.
- Chunk because of the token limit, for sharper vectors, and to send less to the LLM. Overlap keeps boundary sentences whole.
- Deterministic ids plus `upsert` make ingestion safe to re-run. But stale chunks remain when a document shrinks, so rebuild the index after changing data.
- Data quality beats model tuning: check what is actually inside your chunks.
- Choose embedding models (MiniLM, Voyage, OpenAI) by **measuring** retrieval with evals.

---

## 4. Gotchas we hit

| What happened | Cause | Fix / lesson |
| --- | --- | --- |
| `AttributeError: 'str' object has no attribute 'items'` inside `upsert` | `metadatas=[c["source"] for c in chunks]` built a list of **strings** | Metadata must be a list of **dicts**: `[{"source": c["source"]} for c in chunks]` |
| `ValueError: too many values to unpack (expected 2)` | `for _, source in chunks` tried to unpack **dicts**, which iterate over their 3 keys | Unpack tuples; index dicts by key: `c["source"]` |
| Confusion over `res["documents"][0]` | `query()` returns one inner list per question | With one question, always take `[0]`. Print `res.keys()` when unsure of a shape |
| A test expected 3 chunks for 1,500 characters and failed | Wrong mental arithmetic: the 2nd chunk (700–1500) already reaches the end | `chunk_text` was right. Work out the boundaries on paper |
| `ModuleNotFoundError: No module named 'src'` in a new test | `conftest.py` puts `src/` on the path, so tests import `from ingest import ...` | Match the existing import style |
| The first Chroma run paused for a while | The 80 MB MiniLM model was downloading | It happens only once; it's cached in `~/.cache/chroma` |
| Chunks had no code examples | The FastAPI docs use `{* ... *}` placeholders instead of inline code | Fixed in week 6 by inlining the real files before indexing |
| Old chunks could remain after documents changed | `upsert` only replaces the ids you send | `rm -rf chroma_db` and re-ingest after changing data |

How to read a Python traceback: start at the **bottom** line (what went wrong), then go up to the deepest frame that is in **your** file. Skip the frames inside `site-packages`, which is library code.

---

## 5. Revision questions

1. What are the two halves of a RAG system, and how often does each run?
2. Why does keyword search fail for "How do I tell the client a resource doesn't exist?"
3. What is an embedding, and why are its individual numbers meaningless?
4. Describe the four steps that turn a sentence into a MiniLM vector.
5. What does "contrastive learning" mean, and what data was MiniLM trained on?
6. What happens to text beyond MiniLM's input limit, and what does that imply for chunking?
7. For normalised vectors, how are squared L2 distance and cosine similarity related? Does the choice change the ranking?
8. In the drill, the 404 sentence scored 1.316 and the banana sentence 2.121. What do those numbers mean?
9. Why must the question and the chunks be embedded with the same model?
10. Why does Chroma's `query()` return lists of lists?
11. You search "Eiffel Tower" with `where={"topic": "food"}`. What comes back, and why?
12. What are the trade-offs of smaller versus larger chunks? What is the overlap for?
13. Why do deterministic ids plus `upsert` make ingestion safe to re-run? What problem remains?
14. A chunk contains `{* ../../docs_src/x.py hl[1] *}`. Why is that a problem for RAG?
15. How should you decide between MiniLM and a paid embedding model?

---

## Answers

1. **Indexing** (chunk → embed → store) runs once, whenever the documents change. **Answering** (embed the question → find the nearest chunks → LLM → answer) runs for every question.
2. The question and the document share no important words. Keyword search matches spelling, not meaning, so it finds nothing.
3. It's a fixed-length vector representing a text's meaning. The dimensions came out of training and have no individual labels. Only the distances between whole vectors are meaningful.
4. (1) Split into word pieces and look up their IDs. (2) Give each token a starting vector. (3) Six attention layers let each token adjust its vector based on the others, adding context. (4) Average the token vectors (mean pooling) and scale the result to length 1.
5. Training that pulls the vectors of related pairs together and pushes unrelated texts apart. MiniLM was trained this way on about a billion pairs: questions and answers, queries and clicked pages, titles and articles, paraphrases.
6. It is silently cut off and never influences the vector. So chunks must be small enough to fit (about 256 tokens); our 800 characters is roughly 200 tokens.
7. `squared L2 = 2 − 2 × cosine similarity`. Since one is a fixed function of the other, the ranking is identical; only the scale differs.
8. They are squared L2 distances. 1.316 is clearly closer (related), and about 2.0 or more corresponds to cosine similarity around 0, meaning unrelated.
9. Each model produces its own "coordinate system". Vectors from different models can't be compared meaningfully.
10. `query_texts` takes a list of questions. The outer list has one entry per question, and the inner list holds that question's results, so use `[0]` for a single question.
11. The food sentences. The `where` filter removes all other topics **before** ranking, so the travel sentence is never a candidate.
12. Smaller chunks give sharper vectors but less context and may split explanations; larger ones give more context but blurrier vectors. The overlap keeps a sentence on a boundary whole in at least one chunk.
13. The same chunk always gets the same id, and `upsert` replaces rows with matching ids instead of duplicating them. The remaining problem is **stale chunks**: if a document shrinks, its old trailing chunks stay in the index. Rebuild after changing data.
14. Neither the embedding model nor the LLM can follow the reference, so the chunk is missing its code. Search can't match it on code, and the answer can't show code. Inline the referenced file during data preparation.
15. Build an eval set and measure retrieval with each model on your own data, weighing quality against cost, privacy and speed. Remember that switching models means re-embedding everything.
