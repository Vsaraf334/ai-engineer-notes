# Week 7: Postgres, pgvector and hybrid search

This week the RAG system's storage moved from Chroma to **Postgres with the pgvector extension**, running locally in Docker through Docker Compose. The migration was proven with the eval suite: both backends returned identical chunks, in identical order, for all 24 questions. Integration tests run against a real database and skip cleanly when it isn't running. Finally, **hybrid search** (vector + keyword search, merged with Reciprocal Rank Fusion) was built, measured, found to be worse on this data, and moved to a branch with a written report.

**Built in:** [github.com/Vsaraf334/rag-docs](https://github.com/Vsaraf334/rag-docs): `src/pg_store.py`, `compose.yaml`, `tests/test_pg_store.py`, `evals/run_eval.py`, `docs/hybrid-search.md`, and the branch `experiment/hybrid-search`.

---

## 1. Concepts

### 1.1 Postgres, and why it's used for vectors

**PostgreSQL** is a relational database, in the same family as MySQL, Oracle and SQL Server. It's free, open source, and one of the most common databases in companies.

Ways this project has stored data so far:

| Storage | Where the data lives | Survives a restart | Shared by several apps or servers |
| --- | --- | --- | --- |
| chat-service `conversations = {}` | Python memory | No | No |
| Chroma (`chroma_db/`) | Files in the project folder | Yes | No, local to one program |
| **Postgres** | A database server reached over a port | Yes | Yes |

Why teams put vectors in Postgres instead of a separate vector database:

- **It's already there.** Most companies run Postgres, with backups, monitoring, access control and people who know how to operate it.
- **Vectors sit next to the application's data.** A single query can combine meaning-based search with ordinary SQL, for example *"the 8 nearest chunks that this user is allowed to see"*: `WHERE tenant_id = ... ORDER BY embedding <=> ... LIMIT 8`.
- **One system instead of two.** Nothing has to be kept in sync between a vector database and the main database.

### 1.2 What "local" means here

Postgres runs in a Docker container. On a Mac, containers run inside a small Linux virtual machine managed by Docker Desktop:

```
Your Mac
 ├── Python code ── connects to 127.0.0.1:5432 ──┐
 │                                                │  port forwarding (-p / ports:)
 └── Docker Desktop's Linux VM                    │
       └── container (Postgres 17 + pgvector) ◄───┘  listening on 5432 inside
             └── data folder ──► Docker volume (on the VM's disk)
```

- **Nothing is in the cloud.** The server runs on your machine, inside the VM. `SELECT version()` reports Debian Linux on aarch64 (the Mac's ARM chip).
- **Port forwarding** makes the container reachable as if Postgres were installed on the Mac.
- **The bind address matters:**
  - `-p 5432:5432` binds to `0.0.0.0`, which means **every network interface**. Other devices on the same Wi-Fi could try to connect, and the password is only `localdev`.
  - `-p 127.0.0.1:5432:5432` binds to **localhost only**, so only programs on your Mac can connect. Always prefer this for local development.
- **Volumes keep the data.** The container can be deleted and recreated, and the tables survive because they live in the volume, not in the container. (This was tested: deleting the container left all rows intact.)

In production it's the same Postgres, on a server or a managed service (AWS RDS, Supabase, Neon). Only the connection address changes.

### 1.3 pgvector

pgvector is a Postgres **extension**, a plugin that adds:

- a column type **`vector(n)`**: a fixed-length list of `n` numbers. MiniLM produces 384 numbers, so the column is `vector(384)`.
- **distance operators:**

  | Operator | Distance | Use with |
  | --- | --- | --- |
  | `<=>` | Cosine distance (0 = same direction, ~1 = unrelated, 2 = opposite) | Text embeddings (what we used) |
  | `<->` | Euclidean (L2) distance | Models that call for L2 |
  | `<#>` | Negative inner product | Normalised vectors, when speed matters |

- **index types** for large tables (see 1.6).

It has to be switched on once per database: `CREATE EXTENSION IF NOT EXISTS vector;`. Without it, Postgres has no idea what a vector is.

**Nearest-neighbour search is plain SQL:**

```sql
SELECT text, source, embedding <=> '[...query vector...]' AS distance
FROM chunks
ORDER BY distance
LIMIT 8;
```

This is exactly what Chroma's `col.query(query_texts=[q], n_results=8)` did, written as SQL.

### 1.4 You embed the text yourself now

Chroma hid a step: `col.add(documents=[...])` ran the embedding model for you. **Postgres only stores numbers**, so the code has to call the model itself:

```
Chroma:    text ──► col.add() ──► (Chroma embeds) ──► stored
Postgres:  text ──► embed()   ──► 384 numbers     ──► INSERT
```

To keep results comparable, the same model is used: Chroma's `DefaultEmbeddingFunction` (all-MiniLM-L6-v2). This was **verified before writing any code**: embedding a chunk directly gave a vector whose maximum difference from the one Chroma had stored was `0.0`.

**Rule: the question and the chunks must be embedded with the same model.** Vectors from different models are like coordinates on different maps, so comparing them means nothing. Changing the model means re-embedding everything.

### 1.5 Why Chroma's distances were exactly double

For the same chunk and question, Chroma reported 1.057 and pgvector reported 0.529. Across all 8 results, every pgvector distance was exactly half of Chroma's, **and the order was identical**.

- Chroma's default distance is **squared Euclidean** distance.
- `<=>` is **cosine distance** = 1 − cosine similarity.
- For vectors scaled to length 1 (MiniLM's are): squared Euclidean distance = 2 × (1 − cosine similarity) = **2 × cosine distance**.

A different scale, the same ranking. Never compare distance numbers *across* systems, only rankings.

### 1.6 Exact search vs approximate indexes

Without an index, `ORDER BY embedding <=> q LIMIT 8` computes the distance to **every row**, then sorts. That's an **exact** search (a sequential scan). With 135 rows it takes microseconds.

At millions of rows, scanning everything per question is too slow. pgvector offers approximate nearest neighbour (ANN) indexes:

- **HNSW** (Hierarchical Navigable Small World): a layered graph where each vector links to its neighbours. Search jumps through the graph instead of checking every row. It's very fast and has high recall, at the cost of slower inserts and more memory.
- **IVFFlat**: clusters the vectors and only searches the nearest clusters.

**"Approximate" means it can occasionally miss a true nearest neighbour** in exchange for big speed gains. It's like a B-tree index for `WHERE`, but for "nearest". We didn't need one at 135 rows. It's listed under the project's next steps.

### 1.7 psycopg 3: the Postgres driver (JDBC for Python)

| psycopg | JDBC equivalent |
| --- | --- |
| `psycopg.connect("postgresql://user:pass@host:5432/db")` | `DriverManager.getConnection("jdbc:postgresql://host:5432/db", user, pass)` |
| `autocommit=True` | `conn.setAutoCommit(true)` |
| `%s` in SQL | `?` in a `PreparedStatement` |
| `conn.execute(sql, params)` | `ps.setX(...)` + `ps.execute()` |
| `cur.executemany(sql, rows)` | `addBatch()` per row, then `executeBatch()` |
| `with connect() as conn:` | try-with-resources: closed automatically |
| `.fetchone()` | `rs.next()` + read one row (returns `None` if there are no rows) |
| `.fetchall()` | Loop over the whole `ResultSet` into a list |

**`register_vector(conn)`** (from the `pgvector` package) teaches psycopg how to send numpy arrays as Postgres `vector` values and read them back. It needs the extension to exist first, because it looks up the `vector` type in the database.

**The connection string format** is `postgresql://user:password@host:port/database`. It lives in `.env` as `DATABASE_URL`, because in real life it contains a secret.

### 1.8 Placeholders are for values only

Placeholders (`%s` in psycopg, `?` in JDBC) protect against **SQL injection**: the database receives the value separately from the SQL, so whatever is passed in is treated as **data, never as SQL**. Never build SQL with f-strings or string concatenation.

The flip side: **a placeholder can't be a column name, table name or keyword.**

```sql
ORDER BY %s        -- wrong: with a value of 8 this sorts by the constant 8, so there's no real ordering
ORDER BY distance  -- right: identifiers are written directly in the SQL
```

| Part of the query | How to write it |
| --- | --- |
| Values from code (vectors, numbers, user text) | `%s` placeholders |
| Column names, aliases, table names, `ASC`/`DESC` | Written directly in the SQL text |

### 1.9 Upsert: `ON CONFLICT ... DO UPDATE`

```sql
INSERT INTO chunks (id, source, text, embedding) VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE
SET source = EXCLUDED.source, text = EXCLUDED.text, embedding = EXCLUDED.embedding
```

- If no row with that `id` exists, insert it.
- If one exists (a conflict on the primary key), **update** it instead of failing.
- **`EXCLUDED`** is a special name meaning *"the row you tried to insert"*.

It's the same job as Chroma's `col.upsert()` and JPA's `merge()`. Combined with deterministic ids (`body.md:3`), running ingest twice leaves 135 rows, not 270.

### 1.10 Connection pools

`retrieve_pg()` opens a **new connection on every call**. Opening a Postgres connection costs a network round trip, authentication and a server process, so it's fine for an eval but too slow for a web service handling many requests.

A **connection pool** keeps a few connections open and lends them out: HikariCP in Java, `psycopg_pool.ConnectionPool` in Python. It's a next step for when this code sits behind an API.

### 1.11 Docker Compose

Compose describes containers **in a file in the repo** instead of a long `docker run` command. It's `pom.xml` for running services: anyone who clones the repo runs `docker compose up -d --wait`.

| `docker run` flag | `compose.yaml` key |
| --- | --- |
| `--name rag-postgres` | the service name under `services:` |
| image argument | `image:` |
| `-e KEY=value` | `environment:` |
| `-p 127.0.0.1:5432:5432` | `ports:` |
| `-v name:/path` | `volumes:` (plus a top-level `volumes:` block declaring the volume) |
| — | `healthcheck:` |

**Health checks.** A container being "running" doesn't mean Postgres accepts connections yet; it takes a second or two to start. The health check runs `pg_isready` every 2 seconds and marks the service **healthy** when it succeeds. `docker compose up -d --wait` waits for that, so scripts never connect too early.

**Lifecycle commands:**

| Command | Effect |
| --- | --- |
| `docker compose up -d --wait` | Start in the background and wait until healthy |
| `docker compose stop` | Stop; the data stays |
| `docker compose down` | Remove the container and network; **the volume (data) stays** |
| `docker compose down -v` | Remove everything **including the volume**, wiping the data |
| `docker compose exec postgres psql -U postgres` | A SQL prompt, addressed by **service** name |

**Volume naming:** Compose prefixes names with the project folder, so `pgdata` became `rag-docs_pgdata`, and the container became `rag-docs-postgres-1`.

### 1.12 Integration tests

Unit tests replace the outside world with mocks. **Integration tests** run against the real thing, here a real Postgres. Three problems need solving:

| Problem | Solution used | Java equivalent |
| --- | --- | --- |
| Postgres might not be running | Try to connect with a 2-second timeout; on failure, `pytest.skip(...)` with a clear message | `Assumptions.assumeTrue(...)` |
| Don't touch the real data | A separate database, `rag_test`, on the same server | A test profile / Testcontainers |
| Tests must not affect each other | `TRUNCATE chunks` before every test | `@BeforeEach` cleanup / `@Transactional` rollback |

Pytest tools that made this work:

- **`@pytest.fixture(scope="module")`** runs once per test file. That's JUnit's `@BeforeAll`; the default (per test) is `@BeforeEach`.
- **`yield` in a fixture** splits it into setup (before `yield`) and teardown (after). The test runs at the `yield`.
- **`monkeypatch.setenv("DATABASE_URL", ...)`** changes an environment variable **only for that test**. Because `connect()` reads `DATABASE_URL`, the production code talks to `rag_test` without any change.

A skipped test isn't a failure: a missing database is an environment problem, not a code bug.

### 1.13 Strategy pattern in Python: a dict of functions

Both backends follow the **same contract**: `(question, top_k) → [{"text", "source", "distance"}, ...]`. So the eval can choose one at runtime:

```python
RETRIEVERS = {"chroma": retrieve, "pg": retrieve_pg}   # no () → the functions themselves
retriever = RETRIEVERS[backend]
chunks = retriever(q["question"], top_k=top_k)
```

- **A function name without `()`** is the function as a value, like a Java method reference (`ChromaSearch::retrieve`).
- **Passing `retriever` into `evaluate()`** is dependency injection: `evaluate` doesn't know which backend it's using.
- In Java this would be a `Retriever` interface with two implementations, in a `Map<String, Retriever>`.

**argparse** turns command-line text into validated values: `--backend` with `choices=` (only `chroma`/`pg` allowed), `--top-k` with `type=int`, defaults, and a free `--help`. It's like picocli or Commons CLI, or Pydantic for command-line input.

### 1.14 Proving a migration: deterministic checks beat noisy scores

The Postgres eval run gave 19/20 retrieval and **18/20 answers**, one fewer than Chroma's 19/20. Was the migration broken?

Retrieval is deterministic, so it can be compared exactly: **for all 24 questions, both backends returned the same chunks in the same order.** So the answer difference can only come from the LLM steps. Claude's answer and the judge's verdict vary a little between runs, and a borderline answer can tip either way.

Lessons:
- **One question in 20 is 5%, and that's within noise.** Don't call it a regression or an improvement.
- **Compare the deterministic parts directly.** "Identical on 24/24" is much stronger proof than two accuracy numbers.
- To measure small effects, average several runs or use more questions.

### 1.15 Hybrid search (summary)

> The full write-up, with the SQL explained line by line, a worked RRF example and revision questions, is in [rag-docs/docs/hybrid-search.md](https://github.com/Vsaraf334/rag-docs/blob/main/docs/hybrid-search.md). This is a summary.

**The idea:** vector search matches **meaning** and keyword search matches **exact words**. Run both and merge, so that a chunk ranked well by both wins.

**Keyword search in Postgres:**
- `to_tsvector('english', text)` drops stop words and **stems** the rest: "values" → `valu`, "parameter" → `paramet`.
- A **generated column** (`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`) keeps those words up to date automatically.
- A **GIN index** is an inverted index (word → rows), the same idea as Lucene/Elasticsearch.
- `words @@ query` means "matches", and `ts_rank_cd` gives a relevance score.
- **Gotcha:** `plainto_tsquery` ANDs every word, giving **0 matches** for a natural question. Replacing `&` with `|` (OR) gave 77 candidates.

**Merging with Reciprocal Rank Fusion (RRF).** The two scores are in different units, so use only ranks:

```
score = 1/(60 + vector rank) + 1/(60 + keyword rank)       (0 if missing from a list)
```

All of it ran in **one SQL query**: two CTEs rank each method's top 50 with `row_number() OVER (...)`, a `FULL OUTER JOIN` keeps chunks found by either one, and `coalesce(..., 0)` turns a missing rank into 0.

**The result was worse, and that's a real result:**

| Question | What hybrid did |
| --- | --- |
| Restrict a URL parameter to fixed values | A "hit", but the page **introduction**, with no Enum content |
| Text where the URL expects a number | **New miss**: "client", "text" and "number" matched handling-errors chunks |
| Run code after sending the **response** | The `add_task` chunks were displaced by `response-model.md`; the answer became partial |

Keyword matching helps when users type **rare exact identifiers** (error codes, SKUs, function names). These questions were natural language, so common words added noise. The remaining miss is a **vocabulary mismatch** ("fixed set of allowed values" vs "Enum"), which keyword search can't bridge because it only matches words that are in the question.

**What happened to the code:** it was committed on the `experiment/hybrid-search` branch with the measurements in the commit message, and documented in `docs/hybrid-search.md`. `main` stays simple. Nothing is lost, and the reason for the decision is written down.

---

## 2. How we built it

### 2.1 Running Postgres: `compose.yaml`

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_PASSWORD: localdev
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 2s
      timeout: 3s
      retries: 15

volumes:
  pgdata:
```

- `pgvector/pgvector:pg17` is the official Postgres 17 image with pgvector already installed.
- `127.0.0.1:` limits access to your own machine.
- `/var/lib/postgresql/data` is where Postgres keeps its files inside the container, so mounting the volume there persists them.

### 2.2 The schema and the queries: `src/pg_store.py`

```python
embed = DefaultEmbeddingFunction()  # all-MiniLM-L6-v2, the same model Chroma used

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        text PRIMARY KEY,
    source    text NOT NULL,
    text      text NOT NULL,
    embedding vector(384) NOT NULL
);
"""

UPSERT = """
INSERT INTO chunks (id, source, text, embedding) VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE
SET source = EXCLUDED.source, text = EXCLUDED.text, embedding = EXCLUDED.embedding
"""

SEARCH = """
SELECT text, source, embedding <=> %s AS distance
FROM chunks
ORDER BY distance
LIMIT %s
"""
```

`IF NOT EXISTS` makes the schema **idempotent**: it's safe to run on every connection, so a fresh database (such as `rag_test`) gets the table automatically.

### 2.3 Connecting

```python
def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute(SCHEMA)
    register_vector(conn)  # teach psycopg the vector type (needs the extension first)
    return conn
```

The order matters: run the schema first (it creates the extension), **then** call `register_vector`.

### 2.4 Storing

```python
def store_chunks_pg(conn: psycopg.Connection, chunks: list[dict]) -> None:
    vectors = embed([c["text"] for c in chunks])
    rows = [(c["id"], c["source"], c["text"], v) for c, v in zip(chunks, vectors)]
    with conn.cursor() as cur:
        cur.executemany(UPSERT, rows)
```

1. Embed all chunk texts in one call (135 texts → 135 vectors).
2. `zip` pairs each chunk with its vector, giving one tuple per row, in the same order as the four `%s` placeholders.
3. `executemany` sends them as a batch.

Running the script prints `135 chunks in Postgres`, and running it again still prints 135.

### 2.5 Searching

```python
def retrieve_pg(question: str, top_k: int = 8) -> list[dict]:
    """Same contract as retrieve(): the top_k closest chunks, closest first."""
    query_vector = embed([question])[0]
    with connect() as conn:
        rows = conn.execute(SEARCH, (query_vector, top_k)).fetchall()
    return [{"text": t, "source": s, "distance": d} for t, s, d in rows]
```

- `embed([question])[0]`: the model takes a **list** of texts, so wrap the question in a list and take the first vector.
- `(query_vector, top_k)` fills the two `%s` placeholders in order.
- It returns the same dict keys as Chroma's `retrieve()`, so `generate()` and the eval work unchanged.

Real output for "How do I run code after sending the response?" ranked the same 8 chunks in the same order as Chroma, with distances exactly half of Chroma's (0.432 vs 0.863 for the top result).

### 2.6 The integration-test fixtures: `tests/test_pg_store.py`

```python
ADMIN_URL = os.environ.get(
    "TEST_ADMIN_URL", "postgresql://postgres:localdev@localhost:5432/postgres"
)
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + "/rag_test"


@pytest.fixture(scope="module")
def test_database():
    """Create the rag_test database once per test file, or skip if Postgres is down."""
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip(
            "Postgres is not running; start it with: docker compose up -d --wait"
        )
    with admin:
        if not admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'rag_test'"
        ).fetchone():
            admin.execute("CREATE DATABASE rag_test")
    return TEST_URL


@pytest.fixture
def conn(test_database, monkeypatch):
    """A connection to rag_test with an empty chunks table."""
    monkeypatch.setenv("DATABASE_URL", test_database)
    with connect() as conn:
        conn.execute("TRUNCATE chunks")
        yield conn
```

- `rsplit("/", 1)[0]` cuts off the database name: `.../postgres` → `...` + `/rag_test`.
- `CREATE DATABASE` can't run inside a transaction, which is one reason the admin connection uses `autocommit=True`.

The tests check that storing twice doesn't duplicate, that storing again updates changed text, that retrieval returns the closest chunk first with sorted float distances, that `top_k` is respected, and that **Postgres ranks chunks exactly like Chroma** for three questions:

```python
@pytest.mark.parametrize(
    "question", ["item not found", "dessert", "landmarks in France"]
)
def test_ranks_chunks_like_chroma(conn, tmp_path, question):
    chunks = build_chunks(DOCS)
    store_chunks_pg(conn, chunks)
    store_chunks(chunks, path=tmp_path)
    chroma_order = [r["source"] for r in retrieve(question, top_k=3, path=tmp_path)]
    pg_order = [r["source"] for r in retrieve_pg(question, top_k=3)]
    assert pg_order == chroma_order
```

With Postgres running: 28 passed. With it unreachable: 21 passed, 7 skipped, each with the "start it with" message.

### 2.7 Choosing the backend in the eval: `evals/run_eval.py`

```python
RETRIEVERS: dict[str, Callable[..., list[dict]]] = {
    "chroma": retrieve,
    "pg": retrieve_pg,
}

def evaluate(q: dict, retriever: Callable[..., list[dict]], top_k: int) -> dict:
    chunks = retriever(q["question"], top_k=top_k)
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--backend", choices=RETRIEVERS, default="chroma")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    main(args.backend, args.top_k)
```

- `choices=RETRIEVERS` works because iterating over a dict gives its keys.
- `--top-k` becomes `args.top_k`, since `-` isn't allowed in Python names.
- `--backend mysql` → `invalid choice: 'mysql' (choose from 'chroma', 'pg')`. `--top-k four` → `invalid int value: 'four'`.

### 2.8 Results

| Backend | Retrieval | Answers | Refusals |
| --- | --- | --- | --- |
| Chroma (week 6) | 19/20 | 19/20 | 4/4 |
| Postgres + pgvector | 19/20 | 18/20 | 4/4 |
| Hybrid (branch only) | 19/20 | 18/20 | 4/4 |

Retrieval was identical between Chroma and Postgres on 24/24 questions, so the answer difference is LLM variation.

---

## 3. Key points

- Postgres + pgvector puts vectors **next to application data**, searchable and filterable with ordinary SQL, in a database most companies already run.
- `vector(384)` stores an embedding; `<=>` is cosine distance; nearest neighbours = `ORDER BY embedding <=> q LIMIT k`.
- With Postgres, **you call the embedding model yourself**. Use the same model for chunks and questions.
- Chroma (squared L2) and pgvector cosine distances differ by exactly 2× for normalised vectors; **the ranking is identical**.
- No index = exact scan, fine for small tables. **HNSW** = approximate nearest neighbours, fast at millions of rows.
- psycopg maps to JDBC: `%s` = `?`, `executemany` = batch, `with` = try-with-resources, `fetchone()` may return `None`.
- **Placeholders are for values, never identifiers.** `ORDER BY %s` silently fails to sort.
- `INSERT ... ON CONFLICT (id) DO UPDATE SET col = EXCLUDED.col` is Postgres's upsert.
- Bind local databases to **127.0.0.1**, not 0.0.0.0.
- Docker volumes keep data when containers are deleted. `down` keeps the volume; `down -v` deletes it.
- Compose health checks + `up --wait` mean "start, and wait until it accepts connections".
- Integration tests: **skip** when the dependency is missing, use a **separate test database**, **reset state** before each test.
- A dict of functions plus a shared return shape = the strategy pattern without interfaces.
- **Prove migrations with deterministic checks** (identical retrieval on 24/24), not with noisy LLM accuracy scores.
- Hybrid search helps with exact identifiers. Here it **measured worse**, so it went to a branch with a written report: a negative result is still a result.

---

## 4. Gotchas we hit

| What happened | Cause | Fix |
| --- | --- | --- |
| `ProgrammingError: the query has 3 placeholders but 2 parameters were passed` | `ORDER BY %s`: a placeholder used for a column alias | `ORDER BY distance`. Placeholders are only for values |
| Pylance: "Object of type None is not subscriptable" on `fetchone()[0]` | `fetchone()` returns a row **or `None`** | `row = ...fetchone()`, then `assert row`, then `row[0]` (type narrowing) |
| The first `docker run` exposed Postgres as `0.0.0.0:5432` | `-p 5432:5432` binds every network interface | `-p 127.0.0.1:5432:5432`, kept in `compose.yaml` |
| `Cannot connect to the Docker daemon` | Docker Desktop wasn't running | Open Docker Desktop and wait for the whale icon |
| Postgres eval scored 18/20 answers vs Chroma's 19/20 | LLM output varies between runs; retrieval was identical | Compare the deterministic part (chunks) directly; treat ±1 question as noise |
| Hybrid keyword search matched **0** chunks | `plainto_tsquery` ANDs every word | Replace `&` with `|` to OR the words |
| `syntax error at or near "::"` in the hybrid SQL | A `::` cast directly in the `FROM` list isn't allowed | Wrap it in a subquery: `(SELECT ...::tsquery AS query) AS q` |
| Hybrid's "fix" for the Enum question wasn't real | The retrieval metric checks the **file**, not the section | Inspect the chunks; a section-level metric is a next step |
| Stale `words` column left in the database after removing hybrid from `main` | The schema change outlived the code | `ALTER TABLE chunks DROP COLUMN IF EXISTS words` in both databases |
| A shell demo of argparse returned "unrecognized arguments" | zsh passed `"--backend pg"` as a single argument | Split the words (`eval`), a shell-quoting issue, not a Python one |

---

## 5. Revision questions

1. Give two reasons a company might store embeddings in Postgres rather than in a dedicated vector database.
2. What does `CREATE EXTENSION vector` add, and what happens if you skip it?
3. Write the SQL for "the 5 chunks nearest to a query vector", returning text and distance.
4. Why does the Postgres version need to call the embedding model, when the Chroma version didn't?
5. Chroma said 1.057 and pgvector said 0.529 for the same chunk. Is one of them wrong? Explain.
6. What is HNSW, what does "approximate" mean, and when do you need it?
7. Why is `ORDER BY %s` wrong even when the number of placeholders matches?
8. What does `EXCLUDED` refer to in `ON CONFLICT ... DO UPDATE`?
9. What is the difference between binding a port to `0.0.0.0` and to `127.0.0.1`?
10. What's the difference between `docker compose down` and `docker compose down -v`?
11. Why add a health check when the container already shows as "running"?
12. Name three things the integration tests do so that they're safe and repeatable.
13. What does `scope="module"` change about a fixture, and what does `yield` do in a fixture?
14. How did the eval switch between Chroma and Postgres without changing `evaluate()`'s logic?
15. The Postgres eval scored one answer lower. How was it shown that the migration wasn't the cause?

---

## Answers

1. Postgres is usually already run and operated by the company (backups, monitoring, access control), and vectors sit next to the application's data, so one SQL query can combine similarity search with ordinary filters such as permissions or tenant. There's also no second system to keep in sync.
2. It loads pgvector: the `vector(n)` column type, the distance operators (`<=>`, `<->`, `<#>`) and the vector index types. Without it, `vector(384)` and `<=>` are unknown, and `register_vector` can't find the type.
3. `SELECT text, embedding <=> %s AS distance FROM chunks ORDER BY distance LIMIT 5;` with the query vector passed as the parameter.
4. Chroma ran its embedding function inside `add()` and `query()`. Postgres only stores the numbers you give it, so the code embeds chunks before inserting and embeds the question before searching, with the same model.
5. Neither. Chroma uses squared Euclidean distance and `<=>` uses cosine distance. For length-1 vectors, squared Euclidean = 2 × cosine distance, so the scale differs but the ranking is identical.
6. Hierarchical Navigable Small World: a graph index for nearest-neighbour search. "Approximate" means it can occasionally miss a true nearest neighbour in exchange for being far faster than scanning every row. It's needed when tables grow large (hundreds of thousands to millions of rows), not at 135 rows.
7. Placeholders carry values, not SQL. `ORDER BY %s` with the value 8 sorts by a constant, so the rows come back in no meaningful order, with no error. Column names and aliases must be written in the SQL.
8. The row you tried to insert. `SET text = EXCLUDED.text` copies the new value over the existing row's value.
9. `0.0.0.0` listens on every network interface, so other machines on the network can try to connect. `127.0.0.1` only accepts connections from the same machine.
10. `down` removes the containers and network but keeps the volume, so the data survives. `down -v` also deletes the volume, wiping the data.
11. "Running" only means the process started. Postgres needs a moment before it accepts connections. The health check (`pg_isready`) marks it healthy when it's actually ready, and `up --wait` waits for that.
12. Skip when Postgres is unreachable (2-second timeout); use a separate `rag_test` database so real data is never touched; `TRUNCATE` the table before every test; and point the code at the test database with `monkeypatch.setenv`, which is undone after each test.
13. `scope="module"` runs the fixture once per test file instead of before every test (`@BeforeAll` vs `@BeforeEach`). `yield` hands the value to the test; code after `yield` runs as teardown once the test finishes.
14. Both functions share the same contract. A dict maps names to the functions (`{"chroma": retrieve, "pg": retrieve_pg}`), argparse reads `--backend`, and the chosen function is passed into `evaluate()` as a parameter (dependency injection).
15. By comparing retrieval directly: both backends returned the same chunks in the same order for all 24 questions. The chunks Claude received were identical, so the difference came from run-to-run variation in Claude's answer and the judge, not from Postgres.
