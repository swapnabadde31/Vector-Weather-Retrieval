# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest NWS Weather Text -> pgvector Embeddings (Lakebase)
# MAGIC
# MAGIC The scheduled half of the weather RAG pipeline. It:
# MAGIC
# MAGIC 1. Harvests unstructured text from **api.weather.gov** - Area Forecast
# MAGIC    Discussions, Hazardous Weather Outlooks, Hydrologic Outlooks, and
# MAGIC    active watches/warnings.
# MAGIC 2. Upserts them into `weather_documents`, skipping any product whose
# MAGIC    text hasn't changed since the last run.
# MAGIC 3. Chunks each changed document on its own section boundaries and
# MAGIC    embeds the chunks with `sentence-transformers`.
# MAGIC 4. Writes the vectors into `weather_embeddings` (`pgvector`), with an
# MAGIC    explicit `::vector` cast on every insert - no array staging table
# MAGIC    and no follow-up `UPDATE ... ::vector` pass.
# MAGIC 5. Runs a sample similarity query so a green run proves retrieval works,
# MAGIC    not just that rows landed.
# MAGIC
# MAGIC ## Why this notebook uses `pg8000`, not `psycopg2`
# MAGIC
# MAGIC `psycopg2`'s C extension crashes the sandboxed kernel on Databricks
# MAGIC **Serverless** notebook compute - not a normal Python exception, a hard
# MAGIC `SIGABRT` that kills the whole kernel the moment the module is imported.
# MAGIC `app.py` (a Databricks App, not a notebook) runs in an unsandboxed
# MAGIC container and is unaffected, so it keeps using `psycopg2` via
# MAGIC `lakebase.py` / `weather_store.py`. This notebook does **not** import
# MAGIC either of those modules, and instead reimplements the same DDL, upsert,
# MAGIC and search SQL directly against
# MAGIC [`pg8000`](https://github.com/tlocke/pg8000), a pure-Python driver with
# MAGIC no C extension to crash. If you run this notebook on a classic
# MAGIC (non-serverless) cluster, `psycopg2` would work fine there too - `pg8000`
# MAGIC is used unconditionally anyway so the same notebook runs on both.
# MAGIC
# MAGIC It reuses the same Lakebase secret as the Flask app (scope `database`,
# MAGIC key `lakebase-url`) and needs no API key of its own, because NWS
# MAGIC authenticates with nothing but a User-Agent header.
# MAGIC
# MAGIC Runs on Databricks Serverless (Free Edition) and as a plain script:
# MAGIC `LAKEBASE_URL=postgresql://... python notebooks/ingest_weather_embeddings.py`
# MAGIC
# MAGIC ## Run this top to bottom (`Run all`), not cell-by-cell out of order
# MAGIC
# MAGIC The second cell calls `dbutils.library.restartPython()` right after
# MAGIC `%pip install`, which clears **every** Python variable in the kernel.
# MAGIC If you then re-run a single cell further down in isolation - common
# MAGIC while iterating on, say, just the schema or just the search cell -
# MAGIC without first re-running the cells above it in that same session, you'll
# MAGIC hit `NameError: name '...' is not defined` for whatever variable that
# MAGIC cell expected from earlier (`EMBEDDING_DIM`, `conn`, `client`, `model`,
# MAGIC `pending`, etc.). If you see that, use **Run all**, or right-click the
# MAGIC cell that failed and choose **Run cells above**, rather than re-running
# MAGIC just the one cell.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q 'databricks-sdk>=0.30.0' pg8000 sentence-transformers requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Widgets let a scheduled Job override every knob without editing code.
# MAGIC When there is no `dbutils` (running as a plain script), the same names
# MAGIC are read from environment variables instead.

# COMMAND ----------

# DBTITLE 1,Widgets and parameters
import os

try:
    dbutils  # noqa: B018 - probing for the notebook runtime
    _IN_DATABRICKS = True
except NameError:
    _IN_DATABRICKS = False

_WIDGET_DEFAULTS = {
    "offices": "OKX,LWX,MPX,FWD,SEW,MFL,BOU,OUN,SLC,TAE",
    "product_types": "AFD,HWO,ESF",
    "limit_per_pair": "2",
    "include_alerts": "true",
    "alert_area": "",
    "alert_limit": "50",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": "900",
    "chunk_overlap": "150",
    "max_requests_per_minute": "60",
    "documents_table": "weather_documents",
    "embeddings_table": "weather_embeddings",
    "reembed_all": "false",
    "repo_path": "",
}

if _IN_DATABRICKS:
    for name, default in _WIDGET_DEFAULTS.items():
        dbutils.widgets.text(name, default, name)


def param(name: str) -> str:
    if _IN_DATABRICKS:
        return dbutils.widgets.get(name)
    return os.environ.get(name.upper(), _WIDGET_DEFAULTS[name])


def csv_param(name: str) -> list[str]:
    return [v.strip().upper() for v in param(name).split(",") if v.strip()]


def bool_param(name: str) -> bool:
    return param(name).strip().lower() in ("true", "1", "yes")


OFFICES = csv_param("offices")
PRODUCT_TYPES = csv_param("product_types")
LIMIT_PER_PAIR = int(param("limit_per_pair"))
INCLUDE_ALERTS = bool_param("include_alerts")
ALERT_AREA = param("alert_area").strip() or None
ALERT_LIMIT = int(param("alert_limit"))
EMBEDDING_MODEL = param("embedding_model")
CHUNK_SIZE = int(param("chunk_size"))
CHUNK_OVERLAP = int(param("chunk_overlap"))
MAX_RPM = int(param("max_requests_per_minute"))
DOCUMENTS_TABLE = param("documents_table")
EMBEDDINGS_TABLE = param("embeddings_table")
REEMBED_ALL = bool_param("reembed_all")

print(f"Offices        : {OFFICES}")
print(f"Product types  : {PRODUCT_TYPES}")
print(f"Model          : {EMBEDDING_MODEL}")
print(f"Chunking       : {CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import `weather_client` and `embeddings` only
# MAGIC
# MAGIC These two modules are pure - no `psycopg2` anywhere in their import
# MAGIC chain - so it's safe to import them directly from the repo root. This is
# MAGIC what keeps the notebook's harvesting and chunking logic identical to the
# MAGIC Flask app's: if the app embeds a query one way and this notebook embeds
# MAGIC the corpus a different way, cosine similarity stops meaning anything.
# MAGIC
# MAGIC `weather_store.py` and `lakebase.py` are deliberately **not** imported -
# MAGIC both pull in `psycopg2` at module load time, which is exactly the import
# MAGIC that crashes the serverless kernel. All of the DDL/upsert/search SQL
# MAGIC those modules contain is reimplemented below against `pg8000` instead.

# COMMAND ----------

# DBTITLE 1,Put the repo root on sys.path
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Locate the folder holding weather_client.py, from a notebook or a script."""
    explicit = param("repo_path").strip()
    if explicit:
        return Path(explicit)

    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass
    if _IN_DATABRICKS:
        try:
            notebook_path = (
                dbutils.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .notebookPath()
                .get()
            )
            candidates.append(Path("/Workspace") / notebook_path.lstrip("/"))
            candidates.append((Path("/Workspace") / notebook_path.lstrip("/")).parent.parent)
        except Exception:  # noqa: BLE001 - context API varies across runtimes
            pass
    candidates.append(Path.cwd())
    candidates.append(Path.cwd().parent)

    for candidate in candidates:
        if (candidate / "weather_client.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the repo root (the folder containing weather_client.py). "
        "Set the `repo_path` widget to its absolute path, e.g. "
        "/Workspace/Users/you@example.com/databricks-lakebase-weather-rag"
    )


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print(f"Repo root: {REPO_ROOT}")

import embeddings as emb  # noqa: E402  -  pure, no psycopg2
from weather_client import NWSClient  # noqa: E402  -  pure, no psycopg2

EMBEDDING_DIM = emb.resolve_dimension(EMBEDDING_MODEL)
print(f"Vector width: {EMBEDDING_DIM}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to Lakebase with `pg8000`
# MAGIC
# MAGIC Same secret the Flask app uses (`database/lakebase-url`), parsed into
# MAGIC discrete connection arguments. `pg8000` speaks the Postgres wire
# MAGIC protocol directly in Python - no compiled `libpq` binding to conflict
# MAGIC with the notebook sandbox - and accepts the same `%s`-style parameter
# MAGIC placeholders as `psycopg2`, so the SQL below reads the same way it would
# MAGIC against either driver.

# COMMAND ----------

# DBTITLE 1,Resolve the connection and open a session
import base64
import ssl
from urllib.parse import urlparse

import pg8000.dbapi as pg8000


def resolve_lakebase_url() -> str:
    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        return env_url

    scope = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
    key = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

    if _IN_DATABRICKS:
        # dbutils returns the plaintext directly inside a notebook.
        return dbutils.secrets.get(scope=scope, key=key)

    from databricks.sdk import WorkspaceClient

    value = WorkspaceClient().secrets.get_secret(scope=scope, key=key).value
    return base64.b64decode(value).decode("utf-8")


parsed = urlparse(resolve_lakebase_url())
CONN_PARAMS = {
    "host": parsed.hostname,
    "port": parsed.port or 5432,
    "database": (parsed.path or "").lstrip("/") or "databricks_postgres",
    "user": parsed.username,
    "password": parsed.password,
    "ssl_context": ssl.create_default_context(),
    "timeout": 30,
}

print(f"Host    : {CONN_PARAMS['host']}:{CONN_PARAMS['port']}")
print(f"Database: {CONN_PARAMS['database']}")
print(f"Role    : {CONN_PARAMS['user']}")

conn = pg8000.connect(**CONN_PARAMS)
conn.autocommit = False


def q(sql: str, params: tuple = ()) -> list[tuple]:
    """Execute a query and fetch all rows. Opens/closes its own cursor."""
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        try:
            return cur.fetchall()
        except Exception:  # noqa: BLE001 - statement had no result set
            return []
    finally:
        cur.close()


def x(sql: str, params: tuple = ()) -> None:
    """Execute a statement with no result set expected (DDL, plain writes)."""
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
    finally:
        cur.close()


print("Connected:", q("SELECT version()")[0][0].split(",")[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the schema
# MAGIC
# MAGIC Enables `pgvector`, creates `weather_documents` and `weather_embeddings`,
# MAGIC and builds the HNSW cosine index. Idempotent, so it runs on every
# MAGIC execution and needs no manual SQL step beforehand. (The equivalent
# MAGIC statements, generated from the exact same source of truth, are also in
# MAGIC `sql/` if you would rather apply them by hand through the Lakebase SQL
# MAGIC editor.)
# MAGIC
# MAGIC HNSW rather than IVFFlat: IVFFlat needs a representative sample of rows
# MAGIC to build usable lists and degrades badly when the table is small or
# MAGIC grows in bursts - exactly the shape of a weather corpus that refills
# MAGIC every few hours. HNSW is built incrementally and needs no training pass.
# MAGIC
# MAGIC **If this cell is ever re-run on its own** (e.g. while iterating on the
# MAGIC schema) in a kernel session where the cells above it did *not* also run
# MAGIC - most commonly right after `dbutils.library.restartPython()`, which
# MAGIC clears every Python variable - `EMBEDDING_MODEL`/`EMBEDDING_DIM` would
# MAGIC otherwise raise `NameError`. Both are cheap, pure lookups (a widget read
# MAGIC and a dict lookup, no network or DB calls), so they're recomputed here
# MAGIC rather than trusted from the "Import" cell above. `conn`/`q`/`x` are not
# MAGIC recoverable this way - if those are undefined, the fix is **Run all**
# MAGIC (or "Run cells above"), not a local patch, since re-deriving a live
# MAGIC connection isn't a one-line fix.

# COMMAND ----------

EMBEDDING_MODEL = param("embedding_model")
EMBEDDING_DIM = emb.resolve_dimension(EMBEDDING_MODEL)

DDL = [
    f"""
    CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
        id              TEXT PRIMARY KEY,
        source          TEXT NOT NULL,
        product_code    TEXT NOT NULL,
        product_name    TEXT,
        office_id       TEXT,
        wmo_id          TEXT,
        headline        TEXT,
        area_desc       TEXT,
        severity        TEXT,
        certainty       TEXT,
        urgency         TEXT,
        issued_at       TIMESTAMPTZ,
        effective_at    TIMESTAMPTZ,
        expires_at      TIMESTAMPTZ,
        raw_text        TEXT NOT NULL,
        text_sha256     TEXT NOT NULL,
        char_count      INTEGER NOT NULL DEFAULT 0,
        payload         JSONB,
        synced_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_office ON {DOCUMENTS_TABLE} (office_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_code ON {DOCUMENTS_TABLE} (product_code)",
    f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_issued ON {DOCUMENTS_TABLE} (issued_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_source ON {DOCUMENTS_TABLE} (source)",
    "CREATE EXTENSION IF NOT EXISTS vector",
    f"""
    CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
        id            TEXT PRIMARY KEY,
        document_id   TEXT NOT NULL REFERENCES {DOCUMENTS_TABLE}(id) ON DELETE CASCADE,
        chunk_index   INTEGER NOT NULL,
        chunk_text    TEXT NOT NULL,
        char_count    INTEGER NOT NULL DEFAULT 0,
        product_code  TEXT,
        office_id     TEXT,
        issued_at     TIMESTAMPTZ,
        embedding     VECTOR({EMBEDDING_DIM}) NOT NULL,
        model_name    TEXT NOT NULL,
        embedded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index)
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_document ON {EMBEDDINGS_TABLE} (document_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_model ON {EMBEDDINGS_TABLE} (model_name)",
    f"""
    CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_vector
    ON {EMBEDDINGS_TABLE} USING hnsw (embedding vector_cosine_ops)
    """,
]

for statement in DDL:
    x(statement)
conn.commit()
print(f"Schema ready: {DOCUMENTS_TABLE}, {EMBEDDINGS_TABLE} (VECTOR({EMBEDDING_DIM}))")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Harvest text products and alerts from NWS
# MAGIC
# MAGIC Requests are made serially and spaced to `max_requests_per_minute`.
# MAGIC NWS publishes no hard quota, but sustained parallel bursts get
# MAGIC throttled - and during severe weather, when the corpus is most worth
# MAGIC refreshing, the API is already under load from everyone else. Being a
# MAGIC polite client is the difference between a job that always finishes and
# MAGIC one that fails exactly when it matters.
# MAGIC
# MAGIC Fetching is two-phase by design: the collection endpoint returns
# MAGIC metadata only, so each product's body costs a second call.

# COMMAND ----------

# DBTITLE 1,Fetch documents
import time

started = time.time()
client = NWSClient(max_requests_per_minute=MAX_RPM)

documents = list(
    client.iter_text_products(
        product_types=PRODUCT_TYPES,
        offices=OFFICES,
        limit_per_pair=LIMIT_PER_PAIR,
    )
)
print(f"Fetched {len(documents)} text products in {time.time() - started:.0f}s")

if INCLUDE_ALERTS and ALERT_AREA:
    alerts = client.get_active_alerts(area=ALERT_AREA, limit=ALERT_LIMIT)
    print(f"Fetched {len(alerts)} active alerts")
    documents.extend(alerts)
elif INCLUDE_ALERTS and not ALERT_AREA:
    alerts = client.get_active_alerts(limit=ALERT_LIMIT)
    print(f"Fetched {len(alerts)} active alerts")
    documents.extend(alerts)

if documents:
    sample = documents[0]
    print(f"\nSample: {sample['product_code']} / {sample['office_id']} / {sample['issued_at']}")
    print(sample["raw_text"][:400].replace("\n", " ") + " ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert into `weather_documents`
# MAGIC
# MAGIC One multi-row `INSERT ... VALUES (...),(...),... ON CONFLICT DO UPDATE`
# MAGIC per batch, with `WHERE text_sha256 IS DISTINCT FROM EXCLUDED.text_sha256`
# MAGIC so a row that hasn't actually changed produces no write - `RETURNING id`
# MAGIC then tells us exactly which documents changed. AFDs are reissued on a
# MAGIC fixed schedule but often barely change between issuances, so on a
# MAGIC typical run most documents are no-ops, and only the ones that genuinely
# MAGIC moved get re-embedded in the next cell.
# MAGIC
# MAGIC `pg8000` does not have `psycopg2.extras.execute_values`, so the
# MAGIC multi-row `VALUES` list is built by hand here - `execute_values` was the
# MAGIC only piece of `weather_store.py` that couldn't be ported as-is.

# COMMAND ----------

import hashlib
import json


def text_digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def upsert_documents(documents: list[dict], batch_size: int = 100) -> dict:
    """Upsert documents in batches, returning counts. Mirrors weather_store.upsert_documents."""
    rows = []
    for doc in documents:
        text = doc.get("raw_text") or ""
        if not text.strip():
            continue
        rows.append(
            (
                str(doc["id"]),
                doc.get("source") or "unknown",
                doc.get("product_code") or "UNKNOWN",
                doc.get("product_name"),
                doc.get("office_id"),
                doc.get("wmo_id"),
                doc.get("headline"),
                doc.get("area_desc"),
                doc.get("severity"),
                doc.get("certainty"),
                doc.get("urgency"),
                doc.get("issued_at"),
                doc.get("effective_at"),
                doc.get("expires_at"),
                text,
                text_digest(text),
                len(text),
                json.dumps(doc.get("payload") or {}),
            )
        )

    if not rows:
        return {"received": len(documents), "written": 0, "unchanged": len(documents)}

    written = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        placeholders = ",".join(
            ["(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::timestamptz,"
             "%s::timestamptz,%s,%s,%s,%s::jsonb)"] * len(batch)
        )
        flat = [value for row in batch for value in row]
        sql = f"""
            INSERT INTO {DOCUMENTS_TABLE} (
                id, source, product_code, product_name, office_id, wmo_id,
                headline, area_desc, severity, certainty, urgency,
                issued_at, effective_at, expires_at,
                raw_text, text_sha256, char_count, payload
            ) VALUES {placeholders}
            ON CONFLICT (id) DO UPDATE SET
                source        = EXCLUDED.source,
                product_code  = EXCLUDED.product_code,
                product_name  = EXCLUDED.product_name,
                office_id     = EXCLUDED.office_id,
                wmo_id        = EXCLUDED.wmo_id,
                headline      = EXCLUDED.headline,
                area_desc     = EXCLUDED.area_desc,
                severity      = EXCLUDED.severity,
                certainty     = EXCLUDED.certainty,
                urgency       = EXCLUDED.urgency,
                issued_at     = EXCLUDED.issued_at,
                effective_at  = EXCLUDED.effective_at,
                expires_at    = EXCLUDED.expires_at,
                raw_text      = EXCLUDED.raw_text,
                text_sha256   = EXCLUDED.text_sha256,
                char_count    = EXCLUDED.char_count,
                payload       = EXCLUDED.payload,
                updated_at    = now()
            WHERE {DOCUMENTS_TABLE}.text_sha256 IS DISTINCT FROM EXCLUDED.text_sha256
            RETURNING id
        """
        written += len(q(sql, tuple(flat)))
        conn.commit()

    return {"received": len(documents), "written": written, "unchanged": len(rows) - written}


result = upsert_documents(documents)
print(f"Received : {result['received']}")
print(f"Written  : {result['written']}   (new or changed)")
print(f"Unchanged: {result['unchanged']} (skipped)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select what needs embedding
# MAGIC
# MAGIC A document needs work if it has no chunks for this model, or its
# MAGIC chunks predate its last text change. Set the `reembed_all` widget to
# MAGIC `true` after changing the chunking parameters or the model, when every
# MAGIC existing vector is stale by definition.

# COMMAND ----------

if REEMBED_ALL:
    x(f"DELETE FROM {EMBEDDINGS_TABLE}")
    conn.commit()
    print("reembed_all=true - cleared existing vectors")

pending_rows = q(
    f"""
    SELECT d.id, d.source, d.product_code, d.office_id, d.issued_at, d.raw_text
    FROM {DOCUMENTS_TABLE} d
    WHERE NOT EXISTS (
        SELECT 1 FROM {EMBEDDINGS_TABLE} e
        WHERE e.document_id = d.id
          AND e.model_name = %s
          AND e.embedded_at >= d.updated_at
    )
    ORDER BY d.issued_at DESC NULLS LAST
    """,
    (EMBEDDING_MODEL,),
)
pending = [
    {
        "id": row[0],
        "source": row[1],
        "product_code": row[2],
        "office_id": row[3],
        "issued_at": row[4],
        "raw_text": row[5],
    }
    for row in pending_rows
]
print(f"{len(pending)} documents need embedding with {EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk and embed
# MAGIC
# MAGIC Chunking is section-aware: `embeddings.chunk_weather_text` unwraps the
# MAGIC teletype line breaks, splits on the product's own `.SHORT TERM...` /
# MAGIC `.AVIATION...` / `&&` markers, and only falls back to a sliding
# MAGIC character window inside sections that are still too long for the
# MAGIC model's 256-token context.
# MAGIC
# MAGIC Embedding runs on the driver. all-MiniLM-L6-v2 is ~80 MB and handles a
# MAGIC few thousand chunks per minute on CPU, which covers a corpus of this
# MAGIC size comfortably - and Free Edition serverless has no GPU to
# MAGIC distribute to anyway.
# MAGIC
# MAGIC Writing is delete-then-insert per document, not upsert-by-chunk-index:
# MAGIC when a product is reissued its text changes length, so the new version
# MAGIC may have fewer chunks than the old one. Deleting first means a shorter
# MAGIC reissue can never leave a surplus chunk behind as an orphaned passage
# MAGIC that still matches queries - a stale forecast surfacing as if current.

# COMMAND ----------

# DBTITLE 1,Compute and write vectors
def replace_document_embeddings(
    document_id: str,
    chunks: list[str],
    vectors: list[list[float]],
    product_code,
    office_id,
    issued_at,
) -> int:
    """Delete all vectors for one document, then insert the current set. Mirrors
    weather_store.replace_document_embeddings."""
    x(f"DELETE FROM {EMBEDDINGS_TABLE} WHERE document_id = %s", (document_id,))
    if not chunks:
        conn.commit()
        return 0

    rows = [
        (
            f"{document_id}::{index}",
            document_id,
            index,
            chunk,
            len(chunk),
            product_code,
            office_id,
            issued_at,
            emb.to_pgvector(vector),
            EMBEDDING_MODEL,
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    placeholders = ",".join(
        ["(%s,%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::vector,%s,now())"] * len(rows)
    )
    flat = [value for row in rows for value in row]
    sql = f"""
        INSERT INTO {EMBEDDINGS_TABLE} (
            id, document_id, chunk_index, chunk_text, char_count,
            product_code, office_id, issued_at, embedding, model_name, embedded_at
        ) VALUES {placeholders}
        ON CONFLICT (id) DO UPDATE SET
            chunk_text  = EXCLUDED.chunk_text,
            char_count  = EXCLUDED.char_count,
            embedding   = EXCLUDED.embedding,
            model_name  = EXCLUDED.model_name,
            embedded_at = EXCLUDED.embedded_at
    """
    x(sql, tuple(flat))
    conn.commit()
    return len(chunks)


model = emb.load_model(EMBEDDING_MODEL)

total_chunks = 0
processed = 0
skipped = 0

for i, doc in enumerate(pending, start=1):
    chunks = emb.chunk_weather_text(
        doc["raw_text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
    )
    if not chunks:
        skipped += 1
        continue

    vectors = emb.embed_texts(chunks, model=model, batch_size=32)

    total_chunks += replace_document_embeddings(
        document_id=doc["id"],
        chunks=chunks,
        vectors=vectors,
        product_code=doc.get("product_code"),
        office_id=doc.get("office_id"),
        issued_at=doc.get("issued_at"),
    )
    processed += 1

    if i % 10 == 0 or i == len(pending):
        print(f"  {i}/{len(pending)} documents -> {total_chunks} chunks")

print(f"\nEmbedded {processed} documents into {total_chunks} chunks")
if skipped:
    print(f"Skipped {skipped} documents that produced no usable chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Row counts, then per-product coverage.

# COMMAND ----------

stats_row = q(
    f"""
    SELECT
        (SELECT count(*) FROM {DOCUMENTS_TABLE})                     AS documents,
        (SELECT count(*) FROM {EMBEDDINGS_TABLE})                    AS chunks,
        (SELECT count(DISTINCT document_id) FROM {EMBEDDINGS_TABLE}) AS embedded_documents,
        (SELECT count(DISTINCT office_id) FROM {DOCUMENTS_TABLE})    AS offices,
        (SELECT max(issued_at) FROM {DOCUMENTS_TABLE})               AS latest_issued_at,
        (SELECT max(embedded_at) FROM {EMBEDDINGS_TABLE})            AS latest_embedded_at
    """
)[0]
for label, value in zip(
    ("documents", "chunks", "embedded_documents", "offices", "latest_issued_at", "latest_embedded_at"),
    stats_row,
):
    print(f"{label:22}: {value}")

coverage = q(
    f"""
    SELECT d.product_code,
           count(DISTINCT d.id)     AS documents,
           count(e.id)              AS chunks,
           round(avg(e.char_count)) AS avg_chunk_chars
    FROM {DOCUMENTS_TABLE} d
    LEFT JOIN {EMBEDDINGS_TABLE} e ON e.document_id = d.id
    GROUP BY d.product_code
    ORDER BY documents DESC
    """
)
print("\nproduct_code  documents  chunks  avg_chunk_chars")
for row in coverage:
    print(f"{row[0]:<13} {row[1]:>9} {row[2]:>7} {str(row[3]):>16}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test the retrieval path
# MAGIC
# MAGIC The same query the Flask endpoint serves, run through the same
# MAGIC cosine-similarity SQL `weather_store.search` uses. A run that inserts
# MAGIC rows but retrieves nothing is a failed run, and this is what catches it
# MAGIC - a dimension mismatch or a missing HNSW index shows up here rather than
# MAGIC in production traffic.

# COMMAND ----------

def search(query_vector: list[float], limit: int = 3) -> list[dict]:
    """Grouped-by-document cosine search. Mirrors weather_store.search's default mode."""
    vector_literal = emb.to_pgvector(query_vector)
    candidates = max(limit, limit * 6)
    sql = f"""
        WITH candidates AS (
            SELECT
                e.document_id, e.chunk_text, d.product_code, d.office_id,
                1 - (e.embedding <=> %s::vector) AS similarity
            FROM {EMBEDDINGS_TABLE} e
            JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
            WHERE e.model_name = %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        ),
        best_per_document AS (
            SELECT DISTINCT ON (document_id) *
            FROM candidates
            ORDER BY document_id, similarity DESC
        )
        SELECT * FROM best_per_document
        ORDER BY similarity DESC
        LIMIT %s
    """
    rows = q(sql, (vector_literal, EMBEDDING_MODEL, vector_literal, candidates, limit))
    return [
        {"document_id": r[0], "chunk_text": r[1], "product_code": r[2], "office_id": r[3], "similarity": r[4]}
        for r in rows
    ]


SAMPLE_QUERIES = [
    "flash flood risk this weekend",
    "severe thunderstorms with damaging wind and hail",
    "heavy mountain snow and travel impacts",
]

for query in SAMPLE_QUERIES:
    vector = emb.embed_query(query, EMBEDDING_MODEL)
    hits = search(vector, limit=3)
    print(f"\n=== {query!r} -> {len(hits)} hits")
    for hit in hits:
        preview = " ".join(hit["chunk_text"].split())[:160]
        print(
            f"  {hit['similarity']:.3f}  {hit['product_code']:<6} "
            f"{hit['office_id'] or '--':<4} {preview}..."
        )

if not SAMPLE_QUERIES:
    print("No sample queries configured")

# COMMAND ----------

conn.close()
print("Done.")
