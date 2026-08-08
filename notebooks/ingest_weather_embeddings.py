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
# MAGIC 4. Writes the vectors into `weather_embeddings` (`pgvector`) using
# MAGIC    `psycopg2` with an explicit `::vector` cast - no array staging table
# MAGIC    and no follow-up `UPDATE ... ::vector` pass.
# MAGIC 5. Runs a sample similarity query so a green run proves retrieval works,
# MAGIC    not just that rows landed.
# MAGIC
# MAGIC It reuses the same Lakebase secret as the Flask app (scope `database`,
# MAGIC key `lakebase-url`) and needs no API key of its own, because NWS
# MAGIC authenticates with nothing but a User-Agent header.
# MAGIC
# MAGIC Runs on Databricks Serverless (Free Edition) and as a plain script:
# MAGIC `LAKEBASE_URL=postgresql://... python notebooks/ingest_weather_embeddings.py`

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q 'databricks-sdk>=0.30.0' psycopg2-binary sentence-transformers requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Widgets let a scheduled Job override every knob without editing code.
# MAGIC When there is no `dbutils` (running as a script), the same names are
# MAGIC read from environment variables instead.

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
# MAGIC ## Import the shared pipeline modules
# MAGIC
# MAGIC `weather_client`, `embeddings`, and `weather_store` live at the repo
# MAGIC root. Importing them here rather than copying their logic into the
# MAGIC notebook is what guarantees the notebook chunks and embeds text exactly
# MAGIC the way `/weather/search` embeds the query - if those two ever diverge,
# MAGIC cosine similarity stops meaning anything.

# COMMAND ----------

# DBTITLE 1,Put the repo root on sys.path
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Locate the folder holding weather_store.py, from a notebook or a script."""
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
        if (candidate / "weather_store.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the repo root (the folder containing weather_store.py). "
        "Set the `repo_path` widget to its absolute path, e.g. "
        "/Workspace/Users/you@example.com/databricks-lakebase-weather-rag"
    )


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print(f"Repo root: {REPO_ROOT}")

import embeddings as emb  # noqa: E402
import weather_store  # noqa: E402
from weather_client import NWSClient  # noqa: E402

EMBEDDING_DIM = emb.resolve_dimension(EMBEDDING_MODEL)
print(f"Vector width: {EMBEDDING_DIM}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to Lakebase
# MAGIC
# MAGIC Same secret the Flask app uses. The URL is parsed into discrete
# MAGIC psycopg2 keyword arguments so a password containing URL-reserved
# MAGIC characters survives intact.

# COMMAND ----------

# DBTITLE 1,Resolve the connection and open a session
import base64
from urllib.parse import urlparse

import psycopg2


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
    "dbname": (parsed.path or "").lstrip("/") or "databricks_postgres",
    "user": parsed.username,
    "password": parsed.password,
    "sslmode": "require",
    "connect_timeout": 30,
}

print(f"Host    : {CONN_PARAMS['host']}:{CONN_PARAMS['port']}")
print(f"Database: {CONN_PARAMS['dbname']}")
print(f"Role    : {CONN_PARAMS['user']}")

conn = psycopg2.connect(**CONN_PARAMS)
with conn.cursor() as cur:
    cur.execute("SELECT version()")
    print("Connected:", cur.fetchone()[0].split(",")[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the schema
# MAGIC
# MAGIC `ensure_schema` enables `pgvector`, creates `weather_documents` and
# MAGIC `weather_embeddings`, and builds the HNSW cosine index. Idempotent, so
# MAGIC it runs on every execution and the notebook needs no manual SQL step
# MAGIC beforehand. (The equivalent statements are in `sql/` if you would
# MAGIC rather apply them by hand.)

# COMMAND ----------

weather_store.DOCUMENTS_TABLE = DOCUMENTS_TABLE
weather_store.EMBEDDINGS_TABLE = EMBEDDINGS_TABLE

weather_store.ensure_schema(
    conn,
    dim=EMBEDDING_DIM,
    documents_table=DOCUMENTS_TABLE,
    embeddings_table=EMBEDDINGS_TABLE,
)
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

if INCLUDE_ALERTS:
    alerts = client.get_active_alerts(area=ALERT_AREA, limit=ALERT_LIMIT)
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
# MAGIC The upsert compares a SHA-256 of the body and skips rows whose text is
# MAGIC byte-identical to what is already stored. AFDs are reissued on a fixed
# MAGIC schedule but often barely change between issuances, so on a typical run
# MAGIC most documents are no-ops - and only the ones that genuinely moved get
# MAGIC re-embedded in the next cell.

# COMMAND ----------

result = weather_store.upsert_documents(conn, documents, table=DOCUMENTS_TABLE)
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
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {EMBEDDINGS_TABLE}")
    conn.commit()
    print("reembed_all=true - cleared existing vectors")

pending = weather_store.documents_needing_embedding(
    conn,
    model_name=EMBEDDING_MODEL,
    documents_table=DOCUMENTS_TABLE,
    embeddings_table=EMBEDDINGS_TABLE,
)
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
# MAGIC distribute to anyway. A Spark pandas UDF would be the move at a scale
# MAGIC where the driver becomes the bottleneck.

# COMMAND ----------

# DBTITLE 1,Compute and write vectors
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

    total_chunks += weather_store.replace_document_embeddings(
        conn,
        document_id=doc["id"],
        chunks=chunks,
        vectors=vectors,
        model_name=EMBEDDING_MODEL,
        product_code=doc.get("product_code"),
        office_id=doc.get("office_id"),
        issued_at=doc.get("issued_at"),
        table=EMBEDDINGS_TABLE,
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

stats = weather_store.stats(
    conn, documents_table=DOCUMENTS_TABLE, embeddings_table=EMBEDDINGS_TABLE
)
for key, value in stats.items():
    print(f"{key:22}: {value}")

with conn.cursor() as cur:
    cur.execute(
        f"""
        SELECT d.product_code,
               count(DISTINCT d.id)  AS documents,
               count(e.id)           AS chunks,
               round(avg(e.char_count)) AS avg_chunk_chars
        FROM {DOCUMENTS_TABLE} d
        LEFT JOIN {EMBEDDINGS_TABLE} e ON e.document_id = d.id
        GROUP BY d.product_code
        ORDER BY documents DESC
        """
    )
    print("\nproduct_code  documents  chunks  avg_chunk_chars")
    for row in cur.fetchall():
        print(f"{row[0]:<13} {row[1]:>9} {row[2]:>7} {str(row[3]):>16}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test the retrieval path
# MAGIC
# MAGIC The same query the Flask endpoint serves, run through the same
# MAGIC `weather_store.search` code. A run that inserts rows but retrieves
# MAGIC nothing is a failed run, and this is what catches it - a dimension
# MAGIC mismatch or a missing HNSW index shows up here rather than in
# MAGIC production traffic.

# COMMAND ----------

SAMPLE_QUERIES = [
    "flash flood risk this weekend",
    "severe thunderstorms with damaging wind and hail",
    "heavy mountain snow and travel impacts",
]

for query in SAMPLE_QUERIES:
    vector = emb.embed_query(query, EMBEDDING_MODEL)
    hits = weather_store.search(
        conn,
        query_vector=vector,
        limit=3,
        model_name=EMBEDDING_MODEL,
        documents_table=DOCUMENTS_TABLE,
        embeddings_table=EMBEDDINGS_TABLE,
    )
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
