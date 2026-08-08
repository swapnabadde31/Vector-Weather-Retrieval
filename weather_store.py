"""
Lakebase persistence layer for the weather RAG pipeline.

Owns the DDL for `weather_documents` and `weather_embeddings`, the upsert
logic, and the pgvector similarity query. Deliberately depends on nothing but
psycopg2 - no Flask, no Databricks SDK - so the exact same code runs in the
Databricks App, in the ingestion notebook, and in the local CLI. Every
function takes an open connection rather than opening its own, so callers
control transaction boundaries.

Schema at a glance:

    weather_documents            one row per NWS product or alert
      id (PK)                    upstream product/alert id
      raw_text                   the unstructured body we embed
      text_sha256                change detector - drives re-embedding
      + provenance and filter columns (office, product code, times, severity)

    weather_embeddings           one row per chunk, N rows per document
      document_id (FK, CASCADE)  delete a document, its vectors go with it
      chunk_index                position within the document
      chunk_text                 the passage returned to the caller
      embedding VECTOR(dim)      pgvector column, HNSW + cosine
      model_name                 which model produced this vector

The two-table split matters: documents are the unit of *sync* (idempotent,
keyed on upstream id) and chunks are the unit of *retrieval*. Keeping them
apart means a re-embed with a different model rewrites only the vector table
and never re-fetches from the API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Sequence

from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")

# Upper bound for the `since_hours` search filter (10 years).
MAX_SINCE_HOURS = 24 * 365 * 10


# ------------------------------------------------------------------- DDL --


def documents_ddl(table: str = DOCUMENTS_TABLE) -> list[str]:
    """Statements that create the raw document store and its indexes."""
    return [
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
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
        f"CREATE INDEX IF NOT EXISTS idx_{table}_office ON {table} (office_id)",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_code ON {table} (product_code)",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_issued ON {table} (issued_at DESC)",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_source ON {table} (source)",
    ]


def embeddings_ddl(
    dim: int,
    table: str = EMBEDDINGS_TABLE,
    documents_table: str = DOCUMENTS_TABLE,
) -> list[str]:
    """
    Statements that create the chunk/vector table and its HNSW index.

    HNSW rather than IVFFlat: IVFFlat needs a representative sample of rows
    to build usable lists, and it degrades badly when the table is small or
    grows in bursts - which is exactly the shape of a weather corpus that
    refills every few hours. HNSW is built incrementally and needs no
    training pass.
    """
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id            TEXT PRIMARY KEY,
            document_id   TEXT NOT NULL REFERENCES {documents_table}(id) ON DELETE CASCADE,
            chunk_index   INTEGER NOT NULL,
            chunk_text    TEXT NOT NULL,
            char_count    INTEGER NOT NULL DEFAULT 0,
            product_code  TEXT,
            office_id     TEXT,
            issued_at     TIMESTAMPTZ,
            embedding     VECTOR({dim}) NOT NULL,
            model_name    TEXT NOT NULL,
            embedded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        )
        """,
        f"CREATE INDEX IF NOT EXISTS idx_{table}_document ON {table} (document_id)",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_model ON {table} (model_name)",
        # Cosine distance operator class, matching the `<=>` used in search().
        f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_vector
        ON {table} USING hnsw (embedding vector_cosine_ops)
        """,
    ]


def ensure_schema(
    conn,
    dim: int,
    documents_table: str = DOCUMENTS_TABLE,
    embeddings_table: str = EMBEDDINGS_TABLE,
) -> None:
    """
    Create both tables, the pgvector extension, and all indexes if missing.

    Idempotent - safe to call on every app start and at the top of every
    notebook run. If the vector extension cannot be created (the role lacks
    rights), the error surfaces here rather than at first insert.
    """
    statements = documents_ddl(documents_table) + embeddings_ddl(
        dim, embeddings_table, documents_table
    )
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)
    conn.commit()
    logger.info("Schema ready: %s, %s (dim=%s)", documents_table, embeddings_table, dim)


def drop_schema(conn, documents_table: str = DOCUMENTS_TABLE,
                embeddings_table: str = EMBEDDINGS_TABLE) -> None:
    """Tear both tables down. Used by tests and by `--reset` in the CLI."""
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {embeddings_table}")
        cur.execute(f"DROP TABLE IF EXISTS {documents_table}")
    conn.commit()


# --------------------------------------------------------------- documents --


def text_digest(text: str) -> str:
    """SHA-256 of the document body, used to detect real changes."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def upsert_documents(conn, documents: Sequence[dict], table: str = DOCUMENTS_TABLE) -> dict:
    """
    Insert or update documents, returning {"received", "written", "unchanged"}.

    The `WHERE ... IS DISTINCT FROM` clause on the update is the important
    part: a re-sync that pulls the same product back gets skipped entirely,
    so `updated_at` only moves when the text actually changed. The embedding
    job keys off that, which is what keeps repeated syncs from re-embedding
    a corpus that didn't move.
    """
    if not documents:
        return {"received": 0, "written": 0, "unchanged": 0}

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

    sql = f"""
        INSERT INTO {table} (
            id, source, product_code, product_name, office_id, wmo_id,
            headline, area_desc, severity, certainty, urgency,
            issued_at, effective_at, expires_at,
            raw_text, text_sha256, char_count, payload
        ) VALUES %s
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
        WHERE {table}.text_sha256 IS DISTINCT FROM EXCLUDED.text_sha256
        RETURNING id
    """
    template = (
        "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
        "%s::timestamptz, %s::timestamptz, %s::timestamptz, %s, %s, %s, %s::jsonb)"
    )

    with conn.cursor() as cur:
        written = execute_values(cur, sql, rows, template=template, page_size=100, fetch=True)
    conn.commit()

    written_count = len(written or [])
    return {
        "received": len(documents),
        "written": written_count,
        "unchanged": len(rows) - written_count,
    }


def documents_needing_embedding(
    conn,
    model_name: str,
    limit: int | None = None,
    documents_table: str = DOCUMENTS_TABLE,
    embeddings_table: str = EMBEDDINGS_TABLE,
) -> list[dict]:
    """
    Return documents with no current vectors for this model.

    "Current" means: at least one chunk exists, produced by this model, and
    embedded after the document was last updated. Anything else - never
    embedded, embedded by a different model, or embedded before the text
    changed - comes back for reprocessing.
    """
    sql = f"""
        SELECT d.id, d.source, d.product_code, d.office_id, d.issued_at, d.raw_text
        FROM {documents_table} d
        WHERE NOT EXISTS (
            SELECT 1 FROM {embeddings_table} e
            WHERE e.document_id = d.id
              AND e.model_name = %(model)s
              AND e.embedded_at >= d.updated_at
        )
        ORDER BY d.issued_at DESC NULLS LAST
        {"LIMIT %(limit)s" if limit else ""}
    """
    params: dict[str, Any] = {"model": model_name}
    if limit:
        params["limit"] = limit

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [c[0] for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# -------------------------------------------------------------- embeddings --


def replace_document_embeddings(
    conn,
    document_id: str,
    chunks: Sequence[str],
    vectors: Sequence[Sequence[float]],
    model_name: str,
    product_code: str | None = None,
    office_id: str | None = None,
    issued_at: Any = None,
    table: str = EMBEDDINGS_TABLE,
) -> int:
    """
    Replace all vectors for one document, atomically.

    Delete-then-insert rather than upsert-by-chunk-index: when a product is
    reissued its text changes length, so the new version may have fewer
    chunks than the old one. An upsert would leave the surplus chunks behind
    as orphaned passages that still match queries - stale forecasts surfacing
    as if current. Deleting first makes that impossible.

    Vectors are written straight into the VECTOR column via an explicit
    `%s::vector` cast on a pgvector literal, so there is no second pass to
    convert arrays into vectors.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"chunk/vector length mismatch: {len(chunks)} vs {len(vectors)}")

    from embeddings import to_pgvector

    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE document_id = %s", (document_id,))
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
                to_pgvector(vector),
                model_name,
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        sql = f"""
            INSERT INTO {table} (
                id, document_id, chunk_index, chunk_text, char_count,
                product_code, office_id, issued_at, embedding, model_name, embedded_at
            ) VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                chunk_text  = EXCLUDED.chunk_text,
                char_count  = EXCLUDED.char_count,
                embedding   = EXCLUDED.embedding,
                model_name  = EXCLUDED.model_name,
                embedded_at = EXCLUDED.embedded_at
        """
        template = (
            "(%s, %s, %s, %s, %s, %s, %s, %s::timestamptz, %s::vector, %s, now())"
        )
        execute_values(cur, sql, rows, template=template, page_size=100)

    conn.commit()
    return len(chunks)


# ------------------------------------------------------------------ search --


def search(
    conn,
    query_vector: Sequence[float],
    limit: int = 5,
    model_name: str | None = None,
    office_id: str | None = None,
    product_code: str | None = None,
    source: str | None = None,
    since_hours: int | None = None,
    min_similarity: float | None = None,
    group_by_document: bool = True,
    candidate_multiplier: int = 6,
    documents_table: str = DOCUMENTS_TABLE,
    embeddings_table: str = EMBEDDINGS_TABLE,
) -> list[dict]:
    """
    Rank chunks by cosine similarity to `query_vector`.

    Similarity is `1 - (embedding <=> query)`, i.e. cosine similarity in
    [-1, 1] where 1 is identical. The `<=>` operator is what the HNSW index
    is built on, so the ORDER BY in the inner query is index-backed.

    When `group_by_document` is on (the default), the inner query pulls a
    wider candidate set and the outer query keeps only each document's best
    chunk. Without it, one long AFD whose sections all discuss flooding will
    occupy every slot in the result list and crowd out the other offices.
    """
    from embeddings import to_pgvector

    vector_literal = to_pgvector(query_vector)
    candidates = max(limit, limit * candidate_multiplier) if group_by_document else limit

    filters = ["TRUE"]
    params: dict[str, Any] = {
        "vector": vector_literal,
        "limit": limit,
        "candidates": candidates,
    }

    if model_name:
        filters.append("e.model_name = %(model_name)s")
        params["model_name"] = model_name
    if office_id:
        filters.append("d.office_id = %(office_id)s")
        params["office_id"] = office_id.upper()
    if product_code:
        filters.append("d.product_code = %(product_code)s")
        params["product_code"] = product_code.upper()
    if source:
        filters.append("d.source = %(source)s")
        params["source"] = source
    if since_hours:
        # Clamped: `now() - make_interval(hours => n)` raises
        # DatetimeFieldOverflow once n pushes the result past Postgres'
        # timestamp range, which would turn a silly query parameter into a 500.
        # Ten years is far beyond any useful weather-text window anyway.
        filters.append("d.issued_at >= now() - make_interval(hours => %(since_hours)s)")
        params["since_hours"] = max(1, min(int(since_hours), MAX_SINCE_HOURS))

    where = " AND ".join(filters)

    having = ""
    if min_similarity is not None:
        having = "WHERE similarity >= %(min_similarity)s"
        params["min_similarity"] = float(min_similarity)

    if group_by_document:
        sql = f"""
            WITH candidates AS (
                SELECT
                    e.id            AS chunk_id,
                    e.document_id,
                    e.chunk_index,
                    e.chunk_text,
                    e.model_name,
                    d.source,
                    d.product_code,
                    d.product_name,
                    d.office_id,
                    d.headline,
                    d.area_desc,
                    d.severity,
                    d.issued_at,
                    d.expires_at,
                    1 - (e.embedding <=> %(vector)s::vector) AS similarity
                FROM {embeddings_table} e
                JOIN {documents_table} d ON d.id = e.document_id
                WHERE {where}
                ORDER BY e.embedding <=> %(vector)s::vector
                LIMIT %(candidates)s
            ),
            best_per_document AS (
                SELECT DISTINCT ON (document_id) *
                FROM candidates
                ORDER BY document_id, similarity DESC
            )
            SELECT * FROM best_per_document
            {having}
            ORDER BY similarity DESC
            LIMIT %(limit)s
        """
    else:
        sql = f"""
            WITH candidates AS (
                SELECT
                    e.id            AS chunk_id,
                    e.document_id,
                    e.chunk_index,
                    e.chunk_text,
                    e.model_name,
                    d.source,
                    d.product_code,
                    d.product_name,
                    d.office_id,
                    d.headline,
                    d.area_desc,
                    d.severity,
                    d.issued_at,
                    d.expires_at,
                    1 - (e.embedding <=> %(vector)s::vector) AS similarity
                FROM {embeddings_table} e
                JOIN {documents_table} d ON d.id = e.document_id
                WHERE {where}
                ORDER BY e.embedding <=> %(vector)s::vector
                LIMIT %(limit)s
            )
            SELECT * FROM candidates
            {having}
            ORDER BY similarity DESC
        """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [c[0] for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def stats(
    conn,
    documents_table: str = DOCUMENTS_TABLE,
    embeddings_table: str = EMBEDDINGS_TABLE,
) -> dict:
    """Corpus counts, for the health panel in the UI and for smoke tests."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                (SELECT count(*) FROM {documents_table})                    AS documents,
                (SELECT count(*) FROM {embeddings_table})                   AS chunks,
                (SELECT count(DISTINCT document_id) FROM {embeddings_table}) AS embedded_documents,
                (SELECT count(DISTINCT office_id) FROM {documents_table})   AS offices,
                (SELECT max(issued_at) FROM {documents_table})              AS latest_issued_at,
                (SELECT max(embedded_at) FROM {embeddings_table})           AS latest_embedded_at
            """
        )
        columns = [c[0] for c in cur.description]
        return dict(zip(columns, cur.fetchone()))
