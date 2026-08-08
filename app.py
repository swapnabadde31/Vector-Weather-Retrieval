"""
Weather RAG Databricks App.

Flask API over Lakebase (Postgres + pgvector) that:
  - harvests unstructured NWS text products and alerts   POST /weather/sync
  - chunks and vectorizes them into pgvector             POST /weather/embed
  - answers semantic queries over the corpus             POST /weather/search

The headline endpoint is the last one:

    POST /weather/search {"query": "flash flood risk this weekend"}

Run locally:
    LAKEBASE_URL=postgresql://... python app.py

Deploy on Databricks: point an App at this folder; app.yaml supplies the
command and the secret-scope environment variables.
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request

import embeddings
import lakebase
import weather_store
from weather_client import (
    DEFAULT_OFFICES,
    DEFAULT_PRODUCT_TYPES,
    NWSClient,
    fetch_cpc_outlooks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-rag")

app = Flask(__name__)

EMBEDDING_MODEL = embeddings.DEFAULT_MODEL
EMBEDDING_DIM = embeddings.resolve_dimension(EMBEDDING_MODEL)
CHUNK_SIZE = embeddings.DEFAULT_CHUNK_SIZE
CHUNK_OVERLAP = embeddings.DEFAULT_CHUNK_OVERLAP

# Embedding inside a web request is convenient for small syncs but is not the
# scalable path - the notebook is. This caps how much work a single HTTP
# request will take on before it starts refusing, so a stray call can't hang
# the app for ten minutes.
MAX_INLINE_EMBED_DOCS = int(os.environ.get("MAX_INLINE_EMBED_DOCS", 200))

_schema_ready = False


def ensure_schema_once() -> None:
    """Create tables on first use, then get out of the way."""
    global _schema_ready
    if _schema_ready:
        return
    with lakebase.get_connection() as conn:
        weather_store.ensure_schema(conn, EMBEDDING_DIM)
    _schema_ready = True


@app.errorhandler(Exception)
def handle_exception(err):
    """Always return JSON, so a client's resp.json() never hits an HTML page."""
    logger.exception("Unhandled exception while processing request")
    status = getattr(err, "code", 500)
    if not isinstance(status, int):
        status = 500
    return jsonify({"error": str(err), "type": type(err).__name__}), status


# ------------------------------------------------------------------ pages --


@app.route("/")
def index():
    """Search UI over the ingested weather corpus."""
    return render_template(
        "index.html",
        model=EMBEDDING_MODEL,
        dim=EMBEDDING_DIM,
        offices=DEFAULT_OFFICES,
        product_types=DEFAULT_PRODUCT_TYPES + ["ALERT"],
    )


@app.route("/healthz")
def healthz():
    """Liveness plus a real database round trip."""
    try:
        health = lakebase.healthcheck()
    except Exception as exc:  # noqa: BLE001 - health endpoints report, not raise
        return jsonify({"status": "degraded", "error": str(exc)}), 503
    return jsonify({"status": "ok", "model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM, **health})


@app.route("/weather/stats")
def weather_stats():
    """Corpus counts: documents, chunks, coverage, freshness."""
    ensure_schema_once()
    with lakebase.get_connection() as conn:
        payload = weather_store.stats(conn)
    payload["model"] = EMBEDDING_MODEL
    payload["chunk_size"] = CHUNK_SIZE
    payload["chunk_overlap"] = CHUNK_OVERLAP
    return jsonify(payload)


@app.route("/weather/documents")
def list_documents():
    """Browse synced documents without their vectors."""
    ensure_schema_once()
    limit = min(int(request.args.get("limit", 50)), 500)
    office = request.args.get("office")
    product_code = request.args.get("product_code")

    clauses = ["TRUE"]
    params: dict = {"limit": limit}
    if office:
        clauses.append("office_id = %(office)s")
        params["office"] = office.upper()
    if product_code:
        clauses.append("product_code = %(product_code)s")
        params["product_code"] = product_code.upper()

    rows = lakebase.run_query(
        f"""
        SELECT id, source, product_code, product_name, office_id, headline,
               area_desc, severity, issued_at, char_count, synced_at
        FROM {weather_store.DOCUMENTS_TABLE}
        WHERE {" AND ".join(clauses)}
        ORDER BY issued_at DESC NULLS LAST
        LIMIT %(limit)s
        """,
        params,
    )
    return jsonify({"count": len(rows), "documents": rows})


# ------------------------------------------------------------------- sync --


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """
    Harvest unstructured weather text from NWS and upsert it into Lakebase.

    Body (all optional):
        {
          "offices":         ["OKX", "FWD"],     # WFO ids
          "product_types":   ["AFD", "HWO"],     # NWS text product codes
          "limit_per_pair":  2,                  # products per type per office
          "include_alerts":  true,
          "alert_area":      "TX",               # state/marine code, or list
          "alert_limit":     50,
          "include_cpc":     false,              # CPC extended-range outlooks
          "embed":           false               # also vectorize, inline
        }

    Returns counts rather than the documents themselves - a full sync is a
    few hundred KB of prose and nobody wants that echoed back.
    """
    ensure_schema_once()
    body = request.get_json(silent=True) or {}

    offices = [o.strip().upper() for o in body.get("offices", DEFAULT_OFFICES) if o.strip()]
    product_types = [
        p.strip().upper() for p in body.get("product_types", DEFAULT_PRODUCT_TYPES) if p.strip()
    ]
    limit_per_pair = max(1, min(int(body.get("limit_per_pair", 2)), 10))
    include_alerts = bool(body.get("include_alerts", True))
    include_cpc = bool(body.get("include_cpc", False))

    client = NWSClient()
    documents = list(
        client.iter_text_products(
            product_types=product_types,
            offices=offices,
            limit_per_pair=limit_per_pair,
        )
    )
    product_count = len(documents)

    alert_count = 0
    if include_alerts:
        alerts = client.get_active_alerts(
            area=body.get("alert_area"),
            limit=int(body.get("alert_limit", 50)),
        )
        alert_count = len(alerts)
        documents.extend(alerts)

    cpc_count = 0
    if include_cpc:
        cpc_docs = fetch_cpc_outlooks()
        cpc_count = len(cpc_docs)
        documents.extend(cpc_docs)

    with lakebase.get_connection() as conn:
        result = weather_store.upsert_documents(conn, documents)

    response = {
        "fetched": {
            "text_products": product_count,
            "alerts": alert_count,
            "cpc_outlooks": cpc_count,
        },
        "upserted": result,
        "offices": offices,
        "product_types": product_types,
    }

    if body.get("embed"):
        response["embedded"] = _embed_pending(limit=MAX_INLINE_EMBED_DOCS)

    return jsonify(response)


@app.route("/weather/embed", methods=["POST"])
def weather_embed():
    """
    Vectorize any documents that don't yet have current embeddings.

    Body (optional): {"limit": 50}

    This exists so the pipeline can be driven end-to-end from the API alone,
    which is what makes the whole thing runnable on Databricks Free Edition
    without a job cluster. For a real corpus, schedule the notebook instead -
    a web worker is the wrong place to hold a model and grind through
    thousands of chunks.
    """
    ensure_schema_once()
    body = request.get_json(silent=True) or {}
    limit = min(int(body.get("limit", MAX_INLINE_EMBED_DOCS)), MAX_INLINE_EMBED_DOCS)
    return jsonify(_embed_pending(limit=limit))


def _embed_pending(limit: int) -> dict:
    """Chunk and embed pending documents; shared by /sync?embed and /embed."""
    with lakebase.get_connection() as conn:
        pending = weather_store.documents_needing_embedding(
            conn, model_name=EMBEDDING_MODEL, limit=limit
        )
        if not pending:
            return {"documents": 0, "chunks": 0, "model": EMBEDDING_MODEL}

        model = embeddings.load_model(EMBEDDING_MODEL)
        total_chunks = 0
        processed = 0

        for doc in pending:
            chunks = embeddings.chunk_weather_text(
                doc["raw_text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
            )
            if not chunks:
                continue
            vectors = embeddings.embed_texts(chunks, model=model)
            total_chunks += weather_store.replace_document_embeddings(
                conn,
                document_id=doc["id"],
                chunks=chunks,
                vectors=vectors,
                model_name=EMBEDDING_MODEL,
                product_code=doc.get("product_code"),
                office_id=doc.get("office_id"),
                issued_at=doc.get("issued_at"),
            )
            processed += 1

    return {"documents": processed, "chunks": total_chunks, "model": EMBEDDING_MODEL}


# ----------------------------------------------------------------- search --


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """
    Semantic search over the ingested weather corpus.

        POST /weather/search
        {"query": "flash flood risk this weekend"}

    Optional body fields:
        limit             (default 5, max 50)
        office            filter to one WFO, e.g. "OKX"
        product_code      "AFD" | "HWO" | "ESF" | "ALERT"
        source            "nws_product" | "nws_alert" | "cpc_outlook"
        since_hours       only documents issued in the last N hours
        min_similarity    drop results below this cosine similarity
        group_by_document true (default) returns each document's best chunk
    """
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Send a JSON body with Content-Type: application/json"}), 400

    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Field 'query' is required and cannot be empty"}), 400
    if len(query) > 2000:
        return jsonify({"error": "Field 'query' is limited to 2000 characters"}), 400

    # Coerce the numeric knobs here so malformed input is a 400 with a clear
    # message, not a 500 from deep inside psycopg2.
    try:
        limit = max(1, min(int(body.get("limit", 5)), 50))
        since_hours = body.get("since_hours")
        since_hours = int(since_hours) if since_hours is not None else None
        min_similarity = body.get("min_similarity")
        min_similarity = float(min_similarity) if min_similarity is not None else None
    except (TypeError, ValueError):
        return jsonify(
            {"error": "limit and since_hours must be integers; min_similarity must be a number"}
        ), 400

    if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
        return jsonify({"error": "min_similarity must be between -1 and 1"}), 400

    ensure_schema_once()

    try:
        query_vector = embeddings.embed_query(query, EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to embed query")
        return jsonify({"error": f"Could not embed the query: {exc}"}), 500

    with lakebase.get_connection() as conn:
        rows = weather_store.search(
            conn,
            query_vector=query_vector,
            limit=limit,
            model_name=EMBEDDING_MODEL,
            office_id=body.get("office"),
            product_code=body.get("product_code"),
            source=body.get("source"),
            since_hours=since_hours,
            min_similarity=min_similarity,
            group_by_document=bool(body.get("group_by_document", True)),
        )

    results = [
        {
            "document_id": row["document_id"],
            "chunk_id": row["chunk_id"],
            "chunk_index": row["chunk_index"],
            "similarity": round(float(row["similarity"]), 4),
            "text": row["chunk_text"],
            "source": row["source"],
            "product_code": row["product_code"],
            "product_name": row["product_name"],
            "office_id": row["office_id"],
            "headline": row["headline"],
            "area_desc": row["area_desc"],
            "severity": row["severity"],
            "issued_at": row["issued_at"].isoformat() if row.get("issued_at") else None,
            "expires_at": row["expires_at"].isoformat() if row.get("expires_at") else None,
        }
        for row in rows
    ]

    return jsonify(
        {
            "query": query,
            "model": EMBEDDING_MODEL,
            "count": len(results),
            "results": results,
        }
    )


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", host=host, port=port)
