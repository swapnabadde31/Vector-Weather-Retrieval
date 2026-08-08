#!/usr/bin/env python3
"""
Run the whole weather RAG pipeline from a terminal.

Exists so the pipeline can be exercised without Databricks at all - useful
when developing locally, and useful as an acceptance test after deploying,
because it drives the same modules the app and the notebook drive.

    export LAKEBASE_URL=postgresql://role:pw@host:5432/databricks_postgres?sslmode=require

    python scripts/run_pipeline.py sync --offices OKX,FWD --limit-per-pair 1
    python scripts/run_pipeline.py embed
    python scripts/run_pipeline.py search "flash flood risk this weekend"
    python scripts/run_pipeline.py all --query "damaging wind and large hail"

`all` runs sync -> embed -> search in one pass, which is the fastest way to
prove an end-to-end deployment works.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embeddings  # noqa: E402
import lakebase  # noqa: E402
import weather_store  # noqa: E402
from weather_client import (  # noqa: E402
    DEFAULT_OFFICES,
    DEFAULT_PRODUCT_TYPES,
    NWSClient,
    fetch_cpc_outlooks,
)


def _csv(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    return [v.strip().upper() for v in value.split(",") if v.strip()]


def cmd_sync(args) -> None:
    offices = _csv(args.offices, DEFAULT_OFFICES)
    product_types = _csv(args.product_types, DEFAULT_PRODUCT_TYPES)

    client = NWSClient(max_requests_per_minute=args.rpm)
    print(f"Fetching {product_types} from {len(offices)} offices...")

    documents = list(
        client.iter_text_products(
            product_types=product_types,
            offices=offices,
            limit_per_pair=args.limit_per_pair,
        )
    )
    print(f"  {len(documents)} text products")

    if not args.no_alerts:
        alerts = client.get_active_alerts(area=args.alert_area, limit=args.alert_limit)
        print(f"  {len(alerts)} active alerts")
        documents.extend(alerts)

    if args.include_cpc:
        cpc = fetch_cpc_outlooks()
        print(f"  {len(cpc)} CPC outlooks")
        documents.extend(cpc)

    with lakebase.get_connection() as conn:
        weather_store.ensure_schema(conn, embeddings.resolve_dimension())
        result = weather_store.upsert_documents(conn, documents)

    print(
        f"Upserted: {result['written']} written, {result['unchanged']} unchanged "
        f"({result['received']} received)"
    )


def cmd_embed(args) -> None:
    model_name = args.model or embeddings.DEFAULT_MODEL
    dim = embeddings.resolve_dimension(model_name)

    with lakebase.get_connection() as conn:
        weather_store.ensure_schema(conn, dim)

        if args.reset:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {weather_store.EMBEDDINGS_TABLE}")
            conn.commit()
            print("Cleared existing vectors")

        pending = weather_store.documents_needing_embedding(
            conn, model_name=model_name, limit=getattr(args, "embed_limit", None)
        )
        print(f"{len(pending)} documents need embedding with {model_name}")
        if not pending:
            return

        model = embeddings.load_model(model_name)
        total = 0
        for i, doc in enumerate(pending, start=1):
            chunks = embeddings.chunk_weather_text(
                doc["raw_text"], chunk_size=args.chunk_size, overlap=args.chunk_overlap
            )
            if not chunks:
                continue
            vectors = embeddings.embed_texts(chunks, model=model)
            total += weather_store.replace_document_embeddings(
                conn,
                document_id=doc["id"],
                chunks=chunks,
                vectors=vectors,
                model_name=model_name,
                product_code=doc.get("product_code"),
                office_id=doc.get("office_id"),
                issued_at=doc.get("issued_at"),
            )
            if i % 10 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)} documents -> {total} chunks")

        print(f"Embedded {len(pending)} documents into {total} chunks")


def cmd_search(args) -> None:
    model_name = args.model or embeddings.DEFAULT_MODEL
    vector = embeddings.embed_query(args.query, model_name)

    with lakebase.get_connection() as conn:
        rows = weather_store.search(
            conn,
            query_vector=vector,
            limit=args.limit,
            model_name=model_name,
            office_id=args.office,
            product_code=args.product_code,
            since_hours=args.since_hours,
        )

    if args.json:
        print(
            json.dumps(
                [
                    {
                        **row,
                        "issued_at": row["issued_at"].isoformat() if row.get("issued_at") else None,
                        "expires_at": (
                            row["expires_at"].isoformat() if row.get("expires_at") else None
                        ),
                    }
                    for row in rows
                ],
                indent=2,
            )
        )
        return

    print(f"\nQuery: {args.query!r}  ({len(rows)} results)\n")
    for row in rows:
        header = f"{row['similarity']:.4f}  {row['product_code']}"
        if row.get("office_id"):
            header += f" / {row['office_id']}"
        if row.get("issued_at"):
            header += f"  issued {row['issued_at']:%Y-%m-%d %H:%MZ}"
        print(header)
        if row.get("headline"):
            print(f"        {row['headline'][:110]}")
        preview = " ".join(row["chunk_text"].split())
        print(f"        {preview[:320]}...\n")


def cmd_stats(args) -> None:
    with lakebase.get_connection() as conn:
        payload = weather_store.stats(conn)
    for key, value in payload.items():
        print(f"{key:22}: {value}")


def cmd_all(args) -> None:
    """sync -> embed -> search, sharing one option namespace."""
    cmd_sync(args)
    cmd_embed(args)
    args.query = args.query or "flash flood risk this weekend"
    cmd_search(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_sync_args(p):
        p.add_argument("--offices", help="Comma-separated WFO ids, e.g. OKX,FWD")
        p.add_argument("--product-types", help="Comma-separated NWS codes, e.g. AFD,HWO")
        p.add_argument("--limit-per-pair", type=int, default=2)
        p.add_argument("--no-alerts", action="store_true")
        p.add_argument("--alert-area", help="State/marine code, e.g. TX")
        p.add_argument("--alert-limit", type=int, default=50)
        p.add_argument("--include-cpc", action="store_true", help="Also pull CPC outlooks")
        p.add_argument("--rpm", type=int, default=60, help="Max NWS requests per minute")

    def add_embed_args(p):
        p.add_argument("--model", help="sentence-transformers model id")
        # dest is explicit so `all` can also expose --limit for result count
        # without the two colliding.
        p.add_argument(
            "--embed-limit", dest="embed_limit", type=int,
            help="Max documents to embed in this run (default: all pending)",
        )
        p.add_argument("--chunk-size", type=int, default=embeddings.DEFAULT_CHUNK_SIZE)
        p.add_argument("--chunk-overlap", type=int, default=embeddings.DEFAULT_CHUNK_OVERLAP)
        p.add_argument("--reset", action="store_true", help="Delete all vectors first")

    def add_search_args(p, positional=True, with_model=True):
        if positional:
            p.add_argument("query")
        p.add_argument("--limit", type=int, default=5, help="Results to return")
        p.add_argument("--office")
        p.add_argument("--product-code")
        p.add_argument("--since-hours", type=int)
        p.add_argument("--json", action="store_true")
        if with_model:
            p.add_argument("--model", help="sentence-transformers model id")

    p_sync = sub.add_parser("sync", help="Harvest NWS text into weather_documents")
    add_sync_args(p_sync)
    p_sync.set_defaults(func=cmd_sync)

    p_embed = sub.add_parser("embed", help="Chunk and vectorize pending documents")
    add_embed_args(p_embed)
    p_embed.set_defaults(func=cmd_embed)

    p_search = sub.add_parser("search", help="Semantic search over the corpus")
    add_search_args(p_search)
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="Corpus counts")
    p_stats.set_defaults(func=cmd_stats)

    # `all` shares one option namespace across the three steps. --model is
    # registered once by add_embed_args; --embed-limit caps documents to
    # embed while --limit controls how many results the final search returns.
    p_all = sub.add_parser("all", help="sync -> embed -> search in one pass")
    add_sync_args(p_all)
    add_embed_args(p_all)
    add_search_args(p_all, positional=False, with_model=False)
    p_all.add_argument("--query", default="flash flood risk this weekend")
    p_all.set_defaults(func=cmd_all)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
