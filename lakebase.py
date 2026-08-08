"""
Lakebase (Databricks-managed Postgres) connection helper.

Resolves a single Postgres URL of the form

    postgresql://role:password@host:5432/databricks_postgres?sslmode=require

from one of two places, in order:

  1. The LAKEBASE_URL environment variable - used for local development and
     for the ingestion notebook when it runs outside a Databricks App.
  2. A Databricks secret (scope `database`, key `lakebase-url`), which is how
     the deployed app gets it. `w.secrets.get_secret()` returns a base64 blob,
     so the value is decoded on the way out.

The env-var path is checked first so that `LAKEBASE_URL=... python app.py`
works on a laptop with no Databricks CLI profile configured. The reference
day-2 app read only the secret, which meant its own `.env.example` could not
actually be used locally.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from contextlib import contextmanager
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

_cached_url: str | None = None


def _decode_secret(value: str) -> str:
    """
    Decode a Databricks secret value.

    The SDK returns base64. Some code paths (dbutils inside a notebook) hand
    back the plaintext already, so fall back to the raw string rather than
    corrupting a perfectly good URL.
    """
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return value
    return decoded if decoded.startswith("postgres") else value


def lakebase_url(refresh: bool = False) -> str:
    """Return the Postgres connection URL, caching it after the first lookup."""
    global _cached_url
    if _cached_url and not refresh:
        return _cached_url

    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        _cached_url = env_url
        return _cached_url

    from databricks.sdk import WorkspaceClient

    secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
    _cached_url = _decode_secret(secret.value)
    return _cached_url


def connection_params() -> dict:
    """
    Break the URL into psycopg2 keyword arguments.

    The ingestion notebook connects this way rather than passing the DSN
    string, because it needs the individual pieces for logging and because a
    password containing URL-reserved characters survives the round trip
    better as an explicit kwarg.
    """
    parsed = urlparse(lakebase_url())
    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": (parsed.path or "").lstrip("/") or "databricks_postgres",
        "user": parsed.username,
        "password": parsed.password,
        "sslmode": "require",
    }


@contextmanager
def get_connection(dict_cursor: bool = False, connect_timeout: int = 15):
    """
    Yield a psycopg2 connection, closed on exit.

    `dict_cursor=True` gives RealDictCursor rows for code that wants dicts;
    weather_store builds its own dicts from cursor.description so it works
    either way.
    """
    params = connection_params()
    params["connect_timeout"] = connect_timeout
    if dict_cursor:
        params["cursor_factory"] = RealDictCursor

    conn = psycopg2.connect(**params)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """SQLAlchemy engine, for pandas `read_sql` in notebooks and analysis."""
    from sqlalchemy import create_engine

    return create_engine(lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query and return rows as a list of dicts."""
    with get_connection(dict_cursor=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run a write statement and return the affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def healthcheck() -> dict:
    """Confirm the database answers and report whether pgvector is installed."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            has_vector = cur.fetchone()[0] > 0
    return {"postgres": version.split(",")[0], "pgvector": has_vector}
