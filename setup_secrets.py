"""
One-time setup script: stores the Lakebase connection URL as a Databricks
secret.

That's the entire list. Unlike the day-2 Massive pipeline, this app has no
second secret to create - api.weather.gov authenticates with a descriptive
User-Agent header only, so there is no weather API key to store, rotate, or
accidentally commit.

Run this from a Databricks notebook or terminal with the Databricks CLI
configured:

    python setup_secrets.py

It is safe to re-run: `put_secret` overwrites the existing value.
"""

import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

SCOPE = "database"
KEY = "lakebase-url"

try:
    w.secrets.create_scope(scope=SCOPE)
    print(f"Created secret scope '{SCOPE}'")
except Exception as exc:  # noqa: BLE001 - scope-exists is not worth a special case
    print(f"Secret scope '{SCOPE}' already exists ({exc.__class__.__name__}) - continuing")

url = getpass.getpass(
    "Paste your Lakebase connection URL "
    "(postgresql://role:password@host:5432/databricks_postgres?sslmode=require): "
)
if not url.strip().startswith("postgres"):
    raise SystemExit("That doesn't look like a Postgres URL - aborting without writing anything.")

w.secrets.put_secret(scope=SCOPE, key=KEY, string_value=url.strip())
print(f"Stored secret {SCOPE}/{KEY}")

w.secrets.put_acl(scope=SCOPE, principal="users", permission=workspace.AclPermission.READ)
print(f"Granted read access on scope '{SCOPE}' to all workspace users")

print(
    "\nDone. The Flask app and the ingestion notebook both read this secret "
    "automatically via lakebase.py / weather_store.py - no further config needed.\n"
    "For local development, skip this script entirely and set LAKEBASE_URL in "
    "your .env instead (see .env.example)."
)
