import os
#!/usr/bin/env python3
API_BASE = os.getenv("API_BASE", "https://v3.football.api-sports.io")
"""
WinMatic backend — cleaned and patched:
- Serve /static correctly
- Add /team-logo/{team_id}.png dynamic proxy + disk cache
- Create /static/team-logo/default.png on startup to avoid 404 spam
- Keep all prediction endpoints intact
"""



def _sql_pg_fix(q: str) -> str:
    """
    Convert SQLite-style placeholders/functions to Postgres-safe SQL.
    - '?'  -> '%s'
    - CURRENT_TIMESTAMP -> now()
    """
    try:
        import os
        db = os.getenv("DATABASE_URL", "") or ""
        if db.startswith("postgres"):
            return q.replace("CURRENT_TIMESTAMP", "now()").replace("?", "%s")
    except Exception:
        pass
    return q
import os

API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY', '')

import io
import json
import math
import time
import threading
import base64
import logging
import sqlite3

import os

# === History DB backend (SQLite file OR Neon Postgres) ===
# If DATABASE_URL starts with postgresql://, we use Neon.
# We keep your existing SQL mostly unchanged by translating '?' -> '%s' for psycopg.

def _is_pg_url(url: str) -> bool:
    return bool(url) and url.startswith("postgresql://")

class PGCursor:
    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=None):
        sql = sql.replace("?", "%s")
        return self._cur.execute(sql, params or ())

    def executemany(self, sql, seq_of_params):
        sql = sql.replace("?", "%s")
        return self._cur.executemany(sql, seq_of_params)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def close(self):
        return self._cur.close()

    @property
    def rowcount(self):
        return getattr(self._cur, "rowcount", -1)

class PGConnection:
    def __init__(self, conn):
        self._conn = conn
        self.is_pg = True

    def cursor(self):
        return PGCursor(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def close(self):
        return self._conn.close()

def db_connect():
    """Return a DB connection (Postgres if DATABASE_URL is set, else SQLite)."""
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if url:
        import psycopg
        return psycopg.connect(url)
    return sqlite3.connect(DB_PATH)

