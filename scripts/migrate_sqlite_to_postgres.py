#!/usr/bin/env python3
import argparse
import os
import sqlite3
from typing import Any, Dict, List, Tuple

import psycopg2
from psycopg2.extras import execute_values


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions_history (
  id BIGINT PRIMARY KEY,
  created_utc TEXT,
  league INTEGER,
  fixture_id BIGINT,
  kickoff_utc TEXT,
  home_team TEXT,
  away_team TEXT,
  xg_home DOUBLE PRECISION,
  xg_away DOUBLE PRECISION,
  raw_home DOUBLE PRECISION,
  raw_draw DOUBLE PRECISION,
  raw_away DOUBLE PRECISION,
  cal_home DOUBLE PRECISION,
  cal_draw DOUBLE PRECISION,
  cal_away DOUBLE PRECISION,
  odds_home DOUBLE PRECISION,
  odds_draw DOUBLE PRECISION,
  odds_away DOUBLE PRECISION,
  ev_home DOUBLE PRECISION,
  ev_draw DOUBLE PRECISION,
  ev_away DOUBLE PRECISION,
  mode TEXT,
  meta_json TEXT
);
"""

# Helpful indexes (optional, safe)
CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_predictions_history_league ON predictions_history(league);
CREATE INDEX IF NOT EXISTS idx_predictions_history_fixture ON predictions_history(fixture_id);
CREATE INDEX IF NOT EXISTS idx_predictions_history_kickoff ON predictions_history(kickoff_utc);
"""


def guess_columns(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return [r[1] for r in rows]  # name column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True, help="Path to SQLite DB (e.g. data/predictions_history.db)")
    ap.add_argument("--pg", required=True, help="Postgres connection URL (Render EXTERNAL URL)")
    args = ap.parse_args()

    sqlite_path = args.sqlite
    pg_url = args.pg

    if not os.path.exists(sqlite_path):
        raise SystemExit(f"SQLite file not found: {sqlite_path}")

    # 1) Read all rows from SQLite
    sconn = sqlite3.connect(sqlite_path)
    sconn.row_factory = sqlite3.Row
    scur = sconn.cursor()

    # Your existing table name (from your backend)
    table = "predictions_history"

    # Verify table exists
    scur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if scur.fetchone() is None:
        raise SystemExit(f"SQLite table '{table}' not found in {sqlite_path}")

    cols = guess_columns(scur, table)

    scur.execute(f"SELECT {', '.join(cols)} FROM {table}")
    rows = scur.fetchall()
    print(f"[sqlite] rows={len(rows)} cols={len(cols)}")

    # 2) Connect Postgres + create table
    pconn = psycopg2.connect(pg_url)
    pconn.autocommit = False
    pcur = pconn.cursor()
    pcur.execute(CREATE_TABLE_SQL)
    pcur.execute(CREATE_INDEXES_SQL)
    pconn.commit()

    # 3) Insert rows in chunks
    # Map SQLite columns -> Postgres columns (only keep those that exist in CREATE_TABLE_SQL)
    # If your SQLite has extra cols, they’ll be ignored.
    pg_cols = [
        "id", "created_utc", "league", "fixture_id", "kickoff_utc",
        "home_team", "away_team",
        "xg_home", "xg_away",
        "raw_home", "raw_draw", "raw_away",
        "cal_home", "cal_draw", "cal_away",
        "odds_home", "odds_draw", "odds_away",
        "ev_home", "ev_draw", "ev_away",
        "mode", "meta_json",
    ]

    # Build values list from sqlite rows (missing -> None)
    def getv(r: sqlite3.Row, k: str):
        return r[k] if k in r.keys() else None

    values: List[Tuple[Any, ...]] = []
    for r in rows:
        values.append(tuple(getv(r, c) for c in pg_cols))

    # Upsert by primary key id
    insert_sql = f"""
      INSERT INTO predictions_history ({', '.join(pg_cols)})
      VALUES %s
      ON CONFLICT (id) DO NOTHING
    """

    chunk = 1000
    inserted = 0
    for i in range(0, len(values), chunk):
        batch = values[i : i + chunk]
        execute_values(pcur, insert_sql, batch, page_size=chunk)
        pconn.commit()
        inserted += len(batch)
        print(f"[pg] upserted {inserted}/{len(values)}")

    # 4) Fix sequence if needed (not critical if you always provide id yourself)
    # Try to set sequence to max(id)+1 if a sequence exists (safe to ignore errors)
    try:
        pcur.execute("SELECT MAX(id) FROM predictions_history")
        mx = pcur.fetchone()[0] or 0
        # Create a sequence if you ever need it (optional)
        pcur.execute("CREATE SEQUENCE IF NOT EXISTS predictions_history_id_seq")
        pcur.execute("SELECT setval('predictions_history_id_seq', %s, true)", (mx,))
        pconn.commit()
        print(f"[pg] sequence set to {mx}")
    except Exception as e:
        pconn.rollback()
        print(f"[pg] sequence step skipped: {e}")

    pcur.close()
    pconn.close()
    scur.close()
    sconn.close()

    print("[ok] migration complete")


if __name__ == "__main__":
    main()

