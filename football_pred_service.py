#!/usr/bin/env python3
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
    - datetime('now') -> now()
    """
    try:
        import os
        db = os.getenv("DATABASE_URL", "") or ""
        if db.startswith("postgres"):
            return q.replace("datetime('now')", "now()").replace("?", "%s")
    except Exception:
        pass
    return q
import os
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
    url = os.environ.get("DATABASE_URL", "")
    if _is_pg_url(url):
        import psycopg
        return PGConnection(psycopg.connect(url))
    # fallback: sqlite
    return sqlite3.connect(DB_PATH)

def get_table_columns(conn, cur, table: str) -> list[str]:
    """
    Return column names for `table` for both Postgres (Neon) and SQLite.
    """
    if getattr(conn, "is_pg", False):
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [r[0] for r in cur.fetchall()]

    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
from pathlib import Path

import math
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import requests
from joblib import dump, load
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends, Header, Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, Response
from fastapi.responses import HTMLResponse


from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.calibration import calibration_curve
from functools import lru_cache


# ==========================================
# 🧩 Ensure predictions_history table exists
# ==========================================
import os, sqlite3

def _norm_result_label(value):
    """
    Map different encodings of match outcome to a common form:
    'home', 'draw', 'away'.
    """
    if value is None:
        return None

    s = str(value).strip().lower()

    if s in ("h", "home", "1"):
        return "home"
    if s in ("a", "away", "2"):
        return "away"
    if s in ("d", "draw", "x"):
        return "draw"

    # Fallback: return as-is
    return s


def ensure_predictions_db() -> None:
    """
    Make sure the predictions_history table exists and has all columns
    used by the API (fixture_id, league, teams, probs, result, etc.).
    Safe to call many times.
    """
    try:
        # Make sure the folder for the DB exists
        db_dir = os.path.dirname(DB_PATH) or "."
        os.makedirs(db_dir, exist_ok=True)

        conn = db_connect()
        cur = conn.cursor()

        # 1) Create the table if it doesn't exist at all.
        #    We start with an id, then we'll add/ensure all other columns.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            );
            """
        )

        # 2) See what columns we currently have
        cur.execute("PRAGMA table_info(predictions_history);")
        existing_cols = {row[1] for row in cur.fetchall()}

        # 3) Columns we want to be sure exist
        expected_cols = {
            "fixture_id": "INTEGER",
            "league": "INTEGER",
            "home_team": "TEXT",
            "away_team": "TEXT",
            "kickoff_utc": "TEXT",
            "model_home_p": "REAL",
            "model_draw_p": "REAL",
            "model_away_p": "REAL",
            "predicted_side": "TEXT",
            "edge_value": "REAL",
            "actual_result": "TEXT",
            # optional JSON payload field for backwards compatibility
            "payload": "TEXT",

            # --- market implied probs / odds (optional) ---
            "market_home_p": "REAL",
            "market_draw_p": "REAL",
            "market_away_p": "REAL",
            "market_home_odds": "REAL",
            "market_draw_odds": "REAL",
            "market_away_odds": "REAL",
            "market_bookmaker": "TEXT",
            "market_bet_name": "TEXT",
        }

        # 4) Add any missing columns
        for name, coltype in expected_cols.items():
            if name not in existing_cols:
                cur.execute(
                    f"ALTER TABLE predictions_history "
                    f"ADD COLUMN {name} {coltype}"
                )

        conn.commit()
    except Exception as e:
        # Don't crash the app, just log a warning
        try:
            logger.warning("[DB] ensure_predictions_db failed: %s", e)
        except Exception:
            # logger might not exist yet at import time in some setups
            print("[DB] ensure_predictions_db failed:", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass



# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

if not API_FOOTBALL_KEY:
    raise RuntimeError("API_FOOTBALL_KEY environment variable is not set")

API_BASE = "https://v3.football.api-sports.io"

# Odds endpoints often have limited historical availability.
# To avoid wasting API quota, we only attempt odds fetches within this window.
ODDS_LOOKBACK_DAYS = int(os.getenv("ODDS_LOOKBACK_DAYS", "21"))   # past days
ODDS_FUTURE_DAYS = int(os.getenv("ODDS_FUTURE_DAYS", "10"))       # upcoming days
ODDS_MAX_CALLS_PER_RUN = int(os.getenv("ODDS_MAX_CALLS_PER_RUN", "60"))

def _within_odds_window(kickoff_utc: Any) -> bool:
    """Return True if kickoff is within a (now - lookback) .. (now + future) window."""
    try:
        if not kickoff_utc:
            return False
        if isinstance(kickoff_utc, str):
            dt = datetime.datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
        elif isinstance(kickoff_utc, datetime.datetime):
            dt = kickoff_utc
        else:
            return False
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - datetime.timedelta(days=ODDS_LOOKBACK_DAYS)) <= dt <= (now + datetime.timedelta(days=ODDS_FUTURE_DAYS))
    except Exception:
        return False


ART = "artifacts"
os.makedirs(ART, exist_ok=True)

SNAPSHOT_DIR = os.path.join(ART, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

API_CACHE_FILE = os.path.join(ART, "api_cache.json")
API_DISK_CACHE_FILE = os.path.join(ART, "api_disk_cache.json")
CACHE_ONLY_MODE = os.getenv("WINMATIC_CACHE_ONLY", "0") == "1"
API_QUOTA_EXHAUSTED = False  # becomes True after daily limit is hit
API_QUOTA_EXHAUSTED_UNTIL = 0.0  # unix ts (UTC midnight-ish)
API_RATE_LIMIT_UNTIL = 0.0       # unix ts (short cooldown)

def _next_utc_midnight_ts() -> float:
    dt = datetime.now(timezone.utc)
    next_mid = (dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
    return next_mid.timestamp()



LAST_QUOTA_RECHECK_TS = 0.0

def _recheck_api_daily_quota() -> dict:
    """Recheck daily quota using /status. Returns dict with remaining/current/limit."""
    global LAST_QUOTA_RECHECK_TS
    now = time.time()
    if now - LAST_QUOTA_RECHECK_TS < 10:
        return {"ok": False, "reason": "throttled"}
    LAST_QUOTA_RECHECK_TS = now

    try:
        url = API_BASE + "/status"
        headers = {"x-apisports-key": API_FOOTBALL_KEY}
        r = requests.get(url, headers=headers, timeout=10)
        # Prefer headers if present
        hdr_rem = r.headers.get("x-ratelimit-requests-remaining") or r.headers.get("x-requests-remaining")
        if hdr_rem is not None and str(hdr_rem).strip().isdigit():
            return {"ok": True, "remaining": int(str(hdr_rem).strip()), "source": "headers"}
        # Fallback to JSON body
        js = r.json()
        req = (js.get("response") or {}).get("requests") or {}
        cur = req.get("current")
        lim = req.get("limit_day")
        if isinstance(cur, int) and isinstance(lim, int) and lim >= cur:
            return {"ok": True, "remaining": lim - cur, "current": cur, "limit_day": lim, "source": "json"}
        return {"ok": False, "reason": "unparseable"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    dt = datetime.now(timezone.utc)
    next_mid = (dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
    return next_mid.timestamp()

DB_PATH = os.path.join("data", "predictions_history.db")
os.makedirs("data", exist_ok=True)


DEFAULT_LEAGUE = 39  # Premier League

# ============================================================
# ADMIN TOKEN (simple protection for sensitive endpoints)
# ============================================================
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


def require_admin(
    x_admin_token: str | None = Header(None, alias="X-Admin-Token")
) -> None:
    """
    Simple admin guard:

    - If ADMIN_TOKEN is NOT set:
        -> do nothing (useful for local/dev)
    - If ADMIN_TOKEN IS set:
        -> require matching X-Admin-Token header
    """
    # Dev mode: no ADMIN_TOKEN configured -> don't enforce anything
    if not ADMIN_TOKEN:
        return

    # Prod mode: ADMIN_TOKEN set -> header required
    if x_admin_token is None:
        raise HTTPException(
            status_code=401,
            detail="X-Admin-Token header required.",
        )

    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin token.",
        )


DEFAULT_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024,2025]
MAX_DATE_RANGE_DAYS = 14

TARGET_COLS = ["home_goals", "away_goals"]

FEATURE_COLS_BASE = [
    "home_team_idx",
    "away_team_idx",
    "home_advantage",
    "home_att_str",
    "home_def_str",
    "away_att_str",
    "away_def_str",
    "home_elo",
    "away_elo",
]


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("winmatic")
logger.setLevel(logging.INFO)

# ---- Lightweight API usage tracking (helps with quota + performance debugging) ----
# (Counts HTTP calls only; cache hits are not counted)
API_USAGE_LOCK = threading.Lock()
API_USAGE = {
    "total_http_calls": 0,
    "by_path": {},          # path -> count
    "last_headers": {},     # last seen rate/quota headers
}

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

# ============================================================
# API CACHE HELPERS
# ============================================================

_CACHE_MEMO: Dict[str, Dict[str, Any]] = {}

def _load_cache_file(path: str) -> Dict[str, Any]:
    if path in _CACHE_MEMO:
        return _CACHE_MEMO[path]
    if not os.path.exists(path):
        data: Dict[str, Any] = {}
    else:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load cache file %s: %s", path, e)
            data = {}
    _CACHE_MEMO[path] = data
    return data

def _build_cache_query_variants(params: Dict[str, Any]) -> List[str]:
    items = [(k, v) for k, v in (params or {}).items() if v is not None]
    if not items:
        return [""]
    variants: List[str] = []
    seen: set = set()
    n = len(items)
    for mask in range((1 << n) - 1, -1, -1):
        subset = [items[i] for i in range(n) if mask & (1 << i)]
        orders = [subset]
        if len(subset) > 1:
            orders.append(sorted(subset))
        for order in orders:
            pairs = [f"{key}={value}" for key, value in order]
            query = "&".join(pairs)
            if query not in seen:
                seen.add(query)
                variants.append(query)
    if "" not in seen:
        variants.append("")
    return variants

def _cache_keys(path: str, params: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    clean_path = path.lstrip("/") or path
    queries = _build_cache_query_variants(params or {})
    question_keys: List[str] = []
    pipe_keys: List[str] = []
    for query in queries:
        if query:
            question_keys.append(f"{clean_path}?{query}")
            pipe_keys.append(f"{clean_path}|{query}")
        else:
            question_keys.append(clean_path)
            pipe_keys.append(clean_path)
    question_keys = list(dict.fromkeys(question_keys))
    pipe_keys = list(dict.fromkeys(pipe_keys))
    return question_keys, pipe_keys

def cached_api_response(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q_keys, pipe_keys = _cache_keys(path, params)
    api_cache = _load_cache_file(API_CACHE_FILE)
    for key in q_keys:
        if key in api_cache:
            entry = api_cache[key]
            if isinstance(entry, dict) and "data" in entry:
                logger.info("[API CACHE HIT] %s (api_cache)", key)
                return {"response": entry["data"]}
    disk_cache = _load_cache_file(API_DISK_CACHE_FILE)
    for key in pipe_keys:
        if key in disk_cache:
            entry = disk_cache[key]
            if isinstance(entry, dict) and "data" in entry:
                logger.info("[API CACHE HIT] %s (api_disk_cache)", key)
                return {"response": entry["data"]}
    return None

def cached_upcoming_fixtures(league: int, season: int, next_count: int = 50) -> List[Dict[str, Any]]:
    cached = cached_api_response("/fixtures", {"league": league, "season": season, "next": next_count})
    if cached:
        logger.info("[API CACHE MODE] using cached upcoming fixtures league=%s season=%s", league, season)
        return cached.get("response", []) or []
    return []

def _list_snapshot_files(prefix: str) -> List[str]:
    if not os.path.isdir(SNAPSHOT_DIR):
        return []
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.startswith(prefix) and f.endswith(".json")]
    files.sort(reverse=True)
    return [os.path.join(SNAPSHOT_DIR, f) for f in files]

def load_snapshot_predictions(league: int, days_ahead: int = 7, label: str = "upcoming") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    prefix = f"{label}_{league}_{days_ahead}_"
    candidates = _list_snapshot_files(prefix)
    if not candidates:
        prefix = f"{label}_{league}_"
        candidates = _list_snapshot_files(prefix)
    for path in candidates:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            fixtures = data.get("fixtures")
            if fixtures:
                logger.warning("[SNAPSHOT LOAD] using snapshot league=%s file=%s fixtures=%s",
                               league, os.path.basename(path), len(fixtures))
                return fixtures, path
        except Exception as exc:
            logger.warning("Failed to load snapshot %s: %s", path, exc)
    return [], None

def is_cache_only_mode() -> bool:
    """
    Returns True if we're running in developer cache-only mode.
    """
    return os.getenv("WINMATIC_CACHE_ONLY", "0") == "1"

# ============================================================
# UTILS
# ============================================================


def api_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    global API_QUOTA_EXHAUSTED_UNTIL, API_RATE_LIMIT_UNTIL, API_QUOTA_EXHAUSTED
    """Call API-FOOTBALL with cache + daily-quota protection."""
    global API_QUOTA_EXHAUSTED

    # 🧩 Developer mode: skip all live API calls
    if is_cache_only_mode():
        cached = try_cache(path, params, reason="cache-only-mode")
        if cached is not None:
            logger.info("[API CACHE MODE] served=%s reason=cache-only-mode", path)
            return cached
        raise HTTPException(
            status_code=503,
            detail="Cache-only mode active: no live API requests allowed."
        )

    def try_cache(reason: str) -> Optional[Dict[str, Any]]:
        cached = cached_api_response(path, params)
        if cached:
            logger.info("[API CACHE MODE] served=%s reason=%s", path, reason)
        return cached

    # --- Handle missing key -------------------------------------------------
    if not API_FOOTBALL_KEY:
        cached = try_cache("missing-key")
        if cached:
            return cached
        raise HTTPException(
            status_code=500,
            detail="API_FOOTBALL_KEY not configured in environment"
        )

    # --- Stop if rate limit / daily quota already hit ------------------------
    now = time.time()

    # Short cooldown: rate limit (seconds/minutes)
    if API_RATE_LIMIT_UNTIL and now < API_RATE_LIMIT_UNTIL:
        cached = try_cache("rate-limited")
        if cached:
            return cached
        raise HTTPException(
            status_code=429,
            detail=f"API-FOOTBALL rate limit active; retry in {int(API_RATE_LIMIT_UNTIL - now)}s"
        )

    # Daily quota (until next UTC midnight-ish)
    if API_QUOTA_EXHAUSTED_UNTIL and now < API_QUOTA_EXHAUSTED_UNTIL:
        # stale quota flag protection: recheck /status once when blocked
        chk = _recheck_api_daily_quota()
        if chk.get('ok') and isinstance(chk.get('remaining'), int) and chk['remaining'] > 0:
            API_QUOTA_EXHAUSTED_UNTIL = 0.0
            API_QUOTA_EXHAUSTED = False
        else:
                cached = try_cache("quota-exhausted")
                if cached:
                    return cached
        raise HTTPException(
            status_code=429,
            detail="API-FOOTBALL daily request limit already reached (quota exhausted)."
        )

    # Legacy boolean (avoid permanent lock if it was set incorrectly)
    if API_QUOTA_EXHAUSTED:
        API_QUOTA_EXHAUSTED_UNTIL = max(API_QUOTA_EXHAUSTED_UNTIL, _next_utc_midnight_ts())
        API_QUOTA_EXHAUSTED = False

    # --- Cache-only mode toggle ---------------------------------------------
    if CACHE_ONLY_MODE:
        cached = try_cache("cache-only-mode")
        if cached:
            return cached
        raise HTTPException(
            status_code=503,
            detail="Cache-only mode enabled but no cached data for request."
        )

    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    # Build URL robustly (avoid '...ioodds' when path lacks a leading slash)
    from urllib.parse import urljoin
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
        url = path
    else:
        base = (API_BASE or "").rstrip("/") + "/"
        p = (path or "").lstrip("/")
        url = urljoin(base, p)
    logger.info("[API CALL] %s %s", url, params)

    # --- Perform the request ------------------------------------------------
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        # Track real HTTP calls (cache hits don't reach this line)
        try:
            with API_USAGE_LOCK:
                API_USAGE['total_http_calls'] = API_USAGE.get('total_http_calls', 0) + 1
                API_USAGE['by_path'][path] = API_USAGE.get('by_path', {}).get(path, 0) + 1
                hdr = {}
                for k in [
                    'x-requests-remaining','x-requests-limit',
                    'x-ratelimit-remaining','x-ratelimit-limit','x-ratelimit-reset',
                    'x-rate-limit-remaining','x-rate-limit-limit','x-rate-limit-reset',
                ]:
                    if k in (resp.headers or {}):
                        hdr[k] = resp.headers.get(k)
                if hdr:
                    API_USAGE['last_headers'] = hdr
        except Exception:
            pass
        data = resp.json()

        if data is None:

            data = {}
    except Exception as e:
        logger.warning("[API ERROR] %s", e)
        cached = try_cache("network-error")
        if cached:
            return cached
        raise HTTPException(status_code=502, detail=str(e))

    # --- Non-200 status codes -----------------------------------------------
    if resp.status_code != 200:
        # If 429, treat as short rate-limit cooldown unless daily quota headers indicate otherwise
        if resp.status_code == 429:
            try:
                daily_rem = (resp.headers.get("x-requests-remaining") or "").strip()
                rate_rem = (resp.headers.get("x-ratelimit-remaining") or resp.headers.get("x-rate-limit-remaining") or "").strip()
                rate_reset = (resp.headers.get("x-ratelimit-reset") or resp.headers.get("x-rate-limit-reset") or "").strip()
                now = time.time()
                if daily_rem == "0":
                    globals()["API_QUOTA_EXHAUSTED_UNTIL"] = max(globals()["API_QUOTA_EXHAUSTED_UNTIL"], _next_utc_midnight_ts())
                else:
                    wait = 60
                    if rate_reset.isdigit():
                        # Some APIs provide epoch seconds; some provide seconds-from-now. Handle both.
                        rr = int(rate_reset)
                        if rr > now:
                            wait = max(wait, int(rr - now))
                        else:
                            wait = max(wait, rr)
                    globals()["API_RATE_LIMIT_UNTIL"] = max(globals()["API_RATE_LIMIT_UNTIL"], now + wait)
            except Exception:
                pass

        cached = try_cache(f"http-{resp.status_code}")
        if cached:
            return cached
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"API-FOOTBALL error: {data}"
        )

    # --- API-level errors (daily limit etc.) --------------------------------
    if "errors" in data and data["errors"]:
        logger.error("[API ERRORS] %s", data["errors"])

        # Detect “request limit” messages
        try:
            errs = data["errors"]
            msg = ""
            if isinstance(errs, dict) and "requests" in errs:
                msg = str(errs["requests"])
            elif isinstance(errs, (list, tuple)) and errs:
                msg = str(errs[0])
            else:
                msg = str(errs)

            lower_msg = msg.lower()
            if "request limit" in lower_msg or (
                "limit" in lower_msg and "request" in lower_msg
            ):
                API_QUOTA_EXHAUSTED = True
                logger.warning(
                    "[API QUOTA] Daily request limit reached. "
                    "API_QUOTA_EXHAUSTED set to True."
                )
        except Exception:
            pass

        cached = try_cache("api-error")
        if cached:
            return cached
        raise HTTPException(
            status_code=502,
            detail=f"API-FOOTBALL error: {data['errors']}"
        )

    # --- Happy path ---------------------------------------------------------
    return (data or {})
    if resp.status_code != 200:
            cached = try_cache(f"http-{resp.status_code}")
            if cached:
                return cached
            raise HTTPException(status_code=resp.status_code, detail=f"API-FOOTBALL error: {data}")

    if "errors" in data and data["errors"]:
            logger.error("[API ERRORS] %s", data["errors"])
            cached = try_cache("api-error")
            if cached:
                return cached
            raise HTTPException(status_code=502, detail=f"API-FOOTBALL error: {data['errors']}")

    return data

    raise HTTPException(status_code=502, detail="API-FOOTBALL retries exhausted")

    

# ---------------------------------------------------------------------------
# Fixture logo helper (used by /results/recent)
# NOTE: must never raise (otherwise /results/recent returns 500 and the UI breaks).
# We cache by fixture_id to avoid eating API-FOOTBALL quota on every page load.
# ---------------------------------------------------------------------------
_FIXTURE_LOGO_CACHE = {}  # fixture_id -> {"ts": float, "home": str|None, "away": str|None}
_FIXTURE_LOGO_TTL_SEC = int(os.getenv("FIXTURE_LOGO_TTL_SEC", "21600"))  # 6h default



from functools import lru_cache
from typing import Optional, Tuple

@lru_cache(maxsize=512)
def get_fixture_logos(fixture_id: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort: fetch home/away logo URLs for a fixture.

    IMPORTANT: must NEVER raise (quota/network/weird data -> (None, None)).
    """
    try:
        data = api_get("/fixtures", {"id": int(fixture_id)}) or {}
        if not isinstance(data, dict):
            return None, None

        resp = data.get("response") or []
        if not resp:
            return None, None

        teams = (resp[0] or {}).get("teams") or {}
        home_logo = (teams.get("home") or {}).get("logo")
        away_logo = (teams.get("away") or {}).get("logo")
        return home_logo, away_logo
    except Exception as e:
        try:
            logger.warning("[RESULTS LOGOS] fixture=%s failed: %s", fixture_id, e)
        except Exception:
            pass
        return None, None

def get_fixture_logos(fixture_id: int):
    """Best-effort: return (home_logo_url, away_logo_url) for a fixture.

    - Uses API-FOOTBALL /fixtures?id=<fixture_id>
    - Caches results for FIXTURE_LOGO_TTL_SEC seconds
    - NEVER raises (returns (None, None) on any error)
    """
    try:
        now = time.time()
        ent = _FIXTURE_LOGO_CACHE.get(int(fixture_id))
        if ent and (now - float(ent.get("ts", 0.0)) < _FIXTURE_LOGO_TTL_SEC):
            return ent.get("home"), ent.get("away")

        j = api_get("/fixtures", {"id": int(fixture_id)})
        resp = (j or {}).get("response") or []
        if not resp:
            return None, None

        fx = resp[0] or {}
        teams = fx.get("teams") or {}
        home_logo = (teams.get("home") or {}).get("logo")
        away_logo = (teams.get("away") or {}).get("logo")

        _FIXTURE_LOGO_CACHE[int(fixture_id)] = {"ts": now, "home": home_logo, "away": away_logo}
        return home_logo, away_logo
    except Exception:
        return None, None

def current_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1

# ============================================================
# HISTORY DB
# ============================================================

def init_history_db() -> None:
    os.makedirs(ART, exist_ok=True)
    conn = db_connect()

    # If we're on Neon (Postgres), create the table with a proper UNIQUE constraint.
    if getattr(conn, "is_pg", False):
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions_history (
          id BIGSERIAL PRIMARY KEY,
          fixture_id BIGINT NOT NULL,
          league INT NOT NULL,
          home_team TEXT,
          away_team TEXT,
          kickoff_utc TIMESTAMPTZ NOT NULL,

          model_home_p DOUBLE PRECISION,
          model_draw_p DOUBLE PRECISION,
          model_away_p DOUBLE PRECISION,

          predicted_side TEXT,
          edge_value DOUBLE PRECISION,

          actual_result TEXT,

          payload TEXT,
          created_at TIMESTAMPTZ DEFAULT NOW(),

          market_home_p DOUBLE PRECISION,
          market_draw_p DOUBLE PRECISION,
          market_away_p DOUBLE PRECISION,

          market_home_odds DOUBLE PRECISION,
          market_draw_odds DOUBLE PRECISION,
          market_away_odds DOUBLE PRECISION,

          market_bookmaker TEXT,
          market_bet_name TEXT,

          CONSTRAINT uq_predictions_history UNIQUE (league, fixture_id, kickoff_utc)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS odds_history (
          id BIGSERIAL PRIMARY KEY,
          fixture_id BIGINT NOT NULL,
          league INT NOT NULL,
          kickoff_utc TIMESTAMPTZ NOT NULL,
          snapshot_type TEXT NOT NULL,
          bookmaker TEXT,
          odds_home DOUBLE PRECISION,
          odds_draw DOUBLE PRECISION,
          odds_away DOUBLE PRECISION,
          created_at TIMESTAMPTZ DEFAULT NOW(),
          CONSTRAINT uq_odds_history UNIQUE (league, fixture_id, kickoff_utc, snapshot_type)
        );
        """)
        conn.commit()
        cur.close()
        conn.close()
        return
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league INTEGER NOT NULL,
            fixture_id INTEGER NOT NULL,
            kickoff_utc TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(league, fixture_id, kickoff_utc)
        );
        """
    )
    conn.commit()
    conn.close()
    logger.info("[DB] history.db ready")
def record_predictions_history(league: int, fixtures: list[dict]) -> None:
    """Persist predictions to predictions_history.

    - Upserts by (league, fixture_id, kickoff_utc).
    - Stores bet-side pick when available:
        predicted_side = value_side (if present) else predictions.best_side.
      This keeps ROI/PnL endpoints consistent with edge_value/best_edge.
    - Stores market columns when odds/implied fields are present.
    """
    if not fixtures:
        return

    try:
        ensure_predictions_db()
    except Exception as e:
        logger.warning("[DB] ensure_predictions_db failed: %s", e)

    conn = None
    try:
        conn = db_connect()
        cur = conn.cursor()

        # Ensure unique index for conflict target
        try:
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_history_key "
                "ON predictions_history (league, fixture_id, kickoff_utc);"
            )
        except Exception as e:
            logger.warning("[DB] Could not ensure unique index: %s", e)

        sql = """
            INSERT INTO predictions_history (
                league, fixture_id, home_team, away_team, kickoff_utc,
                model_home_p, model_draw_p, model_away_p,
                predicted_side, edge_value,
                market_home_p, market_draw_p, market_away_p,
                market_home_odds, market_draw_odds, market_away_odds,
                market_bookmaker, market_bet_name,
                payload, created_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, datetime('now')
            )
            ON CONFLICT(league, fixture_id, kickoff_utc) DO UPDATE SET
                home_team = excluded.home_team,
                away_team = excluded.away_team,
                model_home_p = excluded.model_home_p,
                model_draw_p = excluded.model_draw_p,
                model_away_p = excluded.model_away_p,
                predicted_side = COALESCE(excluded.predicted_side, predictions_history.predicted_side),
                edge_value = COALESCE(excluded.edge_value, predictions_history.edge_value),
                market_home_p = COALESCE(excluded.market_home_p, predictions_history.market_home_p),
                market_draw_p = COALESCE(excluded.market_draw_p, predictions_history.market_draw_p),
                market_away_p = COALESCE(excluded.market_away_p, predictions_history.market_away_p),
                market_home_odds = COALESCE(excluded.market_home_odds, predictions_history.market_home_odds),
                market_draw_odds = COALESCE(excluded.market_draw_odds, predictions_history.market_draw_odds),
                market_away_odds = COALESCE(excluded.market_away_odds, predictions_history.market_away_odds),
                market_bookmaker = COALESCE(excluded.market_bookmaker, predictions_history.market_bookmaker),
                market_bet_name = COALESCE(excluded.market_bet_name, predictions_history.market_bet_name),
                payload = excluded.payload,
                created_at = COALESCE(NULLIF(predictions_history.created_at,''), excluded.created_at)
        """

        rows = []
        for f in fixtures:
            fixture_id = f.get("fixture_id") or (f.get("fixture") or {}).get("id")
            kickoff_utc = f.get("kickoff_utc") or (f.get("fixture") or {}).get("date")
            if fixture_id is None or kickoff_utc is None:
                continue

            home_team = f.get("home_team") or f.get("home_name") or (f.get("teams") or {}).get("home", {}).get("name")
            away_team = f.get("away_team") or f.get("away_name") or (f.get("teams") or {}).get("away", {}).get("name")

            preds = f.get("predictions") or {}
            ph = preds.get("home_win_p")
            pd = preds.get("draw_p")
            pa = preds.get("away_win_p")

            # bet-side + edge
            predicted_side = (f.get("value_side") or preds.get("best_side") or f.get("predicted_side"))
            try:
                predicted_side = (predicted_side or "").strip().lower() or None
            except Exception:
                predicted_side = None

            edge_value = f.get("best_edge")
            if edge_value is None:
                edge_value = f.get("edge_value")

            # market fields
            implied = f.get("implied_1x2") or {}
            odds = f.get("odds_1x2") or {}
            mhp = implied.get("home")
            mdp = implied.get("draw")
            map_ = implied.get("away")
            mho = odds.get("home")
            mdo = odds.get("draw")
            mao = odds.get("away")

            mbook = f.get("market_bookmaker")
            mbet = f.get("market_bet_name")

            payload = json.dumps(f, ensure_ascii=False)

            rows.append(
                (
                    int(league), int(fixture_id), home_team, away_team, str(kickoff_utc),
                    ph, pd, pa,
                    predicted_side, edge_value,
                    mhp, mdp, map_,
                    mho, mdo, mao,
                    mbook, mbet,
                    payload,
                )
            )

        if rows:
            cur.executemany(sql, rows)
            conn.commit()

    except Exception as e:
        logger.exception("[DB] record_predictions_history failed: %s", e)
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def record_odds_snapshot(snap: "OddsSnapshot") -> None:
    """Upsert a snapshot into odds_history (Postgres or SQLite)."""
    conn = db_connect()
    try:
        cur = conn.cursor()
        if getattr(conn, "is_pg", False):
            cur.execute(
                """
                INSERT INTO odds_history (
                    league, fixture_id, kickoff_utc, snapshot_type, bookmaker,
                    odds_home, odds_draw, odds_away, created_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s, NOW())
                ON CONFLICT (league, fixture_id, kickoff_utc, snapshot_type)
                DO UPDATE SET
                    bookmaker=EXCLUDED.bookmaker,
                    odds_home=EXCLUDED.odds_home,
                    odds_draw=EXCLUDED.odds_draw,
                    odds_away=EXCLUDED.odds_away,
                    created_at=NOW()
                """,
                (
                    int(snap.league),
                    int(snap.fixture_id),
                    snap.kickoff_utc,
                    snap.snapshot_type,
                    snap.bookmaker,
                    float(snap.odds_home),
                    float(snap.odds_draw),
                    float(snap.odds_away),
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO odds_history (
                    league, fixture_id, kickoff_utc, snapshot_type, bookmaker,
                    odds_home, odds_draw, odds_away, created_at
                )
                VALUES (?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(league, fixture_id, kickoff_utc, snapshot_type)
                DO UPDATE SET
                    bookmaker=excluded.bookmaker,
                    odds_home=excluded.odds_home,
                    odds_draw=excluded.odds_draw,
                    odds_away=excluded.odds_away,
                    created_at=datetime('now')
                """,
                (
                    int(snap.league),
                    int(snap.fixture_id),
                    snap.kickoff_utc,
                    snap.snapshot_type,
                    snap.bookmaker,
                    float(snap.odds_home),
                    float(snap.odds_draw),
                    float(snap.odds_away),
                ),
            )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass




def get_odds_snapshot_by_fixture(league: int, fixture_id: int, snapshot_type: str) -> dict | None:
    """Fetch a snapshot by (league, fixture_id, snapshot_type) ignoring kickoff_utc string mismatch."""
    conn = db_connect()
    try:
        cur = conn.cursor()
        if getattr(conn, "is_pg", False):
            cur.execute(
                """
                SELECT kickoff_utc, bookmaker, odds_home, odds_draw, odds_away, created_at
                FROM odds_history
                WHERE league=%s AND fixture_id=%s AND snapshot_type=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (int(league), int(fixture_id), snapshot_type),
            )
        else:
            cur.execute(
                """
                SELECT kickoff_utc, bookmaker, odds_home, odds_draw, odds_away, created_at
                FROM odds_history
                WHERE league=? AND fixture_id=? AND snapshot_type=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (int(league), int(fixture_id), snapshot_type),
            )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "kickoff_utc": row[0],
            "bookmaker": row[1],
            "odds_home": row[2],
            "odds_draw": row[3],
            "odds_away": row[4],
            "created_at": row[5],
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_odds_snapshot(league: int, fixture_id: int, kickoff_utc: str, snapshot_type: str) -> dict | None:
    """Fetch a single snapshot from odds_history."""
    conn = db_connect()
    try:
        cur = conn.cursor()
        if getattr(conn, "is_pg", False):
            cur.execute(
                """
                SELECT bookmaker, odds_home, odds_draw, odds_away, created_at
                FROM odds_history
                WHERE league=%s AND fixture_id=%s AND kickoff_utc=%s AND snapshot_type=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (int(league), int(fixture_id), kickoff_utc, snapshot_type),
            )
        else:
            cur.execute(
                """
                SELECT bookmaker, odds_home, odds_draw, odds_away, created_at
                FROM odds_history
                WHERE league=? AND fixture_id=? AND kickoff_utc=? AND snapshot_type=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (int(league), int(fixture_id), kickoff_utc, snapshot_type),
            )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "bookmaker": row[0],
            "odds_home": row[1],
            "odds_draw": row[2],
            "odds_away": row[3],
            "created_at": row[4],
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


def model_paths(league_id: int) -> Tuple[str, str]:
    model_path = os.path.join(ART, f"model_{league_id}.joblib")
    meta_path = os.path.join(ART, f"meta_{league_id}.json")
    return model_path, meta_path

def save_model_and_meta(league_id: int, model: Any, meta: Dict[str, Any]) -> None:
    model_path, meta_path = model_paths(league_id)
    dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("[MODEL SAVED] league=%s model=%s meta=%s", league_id, model_path, meta_path)

def load_model_and_meta(league_id: int) -> Tuple[Any, Dict[str, Any]]:
    model_path, meta_path = model_paths(league_id)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise HTTPException(status_code=400, detail=f"No model trained yet for league {league_id}. Call /train first.")
    model = load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta

# ============================================================
# TRAINING
# ============================================================

def fetch_historic_fixtures(league_id: int, seasons: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for season in seasons:
        data = api_get("/fixtures", {"league": league_id, "season": season, "status": "FT"})
        resp = data.get("response", [])
        if not resp:
            logger.warning("[TRAIN] No fixtures returned for league=%s season=%s", league_id, season)
            continue
        for fx in resp:
            fixture = fx.get("fixture", {}) or {}
            teams = fx.get("teams", {}) or {}
            goals = fx.get("goals", {}) or {}
            home = teams.get("home", {}) or {}
            away = teams.get("away", {}) or {}
            if not home or not away:
                continue
            home_id = home.get("id")
            away_id = away.get("id")
            if home_id is None or away_id is None:
                continue
            rows.append(
                {
                    "fixture_id": fixture.get("id"),
                    "league": league_id,
                    "season": season,
                    "date": fixture.get("date"),
                    "home_id": home_id,
                    "away_id": away_id,
                    "home_name": home.get("name"),
                    "away_name": away.get("name"),
                    "home_goals": goals.get("home", 0),
                    "away_goals": goals.get("away", 0),
                }
            )
    if not rows:
        raise HTTPException(status_code=400, detail="No historic fixtures fetched; check league/seasons/API key.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["fixture_id"]).reset_index(drop=True)
    return df

def build_team_strengths(df: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        records.append({"team_id": int(r["home_id"]), "team_name": r.get("home_name"), "gf": float(r["home_goals"]), "ga": float(r["away_goals"])})
        records.append({"team_id": int(r["away_id"]), "team_name": r.get("away_name"), "gf": float(r["away_goals"]), "ga": float(r["home_goals"])})
    tdf = pd.DataFrame(records)
    grouped = tdf.groupby("team_id", as_index=False).agg({"team_name": "first", "gf": "sum", "ga": "sum"})
    matches_home = df.groupby("home_id").size().rename("home_matches")
    matches_away = df.groupby("away_id").size().rename("away_matches")
    matches = pd.concat([matches_home, matches_away], axis=1).fillna(0.0)
    matches["matches"] = matches["home_matches"] + matches["away_matches"]
    matches = matches["matches"].rename_axis("team_id").reset_index()
    grouped = grouped.merge(matches, on="team_id", how="left")
    grouped["matches"] = grouped["matches"].replace(0, np.nan)
    grouped["gf_per_match"] = grouped["gf"] / grouped["matches"]
    grouped["ga_per_match"] = grouped["ga"] / grouped["matches"]
    league_avg_gf = grouped["gf_per_match"].mean()
    league_avg_ga = grouped["ga_per_match"].mean()

    def safe_ratio(x: float, denom: float) -> float:
        if denom is None or denom <= 0: return 1.0
        if x is None or np.isnan(x): return 1.0
        return float(x) / float(denom)

    grouped["attack_strength"] = grouped["gf_per_match"].apply(lambda v: safe_ratio(v, league_avg_gf))
    grouped["defense_strength"] = grouped["ga_per_match"].apply(lambda v: safe_ratio(v, league_avg_ga))
    grouped["rating"] = (grouped["attack_strength"] / grouped["defense_strength"].replace(0, np.nan)).fillna(1.0)

    attack_strength = {int(r["team_id"]): float(r["attack_strength"]) for _, r in grouped.iterrows()}
    defense_strength = {int(r["team_id"]): float(r["defense_strength"]) for _, r in grouped.iterrows()}
    team_summary = {
        int(r["team_id"]): {
            "team_id": int(r["team_id"]),
            "team_name": r["team_name"],
            "matches": int(r["matches"]),
            "gf": float(r["gf"]),
            "ga": float(r["ga"]),
            "attack_strength": float(r["attack_strength"]),
            "defense_strength": float(r["defense_strength"]),
            "rating": float(r["rating"]),
        }
        for _, r in grouped.iterrows()
    }
    return attack_strength, defense_strength, team_summary

def add_elo_features(
    df: pd.DataFrame,
    k_factor: float = 20.0,
    home_advantage_elo: float = 100.0,
    initial_rating: float = 1500.0,
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    Compute simple Elo ratings for each team over time and add
    pre-match Elo features to the dataframe:

    - home_elo: Elo rating of the home team *before* the match
    - away_elo: Elo rating of the away team *before* the match

    Returns a (df_with_elo, elo_ratings_dict) tuple, where elo_ratings_dict
    contains the final Elo for each team_id.
    """
    if df.empty:
        return df, {}

    # Work on a copy sorted by date
    df_sorted = df.copy()
    df_sorted["date_dt"] = pd.to_datetime(df_sorted["date"], utc=True, errors="coerce")
    df_sorted = df_sorted.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index()

    elo: Dict[int, float] = {}
    home_elos: List[float] = []
    away_elos: List[float] = []

    for _, row in df_sorted.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        Rh = elo.get(home_id, initial_rating)
        Ra = elo.get(away_id, initial_rating)

        # Pre-match Elo ratings
        home_elos.append(Rh)
        away_elos.append(Ra)

        # Match outcome
        hg = float(row["home_goals"])
        ag = float(row["away_goals"])
        if hg > ag:
            sh, sa = 1.0, 0.0
        elif hg < ag:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        # Expected score for home, with a small home-advantage in rating space
        diff = (Rh + home_advantage_elo) - Ra
        Eh = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        Ea = 1.0 - Eh

        # Update Elo ratings
        elo[home_id] = Rh + k_factor * (sh - Eh)
        elo[away_id] = Ra + k_factor * (sa - Ea)

    # Attach pre-match Elo to the sorted copy, then map back to original df
    df_sorted["home_elo"] = home_elos
    df_sorted["away_elo"] = away_elos

    df_with_elo = df.copy()
    df_with_elo.loc[df_sorted["index"], "home_elo"] = df_sorted["home_elo"].values
    df_with_elo.loc[df_sorted["index"], "away_elo"] = df_sorted["away_elo"].values

    # Final Elo map
    elo_ratings = {int(tid): float(r) for tid, r in elo.items()}
    return df_with_elo, elo_ratings


def build_training_frame(league_id: int, seasons: List[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build the main training dataframe and metadata for a given league + seasons.
    """
    # 1) Fetch historic fixtures
    df = fetch_historic_fixtures(league_id, seasons)

    # 2) Basic team indexing
    team_ids = pd.unique(df[["home_id", "away_id"]].values.ravel())
    team_index = {int(t): i for i, t in enumerate(sorted(team_ids))}
    df["home_team_idx"] = df["home_id"].map(team_index).astype(float)
    df["away_team_idx"] = df["away_id"].map(team_index).astype(float)
    df["home_advantage"] = 1.0

    # 3) Team strengths (attack / defence / rating)
    attack_strength, defense_strength, team_summary = build_team_strengths(df)
    df["home_att_str"] = df["home_id"].map(attack_strength).astype(float)
    df["home_def_str"] = df["home_id"].map(defense_strength).astype(float)
    df["away_att_str"] = df["away_id"].map(attack_strength).astype(float)
    df["away_def_str"] = df["away_id"].map(defense_strength).astype(float)

    # 4) Elo ratings (pre-match) as extra strength features
    df, elo_ratings = add_elo_features(df)

    # 5) Last-5 form (simple GF / GA)
    form_rows: List[Dict[str, Any]] = []
    for tid in team_ids:
        team_df = df[(df["home_id"] == tid) | (df["away_id"] == tid)].sort_values("date")
        gf_list: List[float] = []
        ga_list: List[float] = []

        for _, r in team_df.iterrows():
            if r["home_id"] == tid:
                gf_list.append(r["home_goals"])
                ga_list.append(r["away_goals"])
            else:
                gf_list.append(r["away_goals"])
                ga_list.append(r["home_goals"])

            if len(gf_list) > 5:
                gf_list.pop(0)
                ga_list.pop(0)

            form_rows.append(
                {
                    "fixture_id": r["fixture_id"],
                    "team_id": tid,
                    "form_gf": float(np.mean(gf_list)),
                    "form_ga": float(np.mean(ga_list)),
                }
            )

    form_df = pd.DataFrame(form_rows)
    df = df.merge(
        form_df.rename(columns={"team_id": "home_id", "form_gf": "home_form_gf", "form_ga": "home_form_ga"})[
            ["fixture_id", "home_id", "home_form_gf", "home_form_ga"]
        ],
        on=["fixture_id", "home_id"],
        how="left",
    )
    df = df.merge(
        form_df.rename(columns={"team_id": "away_id", "form_gf": "away_form_gf", "form_ga": "away_form_ga"})[
            ["fixture_id", "away_id", "away_form_gf", "away_form_ga"]
        ],
        on=["fixture_id", "away_id"],
        how="left",
    )
    df[["home_form_gf", "home_form_ga", "away_form_gf", "away_form_ga"]] = (
        df[["home_form_gf", "home_form_ga", "away_form_gf", "away_form_ga"]]
        .fillna(1.0)
        .astype(float)
    )

    # 6) Add form points, congestion, rest days (NO odds here)
    df = add_form_points_features(df)
    df = add_schedule_congestion_features(df)
    df = add_rest_days_features(df)

    # 7) Proxy stats: shots & possession using team_summary
    team_stats_records: List[Dict[str, Any]] = []
    for tid, info in team_summary.items():
        matches = float(info.get("matches", 0.0)) or np.nan
        gf = float(info.get("gf", 0.0))
        ga = float(info.get("ga", 0.0))
        rating = float(info.get("rating", 1.0))
        team_stats_records.append(
            {
                "team_id": int(tid),
                "matches": matches,
                "gf": gf,
                "ga": ga,
                "rating": rating,
            }
        )

    team_stats_df = pd.DataFrame(team_stats_records)
    team_stats_df["gf_per_match"] = team_stats_df["gf"] / team_stats_df["matches"]
    league_gf_per_match = float(team_stats_df["gf_per_match"].mean())
    team_stats_df["gf_per_match"] = team_stats_df["gf_per_match"].fillna(league_gf_per_match)

    gf_per_match_map = team_stats_df.set_index("team_id")["gf_per_match"].to_dict()
    rating_map = team_stats_df.set_index("team_id")["rating"].to_dict()

    df["home_shots_proxy"] = (
        df["home_id"].map(gf_per_match_map).fillna(league_gf_per_match) * 3.5
    )
    df["away_shots_proxy"] = (
        df["away_id"].map(gf_per_match_map).fillna(league_gf_per_match) * 3.5
    )

    def _possession_proxy(row) -> float:
        rh = float(rating_map.get(row["home_id"], 1.0))
        ra = float(rating_map.get(row["away_id"], 1.0))
        total = rh + ra
        if total <= 0:
            return 0.5
        return rh / total

    df["home_possession_proxy"] = df.apply(_possession_proxy, axis=1)
    df["away_possession_proxy"] = 1.0 - df["home_possession_proxy"]

    league_avg_shots_proxy = float(
        pd.concat([df["home_shots_proxy"], df["away_shots_proxy"]]).mean()
    )

    # 8) Feature + target columns — NOTICE: no *_odd_implied now
    feature_cols = FEATURE_COLS_BASE + [
        "home_form_gf",
        "home_form_ga",
        "away_form_gf",
        "away_form_ga",
        "home_form_pts",
        "away_form_pts",
        "home_matches_last_7",
        "home_matches_last_14",
        "away_matches_last_7",
        "away_matches_last_14",
        "home_rest_days",
        "away_rest_days",
        "home_shots_proxy",
        "away_shots_proxy",
        "home_possession_proxy",
        "away_possession_proxy",
    ]
    target_cols = TARGET_COLS.copy()

    meta: Dict[str, Any] = {
        "league_id": league_id,
        "seasons": seasons,
        "team_index": team_index,
        "attack_strength": attack_strength,
        "defense_strength": defense_strength,
        "team_summary": team_summary,
        "elo_ratings": elo_ratings,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "league_gf_per_match": league_gf_per_match,
        "league_avg_shots_proxy": league_avg_shots_proxy,
    }

    return df, meta



def add_form_points_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add form points features:
    - home_form_pts: average points in last 5 games for home team
    - away_form_pts: same for away team

    Points: win=3, draw=1, loss=0.
    We only use PREVIOUS games for the form (not the current one).
    """
    rows: List[Dict[str, Any]] = []

    # Get all unique team IDs
    team_ids = pd.unique(df[["home_id", "away_id"]].values.ravel())

    for tid in team_ids:
        # All games where this team played (home or away), sorted by date
        team_df = df[(df["home_id"] == tid) | (df["away_id"] == tid)].sort_values("date")

        last_points: List[float] = []

        for _, r in team_df.iterrows():
            # --- form based on PREVIOUS matches only ---
            if len(last_points) == 0:
                # neutral value between win(3) and loss(0)
                form_pts = 1.0
            else:
                form_pts = float(np.mean(last_points))

            rows.append(
                {
                    "fixture_id": r["fixture_id"],
                    "team_id": tid,
                    "form_pts": form_pts,
                }
            )

            # --- now update history with THIS match result ---
            if r["home_id"] == tid:
                gf = r["home_goals"]
                ga = r["away_goals"]
            else:
                gf = r["away_goals"]
                ga = r["home_goals"]

            if gf > ga:
                pts = 3.0
            elif gf == ga:
                pts = 1.0
            else:
                pts = 0.0

            last_points.append(pts)
            if len(last_points) > 5:
                # keep only last 5
                last_points.pop(0)

    form_pts_df = pd.DataFrame(rows)

    # Merge into main df for home team
    df = df.merge(
        form_pts_df.rename(columns={"team_id": "home_id", "form_pts": "home_form_pts"})[
            ["fixture_id", "home_id", "home_form_pts"]
        ],
        on=["fixture_id", "home_id"],
        how="left",
    )

    # Merge for away team
    df = df.merge(
        form_pts_df.rename(columns={"team_id": "away_id", "form_pts": "away_form_pts"})[
            ["fixture_id", "away_id", "away_form_pts"]
        ],
        on=["fixture_id", "away_id"],
        how="left",
    )

    # Fill any missing with neutral value 1.0 and cast to float
    df[["home_form_pts", "away_form_pts"]] = df[["home_form_pts", "away_form_pts"]].fillna(1.0).astype(float)

    return df

def add_schedule_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule congestion features:
    - home_matches_last_7 / away_matches_last_7: number of matches in last 7 days
    - home_matches_last_14 / away_matches_last_14: number of matches in last 14 days

    Only previous matches are counted (not including the current one).
    """
    # Ensure we have a datetime column
    if "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    rows: List[Dict[str, Any]] = []

    team_ids = pd.unique(df[["home_id", "away_id"]].values.ravel())

    for tid in team_ids:
        # All games for this team, ordered in time
        team_df = df[(df["home_id"] == tid) | (df["away_id"] == tid)].sort_values("date_dt")

        past_dates: List[pd.Timestamp] = []

        for _, r in team_df.iterrows():
            current_date = r["date_dt"]

            # Count previous matches in last 7 / 14 days
            matches_last_7 = 0
            matches_last_14 = 0

            for d in past_dates:
                delta_days = (current_date - d).days
                if 0 < delta_days <= 7:
                    matches_last_7 += 1
                if 0 < delta_days <= 14:
                    matches_last_14 += 1

            rows.append(
                {
                    "fixture_id": r["fixture_id"],
                    "team_id": tid,
                    "matches_last_7": matches_last_7,
                    "matches_last_14": matches_last_14,
                }
            )

            # Now add this match to history for future rows
            past_dates.append(current_date)

    sched_df = pd.DataFrame(rows)

    # Merge into main df for home team
    df = df.merge(
        sched_df.rename(
            columns={
                "team_id": "home_id",
                "matches_last_7": "home_matches_last_7",
                "matches_last_14": "home_matches_last_14",
            }
        )[["fixture_id", "home_id", "home_matches_last_7", "home_matches_last_14"]],
        on=["fixture_id", "home_id"],
        how="left",
    )

    # Merge for away team
    df = df.merge(
        sched_df.rename(
            columns={
                "team_id": "away_id",
                "matches_last_7": "away_matches_last_7",
                "matches_last_14": "away_matches_last_14",
            }
        )[["fixture_id", "away_id", "away_matches_last_7", "away_matches_last_14"]],
        on=["fixture_id", "away_id"],
        how="left",
    )

    # Fill missing with 0 (if very early in season)
    df[
        [
            "home_matches_last_7",
            "home_matches_last_14",
            "away_matches_last_7",
            "away_matches_last_14",
        ]
    ] = df[
        [
            "home_matches_last_7",
            "home_matches_last_14",
            "away_matches_last_7",
            "away_matches_last_14",
        ]
    ].fillna(0.0).astype(float)

    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add odds-implied 1X2 probabilities as features.

    IMPORTANT (quota): API-Football odds coverage is often limited for older fixtures.
    We only fetch odds for fixtures within a configurable time window, otherwise we keep
    a neutral fallback (1/3, 1/3, 1/3).

    Env vars:
      - ODDS_LOOKBACK_DAYS (default 21)
      - ODDS_FUTURE_DAYS (default 10)
      - ODDS_MAX_CALLS_PER_RUN (default 60)
    """
    if df is None or df.empty or "fixture_id" not in df.columns:
        return df

    # Ensure columns exist
    for c in ["home_odd_implied", "draw_odd_implied", "away_odd_implied"]:
        if c not in df.columns:
            df[c] = 1.0 / 3.0

    calls = 0

    # Iterate rows; fetch odds only when within odds window and fixture_id is valid
    for i, row in df.iterrows():
        fid = row.get("fixture_id")
        try:
            fid_int = int(fid)
        except Exception:
            continue

        kickoff = (
            row.get("kickoff_utc")
            or row.get("date")
            or row.get("kickoff")
            or row.get("fixture_date")
        )

        if not _within_odds_window(kickoff):
            continue

        if calls >= ODDS_MAX_CALLS_PER_RUN:
            logger.info("[ODDS] reached ODDS_MAX_CALLS_PER_RUN=%s; skipping remaining odds fetches", ODDS_MAX_CALLS_PER_RUN)
            break

        try:
            payload = api_get("/odds", {"fixture": fid_int})
            scan = scan_market_odds(payload)
            calls += 1

            if scan and scan.get("found"):
                probs = (scan.get("selected") or {}).get("probs") or {}
                ph = probs.get("home"); pd = probs.get("draw"); pa = probs.get("away")
                if ph is not None and pd is not None and pa is not None:
                    df.at[i, "home_odd_implied"] = float(ph)
                    df.at[i, "draw_odd_implied"] = float(pd)
                    df.at[i, "away_odd_implied"] = float(pa)
        except Exception:
            # Keep defaults on any error
            continue

    return df

def add_rest_days_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rest-days features:
    - home_rest_days: days since the home team last played
    - away_rest_days: days since the away team last played

    Only previous matches are counted (not including the current one).
    """
    # Ensure we have a datetime column
    if "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    rows: List[Dict[str, Any]] = []
    team_ids = pd.unique(df[["home_id", "away_id"]].values.ravel())

    for tid in team_ids:
        # All games for this team, ordered in time
        team_df = df[(df["home_id"] == tid) | (df["away_id"] == tid)].sort_values("date_dt")
        last_date = None

        for _, r in team_df.iterrows():
            current_date = r["date_dt"]
            if pd.isna(current_date):
                continue

            if last_date is None:
                rest = 7.0  # neutral default for first game
            else:
                delta_days = (current_date - last_date).days
                if delta_days < 0:
                    delta_days = 0
                rest = float(max(0, min(delta_days, 21)))  # cap at 21

            if r["home_id"] == tid:
                rows.append({
                    "fixture_id": r["fixture_id"],
                    "home_rest_days": rest,
                })
            if r["away_id"] == tid:
                rows.append({
                    "fixture_id": r["fixture_id"],
                    "away_rest_days": rest,
                })

            last_date = current_date

    rest_df = pd.DataFrame(rows)

    # Merge back into the main frame
    df = df.merge(
        rest_df[["fixture_id", "home_rest_days"]].drop_duplicates("fixture_id"),
        on="fixture_id",
        how="left",
    )
    df = df.merge(
        rest_df[["fixture_id", "away_rest_days"]].drop_duplicates("fixture_id"),
        on="fixture_id",
        how="left",
    )

    # Fill missing rest with a neutral value
    df["home_rest_days"] = df["home_rest_days"].fillna(7.0).astype(float)
    df["away_rest_days"] = df["away_rest_days"].fillna(7.0).astype(float)

    return df


def _poisson_outcome_probs(mu_home: float, mu_away: float, max_goals: int = 6) -> Tuple[float, float, float]:
    """
    Turn expected home/away goals (mu_home, mu_away) into
    1X2 probabilities using a simple Poisson model.

    Returns:
        (p_home_win, p_draw, p_away_win)
    """
    # Avoid weird negative or zero values
    mu_home = max(mu_home, 0.0001)
    mu_away = max(mu_away, 0.0001)

    def pois(mu: float, k: int) -> float:
        try:
            return (mu ** k) * math.exp(-mu) / math.factorial(k)
        except OverflowError:
            return 0.0

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    # Sum probabilities up to max_goals goals for each team
    for hg in range(0, max_goals + 1):
        for ag in range(0, max_goals + 1):
            p = pois(mu_home, hg) * pois(mu_away, ag)
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away
    if total <= 0:
        # fallback to something reasonable
        return 1.0 / 3, 1.0 / 3, 1.0 / 3

    return p_home / total, p_draw / total, p_away / total


def _multiclass_logloss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    """
    Log loss for 3-class (home/draw/away) probabilities.

    y_true: shape (n,) with labels in {0, 1, 2}
            0 = home win, 1 = draw, 2 = away win
    proba: shape (n, 3) with probabilities [p_home, p_draw, p_away]
    """
    proba = np.clip(proba, eps, 1.0 - eps)
    # pick the probability of the correct class
    logp = np.log(proba[np.arange(len(y_true)), y_true])
    return float(-np.mean(logp))


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Multi-class Brier score.
    Lower is better. 0 would be perfect predictions.

    y_true: shape (n,) with labels in {0, 1, 2}
    proba: shape (n, 3) with probabilities [p_home, p_draw, p_away]
    """
    n = len(y_true)
    one_hot = np.zeros_like(proba)
    # make a one-hot encoding for the true class
    one_hot[np.arange(n), y_true] = 1.0
    # mean squared error between predicted probs and one-hot targets
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def train_model(league_id: int, seasons: List[int]) -> Dict[str, Any]:
    """
    Train + evaluate model with a time-based split.

    - Build full training frame for given league + seasons.
    - Sort by date and split (80% train, 20% test).
    - Train RandomForest on train set only.
    - Evaluate 1X2 probabilities on test set via Poisson.
    - Store metrics and model metadata.
    """
    # Build the big training dataframe + meta info
    df, meta = build_training_frame(league_id, seasons)

    # --- Make sure "date" is a proper datetime and sorted in time ---
    df["date_dt"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)

    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]

    X_all = df[feature_cols].astype(float).values
    y_all = df[target_cols].astype(float).values

    n_samples = X_all.shape[0]
    if n_samples < 50:
        # tiny safety check – prevents nonsense splits when data is too small
        raise HTTPException(status_code=400, detail=f"Not enough samples ({n_samples}) to train a robust model.")

    # --- Time-based split: first 80% → train, last 20% → test ---
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    logger.info(
        "[TRAIN] league=%s samples_total=%s train=%s test=%s features=%s targets=%s",
        league_id, n_samples, X_train.shape[0], X_test.shape[0],
        len(feature_cols), len(target_cols),
    )

        # --- Train the RandomForest model on the training part only ---
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=140,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        )
    )
    model.fit(X_train, y_train)

        # --- Evaluate on the holdout (test) part: build 1X2 probabilities ---
    y_pred_test = model.predict(X_test)

    y_true_outcome: List[int] = []
    proba_list: List[List[float]] = []

    for (true_home, true_away), (mu_home, mu_away) in zip(y_test, y_pred_test):
        # True label: 0 = home win, 1 = draw, 2 = away win
        if true_home > true_away:
            label = 0
        elif true_home == true_away:
            label = 1
        else:
            label = 2
        y_true_outcome.append(label)

        # Turn predicted goals into 1X2 probabilities using Poisson
        p_home, p_draw, p_away = _poisson_outcome_probs(float(mu_home), float(mu_away))
        proba_list.append([p_home, p_draw, p_away])

    y_true_outcome_arr = np.array(y_true_outcome, dtype=int)
    proba_arr = np.array(proba_list, dtype=float)

    # --- Hit-rate metrics on the holdout set ---
    if len(y_true_outcome_arr) > 0:
        # model's most likely outcome (0/1/2) for each match
        top_idx = np.argmax(proba_arr, axis=1)
        # actual hit rate: how often top pick is correct
        hit_rate_actual = float(np.mean(top_idx == y_true_outcome_arr))
        # expected hit rate: average of max predicted probability
        hit_rate_expected = float(np.max(proba_arr, axis=1).mean())

        # baseline: always predict the most common outcome in the test set
        counts = np.bincount(y_true_outcome_arr, minlength=3)
        baseline_class = int(np.argmax(counts))
        baseline_hit_rate = float(np.mean(y_true_outcome_arr == baseline_class))

        # edge vs this simple baseline (in absolute probability terms)
        edge_vs_baseline = float(hit_rate_actual - baseline_hit_rate)
    else:
        hit_rate_actual = float("nan")
        hit_rate_expected = float("nan")
        baseline_hit_rate = float("nan")
        edge_vs_baseline = float("nan")

        # --- Market baseline using implied odds (if available) ---
    try:
        market_hit_rate = float("nan")
        edge_vs_market = float("nan")

        # feature_cols, X_test, y_true_outcome_arr are already defined earlier in train_model
        if (
            "home_odd_implied" in feature_cols
            and "draw_odd_implied" in feature_cols
            and "away_odd_implied" in feature_cols
            and X_test is not None
            and X_test.shape[0] > 0
        ):
            idx_home_odd = feature_cols.index("home_odd_implied")
            idx_draw_odd = feature_cols.index("draw_odd_implied")
            idx_away_odd = feature_cols.index("away_odd_implied")

            # bookmaker implied probabilities from the test set
            market_probs = X_test[:, [idx_home_odd, idx_draw_odd, idx_away_odd]]

            # make sure they look like valid probabilities
            market_probs = np.clip(market_probs, 1e-6, 1.0)
            row_sums = market_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0  # avoid divide-by-zero
            market_probs = market_probs / row_sums

            # market's top-pick outcome (0=home, 1=draw, 2=away)
            market_top = np.argmax(market_probs, axis=1)
            market_hit_rate = float(np.mean(market_top == y_true_outcome_arr))

            if not math.isnan(hit_rate_actual):
                edge_vs_market = float(hit_rate_actual - market_hit_rate)
    except Exception as e:
        logger.warning("[TRAIN] market baseline calculation failed: %s", e)
        market_hit_rate = float("nan")
        edge_vs_market = float("nan")


        # --- Global logloss + Brier ---
    logloss_1x2 = _multiclass_logloss(y_true_outcome_arr, proba_arr)
    brier_1x2 = _multiclass_brier(y_true_outcome_arr, proba_arr)

    # --- Calibration curves for home / draw / away ---
    calibration_data = None
    try:
        calib: Dict[str, Dict[str, List[float]]] = {}
        labels_names = ["home", "draw", "away"]

        for idx, name in enumerate(labels_names):
            # binary target: did this outcome actually happen?
            y_bin = (y_true_outcome_arr == idx).astype(int)
            # predicted probability for this outcome
            p_hat = proba_arr[:, idx]

            frac_pos, mean_pred = calibration_curve(
                y_bin,
                p_hat,
                n_bins=10,
                strategy="uniform",
            )

            calib[name] = {
                "predicted": mean_pred.tolist(),  # x-axis
                "observed": frac_pos.tolist(),     # y-axis
            }

        calibration_data = calib
    except Exception as e:
        logger.warning("[TRAIN] calibration_curve failed: %s", e)

    # --- Helper: make floats JSON-safe (no NaN/Inf) ---
    def _safe_float(val):
        if val is None:
            return None
        if isinstance(val, (float, np.floating)):
            v = float(val)
            return v if math.isfinite(v) else None
        # ints or other numeric types
        try:
            return float(val)
        except Exception:
            return None

    metrics = {
        "samples_total": int(n_samples),
        "samples_train": int(X_train.shape[0]),
        "samples_test": int(X_test.shape[0]),
        "logloss_1x2": _safe_float(round(logloss_1x2, 5)),
        "brier_1x2": _safe_float(round(brier_1x2, 5)),
        "hit_rate_expected": _safe_float(round(hit_rate_expected, 4)),
        "hit_rate_actual": _safe_float(round(hit_rate_actual, 4)),
        "baseline_hit_rate": _safe_float(round(baseline_hit_rate, 4)),
        "edge_vs_baseline": _safe_float(round(edge_vs_baseline, 4)),
        "market_hit_rate": _safe_float(round(market_hit_rate, 4)),
        "edge_vs_market": _safe_float(round(edge_vs_market, 4)),
        "calibration": calibration_data,
    }


    # attach metrics to meta so you can inspect them later if needed
    meta["metrics"] = metrics

     # Save when this model was trained (UTC time)
    meta["trained_at"] = datetime.now(timezone.utc).isoformat()

    # Persist model + meta (including metrics)
    save_model_and_meta(league_id, model, meta)

    # This is what the /train API will return in "info"
    return {
        "league": league_id,
        "seasons": seasons,
        "features": feature_cols,
        "targets": target_cols,
        "metrics": metrics,
    }

# ============================================================
# PREDICTIONS
# ============================================================

# Small in-memory cache so we don't refetch history for every call
_live_history_cache: Dict[Tuple[int, Tuple[int, ...]], pd.DataFrame] = {}


def get_history_df_for_meta(meta: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Load (and cache) historical fixtures for this league + seasons,
    so we can compute live form / rest for upcoming fixtures.
    """
    league_id = int(meta.get("league_id", 0) or 0)
    seasons = meta.get("seasons") or []
    if not league_id or not seasons:
        return None

    key = (league_id, tuple(int(s) for s in seasons))
    if key in _live_history_cache:
        return _live_history_cache[key]

    try:
        # Reuse the same function you already use during training
        df_hist = fetch_historic_fixtures(league_id, list(seasons))
    except HTTPException as exc:
        logger.warning(
            "[LIVE FORM] could not fetch historic fixtures for league=%s seasons=%s: %s",
            league_id,
            seasons,
            getattr(exc, "detail", exc),
        )
        return None
    except Exception as exc:
        logger.warning(
            "[LIVE FORM] error fetching historic fixtures for league=%s seasons=%s: %s",
            league_id,
            seasons,
            exc,
        )
        return None

    # Ensure we have a datetime column
    df_hist["date_dt"] = pd.to_datetime(df_hist["date"], utc=True, errors="coerce")

    _live_history_cache[key] = df_hist
    return df_hist


def compute_live_team_features(
    team_id: int,
    fixture_dt: datetime,
    history_df: pd.DataFrame,
    league_gf_per_match: float,
) -> Dict[str, float]:
    """
    Compute form (GF/GA/points), schedule congestion, and rest-days
    for a single team, up to (but not including) fixture_dt.
    """
    # All past games where this team played
    team_df = history_df[
        ((history_df["home_id"] == team_id) | (history_df["away_id"] == team_id))
        & (history_df["date_dt"] < fixture_dt)
    ].sort_values("date_dt")

    if team_df.empty:
        # Very early in season or new team: neutral defaults
        return {
            "form_gf": league_gf_per_match,
            "form_ga": league_gf_per_match,
            "form_pts": 1.0,          # neutral average points
            "matches_last_7": 0.0,
            "matches_last_14": 0.0,
            "rest_days": 7.0,
        }

    # ---- Last-5 goal form + points ----
    last5 = team_df.tail(5)
    gf_list: List[float] = []
    ga_list: List[float] = []
    pts_list: List[float] = []

    for _, r in last5.iterrows():
        if r["home_id"] == team_id:
            gf = float(r["home_goals"])
            ga = float(r["away_goals"])
        else:
            gf = float(r["away_goals"])
            ga = float(r["home_goals"])

        gf_list.append(gf)
        ga_list.append(ga)

        # Points: win=3, draw=1, loss=0 (same as training)
        if gf > ga:
            pts_list.append(3.0)
        elif gf == ga:
            pts_list.append(1.0)
        else:
            pts_list.append(0.0)

    form_gf = float(np.mean(gf_list)) if gf_list else league_gf_per_match
    form_ga = float(np.mean(ga_list)) if ga_list else league_gf_per_match
    # Same scale as training: 0..3, neutral ~1
    form_pts = float(np.mean(pts_list)) if pts_list else 1.0

    # ---- Schedule congestion: matches in last 7 / 14 days ----
    window7 = fixture_dt - timedelta(days=7)
    window14 = fixture_dt - timedelta(days=14)

    recent7 = team_df[team_df["date_dt"] >= window7]
    recent14 = team_df[team_df["date_dt"] >= window14]

    matches_last_7 = float(len(recent7))
    matches_last_14 = float(len(recent14))

    # ---- Rest days: days since last match ----
    last_match_date = team_df["date_dt"].iloc[-1]
    if pd.isna(last_match_date):
        rest_days = 7.0
    else:
        diff = fixture_dt - last_match_date
        rest_days = float(diff.days) if diff.days >= 0 else 7.0

    return {
        "form_gf": form_gf,
        "form_ga": form_ga,
        "form_pts": form_pts,
        "matches_last_7": matches_last_7,
        "matches_last_14": matches_last_14,
        "rest_days": rest_days,
    }


def make_feature_row_for_fixture(fx: Dict[str, Any], meta: Dict[str, Any]) -> np.ndarray:
    """
    Build a feature row for a single upcoming fixture, using the same columns
    as used during training (meta["feature_cols"]).

    This version:
    - uses team strengths (attack/defence, rating)
    - uses Elo ratings from training (meta["elo_ratings"])
    - computes *live* form, schedule congestion, and rest-days from past fixtures
    - uses realistic shots & possession proxies based on team stats
    """
    # Look up team index and strength tables from meta
    team_index = {str(k): int(v) for k, v in meta["team_index"].items()}
    attack_strength = {int(k): float(v) for k, v in meta["attack_strength"].items()}
    defense_strength = {int(k): float(v) for k, v in meta["defense_strength"].items()}
    team_summary = meta.get("team_summary", {}) or {}

    # Elo ratings (final values from training)
    elo_ratings_raw = meta.get("elo_ratings") or {}
    elo_ratings = {int(k): float(v) for k, v in elo_ratings_raw.items()}
    initial_elo = 1500.0

    # League-level averages
    league_gf_per_match = float(meta.get("league_gf_per_match", 1.3))
    league_avg_shots = float(meta.get("league_avg_shots_proxy", 10.0))

    # Build helper maps from team_summary for shots / possession proxies
    gf_per_match_map: Dict[int, float] = {}
    rating_map: Dict[int, float] = {}

    for key, info in team_summary.items():
        # key might be the team ID or a string
        try:
            tid = int(info.get("team_id", key))
        except Exception:
            try:
                tid = int(key)
            except Exception:
                continue

        matches = float(info.get("matches", 0.0) or 0.0)
        gf = float(info.get("gf", 0.0) or 0.0)
        rating = float(info.get("rating", 1.0) or 1.0)

        if matches > 0:
            gf_per_match_map[tid] = gf / matches
        else:
            gf_per_match_map[tid] = league_gf_per_match
        rating_map[tid] = rating

    # Try to load historical fixtures so we can compute live form/rest
    history_df = get_history_df_for_meta(meta)

    # Parse fixture date (UTC)
    fixture_info = fx.get("fixture", {}) or {}
    fixture_date_str = fixture_info.get("date")
    fixture_dt: Optional[datetime] = None
    if fixture_date_str:
        try:
            fixture_dt = pd.to_datetime(fixture_date_str, utc=True, errors="coerce")
        except Exception:
            fixture_dt = None

    # IDs of the teams in this fixture
    home_id = fx["teams"]["home"]["id"]
    away_id = fx["teams"]["away"]["id"]

    # Make sure we have entries for these teams (in case of new promotions etc.)
    def ensure_team(tid: int) -> None:
        tid_str = str(tid)
        if tid_str not in team_index:
            team_index[tid_str] = max(team_index.values(), default=0) + 1
        if tid not in attack_strength:
            attack_strength[tid] = 1.0
        if tid not in defense_strength:
            defense_strength[tid] = 1.0

    ensure_team(home_id)
    ensure_team(away_id)

    # Basic numeric values for this fixture
    home_idx = float(team_index[str(home_id)])
    away_idx = float(team_index[str(away_id)])
    home_att = float(attack_strength[home_id])
    home_def = float(defense_strength[home_id])
    away_att = float(attack_strength[away_id])
    away_def = float(defense_strength[away_id])

    # Elo ratings for this fixture (we just use the final ratings from training)
    home_elo = float(elo_ratings.get(home_id, initial_elo))
    away_elo = float(elo_ratings.get(away_id, initial_elo))

    # ---- Live form & schedule features ----
    # Defaults (if we can't compute from history)
    home_form_gf = league_gf_per_match
    home_form_ga = league_gf_per_match
    away_form_gf = league_gf_per_match
    away_form_ga = league_gf_per_match
    home_form_pts = 1.0
    away_form_pts = 1.0
    home_matches_last_7 = 0.0
    home_matches_last_14 = 0.0
    away_matches_last_7 = 0.0
    away_matches_last_14 = 0.0
    home_rest_days = 7.0
    away_rest_days = 7.0

    if history_df is not None and fixture_dt is not None and not pd.isna(fixture_dt):
        home_stats = compute_live_team_features(home_id, fixture_dt, history_df, league_gf_per_match)
        away_stats = compute_live_team_features(away_id, fixture_dt, history_df, league_gf_per_match)

        home_form_gf = home_stats["form_gf"]
        home_form_ga = home_stats["form_ga"]
        home_form_pts = home_stats["form_pts"]
        home_matches_last_7 = home_stats["matches_last_7"]
        home_matches_last_14 = home_stats["matches_last_14"]
        home_rest_days = home_stats["rest_days"]

        away_form_gf = away_stats["form_gf"]
        away_form_ga = away_stats["form_ga"]
        away_form_pts = away_stats["form_pts"]
        away_matches_last_7 = away_stats["matches_last_7"]
        away_matches_last_14 = away_stats["matches_last_14"]
        away_rest_days = away_stats["rest_days"]

    # ---- Shots & possession proxies ----
    home_gf_pm = gf_per_match_map.get(home_id, league_gf_per_match)
    away_gf_pm = gf_per_match_map.get(away_id, league_gf_per_match)

    home_shots_proxy = home_gf_pm * 3.5 if home_gf_pm > 0 else league_avg_shots
    away_shots_proxy = away_gf_pm * 3.5 if away_gf_pm > 0 else league_avg_shots

    rh = float(rating_map.get(home_id, 1.0))
    ra = float(rating_map.get(away_id, 1.0))
    total_rating = rh + ra
    if total_rating <= 0:
        home_possession_proxy = 0.5
        away_possession_proxy = 0.5
    else:
        home_possession_proxy = rh / total_rating
        away_possession_proxy = 1.0 - home_possession_proxy

    # Build feature dict with all required columns
    feat: Dict[str, float] = {}
    for c in meta["feature_cols"]:
        # --- core team / rating features ---
        if c == "home_team_idx":
            feat[c] = home_idx
        elif c == "away_team_idx":
            feat[c] = away_idx
        elif c == "home_advantage":
            feat[c] = 1.0
        elif c == "home_att_str":
            feat[c] = home_att
        elif c == "home_def_str":
            feat[c] = home_def
        elif c == "away_att_str":
            feat[c] = away_att
        elif c == "away_def_str":
            feat[c] = away_def

        # --- Elo features ---
        elif c == "home_elo":
            feat[c] = home_elo
        elif c == "away_elo":
            feat[c] = away_elo

        # --- form features (live if available, otherwise defaults) ---
        elif c == "home_form_gf":
            feat[c] = home_form_gf
        elif c == "home_form_ga":
            feat[c] = home_form_ga
        elif c == "away_form_gf":
            feat[c] = away_form_gf
        elif c == "away_form_ga":
            feat[c] = away_form_ga
        elif c == "home_form_pts":
            feat[c] = home_form_pts
        elif c == "away_form_pts":
            feat[c] = away_form_pts

        # --- schedule congestion ---
        elif c == "home_matches_last_7":
            feat[c] = home_matches_last_7
        elif c == "home_matches_last_14":
            feat[c] = home_matches_last_14
        elif c == "away_matches_last_7":
            feat[c] = away_matches_last_7
        elif c == "away_matches_last_14":
            feat[c] = away_matches_last_14

        # --- rest days ---
        elif c == "home_rest_days":
            feat[c] = home_rest_days
        elif c == "away_rest_days":
            feat[c] = away_rest_days

        # --- proxy stats: shots & possession ---
        elif c == "home_shots_proxy":
            feat[c] = home_shots_proxy
        elif c == "away_shots_proxy":
            feat[c] = away_shots_proxy
        elif c == "home_possession_proxy":
            feat[c] = home_possession_proxy
        elif c == "away_possession_proxy":
            feat[c] = away_possession_proxy

        # --- anything else not explicitly handled ---
        else:
            # Any extra feature we don't explicitly handle → default to 0
            feat[c] = 0.0

    # Turn dict into a 2D numpy array in the correct column order
    x = np.array([[feat[c] for c in meta["feature_cols"]]], dtype=float)
    return x



def derive_extra_stats(home_goals: float, away_goals: float) -> Dict[str, float]:
    hg = float(home_goals); ag = float(away_goals)
    home_sot = max(2.0, hg * 3.0); away_sot = max(2.0, ag * 3.0)
    home_corners = max(3.0, 4.0 + hg * 2.0); away_corners = max(3.0, 4.0 + ag * 2.0)
    home_yellows = 1.5 + 0.4 * hg; away_yellows = 1.5 + 0.4 * ag
    home_reds = 0.05 + 0.03 * max(0.0, hg - ag); away_reds = 0.05 + 0.03 * max(0.0, ag - hg)
    return {
        "home_sot": round(home_sot, 2),
        "away_sot": round(away_sot, 2),
        "home_corners": round(home_corners, 2),
        "away_corners": round(away_corners, 2),
        "home_yellows": round(home_yellows, 2),
        "away_yellows": round(away_yellows, 2),
        "home_reds": round(home_reds, 2),
        "away_reds": round(away_reds, 2),
    }

def build_reasoning_for_prediction(pred: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """
    Build a short natural-language explanation for a single match prediction.

    Works with either:
    - nested 'predictions' dict (home_goals, home_win_p, etc.), or
    - flat keys like 'pred_home_goals', 'prob_home_win' (fallback).
    """
    try:
        home = pred.get("home_name", "Home")
        away = pred.get("away_name", "Away")

        preds = pred.get("predictions") or {}

        # Probabilities: try nested first, then flat keys as fallback
        ph = float(preds.get("home_win_p", pred.get("prob_home_win", 0.0)))
        pd = float(preds.get("draw_p", pred.get("prob_draw", 0.0)))
        pa = float(preds.get("away_win_p", pred.get("prob_away_win", 0.0)))

        # Goals: try nested first, then flat keys as fallback
        xh = float(preds.get("home_goals", pred.get("pred_home_goals", 0.0)))
        xa = float(preds.get("away_goals", pred.get("pred_away_goals", 0.0)))

        home_id = pred.get("home_id")
        away_id = pred.get("away_id")

        att_raw = meta.get("attack_strength", {}) or {}
        def_raw = meta.get("defense_strength", {}) or {}

        def _get_team_rating(d: Dict[str, Any], tid: Any) -> float:
            if tid is None:
                return 1.0
            return float(d.get(str(tid), 1.0))

        home_att = _get_team_rating(att_raw, home_id)
        away_att = _get_team_rating(att_raw, away_id)
        home_def = _get_team_rating(def_raw, home_id)
        away_def = _get_team_rating(def_raw, away_id)

        parts: List[str] = []

        # Who is favourite?
        probs = {"home": ph, "draw": pd, "away": pa}
        fav_side = max(probs, key=lambda k: probs[k])
        fav_prob = probs[fav_side]
        sorted_probs = sorted(probs.values(), reverse=True)
        second_prob = sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
        margin = fav_prob - second_prob

        # Base sentence
        if fav_side == "home":
            if margin >= 0.15:
                parts.append(f"Model sees {home} as a clear favourite at home against {away}.")
            elif margin >= 0.07:
                parts.append(f"Model slightly favours {home} at home against {away}.")
            else:
                parts.append(f"Model sees {home} vs {away} as fairly balanced with a small home edge.")
        elif fav_side == "away":
            if margin >= 0.15:
                parts.append(f"Model sees {away} as a strong away favourite against {home}.")
            elif margin >= 0.07:
                parts.append(f"Model gives a small edge to {away} away to {home}.")
            else:
                parts.append(f"Model sees {home} vs {away} as quite balanced with a tiny edge to the away side.")
        else:  # draw favourite (rare)
            parts.append(f"Model expects a very tight match between {home} and {away}, with a high chance of a draw.")

        # Expected/predicted goals
        parts.append(f"It expects roughly {xh:.2f} : {xa:.2f} goals.")

        # Attack strength comparison
        att_diff = home_att - away_att
        if att_diff >= 0.2:
            parts.append(f"{home} has a stronger attacking rating than {away}.")
        elif att_diff <= -0.2:
            parts.append(f"{away} has a stronger attacking rating than {home}.")

        # Defence strength comparison
        def_diff = away_def - home_def  # how much better away defence is vs home
        if def_diff >= 0.2:
            parts.append(f"{away}'s defence looks stronger on the numbers.")
        elif def_diff <= -0.2:
            parts.append(f"{home}'s defence looks stronger on the numbers.")

        # Draw probability commentary
        if pd >= 0.28:
            parts.append("The model also gives a decent chance of a draw.")

        # Generic nod to form & schedule (which are baked into the model features)
        parts.append("These ratings reflect team strength, recent performances and schedule intensity.")

        return " ".join(parts)
    except Exception:
        # Fallback in case anything goes wrong
        return "Model generated this prediction based on team strength, recent results and schedule data."

def build_predictions_for_fixtures_old(
    fixtures: List[Dict[str, Any]],
    model: Any,
    meta: Dict[str, Any],
    league: int,
    season: int,
    window_start: datetime,
    window_end: datetime
) -> List[Dict[str, Any]]:
    # Keep backward compat but avoid duplicated logic drifting.
    return build_predictions_for_fixtures(
        fixtures=fixtures,
        model=model,
        meta=meta,
        league=league,
        season=season,
        window_start=window_start,
        window_end=window_end,
    )

def build_players_to_score_for_fixture(fixture: Dict[str, Any], league: int, season: int) -> List[Dict[str, Any]]:
    # minimal-safe (unchanged from your version)
    try:
        top = fetch_top_scorers(league, season)
    except HTTPException as e:
        logger.warning("Topscorers fetch failed: %s", e.detail)
        return []
    except Exception as e:
        logger.warning("Topscorers fetch failed: %s", e)
        return []
    teams = fixture.get("teams", {}) or {}
    home_team = teams.get("home", {}) or {}
    away_team = teams.get("away", {}) or {}
    home_id = home_team.get("id"); away_id = away_team.get("id")
    if home_id is None or away_id is None:
        return []
    home_players = [p for p in top if p.get("team_id") == home_id]
    away_players = [p for p in top if p.get("team_id") == away_id]
    home_players.sort(key=lambda x: x.get("goals", 0), reverse=True)
    away_players.sort(key=lambda x: x.get("goals", 0), reverse=True)

    selected: List[Dict[str, Any]] = []
    def estimate_xg_anytime(goals: float, apps: float, rank: int) -> float:
        gpg = goals if not apps else goals / apps
        base = 0.15 + 0.7 * min(1.5, gpg)
        if rank > 1: base *= 0.9
        if rank > 2: base *= 0.85
        return float(max(0.15, min(0.85, base)))

    for rank, p in enumerate(home_players[:2], start=1):
        goals = float(p.get("goals", 0) or 0.0); apps = float(p.get("appearances", 0) or 0.0)
        selected.append({"name": p.get("name"), "team": p.get("team_name") or home_team.get("name"),
                         "xg_anytime": round(estimate_xg_anytime(goals, apps, rank), 3), "photo": p.get("photo")})
    for rank, p in enumerate(away_players[:2], start=1):
        goals = float(p.get("goals", 0) or 0.0); apps = float(p.get("appearances", 0) or 0.0)
        selected.append({"name": p.get("name"), "team": p.get("team_name") or away_team.get("name"),
                         "xg_anytime": round(estimate_xg_anytime(goals, apps, rank), 3), "photo": p.get("photo")})
    return selected

def decimal_to_implied_prob(odd: float) -> float:
    """
    Convert decimal odds (e.g. 2.10) to implied probability (e.g. ~0.476).
    If odds are invalid, return 0.0.
    """
    try:
        odd = float(odd)
    except Exception:
        return 0.0
    if odd <= 1.0:
        return 0.0
    return 1.0 / odd


def extract_match_winner_odds(odds_payload: dict):
    """Backward-compatible: returns decimal odds dict {'home','draw','away'} or None."""
    odds, _meta = extract_market_odds_1x2_with_meta(odds_payload)
    return odds
def odds_to_implied_probs_1x2(odds: dict) -> Optional[Dict[str, float]]:
    """Convert decimal 1X2 odds into normalized implied probabilities.

    Returns dict {home, draw, away} or None if odds missing/invalid.
    """
    try:
        if not isinstance(odds, dict):
            return None
        oh = odds.get("home")
        od = odds.get("draw")
        oa = odds.get("away")
        if oh is None or od is None or oa is None:
            return None
        oh = float(oh); od = float(od); oa = float(oa)
        if oh <= 0 or od <= 0 or oa <= 0:
            return None
        ph = 1.0 / oh
        pd = 1.0 / od
        pa = 1.0 / oa
        s = ph + pd + pa
        if s <= 0:
            return None
        return {"home": ph / s, "draw": pd / s, "away": pa / s}
    except Exception:
        return None


def odds_to_probs_1x2(odds_home, odds_draw, odds_away):
    # Decimal odds -> implied probs, then normalize to remove overround
    try:
        oh = float(odds_home); od = float(odds_draw); oa = float(odds_away)
        if oh <= 1e-9 or od <= 1e-9 or oa <= 1e-9:
            return None
    except Exception:
        return None

    ph = 1.0 / oh
    pd = 1.0 / od
    pa = 1.0 / oa
    s = ph + pd + pa
    if s <= 0:
        return None
    return {"home": ph / s, "draw": pd / s, "away": pa / s}


def extract_market_probs_from_api_football_odds(payload: dict):
    """
    API-Football /odds response can vary a lot.
    We scan all bookmakers/bets and accept the first COMPLETE 1X2 set we can turn into
    normalized implied probabilities.

    IMPORTANT: Do not stop scanning if a "complete" set has invalid/missing odds values.
    """
    resp = (payload or {}).get("response") or []
    if not resp:
        return None

    for item in resp:
        bms = item.get("bookmakers") or []
        for bm in bms:
            bets = bm.get("bets") or []
            for bet in bets:
                vals = bet.get("values") or []
                got = {"home": None, "draw": None, "away": None}
                for v in vals:
                    lab = (v.get("value") or "").strip().lower()
                    odd = v.get("odd")
                    if lab in ("home", "1"):
                        got["home"] = odd
                    elif lab in ("draw", "x"):
                        got["draw"] = odd
                    elif lab in ("away", "2"):
                        got["away"] = odd

                if got["home"] and got["draw"] and got["away"]:
                    probs = odds_to_probs_1x2(got["home"], got["draw"], got["away"])
                    if probs:
                        return probs
                    # else: keep scanning

    return None

def scan_market_odds_1x2(payload: dict, *, max_notes: int = 50) -> dict:
    """
    Scan an API-Football /odds payload and try to extract a complete 1X2 (home/draw/away) set.
    Returns a debug-friendly dict explaining what was (and wasn't) found.
    """
    resp = (payload or {}).get("response") or []
    if not resp:
        return {
            "found": False,
            "reason": "no_response",
            "selected": None,
            "stats": {"items": 0, "bookmakers": 0, "bets": 0, "preferred_bets": 0},
            "notes": [],
        }

    def _norm(s) -> str:
        return (s or "").strip().lower()

    def _is_preferred_bet(bet: dict) -> bool:
        name = _norm(bet.get("name"))
        bet_id = bet.get("id")
        if str(bet_id).strip() == "1":  # API-Football commonly uses id=1 for Match Winner
            return True
        return any(k in name for k in ["match winner", "1x2", "full time result", "fulltime result", "result", "winner"])

    def _map_values(vals: list, home_name: str, away_name: str) -> dict:
        got = {"home": None, "draw": None, "away": None}
        h = _norm(home_name)
        a = _norm(away_name)
        for v in vals or []:
            lab = _norm(v.get("value"))
            odd = v.get("odd")
            if lab in ("home", "1"):
                got["home"] = odd
            elif lab in ("draw", "x", "tie"):
                got["draw"] = odd
            elif lab in ("away", "2"):
                got["away"] = odd
            else:
                # Sometimes labels are team names
                if h and (lab == h or (h in lab and got["home"] is None)):
                    got["home"] = odd
                elif a and (lab == a or (a in lab and got["away"] is None)):
                    got["away"] = odd
        return got

    notes = []
    missing_counts = {"home": 0, "draw": 0, "away": 0, "invalid_odds": 0}
    stats = {"items": 0, "bookmakers": 0, "bets": 0, "preferred_bets": 0}

    # Two-pass scan: prefer obvious 1X2 / Match Winner bets, then fall back to any bet.
    for pass_idx in (0, 1):
        preferred_only = (pass_idx == 0)

        for item in resp:
            stats["items"] += 1
            teams = item.get("teams") or {}
            home_name = ((teams.get("home") or {}).get("name")) or ""
            away_name = ((teams.get("away") or {}).get("name")) or ""

            bms = item.get("bookmakers") or []
            if not bms:
                if len(notes) < max_notes:
                    notes.append({"reason": "no_bookmakers", "context": {"item_fixture": (item.get("fixture") or {}).get("id")}})
                continue

            for bm in bms:
                stats["bookmakers"] += 1
                bm_name = bm.get("name") or bm.get("id") or "unknown"
                bets = bm.get("bets") or []
                if not bets:
                    if len(notes) < max_notes:
                        notes.append({"reason": "no_bets", "context": {"bookmaker": bm_name}})
                    continue

                for bet in bets:
                    stats["bets"] += 1
                    bet_name = bet.get("name") or bet.get("id") or "unknown"
                    preferred = _is_preferred_bet(bet)
                    if preferred:
                        stats["preferred_bets"] += 1
                    if preferred_only and not preferred:
                        continue

                    got = _map_values(bet.get("values") or [], home_name, away_name)

                    missing = [k for k in ("home", "draw", "away") if not got.get(k)]
                    if missing:
                        for k in missing:
                            missing_counts[k] += 1
                        if len(notes) < max_notes:
                            notes.append({"reason": "incomplete_1x2", "missing": missing, "context": {"bookmaker": bm_name, "bet": bet_name}})
                        continue

                    probs = odds_to_probs_1x2(got["home"], got["draw"], got["away"])
                    if not probs:
                        missing_counts["invalid_odds"] += 1
                        if len(notes) < max_notes:
                            notes.append({"reason": "invalid_odds", "context": {"bookmaker": bm_name, "bet": bet_name, "odds": got}})
                        continue

                    return {
                        "found": True,
                        "reason": "ok",
                        "selected": {
                            "odds": {"home": float(got["home"]), "draw": float(got["draw"]), "away": float(got["away"])},
                            "probs": probs,
                            "meta": {"bookmaker": str(bm_name), "bet_name": str(bet_name)},
                            "pass": "preferred" if preferred_only else "fallback",
                        },
                        "stats": stats,
                        "missing_counts": missing_counts,
                        "notes": notes,
                    }

    # Nothing found
    reason = "no_1x2_bet_found" if stats.get("preferred_bets", 0) == 0 else "no_complete_1x2_set"
    # A more specific primary reason, if obvious
    if stats.get("bookmakers", 0) == 0:
        reason = "no_bookmakers"
    elif stats.get("bets", 0) == 0:
        reason = "no_bets"
    elif missing_counts["draw"] > 0 and missing_counts["draw"] >= missing_counts["home"] and missing_counts["draw"] >= missing_counts["away"]:
        reason = "missing_draw"
    elif missing_counts["home"] > 0 and missing_counts["home"] >= missing_counts["draw"] and missing_counts["home"] >= missing_counts["away"]:
        reason = "missing_home"
    elif missing_counts["away"] > 0 and missing_counts["away"] >= missing_counts["draw"] and missing_counts["away"] >= missing_counts["home"]:
        reason = "missing_away"

    return {
        "found": False,
        "reason": reason,
        "selected": None,
        "stats": stats,
        "missing_counts": missing_counts,
        "notes": notes,
    }


def extract_market_odds_1x2_with_meta(payload: dict):
    """
    Backwards-compatible wrapper around scan_market_odds_1x2().
    Returns (odds_dict, meta_dict) or (None, None).
    """
    scan = scan_market_odds_1x2(payload, max_notes=0)
    if not scan.get("found"):
        return None, None
    sel = scan.get("selected") or {}
    return sel.get("odds"), sel.get("meta")

def fetch_1x2_odds_for_fixture(fixture_id: int, return_meta: bool = False):
    """Fetch odds from API-Football and extract a complete 1X2 set (robust across bookmakers)."""
    try:
        data = api_get("/odds", {"fixture": fixture_id})
    except Exception:
        return (None, None) if return_meta else None

    odds, meta = extract_market_odds_1x2_with_meta(data)
    if return_meta:
        return odds, (meta or {})
    return odds
import math

def poisson_1x2_probs(lam_home: float, lam_away: float, max_goals: int = 10) -> dict:
    """
    Convert expected goals (xG) -> 1X2 probabilities using independent Poissons.

    Returns: {"home": p_home_win, "draw": p_draw, "away": p_away_win}
    """
    # safety clamps
    try:
        lam_home = float(lam_home)
        lam_away = float(lam_away)
    except Exception:
        return {"home": None, "draw": None, "away": None}

    lam_home = max(0.01, lam_home)
    lam_away = max(0.01, lam_away)

    # Poisson PMF list up to max_goals
    def pmf_list(lam: float) -> list[float]:
        out = []
        e = math.exp(-lam)
        # k=0
        out.append(e)
        # k>=1 using recurrence to be fast/stable
        for k in range(1, max_goals + 1):
            out.append(out[-1] * lam / k)
        return out

    ph = pmf_list(lam_home)
    pa = pmf_list(lam_away)

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    # score matrix sums
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            pij = ph[i] * pa[j]
            if i > j:
                p_home += pij
            elif i == j:
                p_draw += pij
            else:
                p_away += pij

    # tiny renormalization (truncation error)
    s = p_home + p_draw + p_away
    if s > 0:
        p_home /= s
        p_draw /= s
        p_away /= s

    return {"home": p_home, "draw": p_draw, "away": p_away}

# ----------------------------
# 1X2 calibration helpers
# ----------------------------
_CAL_CACHE = {}  # {league: dict}

def load_1x2_calibration(league: int) -> dict:
    """
    Loads artifacts/calibration_<league>.json if present.
    Returns {} if missing/unreadable.
    Cached in-process so we don't hit disk per request.
    """
    global _CAL_CACHE
    if league in _CAL_CACHE:
        return _CAL_CACHE[league] or {}

    try:
        path = Path("artifacts") / f"calibration_{int(league)}.json"
        if not path.exists():
            _CAL_CACHE[league] = {}
            return {}
        cal = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cal, dict):
            cal = {}
        _CAL_CACHE[league] = cal
        return cal
    except Exception:
        _CAL_CACHE[league] = {}
        return {}

def apply_1x2_calibration(probs: dict, cal: dict) -> dict:
    """
    Applies:
      - temperature scaling (T)
      - draw multiplier (draw_mult)
    then renormalizes.

    probs expected keys: home/draw/away with probabilities.
    """
    if not isinstance(probs, dict):
        return probs

    p = {k: probs.get(k) for k in ("home", "draw", "away")}
    if not all(isinstance(p[k], (int, float)) for k in p):
        return probs

    T = float(cal.get("temperature", 1.0)) if isinstance(cal, dict) else 1.0
    dm = float(cal.get("draw_mult", 1.0)) if isinstance(cal, dict) else 1.0
    if T <= 0:
        T = 1.0
    if dm <= 0:
        dm = 1.0

    eps = 1e-12

    # temperature scaling in log space
    logits = {k: math.log(max(eps, float(v))) / T for k, v in p.items()}
    m = max(logits.values())
    ex = {k: math.exp(v - m) for k, v in logits.items()}
    s = sum(ex.values()) or 1.0
    pt = {k: ex[k] / s for k in ex}

    # draw multiplier then renormalize
    pt["draw"] *= dm
    s2 = sum(pt.values()) or 1.0
    pt = {k: pt[k] / s2 for k in pt}

    return pt


def _avg_logloss_1x2(samples: list[dict], cal: dict | None = None) -> float | None:
    """Average negative log likelihood for 1X2 samples.
    samples: [{"actual": "home|draw|away", "probs": {"home":..,"draw":..,"away":..}}, ...]
    If cal provided, apply_1x2_calibration() before scoring.
    """
    if not samples:
        return None
    eps = 1e-12
    s = 0.0
    n = 0
    for row in samples:
        try:
            actual = row.get("actual")
            probs = row.get("probs") or {}
            if cal:
                probs = apply_1x2_calibration(dict(probs), cal) or probs
            p_true = float((probs or {}).get(actual, 0.0))
            s += -math.log(max(eps, p_true))
            n += 1
        except Exception:
            continue
    return (s / n) if n else None


def fit_1x2_calibration(samples: list[dict]) -> dict:
    """Fit temperature + draw_mult by minimizing logloss on provided samples (grid search)."""
    # Guard to avoid overfitting on tiny sets
    if not samples or len(samples) < 30:
        return {"temperature": 1.0, "draw_mult": 1.0, "note": "insufficient_samples"}

    best = None
    best_ll = None

    # Coarse-but-effective grid (tighten later if you want).
    T_values = [round(x, 2) for x in [0.6 + 0.05*i for i in range(int((2.0-0.6)/0.05)+1)]]
    dm_values = [round(x, 2) for x in [0.6 + 0.05*i for i in range(int((1.6-0.6)/0.05)+1)]]

    for T in T_values:
        for dm in dm_values:
            cal = {"temperature": T, "draw_mult": dm}
            ll = _avg_logloss_1x2(samples, cal=cal)
            if ll is None:
                continue
            if (best_ll is None) or (ll < best_ll):
                best_ll = ll
                best = cal

    return best or {"temperature": 1.0, "draw_mult": 1.0}


def save_1x2_calibration(league: int, cal: dict) -> str:
    """Save calibration to artifacts/calibration_<league>.json and refresh in-process cache."""
    path = Path("artifacts")
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"calibration_{int(league)}.json"
    file_path.write_text(json.dumps(cal, indent=2, sort_keys=True), encoding="utf-8")

    # refresh cache
    global _CAL_CACHE
    _CAL_CACHE[int(league)] = dict(cal) if isinstance(cal, dict) else {}
    return str(file_path)


def compute_value_edges(model_probs: dict, odds: dict) -> dict:
    """
    Compute 3-way market implied probs, edges, and expected value (EV) for:
      - home, draw, away

    Returns:
      {
        "market_probs": {"home":..., "draw":..., "away":...},
        "edges": {"home":..., "draw":..., "away":...},   # model_p - market_p
        "evs": {"home":..., "draw":..., "away":...},     # model_p*odds - 1
        "best_side": "home|draw|away|None",
        "best_edge": <best EV> (kept for your current naming),
      }
    """
    sides = ("home", "draw", "away")

    # normalize inputs
    mp = {s: (model_probs.get(s) if model_probs else None) for s in sides}
    od = {s: (odds.get(s) if odds else None) for s in sides}

    # Only compute for sides with valid odds + probs
    valid = []
    for s in sides:
        p = mp[s]
        o = od[s]
        if p is None or o is None:
            continue
        try:
            p = float(p)
            o = float(o)
        except Exception:
            continue
        if p <= 0 or o <= 1e-9:
            continue
        mp[s] = p
        od[s] = o
        valid.append(s)

    if not valid:
        return {
            "market_probs": {"home": None, "draw": None, "away": None},
            "edges": {"home": None, "draw": None, "away": None},
            "evs": {"home": None, "draw": None, "away": None},
            "best_side": None,
            "best_edge": None,
        }

    # Market implied probabilities with overround normalization (3-way)
    inv_sum = sum(1.0 / od[s] for s in valid)
    market_probs = {s: (1.0 / od[s]) / inv_sum for s in sides}
    for s in sides:
        if s not in valid:
            market_probs[s] = None

    # Edges + EVs
    edges = {}
    evs = {}
    for s in sides:
        if s not in valid:
            edges[s] = None
            evs[s] = None
            continue
        edges[s] = mp[s] - market_probs[s]
        evs[s] = mp[s] * od[s] - 1.0

    # Pick best by EV (this naturally allows draw)
    best_side = max(valid, key=lambda s: evs[s])
    best_edge = evs[best_side]

    return {
        "market_probs": market_probs,
        "edges": edges,
        "evs": evs,
        "best_side": best_side,
        "best_edge": best_edge,
    }

   



def _attach_odds_and_value_fields_upcoming(
    fixtures: List[Dict[str, Any]],
    *,
    include_odds: int,
    odds_limit: int,
    min_edge: float,
) -> List[Dict[str, Any]]:
    """
    Attach odds_1x2 + best_edge/value_side onto fixture dicts (in-place).
    Safe: never raises; skips if odds/model probs missing.
    """
    try:
        fixtures = list(fixtures or [])
    except Exception:
        return fixtures or []

    if not include_odds:
        return fixtures

    out: List[Dict[str, Any]] = []
    fetched = 0

    for fx in fixtures:
        try:
            fxid = fx.get("fixture_id")
            if fxid and fetched < max(0, int(odds_limit)):
                # Don't refetch if already present
                odds = fx.get("odds_1x2") or fetch_1x2_odds_for_fixture(int(fxid))
                if odds:
                    fx["odds_1x2"] = odds

                    pred = fx.get("predictions") or {}
                    p_model = {
                        "home": pred.get("home_win_p"),
                        "draw": pred.get("draw_p"),
                        "away": pred.get("away_win_p"),
                    }

                    if all(isinstance(p_model[k], (int, float)) for k in ("home","draw","away")):
                        p_imp = scan_market_odds_1x2(odds) or {}
                        if all(isinstance(p_imp.get(k), (int, float)) for k in ("home","draw","away")):
                            edges = compute_value_edges(p_model, p_imp)
                            fx["market_probs_1x2"] = p_imp
                            fx["edges_1x2"] = edges
                            best_side = max(edges.keys(), key=lambda k: edges.get(k, -1e9))
                            fx["value_side"] = best_side
                            fx["best_edge"] = float(edges.get(best_side, 0.0))
                fetched += 1

            if min_edge and (fx.get("best_edge") is not None):
                try:
                    if float(fx["best_edge"]) < float(min_edge):
                        continue
                except Exception:
                    pass

            out.append(fx)
        except Exception:
            out.append(fx)
            continue

    return out


def filter_fixtures_by_window(fixtures: List[Dict[str, Any]], window_start: datetime, window_end: datetime) -> List[Dict[str, Any]]:
    filtered = []
    for fx in fixtures:
        fixture = fx.get("fixture", {}) or {}
        dt_str = fixture.get("date")
        if not dt_str:
            continue
        try:
            kickoff = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if window_start <= kickoff <= window_end:
            filtered.append(fx)
    return filtered

def parse_date_range_or_400(from_date: str, to_date: Optional[str]) -> Tuple[datetime, datetime, str, str]:
    if not from_date:
        raise HTTPException(status_code=400, detail="from_date is required (YYYY-MM-DD)")
    try:
        start_day = datetime.strptime(from_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid from_date format. Use YYYY-MM-DD.")
    to_input = to_date or from_date
    try:
        end_day = datetime.strptime(to_input, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid to_date format. Use YYYY-MM-DD.")
    if end_day < start_day:
        raise HTTPException(status_code=400, detail="to_date must be on or after from_date.")
    range_days = (end_day - start_day).days + 1
    if range_days > MAX_DATE_RANGE_DAYS:
        raise HTTPException(status_code=400, detail=f"Date range too large. Max {MAX_DATE_RANGE_DAYS} days.")
    window_start = datetime.combine(start_day, datetime.min.time()).replace(tzinfo=timezone.utc)
    window_end = datetime.combine(end_day, datetime.max.time()).replace(tzinfo=timezone.utc)
    return window_start, window_end, start_day.isoformat(), end_day.isoformat()

# ============================================================
# TOPSCORERS (cached)
# ============================================================

TOPSCORERS_CACHE: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
TOPSCORERS_FAIL_UNTIL: Dict[Tuple[int, int], float] = {}

def fetch_top_scorers(league_id: int, season: int) -> List[Dict[str, Any]]:
    key = (league_id, season)
    now = time.time()
    if key in TOPSCORERS_FAIL_UNTIL and now < TOPSCORERS_FAIL_UNTIL[key]:
        return TOPSCORERS_CACHE.get(key, [])
    if key in TOPSCORERS_CACHE:
        return TOPSCORERS_CACHE[key]
    try:
        data = api_get("/players/topscorers", {"league": league_id, "season": season})
    except HTTPException as e:
        # Cache failure briefly to avoid log/request storms
        TOPSCORERS_FAIL_UNTIL[key] = time.time() + 900
        TOPSCORERS_CACHE[key] = []
        return []
    except Exception:
        TOPSCORERS_FAIL_UNTIL[key] = time.time() + 900
        TOPSCORERS_CACHE[key] = []
        return []
    resp = data.get("response", [])
    out: List[Dict[str, Any]] = []
    for row in resp:
        player = row.get("player", {}) or {}
        stats_list = row.get("statistics", []) or []
        if not stats_list:
            continue
        s = stats_list[0]
        team = s.get("team", {}) or {}
        goals_obj = s.get("goals", {}) or {}
        games_obj = s.get("games", {}) or {}
        goals = goals_obj.get("total") or goals_obj.get("league") or 0
        apps = games_obj.get("appearences") or games_obj.get("appearances") or games_obj.get("matches") or 0
        out.append(
            {
                "player_id": player.get("id"),
                "name": player.get("name"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "photo": player.get("photo"),
                "goals": goals or 0,
                "appearances": apps or 0,
            }
        )
    TOPSCORERS_CACHE[key] = out
    return out

# ============================================================
# FASTAPI APP + STATIC + LOGOS
# ============================================================

app = FastAPI(title="WinMatic Predictor (Clean Backend)")

@app.get("/debug/neon")
def debug_neon():
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("select 1")
        row = cur.fetchone()
        return {
            "ok": True,
            "db": "postgres" if getattr(conn, "is_pg", False) else "sqlite",
            "select1": (row[0] if row else None),
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.on_event("startup")
def _startup_init_history_db():
    try:
        init_history_db()
    except Exception:
        logger.exception("init_history_db failed (continuing to boot)")

def _history_table_columns(conn) -> list[str]:
    cur = conn.cursor()
    try:
        if getattr(conn, "is_pg", False):
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'predictions_history'
                ORDER BY ordinal_position
                """
            )
            cols = [r[0] for r in cur.fetchall()]
        else:
            cur.execute("PRAGMA table_info(predictions_history)")
            cols = [r[1] for r in cur.fetchall()]
        return cols
    finally:
        try:
            cur.close()
        except Exception:
            pass

@app.get("/debug/db")


@app.get("/debug/quota")
def debug_quota():
    """Show internal quota/rate-limit flags."""
    if os.getenv("ADMIN_TOKEN"):
        tok = (getattr(globals().get("request"), "headers", {}) or {}).get("x-admin-token")  # best-effort
    chk = _recheck_api_daily_quota()
    return {
        "API_QUOTA_EXHAUSTED": bool(API_QUOTA_EXHAUSTED),
        "API_QUOTA_EXHAUSTED_UNTIL": API_QUOTA_EXHAUSTED_UNTIL,
        "API_RATE_LIMIT_UNTIL": API_RATE_LIMIT_UNTIL,
        "recheck": chk,
        "now": time.time(),
    }


@app.post("/debug/quota/reset")
def debug_quota_reset(admin_token: str = Query(None, description="If ADMIN_TOKEN is set, you must pass it here")):
    """Reset internal quota/rate-limit flags (useful if they get stuck)."""
    expected = os.getenv("ADMIN_TOKEN")
    if expected and admin_token != expected:
        raise HTTPException(status_code=403, detail="ADMIN_TOKEN required")

    global API_QUOTA_EXHAUSTED, API_QUOTA_EXHAUSTED_UNTIL, API_RATE_LIMIT_UNTIL
    API_QUOTA_EXHAUSTED = False
    API_QUOTA_EXHAUSTED_UNTIL = 0.0
    API_RATE_LIMIT_UNTIL = 0.0
    return {"ok": True}
def debug_db():
    import os
    import sqlite3
    from pathlib import Path

    p = Path(DB_PATH)
    info = {
        "cwd": os.getcwd(),
        "db_path": DB_PATH,
        "db_abs": str(p.resolve()),
        "exists": p.exists(),
    }

    try:
        con = db_connect()
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(predictions_history)").fetchall()]
            info["predictions_history_cols"] = cols
        finally:
            con.close()
    except Exception as e:
        info["error"] = repr(e)

    return info


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Serve /static/*
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root → /static/index.html
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# ---------- Team logo cache/proxy ----------
LOGO_DIR = os.path.join("static", "team-logo")
os.makedirs(LOGO_DIR, exist_ok=True)

# 1x1 transparent PNG (base64) to seed /static/team-logo/default.png
_DEFAULT_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)

def _ensure_default_logo():
    default_path = os.path.join(LOGO_DIR, "default.png")
    if not os.path.exists(default_path):
        try:
            with open(default_path, "wb") as f:
                f.write(base64.b64decode(_DEFAULT_PNG_B64))
            logger.info("[LOGO] created default placeholder at %s", default_path)
        except Exception as e:
            logger.warning("[LOGO] failed to create default placeholder: %s", e)

_ensure_default_logo()

def _logo_cache_path(team_id: int) -> str:
    return os.path.join(LOGO_DIR, f"{team_id}.png")

def _fetch_and_cache_logo(team_id: int) -> Optional[bytes]:
    url = f"https://media.api-sports.io/football/teams/{team_id}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.content:
            path = _logo_cache_path(team_id)
            try:
                with open(path, "wb") as f:
                    f.write(r.content)
                logger.info("[LOGO] cached %s → %s", url, path)
            except Exception as e:
                logger.warning("[LOGO] write cache failed: %s", e)
            return r.content
        logger.warning("[LOGO] CDN returned %s for team_id=%s", r.status_code, team_id)
    except Exception as e:
        logger.warning("[LOGO] fetch failed for team_id=%s: %s", team_id, e)
    return None


# Odds endpoints often have limited historical availability.
# To avoid wasting API quota, we only attempt odds fetches within this window.
ODDS_LOOKBACK_DAYS = int(os.getenv("ODDS_LOOKBACK_DAYS", "21"))   # past days
ODDS_FUTURE_DAYS = int(os.getenv("ODDS_FUTURE_DAYS", "10"))       # upcoming days
ODDS_MAX_CALLS_PER_RUN = int(os.getenv("ODDS_MAX_CALLS_PER_RUN", "60"))

def _within_odds_window(kickoff_utc: Any) -> bool:
    """Return True if kickoff is within a (now - lookback) .. (now + future) window."""
    try:
        if not kickoff_utc:
            return False
        if isinstance(kickoff_utc, str):
            dt = datetime.datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
        elif isinstance(kickoff_utc, datetime.datetime):
            dt = kickoff_utc
        else:
            return False
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - datetime.timedelta(days=ODDS_LOOKBACK_DAYS)) <= dt <= (now + datetime.timedelta(days=ODDS_FUTURE_DAYS))
    except Exception:
        return False


@app.get("/debug/api-usage")
def debug_api_usage():
    """Show basic API call counters (HTTP calls only, not cache hits)."""
    try:
        with API_USAGE_LOCK:
            data = {
                "ok": True,
                "total_http_calls": API_USAGE.get("total_http_calls", 0),
                "by_path": dict(API_USAGE.get("by_path", {})),
                "last_headers": dict(API_USAGE.get("last_headers", {})),
            }
        data["by_path_sorted"] = sorted(data["by_path"].items(), key=lambda kv: kv[1], reverse=True)
        return data
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/team-logo/default.png")
def team_logo_default():
    path = os.path.join(LOGO_DIR, "default.png")
    headers = {"Cache-Control": "public, max-age=604800"}
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", headers=headers)
    # Fallback in memory
    return Response(content=base64.b64decode(_DEFAULT_PNG_B64), media_type="image/png", headers=headers)

@app.get("/team-logo/{team_id}.png")
def team_logo(team_id: int):
    """
    Serve cached logo if present; otherwise fetch from API-Sports CDN,
    cache to disk, and return. If still unavailable, return default.
    """
    path = _logo_cache_path(team_id)
    headers = {"Cache-Control": "public, max-age=604800"}
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", headers=headers)
    # try live fetch
    content = _fetch_and_cache_logo(team_id)
    if content:
        return Response(content=content, media_type="image/png", headers=headers)
    # default
    default_path = os.path.join(LOGO_DIR, "default.png")
    if os.path.exists(default_path):
        return FileResponse(default_path, media_type="image/png", headers=headers)
    return Response(content=base64.b64decode(_DEFAULT_PNG_B64), media_type="image/png", headers=headers)

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

# ============================================================
# Pydantic
# ============================================================

class TrainRequest(BaseModel):
    league: int = Field(DEFAULT_LEAGUE)
    seasons: Optional[List[int]] = Field(default=None, description="List of seasons to train on (e.g. [2021,2022,2023])")

class OddsSnapshot(BaseModel):
    league: int
    fixture_id: int
    kickoff_utc: str
    snapshot_type: str = Field("pred", description="e.g. pred, close")
    bookmaker: str | None = None
    odds_home: float
    odds_draw: float
    odds_away: float


# ------------------------------------------------------------
# ⭐ PASTE build_predictions_for_fixtures HERE
# ------------------------------------------------------------

def build_predictions_for_fixtures(
    fixtures: List[Dict[str, Any]],
    model: Any,
    meta: Dict[str, Any],
    league: int,
    season: int,
    window_start: datetime,
    window_end: datetime
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []

    for fx in fixtures:
        fixture = fx.get("fixture", {}) or {}
        teams = fx.get("teams", {}) or {}
        league_obj = fx.get("league", {}) or {}
        goals_obj = fx.get("goals", {}) or {}

        dt_str = fixture.get("date")
        if not dt_str:
            continue

        try:
            kickoff = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except:
            continue

        if not (window_start <= kickoff <= window_end):
            continue

        home_team = teams.get("home", {})
        away_team = teams.get("away", {})

        if not home_team or not away_team:
            continue

        home_id = home_team.get("id")
        away_id = away_team.get("id")
        home_name = home_team.get("name", "Home")
        away_name = away_team.get("name", "Away")

        # Build feature row & predict
        X = make_feature_row_for_fixture(fx, meta)
        y_pred = model.predict(X)[0]

        pred_home_goals = max(0, float(y_pred[0]))
        pred_away_goals = max(0, float(y_pred[1]))

        # Derived stats
        extra = derive_extra_stats(pred_home_goals, pred_away_goals)

        # Poisson probability engine
        def poisson_prob(avg, k):
            try:
                return (avg ** k) * math.exp(-avg) / math.factorial(k)
            except:
                return 0.0

        # --- 1X2 probs (normalized) ---
        home_win_p, draw_p, away_win_p = _poisson_outcome_probs(
            float(pred_home_goals),
            float(pred_away_goals),
            max_goals=10
        )


        # Normalize to sum to 1.0 (important for 1X2 betting + fair comparisons)
        total_p = home_win_p + draw_p + away_win_p
        if total_p > 0:
            home_win_p /= total_p
            draw_p /= total_p
            away_win_p /= total_p
        else:
            home_win_p = draw_p = away_win_p = 1.0 / 3.0

        # Best side (model favourite) + ✅ calibration
        probs = {"home": home_win_p, "draw": draw_p, "away": away_win_p}

        cal = load_1x2_calibration(league)
        probs = apply_1x2_calibration(probs, cal)

        # keep the scalar vars in sync (optional but nice)
        home_win_p = float(probs["home"])
        draw_p     = float(probs["draw"])
        away_win_p = float(probs["away"])

        best_side = max(probs, key=probs.get)
        best_prob = probs[best_side]



        scorers = build_players_to_score_for_fixture(fx, league, season)

        results.append({
            "fixture_id": fixture.get("id"),
            "league_id": league_obj.get("id"),
            "league_name": league_obj.get("name"),
            "kickoff_utc": kickoff.isoformat(),

            "home_id": home_id,
            "home_name": home_name,
            "home_logo": f"/team-logo/{home_id}.png",

            "away_id": away_id,
            "away_name": away_name,
            "away_logo": f"/team-logo/{away_id}.png",

            "predictions": {
                "home_goals": round(pred_home_goals, 2),
                "away_goals": round(pred_away_goals, 2),
                "home_win_p": round(home_win_p, 3),
                "draw_p": round(draw_p, 3),
                "away_win_p": round(away_win_p, 3),

                "best_side": best_side,
                "best_prob": round(best_prob, 4),

                "home_sot": extra["home_sot"],
                "away_sot": extra["away_sot"],

                "home_corners": extra["home_corners"],
                "away_corners": extra["away_corners"],

                "home_yellows": extra["home_yellows"],
                "away_yellows": extra["away_yellows"],
                "home_reds": extra["home_reds"],
                "away_reds": extra["away_reds"]
            },

            "players_to_score": scorers
        })

    return results


# ============================================================
# API ENDPOINTS
# ============================================================

@app.post("/train", dependencies=[Depends(require_admin)])
def api_train(req: "TrainRequest"):
    seasons = req.seasons or DEFAULT_SEASONS
    logger.info("[TRAIN API] league=%s seasons=%s", req.league, seasons)
    info = train_model(req.league, seasons)
    return {"ok": True, "info": info}

@app.get("/predict/upcoming")
def api_predict_upcoming(
    league: int = Query(DEFAULT_LEAGUE),
    days_ahead: int = Query(7, ge=1, le=14),

    include_odds: int = Query(1, ge=0, le=1, description="Attach odds_1x2 + value fields"),
    odds_limit: int = Query(25, ge=0, le=50, description="Max fixtures to fetch odds for"),
    min_edge: float = Query(0.0, ge=0.0, le=1.0, description="Filter out fixtures with best_edge < min_edge"),
):
    # --- ODDS ENRICH: upcoming ---
    def _enrich_upcoming_odds(fixtures):
        """Attach odds_1x2 + best_edge/value_side to fixtures when possible."""
        if not fixtures:

            # --- SAFE_UPCOMING_VALUE_FIELDS_V1 ---
            try:
                # Prefer explicit response dict name if present
                _out = locals().get("out") or locals().get("resp") or None
                if isinstance(_out, dict) and isinstance(_out.get("fixtures"), list):
                    _fxs = _out["fixtures"]
                else:
                    # Fall back to common list variable names
                    _fxs = locals().get("fixtures") or locals().get("xs") or []
            
                if isinstance(_fxs, list):
                    for f in _fxs:
                        if not isinstance(f, dict):
                            continue
                        odds = f.get("odds_1x2") or {}
                        pred = f.get("predictions") or {}
                        model = {
                            "home": pred.get("home_win_p"),
                            "draw": pred.get("draw_p"),
                            "away": pred.get("away_win_p"),
                        }
                        # Guard: only compute when everything is numeric and odds > 0
                        if not (isinstance(odds, dict) and all(isinstance(model[k], (int,float)) for k in ("home","draw","away"))):
                            continue
                        inv = {}
                        for k in ("home","draw","away"):
                            v = odds.get(k)
                            if isinstance(v, (int,float)) and float(v) > 0:
                                inv[k] = 1.0/float(v)
                        total = float(sum(inv.values()))
                        if total <= 0:
                            continue
                        implied = {k: float(inv.get(k,0.0))/total for k in ("home","draw","away")}
                        edges = {k: float(model[k]) - float(implied[k]) for k in ("home","draw","away")}
                        best_side = max(edges, key=lambda kk: edges[kk])
                        f["implied_1x2"] = implied
                        f["value_edges"] = edges
                        f["value_side"] = best_side
                        f["best_edge"] = float(edges[best_side])
            except Exception:
                pass
            # --- end SAFE_UPCOMING_VALUE_FIELDS_V1 ---
            return fixtures
        if "fetch_1x2_odds_for_fixture" not in globals():
            return fixtures
        try:
            max_odds = int(os.getenv("ODDS_MAX_CALLS_PER_RUN", "60"))
        except Exception:
            max_odds = 60

        calls = 0
        for fx in fixtures:
            if calls >= max_odds:
                break
            try:
                if fx.get("odds_1x2") is not None:
                    continue

                fid = fx.get("fixture_id") or fx.get("id")
                if not fid:
                    continue

                kickoff = fx.get("kickoff_utc") or fx.get("kickoff")
                if kickoff and "_within_odds_window" in globals() and not _within_odds_window(kickoff):
                    continue

                odds, meta = fetch_1x2_odds_for_fixture(int(fid), return_meta=True)
                # --- AUTO-SNAPSHOT ODDS for CLV tracking (pred + optional close) ---
                try:
                    # Only if we actually have 1X2 odds in the standard shape
                    if isinstance(locals().get("odds_1x2"), dict) and all(k in odds_1x2 for k in ("home","draw","away")):
                        _fx = locals().get("fx") or locals().get("fixture") or locals().get("f") or {}
                        _kickoff = (locals().get("kickoff_utc") or locals().get("kickoff") or locals().get("kickoff_time")
                                   or str(((_fx.get("fixture") or {}).get("date")) or ""))
                        _league = int(locals().get("league") or locals().get("league_id") or (((_fx.get("league") or {}).get("id")) or 0))
                        _fid = int(locals().get("fixture_id") or locals().get("fid") or (((_fx.get("fixture") or {}).get("id")) or 0))
                        _meta = locals().get("odds_meta") or locals().get("meta") or {}

                        if _league and _fid and _kickoff:
                            _get_by_fixture = globals().get("get_odds_snapshot_by_fixture")
                            def _has_snap(st: str) -> bool:
                                try:
                                    if callable(_get_by_fixture):
                                        return _get_by_fixture(_league, _fid, st) is not None
                                except Exception:
                                    pass
                                return False

                            # Save "pred" snapshot once (do not overwrite)
                            if not _has_snap("pred"):
                                record_odds_snapshot(OddsSnapshot(
                                    league=_league,
                                    fixture_id=_fid,
                                    kickoff_utc=str(_kickoff),
                                    snapshot_type="pred",
                                    bookmaker=_meta.get("bookmaker"),
                                    odds_home=float(odds_1x2["home"]),
                                    odds_draw=float(odds_1x2["draw"]),
                                    odds_away=float(odds_1x2["away"]),
                                ))

                            # Optional: save "close" snapshot if kickoff is soon (default 120 minutes)
                            try:
                                close_min = int(os.getenv("CLOSE_SNAPSHOT_MINUTES", "120"))
                            except Exception:
                                close_min = 120

                            try:
                                ks = str(_kickoff).replace("Z", "+00:00")
                                kdt = datetime.fromisoformat(ks)
                                if kdt.tzinfo is None:
                                    kdt = kdt.replace(tzinfo=timezone.utc)
                                mins = (kdt - datetime.now(timezone.utc)).total_seconds() / 60.0
                                if 0 <= mins <= close_min and not _has_snap("close"):
                                    record_odds_snapshot(OddsSnapshot(
                                        league=_league,
                                        fixture_id=_fid,
                                        kickoff_utc=str(_kickoff),
                                        snapshot_type="close",
                                        bookmaker=_meta.get("bookmaker"),
                                        odds_home=float(odds_1x2["home"]),
                                        odds_draw=float(odds_1x2["draw"]),
                                        odds_away=float(odds_1x2["away"]),
                                    ))
                            except Exception:
                                pass
                except Exception:
                    # Never break /predict/upcoming because snapshot save failed
                    pass
                if not odds:
                    continue

                fx["odds_1x2"] = odds
                fx["odds_meta"] = meta

                preds = fx.get("predictions") or {}
                model_probs = {
                    "home": preds.get("home_win_p"),
                    "draw": preds.get("draw_p"),
                    "away": preds.get("away_win_p"),
                }

                if "compute_value_edges" in globals():
                    ve = compute_value_edges(model_probs, odds)
                    fx["best_edge"] = ve.get("best_edge")
                    fx["value_side"] = ve.get("best_side")
                    fx["edges_1x2"] = ve.get("edges")
                    fx["evs_1x2"] = ve.get("evs")
                    fx["market_probs_1x2"] = ve.get("market_probs")

                calls += 1
            except Exception:
                continue


        # --- UPCOMING_IMPLIED_EDGES_V2 ---
        # Always compute implied_1x2/value_edges/best_edge/value_side when odds+predictions exist.
        try:
            for fx in fixtures:
                if not isinstance(fx, dict):
                    continue
                odds = fx.get("odds_1x2")
                pred = fx.get("predictions") or {}
                if not isinstance(odds, dict):
                    continue

                model = {
                    "home": pred.get("home_win_p"),
                    "draw": pred.get("draw_p"),
                    "away": pred.get("away_win_p"),
                }
                if not all(isinstance(model[k], (int, float)) for k in ("home", "draw", "away")):
                    continue

                inv = {}
                for k in ("home", "draw", "away"):
                    v = odds.get(k)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        inv[k] = 1.0 / float(v)

                total = float(sum(inv.values()))
                if total <= 0:
                    continue

                implied = {k: float(inv.get(k, 0.0)) / total for k in ("home", "draw", "away")}
                edges = {k: float(model[k]) - float(implied[k]) for k in ("home", "draw", "away")}
                best_side = max(edges, key=lambda kk: edges[kk])

                fx["implied_1x2"] = implied
                fx["value_edges"] = edges
                fx["value_side"] = best_side
                fx["best_edge"] = round(float(edges[best_side]), 3)
        except Exception:
            pass
        # --- end UPCOMING_IMPLIED_EDGES_V2 ---
        return fixtures
    try:
        model, meta = load_model_and_meta(league)
    except HTTPException:
        # Fall back to snapshot if model not available
        snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=days_ahead)
        if snapshot:
            snapshot = _enrich_upcoming_odds(snapshot)
            return {
                "ok": True,
                "count": len(snapshot),
                "fixtures": snapshot,
                "source": "snapshot",
                "snapshot_file": os.path.basename(snap_path) if snap_path else None,
            }
        raise
    
       # Always compute implied_1x2/value_edges/best_edge if we have odds + probs
    try:
        odds = fx.get("odds_1x2") or {}
        p = fx.get("predictions") or {}
        if isinstance(odds, dict) and isinstance(p, dict):
            if all(k in odds for k in ("home","draw","away")) and all(k in p for k in ("home_win_p","draw_p","away_win_p")):
                inv = {}
                for k in ("home","draw","away"):
                    v = odds.get(k)
                    if isinstance(v, (int,float)) and float(v) > 0:
                        inv[k] = 1.0/float(v)
                tot = float(sum(inv.values()))
                if tot > 0:
                    implied = {k: float(inv.get(k,0.0))/tot for k in ("home","draw","away")}
                    edges = {
                        "home": float(p["home_win_p"]) - implied["home"],
                        "draw": float(p["draw_p"]) - implied["draw"],
                        "away": float(p["away_win_p"]) - implied["away"],
                    }
                    best = max(edges, key=edges.get)
                    fx["implied_1x2"] = implied
                    fx["value_edges"] = edges
                    fx["value_side"] = best
                    fx["best_edge"] = round(float(edges[best]), 3)
    except Exception:
        pass

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    season = current_season()

    data = api_get("/fixtures", {"league": league, "season": season, "next": 50})
    fixtures = data.get("response", []) or []
    if not fixtures:
        cached_fixtures = cached_upcoming_fixtures(league, season)
        if cached_fixtures:
            fixtures = filter_fixtures_by_window(cached_fixtures, now, end)
            if fixtures:
                logger.info("[PREDICT UPCOMING] served from cached upcoming fixtures league=%s", league)

    results: List[Dict[str, Any]] = []
    if fixtures:
        results = build_predictions_for_fixtures(
            fixtures=fixtures,
            model=model,
            meta=meta,
            league=league,
            season=season,
            window_start=now,
            window_end=end,
        )

    # Enrich predictor payload: ensure 1x2 probs + names + (best-effort) 1x2 odds
    odds_calls = 0
    for fx in (results or []):
        if not isinstance(fx, dict):
            continue
        # name aliases
        if fx.get('home_name') is None and fx.get('home') is not None:
            fx['home_name'] = fx.get('home')
        if fx.get('away_name') is None and fx.get('away') is not None:
            fx['away_name'] = fx.get('away')

        preds = fx.get('predictions') or {}
        if isinstance(preds, dict):
            hxg = preds.get('home_goals')
            axg = preds.get('away_goals')
            if preds.get('home_win_p') is None and hxg is not None and axg is not None:
                try:
                    probs = poisson_1x2_probs(float(hxg), float(axg), max_goals=10)
                    preds['home_win_p'] = round(float(probs.get('home_win_p', 0.0)), 3)
                    preds['draw_p'] = round(float(probs.get('draw_p', 0.0)), 3)
                    preds['away_win_p'] = round(float(probs.get('away_win_p', 0.0)), 3)
                except Exception:
                    pass
            fx['predictions'] = preds

        # odds (optional + capped)
        if (not fx.get('odds_1x2')):
            ko = fx.get('kickoff_utc')
            if odds_calls < ODDS_MAX_CALLS_PER_RUN and _within_odds_window(ko):
                odds_calls += 1
                try:
                    odds = fetch_1x2_odds_for_fixture(int(fx.get('fixture_id')))
                    if odds:
                        fx['odds_1x2'] = odds
                        implied = implied_probs_1x2(odds)
                        if implied and isinstance(implied, dict):
                            # --- STORE_UPCOMING_IMPLIED_EDGES_V1 ---
                            fx['implied_1x2'] = implied
                            p = fx.get('predictions') or {}
                            edges = {}
                            for side, pk in [('home','home_win_p'),('draw','draw_p'),('away','away_win_p')]:
                                mp = implied.get(side)
                                pp = p.get(pk)
                                if mp is not None and pp is not None:
                                    edges[side] = float(pp) - float(mp)

                            if edges:
                                best_side = max(edges, key=lambda k: edges[k])
                                fx['value_edges'] = edges
                                fx['best_edge'] = round(edges[best_side], 3)
                                fx['value_side'] = best_side
                            # --- end STORE_UPCOMING_IMPLIED_EDGES_V1 ---
                except Exception:
                    pass

    if not results:
        snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=days_ahead)
        if snapshot:
            snapshot = _enrich_upcoming_odds(snapshot)
            return {
                "ok": True,
                "count": len(snapshot),
                "fixtures": snapshot,
                "source": "snapshot",
                "snapshot_file": os.path.basename(snap_path) if snap_path else None,
            }
        results = _enrich_upcoming_odds(results)
        return {
            "ok": False,
            "count": 0,
            "fixtures": [],
            "detail": "No fixtures available. Train the model or provide cached data.",
        }

    # 👉 Add reasoning to each result, keeping your existing structure
    for p in results:
        p["reasoning"] = build_reasoning_for_prediction(p, meta)

    results = _enrich_upcoming_odds(results)

    # Save to history (includes odds/value if present)
    record_predictions_history(league, results)
    return {
        "ok": True,
        "count": len(results),
        "fixtures": results,
        "source": "model",
    }

@app.get("/value-bets")
def api_value_bets(
    league: int = Query(DEFAULT_LEAGUE, description="League ID (e.g. 39 = Premier League)"),
    days_ahead: int = Query(7, ge=1, le=14, description="How many days ahead to look for fixtures"),
    min_edge: float = Query(0.05, description="Minimum edge to consider (e.g. 0.05 = 5%)"),
    limit: int = Query(50, ge=1, le=200, description="Max fixtures to return"),
    mode: str = Query("value", description="Pick mode: value, accuracy, or profit"),
    min_prob: float = Query(0.40, ge=0.0, le=1.0, description="Profit filter: minimum model probability for the pick"),
    max_ratio: float = Query(1.35, ge=1.0, description="Profit filter: max (model_p / market_p) allowed to avoid outliers"),
):
    """
    Wrapper around /value/upcoming.

    mode=value     -> best_side/best_edge are EV-based (value_pick/value_pick_ev)
    mode=accuracy  -> best_side/best_edge become model_pick/model_pick_prob (draw allowed)
    mode=profit    -> returns only 'sane' +EV bets (EV>0, p>=min_prob, p/market<=max_ratio)
    """

    resp = api_value_upcoming(
        league=league,
        days_ahead=days_ahead,
        min_edge=min_edge,
        limit=limit,
    )

    # Safety: never return None
    if not isinstance(resp, dict):
        return {
            "ok": False,
            "error": "internal_error",
            "detail": "api_value_upcoming returned non-dict",
            "input": str(resp),
        }

    fixtures = resp.get("fixtures") or []

    # Always ensure value fields exist
    for f in fixtures:
        f.setdefault("value_pick", f.get("best_side"))
        f.setdefault("value_pick_ev", f.get("best_edge"))

    # Persist history
    try:
        record_predictions_history(league, fixtures)
    except Exception as e:
        logger.warning("[DB] Failed to record history: %s", e)

    m = (mode or "").strip().lower()

    # ----------------------------
    # PROFIT MODE (filtered +EV)
    # ----------------------------
    if m in ("profit", "p", "evsafe"):
        rec = []
        for f in fixtures:
            vp = f.get("value_pick")
            ev = f.get("value_pick_ev")

            if vp not in ("home", "draw", "away"):
                continue
            if not isinstance(ev, (int, float)) or ev <= 0:
                continue

            probs = f.get("model_probs") or {}
            mkt = f.get("market_probs") or {}

            p = probs.get(vp)
            q = mkt.get(vp)

            if not isinstance(p, (int, float)) or p < float(min_prob):
                continue
            if not isinstance(q, (int, float)) or q <= 0:
                continue

            ratio = p / q
            if ratio > float(max_ratio):
                continue

            f2 = dict(f)
            f2["sanity_ratio"] = round(ratio, 3)
            f2["best_side"] = vp
            f2["best_edge"] = round(float(ev), 4)
            rec.append(f2)

        rec.sort(key=lambda x: x.get("best_edge") or -1e9, reverse=True)
        resp["fixtures"] = rec[:limit]
        resp["count"] = len(resp["fixtures"])
        resp["source"] = (resp.get("source") or "") + "+profit"
        resp["filters"] = {"min_prob": min_prob, "max_ratio": max_ratio}
        return resp

    # ----------------------------
    # ACCURACY MODE
    # ----------------------------
    if m in ("accuracy", "acc", "model"):
        for f in fixtures:
            probs = f.get("model_probs") or {}
            numeric = {k: v for k, v in probs.items() if k in ("home", "draw", "away") and isinstance(v, (int, float))}
            if numeric:
                ranked = sorted(numeric.items(), key=lambda kv: kv[1], reverse=True)
                mp1, mp1p = ranked[0]
                mp2, mp2p = (ranked[1] if len(ranked) > 1 else (None, None))
            else:
                mp1, mp1p, mp2, mp2p = (None, None, None, None)

            f["model_pick"] = mp1
            f["model_pick_prob"] = (round(float(mp1p), 4) if isinstance(mp1p, (int, float)) else None)

            f["model_pick_1"] = mp1
            f["model_pick_1_prob"] = (round(float(mp1p), 4) if isinstance(mp1p, (int, float)) else None)
            f["model_pick_2"] = mp2
            f["model_pick_2_prob"] = (round(float(mp2p), 4) if isinstance(mp2p, (int, float)) else None)

            f["best_side"] = mp1
            f["best_edge"] = f["model_pick_prob"]

        resp["source"] = (resp.get("source") or "") + "+accuracy"
        return resp

    # ----------------------------
    # VALUE MODE (default)
    # ----------------------------
    for f in fixtures:
        f["best_side"] = f.get("value_pick")
        f["best_edge"] = f.get("value_pick_ev")

    resp["source"] = (resp.get("source") or "") + "+value"
    return resp



@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    league: int = Query(DEFAULT_LEAGUE),
    days_ahead: int = Query(3, ge=1, le=14),
):
    """
    Simple HTML dashboard that shows upcoming predictions (probabilities + reasoning).
    """
    # Reuse the logic from /predict/upcoming directly
    data = api_predict_upcoming(league=league, days_ahead=days_ahead)
    fixtures = data.get("fixtures", [])

    title = f"League {league} predictions (next {days_ahead} days)"

    # Build a very simple HTML page
    rows_html = []
    for fx in fixtures:
        preds = fx.get("predictions") or {}
        home = fx.get("home_name", "Home")
        away = fx.get("away_name", "Away")
        kickoff = fx.get("kickoff_utc", "")
        reasoning = fx.get("reasoning", "")

        ph = preds.get("home_win_p", 0.0)
        pd = preds.get("draw_p", 0.0)
        pa = preds.get("away_win_p", 0.0)
        xh = preds.get("home_goals", 0.0)
        xa = preds.get("away_goals", 0.0)

        row = f"""
        <div class="card">
            <div class="teams">
                <span class="home">{home}</span>
                <span class="vs">vs</span>
                <span class="away">{away}</span>
            </div>
            <div class="kickoff">Kickoff (UTC): {kickoff}</div>
            <div class="probs">
                <strong>Probabilities:</strong>
                Home {ph:.2f} &nbsp;·&nbsp; Draw {pd:.2f} &nbsp;·&nbsp; Away {pa:.2f}
            </div>
            <div class="xg">
                <strong>Expected goals:</strong> {xh:.2f} : {xa:.2f}
            </div>
            <div class="reasoning">
                <strong>Reasoning:</strong> {reasoning}
            </div>
        </div>
        """
        rows_html.append(row)

    body_html = "\n".join(rows_html) if rows_html else "<p>No fixtures found.</p>"

    html = f"""
    <html>
      <head>
        <title>{title}</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            margin: 0;
            padding: 24px;
          }}
          h1 {{
            margin-bottom: 16px;
          }}
          .card {{
            background: #020617;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 12px;
            border: 1px solid #1f2937;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
          }}
          .teams {{
            font-size: 1.1rem;
            margin-bottom: 4px;
          }}
          .home {{
            font-weight: 600;
          }}
          .away {{
            font-weight: 600;
          }}
          .vs {{
            opacity: 0.8;
            margin: 0 4px;
          }}
          .kickoff, .probs, .xg, .reasoning {{
            font-size: 0.9rem;
            margin-top: 4px;
          }}
          .reasoning {{
            margin-top: 8px;
          }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        {body_html}
      </body>
    </html>
    """
    return html

@app.get("/value/upcoming")
def api_value_upcoming(
    league: int = Query(DEFAULT_LEAGUE),
    days_ahead: int = Query(7, ge=1, le=14),
    min_edge: float = Query(0.05, description="Minimum edge to consider (e.g. 0.05 = 5%)"),
    limit: int = Query(5, ge=1, le=50),
):
    """
    Show the top 'value' upcoming matches for a league.

    - Uses the model's 1X2 probabilities
    - Fetches bookmaker 1X2 odds from API-FOOTBALL /odds
    - Compares and returns matches where the model edge >= min_edge
    - Includes the same natural-language reasoning as /predict/upcoming

    This version is quota-safe: it limits the number of live /odds calls
    per request.
    """
    try:
        model, meta = load_model_and_meta(league)
    except HTTPException:
        # If no model, we can't compute value
        raise

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    season = current_season()

    # Get fixtures just like /predict/upcoming
    data = api_get("/fixtures", {"league": league, "season": season, "next": 50})
    fixtures = data.get("response", []) or []
    if not fixtures:
        cached_fixtures = cached_upcoming_fixtures(league, season)
        if cached_fixtures:
            fixtures = filter_fixtures_by_window(cached_fixtures, now, end)
            if fixtures:
                logger.info("[VALUE UPCOMING] served from cached upcoming fixtures league=%s", league)

    if not fixtures:
        raise HTTPException(status_code=404, detail="No upcoming fixtures found to evaluate value.")

    # Model predictions for these fixtures
    predictions = build_predictions_for_fixtures(
        fixtures=fixtures,
        model=model,
        meta=meta,
        league=league,
        season=season,
        window_start=now,
        window_end=end,
    )

    value_rows: List[Dict[str, Any]] = []

    # 🔒 Quota protection: cap live /odds calls per request
    MAX_ODDS_CALLS = 5
    odds_calls = 0

    for pred in predictions:
        if odds_calls >= MAX_ODDS_CALLS:
            # We've already made enough /odds calls – stop here
            break

        fixture_id = pred.get("fixture_id")
        if not fixture_id:
            continue

        preds = pred.get("predictions") or {}

        # --- model predicts goals (xG). convert xG -> 1X2 probs via Poisson ---
        # Try a few likely keys depending on how build_predictions_for_fixtures() packaged it
        xg_home = (
            preds.get("home_goals")
            or preds.get("xg_home")
            or pred.get("home_goals")
            or pred.get("xg_home")
        )
        xg_away = (
            preds.get("away_goals")
            or preds.get("xg_away")
            or pred.get("away_goals")
            or pred.get("xg_away")
        )

        # If xG missing, fall back to whatever probs exist (your old behavior)
        if isinstance(xg_home, (int, float)) and isinstance(xg_away, (int, float)):
            mp = poisson_1x2_probs(float(xg_home), float(xg_away), max_goals=10)
            cal = load_1x2_calibration(league)
            mp = apply_1x2_calibration(mp, cal)

            prob_home = mp["home"]
            prob_draw = mp["draw"]
            prob_away = mp["away"]
        else:
            # Pull model probabilities from nested structure (or flat fallback)
            prob_home = float(preds.get("home_win_p", pred.get("prob_home_win", 0.0)))
            prob_draw = float(preds.get("draw_p", pred.get("prob_draw", 0.0)))
            prob_away = float(preds.get("away_win_p", pred.get("prob_away_win", 0.0)))

            # Prefer xG -> Poisson 1X2 probs if xG exists (this is your real model)
            xg_home = pred.get("xg_home")
            xg_away = pred.get("xg_away")

            if isinstance(xg_home, (int, float)) and isinstance(xg_away, (int, float)):
                mp = poisson_1x2_probs(float(xg_home), float(xg_away), max_goals=10)

                # ✅ apply calibration to the Poisson probs
                cal = load_1x2_calibration(league)
                mp = apply_1x2_calibration(mp, cal)

                prob_home = mp["home"]
                prob_draw = mp["draw"]
                prob_away = mp["away"]



        # This calls api_get("/odds", {"fixture": fixture_id}) under the hood
        odds = fetch_1x2_odds_for_fixture(int(fixture_id))
        odds_calls += 1

        if not odds:
            continue

        odds_home = odds.get("home")
        odds_draw = odds.get("draw")
        odds_away = odds.get("away")

        value_info = compute_value_edges(
            {"home": prob_home, "draw": prob_draw, "away": prob_away},
            {"home": odds_home, "draw": odds_draw, "away": odds_away},
        )
        if not value_info:
            continue

        best_edge = value_info["best_edge"]
        if best_edge < min_edge:
            continue

        # Build natural-language reasoning (same as /predict/upcoming)
        reasoning = build_reasoning_for_prediction(pred, meta)

        # ----------------------------
        # Model pick (can be draw)
        # ----------------------------
        probs_map = {"home": prob_home, "draw": prob_draw, "away": prob_away}
        model_pick = None
        try:
            if all(isinstance(probs_map[s], (int, float)) for s in probs_map):
                model_pick = max(probs_map, key=probs_map.get)
        except Exception:
            model_pick = None

        # ----------------------------
        # All +EV sides (so draw can appear even if not best_side)
        # Use min_edge as the threshold (same as your filter)
        # ----------------------------
        evs_map = value_info.get("evs") or {}
        value_sides = []
        for s in ("home", "draw", "away"):
            v = evs_map.get(s)
            if isinstance(v, (int, float)) and v >= float(min_edge):
               value_sides.append({"side": s, "ev": round(v, 4)})

        value_sides.sort(key=lambda x: x["ev"], reverse=True)
        # Accuracy pick (argmax of model probs) — draw naturally allowed
        # --- model top picks (accuracy): top1 + runner-up ---
        probs = {
            "home": float(prob_home) if prob_home is not None else None,
            "draw": float(prob_draw) if prob_draw is not None else None,
            "away": float(prob_away) if prob_away is not None else None,
        }

        # keep only valid numbers
        probs = {k: v for k, v in probs.items() if isinstance(v, (int, float))}

        model_pick_1 = None
        model_pick_1_prob = None
        model_pick_2 = None
        model_pick_2_prob = None

        if probs:
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            model_pick_1, model_pick_1_prob = ranked[0]
            if len(ranked) > 1:
                model_pick_2, model_pick_2_prob = ranked[1]

        # keep backward-compat fields
        model_pick = model_pick_1
        model_pick_prob = model_pick_1_prob



        # Compact summary row
        value_rows.append(
            {
                "fixture_id": fixture_id,
                "kickoff_utc": pred.get("kickoff_utc"),
                "xg_home": (round(float(xg_home), 3) if isinstance(xg_home, (int, float)) else None),
                "xg_away": (round(float(xg_away), 3) if isinstance(xg_away, (int, float)) else None),
                "league_id": pred.get("league_id"),
                "league_name": pred.get("league_name"),

                "home_id": pred.get("home_id"),
                "home_name": pred.get("home_name"),
                "away_id": pred.get("away_id"),
                "away_name": pred.get("away_name"),

                "model_probs": {
                "home": (round(prob_home, 3) if isinstance(prob_home, (int, float)) else None),
                "draw": (round(prob_draw, 3) if isinstance(prob_draw, (int, float)) else None),
                "away": (round(prob_away, 3) if isinstance(prob_away, (int, float)) else None),
                },

                "bookmaker_odds": {
                    "home": odds_home,
                    "draw": odds_draw,
                    "away": odds_away,
                },
                "market_probs": value_info["market_probs"],
                "edges": value_info["edges"],
                "evs": value_info.get("evs"),
                "best_side": value_info["best_side"],
                "value_pick": value_info["best_side"],
                "value_pick_ev": round(best_edge, 4),
                "model_pick_prob": (round(model_pick_1_prob, 4) if model_pick_1_prob is not None else None),
                "model_pick_1": model_pick_1,
                "model_pick_1_prob": (round(model_pick_1_prob, 4) if model_pick_1_prob is not None else None),
                "model_pick_2": model_pick_2,
                "model_pick_2_prob": (round(model_pick_2_prob, 4) if model_pick_2_prob is not None else None),

                "model_pick": model_pick,
                "value_sides": value_sides,
                "best_edge": round(best_edge, 4),

                # 👉 explanation for this value spot
                "reasoning": reasoning,
            }
        )

    if not value_rows:
        return {
            "ok": True,
            "count": 0,
            "fixtures": [],
            "detail": f"No fixtures with edge >= {min_edge:.2f} found.",
        }

    # Sort by biggest edge first, keep top N
    value_rows.sort(key=lambda r: r["best_edge"], reverse=True)
    value_rows = value_rows[:limit]

    return {
        "ok": True,
        "count": len(value_rows),
        "fixtures": value_rows,
        "source": "model+odds",
        "min_edge": min_edge,
    }

@app.get("/bet-of-day")
def api_bet_of_day(
    league: int = Query(DEFAULT_LEAGUE, description="League ID (e.g. 39 = Premier League)"),
    days_ahead: int = Query(3, ge=1, le=14, description="How many days ahead to look for fixtures"),
    min_edge: float = Query(0.05, description="Minimum edge to consider (e.g. 0.05 = 5%)"),
):
    """
    Return the single best value spot ('Bet of the Day') for a league.

    This is just a thin wrapper around /value/upcoming:
    - Calls the same logic with limit=1
    - Returns either a single fixture or a friendly 'no value spots' message
    """
    # Reuse the /value/upcoming logic with limit=1 so we don't duplicate any odds/model code.
    resp = api_value_upcoming(
        league=league,
        days_ahead=days_ahead,
        min_edge=min_edge,
        limit=1,
    )

    # If /value/upcoming itself failed (ok == False), just forward that
    if not resp.get("ok", False) and resp.get("fixtures") is None:
        return resp

    fixtures = resp.get("fixtures") or []

    if not fixtures:
        # No value spots for this configuration
        return {
            "ok": False,
            "reason": "no_value_spots",
            "message": "No value spots found for this league / window / min_edge.",
            "league": league,
            "days_ahead": days_ahead,
            "min_edge": min_edge,
        }

    # We asked /value/upcoming for limit=1, so take the first fixture
    best_fixture = fixtures[0]

    return {
        "ok": True,
        "league": league,
        "days_ahead": days_ahead,
        "min_edge": min_edge,
        "source": "model+odds",
        "fixture": best_fixture,
    }
    from datetime import datetime, timedelta

@app.post("/results/sync")
def api_results_sync(
    league: int = Query(39, description="League ID"),
    lookback_days: int = Query(21, ge=1, le=365, description="How far back to look for unfinished predictions"),
    max_fixtures: int = Query(50, ge=1, le=300, description="Max fixtures to update per run"),
    dry_run: bool = Query(False, description="If true, don't write to DB"),
    debug: bool = Query(False, description="If true, include skipped_details in response"),
):
    """
    Backfill actual_result for predictions_history rows once matches finish.

    Selects fixture_ids that:
      - match league
      - kickoff_utc is within [now - lookback_days, now)
      - actual_result is NULL/empty
    Then fetches each fixture from API-FOOTBALL and writes actual_result (home/draw/away) when final.

    Uses SQLite (DB_PATH) and "?" placeholders.
    """
    from datetime import datetime, timedelta, timezone
    import sqlite3

    ensure_predictions_db()

    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=int(lookback_days))).isoformat()

    scanned = 0
    updated = 0
    skipped = 0
    errors = []
    skipped_details = []

    # 1) read candidate fixture_ids
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT fixture_id, MAX(kickoff_utc) AS last_kickoff
            FROM predictions_history
            WHERE league = ?
              AND kickoff_utc >= ?
              AND kickoff_utc < ?
              AND (actual_result IS NULL OR TRIM(COALESCE(actual_result,'')) = '')
              AND fixture_id IS NOT NULL
            GROUP BY fixture_id
            ORDER BY MAX(kickoff_utc) DESC
            LIMIT ?
            """,
            (int(league), since, now.isoformat(), int(max_fixtures)),
        )
        fixture_rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return {"ok": False, "league": league, "error": f"DB read failed: {repr(e)}"}

    fixture_ids = []
    for (fid, last_kickoff) in fixture_rows:
        try:
            if fid is None:
                continue
            fixture_ids.append(int(fid))
        except Exception:
            continue

    # 2) resolve each fixture and update
    for fid_int in fixture_ids:
        scanned += 1
        try:
            data = api_get("/fixtures", {"id": fid_int})
            resp = data.get("response") or []
            if not resp:
                skipped += 1
                if debug:
                    skipped_details.append({"fixture_id": fid_int, "reason": "not_found"})
                continue

            fx = resp[0] or {}
            fx_info = fx.get("fixture") or {}
            st = fx_info.get("status") or {}
            status_short = (st.get("short") or "").strip().upper()

            # final statuses (API-FOOTBALL commonly uses FT, AET, PEN for finished)
            if status_short not in ("FT", "AET", "PEN"):
                skipped += 1
                if debug:
                    skipped_details.append({"fixture_id": fid_int, "reason": "not_final", "status": status_short or None})
                continue

            goals = fx.get("goals") or {}
            hg = goals.get("home")
            ag = goals.get("away")
            if hg is None or ag is None:
                skipped += 1
                if debug:
                    skipped_details.append({"fixture_id": fid_int, "reason": "missing_goals", "status": status_short})
                continue

            try:
                hg = int(hg)
                ag = int(ag)
            except Exception:
                skipped += 1
                if debug:
                    skipped_details.append({"fixture_id": fid_int, "reason": "bad_goals", "status": status_short})
                continue

            if hg > ag:
                actual = "home"
            elif ag > hg:
                actual = "away"
            else:
                actual = "draw"

            teams = fx.get("teams") or {}
            home_name = ((teams.get("home") or {}) .get("name"))
            away_name = ((teams.get("away") or {}) .get("name"))
            kickoff_utc = fx_info.get("date")

            if dry_run:
                updated += 1
                continue

            try:
                conn = db_connect()
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE predictions_history
                    SET
                        actual_result = ?,
                        home_team = COALESCE(home_team, ?),
                        away_team = COALESCE(away_team, ?),
                        kickoff_utc = COALESCE(kickoff_utc, ?)
                    WHERE league = ?
                      AND fixture_id = ?
                      AND (actual_result IS NULL OR TRIM(COALESCE(actual_result,'')) = '')
                    """,
                    (actual, home_name, away_name, kickoff_utc, int(league), int(fid_int)),
                )
                conn.commit()
                conn.close()

                if cur.rowcount and cur.rowcount > 0:
                    updated += 1
                else:
                    skipped += 1
                    if debug:
                        skipped_details.append({"fixture_id": fid_int, "reason": "no_rows_updated"})
            except Exception as e:
                errors.append({"fixture_id": fid_int, "error": repr(e)})

        except Exception as e:
            errors.append({"fixture_id": fid_int, "error": repr(e)})

    out = {
        "ok": True,
        "league": league,
        "lookback_days": lookback_days,
        "max_fixtures": max_fixtures,
        "dry_run": dry_run,
        "scanned": scanned,
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:10],
    }
    if debug:
        out["skipped_details"] = skipped_details[:50]
    return out

@app.get("/backtest/1x2")
def api_backtest_1x2(
    league: int = Query(39, description="League ID"),
    season: str = Query(None, description="Season year (e.g. 2025) or \"all\". If omitted, uses current season."),
    last_n: int = Query(200, ge=20, le=2000, description="How many finished fixtures to evaluate (most recent first)"),
    max_goals: int = Query(10, ge=6, le=15, description="Poisson truncation for 1X2 probs"),
    write_db: bool = Query(False, description="If true, write actual_result + model probs into predictions_history"),
    dry_run: bool = Query(False, description="If true, do not write to DB even if write_db=true"),
    sample_limit: int = Query(300, ge=0, le=2000, description="How many per-game rows to include in sample (0 disables sample)"),
    fit_calibration: bool = Query(False, description="Fit & save 1X2 calibration_<league>.json using this backtest set"),
):
    """
    Backtest your model on FINISHED fixtures (FT), computing accuracy + logloss.

    - Fetches last_n finished fixtures via API-FOOTBALL
    - Runs your existing model pipeline to get xG (home_goals, away_goals)
    - Converts xG -> 1X2 probabilities via poisson_1x2_probs()
    - Scores accuracy + logloss
    - Optional: writes results back into predictions_history (so /progress/metrics can work)
    """
    import math
    from datetime import datetime, timezone, timedelta

    model, meta = load_model_and_meta(league)

    # pick season default / allow season=all
    if season is None:
        seasons_used = [current_season()]
        season_label = seasons_used[0]
    elif isinstance(season, str) and season.lower() == "all":
        cs = current_season()
        seasons_used = [cs, cs - 1, cs - 2]
        season_label = "all"
    else:
        seasons_used = [int(season)]
        season_label = seasons_used[0]

    # 1) Fetch finished fixtures (FT) across seasons_used; take the most recent last_n overall.
    fixtures = []
    for s in seasons_used:
        data = api_get("/fixtures", {"league": league, "season": s, "status": "FT"})
        fixtures.extend((data.get("response") or []))
    if not fixtures:
        return {"ok": False, "message": "No finished fixtures found.", "league": league, "season": season_label, "seasons_used": seasons_used}

    # sort newest first
    def _fx_date(fx):
        try:
            return (fx.get("fixture") or {}).get("date") or ""
        except Exception:
            return ""

    fixtures.sort(key=_fx_date, reverse=True)
    fixtures = fixtures[: int(last_n)]

    # 2) Predict on these fixtures using your existing pipeline
    # Use a wide window so nothing is filtered out by date.
    now = datetime.now(timezone.utc)
    window_start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    window_end = now + timedelta(days=1)

    preds = build_predictions_for_fixtures(
        fixtures=fixtures,
        model=model,
        meta=meta,
        league=league,
        season=season,
        window_start=window_start,
        window_end=window_end,
    ) or []

    if not preds:
        return {"ok": False, "message": "No predictions generated for fixtures.", "league": league, "season": season}

    # helper: actual result from fixture goals
    def actual_1x2_from_fixture(fx: dict):
        goals = fx.get("goals") or {}
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            return None
        try:
            hg = int(hg)
            ag = int(ag)
        except Exception:
            return None
        if hg > ag:
            return "home"
        if ag > hg:
            return "away"
        return "draw"

    # helper: find the original fixture record by fixture_id (for actual_result)
    fx_by_id = {}
    for fx in fixtures:
        fid = (fx.get("fixture") or {}).get("id")
        if fid is not None:
            fx_by_id[int(fid)] = fx

    # helper: pull xg values from pred dict (robust to different keys)
    def get_xg(pred: dict):
        # prefer explicit keys
        xg_h = pred.get("xg_home")
        xg_a = pred.get("xg_away")

        # common fallbacks
        if xg_h is None:
            xg_h = pred.get("pred_home_goals") or pred.get("home_goals") or pred.get("xgH")
        if xg_a is None:
            xg_a = pred.get("pred_away_goals") or pred.get("away_goals") or pred.get("xgA")

        # sometimes nested
        p = pred.get("predictions") or {}
        if xg_h is None:
            xg_h = p.get("xg_home") or p.get("home_goals") or p.get("pred_home_goals")
        if xg_a is None:
            xg_a = p.get("xg_away") or p.get("away_goals") or p.get("pred_away_goals")

        try:
            xg_h = float(xg_h) if xg_h is not None else None
            xg_a = float(xg_a) if xg_a is not None else None
        except Exception:
            xg_h, xg_a = None, None

        return xg_h, xg_a

    # 3) Score
    eps = 1e-12
    n = 0
    correct = 0
    logloss_sum = 0.0

    # --- market (odds-implied) baseline metrics (if odds available) ---
    market_n = 0
    market_correct = 0
    market_logloss_sum = 0.0

    # --- CLV (Closing Line Value) using odds_history snapshots ---
    clv_n = 0
    clv_sum_abs = 0.0
    clv_sum_pct = 0.0

    per_game = []

    # DB write counters (for write_db=true)
    db_writes_attempted = 0
    db_writes_ok = 0
    db_write_errors = []

    cal = load_1x2_calibration(league) or {}
    samples_for_cal: list[dict] = []


    for pred in preds:
        fid = pred.get("fixture_id") or (pred.get("fixture") or {}).get("id")
        if fid is None:
            continue
        try:
            fid = int(fid)
        except Exception:
            continue

        fx = fx_by_id.get(fid)
        if not fx:
            continue

        actual = actual_1x2_from_fixture(fx)
        if actual not in ("home", "draw", "away"):
            continue

        xg_home, xg_away = get_xg(pred)
        if not isinstance(xg_home, (int, float)) or not isinstance(xg_away, (int, float)):
            continue

                # ---- RAW (uncalibrated) probs from Poisson on xG ----
        raw = poisson_1x2_probs(xg_home, xg_away, max_goals=int(max_goals)) or {}
        rph = float(raw.get("home", 0.0))
        rpd = float(raw.get("draw", 0.0))
        rpa = float(raw.get("away", 0.0))

        rs = rph + rpd + rpa
        if rs > 0:
            rph, rpd, rpa = rph / rs, rpd / rs, rpa / rs
        else:
            rph = rpd = rpa = 1.0 / 3.0

        if fit_calibration:
            samples_for_cal.append({"actual": actual, "probs": {"home": rph, "draw": rpd, "away": rpa}})

        # ---- CALIBRATED probs (what your API serves) ----
        probs = {"home": rph, "draw": rpd, "away": rpa}
        if cal:
            probs = apply_1x2_calibration(probs, cal) or probs

        ph = float(probs.get("home", 0.0))
        pd = float(probs.get("draw", 0.0))
        pa = float(probs.get("away", 0.0))

        s = ph + pd + pa
        if s > 0:
            ph, pd, pa = ph / s, pd / s, pa / s
        else:
            ph = pd = pa = 1.0 / 3.0

        dist = {"home": ph, "draw": pd, "away": pa}
        pred_side = max(dist, key=dist.get)

        p_true = dist[actual]
        ll = -math.log(max(p_true, eps))


        p_true = {"home": ph, "draw": pd, "away": pa}[actual]
        ll = -math.log(max(p_true, eps))


        # --- Market odds baseline (if available) ---
        market_probs = None
        market_pick = None
        market_ll = None
        mph = mpd = mpa = None
        try:
            odds, odds_meta = fetch_1x2_odds_for_fixture(fid, return_meta=True)
            market_probs = odds_to_implied_probs_1x2(odds) if odds else None
        except Exception:
            market_probs = None

        if market_probs:
            try:
                mph = float(market_probs.get("home", 0.0) or 0.0)
                mpd = float(market_probs.get("draw", 0.0) or 0.0)
                mpa = float(market_probs.get("away", 0.0) or 0.0)
            except Exception:
                mph = mpd = mpa = None

        if isinstance(mph, (int, float)) and isinstance(mpd, (int, float)) and isinstance(mpa, (int, float)):
            ms = float(mph + mpd + mpa)
            if ms > 0:
                mph, mpd, mpa = mph / ms, mpd / ms, mpa / ms
                market_probs = {"home": mph, "draw": mpd, "away": mpa}
                # CLV: compare stored 'pred' snapshot odds vs current odds (treated as close/last)
                try:
                    kickoff = str(((fx.get("fixture") or {}).get("date")) or "")
                    snap_pred = get_odds_snapshot_by_fixture(int(league), int(fid), "pred")
                    if snap_pred and odds:
                        pred_odds_map = {"home": float(snap_pred.get("odds_home")), "draw": float(snap_pred.get("odds_draw")), "away": float(snap_pred.get("odds_away"))}
                        close_odds_map = {"home": float(odds.get("home")), "draw": float(odds.get("draw")), "away": float(odds.get("away"))}
                        po = pred_odds_map.get(pred_side)
                        co = close_odds_map.get(pred_side)
                        if po and co and co > 0:
                            clv_abs = po - co
                            clv_pct = (po / co) - 1.0
                            clv_n += 1
                            clv_sum_abs += clv_abs
                            clv_sum_pct += clv_pct
                except Exception:
                    pass
                try:
                    market_pick = max(market_probs, key=market_probs.get)
                except Exception:
                    market_pick = None
                try:
                    market_ll = -math.log(max(float(market_probs.get(actual, 0.0) or 0.0), eps))
                except Exception:
                    market_ll = None

                market_n += 1
                if market_pick == actual:
                    market_correct += 1
                if isinstance(market_ll, (int, float)):
                    market_logloss_sum += float(market_ll)

        n += 1
        if pred_side == actual:
            correct += 1
        logloss_sum += ll

        per_game.append(
            {
                "fixture_id": fid,
                "kickoff_utc": (fx.get("fixture") or {}).get("date"),
                "home_name": ((fx.get("teams") or {}).get("home") or {}).get("name"),
                "away_name": ((fx.get("teams") or {}).get("away") or {}).get("name"),
                "actual_result": actual,
                "model_pick": pred_side,
                "model_pick_prob": round(dist[pred_side], 4),
                "xg_home": round(xg_home, 3),
                "xg_away": round(xg_away, 3),

                # NEW: raw (pre-calibration)
                "raw_probs": {"home": round(rph, 6), "draw": round(rpd, 6), "away": round(rpa, 6)},

                # NEW: calibrated (what you scored on)
                "calibrated_probs": {"home": round(ph, 6), "draw": round(pd, 6), "away": round(pa, 6)},

                # backward compat (keep name used elsewhere)
                "model_probs": {"home": round(ph, 6), "draw": round(pd, 6), "away": round(pa, 6)},

                # NEW: market implied (from odds)
                "market_probs": (
                    {"home": round(mph, 6), "draw": round(mpd, 6), "away": round(mpa, 6)}
                    if isinstance(mph, (int, float)) and isinstance(mpd, (int, float)) and isinstance(mpa, (int, float))
                    else None
                ),
                "market_pick": market_pick,
                "market_logloss": (round(market_ll, 4) if isinstance(market_ll, (int, float)) else None),

                "logloss": round(ll, 4),
            }
        )


        # 4) Optional: write back into DB so /progress/metrics can work later
        if write_db and not dry_run:
            db_writes_attempted += 1
            try:
                ensure_predictions_db()
                conn = db_connect()
                cur = conn.cursor()

                # Existing columns (schema-flexible)
                cols = [r[1] for r in cur.execute("PRAGMA table_info(predictions_history)").fetchall()]

                # Prepare data to write
                row_data = {}
                def add(col, val):
                    if col in cols:
                        row_data[col] = val

                add("league", int(league))
                add("fixture_id", int(fid))
                add("kickoff_utc", (fx.get("fixture") or {}).get("date"))
                add("home_team", ((fx.get("teams") or {}).get("home") or {}).get("name"))
                add("away_team", ((fx.get("teams") or {}).get("away") or {}).get("name"))

                add("model_home_p", float(ph))
                add("model_draw_p", float(pd))
                add("model_away_p", float(pa))

                # market implied probs + odds (if available)
                if isinstance(mph, (int, float)) and isinstance(mpd, (int, float)) and isinstance(mpa, (int, float)):
                    add("market_home_p", float(mph))
                    add("market_draw_p", float(mpd))
                    add("market_away_p", float(mpa))
                try:
                    if isinstance(odds, dict):
                        oh = odds.get("home")
                        od = odds.get("draw")
                        oa = odds.get("away")
                        if oh is not None and od is not None and oa is not None:
                            add("market_home_odds", float(oh))
                            add("market_draw_odds", float(od))
                            add("market_away_odds", float(oa))
                except Exception:
                    pass

                # Optional: store which bookmaker/bet we used for market odds
                try:
                    bm_name = (odds_meta or {}).get("bookmaker")
                    bet_name = (odds_meta or {}).get("bet_name")
                    if bm_name is not None or bet_name is not None:
                        add("market_bookmaker", bm_name)
                        add("market_bet_name", bet_name)
                except Exception:
                    pass


                add("predicted_side", pred_side)
                add("actual_result", actual)

                # payload if available
                try:
                    add("payload", json.dumps(pred, ensure_ascii=False))
                except Exception:
                    pass

                # ---- Update-first (fills placeholder rows created elsewhere) ----
                # This avoids ON CONFLICT issues and handles cases where kickoff_utc differs in formatting.
                update_cols = []
                update_vals = []
                for c, v in row_data.items():
                    if c in ("id",):
                        continue
                    if c == "actual_result":
                        update_cols.append("actual_result = COALESCE(actual_result, ?)")
                        update_vals.append(v)
                    else:
                        update_cols.append(f"{c} = ?")
                        update_vals.append(v)

                if update_cols:
                    sql_upd = f"UPDATE predictions_history SET {', '.join(update_cols)} WHERE league = ? AND fixture_id = ?"
                    cur.execute(sql_upd, tuple(update_vals) + (int(league), int(fid)))

                # If nothing updated, insert a new row
                if getattr(cur, "rowcount", 0) == 0:
                    insert_cols = [c for c in row_data.keys() if c != "id"]
                    insert_vals = [row_data[c] for c in insert_cols]
                    placeholders = ",".join(["?"] * len(insert_cols))
                    col_sql = ",".join(insert_cols)
                    sql_ins = f"INSERT INTO predictions_history ({col_sql}) VALUES ({placeholders})"
                    cur.execute(sql_ins, tuple(insert_vals))

                conn.commit()
                conn.close()
                db_writes_ok += 1
            except Exception as e:
                try:
                    conn.close()
                except Exception:
                    pass
                db_write_errors.append({"fixture_id": int(fid), "error": repr(e)})

    if n == 0:
        return {"ok": False, "message": "No scorable fixtures (missing goals/xG).", "league": league, "season": season}
    cal_fit_stats = None
    if fit_calibration:
        try:
            raw_ll = _avg_logloss_1x2(samples_for_cal, cal=None)
            cal_new = fit_1x2_calibration(samples_for_cal)
            cal_path = save_1x2_calibration(league, cal_new)
            cal_ll = _avg_logloss_1x2(samples_for_cal, cal=cal_new)
            cal_fit_stats = {
                "saved_to": cal_path,
                "params": cal_new,
                "raw_logloss": (round(raw_ll, 6) if isinstance(raw_ll, (int, float)) else None),
                "cal_logloss": (round(cal_ll, 6) if isinstance(cal_ll, (int, float)) else None),
            }
        except Exception as e:
            try:
                logger.exception("Calibration fit failed")
            except Exception:
                pass
            cal_fit_stats = {"error": repr(e)}

    return {
        "ok": True,
        "league": league,
        "season": season_label,
        "seasons_used": seasons_used,
        "fixtures_scored": n,
        "accuracy": round(correct / n, 4),
        "logloss": round(logloss_sum / n, 4),
        "calibration_fit": cal_fit_stats,
        "market_samples": int(market_n),
        "market_accuracy": (round(market_correct / market_n, 4) if market_n else None),
        "market_logloss": (round(market_logloss_sum / market_n, 4) if market_n else None),
        "delta_logloss_vs_market": (
            round((logloss_sum / n) - (market_logloss_sum / market_n), 5)
            if (n and market_n) else None
        ),
        "clv_n": clv_n,
        "clv_mean_abs": (round(clv_sum_abs / clv_n, 6) if clv_n else None),
        "clv_mean_pct": (round(clv_sum_pct / clv_n, 6) if clv_n else None),
        "write_db": bool(write_db),
        "dry_run": bool(dry_run),
                "fixtures_total": len(fixtures),
        "preds_generated": len(preds),
        "per_game_len": len(per_game),
        "n_used": n,
        "sample_limit": int(sample_limit),
        "sample": (per_game[: min(int(sample_limit), len(per_game))] if int(sample_limit) > 0 else []),
        "db_writes_attempted": (db_writes_attempted if (write_db and not dry_run) else 0),
        "db_writes_ok": (db_writes_ok if (write_db and not dry_run) else 0),
        "db_writes_failed": (len(db_write_errors) if (write_db and not dry_run) else 0),
        "db_write_errors_sample": (db_write_errors[:5] if db_write_errors else []),
    }




@app.post("/odds/snapshot")
def api_odds_snapshot(snap: "OddsSnapshot"):
    """Store an odds snapshot you provide."""
    try:
        record_odds_snapshot(snap)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to record snapshot: {e!r}")


@app.post("/odds/snapshot/fixture/{fixture_id}")
def api_odds_snapshot_from_fixture(
    fixture_id: int = ApiPath(..., description="Fixture ID"),
    snapshot_type: str = Query("pred", description="e.g. pred or close"),
):
    """Fetch current 1X2 odds for a fixture via API-FOOTBALL and store into odds_history."""
    fx_data = api_get("/fixtures", {"id": fixture_id})
    fx_list = (fx_data.get("response") or [])
    if not fx_list:
        raise HTTPException(status_code=404, detail="fixture not found")
    fx = fx_list[0]
    league = int(((fx.get("league") or {}).get("id")) or 0)
    kickoff = (fx.get("fixture") or {}).get("date")
    if not league or not kickoff:
        raise HTTPException(status_code=400, detail="could not infer league/kickoff from fixture")
    odds, meta = fetch_1x2_odds_for_fixture(int(fixture_id), return_meta=True)
    if not odds:
        raise HTTPException(status_code=404, detail="no odds found for fixture")
    snap = OddsSnapshot(
        league=league,
        fixture_id=int(fixture_id),
        kickoff_utc=str(kickoff),
        snapshot_type=str(snapshot_type),
        bookmaker=(meta or {}).get("bookmaker"),
        odds_home=float(odds.get("home")),
        odds_draw=float(odds.get("draw")),
        odds_away=float(odds.get("away")),
    )
    record_odds_snapshot(snap)
    return {"ok": True, "league": league, "fixture_id": int(fixture_id), "snapshot_type": snapshot_type}

@app.get("/progress/metrics")
def progress_metrics(league: int, window_days: int = 60):
    """
    Works for both Postgres (Neon) and SQLite.
    """
    try:
        conn = db_connect()
        cur = conn.cursor()

        # Fetch column list (Postgres first, fallback to SQLite PRAGMA)
        try:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'predictions_history'
                ORDER BY ordinal_position
            """)
            cols = [r[0] for r in cur.fetchall()]
        except Exception:
            cur.execute("PRAGMA table_info(predictions_history)")
            cols = [r[1] for r in cur.fetchall()]

        has_market_cols = all(c in cols for c in (
            "market_home_p","market_draw_p","market_away_p",
            "market_home_odds","market_draw_odds","market_away_odds",
        ))

        # Finished rows in last window_days for the league
        cur.execute("""
            SELECT COUNT(*)
            FROM predictions_history
            WHERE league = %s
              AND kickoff_utc >= (NOW() - (%s || ' days')::interval)
        """, (league, window_days))
        finished_rows = cur.fetchone()[0] or 0

        finished_rows_with_market = 0
        if has_market_cols:
            cur.execute("""
                SELECT COUNT(*)
                FROM predictions_history
                WHERE league = %s
                  AND kickoff_utc >= (NOW() - (%s || ' days')::interval)
                  AND market_home_p IS NOT NULL
                  AND market_draw_p IS NOT NULL
                  AND market_away_p IS NOT NULL
            """, (league, window_days))
            finished_rows_with_market = cur.fetchone()[0] or 0

        cur.close()
        conn.close()

        return {
            "ok": True,
            "league": league,
            "window_days": window_days,
            "has_market_cols": bool(has_market_cols),
            "finished_rows": int(finished_rows),
            "finished_rows_with_market": int(finished_rows_with_market),
        }

    except Exception as e:
        return {"ok": False, "league": league, "window_days": window_days, "error": str(e)}

@app.get("/progress/roi")

def api_progress_roi(
    league: int = Query(39, description="League ID"),
    window_days: int = Query(
        180,
        ge=7,
        le=365,
        description="How many days of history to include"
    ),
    min_edge: float = Query(
        0.05,
        description="Minimum model edge (e.g. 0.05 = 5% edge)"
    ),
) -> Dict[str, Any]:
    """
    Simple ROI-style backtest on stored predictions_history.

    Changes vs previous version:
    - Deduplicates by (fixture_id, league) and uses ONLY the latest row (max id)
      for each fixture. This avoids double-counting when you called /value-bets
      multiple times for the same game.

    Assumptions:
    - We bet 1 unit on the model's predicted_side for each match where:
        * league matches
        * kickoff_utc is within the last `window_days`
        * actual_result is known
        * edge_value >= min_edge
    - Payout model is simplified:
        * Win  -> +1 unit
        * Loss -> -1 unit
    """
    from datetime import datetime, timedelta, timezone

    ensure_predictions_db()

    # Time window lower bound (UTC)
    since_ts = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    try:
        conn = db_connect()
        cur = conn.cursor()
        # We ALSO select the primary key id so we can pick the latest row per fixture
        cur.execute(
            """
            SELECT
                id,
                fixture_id,
                kickoff_utc,
                model_home_p,
                model_draw_p,
                model_away_p,
                predicted_side,
                edge_value,
                actual_result
            FROM predictions_history
            WHERE league = ?
              AND kickoff_utc >= ?
              AND actual_result IS NOT NULL
            ORDER BY id DESC
            """,
            (league, since_ts),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # --- Deduplicate: keep only the latest row for each fixture_id ---
    latest_by_fixture: Dict[int, Tuple] = {}
    for (
        row_id,
        fixture_id,
        kickoff_utc,
        ph,
        pd,
        pa,
        predicted_side,
        edge_value,
        actual_result,
    ) in rows:
        if fixture_id not in latest_by_fixture:
            latest_by_fixture[fixture_id] = (
                row_id,
                fixture_id,
                kickoff_utc,
                ph,
                pd,
                pa,
                predicted_side,
                edge_value,
                actual_result,
            )
        # because we sorted by id DESC, the first time we see a fixture_id
        # is already the latest row, so we can just skip later ones

    bets = 0
    wins = 0
    profit = 0.0

    for (
        row_id,
        fixture_id,
        kickoff_utc,
        ph,
        pd,
        pa,
        predicted_side,
        edge_value,
        actual_result,
    ) in latest_by_fixture.values():
        if predicted_side is None or actual_result is None:
            continue

        try:
            edge = float(edge_value) if edge_value is not None else 0.0
        except (TypeError, ValueError):
            edge = 0.0

        if edge < min_edge:
            continue

        bets += 1

        if predicted_side == actual_result:
            wins += 1
            profit += 1.0
        else:
            profit -= 1.0

    if bets == 0:
        return {
            "ok": True,
            "league": league,
            "window_days": window_days,
            "min_edge": min_edge,
            "bets": 0,
            "wins": 0,
            "hit_rate": None,
            "profit_units": 0.0,
            "roi": None,
            "message": "No bets matching the filters (min_edge/window_days).",
        }

    hit_rate = wins / bets
    roi = profit / bets

    return {
        "ok": True,
        "league": league,
        "window_days": window_days,
        "min_edge": min_edge,
        "bets": bets,
        "wins": wins,
        "hit_rate": round(hit_rate, 3),
        "profit_units": round(profit, 3),
        "roi": round(roi, 3),
        "generated": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/predict/by-date")
def api_predict_by_date(
    league: int = Query(DEFAULT_LEAGUE),
    from_date: Optional[str] = Query(None, description="YYYY-MM-DD start date"),
    to_date: Optional[str] = Query(None, description="YYYY-MM-DD end date (inclusive)"),
    date: Optional[str] = Query(None, description="YYYY-MM-DD (alias for from_date=to_date)"),
):
    # Backward-compatible: allow ?date=YYYY-MM-DD (maps to from_date=to_date)
    if (not from_date and not to_date) and date:
        from_date = date
        to_date = date
    if not from_date:
        raise HTTPException(status_code=422, detail="Missing required query params: from_date & to_date (or date=YYYY-MM-DD)")

    window_start, window_end, from_str, to_str = parse_date_range_or_400(from_date, to_date)
    try:
        model, meta = load_model_and_meta(league)
    except HTTPException:
        snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=MAX_DATE_RANGE_DAYS)
        if snapshot:
            filtered = []
            for fx in snapshot:
                kickoff = fx.get("kickoff_utc")
                if not kickoff:
                    continue
                try:
                    kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                except Exception:
                    continue
                if window_start <= kickoff_dt <= window_end:
                    filtered.append(fx)
            if filtered:
                return {"ok": True, "count": len(filtered), "range": {"from": from_str, "to": to_str},
                        "fixtures": filtered, "source": "snapshot",
                        "snapshot_file": os.path.basename(snap_path) if snap_path else None}
        raise

    season = current_season()
    try:
        data = api_get("/fixtures", {"league": league, "season": season, "from": from_str, "to": to_str})
        fixtures = data.get("response", []) or []
    except HTTPException as exc:
        if exc.status_code in (500, 502, 503):
            fixtures = []
            logger.warning("[PREDICT BY DATE] API unavailable (%s). Falling back to cached fixtures.", exc.detail)
        else:
            raise

    if not fixtures:
        cached_fixtures = cached_upcoming_fixtures(league, season)
        if cached_fixtures:
            fixtures = filter_fixtures_by_window(cached_fixtures, window_start, window_end)
            if fixtures:
                logger.info("[PREDICT BY DATE] served from cached upcoming fixtures league=%s", league)

    results = build_predictions_for_fixtures(
        fixtures=fixtures, model=model, meta=meta, league=league, season=season,
        window_start=window_start, window_end=window_end
    )
    if results:
        record_predictions_history(league, results)
        return {"ok": True, "count": len(results), "range": {"from": from_str, "to": to_str}, "fixtures": results, "source": "model"}

    snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=MAX_DATE_RANGE_DAYS)
    if snapshot:
        filtered = []
        for fx in snapshot:
            kickoff = fx.get("kickoff_utc")
            if not kickoff:
                continue
            try:
                kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            except Exception:
                continue
            if window_start <= kickoff_dt <= window_end:
                filtered.append(fx)
        if filtered:
            return {"ok": True, "count": len(filtered), "range": {"from": from_str, "to": to_str},
                    "fixtures": filtered, "source": "snapshot",
                    "snapshot_file": os.path.basename(snap_path) if snap_path else None}

    return {"ok": False, "count": 0, "range": {"from": from_str, "to": to_str}, "fixtures": [],
            "detail": "No fixtures available for the requested window."}

@app.get("/history")
def api_history(
    league: int = Query(DEFAULT_LEAGUE),
    limit: int = Query(50, ge=1, le=500),
):
    """
    Return recent predictions from predictions_history for a given league,
    deduplicated so you only see ONE row per fixture (the latest).
    """
    try:
        ensure_predictions_db()

        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                fixture_id,
                league,
                home_team,
                away_team,
                kickoff_utc,
                model_home_p,
                model_draw_p,
                model_away_p,
                predicted_side,
                edge_value,
                actual_result
            FROM predictions_history
            WHERE league = ?
            ORDER BY kickoff_utc DESC, id DESC
            """,
            (league,),
        )
        rows = cur.fetchall()
        conn.close()

        latest_by_fixture = {}
        for (
            row_id,
            fixture_id,
            league_id,
            home_team,
            away_team,
            kickoff_utc,
            ph,
            pd,
            pa,
            predicted_side,
            edge_value,
            actual_result,
        ) in rows:
            if fixture_id in latest_by_fixture:
                continue
            latest_by_fixture[fixture_id] = {
                "id": row_id,
                "fixture_id": fixture_id,
                "league": league_id,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_utc": kickoff_utc,
                "model_home_p": ph,
                "model_draw_p": pd,
                "model_away_p": pa,
                "predicted_side": predicted_side,
                "edge_value": edge_value,
                "actual_result": actual_result,
            }

        fixtures = list(latest_by_fixture.values())
        fixtures.sort(key=lambda f: f["kickoff_utc"], reverse=True)
        fixtures = fixtures[:limit]

        return {"ok": True, "count": len(fixtures), "fixtures": fixtures}
    except Exception as e:
        logger.error("History fetch failed: %s", e)
        return {"ok": False, "count": 0, "fixtures": [], "error": str(e)}



@app.get("/metrics/pnl-history")
def api_pnl_history(
    league: int = Query(39, description="League ID, e.g. 39 = Premier League"),
    min_edge: float = Query(
        0.0,
        description="Minimum model edge to include (e.g. 0.05 = 5% edge)",
    ),
):
    """
    Compute a simple PnL history from predictions_history.

    - 1 unit flat stake per finished prediction
    - Bet side = predicted_side
    - Win  -> +1 unit
    - Loss -> -1 unit

    Filters:
      * league matches
      * actual_result IS NOT NULL
      * edge_value >= min_edge (after casting to float, NULL -> 0.0)
    """
    ensure_predictions_db()

    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                fixture_id,
                home_team,
                away_team,
                kickoff_utc,
                predicted_side,
                edge_value,
                actual_result
            FROM predictions_history
            WHERE league = ?
              AND actual_result IS NOT NULL
            ORDER BY kickoff_utc ASC
            """,
            (league,),
        )
        rows = cur.fetchall()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    n_bets = 0
    wins = 0
    cum_profit = 0.0
    points = []

    for (
        fixture_id,
        home_team,
        away_team,
        kickoff_utc,
        predicted_side,
        edge_value,
        actual_result,
    ) in rows:
        if not predicted_side or not actual_result:
            continue

        # Turn edge_value into a float; NULL or bad values -> 0.0
        try:
            edge = float(edge_value) if edge_value is not None else 0.0
        except (TypeError, ValueError):
            edge = 0.0

        # 🔑 Apply the min_edge filter
        if edge < min_edge:
            continue

        win = (actual_result == predicted_side)
        profit = 1.0 if win else -1.0

        n_bets += 1
        if win:
            wins += 1

        cum_profit += profit
        roi_so_far = cum_profit / n_bets if n_bets else 0.0

        points.append(
            {
                "index": n_bets,
                "fixture_id": fixture_id,
                "kickoff_utc": kickoff_utc,
                "home_team": home_team,
                "away_team": away_team,
                "bet_side": predicted_side,
                "actual_result": actual_result,
                "edge_value": edge,
                "win": win,
                "profit": profit,
                "cum_profit": cum_profit,
                "roi": roi_so_far,
                "stake_flat": 1.0,
            }
        )

    if n_bets == 0:
        return {
            "ok": True,
            "league": league,
            "n_bets": 0,
            "wins": 0,
            "total_profit": 0.0,
            "roi_flat": 0.0,
            "points": [],
            "min_edge": min_edge,
        }

    roi_flat = cum_profit / n_bets

    return {
        "ok": True,
        "league": league,
        "n_bets": n_bets,
        "wins": wins,
        "total_profit": round(cum_profit, 3),
        "roi_flat": round(roi_flat, 3),
        "points": points,
        "min_edge": min_edge,
    }


@app.get("/metrics/roi-by-league")
def api_roi_by_league(
    min_edge: float = Query(0.0, description="Minimum edge filter on edge_value")
):
    """
    Flat-stake ROI per league based on predictions_history.
    1 unit per bet; +1 win, -1 loss.
    """
    ensure_predictions_db()

    conn = db_connect()
    cur = conn.cursor()

    edge_filter = ""
    params = []
    if min_edge > 0:
        edge_filter = "AND edge_value IS NOT NULL AND edge_value >= ?"
        params.append(min_edge)

    cur.execute(
        f"""
        SELECT
            league,
            COUNT(*) AS n_bets,
            SUM(CASE WHEN actual_result = predicted_side THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN actual_result = predicted_side THEN 1 ELSE -1 END) AS total_profit
        FROM predictions_history
        WHERE actual_result IS NOT NULL
        {edge_filter}
        GROUP BY league
        ORDER BY CAST(total_profit AS FLOAT) / COUNT(*) DESC
        """
        ,
        params,
    )

    rows = cur.fetchall()
    conn.close()

    leagues = []
    for league, n_bets, wins, total_profit in rows:
        roi_flat = float(total_profit) / float(n_bets) if n_bets else 0.0
        leagues.append(
            {
                "league": league,
                "n_bets": n_bets,
                "wins": wins,
                "total_profit": total_profit,
                "roi_flat": roi_flat,
            }
        )

    return {"ok": True, "leagues": leagues, "min_edge": min_edge}

@app.get("/metrics/pnl-debug")
def api_pnl_debug(
    league: int = Query(39),
    limit: int = Query(20, description="How many rows to inspect"),
):
    """
    Debug view for finished bets in predictions_history.

    - ONE row per fixture (latest row per fixture_id)
    - Only rows with actual_result IS NOT NULL
    """
    ensure_predictions_db()
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            fixture_id,
            home_team,
            away_team,
            kickoff_utc,
            predicted_side,
            edge_value,
            actual_result
        FROM predictions_history
        WHERE league = ?
          AND actual_result IS NOT NULL
        ORDER BY kickoff_utc DESC, id DESC
        """,
        (league,),
    )
    rows = cur.fetchall()
    conn.close()

    latest_by_fixture = {}
    for (
        row_id,
        fixture_id,
        home_team,
        away_team,
        kickoff_utc,
        predicted_side,
        edge_value,
        actual_result,
    ) in rows:
        if fixture_id in latest_by_fixture:
            continue
        latest_by_fixture[fixture_id] = {
            "id": row_id,
            "fixture_id": fixture_id,
            "kickoff_utc": kickoff_utc,
            "home_team": home_team,
            "away_team": away_team,
            "predicted_side": predicted_side,
            "actual_result": actual_result,
            "edge_value": edge_value,
            "correct": predicted_side == actual_result,
        }

    samples = list(latest_by_fixture.values())
    samples.sort(key=lambda r: r["kickoff_utc"], reverse=True)
    samples = samples[:limit]

    return {"ok": True, "league": league, "n": len(samples), "samples": samples}



@app.get("/metrics/predictions-sanity")
def api_predictions_sanity(
    league: int = Query(39, description="League ID, e.g. 39 = Premier League"),
):
    """
    Sanity check for predictions_history for a given league.

    Returns TWO views:
    - per-row metrics  : every DB row counts
    - per-fixture metrics : only the LATEST row per fixture_id counts

    This helps you see the impact of scanning the same game multiple times.
    """
    ensure_predictions_db()

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            fixture_id,
            predicted_side,
            actual_result
        FROM predictions_history
        WHERE league = ?
          AND actual_result IS NOT NULL
        ORDER BY id DESC
        """,
        (league,),
    )
    rows = cur.fetchall()
    conn.close()

    # ---------- PER-ROW ----------
    total_rows = len(rows)
    correct_rows = 0
    for (_row_id, _fixture_id, pred, actual) in rows:
        if pred and actual and pred == actual:
            correct_rows += 1
    accuracy_rows = (correct_rows / total_rows) if total_rows else 0.0

    # ---------- PER-FIXTURE (LATEST ROW ONLY) ----------
    latest_by_fixture = {}
    for (
        row_id,
        fixture_id,
        predicted_side,
        actual_result,
    ) in rows:
        # because we ordered by id DESC, first time we see fixture_id is the latest
        if fixture_id in latest_by_fixture:
            continue
        latest_by_fixture[fixture_id] = (predicted_side, actual_result)

    total_fixtures = len(latest_by_fixture)
    correct_fixtures = 0
    for fixture_id, (pred, actual) in latest_by_fixture.items():
        if pred and actual and pred == actual:
            correct_fixtures += 1
    accuracy_fixtures = (correct_fixtures / total_fixtures) if total_fixtures else 0.0

    return {
        "ok": True,
        "league": league,
        # per-row view
        "total_rows": total_rows,
        "correct_rows": correct_rows,
        "accuracy_rows": accuracy_rows,
        # per-fixture view
        "total_fixtures": total_fixtures,
        "correct_fixtures": correct_fixtures,
        "accuracy_fixtures": accuracy_fixtures,
    }

      

@app.get(
    "/debug/fixture/{fixture_id}",
    response_description="""
    Debug a single fixture's saved record(s).

    Note: this is best-effort and only reads from the local SQLite history DB.
    """,
)
def debug_fixture(fixture_id: int, league: int, include_history: bool = True):
    db_path = HISTORY_DB_PATH
    if not os.path.exists(db_path):
        return {"ok": False, "error": f"History DB not found: {db_path}", "fixture_id": fixture_id, "league": league}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT
                id,
                fixture_id,
                league,
                home_team,
                away_team,
                kickoff_utc,
                model_home_p,
                model_draw_p,
                model_away_p,
                predicted_side,
                edge_value,
                actual_result,
                market_home_p,
                market_draw_p,
                market_away_p,
                market_home_odds,
                market_draw_odds,
                market_away_odds,
                market_bookmaker,
                market_bet_name
            FROM predictions_history
            WHERE fixture_id = ? AND league = ?
            ORDER BY id DESC
            """,
            (fixture_id, league),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"ok": False, "error": "no records found", "fixture_id": fixture_id, "league": league}

    (
        row_id,
        fx_id,
        lg,
        home_team,
        away_team,
        kickoff_utc,
        ph,
        pd,
        pa,
        predicted_side,
        edge_value,
        actual_result,
        mkh,
        mkd,
        mka,
        moh,
        mod,
        moa,
        mbm,
        mbet,
    ) = rows[0]

    latest = {
        "fixture_id": fx_id,
        "league": lg,
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_utc": kickoff_utc,
        "model_probs": {"home": ph, "draw": pd, "away": pa} if ph is not None else None,
        "predicted_side": predicted_side,
        "edge_value": edge_value,
        "actual_result": actual_result,
    }

    if mkh is not None and mkd is not None and mka is not None:
        latest["market_probs"] = {"home": mkh, "draw": mkd, "away": mka}
    if moh is not None and mod is not None and moa is not None:
        latest["market_odds"] = {"home": moh, "draw": mod, "away": moa}
    if (mbm is not None and str(mbm).strip()) or (mbet is not None and str(mbet).strip()):
        latest["market_meta"] = {"bookmaker": mbm, "bet_name": mbet}

    history = []
    if include_history:
        for (
            row_id,
            fx_id,
            lg,
            home_team,
            away_team,
            kickoff_utc,
            ph,
            pd,
            pa,
            predicted_side,
            edge_value,
            actual_result,
            mkh,
            mkd,
            mka,
            moh,
            mod,
            moa,
            mbm,
            mbet,
        ) in rows:
            history.append(
                {
                    "row_id": row_id,
                    "fixture_id": fx_id,
                    "league": lg,
                    "home_team": home_team,
                    "away_team": away_team,
                    "kickoff_utc": kickoff_utc,
                    "model_home_p": ph,
                    "model_draw_p": pd,
                    "model_away_p": pa,
                    "predicted_side": predicted_side,
                    "edge_value": edge_value,
                    "actual_result": actual_result,
                    "market_home_p": mkh,
                    "market_draw_p": mkd,
                    "market_away_p": mka,
                    "market_home_odds": moh,
                    "market_draw_odds": mod,
                    "market_away_odds": moa,
                    "market_bookmaker": mbm,
                    "market_bet_name": mbet,
                }
            )

    return {"ok": True, "fixture_id": fixture_id, "league": league, "latest": latest, "num_records": len(rows), "history": history}


@app.get("/debug/odds-scan/{fixture_id}")
def debug_odds_scan(fixture_id: int):
    """
    Fetch /odds for a fixture and explain why we did/didn't find a complete 1X2 set.
    """
    try:
        payload = api_get("/odds", {"fixture": fixture_id})
    except Exception as e:
        return {"ok": False, "fixture_id": fixture_id, "error": str(e)}

    scan = scan_market_odds_1x2(payload, max_notes=80)
    return {
        "ok": True,
        "fixture_id": fixture_id,
        "api": {"results": (payload or {}).get("results"), "errors": (payload or {}).get("errors")},
        "scan": scan,
    }

@app.get("/debug/leagues")
def debug_leagues():
    """
    TEMP: list all current-season leagues from API-FOOTBALL
    so you can pick the 40 you want.
    """
    data = api_get("/leagues", {"current": "true"})
    out = []
    for row in data.get("response", []) or []:
        league_obj = row.get("league", {}) or {}
        country_obj = row.get("country", {}) or {}

        out.append({
            "id": league_obj.get("id"),
            "name": league_obj.get("name"),
            "type": league_obj.get("type"),
            "country": country_obj.get("name"),
        })

    # Sort nicely by country then league name
    out.sort(key=lambda x: ((x["country"] or ""), (x["name"] or "")))
    return out

@app.get("/debug/pending-results")
def debug_pending_results(
    league: int = Query(39, description="League ID, e.g. 39 = Premier League"),
    limit: int = Query(50, description="How many pending fixtures to show"),
):
    """
    Show fixtures in predictions_history that have NO actual_result yet
    for a given league. Helps debug why /update-results didn't update anything.
    """
    ensure_predictions_db()

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            fixture_id,
            home_team,
            away_team,
            kickoff_utc,
            actual_result
        FROM predictions_history
        WHERE league = ?
          AND (actual_result IS NULL OR actual_result = '')
        ORDER BY kickoff_utc ASC
        LIMIT ?
        """,
        (league, limit),
    )
    rows = cur.fetchall()
    conn.close()

    fixtures = []
    for fixture_id, home_team, away_team, kickoff_utc, actual_result in rows:
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_utc": kickoff_utc,
                "actual_result": actual_result,
            }
        )

    return {
        "ok": True,
        "league": league,
        "pending": len(fixtures),
        "fixtures": fixtures,
    }


# =========================================================
# 📊 RECENT RESULTS (for Results page)
# =========================================================
from datetime import datetime, timezone, timedelta  # make sure this is imported at top


@app.get("/results/recent")
def api_recent_results(league: int = 39, limit: int = 20):
    """
    Return the most recent fixtures with known results from predictions_history.
    Used by static/results.html.

    Must NEVER 500: always return JSON with ok: true/false.
    Logos are best-effort.
    """
    try:
        ensure_predictions_db()
        limit = max(1, min(int(limit), 100))

        conn = db_connect()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT fixture_id, league, kickoff_utc, home_team, away_team, predicted_side, actual_result
            FROM predictions_history
            WHERE league = ?
              AND actual_result IS NOT NULL
              AND TRIM(actual_result) <> ''
            ORDER BY kickoff_utc DESC
            LIMIT ?
            """,
            (league, limit),
        )
        rows = cur.fetchall()
        conn.close()

        fixtures = []
        for fixture_id, league_id, kickoff, home, away, predicted, actual in rows:
            # Super-safe: even if helper is broken, don't fail endpoint
            try:
                home_logo, away_logo = get_fixture_logos(int(fixture_id))
            except Exception:
                home_logo, away_logo = None, None

            fixtures.append({
                "fixture_id": fixture_id,
                "league": league_id,
                "kickoff_utc": kickoff,
                "home_team": home,
                "away_team": away,
                "predicted_side": predicted,
                "actual_result": actual,
                "home_logo": home_logo,
                "away_logo": away_logo,
            })

        return {"ok": True, "league": league, "limit": limit, "fixtures": fixtures}

    except Exception as e:
        try:
            logger.exception("[RESULTS] /results/recent failed")
        except Exception:
            pass
        return {"ok": False, "league": league, "limit": limit, "fixtures": [], "error": str(e)}

@app.get("/update-results", dependencies=[Depends(require_admin)])
def api_update_results(
    league: int = Query(39, description="League ID"),
    max_updates: int = Query(50, ge=1, le=200, description="Max fixtures to update in one call"),
):
    """
    Checks API-FOOTBALL for finished fixtures and updates their actual_result
    in predictions_history.
    """
    ensure_predictions_db()

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT fixture_id
        FROM predictions_history
        WHERE league = ?
          AND (actual_result IS NULL OR actual_result = '')
        """,
        (league,),
    )
    rows = cur.fetchall()
    conn.close()

    pending_ids = [r[0] for r in rows]
    if not pending_ids:
        return {"ok": True, "updated": 0, "message": "No fixtures pending."}

    updated = 0

    conn = db_connect()
    cur = conn.cursor()

    for fx_id in pending_ids:
        if updated >= max_updates:
            break

        try:
            data = api_get("/fixtures", {"id": fx_id})
        except HTTPException as e:
            logger.warning("[UPDATE] API error for fixture %s: %s", fx_id, e.detail)
            continue

        resp = data.get("response") or []
        if not resp:
            continue

        row = resp[0]
        fixture = row.get("fixture", {}) or {}
        status = (fixture.get("status") or {}).get("short")

        # Only finished statuses
        if status not in ("FT", "AET", "PEN"):
            continue

        goals = row.get("goals", {}) or {}
        home_goals = goals.get("home")
        away_goals = goals.get("away")

        if home_goals is None or away_goals is None:
            continue

        if home_goals > away_goals:
            result = "home"
        elif away_goals > home_goals:
            result = "away"
        else:
            result = "draw"

        cur.execute(
            """
            UPDATE predictions_history
            SET actual_result = ?
            WHERE fixture_id = ? AND league = ?
            """,
            (result, fx_id, league),
        )
        conn.commit()
        updated += 1
        logger.info("[UPDATE] Fixture %s marked as %s", fx_id, result)

    conn.close()
    return {
        "ok": True,
        "updated": updated,
        "message": f"Updated {updated} finished fixtures for league {league}.",
    }


# === PATCH FIX: _within_odds_window override (prevents datetime.datetime bug) ===

from datetime import datetime as _dt, timezone as _tz, timedelta as _td

def _within_odds_window(kickoff_utc) -> bool:
    """
    True if kickoff is within the odds-fetch window.
    Accepts str (ISO), datetime, or None.
    This overrides a buggy duplicate definition later in the file that used `datetime.datetime`.
    """
    try:
        if kickoff_utc is None:
            return False

        if isinstance(kickoff_utc, str):
            dt_str = kickoff_utc.strip().replace("Z", "+00:00")
            ko = _dt.fromisoformat(dt_str)
        else:
            ko = kickoff_utc

        # Ensure timezone-aware UTC
        if getattr(ko, "tzinfo", None) is None:
            ko = ko.replace(tzinfo=_tz.utc)
        ko = ko.astimezone(_tz.utc)

        now = _dt.now(_tz.utc)
        # allow a small “just started” grace window + configurable lookback/future
        return (now - _td(days=int(ODDS_LOOKBACK_DAYS))) <= ko <= (now + _td(days=int(ODDS_FUTURE_DAYS)))
    except Exception:
        return False

@app.get("/debug/history-count")
def debug_history_count(league: int | None = None):
    import sqlite3
    try:
        conn = db_connect()
        cur = conn.cursor()
        if league is None:
            cur.execute("SELECT COUNT(*) FROM predictions_history")
            n = cur.fetchone()[0]
            conn.close()
            return {"ok": True, "predictions_history_count": int(n)}
        else:
            cur.execute("SELECT COUNT(*) FROM predictions_history WHERE league = ?", (int(league),))
            n = cur.fetchone()[0]
            conn.close()
            return {"ok": True, "league": int(league), "predictions_history_count": int(n)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/admin/backfill-history")
def admin_backfill_history(
    league: int = 39,
    window_days: int = 60,
    limit: int = 5000,
    dry_run: bool = False,
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
):
    """
    Backfill structured columns from payload JSON for rows missing market/model fields.
    This lets /progress/metrics compute market baseline immediately.
    """
    require_admin(x_admin_token)

    ensure_predictions_db()

    conn = db_connect()
    cur = conn.cursor()

    # Which columns exist?
    cur.execute("PRAGMA table_info(predictions_history);")
    cols = {row[1] for row in cur.fetchall()}

    need_cols = {"payload", "market_home_p", "market_draw_p", "market_away_p", "model_home_p", "model_draw_p", "model_away_p"}
    if not need_cols.issubset(cols):
        conn.close()
        return {"ok": False, "error": f"Missing required columns: {sorted(need_cols - cols)}"}

    cutoff = (datetime.utcnow() - timedelta(days=int(window_days))).isoformat()

    cur.execute(
        """
        SELECT id, payload
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND payload IS NOT NULL
          AND (
            market_home_p IS NULL OR market_draw_p IS NULL OR market_away_p IS NULL
            OR model_home_p IS NULL OR model_draw_p IS NULL OR model_away_p IS NULL
            OR predicted_side IS NULL
          )
        ORDER BY kickoff_utc DESC
        LIMIT ?
        """,
        (int(league), cutoff, int(limit)),
    )
    rows = cur.fetchall()

    updated = 0
    errors = 0

    for row_id, payload in rows:
        try:
            f = json.loads(payload)

            preds = f.get("predictions") or {}
            odds = f.get("odds_1x2") or {}
            implied = f.get("implied_1x2") or {}
            v_edges = f.get("value_edges") or {}

            predicted_side = (preds.get("best_side") or "").strip().lower() or None

            edge_value = None
            if predicted_side in ("home", "draw", "away"):
                try:
                    edge_value = float(v_edges.get(predicted_side)) if v_edges.get(predicted_side) is not None else None
                except Exception:
                    edge_value = None

            sets = []
            vals = []

            def set_if(col, val):
                if col in cols:
                    sets.append(f"{col} = COALESCE({col}, ?)")
                    vals.append(val)

            set_if("home_team", f.get("home_name") or f.get("home_team"))
            set_if("away_team", f.get("away_name") or f.get("away_team"))

            set_if("model_home_p", preds.get("home_win_p"))
            set_if("model_draw_p", preds.get("draw_p"))
            set_if("model_away_p", preds.get("away_win_p"))

            set_if("predicted_side", predicted_side)
            set_if("edge_value", edge_value)

            set_if("market_home_p", implied.get("home"))
            set_if("market_draw_p", implied.get("draw"))
            set_if("market_away_p", implied.get("away"))

            set_if("market_home_odds", odds.get("home"))
            set_if("market_draw_odds", odds.get("draw"))
            set_if("market_away_odds", odds.get("away"))

            set_if("value_side", f.get("value_side"))
            set_if("best_edge", f.get("best_edge"))

            if sets and not dry_run:
                cur.execute(f"UPDATE predictions_history SET {', '.join(sets)} WHERE id = ?", tuple(vals) + (row_id,))
                updated += 1

        except Exception:
            errors += 1

    if not dry_run:
        conn.commit()
    conn.close()

    return {
        "ok": True,
        "league": league,
        "window_days": window_days,
        "scanned": len(rows),
        "updated": updated,
        "errors": errors,
        "dry_run": dry_run,
    }

@app.get("/debug/market-samples")
def debug_market_samples(league: int = 39, window_days: int = 60):
    import sqlite3
    from datetime import datetime, timedelta

    ensure_predictions_db()
    cutoff = (datetime.utcnow() - timedelta(days=int(window_days))).isoformat()

    conn = db_connect()
    cur = conn.cursor()

    # total finished rows in window
    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND actual_result IS NOT NULL
        """,
        (int(league), cutoff),
    )
    finished = int(cur.fetchone()[0])

    # finished rows with complete market probs
    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND actual_result IS NOT NULL
          AND market_home_p IS NOT NULL
          AND market_draw_p IS NOT NULL
          AND market_away_p IS NOT NULL
        """,
        (int(league), cutoff),
    )
    with_market = int(cur.fetchone()[0])

    conn.close()
    return {
        "ok": True,
        "league": league,
        "window_days": window_days,
        "finished_rows": finished,
        "finished_rows_with_market": with_market,
    }

@app.post("/admin/backfill-market-from-payload")
def admin_backfill_market_from_payload(
    league: int = 39,
    window_days: int = 120,
    limit: int = 5000,
    dry_run: bool = False,
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    ensure_predictions_db()

    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(days=int(window_days))).isoformat()

    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, payload
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND payload IS NOT NULL
          AND (
            market_home_p IS NULL OR market_draw_p IS NULL OR market_away_p IS NULL
            OR market_home_odds IS NULL OR market_draw_odds IS NULL OR market_away_odds IS NULL
            OR predicted_side IS NULL
            OR model_home_p IS NULL OR model_draw_p IS NULL OR model_away_p IS NULL
          )
        ORDER BY kickoff_utc DESC
        LIMIT ?
        """,
        (int(league), cutoff, int(limit)),
    )
    rows = cur.fetchall()

    updated = 0

    skipped = 0

    error_samples = []  # first N errors

    skipped = 0
    errors = 0

    for row_id, payload in rows:
        try:
            try:
                f = json.loads(payload)
            except Exception:
                import ast
                f = ast.literal_eval(payload)
        except Exception as e:
            errors += 1
            if len(error_samples) < 10:
                error_samples.append({"stage":"payload_parse","err":str(e),"row_id":row_id})
            continue

            implied = f.get("implied_1x2") or {}

            predicted_side = (preds.get("best_side") or "").strip().lower() or None

            sets = []
            vals = []

            def put(col, val):
                sets.append(f"{col} = COALESCE({col}, ?)")
                vals.append(val)

            put("model_home_p", preds.get("home_win_p"))
            put("model_draw_p", preds.get("draw_p"))
            put("model_away_p", preds.get("away_win_p"))
            put("predicted_side", predicted_side)

            put("market_home_p", implied.get("home"))
            put("market_draw_p", implied.get("draw"))
            put("market_away_p", implied.get("away"))

            put("market_home_odds", odds.get("home"))
            put("market_draw_odds", odds.get("draw"))
            put("market_away_odds", odds.get("away"))

            if not dry_run:
                cur.execute(
                    f"UPDATE predictions_history SET {', '.join(sets)} WHERE id = ?",
                    tuple(vals) + (row_id,),
                )
            updated += 1  # only counted when we applied a non-empty update
        except Exception as e:
            errors += 1

    if not dry_run:
        conn.commit()
    conn.close()

    return {"ok": True, "league": league, "window_days": window_days, "scanned": len(rows), "updated": updated, "errors": errors, "skipped": skipped, "dry_run": dry_run,
        "error_samples": error_samples
    }

@app.get("/debug/market-finished-sample")
def debug_market_finished_sample(league: int = 39, window_days: int = 60, limit: int = 10):
    from datetime import datetime, timedelta
    ensure_predictions_db()
    cutoff = (datetime.utcnow() - timedelta(days=int(window_days))).isoformat()

    conn = db_connect()
    cur = conn.cursor()

    # Counts
    cur.execute("""
        SELECT COUNT(*)
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND actual_result IS NOT NULL
    """, (int(league), cutoff))
    finished = int(cur.fetchone()[0])

    cur.execute("""
        SELECT COUNT(*)
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND actual_result IS NOT NULL
          AND market_home_p IS NOT NULL
          AND market_draw_p IS NOT NULL
          AND market_away_p IS NOT NULL
    """, (int(league), cutoff))
    finished_with_market = int(cur.fetchone()[0])

    # Sample rows to inspect what is NULL
    cur.execute("""
        SELECT id, fixture_id, kickoff_utc, actual_result,
               market_home_p, market_draw_p, market_away_p,
               market_home_odds, market_draw_odds, market_away_odds
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND actual_result IS NOT NULL
        ORDER BY kickoff_utc DESC
        LIMIT ?
    """, (int(league), cutoff, int(limit)))
    sample = [
        {
            "id": r[0], "fixture_id": r[1], "kickoff_utc": r[2], "actual_result": r[3],
            "market_p": {"home": r[4], "draw": r[5], "away": r[6]},
            "market_odds": {"home": r[7], "draw": r[8], "away": r[9]},
        }
        for r in cur.fetchall()
    ]

    conn.close()
    return {
        "ok": True,
        "league": league,
        "window_days": window_days,
        "finished_rows": finished,
        "finished_rows_with_market": finished_with_market,
        "sample": sample,
    }

@app.get("/debug/versions")
def debug_versions():
    import sys
    import sklearn
    return {
        "python": sys.version,
        "sklearn": sklearn.__version__,
    }

@app.get("/debug/build-info")
def debug_build_info():
    import os, sys
    try:
        import sklearn
        sklearn_v = sklearn.__version__
    except Exception:
        sklearn_v = None

    sha = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT") or None
    return {"python": sys.version, "sklearn": sklearn_v, "commit": sha}

@app.get("/__versions")
def versions_root():
    import sys
    import sklearn
    return {"python": sys.version, "sklearn": sklearn.__version__,
        "skipped": skipped,
        "error_samples": error_samples
    }

@app.get("/debug/payload-market-stats")
def debug_payload_market_stats(league: int = 39, window_days: int = 180):
    import datetime as dt
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).isoformat()

    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM predictions_history WHERE league = ? AND kickoff_utc >= ?",
        (int(league), cutoff),
    )
    total = int(cur.fetchone()[0])

    # Robust: treat payload as text and count presence of keys
    cur.execute(
        "SELECT COUNT(*) FROM predictions_history WHERE league = ? AND kickoff_utc >= ? AND payload LIKE '%\"odds_1x2\"%'",
        (int(league), cutoff),
    )
    has_odds_key = int(cur.fetchone()[0])

    cur.execute(
        "SELECT COUNT(*) FROM predictions_history WHERE league = ? AND kickoff_utc >= ? AND payload LIKE '%\"implied_1x2\"%'",
        (int(league), cutoff),
    )
    has_implied_key = int(cur.fetchone()[0])

    conn.close()
    return {
        "ok": True,
        "league": league,
        "window_days": window_days,
        "rows_total": total,
        "payload_has_odds_1x2_key": has_odds_key,
        "payload_has_implied_1x2_key": has_implied_key,
    }

def debug_payload_market_stats(league: int = 39, window_days: int = 180):
    """
    Quick sanity: do we even have odds/implied stored inside payload JSON strings?
    Uses SQL LIKE (no JSON parsing needed).
    """
    import datetime as _dt

    cutoff = (_dt.datetime.utcnow() - _dt.timedelta(days=int(window_days))).isoformat()

    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM predictions_history WHERE league=? AND kickoff_utc >= ?",
        (int(league), cutoff),
    )
    total = int(cur.fetchone()[0])

    cur.execute(
        "SELECT COUNT(*) FROM predictions_history WHERE league=? AND kickoff_utc >= ? AND payload LIKE '%\"odds_1x2\"%'",
        (int(league), cutoff),
    )
    has_odds_key = int(cur.fetchone()[0])

    cur.execute(
        "SELECT COUNT(*) FROM predictions_history WHERE league=? AND kickoff_utc >= ? AND payload LIKE '%\"implied_1x2\"%'",
        (int(league), cutoff),
    )
    has_implied_key = int(cur.fetchone()[0])

    conn.close()
    return {
        "ok": True,
        "league": league,
        "window_days": window_days,
        "rows_total": total,
        "payload_has_odds_1x2_key": has_odds_key,
        "payload_has_implied_1x2_key": has_implied_key,
    }

import time
from urllib.parse import urlparse

def _get_api_sports_key():
    # Try common env var names you may already be using
    for k in ("API_SPORTS_KEY","APISPORTS_KEY","API_FOOTBALL_KEY","FOOTBALL_API_KEY"):
        v = os.environ.get(k)
        if v:
            return v
    return None

def _extract_1x2_from_odds_api(payload: dict):
    """
    API-Sports odds response shape varies by plan/endpoint; this tries to find:
    bookmaker -> bet ("Match Winner"/"1X2") -> values {Home/Draw/Away}.
    Returns (odds_dict, bookmaker_name, bet_name) or (None, None, None).
    """
    resp = payload.get("response") or []
    if not resp:
        return None, None, None

    # Sometimes response is list with one element
    node = resp[0] if isinstance(resp, list) else resp
    bookmakers = node.get("bookmakers") or []
    if not bookmakers:
        return None, None, None

    preferred_bets = {"Match Winner", "1X2", "1x2", "Full Time Result"}

    for bm in bookmakers:
        bm_name = bm.get("name")
        bets = bm.get("bets") or []
        for bet in bets:
            bet_name = bet.get("name")
            if bet_name not in preferred_bets and (bet_name or "").lower() not in {b.lower() for b in preferred_bets}:
                continue
            values = bet.get("values") or []
            odds = {"home": None, "draw": None, "away": None}
            for v in values:
                label = (v.get("value") or "").strip().lower()
                odd = v.get("odd")
                try:
                    oddf = float(odd) if odd is not None else None
                except Exception:
                    oddf = None
                if label in ("home", "1"):
                    odds["home"] = oddf
                elif label in ("draw", "x"):
                    odds["draw"] = oddf
                elif label in ("away", "2"):
                    odds["away"] = oddf

            if odds["home"] and odds["draw"] and odds["away"]:
                return odds, bm_name, bet_name

    return None, None, None

def _implied_probs_from_odds(odds: dict):
    # Normalize implieds to sum to 1 (removes overround)
    inv = {}
    for k in ("home","draw","away"):
        o = odds.get(k)
        inv[k] = (1.0 / o) if o and o > 0 else None
    if not inv["home"] or not inv["draw"] or not inv["away"]:
        return None
    s = inv["home"] + inv["draw"] + inv["away"]
    return {"home": inv["home"]/s, "draw": inv["draw"]/s, "away": inv["away"]/s}

@app.post("/admin/backfill-market-from-api")
def admin_backfill_market_from_api(
    league: int | None = None,
    window_days: int = 180,
    limit: int = 5000,
    sleep_ms: int = 200,
    dry_run: bool = False,
    admin=Depends(require_admin),
):
    """
    Backfill market odds/probabilities for finished fixtures by calling API-Sports odds endpoint.
    If league is omitted, it will backfill for ALL leagues present in predictions_history.
    """
    key = _get_api_sports_key()
    if not key:
        raise HTTPException(status_code=500, detail="Missing API-Sports key env var (API_SPORTS_KEY/APISPORTS_KEY/etc).")

    import requests

    conn = db_connect()
    cur = conn.cursor()

    # Determine leagues to process
    leagues = []
    if league is not None:
        leagues = [int(league)]
    else:
        cur.execute("SELECT DISTINCT league FROM predictions_history ORDER BY league")
        leagues = [int(r[0]) for r in cur.fetchall()]

    scanned = updated = skipped = errors = 0
    error_samples = []

    cutoff_expr = None
    # Postgres vs sqlite date math
    if getattr(conn, "is_pg", False):
        cutoff_expr = f"NOW() - INTERVAL '{int(window_days)} days'"
        where_time = f"kickoff_utc >= {cutoff_expr}"
    else:
        # sqlite stores kickoff_utc as text; compare lexicographically works for ISO8601
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(window_days))).isoformat()
        where_time = "kickoff_utc >= ?"

    def select_rows_for_league(lg: int):
        if getattr(conn, "is_pg", False):
            cur.execute(f"""
                SELECT id, fixture_id
                FROM predictions_history
                WHERE league = %s
                  AND {where_time}
                  AND actual_result IS NOT NULL
                  AND (
                       market_home_odds IS NULL OR market_draw_odds IS NULL OR market_away_odds IS NULL
                    OR market_home_p   IS NULL OR market_draw_p   IS NULL OR market_away_p   IS NULL
                  )
                ORDER BY kickoff_utc DESC
                LIMIT %s
            """, (lg, int(limit)))
        else:
            cur.execute(f"""
                SELECT id, fixture_id
                FROM predictions_history
                WHERE league = ?
                  AND {where_time}
                  AND actual_result IS NOT NULL
                  AND (
                       market_home_odds IS NULL OR market_draw_odds IS NULL OR market_away_odds IS NULL
                    OR market_home_p   IS NULL OR market_draw_p   IS NULL OR market_away_p   IS NULL
                  )
                ORDER BY kickoff_utc DESC
                LIMIT ?
            """, (lg, cutoff, int(limit)))
        return cur.fetchall()

    for lg in leagues:
        rows = select_rows_for_league(lg)
        for row_id, fixture_id in rows:
            scanned += 1
            try:
                r = requests.get(
                    "https://v3.football.api-sports.io/odds",
                    headers={"x-apisports-key": key},
                    params={"fixture": int(fixture_id)},
                    timeout=20,
                )
                if r.status_code != 200:
                    skipped += 1
                    if len(error_samples) < 10:
                        error_samples.append({"stage":"http", "fixture_id":int(fixture_id), "status":r.status_code})
                    continue

                data = r.json()
                odds, bm_name, bet_name = _extract_1x2_from_odds_api(data)
                if not odds:
                    skipped += 1
                    continue

                implied = _implied_probs_from_odds(odds)
                if not implied:
                    skipped += 1
                    continue

                if not dry_run:
                    cur.execute("""
                        UPDATE predictions_history
                        SET market_home_odds = ?, market_draw_odds = ?, market_away_odds = ?,
                            market_home_p = ?,    market_draw_p = ?,    market_away_p = ?,
                            market_bookmaker = ?, market_bet_name = ?
                        WHERE id = ?
                    """, (
                        odds["home"], odds["draw"], odds["away"],
                        implied["home"], implied["draw"], implied["away"],
                        bm_name, bet_name,
                        row_id
                    ))
                updated += 1

                if sleep_ms:
                    time.sleep(max(0, int(sleep_ms)) / 1000.0)

            except Exception as e:
                errors += 1
                if len(error_samples) < 10:
                    error_samples.append({"stage":"exception", "fixture_id":int(fixture_id), "err":str(e)})

    if not dry_run:
        conn.commit()
    try:
        cur.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass

    return {
        "ok": True,
        "leagues": leagues,
        "window_days": window_days,
        "limit": limit,
        "sleep_ms": sleep_ms,
        "scanned": scanned,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run,
        "error_samples": error_samples,
    }
