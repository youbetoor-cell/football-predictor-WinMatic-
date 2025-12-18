#!/usr/bin/env python3
"""
WinMatic backend — cleaned and patched:
- Serve /static correctly
- Add /team-logo/{team_id}.png dynamic proxy + disk cache
- Create /static/team-logo/default.png on startup to avoid 404 spam
- Keep all prediction endpoints intact
"""

import os
import io
import json
import math
import time
import base64
import logging
import sqlite3
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

        conn = sqlite3.connect(DB_PATH)
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

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

SNAPSHOT_DIR = os.path.join(ART, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

API_CACHE_FILE = os.path.join(ART, "api_cache.json")
API_DISK_CACHE_FILE = os.path.join(ART, "api_disk_cache.json")
CACHE_ONLY_MODE = os.getenv("WINMATIC_CACHE_ONLY", "0") == "1"
API_QUOTA_EXHAUSTED = False  # becomes True after daily limit is hit

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

    # --- Stop if daily quota already hit ------------------------------------
    if API_QUOTA_EXHAUSTED:
        cached = try_cache("quota-exhausted")
        if cached:
            return cached
        raise HTTPException(
            status_code=429,
            detail="API-FOOTBALL daily request limit already reached (quota exhausted)."
        )

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
    url = f"{API_BASE}{path}"
    logger.info("[API CALL] %s %s", url, params)

    # --- Perform the request ------------------------------------------------
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()
    except Exception as e:
        logger.warning("[API ERROR] %s", e)
        cached = try_cache("network-error")
        if cached:
            return cached
        raise HTTPException(status_code=502, detail=str(e))

    # --- Non-200 status codes -----------------------------------------------
    if resp.status_code != 200:
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
    return data


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

    

def current_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1

# ============================================================
# HISTORY DB
# ============================================================

def init_history_db() -> None:
    os.makedirs(ART, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
    # Odds cache table (for 1X2 odds, to reduce API calls)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_cache (
            fixture_id INTEGER PRIMARY KEY,
            league INTEGER,
            kickoff_utc TEXT,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            updated_utc TEXT
        );
        """
    )

    conn.commit()
    conn.close()
    logger.info("[DB] history.db ready")

init_history_db()

# ============================================================
# ODDS CACHE (SQLite) — reduce /odds API calls
# ============================================================

ODDS_CACHE_TTL_SECONDS = int(os.getenv("ODDS_CACHE_TTL_SECONDS", "21600") or 21600)  # 6h
ODDS_CACHE_MAX_ROWS = int(os.getenv("ODDS_CACHE_MAX_ROWS", "50000") or 50000)

def get_cached_odds(fixture_id: int) -> Optional[Dict[str, float]]:
    """Return cached 1X2 odds for fixture_id if fresh, else None."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            "SELECT odds_home, odds_draw, odds_away, updated_utc FROM odds_cache WHERE fixture_id=?",
            (int(fixture_id),),
        )
        row = cur.fetchone()
        con.close()

        if not row:
            return None

        oh, od, oa, updated = row
        if not updated:
            return None

        try:
            updated_dt = datetime.fromisoformat(str(updated).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None

        age = (datetime.utcnow() - updated_dt).total_seconds()
        if age > ODDS_CACHE_TTL_SECONDS:
            return None

        if oh is None or od is None or oa is None:
            return None

        return {"home": float(oh), "draw": float(od), "away": float(oa)}
    except Exception:
        return None

def set_cached_odds(
    fixture_id: int,
    odds: Dict[str, float],
    league: Optional[int] = None,
    kickoff_utc: Optional[str] = None,
) -> None:
    """Upsert cached odds for a fixture (best-effort)."""
    try:
        oh = float(odds.get("home")) if odds and odds.get("home") else None
        od = float(odds.get("draw")) if odds and odds.get("draw") else None
        oa = float(odds.get("away")) if odds and odds.get("away") else None
        if oh is None or od is None or oa is None:
            return

        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO odds_cache
              (fixture_id, league, kickoff_utc, odds_home, odds_draw, odds_away, updated_utc)
            VALUES (?,?,?,?,?,?,?);
            """,
            (
                int(fixture_id),
                int(league) if league is not None else None,
                str(kickoff_utc) if kickoff_utc is not None else None,
                oh, od, oa,
                datetime.utcnow().isoformat(),
            ),
        )
        con.commit()

        # keep table from growing forever
        try:
            cur.execute(
                f"""
                DELETE FROM odds_cache
                WHERE fixture_id IN (
                    SELECT fixture_id FROM odds_cache
                    ORDER BY updated_utc DESC
                    LIMIT -1 OFFSET {ODDS_CACHE_MAX_ROWS}
                );
                """
            )
            con.commit()
        except Exception:
            pass

        con.close()
    except Exception:
        return


def record_predictions_history(league: int, fixtures: list[dict]) -> None:
    """
    Persist prediction payloads for later analysis.

    Upserts on (league, fixture_id, kickoff_utc) so we don't create duplicates,
    and never writes blank created_at.
    """
    if not fixtures:
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Ensure unique key exists (needed for ON CONFLICT)
        try:
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_history_key "
                "ON predictions_history (league, fixture_id, kickoff_utc);"
            )
        except Exception as e:
            logger.warning("[DB] Could not ensure unique index: %s", e)

        rows: list[tuple] = []
        for f in fixtures:
            fixture_id = f.get("fixture_id") or (f.get("fixture") or {}).get("id")
            kickoff_utc = f.get("kickoff_utc") or (f.get("fixture") or {}).get("date")
            if fixture_id is None or kickoff_utc is None:
                continue

            payload = json.dumps(f, ensure_ascii=False)
            rows.append((int(league), int(fixture_id), str(kickoff_utc), payload))

        if not rows:
            return

        cur.executemany(
            """
            INSERT INTO predictions_history (league, fixture_id, kickoff_utc, payload, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(league, fixture_id, kickoff_utc) DO UPDATE SET
              payload = excluded.payload,
              created_at = COALESCE(NULLIF(predictions_history.created_at,''), excluded.created_at)
            """,
            rows,
        )
        conn.commit()

    except Exception as e:
        logger.warning("Failed to record history: %s", e)
    finally:
        if conn is not None:
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
    Add bookmaker closing odds (implied probabilities) for each fixture.

    Creates columns:
    - home_odd_implied
    - draw_odd_implied
    - away_odd_implied

    If odds are missing or the API fails, falls back to 1/3, 1/3, 1/3.
    """
    api_key = os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        logger.warning("[TRAIN] API_FOOTBALL_KEY not set, using neutral odds features.")
        df["home_odd_implied"] = 1.0 / 3.0
        df["draw_odd_implied"] = 1.0 / 3.0
        df["away_odd_implied"] = 1.0 / 3.0
        return df

    odds_rows: List[Dict[str, Any]] = []
    fixture_ids = df["fixture_id"].dropna().unique().tolist()

    for fid in fixture_ids:
        fid_int = int(fid)
        p_home_implied = p_draw_implied = p_away_implied = 1.0 / 3.0

        try:
            url_odds = f"https://v3.football.api-sports.io/odds?fixture={fid_int}"
            headers = {"x-apisports-key": api_key}
            r_odds = requests.get(url_odds, headers=headers, timeout=10)
            r_odds.raise_for_status()
            response = r_odds.json().get("response", [])

            if response:
                # Prefer Bet365 if present, otherwise just take the first bookmaker
                bookmaker_entry = next(
                    (x for x in response if x.get("bookmaker", {}).get("name") == "Bet365"),
                    response[0],
                )

                bets = bookmaker_entry.get("bets", [])
                if bets:
                    # Assume first "Match Winner" style market has 3 outcomes: 1X2
                    values = bets[0].get("values", [])
                    if len(values) >= 3:
                        odds_home = float(values[0].get("odd"))
                        odds_draw = float(values[1].get("odd"))
                        odds_away = float(values[2].get("odd"))

                        inv_sum = (1.0 / odds_home) + (1.0 / odds_draw) + (1.0 / odds_away)
                        if inv_sum > 0:
                            p_home_implied = (1.0 / odds_home) / inv_sum
                            p_draw_implied = (1.0 / odds_draw) / inv_sum
                            p_away_implied = (1.0 / odds_away) / inv_sum
        except Exception as e:
            logger.warning("[TRAIN] odds fetch failed for fixture %s: %s", fid_int, e)

        odds_rows.append(
            {
                "fixture_id": fid_int,
                "home_odd_implied": p_home_implied,
                "draw_odd_implied": p_draw_implied,
                "away_odd_implied": p_away_implied,
            }
        )

    odds_df = pd.DataFrame(odds_rows)

    df = df.merge(odds_df, on="fixture_id", how="left")
    # Fill any missing with neutral 1/3
    for col in ["home_odd_implied", "draw_odd_implied", "away_odd_implied"]:
        df[col] = df[col].fillna(1.0 / 3.0).astype(float)

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


def extract_match_winner_odds(data: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Extract 1X2 (Match Winner) odds from an API-FOOTBALL /odds response.

    Returns a dict like:
      {"home": 1.85, "draw": 3.8, "away": 4.2}
    or None if not found.
    """
    resp = data.get("response") or []
    if not resp:
        return None

    entry = resp[0]  # first fixture
    bookmakers = entry.get("bookmakers") or []
    if not bookmakers:
        return None

    # For simplicity, just use the first bookmaker that has a Match Winner / 1X2 market
    for bm in bookmakers:
        bets = bm.get("bets") or []
        for bet in bets:
            name = (bet.get("name") or bet.get("label") or "").lower()
            if "match winner" in name or "1x2" in name:
                values = bet.get("values") or []
                out: Dict[str, float] = {}
                for v in values:
                    side = (v.get("value") or "").lower()
                    odd_str = v.get("odd")
                    try:
                        odd_val = float(odd_str)
                    except Exception:
                        continue

                    if "home" in side or side == "1":
                        out["home"] = odd_val
                    elif "away" in side or side == "2":
                        out["away"] = odd_val
                    elif "draw" in side or side == "x":
                        out["draw"] = odd_val

                if len(out) >= 2:
                    return out

    return None


def fetch_1x2_odds_for_fixture(fixture_id: int) -> Optional[Dict[str, float]]:
    """
    Call API-FOOTBALL /odds for a single fixture and return 1X2 odds.
    """
    # Odds cache (fast path)
    cached = get_cached_odds(int(fixture_id))
    if cached:
        return cached

    try:
        data = api_get("/odds", {"fixture": fixture_id})
    except HTTPException as e:
        logger.warning("[ODDS] HTTP error fixture=%s: %s", fixture_id, e.detail)
        return None
    except Exception as e:
        logger.warning("[ODDS] error fixture=%s: %s", fixture_id, e)
        return None

    odds = extract_match_winner_odds(data)
    if not odds:
        logger.info("[ODDS] no 1X2 odds for fixture=%s", fixture_id)
    
    try:
        set_cached_odds(int(fixture_id), odds)
    except Exception:
        pass

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

def fetch_top_scorers(league_id: int, season: int) -> List[Dict[str, Any]]:
    key = (league_id, season)
    if key in TOPSCORERS_CACHE:
        return TOPSCORERS_CACHE[key]
    data = api_get("/players/topscorers", {"league": league_id, "season": season})
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

@app.get("/debug/db")
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
        con = sqlite3.connect(DB_PATH)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(predictions_history)").fetchall()]
            info["predictions_history_cols"] = cols
        finally:
            con.close()
    except Exception as e:
        info["error"] = repr(e)

    return info




@app.get("/debug/routes", dependencies=[Depends(require_admin)])
def debug_routes():
    """List registered routes (admin-only)."""
    out = []
    try:
        for r in app.routes:
            path = getattr(r, "path", None)
            methods = sorted(list(getattr(r, "methods", []) or []))
            name = getattr(r, "name", None)
            if path:
                out.append({"path": path, "methods": methods, "name": name})
        out.sort(key=lambda x: x["path"])
    except Exception as e:
        return {"ok": False, "error": str(e), "routes": []}

    return {"ok": True, "count": len(out), "routes": out}

@app.get("/debug/odds-cache", dependencies=[Depends(require_admin)])
def debug_odds_cache(limit: int = Query(10, ge=1, le=200)):
    """Inspect odds_cache (admin-only)."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        # table may not exist if someone deployed before adding it
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS odds_cache (
                fixture_id INTEGER PRIMARY KEY,
                league INTEGER,
                kickoff_utc TEXT,
                odds_home REAL,
                odds_draw REAL,
                odds_away REAL,
                updated_utc TEXT
            );
            """
        )
        con.commit()

        cur.execute("SELECT COUNT(*) FROM odds_cache;")
        total = int(cur.fetchone()[0])

        cur.execute(
            "SELECT fixture_id, league, kickoff_utc, odds_home, odds_draw, odds_away, updated_utc "
            "FROM odds_cache ORDER BY updated_utc DESC LIMIT ?;",
            (int(limit),),
        )
        rows = cur.fetchall()

        cutoff = (datetime.utcnow() - timedelta(seconds=ODDS_CACHE_TTL_SECONDS)).isoformat()
        cur.execute("SELECT COUNT(*) FROM odds_cache WHERE updated_utc < ?;", (cutoff,))
        expired = int(cur.fetchone()[0])

        con.close()

        sample = []
        for fx_id, lg, ko, oh, od, oa, upd in rows:
            sample.append({
                "fixture_id": fx_id,
                "league": lg,
                "kickoff_utc": ko,
                "odds": {"home": oh, "draw": od, "away": oa},
                "updated_utc": upd,
            })

        return {"ok": True, "total": total, "expired": expired, "sample": sample}
    except Exception as e:
        return {"ok": False, "error": str(e)}
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
def api_train(req: TrainRequest):
    seasons = req.seasons or DEFAULT_SEASONS
    logger.info("[TRAIN API] league=%s seasons=%s", req.league, seasons)
    info = train_model(req.league, seasons)
    return {"ok": True, "info": info}

@app.get("/predict/upcoming")
def api_predict_upcoming(
    league: int = Query(DEFAULT_LEAGUE),
    days_ahead: int = Query(7, ge=1, le=14),
):
    try:
        model, meta = load_model_and_meta(league)
    except HTTPException:
        # Fall back to snapshot if model not available
        snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=days_ahead)
        if snapshot:
            return {
                "ok": True,
                "count": len(snapshot),
                "fixtures": snapshot,
                "source": "snapshot",
                "snapshot_file": os.path.basename(snap_path) if snap_path else None,
            }
        raise

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

    if not results:
        snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=days_ahead)
        if snapshot:
            return {
                "ok": True,
                "count": len(snapshot),
                "fixtures": snapshot,
                "source": "snapshot",
                "snapshot_file": os.path.basename(snap_path) if snap_path else None,
            }
        return {
            "ok": False,
            "count": 0,
            "fixtures": [],
            "detail": "No fixtures available. Train the model or provide cached data.",
        }

    # 👉 Add reasoning to each result, keeping your existing structure
    for p in results:
        p["reasoning"] = build_reasoning_for_prediction(p, meta)

    # Save to history (now including reasoning)
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
):
    """
    Backfill actual_result for predictions_history rows once matches finish.

    It updates rows where:
      - league matches
      - actual_result is NULL/blank
      - kickoff_utc is in the past (within lookback window)
    """
    from datetime import datetime, timedelta, timezone

    ensure_predictions_db()

    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=lookback_days)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # candidates: finished by time, missing actual_result
    cur.execute(
        """
        SELECT DISTINCT fixture_id
        FROM predictions_history
        WHERE league = ?
          AND kickoff_utc >= ?
          AND kickoff_utc < ?
          AND (actual_result IS NULL OR TRIM(actual_result) = '')
          AND fixture_id IS NOT NULL
        ORDER BY kickoff_utc DESC
        LIMIT ?
        """,
        (league, since, now.isoformat(), max_fixtures),
    )
    fixture_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    updated = 0
    scanned = 0
    skipped = 0
    errors = []

    for fid in fixture_ids:
        scanned += 1
        try:
            data = api_get("/fixtures", {"id": int(fid)})
            resp = (data.get("response") or [])
            if not resp:
                skipped += 1
                continue

            fx = resp[0]
            goals = fx.get("goals") or {}
            hg = goals.get("home")
            ag = goals.get("away")

            # if API doesn’t have goals yet, skip
            if hg is None or ag is None:
                skipped += 1
                continue

            try:
                hg = int(hg)
                ag = int(ag)
            except Exception:
                skipped += 1
                continue

            if hg > ag:
                actual = "home"
            elif ag > hg:
                actual = "away"
            else:
                actual = "draw"

            if dry_run:
                updated += 1
                continue

            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE predictions_history
                SET actual_result = ?
                WHERE league = ?
                  AND fixture_id = ?
                  AND (actual_result IS NULL OR TRIM(actual_result) = '')
                """,
                (actual, league, int(fid)),
            )
            conn.commit()
            conn.close()

            updated += 1

        except Exception as e:
            errors.append({"fixture_id": fid, "error": repr(e)})
            # keep going

    return {
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


    # =====================================================
# 📊 RESULTS + METRICS ENDPOINTS (WORKING WITH EXISTING DB)
# =====================================================
from datetime import datetime, timedelta
import json
import sqlite3
import math

from pathlib import Path
import sqlite3

os.makedirs("data", exist_ok=True)


@app.get("/backtest/1x2")
def api_backtest_1x2(
    league: int = Query(39, description="League ID"),
    season: int = Query(None, description="Season year (e.g. 2025). If omitted, uses current season."),
    last_n: int = Query(200, ge=20, le=2000, description="How many finished fixtures to evaluate (most recent first)"),
    max_goals: int = Query(10, ge=6, le=15, description="Poisson truncation for 1X2 probs"),
    write_db: bool = Query(False, description="If true, write actual_result + model probs into predictions_history"),
    dry_run: bool = Query(False, description="If true, do not write to DB even if write_db=true"),
    sample_limit: int = Query(300, ge=0, le=2000, description="How many per-game rows to include in sample (0 disables sample)"),
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

    # pick season default
    if season is None:
        season = current_season()

    # 1) Fetch finished fixtures
    # API-FOOTBALL supports status=FT; we then take the most recent last_n by kickoff date.
    data = api_get("/fixtures", {"league": league, "season": season, "status": "FT"})
    fixtures = (data.get("response") or [])
    if not fixtures:
        return {"ok": False, "message": "No finished fixtures found.", "league": league, "season": season}

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

    per_game = []

    cal = load_1x2_calibration(league) or {}


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

                "logloss": round(ll, 4),
            }
        )


        # 4) Optional: write back into DB so /progress/metrics can work later
        if write_db and not dry_run:
            try:
                ensure_predictions_db()
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()

                cols = [r[1] for r in cur.execute("PRAGMA table_info(predictions_history)").fetchall()]
                # build dynamic upsert only with columns that exist
                insert_cols = []
                insert_vals = []
                def add(col, val):
                    if col in cols:
                        insert_cols.append(col)
                        insert_vals.append(val)

                add("league", int(league))
                add("fixture_id", int(fid))
                add("kickoff_utc", (fx.get("fixture") or {}).get("date"))
                add("home_team", ((fx.get("teams") or {}).get("home") or {}).get("name"))
                add("away_team", ((fx.get("teams") or {}).get("away") or {}).get("name"))

                add("model_home_p", float(ph))
                add("model_draw_p", float(pd))
                add("model_away_p", float(pa))

                add("predicted_side", pred_side)
                add("actual_result", actual)

                # payload if available
                try:
                    add("payload", json.dumps(pred, ensure_ascii=False))
                except Exception:
                    pass

                # build query
                placeholders = ",".join(["?"] * len(insert_cols))
                col_sql = ",".join(insert_cols)

                # update set (don’t overwrite actual_result if already set)
                set_parts = []
                for c in insert_cols:
                    if c in ("id",):
                        continue
                    if c == "actual_result":
                        set_parts.append(f"{c} = COALESCE(predictions_history.actual_result, excluded.actual_result)")
                    else:
                        set_parts.append(f"{c} = excluded.{c}")
                set_sql = ",\n".join(set_parts)

                sql = f"""
                INSERT INTO predictions_history ({col_sql})
                VALUES ({placeholders})
                ON CONFLICT(league, fixture_id, kickoff_utc) DO UPDATE SET
                {set_sql}
                """

                cur.execute(sql, tuple(insert_vals))
                conn.commit()
                conn.close()
            except Exception:
                # do not fail whole backtest if DB write hiccups
                pass

    if n == 0:
        return {"ok": False, "message": "No scorable fixtures (missing goals/xG).", "league": league, "season": season}

    return {
        "ok": True,
        "league": league,
        "season": season,
        "fixtures_scored": n,
        "accuracy": round(correct / n, 4),
        "logloss": round(logloss_sum / n, 4),
        "write_db": bool(write_db),
        "dry_run": bool(dry_run),
                "fixtures_total": len(fixtures),
        "preds_generated": len(preds),
        "per_game_len": len(per_game),
        "n_used": n,
        "sample_limit": int(sample_limit),
        "sample": (per_game[: min(int(sample_limit), len(per_game))] if int(sample_limit) > 0 else []),
    }


@app.get("/progress/metrics")
def api_progress_metrics(
    league: int = Query(39, description="League ID"),
    window_days: int = Query(60, ge=7, le=365, description="Days of history to evaluate"),
):
    """
    Progress stats (accuracy + log loss) for recent predictions in predictions_history.

    IMPORTANT:
    - We dedupe by fixture_id and use ONLY the latest row per fixture
      so scanning the same game multiple times doesn't inflate the metrics.
    """
    import math
    from datetime import datetime, timedelta, timezone

    ensure_predictions_db()

    since_ts = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # include id + fixture_id so we can pick the latest row per fixture
    cur.execute(
        """
        SELECT
            id,
            fixture_id,
            model_home_p,
            model_draw_p,
            model_away_p,
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

    if not rows:
        return {
            "ok": False,
            "message": "No records with actual results in this window.",
            "league": league,
            "window_days": window_days,
        }

    # ------ dedupe per fixture: latest row only ------
    latest_by_fixture = {}
    for (
        row_id,
        fixture_id,
        ph,
        pd,
        pa,
        actual_result,
    ) in rows:
        # rows are id DESC, so first one we see for a fixture_id is the latest
        if fixture_id in latest_by_fixture:
            continue
        latest_by_fixture[fixture_id] = (ph, pd, pa, actual_result)

    total = len(latest_by_fixture)
    if total == 0:
        return {
            "ok": False,
            "message": "No fixtures after deduplication.",
            "league": league,
            "window_days": window_days,
        }

    correct = 0
    log_losses = []

    for (ph, pd, pa, actual) in latest_by_fixture.values():
        probs = {"home": ph, "draw": pd, "away": pa}
        predicted = max(probs, key=probs.get)

        if predicted == actual:
            correct += 1

        # log loss on the TRUE outcome
        p_true = max(probs.get(actual, 1e-6), 1e-6)
        log_losses.append(-math.log(p_true))

    accuracy = round(correct / total, 3)
    avg_log_loss = round(sum(log_losses) / total, 3)

    return {
        "ok": True,
        "league": league,
        "sample_size": total,      # per-fixture, same meaning as total_fixtures
        "accuracy": accuracy,      # should roughly align with accuracy_fixtures
        "log_loss": avg_log_loss,
        "window_days": window_days,
        "generated": datetime.now(timezone.utc).isoformat(),
    }


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
        conn = sqlite3.connect(DB_PATH)
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
    from_date: str = Query(..., description="YYYY-MM-DD start date"),
    to_date: Optional[str] = Query(None, description="YYYY-MM-DD end date (inclusive)"),
):
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

        conn = sqlite3.connect(DB_PATH)
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
        conn = sqlite3.connect(DB_PATH)
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

    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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

    conn = sqlite3.connect(DB_PATH)
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

      

@app.get("/debug/fixture/{fixture_id}")
def debug_fixture(
    fixture_id: int = ApiPath(..., description="Fixture ID (as stored in predictions_history)"),
    league: int = Query(DEFAULT_LEAGUE, description="League ID, e.g. 39 = Premier League"),
    include_history: bool = Query(
        True,
        description="If true, include all DB rows for this fixture (not just the latest one)"
    ),
) -> Dict[str, Any]:
    """
    Debug endpoint to inspect what the model predicted for a single fixture.

    It ONLY reads from predictions_history (no external API calls), so it is
    safe to use even when API-FOOTBALL quota is exhausted.

    Returns:
    - basic fixture info (teams, kickoff time)
    - model probabilities (home/draw/away)
    - predicted side
    - edge_value (if any)
    - actual_result (if filled by /update-results)
    - optionally: all historical rows for this fixture in the DB
    """
    ensure_predictions_db()

    try:
        conn = sqlite3.connect(DB_PATH)
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
            WHERE fixture_id = ?
              AND league = ?
            ORDER BY id DESC
            """,
            (fixture_id, league),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        logger.error("[DEBUG FIXTURE] DB error: %s", e)
        return {"ok": False, "error": str(e)}

    if not rows:
        return {
            "ok": False,
            "fixture_id": fixture_id,
            "league": league,
            "error": "No predictions found for this fixture in predictions_history.",
        }

    # Latest record (the first row because we ordered by id DESC)
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
    ) = rows[0]

    # Build a clean "latest" summary
    latest = {
        "fixture_id": fx_id,
        "league": lg,
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_utc": kickoff_utc,
        "model_probs": {
            "home": ph,
            "draw": pd,
            "away": pa,
        },
        "predicted_side": predicted_side,
        "edge_value": edge_value,
        "actual_result": actual_result,
    }

    # Optionally include the full DB history for this fixture
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
            pred_side,
            edge,
            result,
        ) in rows:
            history.append({
                "row_id": row_id,
                "fixture_id": fx_id,
                "league": lg,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_utc": kickoff_utc,
                "model_home_p": ph,
                "model_draw_p": pd,
                "model_away_p": pa,
                "predicted_side": pred_side,
                "edge_value": edge,
                "actual_result": result,
            })

    return {
        "ok": True,
        "fixture_id": fixture_id,
        "league": league,
        "latest": latest,
        "num_records": len(rows),
        "history": history if include_history else None,
    }


@app.get("/team-strength")
def api_team_strength(league: int = Query(DEFAULT_LEAGUE)):
    _, meta = load_model_and_meta(league)
    team_summary = meta.get("team_summary", {})
    teams = list(team_summary.values())
    teams.sort(key=lambda r: r.get("rating", 1.0), reverse=True)
    return {"ok": True, "league": league, "teams": teams}

@app.get("/model-info")
def api_model_info(league: int = Query(DEFAULT_LEAGUE)):
    """
    Show information about the currently trained model for a league:
    - seasons used
    - features and targets
    - evaluation metrics (logloss, Brier)
    - when it was trained
    """
    _, meta = load_model_and_meta(league)

    info = {
        "league": league,
        "seasons": meta.get("seasons"),
        "features": meta.get("feature_cols"),
        "targets": meta.get("target_cols"),
        "metrics": meta.get("metrics"),
        "trained_at": meta.get("trained_at"),
    }

    return {"ok": True, "info": info}

@lru_cache(maxsize=512)
def get_fixture_logos(fixture_id: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Small cache so we only call API-FOOTBALL once per fixture
    when building /results/recent cards with real team logos.

    Returns (home_logo_url, away_logo_url).
    If anything fails, both are None.
    """
    try:
        data = api_get("/fixtures", {"id": fixture_id})
    except HTTPException as e:
        logger.warning("[RESULTS LOGOS] API error for fixture %s: %s", fixture_id, e.detail)
        return None, None

    resp = data.get("response") or []
    if not resp:
        return None, None

    teams = resp[0].get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}

    return home.get("logo"), away.get("logo")

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

    conn = sqlite3.connect(DB_PATH)
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
def api_recent_results(
    league: int = Query(39, description="League ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of fixtures to show"),
):
    """
    Return the most recent UNIQUE fixtures (by fixture_id) that have a known result.
    Used by static/results.html.

    Enriches each fixture with home_logo / away_logo using API-FOOTBALL.
    """
    ensure_predictions_db()

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Get all finished predictions for this league, newest first
        cur.execute(
            """
            SELECT
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
              AND predicted_side IS NOT NULL
              AND actual_result IS NOT NULL
            ORDER BY kickoff_utc DESC
            """,
            (league,),
        )
        rows = cur.fetchall()
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    fixtures = []
    seen_fixture_ids = set()

    for (
        fixture_id,
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
    ) in rows:
        if fixture_id in seen_fixture_ids:
            continue  # skip duplicates

        seen_fixture_ids.add(fixture_id)
        won = bool(actual_result and predicted_side and actual_result == predicted_side)

        # ✅ real logos from API-FOOTBALL (cached per fixture)
        home_logo, away_logo = get_fixture_logos(fixture_id)

        fixtures.append(
            {
                "fixture_id": fixture_id,
                "league": lg,
                "home_team": home_team,
                "away_team": away_team,
                "home_logo": home_logo,
                "away_logo": away_logo,
                "kickoff_utc": kickoff_utc,
                "model_probs": {"home": ph, "draw": pd, "away": pa},
                "predicted_side": predicted_side,
                "edge_value": edge_value,
                "actual_result": actual_result,
                "won": won,
            }
        )

        if len(fixtures) >= limit:
            break

    return {
        "ok": True,
        "count": len(fixtures),
        "fixtures": fixtures,
    }



# =========================================================
# ⚽ AUTO-UPDATE ACTUAL RESULTS AFTER MATCHES FINISH
# =========================================================
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

    conn = sqlite3.connect(DB_PATH)
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

    conn = sqlite3.connect(DB_PATH)
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
