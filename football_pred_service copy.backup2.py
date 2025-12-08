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
from typing import Any, Dict, List, Optional, Tuple

import math
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import requests
from joblib import dump, load
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, Response

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_BASE = "https://v3.football.api-sports.io"

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

SNAPSHOT_DIR = os.path.join(ART, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

API_CACHE_FILE = os.path.join(ART, "api_cache.json")
API_DISK_CACHE_FILE = os.path.join(ART, "api_disk_cache.json")
CACHE_ONLY_MODE = os.getenv("WINMATIC_CACHE_ONLY", "0") == "1"

DB_PATH = os.path.join(ART, "history.db")

DEFAULT_LEAGUE = 39  # Premier League
DEFAULT_SEASONS = [2021, 2022, 2023, 2024]
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

# ============================================================
# UTILS
# ============================================================

def api_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    def try_cache(reason: str) -> Optional[Dict[str, Any]]:
        cached = cached_api_response(path, params)
        if cached:
            logger.info("[API CACHE MODE] served=%s reason=%s", path, reason)
        return cached

    if not API_FOOTBALL_KEY:
        cached = try_cache("missing-key")
        if cached:
            return cached
        raise HTTPException(status_code=500, detail="API_FOOTBALL_KEY not configured in environment")

    if CACHE_ONLY_MODE:
        cached = try_cache("cache-only-mode")
        if cached:
            return cached
        raise HTTPException(status_code=503, detail="Cache-only mode enabled but no cached data for request.")

    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    url = f"{API_BASE}{path}"
    logger.info("[API CALL] %s %s", url, params)

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            data = resp.json()
        except Exception as e:
            logger.warning("[API ERROR] attempt=%s %s", attempt, e)
            if attempt == 3:
                cached = try_cache("network-error")
                if cached:
                    return cached
                raise HTTPException(status_code=502, detail=str(e))
            time.sleep(1)
            continue

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
    conn.commit()
    conn.close()
    logger.info("[DB] history.db ready")

init_history_db()

def record_predictions_history(league: int, fixtures: List[Dict[str, Any]]) -> None:
    if not fixtures:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        for f in fixtures:
            cur.execute(
                """
                INSERT OR IGNORE INTO predictions_history
                (league, fixture_id, kickoff_utc, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (league, int(f["fixture_id"]), f["kickoff_utc"], json.dumps(f), now),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Failed to record history: %s", e)

# ============================================================
# MODEL IO
# ============================================================

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

def build_training_frame(league_id: int, seasons: List[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = fetch_historic_fixtures(league_id, seasons)
    team_ids = pd.unique(df[["home_id", "away_id"]].values.ravel())
    team_index = {int(t): i for i, t in enumerate(sorted(team_ids))}
    df["home_team_idx"] = df["home_id"].map(team_index).astype(float)
    df["away_team_idx"] = df["away_id"].map(team_index).astype(float)
    df["home_advantage"] = 1.0

    attack_strength, defense_strength, team_summary = build_team_strengths(df)
    df["home_att_str"] = df["home_id"].map(attack_strength).astype(float)
    df["home_def_str"] = df["home_id"].map(defense_strength).astype(float)
    df["away_att_str"] = df["away_id"].map(attack_strength).astype(float)
    df["away_def_str"] = df["away_id"].map(defense_strength).astype(float)

    # last-5 form (simple)
    form_rows = []
    for tid in team_ids:
        team_df = df[(df["home_id"] == tid) | (df["away_id"] == tid)].sort_values("date")
        gf_list, ga_list = [], []
        for _, r in team_df.iterrows():
            if r["home_id"] == tid:
                gf_list.append(r["home_goals"]); ga_list.append(r["away_goals"])
            else:
                gf_list.append(r["away_goals"]); ga_list.append(r["home_goals"])
            if len(gf_list) > 5:
                gf_list.pop(0); ga_list.pop(0)
            form_rows.append({"fixture_id": r["fixture_id"], "team_id": tid, "form_gf": np.mean(gf_list), "form_ga": np.mean(ga_list)})
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
    df[["home_form_gf","home_form_ga","away_form_gf","away_form_ga"]] = \
        df[["home_form_gf","home_form_ga","away_form_gf","away_form_ga"]].fillna(1.0).astype(float)

    feature_cols = FEATURE_COLS_BASE + ["home_form_gf","home_form_ga","away_form_gf","away_form_ga"]
    target_cols = TARGET_COLS.copy()

    X = df[feature_cols].astype(float).values
    y = df[target_cols].astype(float).values

    meta: Dict[str, Any] = {
        "league_id": league_id,
        "seasons": seasons,
        "team_index": team_index,
        "attack_strength": attack_strength,
        "defense_strength": defense_strength,
        "team_summary": team_summary,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }
    return df, meta

def train_model(league_id: int, seasons: List[int]) -> Dict[str, Any]:
    df, meta = build_training_frame(league_id, seasons)
    X = df[meta["feature_cols"]].astype(float).values
    y = df[meta["target_cols"]].astype(float).values
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=140, max_depth=14, random_state=42, n_jobs=-1))
    logger.info("[TRAIN] league=%s samples=%s features=%s targets=%s", league_id, X.shape[0], X.shape[1], y.shape[1])
    model.fit(X, y)
    save_model_and_meta(league_id, model, meta)
    return {"league": league_id, "samples": int(X.shape[0]), "features": meta["feature_cols"], "targets": meta["target_cols"]}

# ============================================================
# PREDICTIONS
# ============================================================

def make_feature_row_for_fixture(fx: Dict[str, Any], meta: Dict[str, Any]) -> np.ndarray:
    team_index = {str(k): int(v) for k, v in meta["team_index"].items()}
    attack_strength = {str(k): float(v) for k, v in meta["attack_strength"].items()}
    defense_strength = {str(k): float(v) for k, v in meta["defense_strength"].items()}

    home_id = fx["teams"]["home"]["id"]
    away_id = fx["teams"]["away"]["id"]

    def ensure_team(tid: int) -> None:
        tid_str = str(tid)
        if tid_str not in team_index:
            team_index[tid_str] = max(team_index.values(), default=0) + 1
        if tid_str not in attack_strength:
            attack_strength[tid_str] = 1.0
        if tid_str not in defense_strength:
            defense_strength[tid_str] = 1.0

    ensure_team(home_id); ensure_team(away_id)

    home_idx = float(team_index[str(home_id)])
    away_idx = float(team_index[str(away_id)])
    home_att = float(attack_strength[str(home_id)])
    home_def = float(defense_strength[str(home_id)])
    away_att = float(attack_strength[str(away_id)])
    away_def = float(defense_strength[str(away_id)])

    feat: Dict[str, float] = {}
    for c in meta["feature_cols"]:
        if c == "home_team_idx": feat[c] = home_idx
        elif c == "away_team_idx": feat[c] = away_idx
        elif c == "home_advantage": feat[c] = 1.0
        elif c == "home_att_str": feat[c] = home_att
        elif c == "home_def_str": feat[c] = home_def
        elif c == "away_att_str": feat[c] = away_att
        elif c == "away_def_str": feat[c] = away_def
        elif c in ["home_form_gf","home_form_ga","away_form_gf","away_form_ga"]:
            feat[c] = 1.2
        else:
            feat[c] = 0.0
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

        # -----------------------------------------
        # ✓ Kickoff time
        # -----------------------------------------
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

        # -----------------------------------------
        # ✓ Feature row → model prediction
        # -----------------------------------------
        X = make_feature_row_for_fixture(fx, meta)
        y_pred = model.predict(X)[0]

        pred_home_goals = max(0, float(y_pred[0]))
        pred_away_goals = max(0, float(y_pred[1]))

        # -----------------------------------------
        # ✓ Stats (corners, SOT, cards…)
        # -----------------------------------------
        extra = derive_extra_stats(pred_home_goals, pred_away_goals)

        # -----------------------------------------
        # ✓ Probability engine (simple Poisson)
        # -----------------------------------------
        def poisson_prob(avg, k):
            try:
                return (avg ** k) * math.exp(-avg) / math.factorial(k)
            except:
                return 0.0

        home_win_p = 0.0
        draw_p = 0.0
        away_win_p = 0.0

        for hg in range(0, 6):
            for ag in range(0, 6):
                p = poisson_prob(pred_home_goals, hg) * poisson_prob(pred_away_goals, ag)
                if hg > ag:
                    home_win_p += p
                elif hg == ag:
                    draw_p += p
                else:
                    away_win_p += p

        # -----------------------------------------
        # ✓ Likely scorers (your existing function)
        # -----------------------------------------
        scorers = build_players_to_score_for_fixture(fx, league, season)

        # -----------------------------------------
        # ✓ Final structured result
        # -----------------------------------------
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

            "pred_home_goals": round(pred_home_goals, 2),
            "pred_away_goals": round(pred_away_goals, 2),

            "prob_home_win": round(home_win_p, 3),
            "prob_draw": round(draw_p, 3),
            "prob_away_win": round(away_win_p, 3),

            "stats": {
                "xg": [round(pred_home_goals, 2), round(pred_away_goals, 2)],
                "shots_on_target": [extra["home_sot"], extra["away_sot"]],
                "corners": [extra["home_corners"], extra["away_corners"]],
                "yellows": [extra["home_yellows"], extra["away_yellows"]],
                "reds": [extra["home_reds"], extra["away_reds"]],
            },

            "likely_scorers": scorers
        })

    return results

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

        home_win_p = 0.0
        draw_p = 0.0
        away_win_p = 0.0

        for hg in range(0, 6):
            for ag in range(0, 6):
                p = poisson_prob(pred_home_goals, hg) * poisson_prob(pred_away_goals, ag)
                if hg > ag:
                    home_win_p += p
                elif hg == ag:
                    draw_p += p
                else:
                    away_win_p += p

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

@app.post("/train")
def api_train(req: TrainRequest):
    seasons = req.seasons or DEFAULT_SEASONS
    logger.info("[TRAIN API] league=%s seasons=%s", req.league, seasons)
    info = train_model(req.league, seasons)
    return {"ok": True, "info": info}

@app.get("/predict/upcoming")
def api_predict_upcoming(league: int = Query(DEFAULT_LEAGUE), days_ahead: int = Query(7, ge=1, le=14)):
    try:
        model, meta = load_model_and_meta(league)
    except HTTPException:
        snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=days_ahead)
        if snapshot:
            return {"ok": True, "count": len(snapshot), "fixtures": snapshot, "source": "snapshot",
                    "snapshot_file": os.path.basename(snap_path) if snap_path else None}
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

    results = []
    if fixtures:
        results = build_predictions_for_fixtures(
            fixtures=fixtures, model=model, meta=meta, league=league, season=season,
            window_start=now, window_end=end
        )

    if not results:
        snapshot, snap_path = load_snapshot_predictions(league=league, days_ahead=days_ahead)
        if snapshot:
            return {"ok": True, "count": len(snapshot), "fixtures": snapshot, "source": "snapshot",
                    "snapshot_file": os.path.basename(snap_path) if snap_path else None}
        return {"ok": False, "count": 0, "fixtures": [], "detail": "No fixtures available. Train the model or provide cached data."}

    record_predictions_history(league, results)
    return {"ok": True, "count": len(results), "fixtures": results, "source": "model"}

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
def api_history(league: int = Query(DEFAULT_LEAGUE), limit: int = Query(50, ge=1, le=500)):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT payload
            FROM predictions_history
            WHERE league = ?
            ORDER BY kickoff_utc DESC
            LIMIT ?
            """,
            (league, limit),
        )
        rows = cur.fetchall()
        conn.close()
        fixtures = [json.loads(r[0]) for r in rows]
        return {"ok": True, "count": len(fixtures), "fixtures": fixtures}
    except Exception as e:
        logger.error("History fetch failed: %s", e)
        return {"ok": False, "count": 0, "fixtures": [], "error": str(e)}

@app.get("/team-strength")
def api_team_strength(league: int = Query(DEFAULT_LEAGUE)):
    _, meta = load_model_and_meta(league)
    team_summary = meta.get("team_summary", {})
    teams = list(team_summary.values())
    teams.sort(key=lambda r: r.get("rating", 1.0), reverse=True)
    return {"ok": True, "league": league, "teams": teams}
