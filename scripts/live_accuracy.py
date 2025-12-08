"""
Compute LIVE accuracy metrics for WinMatic using:
- Stored predictions in artifacts/history.db (predictions_history table)
- Actual results from API-Football

Usage:
    source env/bin/activate
    python scripts/live_accuracy.py

This does NOT change your API or database schema.
"""

import os
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv

# --- CONFIG ---
DB_PATH = os.path.join("artifacts", "history.db")
LEAGUE_ID = 39  # Premier League
DAYS_BACK = 30  # how many days back to evaluate

load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY", "")


def _multiclass_logloss(y_true: List[int], probs: List[List[float]]) -> float:
    """Same idea as your backend: log-loss for 3-way probabilities."""
    eps = 1e-15
    losses = []
    for y, p in zip(y_true, probs):
        p_clipped = [max(min(v, 1.0 - eps), eps) for v in p]
        losses.append(-math.log(p_clipped[y]))
    return sum(losses) / len(losses) if losses else float("nan")


def _multiclass_brier(y_true: List[int], probs: List[List[float]]) -> float:
    """Same idea as in backend: Brier score for 3-way."""
    scores = []
    for y, p in zip(y_true, probs):
        target = [0.0, 0.0, 0.0]
        target[y] = 1.0
        scores.append(sum((pi - ti) ** 2 for pi, ti in zip(p, target)))
    return sum(scores) / len(scores) if scores else float("nan")


def fetch_finished_fixtures(league: int, days_back: int) -> List[Dict[str, Any]]:
    """Fetch finished fixtures from API-Football for the last N days."""
    if not API_KEY:
        raise RuntimeError("API_FOOTBALL_KEY is not set in .env")

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days_back)

    url = (
        "https://v3.football.api-sports.io/fixtures"
        f"?league={league}&season={end.year}&from={start}&to={end}"
    )
    headers = {"x-apisports-key": API_KEY}

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    fixtures = data.get("response", [])
    # Only keep finished matches
    finished = []
    for fx in fixtures:
        status = fx.get("fixture", {}).get("status", {}).get("short")
        if status in {"FT", "AET", "PEN"}:
            finished.append(fx)
    return finished


def load_predictions_from_db() -> Dict[int, Dict[str, Any]]:
    """
    Load stored predictions from predictions_history table.
    Keyed by fixture_id.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT league, fixture_id, kickoff_utc, payload
        FROM predictions_history
        WHERE league = ?
        """,
        (LEAGUE_ID,),
    )
    rows = cur.fetchall()
    conn.close()

    preds: Dict[int, Dict[str, Any]] = {}
    for league, fid, kickoff, payload in rows:
        try:
            obj = json.loads(payload)
            preds[int(fid)] = obj
        except Exception:
            continue
    return preds


def result_code_from_goals(home_goals: int, away_goals: int) -> int:
    """Map actual goals to 3-way result: 0=home, 1=draw, 2=away."""
    if home_goals > away_goals:
        return 0
    elif home_goals == away_goals:
        return 1
    else:
        return 2


def main() -> None:
    print(f"📊 Evaluating live accuracy for league {LEAGUE_ID}, last {DAYS_BACK} days")
    preds_by_fixture = load_predictions_from_db()
    print(f"Loaded {len(preds_by_fixture)} stored predictions from {DB_PATH}")

    finished_fixtures = fetch_finished_fixtures(LEAGUE_ID, DAYS_BACK)
    print(f"Fetched {len(finished_fixtures)} finished fixtures from API-Football")

    y_true: List[int] = []
    proba: List[List[float]] = []
    hit_count = 0
    total_used = 0

    for fx in finished_fixtures:
        fixture = fx.get("fixture", {})
        goals = fx.get("goals", {})

        fid = int(fixture.get("id"))
        if fid not in preds_by_fixture:
            # no stored prediction for this fixture (maybe you didn't call /predict/upcoming)
            continue

        pred_obj = preds_by_fixture[fid]
        pred_block = pred_obj.get("predictions") or {}

        try:
            p_home = float(pred_block.get("home_win_p"))
            p_draw = float(pred_block.get("draw_p"))
            p_away = float(pred_block.get("away_win_p"))
        except (TypeError, ValueError):
            continue

        if any(math.isnan(v) for v in [p_home, p_draw, p_away]):
            continue

        # normalize just in case
        s = p_home + p_draw + p_away
        if s <= 0:
            continue
        p_home /= s
        p_draw /= s
        p_away /= s

        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            continue

        hg_int = int(hg)
        ag_int = int(ag)
        y = result_code_from_goals(hg_int, ag_int)

        y_true.append(y)
        proba.append([p_home, p_draw, p_away])
        total_used += 1

        # hit rate: was the top prob outcome correct?
        top_idx = max(range(3), key=lambda i: [p_home, p_draw, p_away][i])
        if top_idx == y:
            hit_count += 1

    if not y_true:
        print("⚠️ No overlapping matches with both predictions and final scores.")
        print("   Make sure you call /predict/upcoming regularly *before* matches are played.")
        return

    logloss = _multiclass_logloss(y_true, proba)
    brier = _multiclass_brier(y_true, proba)
    hit_rate = hit_count / total_used if total_used else float("nan")

    print("\n✅ LIVE ACCURACY (last {} days)".format(DAYS_BACK))
    print(f"Matches evaluated : {total_used}")
    print(f"Logloss (1X2)     : {logloss:.4f}")
    print(f"Brier (1X2)       : {brier:.4f}")
    print(f"Hit rate (top pick correct) : {hit_rate*100:5.1f}%")

    print("\nNote: this is based only on matches where:")
    print("  - You had a stored prediction in history.db")
    print("  - The match is now finished according to API-Football\n")


if __name__ == "__main__":
    main()
