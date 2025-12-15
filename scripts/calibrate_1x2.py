import json, math, os
from datetime import datetime, timezone
import requests

LEAGUE = int(os.environ.get("LEAGUE", "39"))
LAST_N = int(os.environ.get("LAST_N", "300"))
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", str(LAST_N)))
OUT_DIR = os.environ.get("OUT_DIR", "artifacts")
OUT_PATH = os.path.join(OUT_DIR, f"calibration_{LEAGUE}.json")

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8001")

def fetch_backtest():
    url = f"{BASE_URL}/backtest/1x2?league={LEAGUE}&season=2025&last_n={LAST_N}&sample_limit={SAMPLE_LIMIT}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json(), url

def apply_temp_drawmult(p, T, dm):
    eps = 1e-15
    ph = max(eps, float(p["home"]))
    pd = max(eps, float(p["draw"]))
    pa = max(eps, float(p["away"]))

    invT = 1.0 / float(T)
    ph = ph ** invT
    pd = pd ** invT
    pa = pa ** invT

    pd = pd * float(dm)

    s = ph + pd + pa
    if s <= 0:
        return {"home": 1/3, "draw": 1/3, "away": 1/3}
    return {"home": ph/s, "draw": pd/s, "away": pa/s}

def logloss(p_true):
    return -math.log(max(p_true, 1e-15))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bt, _ = fetch_backtest()
    sample = bt.get("sample") or []
    fixtures_scored = int(bt.get("fixtures_scored") or 0)

    rows = []
    for r in sample:
        actual = r.get("actual_result")
        raw = r.get("raw_probs") or r.get("model_probs")
        if actual not in ("home", "draw", "away"):
            continue
        if not raw or any(k not in raw for k in ("home", "draw", "away")):
            continue
        rows.append((actual, raw))

    n = len(rows)
    print("usable_rows:", n, "fixtures_scored:", fixtures_scored, "sample_len:", len(sample))
    if n < 50:
        print("WARNING: too few usable rows — calibration may be unstable.")

    fine = []
    Ts = [0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60]
    dms = [0.70, 0.80, 0.90, 1.00, 1.10]

    for T in Ts:
        for dm in dms:
            ll = 0.0
            correct = 0
            for actual, raw in rows:
                probs = apply_temp_drawmult(raw, T, dm)
                ll += logloss(probs[actual])
                pred = max(probs, key=probs.get)
                if pred == actual:
                    correct += 1
            ll /= max(1, n)
            acc = correct / max(1, n)
            fine.append((ll, -acc, T, dm))

    fine.sort()
    best_ll, neg_acc, best_T, best_dm = fine[0]
    best_acc = -neg_acc

    cal = {
        "temperature": float(best_T),
        "draw_mult": float(best_dm),
        "fitted_on": {
            "league": LEAGUE,
            "season": bt.get("season"),
            "fixtures_scored": fixtures_scored,
            "n_used": n,
            "source_url": f"{BASE_URL}/backtest/1x2?league={LEAGUE}&season={bt.get('season')}&last_n={LAST_N}&sample_limit={SAMPLE_LIMIT}",
        },
        "metrics": {
            "logloss": round(float(best_ll), 6),
            "accuracy": round(float(best_acc), 6),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    print("WROTE:", OUT_PATH)
    print("BEST: temperature=%s draw_mult=%s  logloss=%.6f  accuracy=%.4f  n=%d" % (best_T, best_dm, best_ll, best_acc, n))

if __name__ == "__main__":
    main()
