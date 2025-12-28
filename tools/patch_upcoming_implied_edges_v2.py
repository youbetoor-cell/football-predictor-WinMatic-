#!/usr/bin/env python3
from pathlib import Path

TARGET = Path("football_pred_service.py")
MARK = "# --- UPCOMING_IMPLIED_EDGES_V2 ---"

BLOCK = r'''
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
'''

def main():
    s = TARGET.read_text(encoding="utf-8")
    if MARK in s:
        print("ℹ️ Patch already present.")
        return

    needle = "\n        return fixtures\n    try:"
    i = s.find(needle)
    if i == -1:
        raise SystemExit("❌ Could not find insertion point (expected: return fixtures then 'try:').")

    insert_at = i + 1  # insert before the final 'return fixtures'
    out = s[:insert_at] + BLOCK + s[insert_at:]
    TARGET.write_text(out, encoding="utf-8")
    print("✅ Inserted UPCOMING_IMPLIED_EDGES_V2")

if __name__ == "__main__":
    main()
