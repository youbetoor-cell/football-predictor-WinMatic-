#!/usr/bin/env python3
from pathlib import Path
import re, sys

P = Path("football_pred_service.py")
s = P.read_text(encoding="utf-8")

BLOCK = r'''
# --- value fields from odds + model probs (predictions.*) ---
pred = f.get("predictions") or {}
model = {
    "home": pred.get("home_win_p"),
    "draw": pred.get("draw_p"),
    "away": pred.get("away_win_p"),
}
if f.get("odds_1x2") and all(isinstance(model[k], (int, float)) for k in ("home","draw","away")):
    odds = f["odds_1x2"]
    inv = {k: (1.0/float(odds[k])) for k in ("home","draw","away") if odds.get(k)}
    total = sum(inv.values()) or 0.0
    if total > 0:
        implied = {k: inv.get(k, 0.0)/total for k in ("home","draw","away")}
        f["implied_1x2"] = implied
        edges = {k: float(model[k]) - float(implied[k]) for k in ("home","draw","away")}
        f["value_edges"] = edges
        best_side = max(edges, key=lambda k: edges[k])
        f["value_side"] = best_side
        f["best_edge"] = float(edges[best_side])
# --- end value fields ---
'''

# Find a good insertion point: right after setting odds_1x2 on fixture dict
# Matches e.g. f["odds_1x2"] = {...}
m = re.search(r'^\s*f\[\s*[\'"]odds_1x2[\'"]\s*\]\s*=\s*.+\n', s, flags=re.M)
if not m:
    print("❌ Could not find where f['odds_1x2'] is assigned in football_pred_service.py")
    sys.exit(1)

# Prevent double-insert
if "value fields from odds + model probs" in s:
    print("ℹ️ Patch already present (value fields block found). No change.")
    sys.exit(0)

insert_at = m.end()
indent = re.match(r'^\s*', m.group(0)).group(0)
block = "\n".join(indent + line if line.strip() else line for line in BLOCK.strip("\n").splitlines()) + "\n"

bak = P.with_suffix(".py.bak_upcoming_value_from_predictions")
bak.write_text(s, encoding="utf-8")
P.write_text(s[:insert_at] + block + s[insert_at:], encoding="utf-8")
print(f"✅ Patched upcoming: implied_1x2/value_edges/best_edge. Backup: {bak}")
