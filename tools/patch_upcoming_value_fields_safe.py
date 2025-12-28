#!/usr/bin/env python3
from pathlib import Path
import re, sys

P = Path("football_pred_service.py")
s = P.read_text(encoding="utf-8")

decor = re.search(r'^\s*@app\.get\(\s*["\']/predict/upcoming["\']', s, flags=re.M)
if not decor:
    raise SystemExit("❌ Could not find @app.get('/predict/upcoming')")

# find function def after decorator
fn = re.search(r'^\s*def\s+([A-Za-z0-9_]+)\s*\(.*\)\s*:\s*$', s[decor.end():], flags=re.M)
if not fn:
    raise SystemExit("❌ Could not find function def after upcoming decorator")

fn_start = decor.end() + fn.start()
m_next = re.search(r'^\s*@app\.', s[fn_start+1:], flags=re.M)
fn_end = (fn_start + 1 + m_next.start()) if m_next else len(s)
segment = s[fn_start:fn_end]

MARK = "SAFE_UPCOMING_VALUE_FIELDS_V1"
if MARK in segment:
    print("ℹ️ Patch already present. No change.")
    sys.exit(0)

# find first return in function
ret = re.search(r'^\s*return\b', segment, flags=re.M)
if not ret:
    raise SystemExit("❌ Could not find a return in upcoming handler")

indent = re.match(r'^(\s*)return\b', segment[ret.start():], flags=re.M).group(1)

BLOCK = f"""
{indent}# --- {MARK} ---
{indent}try:
{indent}    # Prefer explicit response dict name if present
{indent}    _out = locals().get("out") or locals().get("resp") or None
{indent}    if isinstance(_out, dict) and isinstance(_out.get("fixtures"), list):
{indent}        _fxs = _out["fixtures"]
{indent}    else:
{indent}        # Fall back to common list variable names
{indent}        _fxs = locals().get("fixtures") or locals().get("xs") or []
{indent}
{indent}    if isinstance(_fxs, list):
{indent}        for f in _fxs:
{indent}            if not isinstance(f, dict):
{indent}                continue
{indent}            odds = f.get("odds_1x2") or {{}}
{indent}            pred = f.get("predictions") or {{}}
{indent}            model = {{
{indent}                "home": pred.get("home_win_p"),
{indent}                "draw": pred.get("draw_p"),
{indent}                "away": pred.get("away_win_p"),
{indent}            }}
{indent}            # Guard: only compute when everything is numeric and odds > 0
{indent}            if not (isinstance(odds, dict) and all(isinstance(model[k], (int,float)) for k in ("home","draw","away"))):
{indent}                continue
{indent}            inv = {{}}
{indent}            for k in ("home","draw","away"):
{indent}                v = odds.get(k)
{indent}                if isinstance(v, (int,float)) and float(v) > 0:
{indent}                    inv[k] = 1.0/float(v)
{indent}            total = float(sum(inv.values()))
{indent}            if total <= 0:
{indent}                continue
{indent}            implied = {{k: float(inv.get(k,0.0))/total for k in ("home","draw","away")}}
{indent}            edges = {{k: float(model[k]) - float(implied[k]) for k in ("home","draw","away")}}
{indent}            best_side = max(edges, key=lambda kk: edges[kk])
{indent}            f["implied_1x2"] = implied
{indent}            f["value_edges"] = edges
{indent}            f["value_side"] = best_side
{indent}            f["best_edge"] = float(edges[best_side])
{indent}except Exception:
{indent}    pass
{indent}# --- end {MARK} ---
"""

insert_at = fn_start + ret.start()
bak = P.with_suffix(".py.bak_upcoming_value_safe")
bak.write_text(s, encoding="utf-8")
P.write_text(s[:insert_at] + BLOCK + s[insert_at:], encoding="utf-8")
print(f"✅ Patched upcoming (safe) before return. Backup: {bak}")
