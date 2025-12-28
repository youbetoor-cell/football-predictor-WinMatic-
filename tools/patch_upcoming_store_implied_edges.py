#!/usr/bin/env python3
from pathlib import Path

P = Path("football_pred_service.py")

def main():
    s = P.read_text(encoding="utf-8")

    marker = "# --- STORE_UPCOMING_IMPLIED_EDGES_V1 ---"
    if marker in s:
        print("ℹ️ Patch already present.")
        return

    old = """                        implied = implied_probs_1x2(odds)
                        if implied and isinstance(implied, dict):
                            p = fx.get('predictions') or {}
                            edges = {}
                            for side, pk in [('home','home_win_p'),('draw','draw_p'),('away','away_win_p')]:
                                mp = implied.get(side)
                                pp = p.get(pk)
                                if mp is not None and pp is not None:
                                    edges[side] = float(pp) - float(mp)
                            if edges:
                                best_side = max(edges, key=lambda k: edges[k])
                                fx['best_edge'] = round(edges[best_side], 3)
                                fx['value_side'] = best_side
"""

    new = """                        implied = implied_probs_1x2(odds)
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
"""

    if old not in s:
        raise SystemExit("❌ Could not find the exact implied/edges block to patch (file changed).")

    s2 = s.replace(old, new)
    P.write_text(s2, encoding="utf-8")
    print("✅ Patched: store implied_1x2/value_edges/best_edge/value_side on upcoming.")

if __name__ == "__main__":
    main()
