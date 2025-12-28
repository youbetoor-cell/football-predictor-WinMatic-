#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("football_pred_service.py")

HELPER_NAME = "_attach_odds_and_value_fields_upcoming"

HELPER_CODE = r'''
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
'''.lstrip("\n")


def extract_function(src: str, func_name: str) -> tuple[int, int, str] | None:
    # Returns (start, end, block)
    m = re.search(rf"^def\s+{re.escape(func_name)}\s*\(", src, flags=re.M)
    if not m:
        return None
    start = m.start()
    # end = next top-level def or decorator
    m2 = re.search(r"^(def\s+|@app\.)", src[m.end():], flags=re.M)
    if not m2:
        end = len(src)
    else:
        end = m.end() + m2.start()
    return start, end, src[start:end]


def main():
    if not TARGET.exists():
        raise SystemExit("❌ football_pred_service.py not found (run this from repo root).")

    src = TARGET.read_text(encoding="utf-8", errors="ignore")
    bak = TARGET.with_suffix(".py.bak")
    bak.write_text(src, encoding="utf-8")

    changed = 0

    # 1) Fix TrainRequest NameError at import time (quote the annotation)
    new_src = re.sub(
        r"def\s+api_train\s*\(\s*req\s*:\s*TrainRequest\s*\)",
        'def api_train(req: "TrainRequest")',
        src,
    )
    if new_src != src:
        src = new_src
        changed += 1
        print("✅ Patched: api_train(req: TrainRequest) -> api_train(req: \"TrainRequest\")")

    # 2) Inject helper function once (place after compute_value_edges if possible)
    if HELPER_NAME not in src:
        ins_pos = None
        m = re.search(r"^def\s+compute_value_edges\s*\(", src, flags=re.M)
        if m:
            # insert after that function block
            block = extract_function(src, "compute_value_edges")
            if block:
                _, end, _ = block
                ins_pos = end
        if ins_pos is None:
            ins_pos = len(src)

        src = src[:ins_pos] + "\n\n" + HELPER_CODE + "\n\n" + src[ins_pos:]
        changed += 1
        print("✅ Injected helper:", HELPER_NAME)

    # 3) Patch api_predict_upcoming signature to include odds params (default include_odds=1)
    blk = extract_function(src, "api_predict_upcoming")
    if not blk:
        raise SystemExit("❌ Could not find def api_predict_upcoming(...).")
    start, end, block = blk

    # add params
    m_sig = re.search(r"def\s+api_predict_upcoming\s*\((.*?)\)\s*:", block, flags=re.S)
    if not m_sig:
        raise SystemExit("❌ Could not parse api_predict_upcoming signature.")
    sig = m_sig.group(1)

    if "include_odds" not in sig:
        insert = (
            '\n    include_odds: int = Query(1, ge=0, le=1, description="Attach odds_1x2 + value fields"),'
            '\n    odds_limit: int = Query(25, ge=0, le=50, description="Max fixtures to fetch odds for"),'
            '\n    min_edge: float = Query(0.0, ge=0.0, le=1.0, description="Filter out fixtures with best_edge < min_edge"),\n'
        )
        sig2 = re.sub(
            r"(days_ahead:\s*int\s*=\s*Query\([^\n]*\)\s*,?\n)",
            r"\1" + insert,
            sig,
            count=1,
        )
        if sig2 == sig:
            sig2 = sig + insert

        block2 = block[:m_sig.start(1)] + sig2 + block[m_sig.end(1):]
        block = block2
        changed += 1
        print("✅ Patched: api_predict_upcoming signature now supports include_odds/odds_limit/min_edge")

    # 4) Patch body: unify returns + attach odds to BOTH model & snapshot payloads
    # Replace the "if not results: ... return snapshot ... return model" region if present
    pat = re.compile(
        r"""
        if\s+not\s+results\s*:\s*\n
        (?P<ind>\s+)snap\s*=\s*load_upcoming_snapshot\([^\n]*\)\s*\n
        (?P=ind)if\s+snap\s*:\s*\n
        (?P=ind)\s+return\s+\{.*?"source"\s*:\s*"snapshot".*?\}\s*\n
        \s*\n
        (?P=ind)return\s+\{.*?"source"\s*:\s*"model".*?\}\s*
        """,
        re.S | re.X,
    )
    mm = pat.search(block)
    if mm:
        ind = mm.group("ind")
        repl = f'''if not results:
{ind}snap = load_upcoming_snapshot(league=league, days_ahead=days_ahead)
{ind}if snap:
{ind}    fixtures_payload = snap.get("fixtures") or []
{ind}    source = "snapshot"
{ind}    snapshot_file = snap.get("snapshot_file")
{ind}else:
{ind}    fixtures_payload = []
{ind}    source = "model"
{ind}    snapshot_file = None
{ind}else:
{ind}fixtures_payload = results
{ind}source = "model"
{ind}snapshot_file = None

{ind}fixtures_payload = _attach_odds_and_value_fields_upcoming(
{ind}    fixtures_payload,
{ind}    include_odds=include_odds,
{ind}    odds_limit=odds_limit,
{ind}    min_edge=min_edge,
{ind})

{ind}resp = {{
{ind}    "ok": True,
{ind}    "count": len(fixtures_payload),
{ind}    "fixtures": fixtures_payload,
{ind}    "source": source,
{ind}}}
{ind}if snapshot_file:
{ind}    resp["snapshot_file"] = snapshot_file
{ind}return resp
'''
        block = block[:mm.start()] + repl + block[mm.end():]
        changed += 1
        print("✅ Patched: /predict/upcoming now attaches odds+edges for model AND snapshot responses")
    else:
        print("⚠️ Could not match the exact snapshot-return block in api_predict_upcoming.")
        print("   If your function structure is different, paste your api_predict_upcoming() here and I’ll adapt the patch.")

    # write back
    src = src[:start] + block + src[end:]
    TARGET.write_text(src, encoding="utf-8")

    if changed:
        print(f"\n✅ Done. Backup written to: {bak}")
    else:
        print("\nℹ️ No changes were necessary (already patched).")


if __name__ == "__main__":
    main()
