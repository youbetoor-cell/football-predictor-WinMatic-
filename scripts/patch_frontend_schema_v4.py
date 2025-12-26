#!/usr/bin/env python3
from __future__ import annotations

import datetime
import re
import shutil
import sys
from pathlib import Path


def ensure_import(code: str, import_line: str) -> tuple[str, bool]:
    if import_line in code:
        return code, False
    imports = list(re.finditer(r'^(from\s+\S+\s+import\s+.*|import\s+.*)$', code, re.M))
    if not imports:
        return import_line + "\n" + code, True
    insert_pos = imports[-1].end()
    return code[:insert_pos] + "\n" + import_line + code[insert_pos:], True


def ensure_patch_version(code: str) -> str:
    if re.search(r'^\s*PATCH_VERSION\s*=', code, re.M):
        return code
    pv = f'\nPATCH_VERSION = "patch-{datetime.date.today().isoformat()}-v4"\n'
    imports = list(re.finditer(r'^(from\s+\S+\s+import\s+.*|import\s+.*)$', code, re.M))
    insert_pos = imports[-1].end() if imports else 0
    return code[:insert_pos] + pv + code[insert_pos:]


def ensure_poisson_1x2(code: str) -> str:
    if re.search(r'^def\s+poisson_1x2_probs\b', code, re.M):
        return code

    fn = r'''
def poisson_1x2_probs(lam_home: float, lam_away: float, max_goals: int = 10) -> dict:
    """Compute 1X2 probabilities using independent Poisson goals."""
    try:
        lam_home = float(lam_home)
        lam_away = float(lam_away)
    except Exception:
        return {"home": None, "draw": None, "away": None}

    import math

    def pmf(lam: float, k: int) -> float:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    p_home = p_draw = p_away = 0.0
    for hg in range(0, max_goals + 1):
        ph = pmf(lam_home, hg)
        for ag in range(0, max_goals + 1):
            pa = pmf(lam_away, ag)
            p = ph * pa
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p

    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s

    return {"home": p_home, "draw": p_draw, "away": p_away}
'''.strip() + "\n\n"

    m = re.search(r'^PATCH_VERSION\s*=.*$', code, re.M)
    insert_pos = m.end() + 1 if m else 0
    return code[:insert_pos] + fn + code[insert_pos:]


def patch_health(code: str) -> str:
    m = re.search(
        r'@app\.get\("/health"\)\s*\n(def\s+\w+\([^)]*\):\s*\n)([\s\S]*?)(?=\n@app\.|\n^def\s|\Z)',
        code,
        re.M,
    )
    if not m:
        return code

    block = code[m.start():m.end()]
    if "version" in block and "commit" in block:
        return code

    fn_sig = m.group(1)
    body = (
        "    commit = os.getenv('RENDER_GIT_COMMIT') or os.getenv('GIT_COMMIT')\n"
        "    return {'ok': True, 'ts': datetime.utcnow().isoformat(), 'version': PATCH_VERSION, 'commit': commit}\n"
    )
    new_block = '@app.get("/health")\n' + fn_sig + body
    return code[:m.start()] + new_block + code[m.end():]


def patch_by_date_alias(code: str) -> str:
    # Patch the /predict/by-date handler to support ?date=YYYY-MM-DD
    m = re.search(r'@app\.get\("/predict/by-date"[^\n]*\)\s*\n(def\s+\w+\([\s\S]*?\)\s*:\s*\n)', code, re.M)
    if not m:
        return code

    sig = m.group(1)
    if "date:" in sig:
        return code
    if "from_date" not in sig:
        return code

    name = re.search(r'def\s+(\w+)\(', sig).group(1)
    new_sig = (
        f"def {name}(\n"
        "    league: int = Query(DEFAULT_LEAGUE),\n"
        "    from_date: Optional[str] = Query(None, description=\"YYYY-MM-DD start date\"),\n"
        "    to_date: Optional[str] = Query(None, description=\"YYYY-MM-DD end date (inclusive)\"),\n"
        "    date: Optional[str] = Query(None, description=\"YYYY-MM-DD (alias: from_date=to_date)\"),\n"
        "):\n"
    )

    code = code[:m.start(1)] + new_sig + code[m.end(1):]

    # Insert alias logic immediately after the new signature
    m2 = re.search(rf'def\s+{re.escape(name)}\([\s\S]*?\)\s*:\s*\n', code[m.start():], re.M)
    if not m2:
        return code
    body_start = m.start() + m2.end()

    inject = (
        "    # Backward-compatible: allow ?date=YYYY-MM-DD (maps to from_date=to_date)\n"
        "    if (from_date is None or to_date is None) and date:\n"
        "        from_date = from_date or date\n"
        "        to_date = to_date or date\n"
        "    if not from_date:\n"
        "        raise HTTPException(status_code=422, detail=\"Missing required query param: from_date (or date=YYYY-MM-DD)\")\n"
        "    if not to_date:\n"
        "        to_date = from_date\n\n"
    )

    if inject.strip() not in code:
        code = code[:body_start] + inject + code[body_start:]

    return code


def replace_winematic_middlewares_and_append(code: str) -> str:
    # Remove any existing _winmatic_* middlewares to avoid double-normalizing.
    code = re.sub(
        r'@app\.middleware\("http"\)\s*\nasync def\s+_winmatic_[\s\S]*?(?=\n@app\.|\n^def\s|\Z)',
        "",
        code,
        flags=re.M,
    )

    mw = r'''
@app.middleware("http")
async def _winmatic_frontend_compat_mw(request: Request, call_next):
    """
    Frontend compatibility layer (no schema-breaking changes to business logic).

    Ensures the UI keeps working even when the backend returns:
      - `fixtures` instead of `predictions`
      - `home` / `away` instead of `home_name` / `away_name`
      - missing 1X2 probabilities (derive from Poisson using predicted goals)
    """
    resp = await call_next(request)

    if request.url.path not in ("/predict/upcoming", "/value/upcoming", "/results/recent"):
        return resp

    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" not in ctype:
        return resp

    body = b""
    async for chunk in resp.body_iterator:
        body += chunk

    try:
        data = _json.loads(body.decode("utf-8") or "{}")
    except Exception:
        return JSONResponse(content={"ok": False, "error": "Invalid JSON"}, status_code=200)

    def _norm_fixture(fx: Any) -> Any:
        if not isinstance(fx, dict):
            return fx

        # Names: support both old+new keys
        if not fx.get("home_name") and fx.get("home"):
            fx["home_name"] = fx.get("home")
        if not fx.get("away_name") and fx.get("away"):
            fx["away_name"] = fx.get("away")

        # Keep legacy keys too
        if not fx.get("home") and fx.get("home_name"):
            fx["home"] = fx.get("home_name")
        if not fx.get("away") and fx.get("away_name"):
            fx["away"] = fx.get("away_name")

        preds = fx.get("predictions")
        if isinstance(preds, dict):
            hg = preds.get("home_goals")
            ag = preds.get("away_goals")
            if hg is not None and ag is not None:
                try:
                    probs = poisson_1x2_probs(float(hg), float(ag))
                    preds.setdefault("home_win_p", probs.get("home"))
                    preds.setdefault("draw_p", probs.get("draw"))
                    preds.setdefault("away_win_p", probs.get("away"))

                    # extra aliases (some pages use these)
                    preds.setdefault("prob_home_win", preds.get("home_win_p"))
                    preds.setdefault("prob_draw", preds.get("draw_p"))
                    preds.setdefault("prob_away_win", preds.get("away_win_p"))
                except Exception:
                    pass

        # Safe defaults so the UI doesn't hide sections
        if "odds_1x2" not in fx or fx.get("odds_1x2") is None:
            fx["odds_1x2"] = {}
        fx.setdefault("best_edge", None)
        fx.setdefault("value_side", None)

        return fx

    if isinstance(data, dict):
        if isinstance(data.get("fixtures"), list):
            data["fixtures"] = [_norm_fixture(x) for x in (data.get("fixtures") or [])]
            data.setdefault("predictions", data["fixtures"])
        elif isinstance(data.get("predictions"), list):
            data["predictions"] = [_norm_fixture(x) for x in (data.get("predictions") or [])]
            data.setdefault("fixtures", data["predictions"])

    headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
    return JSONResponse(content=data, status_code=resp.status_code, headers=headers)
'''.strip() + "\n"

    return code.rstrip() + "\n\n" + mw


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: scripts/patch_frontend_schema_v4.py path/to/football_pred_service.py")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"❌ File not found: {path}")
        return 2

    orig = path.read_text(encoding="utf-8", errors="ignore")
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"✅ Backup written: {bak}")

    code = orig
    code = ensure_patch_version(code)

    # Make sure required imports exist (idempotent)
    for line in [
        "import os",
        "from datetime import datetime",
        "import json as _json",
        "from typing import Any, Optional",
        "from fastapi import Request, Query, HTTPException",
        "from fastapi.responses import JSONResponse",
    ]:
        code, _ = ensure_import(code, line)

    code = ensure_poisson_1x2(code)
    code = patch_health(code)
    code = patch_by_date_alias(code)
    code = replace_winematic_middlewares_and_append(code)

    path.write_text(code, encoding="utf-8")
    print(f"✅ Patched: {path}")
    print("Next: python3 -m py_compile football_pred_service.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
