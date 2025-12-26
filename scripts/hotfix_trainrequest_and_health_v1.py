#!/usr/bin/env python3
from __future__ import annotations
import re, sys, shutil
from pathlib import Path
from datetime import date

def ensure_import(text: str, line: str) -> str:
    if re.search(r'^\s*' + re.escape(line) + r'\s*$', text, flags=re.M):
        return text
    # insert after last import
    imports = list(re.finditer(r'^(from\s+\S+\s+import\s+.*|import\s+.*)$', text, flags=re.M))
    if not imports:
        return line + "\n" + text
    pos = imports[-1].end()
    return text[:pos] + "\n" + line + text[pos:]

def main():
    if len(sys.argv) != 2:
        print("Usage: scripts/hotfix_trainrequest_and_health_v1.py football_pred_service.py")
        return 2

    p = Path(sys.argv[1])
    s = p.read_text(encoding="utf-8", errors="ignore")

    bak = p.with_suffix(p.suffix + ".bak")
    shutil.copy2(p, bak)
    print(f"✅ Backup: {bak}")

    # Add PATCH_VERSION if missing
    if not re.search(r'^\s*PATCH_VERSION\s*=', s, flags=re.M):
        pv = f'\nPATCH_VERSION = "patch-{date.today().isoformat()}-trainrequest-health-v1"\n'
        # add after imports
        imports = list(re.finditer(r'^(from\s+\S+\s+import\s+.*|import\s+.*)$', s, flags=re.M))
        pos = imports[-1].end() if imports else 0
        s = s[:pos] + pv + s[pos:]
        print("✅ Added PATCH_VERSION")

    # Ensure imports needed for TrainRequest and /health
    s = ensure_import(s, "import os")
    s = ensure_import(s, "from datetime import datetime")
    s = ensure_import(s, "from typing import Optional, List")
    s = ensure_import(s, "from pydantic import BaseModel, Field")

    # Insert TrainRequest if missing
    if not re.search(r'^\s*class\s+TrainRequest\b', s, flags=re.M):
        m = re.search(r'^\s*def\s+api_train\s*\(', s, flags=re.M)
        if not m:
            print("❌ Could not find def api_train(...). Aborting.")
            return 1

        insert_at = m.start()
        block = (
            "\n\nclass TrainRequest(BaseModel):\n"
            "    league: int = Field(DEFAULT_LEAGUE, description=\"League ID\")\n"
            "    seasons: Optional[List[int]] = Field(default=None, description=\"Seasons list, e.g. [2023, 2024]\")\n\n"
        )
        s = s[:insert_at] + block + s[insert_at:]
        print("✅ Inserted TrainRequest above api_train()")
    else:
        print("ℹ️ TrainRequest already exists")

    # Patch /health to include version + commit (if the handler exists)
    # It’s OK if this doesn’t match; we won’t break anything.
    health = re.search(
        r'@app\.get\("/health"\)\s*\n(def\s+\w+\([^)]*\):\s*\n)([\s\S]*?)(?=\n@app\.|\n^def\s|\Z)',
        s,
        flags=re.M,
    )
    if health:
        block = s[health.start():health.end()]
        if ("version" not in block) or ("commit" not in block):
            fn_sig = health.group(1)
            new_body = (
                "    commit = os.getenv('RENDER_GIT_COMMIT') or os.getenv('GIT_COMMIT')\n"
                "    return {'ok': True, 'ts': datetime.utcnow().isoformat(), 'version': PATCH_VERSION, 'commit': commit}\n"
            )
            new_block = '@app.get("/health")\n' + fn_sig + new_body
            s = s[:health.start()] + new_block + s[health.end():]
            print("✅ Patched /health to include version + commit")
        else:
            print("ℹ️ /health already includes version/commit")
    else:
        print("⚠️ Could not find /health handler block to patch (skipped)")

    p.write_text(s, encoding="utf-8")
    print(f"✅ Wrote: {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
