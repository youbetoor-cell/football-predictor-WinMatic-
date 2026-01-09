import os, json, glob, re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

def _parse_iso_maybe(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def _ts_to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _age_minutes(dt: Optional[datetime]) -> Optional[int]:
    if not dt:
        return None
    return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 60))

def _infer_ts_from_filename(path: str) -> Optional[datetime]:
    # supports ..._YYYYMMDD.json or ..._YYYYMMDDHHMMSS.json
    m = re.search(r'_(\d{8})(\d{6})?\.json$', os.path.basename(path))
    if not m:
        return None
    ymd = m.group(1)
    hms = m.group(2) or "000000"
    try:
        dt = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def load_latest_upcoming_snapshot(artifacts_dir: str, league: int, days_ahead: int) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Returns (payload, snapshot_iso, snapshot_file)
    Searches common snapshot patterns and picks the newest by filename timestamp.
    """
    snap_dir = os.path.join(artifacts_dir, "snapshots")
    patterns = [
        os.path.join(snap_dir, f"upcoming_{league}_{days_ahead}_*.json"),
        os.path.join(snap_dir, f"pred_{league}_{days_ahead}_*.json"),
        os.path.join(snap_dir, f"upcoming_{league}_{days_ahead}.json"),
        os.path.join(snap_dir, f"pred_{league}_{days_ahead}.json"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    if not candidates:
        return None, None, None

    # choose newest by inferred timestamp, fallback to mtime
    def sort_key(p: str):
        dt = _infer_ts_from_filename(p)
        if dt:
            return dt.timestamp()
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0

    candidates.sort(key=sort_key, reverse=True)
    path = candidates[0]

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None, None, path

    # allow either {"payload": {...}} or raw payload dict
    payload = obj.get("payload") if isinstance(obj, dict) and "payload" in obj else obj

    # best-effort timestamp extraction
    ts = None
    if isinstance(obj, dict):
        ts = _parse_iso_maybe(obj.get("snapshot_utc") or obj.get("snapshot_ts") or obj.get("created_utc") or obj.get("created_at"))
    if not ts:
        ts = _infer_ts_from_filename(path)

    snapshot_iso = _ts_to_iso(ts) if ts else None
    return payload if isinstance(payload, dict) else {"data": payload}, snapshot_iso, os.path.basename(path)

def attach_snapshot_meta(resp: Dict[str, Any], snapshot_iso: Optional[str], snapshot_file: Optional[str], source: str) -> Dict[str, Any]:
    dt = _parse_iso_maybe(snapshot_iso) if snapshot_iso else None
    resp = dict(resp)
    resp["source"] = source
    resp["snapshot_utc"] = snapshot_iso
    resp["snapshot_age_minutes"] = _age_minutes(dt)
    if snapshot_file:
        resp["snapshot_file"] = snapshot_file
    return resp
