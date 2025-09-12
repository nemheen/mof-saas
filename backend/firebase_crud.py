# mof-application/mof-api/firebase_crud.py
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import asyncio

from firebase_admin import firestore

# ---------------- Utilities ----------------

def _doc_id(val: Any) -> str:
    """Firestore path segments must be strings and cannot contain '/'. """
    s = str(val)
    return s.replace("/", "_").strip()

def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def _run_blocking(func, *args, **kwargs):
    # Run Firestore client calls in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))

# ---------------- Favorites ----------------

async def add_to_favorites(user_id: Any, mof_id: Any, filename: Optional[str] = None) -> None:
    uid = _doc_id(user_id)
    mid = _doc_id(mof_id)
    db = firestore.client()
    doc_ref = db.collection("users").document(uid).collection("favorites").document(mid)
    payload = {
        "filename": filename or mid,
        "added_at": firestore.SERVER_TIMESTAMP,
    }
    await _run_blocking(doc_ref.set, payload, True)

async def remove_from_favorites(user_id: Any, mof_id: Any) -> None:
    uid = _doc_id(user_id)
    mid = _doc_id(mof_id)
    db = firestore.client()
    doc_ref = db.collection("users").document(uid).collection("favorites").document(mid)
    await _run_blocking(doc_ref.delete)

async def list_favorites(user_id: Any, limit: int = 50, cursor: Optional[str] = None) -> Dict[str, Any]:
    uid = _doc_id(user_id)
    db = firestore.client()
    base = (
        db.collection("users").document(uid)
        .collection("favorites")
        .order_by("added_at", direction=firestore.Query.DESCENDING)
    )
    if cursor:
        cursor_ref = (
            db.collection("users").document(uid)
            .collection("favorites").document(cursor)
        )
        snap = await _run_blocking(cursor_ref.get)
        q = base.start_after(snap) if snap.exists else base
    else:
        q = base
    q = q.limit(limit)

    snaps = await _run_blocking(q.stream)
    items: List[Dict[str, Any]] = []
    last_id: Optional[str] = None
    for s in snaps:
        data = s.to_dict() or {}
        data["id"] = s.id
        items.append(data)
        last_id = s.id

    return {"items": items, "next_cursor": last_id if len(items) == limit else None}

# ---------------- History ----------------

async def add_user_activity(user_id: Any, activity_type: str, details: Dict[str, Any]) -> None:
    uid = _doc_id(user_id)
    db = firestore.client()
    col = db.collection("users").document(uid).collection("history")
    payload = {"type": activity_type, "details": details, "ts": firestore.SERVER_TIMESTAMP}
    await _run_blocking(col.add, payload)

async def get_user_activities(user_id: Any, limit: int = 20) -> Dict[str, Any]:
    uid = _doc_id(user_id)
    db = firestore.client()
    col = (
        db.collection("users").document(uid)
        .collection("history")
        .order_by("ts", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    snaps = await _run_blocking(col.stream)
    items: List[Dict[str, Any]] = []
    for s in snaps:
        data = s.to_dict() or {}
        data["id"] = s.id
        ts = data.get("ts")
        if isinstance(ts, datetime):
            data["ts_iso"] = _utc_iso(ts)
        items.append(data)
    return {"items": items}

# ---------------- My Files ----------------

async def add_prediction_to_my_files(
    user_id: Any,
    filename: str,
    prediction: Dict[str, Any],
    plot: str,
    adsorption_data: Dict[str, Any],
) -> None:
    uid = _doc_id(user_id)
    safe_name = _doc_id(filename)
    db = firestore.client()
    doc_ref = db.collection("users").document(uid).collection("my_files").document(safe_name)
    payload = {
        "filename": filename,
        "prediction": prediction,
        "plot_base64": plot,
        "adsorption_data": adsorption_data,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }
    await _run_blocking(doc_ref.set, payload, True)
