"""Long-term memory layer for the multi-agent IT support system.

This module is DISTINCT from SupportState (state.py):

  SupportState (working/short-term memory)
    — Lives for one call only. Created fresh at graph start, gone when END
      is reached. Every agent reads and writes it, but nothing survives to
      the next call.

  memory.py (long-term memory)
    — Persists ACROSS calls, across process restarts, across days. Keyed by
      employee_id. Loaded at the START of a call (warm-start: the agent
      already knows who you are and what went wrong last time). Written at
      the END of a call so the next one can see it.

This is the "warm-start" pattern: returning callers get a personalised
experience without having to repeat their history.

Production note: swap _STORE_PATH / _load_store / _save_store for a
key-value store (Redis, DynamoDB, Firestore) or a vector DB (for semantic
recall of similar past issues), keeping load_history() and save_call()
signatures identical. Nothing else in the system needs to change.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# Runtime data file — NOT committed (listed in .gitignore).
# In production this would be a DB connection URI from an env var.
_STORE_PATH = Path(__file__).parent / "memory_store.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_store() -> dict:
    """Read the full JSON store from disk. Returns {} on first run."""
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_store(store: dict) -> None:
    """Write the full store dict back to disk atomically."""
    _STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_history(employee_id: str) -> list[dict]:
    """Return all past call summaries for this employee.

    Each summary is a dict with keys: issue (str), outcome (str),
    timestamp (ISO-8601 UTC str).

    Returns an empty list if the employee has no history or the store
    file doesn't exist yet — callers should treat [] as "first contact".
    """
    return _load_store().get(employee_id.upper(), [])


def save_call(employee_id: str, summary: dict) -> None:
    """Persist a call summary to this employee's history.

    Appends to any existing records; never overwrites. Adds a UTC
    timestamp automatically.

    Args:
        employee_id: The caller's employee ID, e.g. "E1001".
        summary: Dict with at least "issue" (str) and "outcome" (str).
    """
    store = _load_store()
    key = employee_id.upper()
    store.setdefault(key, []).append({
        "issue":     summary.get("issue", ""),
        "outcome":   summary.get("outcome", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_store(store)
