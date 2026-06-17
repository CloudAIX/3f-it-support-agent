"""Long-term memory layer for the multi-agent IT support system.

This module is DISTINCT from SupportState (state.py):

  SupportState (working/short-term memory)
    — Lives for one call only. Created fresh at graph start, gone when END
      is reached. Every agent reads and writes it, but nothing survives to
      the next call.

  memory.py (long-term memory)
    — Persists ACROSS calls, across process restarts, across days. Keyed by
      employee_id. Two categories of stored data:

      history       — a log of past calls (issue, outcome, timestamp).
                      Written by review_agent at the END of each call.
                      Read by intake_agent for the warm-start.

      pending       — open tickets, recent notifications, flagged concerns.
                      Written by external systems (IVR, ticketing, ops).
                      Read by intake_agent to compose a proactive greeting.

The proactive greeting pattern: at the start of each call, intake_agent loads
both history and pending context so the agent can open with "Hi Priya — I see
we raised a ticket about your VPN yesterday, is that what you're calling about?"
instead of a generic "how can I help".

Production note: swap _STORE_PATH / _load_store / _save_store for a
key-value store (Redis, DynamoDB, Firestore) or a vector DB (for semantic
recall of similar past issues). The public API (load_history, save_call,
get_pending_context, set_pending_context) stays the same.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# Runtime data — NOT committed (listed in .gitignore).
_STORE_PATH = Path(__file__).parent / "memory_store.json"

# Store schema per employee_id key:
# {
#   "history": [{"issue": str, "outcome": str, "timestamp": str}, ...],
#   "pending": [{"type": str, ...arbitrary fields...}, ...]
# }


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
    _STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _record(key: str, store: dict) -> dict:
    """Return the employee record dict, migrating the legacy list format."""
    raw = store.get(key, {})
    if isinstance(raw, list):
        # Migrate from previous flat list-of-history-items layout.
        return {"history": raw, "pending": []}
    return raw


# ---------------------------------------------------------------------------
# Public API — history
# ---------------------------------------------------------------------------

def load_history(employee_id: str) -> list[dict]:
    """Return all past call summaries for this employee ([] if none)."""
    return _record(employee_id.upper(), _load_store()).get("history", [])


def save_call(employee_id: str, summary: dict) -> None:
    """Append a call summary to this employee's history and persist it.

    Adds a UTC timestamp automatically.
    Args:
        summary: dict with at least "issue" (str) and "outcome" (str).
    """
    store = _load_store()
    key = employee_id.upper()
    rec = _record(key, store)
    rec.setdefault("history", []).append({
        "issue":     summary.get("issue", ""),
        "outcome":   summary.get("outcome", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    store[key] = rec
    _save_store(store)


# ---------------------------------------------------------------------------
# Public API — pending context
# ---------------------------------------------------------------------------

def get_pending_context(employee_id: str) -> list[dict]:
    """Return pending context items for this employee ([] if none).

    Each item is a free-form dict. Recognised 'type' values used in this
    system: "open_ticket", "notification". In production these would be
    pushed from the ticketing system or IVR platform.
    """
    return _record(employee_id.upper(), _load_store()).get("pending", [])


def set_pending_context(employee_id: str, items: list[dict]) -> None:
    """Overwrite the pending context list for this employee.

    Used by external systems (IVR, ticketing, ops) to register open items
    before the call is answered. Also used by the demo seed in __main__.
    """
    store = _load_store()
    key = employee_id.upper()
    rec = _record(key, store)
    rec["pending"] = items
    store[key] = rec
    _save_store(store)
