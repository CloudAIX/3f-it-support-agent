"""LangChain tool definitions for the multi-agent IT support system.

Four tools mirror the FastAPI endpoints in the voice agent but are implemented
as standalone LangChain @tool functions so LangGraph nodes can call them
directly without an HTTP round-trip. Stub data is duplicated here intentionally
— tools.py is self-contained and does not import from the parent main.py.

READ tools (lookup_employee, search_kb) are autonomous: agents may call them
freely without caller confirmation.

WRITE tools (create_ticket, escalate) are gated: the agent must confirm with
the caller before invoking them, and every call is logged.
"""

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Stub data — swap for real DB / ITSM calls without changing tool contracts
# ---------------------------------------------------------------------------

_EMPLOYEES = {
    "E1001": {"name": "Priya Nair",      "department": "Finance",        "verified": True},
    "E1002": {"name": "Tom Becker",      "department": "Engineering",    "verified": True},
    "E1003": {"name": "Sara Lindqvist",  "department": "Sales",          "verified": False},
    "A10":   {"name": "Nathan",          "department": "Innovation Lab", "verified": True},
}

_KB = [
    {
        "id": "KB001",
        "title": "Password reset",
        "keywords": ["password", "reset", "locked out", "login", "log in", "sign in", "signin"],
        "steps": [
            "Go to the self-service portal at portal.example.com.",
            "Select 'Forgot password' and enter your work email.",
            "Follow the emailed link and set a new password.",
            "If no email arrives within five minutes the account may be locked — escalate.",
        ],
    },
    {
        "id": "KB002",
        "title": "VPN or remote access",
        "keywords": ["vpn", "remote", "access", "connect", "network", "offsite", "off site", "work from home"],
        "steps": [
            "Confirm the VPN client is installed and up to date.",
            "Sign in with your work email and password.",
            "If sign-in fails, restart the client and try once more.",
            "If it still fails the account may not be enabled for remote access — escalate.",
        ],
    },
    {
        "id": "KB003",
        "title": "Software install request",
        "keywords": ["install", "software", "application", "app", "program", "download"],
        "steps": [
            "Open the company software catalogue from the desktop.",
            "Search for the application and select 'Request install'.",
            "Approved apps install automatically within an hour.",
            "If the app is not in the catalogue it needs manager approval — raise a ticket.",
        ],
    },
    {
        "id": "KB004",
        "title": "Printer not working",
        "keywords": ["printer", "print", "printing", "scan", "scanner", "paper jam"],
        "steps": [
            "Check the printer is powered on and shows no error light.",
            "Confirm the correct printer is selected on your computer.",
            "Remove and re-add the printer from system settings if it is missing.",
            "If the printer shows a hardware fault, raise a ticket for on-site support.",
        ],
    },
    {
        "id": "KB005",
        "title": "Email or Outlook issues",
        "keywords": ["email", "outlook", "mail", "calendar", "inbox", "send", "receive"],
        "steps": [
            "Close and reopen the email client.",
            "Confirm you are connected to the internet or VPN.",
            "Check that your mailbox is not full.",
            "If mail still will not send or receive, raise a ticket.",
        ],
    },
]

_ticket_counter = 1000
_handoff_counter = 5000


# ---------------------------------------------------------------------------
# Tool 1 — READ (autonomous)
# ---------------------------------------------------------------------------

@tool
def lookup_employee(employee_id: str) -> dict:
    """Look up an employee record by their employee ID.

    Call this first, before helping with any IT issue, to verify who is
    calling. It is read-only and safe to call without caller confirmation.

    Args:
        employee_id: The caller's unique employee ID, e.g. "E1001".
            Matching is case-insensitive and ignores surrounding spaces.

    Returns:
        A dict with:
          found (bool) — True if a record matched.
          name (str | None) — The employee's display name, or None.
          department (str | None) — The employee's department, or None.
          verified (bool) — True if the employee is marked as verified.
          message (str | None) — Error message when found is False.

    On failure (found=False): ask the caller once to repeat their ID.
    If it still does not match, escalate to a human.
    """
    key = employee_id.strip().upper()
    record = _EMPLOYEES.get(key)
    if record is None:
        return {
            "found": False,
            "name": None,
            "department": None,
            "verified": False,
            "message": f"No employee record found for ID '{employee_id}'.",
        }
    return {
        "found": True,
        "name": record["name"],
        "department": record["department"],
        "verified": record["verified"],
        "message": None,
    }


# ---------------------------------------------------------------------------
# Tool 2 — READ (autonomous)
# ---------------------------------------------------------------------------

@tool
def search_kb(issue_description: str) -> dict:
    """Search the IT knowledge base for resolution steps matching the caller's issue.

    Call this after verifying the caller, to find a fix for a routine IT
    problem. It is read-only and safe to call without caller confirmation.
    Pass the caller's issue in their own words; the tool matches against
    known topics: password reset, VPN/remote access, software install,
    printer problems, and email issues.

    Args:
        issue_description: The caller's IT issue in plain language,
            e.g. "I can't connect to the VPN" or "my password isn't working".

    Returns:
        A dict with:
          found (bool) — True if an article matched.
          entry_id (str | None) — KB article ID, e.g. "KB002".
          title (str | None) — Article title.
          steps (list[str] | None) — Ordered resolution steps to read to the caller.
          message (str | None) — Error message when found is False.

    On failure (found=False): do not guess an answer. Escalate to a human.
    """
    text = issue_description.lower()
    best_entry, best_score = None, 0
    for entry in _KB:
        score = sum(1 for kw in entry["keywords"] if kw in text)
        if score > best_score:
            best_score, best_entry = score, entry

    if best_entry is None:
        return {
            "found": False,
            "entry_id": None,
            "title": None,
            "steps": None,
            "message": "No knowledge base article matched the issue described.",
        }
    return {
        "found": True,
        "entry_id": best_entry["id"],
        "title": best_entry["title"],
        "steps": best_entry["steps"],
        "message": None,
    }


# ---------------------------------------------------------------------------
# Tool 3 — WRITE (gated: confirm with caller before calling)
# ---------------------------------------------------------------------------

@tool
def create_ticket(employee_id: str, category: str, description: str) -> dict:
    """Create an IT support ticket for the caller's unresolved issue.

    WRITE tool — only call this after telling the caller you are raising a
    ticket and receiving their spoken agreement. Every call is logged.

    Use this when self-service steps did not resolve the issue, or when the
    issue requires a human to action it (e.g. a software install needing
    manager approval).

    Args:
        employee_id: The verified caller's employee ID, e.g. "E1001".
        category: Short issue category, e.g. "VPN", "password", "software".
        description: A clear one-line summary of the caller's issue.

    Returns:
        A dict with:
          created (bool) — True if the ticket was raised successfully.
          ticket_id (str | None) — Ticket ID to read back to the caller, e.g. "TKT1001".
          status (str | None) — Ticket status, e.g. "open".
          message (str | None) — Error message when created is False.

    On failure (created=False): retry once with a corrected description.
    If it fails again, escalate to a human and tell the caller.
    """
    global _ticket_counter

    if not description.strip():
        return {
            "created": False,
            "ticket_id": None,
            "status": None,
            "message": "Could not create ticket: the issue description was empty.",
        }

    _ticket_counter += 1
    ticket_id = f"TKT{_ticket_counter}"
    print(f"[LOG] create_ticket OK — {ticket_id} for {employee_id} ({category})")
    return {
        "created": True,
        "ticket_id": ticket_id,
        "status": "open",
        "message": None,
    }


# ---------------------------------------------------------------------------
# Tool 4 — WRITE (gated: always log; carry attempt_summary for warm handoff)
# ---------------------------------------------------------------------------

@tool
def escalate(employee_id: str, issue: str, attempt_summary: str, reason: str) -> dict:
    """Hand the call off to a human support agent.

    WRITE tool — the catch-all for every failure path. Call it when:
      - the caller's ID still does not match after one retry;
      - the knowledge base has no article for the issue;
      - creating a ticket has failed twice; or
      - you are not confident you can resolve the issue safely.

    Before calling, tell the caller you are connecting them to a person.
    Pass a clear attempt_summary so the human knows what was already tried
    and the caller does not have to repeat themselves.

    Args:
        employee_id: The caller's employee ID, or "unknown" if unverified.
        issue: A short summary of the caller's issue.
        attempt_summary: What has already been tried, so the human does not repeat it.
        reason: Why the call is being escalated.

    Returns:
        A dict with:
          escalated (bool) — Always True on success.
          handoff_id (str) — Handoff reference ID to read back to the caller.
          message (str) — Confirmation message.
    """
    global _handoff_counter

    _handoff_counter += 1
    handoff_id = f"HND{_handoff_counter}"
    print(
        f"[LOG] escalate OK — {handoff_id} for {employee_id} | "
        f"reason: {reason} | tried: {attempt_summary}"
    )
    return {
        "escalated": True,
        "handoff_id": handoff_id,
        "message": "Call handed off to a human support agent.",
    }
