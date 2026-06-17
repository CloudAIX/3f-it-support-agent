"""3F — Multi-Agent IT Support Voice Agent (ITSM): backend tools.

This FastAPI app exposes the tools the ElevenLabs Conversational AI agent
calls via server-tools / webhooks. Each endpoint is one tool.

Tool 1 of 4: lookup_employee (read, autonomous).
Tool 2 of 4: search_kb (read, autonomous).
Tool 3 of 4: create_ticket (write, gated — confirm with caller first, then log).
Tool 4 of 4: escalate (write, gated — the catch-all for every failure path; log the handoff).

Plus post_call_review: takes a call transcript and returns a structured review
{resolved, correct_path, followup}. This is the one endpoint that calls a model
(Nebius Token Factory / Llama 3.3 70B) — it satisfies the brief's "at least one
Nebius model call" requirement.
"""

import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("3f")

# Nebius Token Factory — OpenAI-compatible client. Same setup as Week 2.
# Key and model name come from .env, never hardcoded. The client is created
# lazily (only when post_call_review runs) so the other four tools still work
# even if the Nebius key is not set.
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
_nebius_client: OpenAI | None = None


def _get_nebius() -> OpenAI:
    """Create the Nebius client on first use. Raises if the key is missing."""
    global _nebius_client
    if _nebius_client is None:
        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY is not set. Add it to your .env file.")
        _nebius_client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=key,
        )
    return _nebius_client

app = FastAPI(
    title="3F IT Support Voice Agent — Backend Tools",
    description="Stubbed tool endpoints called by the ElevenLabs voice agent.",
    version="0.1.0",
)


# --- Stub datastore -------------------------------------------------------
# Fake data only. Swapping this dict for a real DB changes nothing in the
# agent machinery — the tool contract (args in, shape out) stays identical.
_EMPLOYEE_RECORDS = {
    "E1001": {"name": "Priya Nair",      "department": "Finance",        "verified": True},
    "E1002": {"name": "Tom Becker",      "department": "Engineering",    "verified": True},
    "E1003": {"name": "Sara Lindqvist",  "department": "Sales",          "verified": False},
    "A10":   {"name": "Nathan",          "department": "Innovation Lab", "verified": True},
}


# Knowledge base stub. Each entry has trigger keywords and resolution steps.
# A real build would swap this for a document store; the tool contract is
# unchanged. Deliberately NOT a vector DB / RAG — the KB is too small to
# justify it (MINT principle), so matching is simple keyword overlap.
_KB_ENTRIES = [
    {
        "id": "KB001",
        "title": "Password reset",
        "keywords": ["password", "reset", "locked out", "log in", "login", "signin", "sign in"],
        "steps": [
            "Go to the self-service portal at portal.example.com.",
            "Select 'Forgot password' and enter your work email.",
            "Follow the emailed link and set a new password.",
            "If no email arrives within five minutes, the account may be locked — escalate.",
        ],
    },
    {
        "id": "KB002",
        "title": "VPN or remote access",
        "keywords": ["vpn", "remote", "access", "connect", "network", "off site", "offsite", "work from home"],
        "steps": [
            "Confirm the VPN client is installed and up to date.",
            "Sign in with your work email and password.",
            "If sign-in fails, restart the client and try once more.",
            "If it still fails, the account may not be enabled for remote access — escalate.",
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
            "If the app is not in the catalogue, it needs manager approval — raise a ticket.",
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


# Ticket store stub. Created tickets are kept in memory and logged. A real
# build would write to an ITSM system (e.g. ServiceNow, Jira); the tool
# contract is unchanged. _ticket_counter gives each ticket a readable ID.
_TICKETS: dict[str, dict] = {}
_ticket_counter = 1000
_handoff_counter = 5000


# --- Schemas --------------------------------------------------------------
class LookupEmployeeRequest(BaseModel):
    employee_id: str = Field(
        ...,
        description="The employee's unique ID, e.g. 'E1001'. Case-insensitive.",
        examples=["E1001"],
    )


class EmployeeRecord(BaseModel):
    found: bool
    employee_id: str | None = None
    name: str | None = None
    department: str | None = None
    verification_status: str | None = None  # "verified" or "unverified"
    message: str | None = None


class SearchKbRequest(BaseModel):
    issue_description: str = Field(
        ...,
        description="The caller's IT issue in their own words, e.g. 'I can't connect to the VPN'.",
        examples=["I can't connect to the VPN"],
    )


class KbResult(BaseModel):
    found: bool
    entry_id: str | None = None
    title: str | None = None
    steps: list[str] | None = None
    message: str | None = None


class CreateTicketRequest(BaseModel):
    employee_id: str = Field(
        ...,
        description="The verified caller's employee ID, e.g. 'E1001'.",
        examples=["E1001"],
    )
    category: str = Field(
        ...,
        description="Short issue category, e.g. 'VPN', 'password', 'software'.",
        examples=["VPN"],
    )
    description: str = Field(
        ...,
        description="A clear one-line summary of the caller's issue.",
        examples=["Cannot connect to VPN after password change."],
    )


class TicketResult(BaseModel):
    created: bool
    ticket_id: str | None = None
    status: str | None = None  # e.g. "open"
    message: str | None = None


class EscalateRequest(BaseModel):
    employee_id: str = Field(
        ...,
        description="The caller's employee ID, or 'unknown' if not verified.",
        examples=["E1001"],
    )
    issue: str = Field(
        ...,
        description="A short summary of the caller's issue.",
        examples=["Cannot connect to VPN; password reset did not help."],
    )
    attempt_summary: str = Field(
        ...,
        description="What the agent already tried, so the human does not repeat it.",
        examples=["Verified caller, searched KB, walked through VPN steps — still failing."],
    )
    reason: str = Field(
        ...,
        description="Why the call is being escalated.",
        examples=["KB steps did not resolve the issue."],
    )


class EscalationResult(BaseModel):
    escalated: bool
    handoff_id: str | None = None
    message: str | None = None


class PostCallReviewRequest(BaseModel):
    transcript: str = Field(
        ...,
        description="The full text of the call between the agent and the caller.",
        examples=["Agent: Hello, can I have your employee ID? Caller: E1001..."],
    )


class PostCallReview(BaseModel):
    resolved: bool
    correct_path: bool
    followup: str
    raw_model_output: str | None = None  # kept for the write-up / debugging


# --- Tool 1: lookup_employee ---------------------------------------------
@app.post("/lookup_employee", response_model=EmployeeRecord)
def lookup_employee(request: LookupEmployeeRequest) -> EmployeeRecord:
    """Look up an employee record by their employee ID.

    Use this tool first, to verify who is calling before helping with any
    IT issue. It is read-only and safe to call without confirmation.

    Args:
        employee_id (str): The caller's unique employee ID, for example
            "E1001". Matching is case-insensitive and ignores surrounding
            spaces. This is the only input.

    Returns:
        An object describing the lookup result.
        On success (found = true): employee_id, name, department, and
        verification_status, which is "verified" or "unverified".
        On failure (found = false): a short message saying no record
        matched. When this happens, ask the caller once to repeat their
        employee ID. If it still does not match, escalate to a human.
    """
    key = request.employee_id.strip().upper()
    record = _EMPLOYEE_RECORDS.get(key)

    if record is None:
        return EmployeeRecord(
            found=False,
            message=f"No employee record found for ID '{request.employee_id}'.",
        )

    return EmployeeRecord(
        found=True,
        employee_id=key,
        name=record["name"],
        department=record["department"],
        verification_status="verified" if record["verified"] else "unverified",
    )


# --- Tool 2: search_kb ----------------------------------------------------
@app.post("/search_kb", response_model=KbResult)
def search_kb(request: SearchKbRequest) -> KbResult:
    """Search the IT support knowledge base for a fix to the caller's issue.

    Use this tool after the caller is verified, to find resolution steps for
    a routine IT problem. It is read-only and safe to call without
    confirmation. Pass the caller's issue in their own words; the tool
    matches it against known topics such as password reset, VPN or remote
    access, software install, printer problems, and email issues.

    Args:
        issue_description (str): The caller's IT issue described in plain
            language, for example "I can't connect to the VPN" or "my
            password isn't working". This is the only input.

    Returns:
        An object describing the search result.
        On a match (found = true): entry_id, title, and steps, which is an
        ordered list of resolution steps to read to the caller.
        On no match (found = false): a short message saying nothing matched.
        When this happens, do not guess an answer. Escalate to a human.
    """
    text = request.issue_description.lower()

    best_entry = None
    best_score = 0
    for entry in _KB_ENTRIES:
        score = sum(1 for kw in entry["keywords"] if kw in text)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None:
        return KbResult(
            found=False,
            message="No knowledge base article matched the issue described.",
        )

    return KbResult(
        found=True,
        entry_id=best_entry["id"],
        title=best_entry["title"],
        steps=best_entry["steps"],
    )


# --- Tool 3: create_ticket ------------------------------------------------
@app.post("/create_ticket", response_model=TicketResult)
def create_ticket(request: CreateTicketRequest) -> TicketResult:
    """Create an IT support ticket for the caller's issue.

    This tool WRITES data, so it is gated. Only call it AFTER you have told
    the caller you are about to raise a ticket and they have agreed. Do not
    raise a ticket without the caller's spoken confirmation. Every ticket is
    logged when created.

    Use this when self-service steps did not resolve the issue, or when the
    issue needs a human to action it later (for example a software install
    that needs approval).

    Args:
        employee_id (str): The verified caller's employee ID, e.g. "E1001".
        category (str): A short issue category, e.g. "VPN", "password",
            "software".
        description (str): A clear one-line summary of the caller's issue.

    Returns:
        An object describing the result.
        On success (created = true): ticket_id and status (e.g. "open").
        Read the ticket_id back to the caller.
        On failure (created = false): a short message. When this happens,
        try once more. If it still fails, escalate to a human and tell the
        caller.
    """
    global _ticket_counter

    # Stub failure path: a blank description means we cannot raise a useful
    # ticket. A real system would have its own validation. This gives the
    # agent a failure branch to handle (retry once, then escalate).
    if not request.description.strip():
        logger.info("create_ticket FAILED — empty description for %s", request.employee_id)
        return TicketResult(
            created=False,
            message="Could not create ticket: the issue description was empty.",
        )

    _ticket_counter += 1
    ticket_id = f"TKT{_ticket_counter}"
    ticket = {
        "ticket_id": ticket_id,
        "employee_id": request.employee_id,
        "category": request.category,
        "description": request.description,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _TICKETS[ticket_id] = ticket
    logger.info("create_ticket OK — %s for %s (%s)", ticket_id, request.employee_id, request.category)

    return TicketResult(created=True, ticket_id=ticket_id, status="open")


# --- Tool 4: escalate -----------------------------------------------------
@app.post("/escalate", response_model=EscalationResult)
def escalate(request: EscalateRequest) -> EscalationResult:
    """Hand the call off to a human support agent.

    This is the catch-all for every failure path. Call it when:
      - the caller's employee ID still does not match after one retry;
      - the knowledge base has no article for the issue;
      - creating a ticket failed twice; or
      - you are not confident you can resolve the issue safely.

    This tool WRITES a handoff, so it is gated and always logged. Before
    calling it, tell the caller you are connecting them to a person. Pass a
    clear attempt_summary so the human knows what was already tried and the
    caller does not have to repeat themselves.

    Args:
        employee_id (str): The caller's employee ID, or "unknown" if they
            could not be verified.
        issue (str): A short summary of the caller's issue.
        attempt_summary (str): What you already tried, so the human does not
            repeat it.
        reason (str): Why the call is being escalated.

    Returns:
        An object describing the handoff.
        On success (escalated = true): a handoff_id. Read it back to the
        caller and tell them a person will take over.
    """
    global _handoff_counter

    _handoff_counter += 1
    handoff_id = f"HND{_handoff_counter}"
    logger.info(
        "escalate OK — %s for %s | reason: %s | tried: %s",
        handoff_id,
        request.employee_id,
        request.reason,
        request.attempt_summary,
    )

    return EscalationResult(
        escalated=True,
        handoff_id=handoff_id,
        message="Call handed off to a human support agent.",
    )


# --- Post-call review (calls Nebius / Llama) ------------------------------
_REVIEW_SYSTEM = (
    "You review IT support call transcripts. Read the transcript and decide "
    "three things. Reply with ONLY a JSON object, no other text, no markdown "
    "fences. The JSON must have exactly these keys: "
    '"resolved" (true if the caller\'s issue was fixed on the call, else false), '
    '"correct_path" (true if the agent followed a sensible process: verify the '
    'caller, search the knowledge base, then create a ticket or escalate when '
    'needed; else false), and '
    '"followup" (a short plain-English sentence saying what should happen next, '
    'or "None" if nothing is needed).'
)


@app.post("/post_call_review", response_model=PostCallReview)
def post_call_review(request: PostCallReviewRequest) -> PostCallReview:
    """Review a finished call transcript and return a structured summary.

    Call this once at the end of a call. It sends the transcript to a model
    (Nebius Token Factory) and returns a structured review used for quality
    monitoring. It does not change anything the caller sees.

    Args:
        transcript (str): The full text of the call between the agent and
            the caller.

    Returns:
        An object with three fields:
        resolved (bool) — was the caller's issue fixed on the call.
        correct_path (bool) — did the agent follow a sensible process.
        followup (str) — a short note on what should happen next, or "None".
    """
    try:
        client = _get_nebius()
        resp = client.chat.completions.create(
            model=NEBIUS_MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": _REVIEW_SYSTEM},
                {"role": "user", "content": f"TRANSCRIPT:\n{request.transcript}"},
            ],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as err:  # noqa: BLE001 — surface any API/config error cleanly
        logger.info("post_call_review API FAILED — %s", err)
        return PostCallReview(
            resolved=False,
            correct_path=False,
            followup="Automated review unavailable. Flag for manual review.",
            raw_model_output=str(err),
        )

    # Parse defensively: strip any stray markdown fences, then load JSON.
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        review = PostCallReview(
            resolved=bool(data["resolved"]),
            correct_path=bool(data["correct_path"]),
            followup=str(data["followup"]),
            raw_model_output=raw,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        # If the model returns something we can't parse, fail safe: flag for
        # human review rather than guessing. This is itself a failure path.
        logger.info("post_call_review PARSE FAILED — %s | raw: %s", err, raw)
        review = PostCallReview(
            resolved=False,
            correct_path=False,
            followup="Could not parse the automated review. Flag for manual review.",
            raw_model_output=raw,
        )

    logger.info(
        "post_call_review — resolved=%s correct_path=%s",
        review.resolved,
        review.correct_path,
    )
    return review


@app.get("/health")
def health() -> dict:
    """Simple health check. Returns {"status": "ok"} when the app is running."""
    return {"status": "ok"}