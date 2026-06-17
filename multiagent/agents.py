"""Agent node functions for the multi-agent IT support LangGraph.

Each function is one node in the graph. It receives the full SupportState,
does exactly one job, and returns a dict of only the fields it updates.
LangGraph merges those partial updates back into the shared state.

Three agents call the LLM (Nebius/Llama): intake_agent (greeting),
tone_agent (emotion), and review_agent (post-call review).
Two agents call tools only (no LLM): knowledge_agent, action_agent.
Model calls are kept minimal — use the LLM only where judgement over
unstructured text is needed.
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from memory import get_pending_context, load_history, save_call
from state import SupportState
from tools import create_ticket, escalate, lookup_employee, search_kb

load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY"),
    model="meta-llama/Llama-3.3-70B-Instruct",
)

_GREETING_SYSTEM = (
    "You write a short, warm, spoken-style greeting for an IT support agent "
    "answering an inbound call. The caller's identity is already known — the IVR "
    "passed their employee ID and we've looked them up. "
    "Rules: address the caller by first name only; 1-2 sentences maximum; "
    "natural spoken language, no corporate jargon; never ask for their name or ID. "
    "If there are pending context items (open tickets, recent notifications), "
    "reference the single most relevant one to pre-empt why they may be calling. "
    "Example with context: 'Hi Priya — I can see we raised a ticket about your VPN "
    "yesterday, is that what you're calling about?' "
    "Example without context: 'Hi Tom, great to have you — what can I help you with today?' "
    "Reply with ONLY the greeting text, nothing else."
)

_TONE_SYSTEM = (
    "You detect emotional tone in IT support messages from the caller's WORDS only "
    "(not audio or intonation). Reply with ONLY a JSON object — no other text, no "
    "markdown fences. Keys: "
    '"tone" (exactly one of: "calm", "frustrated", "upset", "urgent") and '
    '"empathy_note" (one short sentence the support agent should say to acknowledge '
    "the caller's feeling before addressing the issue, or empty string if tone is calm). "
    "Be generous with 'frustrated' — any sign of repeated effort, caps, exclamations, "
    "or deadline pressure counts."
)

_REVIEW_SYSTEM = (
    "You review IT support interactions. Based on the information provided, "
    "decide three things. Reply with ONLY a JSON object — no other text, no "
    "markdown fences. Keys: "
    '"resolved" (true if the caller\'s issue was fixed, else false), '
    '"correct_path" (true if the agent followed a sensible process — verify '
    "the caller, classify tone, search the KB, then resolve or escalate appropriately; "
    'else false), "followup" (a short plain-English sentence on what should happen '
    'next, or "None" if nothing is needed).'
)

# Tones that trigger fast-track escalation when no KB fix is found.
# An upset caller must not be subjected to a ticket-approval interrupt —
# connect them to a human directly without making them wait for a gate.
_HIGH_URGENCY_TONES = {"frustrated", "upset", "urgent"}


# ---------------------------------------------------------------------------
# Agent 1 — tool + LLM (lookup_employee, load memory, generate greeting)
# ---------------------------------------------------------------------------

def intake_agent(state: SupportState) -> dict:
    """Verify the caller, load long-term memory, and compose a proactive greeting.

    Three things happen in sequence:

    1. Tool call — lookup_employee: verifies identity, writes name/department/verified.

    2. Memory load — loads call history and pending context (open tickets,
       notifications) from memory.py for this employee.

    3. LLM call — uses Nebius/Llama to compose a 1-2 sentence spoken greeting.
       If pending context exists, the greeting pre-empts the likely reason for
       calling ("I see we raised a ticket about your VPN yesterday — is that
       what you're calling about?"). If no context, it gives a warm friendly
       opening. Fails safe to a generic greeting on any LLM error.
    """
    # --- Step 1: verify identity -------------------------------------------
    result = lookup_employee.invoke({"employee_id": state["employee_id"]})

    if result["found"] and result["verified"]:
        note = f"intake: verified {result['name']} ({result['department']})"
    elif result["found"]:
        note = f"intake: found {result['name']} but marked unverified"
    else:
        note = f"intake: no employee record matched '{state['employee_id']}'"

    name = result.get("name") or "there"
    first_name = name.split()[0] if name and name != "there" else name

    # --- Step 2: load long-term memory -------------------------------------
    history = load_history(state["employee_id"])
    pending = get_pending_context(state["employee_id"])

    if history:
        last = history[-1]
        note_mem = (
            f"intake: returning caller — {len(history)} previous contact(s), "
            f"last issue: '{last['issue']}' (outcome: {last['outcome']})"
        )
    else:
        note_mem = "intake: first-time caller — no prior history"

    note_pending = (
        f"intake: {len(pending)} pending context item(s) loaded"
        if pending else "intake: no pending context"
    )

    # --- Step 3: compose proactive greeting via LLM ------------------------
    # Build the user message: name + pending context + abbreviated history.
    pending_text = (
        json.dumps(pending, indent=2) if pending else "None"
    )
    last_call_text = (
        f"issue='{history[-1]['issue']}', outcome='{history[-1]['outcome']}'"
        if history else "None"
    )
    user_msg = (
        f"Caller first name: {first_name}\n"
        f"Pending context items: {pending_text}\n"
        f"Last call on record: {last_call_text}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_GREETING_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        greeting = (response.content or "").strip().strip('"')
    except Exception:
        greeting = f"Hi {first_name}, thanks for calling — what can I help you with today?"

    note_greeting = f"intake: greeting generated ({'with pending context' if pending else 'no pending context'})"

    return {
        "employee_name":  result.get("name"),
        "department":     result.get("department"),
        "verified":       bool(result["found"] and result["verified"]),
        "past_history":   history,
        "pending_context": pending,
        "greeting":       greeting,
        "attempts":       [note, note_mem, note_pending, note_greeting],
        "step_count":     state["step_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Agent 2 — LLM call (Nebius/Llama — tone classification from text)
# ---------------------------------------------------------------------------

def tone_agent(state: SupportState) -> dict:
    """Classify the caller's emotional tone from their issue_text using the LLM.

    Analyses the caller's WORDS only (text-based — audio/intonation is a
    separate voice-layer concern). Writes emotional_tone and empathy_note to
    state. The tone drives two downstream behaviours:
      1. empathy_note is available for the agent to open with before the fix.
      2. frustrated/upset/urgent + no KB fix → action_agent fast-tracks to
         human escalation instead of surfacing a ticket-approval interrupt.

    Fails safe to tone="calm", empathy_note="" on any error so a detection
    failure never blocks the rest of the call.
    """
    try:
        response = llm.invoke([
            SystemMessage(content=_TONE_SYSTEM),
            HumanMessage(content=f"Caller message: {state['issue_text']}"),
        ])
        raw = response.content or ""
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        tone = str(data.get("tone", "calm")).lower()
        if tone not in ("calm", "frustrated", "upset", "urgent"):
            tone = "calm"
        empathy = str(data.get("empathy_note", ""))
    except Exception:  # fail safe — never block the call on tone failure
        tone = "calm"
        empathy = ""

    note = f"tone: detected '{tone}'" + (" — empathy note ready" if empathy else "")
    return {
        "emotional_tone": tone,
        "empathy_note":   empathy,
        "attempts":       [note],
        "step_count":     state["step_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Agent 3 — tool only, no LLM (calls search_kb)
# ---------------------------------------------------------------------------

def knowledge_agent(state: SupportState) -> dict:
    """Search the knowledge base for a fix that matches the caller's issue.

    Calls search_kb with issue_text. Writes kb_found and kb_steps. A no-match
    is recorded in attempts so action_agent knows to escalate rather than
    attempt a self-service resolution.
    """
    result = search_kb.invoke({"issue_description": state["issue_text"]})

    note = (
        f"knowledge: matched KB article '{result['entry_id']} — {result['title']}'"
        if result["found"]
        else "knowledge: no KB article matched the issue"
    )
    return {
        "kb_found": result["found"],
        "kb_steps": result.get("steps") or [],
        "attempts": [note],
        "step_count": state["step_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Agent 4 — HITL gate + tool calls, tone-aware routing (no LLM)
# ---------------------------------------------------------------------------

def action_agent(state: SupportState) -> dict:
    """Act on the KB result, with tone-aware routing before any write action.

    Happy path (kb_found=True):
      Resolution steps are in state — no write needed regardless of tone.

    Upset caller fast-track (kb_found=False AND tone is high-urgency):
      Skip the ticket-approval interrupt entirely. An upset caller must not
      be made to wait for a HITL gate — call escalate directly and hand off
      to a human immediately. The attempt_summary ensures the human agent
      knows the full context.

    Calm caller HITL gate (kb_found=False AND tone is calm):
      Use interrupt() to propose a ticket and wait for human approval.
        - "yes" → create_ticket, write ticket_id.
        - "no"  → escalate, write escalation_id.
    """
    if state["kb_found"]:
        note = "action: KB steps available — resolution will be delivered to caller"
        return {
            "attempts":   [note],
            "step_count": state["step_count"] + 1,
        }

    tone = state.get("emotional_tone", "calm")
    attempt_summary = "; ".join(state.get("attempts") or []) or "no prior steps"

    if tone in _HIGH_URGENCY_TONES:
        # Fast-track: upset caller + no KB fix → straight to human, no gate.
        # Rationale: interrupt() would pause the graph and add latency; an
        # upset caller with a deadline should not experience that wait.
        result = escalate.invoke({
            "employee_id":     state.get("employee_id") or "unknown",
            "issue":           state["issue_text"],
            "attempt_summary": attempt_summary,
            "reason": (
                f"Caller tone detected as '{tone}' and no KB fix exists. "
                "Fast-tracked to human — skipped ticket-approval gate."
            ),
        })
        note = (
            f"action: tone='{tone}' + no KB fix → "
            f"fast-tracked to human, handoff ID {result['handoff_id']}"
        )
        return {
            "escalation_id": result["handoff_id"],
            "attempts":      [note],
            "step_count":    state["step_count"] + 1,
        }

    # Calm caller: pause and ask for ticket approval before writing anything.
    decision = interrupt(
        f"No KB fix found for: '{state['issue_text']}'. "
        f"Propose raising a ticket for employee "
        f"{state.get('employee_id', 'unknown')}, category 'general'. "
        f"Approve? (yes/no)"
    )

    if decision.strip().lower() == "yes":
        result = create_ticket.invoke({
            "employee_id": state.get("employee_id") or "unknown",
            "category":    "general",
            "description": state["issue_text"],
        })
        note = (
            f"action: human APPROVED — ticket {result.get('ticket_id')} "
            f"raised (status: {result.get('status')})"
        )
        return {
            "ticket_id":  result.get("ticket_id"),
            "attempts":   [note],
            "step_count": state["step_count"] + 1,
        }
    else:
        result = escalate.invoke({
            "employee_id":     state.get("employee_id") or "unknown",
            "issue":           state["issue_text"],
            "attempt_summary": attempt_summary,
            "reason":          "Human rejected ticket proposal; escalating to support staff.",
        })
        note = (
            f"action: human REJECTED ticket — "
            f"escalated, handoff ID {result['handoff_id']}"
        )
        return {
            "escalation_id": result["handoff_id"],
            "attempts":      [note],
            "step_count":    state["step_count"] + 1,
        }


# ---------------------------------------------------------------------------
# Agent 5 — LLM call (Nebius/Llama — structured post-call review + memory save)
# ---------------------------------------------------------------------------

def review_agent(state: SupportState) -> dict:
    """Produce a structured post-call review and persist the call to memory.

    Sends the issue, tone, attempts log, and outcome to Nebius/Llama and
    requests a JSON review: {resolved, correct_path, followup}. Parses
    defensively. Then calls save_call() so the next call by this employee
    sees this one in their warm-start history.
    """
    outcome = (
        f"Ticket raised: {state.get('ticket_id')}" if state.get("ticket_id")
        else f"Escalated to human: {state.get('escalation_id')}" if state.get("escalation_id")
        else "Resolved via KB steps" if state.get("kb_found")
        else "Unresolved"
    )

    user_content = (
        f"Issue: {state['issue_text']}\n"
        f"Caller tone: {state.get('emotional_tone', 'unknown')}\n"
        f"Employee verified: {state.get('verified', False)}\n"
        f"Steps tried: {'; '.join(state.get('attempts') or [])}\n"
        f"Outcome: {outcome}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_REVIEW_SYSTEM),
            HumanMessage(content=user_content),
        ])
        raw = response.content or ""
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        review = {
            "resolved":     bool(data["resolved"]),
            "correct_path": bool(data["correct_path"]),
            "followup":     str(data["followup"]),
        }
    except Exception as err:
        review = {
            "resolved":     False,
            "correct_path": False,
            "followup":     "manual review needed",
            "_error":       str(err),
        }

    save_call(
        state.get("employee_id") or "unknown",
        {"issue": state["issue_text"], "outcome": outcome},
    )

    note = f"review: resolved={review['resolved']} correct_path={review['correct_path']}"
    return {
        "review":     review,
        "attempts":   [note],
        "step_count": state["step_count"] + 1,
    }
