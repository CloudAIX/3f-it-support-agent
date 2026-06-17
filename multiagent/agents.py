"""Agent node functions for the multi-agent IT support LangGraph.

Each function is one node in the graph. It receives the full SupportState,
does exactly one job (call a tool or call the model), and returns a dict
of only the fields it updates. LangGraph merges those partial updates back
into the shared state automatically.

Three agents call tools only (no LLM): intake_agent, knowledge_agent,
action_agent. One agent calls the model: review_agent (Nebius/Llama). This
is deliberate — use a model only where judgement over unstructured text is
needed; use plain tool calls everywhere else.
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.types import interrupt

from state import SupportState
from tools import create_ticket, escalate, lookup_employee, search_kb

load_dotenv()  # reads .env from this dir or any parent — picks up NEBIUS_API_KEY

# One shared LLM instance. Only review_agent uses it; the other three agents
# call tools with no model round-trip.
llm = ChatOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY"),
    model="meta-llama/Llama-3.3-70B-Instruct",
)

_REVIEW_SYSTEM = (
    "You review IT support interactions. Based on the information provided, "
    "decide three things. Reply with ONLY a JSON object — no other text, no "
    "markdown fences. The JSON must have exactly these keys: "
    '"resolved" (true if the caller\'s issue was fixed, else false), '
    '"correct_path" (true if the agent followed a sensible process — verify '
    "the caller, search the knowledge base, then resolve or escalate; else "
    'false), "followup" (a short plain-English sentence on what should happen '
    'next, or "None" if nothing is needed).'
)


# ---------------------------------------------------------------------------
# Agent 1 — tool only, no LLM (calls lookup_employee)
# ---------------------------------------------------------------------------

def intake_agent(state: SupportState) -> dict:
    """Verify the caller's identity by looking up their employee ID.

    Calls lookup_employee with the employee_id in state. On success writes
    employee_name, department, and verified=True. On failure (not found or
    unverified) writes verified=False and notes the failure in attempts so
    downstream agents and the escalation summary have a clear record.
    """
    result = lookup_employee.invoke({"employee_id": state["employee_id"]})

    if result["found"] and result["verified"]:
        note = f"intake: verified {result['name']} ({result['department']})"
    elif result["found"]:
        note = f"intake: found {result['name']} but marked unverified"
    else:
        note = f"intake: no employee record matched '{state['employee_id']}'"

    return {
        "employee_name": result.get("name"),
        "department": result.get("department"),
        "verified": bool(result["found"] and result["verified"]),
        "attempts": [note],
        "step_count": state["step_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Agent 2 — tool only, no LLM (calls search_kb)
# ---------------------------------------------------------------------------

def knowledge_agent(state: SupportState) -> dict:
    """Search the knowledge base for a fix that matches the caller's issue.

    Calls search_kb with issue_text from state. Writes kb_found and kb_steps.
    If no article matched, records this clearly in attempts so action_agent
    knows to escalate rather than attempt a self-service resolution.
    """
    result = search_kb.invoke({"issue_description": state["issue_text"]})

    if result["found"]:
        note = (
            f"knowledge: matched KB article "
            f"'{result['entry_id']} — {result['title']}'"
        )
    else:
        note = "knowledge: no KB article matched the issue"

    return {
        "kb_found": result["found"],
        "kb_steps": result.get("steps") or [],
        "attempts": [note],
        "step_count": state["step_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Agent 3 — HITL gate + tool calls (no LLM)
# ---------------------------------------------------------------------------

def action_agent(state: SupportState) -> dict:
    """Act on the KB result, with a human approval gate before any write.

    Happy path (kb_found=True): resolution steps are already in state and
    will be delivered to the caller — no write needed, just log and return.

    HITL path (kb_found=False): calls interrupt() to pause the graph and
    surface a ticket proposal to the human operator. The graph is frozen
    here until the caller resumes it with Command(resume="yes"|"no"):
      - "yes" → call create_ticket, write ticket_id to state.
      - "no"  → call escalate with the full attempt_summary, write
                 escalation_id to state.
    The human's decision is recorded in attempts either way.
    """
    if state["kb_found"]:
        note = "action: KB steps available — resolution will be delivered to caller"
        return {
            "attempts": [note],
            "step_count": state["step_count"] + 1,
        }

    # No KB fix found — pause here and ask a human before writing anything.
    # interrupt() checkpoints the current state and hands control back to the
    # caller. When graph.invoke(Command(resume=value), config=same_config) is
    # called, execution resumes from the next line with decision = value.
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
            "ticket_id": result.get("ticket_id"),
            "attempts":  [note],
            "step_count": state["step_count"] + 1,
        }
    else:
        attempt_summary = "; ".join(state.get("attempts") or []) or "no prior steps"
        result = escalate.invoke({
            "employee_id": state.get("employee_id") or "unknown",
            "issue":           state["issue_text"],
            "attempt_summary": attempt_summary,
            "reason": "Human rejected ticket proposal; escalating to support staff.",
        })
        note = (
            f"action: human REJECTED ticket — "
            f"escalated, handoff ID {result['handoff_id']}"
        )
        return {
            "escalation_id": result["handoff_id"],
            "attempts":  [note],
            "step_count": state["step_count"] + 1,
        }


# ---------------------------------------------------------------------------
# Agent 4 — LLM call (Nebius / Llama 3.3 70B — the only model call in graph)
# ---------------------------------------------------------------------------

def review_agent(state: SupportState) -> dict:
    """Produce a structured post-call review using the LLM.

    Sends the issue, the full attempts log, and the outcome to Nebius/Llama
    and asks for a JSON object: {resolved, correct_path, followup}. Parses
    defensively — strips any stray markdown fences before JSON parsing. On
    any failure (API error, malformed JSON, missing keys) fails safe:
    resolved=False, correct_path=False, followup="manual review needed".
    The call never crashes the graph.
    """
    outcome = (
        f"Ticket raised: {state.get('ticket_id')}" if state.get("ticket_id")
        else f"Escalated to human: {state.get('escalation_id')}" if state.get("escalation_id")
        else "Resolved via KB steps" if state.get("kb_found")
        else "Unresolved"
    )

    user_content = (
        f"Issue: {state['issue_text']}\n"
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
            "resolved": bool(data["resolved"]),
            "correct_path": bool(data["correct_path"]),
            "followup": str(data["followup"]),
        }
    except Exception as err:  # fail safe — never let review crash the graph
        review = {
            "resolved": False,
            "correct_path": False,
            "followup": "manual review needed",
            "_error": str(err),
        }

    note = (
        f"review: resolved={review['resolved']} "
        f"correct_path={review['correct_path']}"
    )
    return {
        "review": review,
        "attempts": [note],
        "step_count": state["step_count"] + 1,
    }
