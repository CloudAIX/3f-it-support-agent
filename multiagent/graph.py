"""LangGraph graph for the multi-agent IT support system.

Full pipeline with tone-aware routing:

  Flow (happy path, calm caller):
    START → intake → tone → knowledge → action → review → END

  Flow (unverified caller):
    START → intake → tone → action (escalates) → review → END

  Flow (upset/frustrated/urgent caller, no KB match):
    START → intake → tone → knowledge → action (fast-track escalate) → review → END

  Flow (calm caller, no KB match):
    START → intake → tone → knowledge → action (HITL interrupt) → review → END
"""

import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents import (
    action_agent,
    intake_agent,
    knowledge_agent,
    review_agent,
    tone_agent,
)
from state import SupportState

MAX_STEPS = 10  # raised from 8 to accommodate the extra tone node


# ---------------------------------------------------------------------------
# Router — runs after tone_agent; reads verified + step_count
# ---------------------------------------------------------------------------

def route_after_tone(state: SupportState) -> str:
    """Decide what runs after tone_agent.

    Normal routing:
      verified=True  → knowledge (search KB, then action, then review)
      verified=False → action   (escalate immediately, then review)

    Guardrail: if step_count has reached MAX_STEPS, skip straight to review.
    This cap prevents runaway loops regardless of graph state.
    """
    if state["step_count"] >= MAX_STEPS:  # runaway-loop guardrail
        return "review"
    return "knowledge" if state["verified"] else "action"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(SupportState)

builder.add_node("intake",    intake_agent)
builder.add_node("tone",      tone_agent)      # NEW — classifies emotional tone
builder.add_node("knowledge", knowledge_agent)
builder.add_node("action",    action_agent)
builder.add_node("review",    review_agent)

# Fixed edges
builder.add_edge(START,       "intake")
builder.add_edge("intake",    "tone")          # tone always runs after intake
builder.add_edge("knowledge", "action")
builder.add_edge("action",    "review")
builder.add_edge("review",    END)

# Conditional edge: tone → (knowledge | action | review)
builder.add_conditional_edges(
    "tone",
    route_after_tone,
    {
        "knowledge": "knowledge",
        "action":    "action",
        "review":    "review",   # only reached when MAX_STEPS is exceeded
    },
)

graph = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def _print_final(label: str, final: dict) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {label}")
    print(f"{'─' * 62}")
    print(f"  Employee       : {final.get('employee_name')} ({final.get('department')})")
    print(f"  Verified       : {final.get('verified')}")
    tone = final.get("emotional_tone", "—")
    print(f"  Detected tone  : {tone}")
    empathy = final.get("empathy_note", "")
    if empathy:
        print(f"  Empathy note   : \"{empathy}\"")
    print(f"  KB found       : {final.get('kb_found')}")
    print(f"  Ticket ID      : {final.get('ticket_id') or '(none)'}")
    print(f"  Escalation ID  : {final.get('escalation_id') or '(none)'}")
    print(f"  Steps taken    : {final.get('step_count')}")
    print(f"\n  Attempts log:")
    for i, entry in enumerate(final.get("attempts") or [], 1):
        print(f"    {i}. {entry}")
    review = final.get("review")
    if review:
        print(f"\n  Post-call review:")
        print(f"    " + json.dumps(review, indent=4).replace("\n", "\n    "))


# ---------------------------------------------------------------------------
# Demo: Run A (calm caller) vs Run B (upset caller, different path)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store_path = Path(__file__).parent / "memory_store.json"
    if store_path.exists():
        store_path.unlink()
        print("(cleared memory_store.json for a clean demo)\n")

    _BLANK: SupportState = {
        "issue_text":     "",
        "employee_id":    "",
        "employee_name":  None,
        "department":     None,
        "verified":       False,
        "emotional_tone": "",
        "empathy_note":   "",
        "kb_found":       False,
        "kb_steps":       [],
        "ticket_id":      None,
        "escalation_id":  None,
        "attempts":       [],
        "confidence":     1.0,
        "step_count":     0,
        "past_history":   [],
        "review":         None,
    }

    # -----------------------------------------------------------------------
    # RUN A — calm caller, routine VPN issue.
    # Expected: tone="calm", KB match → resolved via steps, no escalation.
    # -----------------------------------------------------------------------
    print("=" * 62)
    print("  RUN A — Calm caller, routine issue")
    print("=" * 62)
    issue_a = "I can't connect to the VPN today. Can you help?"
    print(f"  Issue : \"{issue_a}\"")

    config_a = {"configurable": {"thread_id": "run-a"}}
    final_a = graph.invoke(
        {**_BLANK, "employee_id": "E1001", "issue_text": issue_a},
        config=config_a,
    )
    _print_final("RUN A result — calm path", final_a)

    # -----------------------------------------------------------------------
    # RUN B — upset caller, unrecognised issue (no KB match).
    # Expected: tone="frustrated"/"upset"/"urgent", no KB match →
    # action_agent fast-tracks straight to escalation WITHOUT pausing for
    # a ticket-approval interrupt. Empathy note is set for the agent.
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 62)
    print("  RUN B — Upset caller, unrecognised issue")
    print("=" * 62)
    issue_b = (
        "This is the THIRD time I've called!! My entire computer is dead — "
        "screen, keyboard, everything — and I have a board presentation in "
        "30 minutes. Nobody has helped me and I am absolutely FURIOUS. "
        "Fix this NOW."
    )
    print(f"  Issue : \"{issue_b}\"")

    config_b = {"configurable": {"thread_id": "run-b"}}
    final_b = graph.invoke(
        {**_BLANK, "employee_id": "E1002", "issue_text": issue_b},
        config=config_b,
    )
    _print_final("RUN B result — fast-track escalation path", final_b)

    # Summary: show side-by-side what changed
    print("\n\n" + "=" * 62)
    print("  ROUTING COMPARISON")
    print("=" * 62)
    print(f"  {'':30s}  {'RUN A':>12}  {'RUN B':>12}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*12}")
    print(f"  {'Detected tone':<30}  {final_a.get('emotional_tone','—'):>12}  {final_b.get('emotional_tone','—'):>12}")
    print(f"  {'KB match':<30}  {str(final_a.get('kb_found')):>12}  {str(final_b.get('kb_found')):>12}")
    print(f"  {'Ticket raised':<30}  {str(bool(final_a.get('ticket_id'))):>12}  {str(bool(final_b.get('ticket_id'))):>12}")
    print(f"  {'Escalated':<30}  {str(bool(final_a.get('escalation_id'))):>12}  {str(bool(final_b.get('escalation_id'))):>12}")
    print(f"  {'HITL interrupt triggered':<30}  {'No':>12}  {'No (bypassed)':>12}")
    print(f"  {'Empathy note set':<30}  {str(bool(final_a.get('empathy_note'))):>12}  {str(bool(final_b.get('empathy_note'))):>12}")
    print()
