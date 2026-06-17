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
    greeting = final.get("greeting", "")
    if greeting:
        print(f"\n  *** GREETING ***")
        print(f"  \"{greeting}\"")
        print()
    pending = final.get("pending_context") or []
    print(f"  Pending items  : {len(pending)} loaded")
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
# Demo: proactive greeting (E1001 with pending context vs E1002 without)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from memory import set_pending_context

    store_path = Path(__file__).parent / "memory_store.json"
    if store_path.exists():
        store_path.unlink()
        print("(cleared memory_store.json for a clean demo)\n")

    # Pre-seed E1001's pending context — simulates what the ticketing system
    # and IVR platform would have written before the call was answered.
    set_pending_context("E1001", [
        {
            "type":      "open_ticket",
            "ticket_id": "TKT1001",
            "issue":     "VPN not working after password change",
            "raised":    "yesterday",
            "status":    "open",
        },
        {
            "type":    "notification",
            "message": "VPN maintenance window completed at 08:00 this morning",
            "sent":    "this morning",
        },
    ])
    print("(pre-seeded E1001 pending context: open VPN ticket + maintenance notification)\n")

    _BLANK: SupportState = {
        "issue_text":      "",
        "employee_id":     "",
        "employee_name":   None,
        "department":      None,
        "verified":        False,
        "emotional_tone":  "",
        "empathy_note":    "",
        "kb_found":        False,
        "kb_steps":        [],
        "ticket_id":       None,
        "escalation_id":   None,
        "attempts":        [],
        "confidence":      1.0,
        "step_count":      0,
        "past_history":    [],
        "pending_context": [],
        "greeting":        "",
        "review":          None,
    }

    # -----------------------------------------------------------------------
    # RUN A — E1001, has pending context (open VPN ticket + notification).
    # Expected greeting: references TKT1001 / VPN ticket proactively.
    # -----------------------------------------------------------------------
    print("=" * 62)
    print("  RUN A — E1001 (pending context: open VPN ticket)")
    print("=" * 62)
    issue_a = "Yes, calling about my VPN — still not working after the maintenance"
    print(f"  Caller says: \"{issue_a}\"")
    print("  (greeting is generated BEFORE caller states their issue)\n")

    config_a = {"configurable": {"thread_id": "greet-a"}}
    final_a = graph.invoke(
        {**_BLANK, "employee_id": "E1001", "issue_text": issue_a},
        config=config_a,
    )
    _print_final("RUN A — proactive greeting with pending context", final_a)

    # -----------------------------------------------------------------------
    # RUN B — E1002, no pending context, first-time caller.
    # Expected greeting: warm generic opening, no pre-emption.
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 62)
    print("  RUN B — E1002 (no pending context, first-time caller)")
    print("=" * 62)
    issue_b = "I can't print anything — the printer just shows an error"
    print(f"  Caller says: \"{issue_b}\"")
    print("  (greeting is generated BEFORE caller states their issue)\n")

    config_b = {"configurable": {"thread_id": "greet-b"}}
    final_b = graph.invoke(
        {**_BLANK, "employee_id": "E1002", "issue_text": issue_b},
        config=config_b,
    )
    _print_final("RUN B — generic greeting, no pending context", final_b)

    # Side-by-side comparison
    print("\n\n" + "=" * 62)
    print("  GREETING COMPARISON")
    print("=" * 62)
    print(f"  RUN A greeting  : \"{final_a.get('greeting', '')}\"")
    print(f"  RUN B greeting  : \"{final_b.get('greeting', '')}\"")
    print(f"\n  Pending items   : {len(final_a.get('pending_context') or [])} (A) vs "
          f"{len(final_b.get('pending_context') or [])} (B)")
    print(f"  Prior history   : {len(final_a.get('past_history') or [])} (A) vs "
          f"{len(final_b.get('past_history') or [])} (B)")
    print()
