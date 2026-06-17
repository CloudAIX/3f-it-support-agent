"""LangGraph graph for the multi-agent IT support system.

Wires intake → knowledge/action → action → review into a compiled graph.
One conditional branch (post-intake) routes based on caller verification.
A max-steps guardrail prevents runaway loops if the graph ever gets stuck.

Flow (happy path):
  START → intake → knowledge → action → review → END

Flow (unverified caller):
  START → intake → action (escalates) → review → END
"""

import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents import action_agent, intake_agent, knowledge_agent, review_agent
from state import SupportState

# Runaway-loop guardrail. Each agent increments step_count by 1, so with
# four agents the normal maximum is 4. Setting the cap at 8 gives ample
# headroom for future nodes while still catching any accidental cycle.
MAX_STEPS = 8


# ---------------------------------------------------------------------------
# Router — the only conditional branch in the graph
# ---------------------------------------------------------------------------

def route_after_intake(state: SupportState) -> str:
    """Decide what runs after intake_agent.

    Normal routing:
      verified=True  → knowledge (search KB, then action, then review)
      verified=False → action   (escalate immediately, then review)

    Guardrail: if step_count has already reached MAX_STEPS, skip straight to
    review regardless of verification status. This is the runaway-loop cap —
    it ensures a misconfigured or looping graph cannot spin indefinitely.
    """
    if state["step_count"] >= MAX_STEPS:  # runaway-loop guardrail
        return "review"
    return "knowledge" if state["verified"] else "action"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(SupportState)

# Nodes — one per agent function
builder.add_node("intake",    intake_agent)
builder.add_node("knowledge", knowledge_agent)
builder.add_node("action",    action_agent)
builder.add_node("review",    review_agent)

# Fixed edges
builder.add_edge(START,       "intake")
builder.add_edge("knowledge", "action")
builder.add_edge("action",    "review")
builder.add_edge("review",    END)

# Conditional edge: intake → (knowledge | action | review)
# path_map makes the routing explicit — each string the router returns maps
# to a named node, so renaming a node won't silently break the branch.
builder.add_conditional_edges(
    "intake",
    route_after_intake,
    {
        "knowledge": "knowledge",
        "action":    "action",
        "review":    "review",   # only reached when MAX_STEPS is exceeded
    },
)

# MemorySaver is required for interrupt/resume — it checkpoints state between
# the first invoke (which pauses at interrupt) and the second (which resumes).
graph = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Helper for printing a final state block
# ---------------------------------------------------------------------------

def _print_final(label: str, final: dict) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    print(f"  Verified       : {final.get('verified')}  ({final.get('employee_name')}, {final.get('department')})")
    print(f"  KB found       : {final.get('kb_found')}")
    print(f"  Ticket ID      : {final.get('ticket_id') or '(none)'}")
    print(f"  Escalation ID  : {final.get('escalation_id') or '(none)'}")
    print(f"  Steps taken    : {final.get('step_count')}")
    print(f"\n  Attempts log:")
    for i, entry in enumerate(final.get("attempts") or [], 1):
        print(f"    {i}. {entry}")
    print(f"\n  Post-call review:")
    review = final.get("review")
    if review:
        print(f"    " + json.dumps(review, indent=4).replace("\n", "\n    "))
    else:
        print("    (none)")


# ---------------------------------------------------------------------------
# Two-run demo: happy path + HITL interrupt/resume
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    _BLANK: SupportState = {
        "issue_text":    "",
        "employee_id":   "",
        "employee_name": None,
        "department":    None,
        "verified":      False,
        "kb_found":      False,
        "kb_steps":      [],
        "ticket_id":     None,
        "escalation_id": None,
        "attempts":      [],
        "confidence":    1.0,
        "step_count":    0,
        "review":        None,
    }

    # -----------------------------------------------------------------------
    # RUN 1 — happy path: VPN issue, KB match, no interrupt
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  RUN 1 — Happy path (KB match, no HITL gate)")
    print("=" * 55)
    print("  Employee : E1001   Issue : I can't connect to the VPN")

    config_1 = {"configurable": {"thread_id": "run-1"}}
    final_1 = graph.invoke(
        {**_BLANK, "employee_id": "E1001", "issue_text": "I can't connect to the VPN"},
        config=config_1,
    )
    _print_final("RUN 1 — completed without interrupt", final_1)

    # -----------------------------------------------------------------------
    # RUN 2 — HITL path: unknown issue, no KB match, interrupt fires
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 55)
    print("  RUN 2 — HITL path (no KB match, interrupt triggered)")
    print("=" * 55)
    print("  Employee : E1002   Issue : my chair is broken")

    config_2 = {"configurable": {"thread_id": "run-2"}}

    # First invoke — runs intake + knowledge, then hits interrupt() in action.
    # graph.invoke() returns the accumulated state at the pause point.
    state_at_pause = graph.invoke(
        {**_BLANK, "employee_id": "E1002", "issue_text": "my chair is broken"},
        config=config_2,
    )

    # Read the interrupt message from the checkpointer snapshot.
    snapshot = graph.get_state(config_2)
    intr_value = None
    for task in snapshot.tasks:
        for intr in task.interrupts:
            intr_value = intr.value

    print(f"\n  ┌─ GRAPH PAUSED ──────────────────────────────────")
    print(f"  │  Interrupt : {intr_value}")
    print(f"  │  Pending   : {snapshot.next}")
    print(f"  │  State so far — verified: {state_at_pause.get('verified')}, "
          f"kb_found: {state_at_pause.get('kb_found')}, "
          f"step_count: {state_at_pause.get('step_count')}")
    print(f"  └─────────────────────────────────────────────────")

    # Human approves the ticket proposal.
    print(f"\n  [HUMAN DECISION → 'yes' (approve ticket)]")
    final_2 = graph.invoke(Command(resume="yes"), config=config_2)
    _print_final("RUN 2 — completed after human approval", final_2)

    print()
