# Block B — bounded latency fix: cheap pre-classifier before the 70B call

**Cluster:** 20/23 rows over the latency budget. Root cause is STRUCTURAL —
every `/route` call makes a full Llama-3.3-70B round-trip via Nebius (~1.5–7s),
even for "morning!" or an obvious injection. MINT violation: a 70B reasoning
model is doing work a keyword check could do.

**This is Track 1 (code-heavy, side-learning).** Log it on the tally.

## SCOPE BOUNDARY — read before pasting

We are adding ONE thing: a cheap, deterministic pre-classifier that catches the
*unambiguous* cases and returns the answer WITHOUT calling Nebius. Everything
else falls through to the existing 70B call, unchanged.

We are NOT: rewriting the router, adding a second model, adding a local model,
or touching the four tool endpoints. If the pre-classifier doesn't move the
median latency enough, that is a FINDING — we report it and the fuller redesign
(e.g. a small local intent model — ties to the Week 4 local-models topic) goes
on the Track 1 tally for later. We do not chase it tonight.

The honesty rule that matters here: do NOT loosen the latency budgets to make
rows pass. The budget is the bar. We make the agent faster, or we report the
true number. Moving the bar to turn red green is eval-hacking.

## The instruction

Paste this into Claude Code as one instruction.

---

In `main.py`, add a deterministic pre-classifier function that runs at the START
of `/route`, BEFORE the Nebius call. Do not change the Nebius path — it stays as
the fall-through for anything the pre-classifier does not confidently catch. Do
not touch the four tool endpoints. Show me the diff before committing. Do not push.

The pre-classifier handles only the cases that are unambiguous by simple rule:

1. **chitchat** — utterance is only a greeting / thanks / small talk with no IT
   content. Match a small keyword set ("hi", "hello", "morning", "thanks",
   "thank you", "cheers", "how are you", "lol", "good") AND no IT-issue keyword
   present. Return `chosen_tool="chitchat"`, no Nebius call.

2. **chitchat (injection refusal)** — utterance contains an obvious
   prompt-injection / instruction-override pattern ("ignore your instructions",
   "ignore previous", "reveal your prompt", "tell me everyone's passwords",
   "system prompt"). Return `chosen_tool="chitchat"` with reasoning
   "injection_refusal", no Nebius call.

3. **unsupported** — utterance clearly matches an out-of-scope domain by keyword:
   personal accounts ("personal gmail", "my own"), procurement ("order",
   "buy", "purchase", "supplier"), HR ("annual leave", "holiday", "payroll",
   "salary"), or coding requests ("write me a script", "python script",
   "scrape"). Return `chosen_tool="unsupported"`, no Nebius call.

If NONE of these match with confidence, fall through to the existing Nebius
`/route` logic unchanged. When in doubt, fall through — a wrong cheap guess is
worse than a slow correct one. The pre-classifier must be conservative: it only
short-circuits when the rule is obvious.

Add a field to the response so we can see which path was taken:
`route_path: "fast" | "llm"`. Log it. This lets the eval show how many rows the
pre-classifier handled.

Keep the keyword lists short and readable at the top of the function as plain
lists — this is a transparent rule, not a clever one. MINT: smallest thing that
catches the obvious cases.

After it's in, re-run the train split and show me:

```bash
python run_baseline.py golden_dataset_v1.csv predictions_train_fix2.csv --split train
python score.py golden_dataset_v1.csv cost_latency_budgets.csv predictions_train_fix2.csv --split train
```

Report three things:
1. **Median latency** before vs after (the headline number).
2. **How many rows took the `fast` path** vs `llm` (from route_path).
3. **Did quality hold?** The pre-classifier must NOT misroute — if any chitchat/
   unsupported row that the 70B got right is now wrong, the rules are too
   greedy. Quality regression kills this fix regardless of the latency win.
