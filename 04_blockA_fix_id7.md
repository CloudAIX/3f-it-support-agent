# Block A — fix the id=7 routing miss (Phase 3, fix 1 of 2)

**Cluster:** "Can't log in." routed to `lookup_employee` (identity) instead of
`search_kb` (it's a login/password support issue). One real failure, one root
cause: the `/route` system prompt biases short identity-shaped utterances toward
employee lookup.

Paste this into Claude Code as one instruction.

---

In `main.py`, edit ONLY the `/route` endpoint's system prompt. Do not touch any
other endpoint. Show me the diff before committing. Do not push.

Add a short disambiguation rule to the routing instructions, in plain words:

> A caller describing a problem they are HAVING — "can't log in", "password not
> working", "locked out", "email won't send", "no internet" — is asking for a
> FIX. Route these to `search_kb`, not `lookup_employee`. Only route to
> `lookup_employee` when the caller is asking to verify or pull up an account /
> record, or gives an employee ID to be checked. "Can't log in" is a support
> issue (search_kb), not an identity lookup.

Keep the change minimal — one rule added, nothing else reworded. Then re-run the
train split and show me the new quality table next to the baseline:

```bash
python run_baseline.py golden_dataset_v1.csv predictions_train_fix1.csv --split train
python score.py golden_dataset_v1.csv cost_latency_budgets.csv predictions_train_fix1.csv --split train
```

Report: did id=7 flip to a pass? Did anything else regress (watch the other
`search_kb` and `lookup_employee` rows — a disambiguation rule can over-correct)?
That regression check is the point — one change, measured, before the next.
