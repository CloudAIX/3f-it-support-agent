# LLM-as-a-Judge for 3F — Claude Code runbook (Track 1 side-learning)

Built Arvind's way: a judge does NOT re-do routing. It catches ONE failure mode,
calibrated against HUMAN labels, scored as precision/recall/F1. We build two
judges (vague-utterance, gated-safety) and compare two Nebius models for each.

**This is Track 1.** Your submission stays the Track 2 report. This enriches the
repo and your understanding. Confluence write-up is optional and noted at the end.

Run each numbered block as ONE input. Standing rules: show diff before commit,
do not push. Files go in a `judge/` folder in the repo.

---

## Phase 0 — set up

> Put `judge_runner.py`, `judge_score.py`, and `judge_calibration_set.csv` in a
> new `judge/` folder in the repo. Confirm `NEBIUS_API_KEY` is set (same key 3F
> uses). Tell me the two Nebius model strings your account actually serves — one
> ~70B (matches the router) and one LARGER (Ash's point: the judge should be a
> bigger/more general model than the one it judges). If you only have one size,
> we compare 70B vs a different 70B-class model. Do not commit yet.

## Phase 1 — HUMAN-LABEL the calibration set (the actual calibration)

> Open `judge_calibration_set.csv`. The `vague_failure` and `safety_failure`
> columns are PRE-FILLED as a hypothesis — my job now is to REVIEW and CORRECT
> them as the person who knows the product. This human labelling IS the source
> of truth the judge gets scored against; if it's wrong, everything downstream
> is wrong. Walk me through each row where you think my pre-fill might be off,
> one at a time, and let me confirm or flip the label. Then save. Show the diff.

## Phase 2 — run JUDGE 1 (vague-utterance), both models

> Run the vague-utterance judge with model A, then model B:
> ```
> python judge/judge_runner.py judge/judge_calibration_set.csv vague_utterance MODEL_A judge/preds_vague_A.csv
> python judge/judge_runner.py judge/judge_calibration_set.csv vague_utterance MODEL_B judge/preds_vague_B.csv
> ```
> Then score and compare against the human labels:
> ```
> python judge/judge_score.py judge/judge_calibration_set.csv judge/preds_vague_A.csv vague_failure --compare judge/preds_vague_B.csv
> ```
> Show me both models' precision/recall/F1/agreement and the verdict. Tell me
> which model to keep as the vague-utterance judge, and WHY (for an alert judge,
> weigh precision — don't cry wolf).

## Phase 3 — run JUDGE 2 (gated-safety), both models

> Same again for the safety judge:
> ```
> python judge/judge_runner.py judge/judge_calibration_set.csv gated_safety MODEL_A judge/preds_safety_A.csv
> python judge/judge_runner.py judge/judge_calibration_set.csv gated_safety MODEL_B judge/preds_safety_B.csv
> python judge/judge_score.py judge/judge_calibration_set.csv judge/preds_safety_A.csv safety_failure --compare judge/preds_safety_B.csv
> ```
> Show me the comparison. For the SAFETY judge, precision AND recall both matter
> — a missed write-without-approval is the dangerous case, so don't let recall
> drop. Tell me which model to keep.

## Phase 4 — if the best judge still isn't good enough

> If the kept model's precision/recall is weak, do NOT jump to fine-tuning
> (Arvind: fine-tuning a judge is expensive, reach for it late). First iterate
> the PROMPT in `judge_runner.py` — sharpen the rubric, add one or two worked
> examples of each failure, tighten the definition. Re-run Phase 2/3. Show me
> the before/after. Only if prompt iteration plateaus do we log fine-tuning as a
> Track 1 item. Show diff, do not push.

---

## What this gives you (and what it doesn't)

- A calibrated judge means: in production, this model+prompt can flag
  vague-utterance misroutes and write-without-approval misses at a known
  precision/recall, WITHOUT a human reviewing every call. That's the scale
  point — the judge is only trustworthy because you proved its agreement with a
  human on a labelled set first.
- It does NOT replace the router eval. The router eval (score.py) measures
  routing quality. The judge measures whether ONE failure mode is occurring,
  cheaply, at scale, online. Different jobs.

## Optional — Confluence write-up (Track 1 note)

If you want this in Confluence: a short page titled "3F — LLM-as-a-Judge
calibration (Track 1)" with the method (one failure mode per judge, calibrated
vs human), the two-model comparison table, and which model you kept for each
judge and why. Draft it here clean and paste it across by hand, like the rest.

## Scope line

Two judges is the ceiling for now. Do NOT add a third failure mode, RAGAS,
trajectory scoring, or production online-evaluation tonight — those are logged
Track 1 / Week 6 items. Build these two, calibrate them, keep the better model,
and that is a complete, genuinely-beyond-the-baseline piece of eval work.
