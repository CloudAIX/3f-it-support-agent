# 3F — Track 1 (code-heavy) running tally

Track 2 (no-code/low-code) is the submission track. These are the code-heavy
pieces, logged so they can be picked up slowly without getting lost.

## Built (this eval cycle)
1. `/route` endpoint — utterance → tool decision via Nebius (Llama 3.3 70B),
   six-target vocabulary matching the golden dataset. Makes the routing decision
   scriptable so the golden dataset can score it. STATUS: live, tested.
2. `run_baseline.py` — eval runner; calls /route per golden row, captures
   chosen tool + approval flag + latency. STATUS: working; baseline run done.
3. `score.py` v2 — three-axis scorer (quality F1 / HOTL safety gate /
   cost+latency budgets) with --split flag. STATUS: working, smoke-tested.
4. Held-out validation split (5 rows, seed 42, stratified). STATUS: in dataset.

## To pick up later (logged, not tonight)
- **Token-usage passthrough in /route** — return Nebius `usage` so the
  `total_tokens` budget axis actually measures. Currently blank/untested.
  Small change. Until done, do NOT report token compliance as passing.
- **Latency redesign — local-models angle (Evals/finetuning bridge)** — if the
  keyword pre-classifier (Block B) doesn't move median latency enough, the
  fuller fix is a small LOCAL intent model (Ollama / a LoRA-tuned classifier)
  doing cheap routing, reserving the 70B Nebius call for genuinely ambiguous
  utterances. This is the clean link to the syllabus's local-models topic.
- **Hamel evals-skills toolkit (hamelsmu/evals-skills)** — MIT-licensed Claude
  Code plugin (v0.2.0, 7 skills), installed as a learning resource. NOT a 3F
  deliverable. Likely forked/cloned into my GitHub — it's a reference toolkit,
  not capstone work. Could run its `error-analysis` skill over the 3F failures
  as a Track 1 exercise later. Hamel's error-analysis method already credited
  in the 3F eval README.

## Naming note for public artefacts
Label the eval work by SYLLABUS TOPIC ("Evals"), not "Week 4". The live sessions
ran evals under a Week-4 banner due to a course reorder, but the published
syllabus lists Evals as Week 5 / Finetuning & local models as Week 4. Public
repo + Confluence should say "Evals" to avoid a visible mismatch.
