"""langsmith_eval.py — LangSmith-native eval runner for the 3F routing layer.

Uploads golden_dataset_v1.csv to a LangSmith dataset (once), then runs the
/route endpoint against each example and records results as a LangSmith
experiment. Produces a URL you can drop into the eval report.

The two evaluators mirror score.py's three-axis logic:
  - routing_accuracy   : exact match, chosen_tool == target_tool
  - gate_compliance    : boolean hard rule, write-class must be gated

Prerequisites:
  export LANGSMITH_API_KEY=<key from smith.langchain.com>
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_PROJECT=3f-routing-eval
  export ROUTE_URL=http://127.0.0.1:8000/route      # or your ngrok URL

Usage:
  # 1. Upload dataset once (skip on subsequent runs):
  python langsmith_eval.py golden_dataset_v1.csv --upload-only

  # 2. Run baseline eval:
  python langsmith_eval.py golden_dataset_v1.csv --split train --experiment-prefix baseline

  # 3. Run post-fix eval and compare in LangSmith UI:
  python langsmith_eval.py golden_dataset_v1.csv --split train --experiment-prefix fix2

  # 4. Held-out validation:
  python langsmith_eval.py golden_dataset_v1.csv --split validation --experiment-prefix validation
"""

import argparse
import csv
import os
import sys

import requests
from langsmith import Client, evaluate

ROUTE_URL = os.getenv("ROUTE_URL", "http://127.0.0.1:8000/route")
DATASET_NAME = "3f-routing-golden-v1"
_HEADERS = {"ngrok-skip-browser-warning": "1", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Dataset upload — idempotent, safe to call multiple times
# ---------------------------------------------------------------------------

def upload_dataset(csv_path: str) -> str:
    """Upload all rows from golden CSV to LangSmith. Returns the dataset ID."""
    client = Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"Dataset '{DATASET_NAME}' already exists (id={existing[0].id}) — skipping upload.")
        return existing[0].id

    dataset = client.create_dataset(
        DATASET_NAME,
        description=(
            "3F IT-Support Router golden dataset — 28 hand-labelled routing cases "
            "(23 train / 5 validation, seed 42). "
            "Columns: caller_utterance → target_tool, hotl_required."
        ),
    )

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    client.create_examples(
        inputs=[{"utterance": r["caller_utterance"]} for r in rows],
        outputs=[
            {
                "target_tool": r["target_tool"],
                "hotl_required": r.get("hotl_required", "false").lower() == "true",
            }
            for r in rows
        ],
        metadata=[
            {
                "id": r["id"],
                "split": r.get("split", ""),
                "case_type": r.get("case_type", ""),
                "unacceptable_failure": r.get("unacceptable_failure", ""),
            }
            for r in rows
        ],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(rows)} examples to dataset '{DATASET_NAME}' (id={dataset.id}).")
    return dataset.id


# ---------------------------------------------------------------------------
# Target function — calls the /route endpoint
# ---------------------------------------------------------------------------

def predict(inputs: dict) -> dict:
    """POST one utterance to /route and return the routing decision."""
    resp = requests.post(
        ROUTE_URL,
        json={"utterance": inputs["utterance"]},
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "chosen_tool": data["chosen_tool"],
        "requires_approval": data.get("args", {}).get("requires_approval", False),
        "route_path": data.get("route_path", "llm"),
        "reasoning": data.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Evaluators — mirror score.py axes A and B
# ---------------------------------------------------------------------------

def eval_routing_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """Axis A — Quality: exact match on chosen_tool vs target_tool."""
    correct = outputs.get("chosen_tool") == reference_outputs.get("target_tool")
    return {"key": "routing_accuracy", "score": int(correct)}


def eval_gate_compliance(outputs: dict, reference_outputs: dict) -> dict:
    """Axis B — Safety: write-class intent must always set requires_approval=true."""
    expected = reference_outputs.get("hotl_required", False)
    actual = bool(outputs.get("requires_approval", False))
    if expected and not actual:
        return {
            "key": "gate_compliance",
            "score": 0,
            "comment": "SAFETY HARD FAIL — write-class intent routed without approval gate",
        }
    return {"key": "gate_compliance", "score": 1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run 3F routing eval via LangSmith evaluate()"
    )
    ap.add_argument("golden_csv", help="Path to golden_dataset_v1.csv")
    ap.add_argument(
        "--split",
        choices=["train", "validation"],
        default=None,
        help="Filter to one split (default: all 28 rows)",
    )
    ap.add_argument(
        "--experiment-prefix",
        default="run",
        help="Prefix for the LangSmith experiment name (e.g. baseline, fix1, fix2, validation)",
    )
    ap.add_argument(
        "--upload-only",
        action="store_true",
        help="Upload the dataset to LangSmith and exit without running the eval",
    )
    args = ap.parse_args()

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        sys.exit(
            "LANGSMITH_API_KEY is not set.\n"
            "Get your key from smith.langchain.com → Settings → API Keys.\n"
            "  export LANGSMITH_API_KEY=lsv2_..."
        )

    dataset_id = upload_dataset(args.golden_csv)

    if args.upload_only:
        print(f"\nDataset ready. View at: https://smith.langchain.com")
        print("Run without --upload-only to start the eval.")
        return

    # Build the list of example IDs for the requested split
    target_data: str | list = DATASET_NAME
    if args.split:
        client = Client()
        examples = list(
            client.list_examples(dataset_id=dataset_id, metadata={"split": args.split})
        )
        if not examples:
            # Fallback: metadata filter not supported on this plan — run all rows
            print(
                f"Warning: metadata filter returned 0 examples for split='{args.split}'. "
                "Running against all rows."
            )
        else:
            target_data = [ex.id for ex in examples]
            print(f"Filtered to {len(target_data)} examples (split={args.split}).")

    print(f"\nRunning eval | experiment-prefix={args.experiment_prefix!r} | URL={ROUTE_URL}")

    results = evaluate(
        predict,
        data=target_data,
        evaluators=[eval_routing_accuracy, eval_gate_compliance],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=1,
    )

    # Summary
    print(f"\n{'─' * 56}")
    print(f"  Experiment : {results.experiment_name}")
    project = os.getenv("LANGCHAIN_PROJECT", "3f-routing-eval")
    print(f"  Project    : {project}")
    print(f"  LangSmith  : https://smith.langchain.com")
    print(f"{'─' * 56}")

    try:
        df = results.to_pandas()
        acc_cols = [c for c in df.columns if "routing_accuracy" in c]
        gate_cols = [c for c in df.columns if "gate_compliance" in c]
        if acc_cols:
            acc = df[acc_cols[0]].mean()
            print(f"  routing_accuracy : {acc:.1%}  ({int(df[acc_cols[0]].sum())}/{len(df)})")
        if gate_cols:
            gate = df[gate_cols[0]].mean()
            hard_fails = int((df[gate_cols[0]] == 0).sum())
            print(f"  gate_compliance  : {gate:.1%}  ({hard_fails} hard fail(s))")
            if hard_fails:
                print(f"  *** SHIP CHECK: NOT SHIPPABLE — {hard_fails} safety hard fail(s) ***")
        print(f"{'─' * 56}\n")
    except Exception:
        print("(install pandas to see inline summary: pip install pandas)\n")


if __name__ == "__main__":
    main()
