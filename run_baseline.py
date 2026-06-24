"""run_baseline.py — POST each utterance to /route and record predictions.

Usage:
    export ROUTE_URL="https://<tunnel>/route"
    python run_baseline.py golden_dataset_v1.csv predictions_train_baseline.csv --split train

Writes a predictions CSV with columns:
    id, predicted_tool, predicted_requires_approval, latency_ms, total_tokens, route_path, raw

total_tokens = prompt_tokens + completion_tokens from the /route response usage field.
Fast-path (pre-classifier) rows have 0 tokens — no LLM call is made.
Latency is wall-clock time for the HTTP round-trip (includes network).
For gated tools (create_ticket, escalate) the budget spec says agent_time_only,
so wall-clock here is conservative (will over-report latency for those rows).
"""

import argparse
import csv
import os
import sys
import time

import requests

ROUTE_URL = os.environ.get("ROUTE_URL", "http://127.0.0.1:8000/route")
HEADERS = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}


def run(golden_path, output_path, split):
    with open(golden_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if split and split != "all":
        rows = [r for r in rows if r.get("split") == split]
        if not rows:
            sys.exit(f"No rows with split={split!r} in {golden_path}")

    print(f"ROUTE_URL : {ROUTE_URL}")
    print(f"Running   : {len(rows)} rows  (split={split})")
    print()

    results = []
    errors = 0
    for i, row in enumerate(rows, 1):
        utterance = row["caller_utterance"]
        row_id = row["id"]
        payload = {"utterance": utterance}

        try:
            t0 = time.perf_counter()
            resp = requests.post(ROUTE_URL, json=payload, timeout=30, headers=HEADERS)
            latency_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            errors += 1
            print(f"  [{i:>2}/{len(rows)}] id={row_id}  ERROR: {e}")
            results.append({
                "id": row_id,
                "predicted_tool": "error",
                "predicted_requires_approval": False,
                "latency_ms": 0,
                "total_tokens": 0,
                "route_path": "error",
                "raw": str(e),
            })
            continue

        predicted_tool = data.get("chosen_tool", "error")
        args = data.get("args", {})
        predicted_requires_approval = args.get("requires_approval", False)
        reasoning = data.get("reasoning", "")
        route_path = data.get("route_path", "llm")
        usage = data.get("usage") or {}
        total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        target = row.get("target_tool", "?")
        match = "✓" if predicted_tool == target else f"✗ (want {target})"
        path_tag = "[fast]" if route_path == "fast" else "[llm] "

        print(f"  [{i:>2}/{len(rows)}] id={row_id}  {path_tag}  {utterance[:38]!r:<40}  "
              f"→ {predicted_tool:<18}  {match}  {latency_ms:.0f}ms")

        results.append({
            "id": row_id,
            "predicted_tool": predicted_tool,
            "predicted_requires_approval": predicted_requires_approval,
            "latency_ms": round(latency_ms, 1),
            "total_tokens": total_tokens,
            "route_path": route_path,
            "raw": reasoning,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "predicted_tool", "predicted_requires_approval",
                        "latency_ms", "total_tokens", "route_path", "raw"],
        )
        writer.writeheader()
        writer.writerows(results)

    correct = sum(
        1 for r, orig in zip(results, rows)
        if r["predicted_tool"] == orig.get("target_tool")
    )
    n_fast = sum(1 for r in results if r.get("route_path") == "fast")
    n_llm = len(results) - n_fast
    fast_lat = [r["latency_ms"] for r in results if r.get("route_path") == "fast" and r["latency_ms"]]
    llm_lat  = [r["latency_ms"] for r in results if r.get("route_path") != "fast" and r["latency_ms"]]
    all_lat  = [r["latency_ms"] for r in results if r["latency_ms"]]

    print(f"\nWrote {len(results)} predictions → {output_path}")
    print(f"Quick accuracy : {correct}/{len(results)} = {correct/len(results):.1%}"
          + (f"  ({errors} errors)" if errors else ""))
    print(f"Route path     : fast={n_fast}  llm={n_llm}")
    if fast_lat:
        print(f"Median latency : fast={sorted(fast_lat)[len(fast_lat)//2]:.0f}ms  "
              f"llm={sorted(llm_lat)[len(llm_lat)//2]:.0f}ms  "
              f"overall={sorted(all_lat)[len(all_lat)//2]:.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden_path")
    ap.add_argument("output_path")
    ap.add_argument("--split", default="train", choices=["all", "train", "validation"])
    args = ap.parse_args()
    run(args.golden_path, args.output_path, args.split)


if __name__ == "__main__":
    main()
