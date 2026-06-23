"""judge_score.py — calibrate the judge against HUMAN labels (Track 1).

This answers Arvind's "who judges the judge?" question. The judge's verdict is
only trustworthy if it AGREES with a human on a labelled set. So we treat the
judge as a binary classifier of one failure mode, and score its agreement with
the human label using precision / recall / F1 — the same classification frame
as the router eval.

Run one model's verdicts vs the human labels:
  python judge_score.py judge_calibration_set.csv preds_vague_modelA.csv vague_failure

Compare two models head to head (Arvind's GPT-4o vs GPT-5 pattern):
  python judge_score.py judge_calibration_set.csv preds_vague_modelA.csv vague_failure \
      --compare preds_vague_modelB.csv

The human label column in the calibration set must match the failure mode:
  - vague_failure   (true/false, hand-labelled)
  - safety_failure  (true/false, hand-labelled)

The model you KEEP is the one with the best agreement with the human — usually
prioritising PRECISION for a safety/alert judge (don't cry wolf), but you
decide based on the use case.
"""

import argparse
import csv
import sys


def load_truth(calibration_csv, label_col):
    truth = {}
    with open(calibration_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if label_col not in r:
                sys.exit(f"Calibration set has no '{label_col}' column. "
                         f"Hand-label it first (true/false per row).")
            truth[r["id"]] = str(r[label_col]).strip().lower() in ("true", "yes", "1")
    return truth


def load_verdicts(preds_csv):
    v = {}
    with open(preds_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v[r["id"]] = str(r["judge_verdict"]).strip().lower() in ("true", "yes", "1")
    return v


def prf(truth, verdicts):
    ids = [i for i in truth if i in verdicts]
    tp = sum(1 for i in ids if truth[i] and verdicts[i])
    fp = sum(1 for i in ids if not truth[i] and verdicts[i])
    fn = sum(1 for i in ids if truth[i] and not verdicts[i])
    tn = sum(1 for i in ids if not truth[i] and not verdicts[i])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    agree = (tp + tn) / len(ids) if ids else 0.0
    return dict(n=len(ids), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, agreement=agree)


def report(name, m):
    print(f"\n--- {name} ---")
    print(f"  rows judged : {m['n']}")
    print(f"  TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")
    print(f"  precision   : {m['precision']:.2f}   (of alerts raised, how many were real)")
    print(f"  recall      : {m['recall']:.2f}   (of real failures, how many caught)")
    print(f"  F1          : {m['f1']:.2f}")
    print(f"  agreement   : {m['agreement']:.1%}   (judge matches human)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration_csv")
    ap.add_argument("preds_csv")
    ap.add_argument("label_col", choices=["vague_failure", "safety_failure"])
    ap.add_argument("--compare", default=None, help="second model's preds csv")
    args = ap.parse_args()

    truth = load_truth(args.calibration_csv, args.label_col)
    n_pos = sum(truth.values())
    print(f"Human-labelled set: {len(truth)} rows, {n_pos} positive "
          f"({args.label_col}=true), {len(truth)-n_pos} negative.")
    if n_pos < 3:
        print("WARNING: very few positive cases — agreement numbers will be "
              "noisy. This is expected for a rare failure mode; treat as "
              "directional. (Same caveat Arvind gave for his imbalanced set.)")

    mA = prf(truth, load_verdicts(args.preds_csv))
    report(f"Model A  ({args.preds_csv})", mA)

    if args.compare:
        mB = prf(truth, load_verdicts(args.compare))
        report(f"Model B  ({args.compare})", mB)
        print("\n=== VERDICT ===")
        better = "A" if mA["f1"] >= mB["f1"] else "B"
        print(f"Higher F1: Model {better}. "
              f"(A={mA['f1']:.2f} vs B={mB['f1']:.2f})")
        print("For a safety/alert judge, also weigh PRECISION: "
              f"A={mA['precision']:.2f} vs B={mB['precision']:.2f}. "
              "Pick the model+prompt you'd trust to alert in production.")
    else:
        print("\nRun again with --compare <model B preds> to pick between models.")


if __name__ == "__main__":
    main()
