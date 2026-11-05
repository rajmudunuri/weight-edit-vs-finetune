"""Score a model on every evaluation set and print a table.

Usage:  uv run python -m wsp.evaluate [--weights path.pt] [--out results.json]
"""

import argparse
import json

import torch

from wsp import data
from wsp.model import load, predict


def accuracy(probs, examples, keep=lambda e: True):
    idx = [i for i, e in enumerate(examples) if keep(e)]
    if not idx:
        return float("nan"), 0
    correct = sum(int((probs[i] > 0.5).item() == examples[i].label) for i in idx)
    return correct / len(idx), len(idx)


def evaluate(tok, model, splits=("dev", "test")):
    results = {}
    for split in splits:
        ex = data.template_set(split)
        p = predict(tok, model, [e.text for e in ex])
        for kind in ("affirmative", "explicit", "implicit"):
            results[f"template_{split}_{kind}"] = accuracy(p, ex, lambda e, k=kind: e.kind == k)

    if "test" in splits:
        ex = data.sst2_validation()
        p = predict(tok, model, [e.text for e in ex])
        results["sst2_val_all"] = accuracy(p, ex)
        results["sst2_val_negation"] = accuracy(p, ex, lambda e: e.kind == "sst2-negation")
        results["sst2_val_no_negation"] = accuracy(p, ex, lambda e: e.kind == "sst2-plain")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="state_dict of an edited model")
    ap.add_argument("--out")
    args = ap.parse_args()

    tok, model = load()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    res = evaluate(tok, model)
    for k, (acc, n) in res.items():
        print(f"{k:32s} {acc:7.1%}  (n={n})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({k: {"accuracy": a, "n": n} for k, (a, n) in res.items()}, f, indent=2)


if __name__ == "__main__":
    main()
