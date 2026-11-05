"""Scenario: you have only a handful of labelled failure examples.

For N in 10, 40, 160, 640 and all 2,080 dev template sentences (3 random subsets each):
  - edit: the v2 recipe (block 1, 64 neurons, alpha 8) built from the subset's cue and
    "not" tokens, plus 512 UNLABELLED real reviews as ordinary-token contrast
  - fine-tune head / LoRA on the same N labelled sentences (no extra labelled data)
Scored on the hidden test set (implicit negation, unseen phrasings) and real SST-2 validation.
"""

import copy
import json
import random

import numpy as np

from wsp import data, finetune
from wsp.edit import PassCounter, apply_edit, build_edit, collect
from wsp.evaluate import evaluate
from wsp.model import load

CUE_OF = {"{S} was anything but {A}.": "but", "Nobody would call {s} {A}.": "nobody",
          "{S} was far from {A}.": "far", "{S} was the opposite of {A}.": "opposite",
          "{S} was by no means {A}.": "means"}
KEYS = ["template_test_implicit", "template_test_explicit", "sst2_val_all"]


def subset(dev, n, seed):
    rng = random.Random(seed)
    if n >= len(dev):
        return dev
    imp = [e for e in dev if e.kind == "implicit"]
    exp = [e for e in dev if e.kind == "explicit"]
    aff = [e for e in dev if e.kind == "affirmative"]
    k = max(2, n * 4 // 10)  # 40% implicit, 40% explicit, 20% affirmative
    return rng.sample(imp, k) + rng.sample(exp, k) + rng.sample(aff, max(1, n - 2 * k))


def run_edit(sub, real):
    tok, base = load()
    cues = {CUE_OF[e.frame] for e in sub if e.kind == "implicit"}
    fit = [e for e in sub if e.kind in ("implicit", "explicit")] + real
    S, k, v, mu = build_edit(*collect(tok, base, [e.text for e in fit], 1, PassCounter()), cues, 64)
    apply_edit(base, 1, S, k, v, mu, 8.0)
    return {key: a for key, (a, _) in evaluate(tok, base, splits=("test",)).items()}


def run_ft(sub, variant, seed):
    import torch
    torch.manual_seed(seed)
    tok, model = load(attn_implementation="eager")
    model = finetune.prepare(model, variant)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=finetune.SETTINGS[variant][0])
    steps = max(30, 3 * (len(sub) + 15) // 16)  # at least 30 steps so tiny sets still train
    rng = random.Random(seed)
    from wsp.model import DEVICE
    for _ in range(steps):
        batch = rng.sample(sub, min(16, len(sub)))
        enc = tok([e.text for e in batch], padding=True, return_tensors="pt").to(DEVICE)
        loss = model(**enc, labels=torch.tensor([e.label for e in batch], device=DEVICE)).loss
        loss.backward(); opt.step(); opt.zero_grad()
    if variant == "lora":
        model = model.merge_and_unload()
    model.eval()
    return {key: a for key, (a, _) in evaluate(tok, model, splits=("test",)).items()}


def main():
    dev = data.template_set("dev")
    real = data.sst2_train_sample(512, seed=0)
    for e in real:
        e.label = -1  # unlabelled: the edit only uses these as ordinary-token contrast
    rows = []
    for n in (10, 40, 160, 640, 2080):
        for seed in range(3 if n < 2080 else 1):
            sub = subset(dev, n, seed)
            for method in ("edit", "head", "lora"):
                res = run_edit(sub, real) if method == "edit" else run_ft(sub, method, seed)
                rows.append({"n": n, "seed": seed, "method": method} | {k: res[k] for k in KEYS})
                print(rows[-1], flush=True)
            json.dump(rows, open("results/scenario_few_labels.json", "w"), indent=1)

    print("\nmeans (implicit test / real reviews)")
    for n in (10, 40, 160, 640, 2080):
        line = f"N={n:5d}"
        for m in ("edit", "head", "lora"):
            r = [x for x in rows if x["n"] == n and x["method"] == m]
            line += f"  {m}: {np.mean([x['template_test_implicit'] for x in r]):.1%} / {np.mean([x['sst2_val_all'] for x in r]):.1%}"
        print(line)


if __name__ == "__main__":
    main()
