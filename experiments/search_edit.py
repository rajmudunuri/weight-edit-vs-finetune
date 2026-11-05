"""Pick block / neurons / alpha on dev only, using leave-one-cue-out to estimate generalisation."""
import copy, itertools, json
import torch
from wsp import data
from wsp.edit import PassCounter, collect, build_edit, apply_edit
from wsp.model import load, predict
from wsp.evaluate import accuracy

CUES = {  # dev implicit frame -> its cue word(s)
    "{S} was anything but {A}.": {"but"},
    "Nobody would call {s} {A}.": {"nobody"},
    "{S} was far from {A}.": {"far"},
    "{S} was the opposite of {A}.": {"opposite"},
    "{S} was by no means {A}.": {"means"},
}
tok, base = load()
dev = data.template_set("dev")
counter = PassCounter()
cache = {}
rows = []
for block, n_neurons, alpha in itertools.product([1, 2], [64, 128, 256], [8.0, 16.0, 32.0]):
    loo = []
    for held, held_cues in CUES.items():
        train_cues = set().union(*[c for f, c in CUES.items() if f != held])
        fit = [e for e in dev if e.frame != held and e.kind in ("implicit", "explicit")]
        key = (block, held)
        if key not in cache:
            cache[key] = collect(tok, base, [e.text for e in fit], block, counter)
        S, k, v, mu = build_edit(*cache[key], train_cues, n_neurons)
        m = copy.deepcopy(base)
        apply_edit(m, block, S, k, v, mu, alpha)
        held_ex = [e for e in dev if e.frame == held]
        guard = [e for e in dev if e.kind != "implicit"]
        p = predict(tok, m, [e.text for e in held_ex + guard]); counter.n += 1
        loo.append((accuracy(p[:len(held_ex)], held_ex)[0], accuracy(p[len(held_ex):], guard)[0]))
    held_acc = sum(x for x, _ in loo) / len(loo); guard_acc = min(g for _, g in loo)
    rows.append(dict(block=block, n=n_neurons, alpha=alpha, loo_implicit=held_acc, min_guard=guard_acc))
    print(f"block {block} n {n_neurons:3d} alpha {alpha:3.1f}  held-out-cue implicit {held_acc:.1%}  worst guard {guard_acc:.1%}", flush=True)
print("forward passes used:", counter.n)
json.dump(rows, open("results/edit_search_2.json", "w"), indent=1)
