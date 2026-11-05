"""v2 search: real SST-2 train text joins the contrast set and the guard (dev side only)."""
import copy, itertools, json
from wsp import data
from wsp.edit import PassCounter, collect, build_edit, apply_edit
from wsp.model import load, predict
from wsp.evaluate import accuracy

CUES = {"{S} was anything but {A}.": {"but"}, "Nobody would call {s} {A}.": {"nobody"},
        "{S} was far from {A}.": {"far"}, "{S} was the opposite of {A}.": {"opposite"},
        "{S} was by no means {A}.": {"means"}}
tok, base = load()
dev = data.template_set("dev")
real = data.sst2_train_sample(512, seed=0)          # contrast tokens
real_guard = data.sst2_train_sample(512, seed=1)    # separate guard sentences
c = PassCounter(); cache = {}
base_real = accuracy(predict(tok, base, [e.text for e in real_guard]), real_guard)[0]; c.n += 1
print(f"baseline accuracy on real-text guard: {base_real:.1%}")
rows = []
for block, n, alpha in itertools.product([1, 2], [32, 64, 128], [8.0, 16.0, 24.0]):
    loo = []
    for held in CUES:
        train_cues = set().union(*[v for f, v in CUES.items() if f != held])
        fit = [e for e in dev if e.frame != held and e.kind in ("implicit", "explicit")] + real
        if (block, held) not in cache:
            cache[(block, held)] = collect(tok, base, [e.text for e in fit], block, c)
        m = copy.deepcopy(base)
        apply_edit(m, block, *build_edit(*cache[(block, held)], train_cues, n), alpha)
        held_ex = [e for e in dev if e.frame == held]; tg = [e for e in dev if e.kind != "implicit"]
        p = predict(tok, m, [e.text for e in held_ex + tg + real_guard]); c.n += 1
        a, b = len(held_ex), len(held_ex) + len(tg)
        loo.append((accuracy(p[:a], held_ex)[0], accuracy(p[a:b], tg)[0], accuracy(p[b:], real_guard)[0]))
    r = dict(block=block, n=n, alpha=alpha, loo_implicit=sum(x[0] for x in loo) / 5,
             template_guard=min(x[1] for x in loo), real_guard=sum(x[2] for x in loo) / 5)
    rows.append(r)
    print(f"block {block} n {n:3d} alpha {alpha:4.1f}  held-out-cue {r['loo_implicit']:.1%}  template guard {r['template_guard']:.1%}  real guard {r['real_guard']:.1%}", flush=True)
print("forward passes used:", c.n)
json.dump(rows, open("results/edit_search_v2.json", "w"), indent=1)
