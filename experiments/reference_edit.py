"""Fit the chosen edit on all dev cues, score it once on test, and compare with random edits of the same size."""
import copy, json, time
import numpy as np, torch
from wsp import data
from wsp.edit import PassCounter, collect, build_edit, apply_edit
from wsp.evaluate import evaluate
from wsp.model import load

BLOCK, N, ALPHA = 1, 128, 16.0          # chosen on dev (results/edit_search_2.json)
CUES = {"but", "nobody", "far", "opposite", "means"}
tok, base = load()
dev = [e for e in data.template_set("dev") if e.kind in ("implicit", "explicit")]

t0 = time.perf_counter()
c = PassCounter()
S, k, v, mu = build_edit(*collect(tok, base, [e.text for e in dev], BLOCK, c), CUES, N)
edited = copy.deepcopy(base)
changed = apply_edit(edited, BLOCK, S, k, v, mu, ALPHA)
fit_seconds = time.perf_counter() - t0
total = sum(p.numel() for p in base.parameters())
torch.save(edited.state_dict(), "results/edited_reference.pt")

out = {"edit": {"block": BLOCK, "neurons": N, "alpha": ALPHA, "weights_changed": changed,
                "fraction_of_model": changed / total, "fit_seconds": fit_seconds,
                "fit_forward_passes": c.n},
       "edited": {k_: a for k_, (a, n) in evaluate(tok, edited).items()}}

# Control: same neurons count, same block, same norms, random directions.
rng = np.random.default_rng(0)
rand = []
for seed in range(5):
    m = copy.deepcopy(base)
    S_r = rng.choice(3072, N, replace=False)
    k_r = rng.standard_normal(N).astype(np.float32); k_r *= np.linalg.norm(k) / np.linalg.norm(k_r)
    v_r = rng.standard_normal(768).astype(np.float32); v_r *= np.linalg.norm(v) / np.linalg.norm(v_r)
    apply_edit(m, BLOCK, S_r, k_r, v_r, mu * 0, ALPHA)
    rand.append({k_: a for k_, (a, n) in evaluate(tok, m).items()})
out["random_control_mean"] = {key: float(np.mean([r[key] for r in rand])) for key in rand[0]}
json.dump(out, open("results/reference_edit.json", "w"), indent=2)

base_res = json.load(open("results/baseline.json"))
print(f"weights changed: {changed:,} of {total:,} ({changed/total:.3%}); fit {fit_seconds:.2f}s, {c.n} forward pass(es)")
print(f"{'set':30s} {'baseline':>9s} {'edited':>9s} {'random':>9s}")
for key in out["edited"]:
    print(f"{key:30s} {base_res[key]['accuracy']:9.1%} {out['edited'][key]:9.1%} {out['random_control_mean'][key]:9.1%}")
