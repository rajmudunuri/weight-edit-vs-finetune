"""Scenario: auditable, reversible changes. Apply edit v2, then remove it, and compare.

Also stacks the edit twice and removes one copy, to show edits compose like patches.
"""

import json

import torch

from wsp import data
from wsp.edit import PassCounter, apply_edit, build_edit, collect
from wsp.model import load, predict

tok, model = load()
original = {k: v.clone() for k, v in model.state_dict().items()}
dev = [e for e in data.template_set("dev") if e.kind in ("implicit", "explicit")] + data.sst2_train_sample(512, seed=0)
S, k, v, mu = build_edit(*collect(tok, model, [e.text for e in dev], 1, PassCounter()),
                         {"but", "nobody", "far", "opposite", "means"}, 64)
val = [e.text for e in data.sst2_validation()]
p0 = predict(tok, model, val)


def diff():
    cur = model.state_dict()
    return max(float((cur[n] - original[n]).abs().max()) for n in original)


apply_edit(model, 1, S, k, v, mu, 8.0)
p_edit = predict(tok, model, val)
apply_edit(model, 1, S, k, v, mu, -8.0)          # remove: the same rule with the opposite sign
after_remove = diff()
p_back = predict(tok, model, val)
apply_edit(model, 1, S, k, v, mu, 8.0); apply_edit(model, 1, S, k, v, mu, 8.0); apply_edit(model, 1, S, k, v, mu, -8.0)
p_stack = predict(tok, model, val)
out = {
    "max_abs_weight_diff_after_remove": after_remove,
    "predictions_changed_by_edit": int(((p_edit > 0.5) != (p0 > 0.5)).sum()),
    "predictions_differing_after_remove": int(((p_back > 0.5) != (p0 > 0.5)).sum()),
    "max_prob_diff_after_remove": float((p_back - p0).abs().max()),
    "stack_two_remove_one_matches_single_edit": int(((p_stack > 0.5) != (p_edit > 0.5)).sum()) == 0,
}
json.dump(out, open("results/scenario_reversible.json", "w"), indent=2)
print(json.dumps(out, indent=2))
