"""Does the model already know a sentence is negated, even for cue words it was never shown?

Closed-form probe (difference of means, no training) on mean-pooled hidden states,
leave-one-cue-out over the 5 dev implicit frames. Dev only.
"""
import numpy as np, torch
from wsp import data
from wsp.model import load, DEVICE
tok, m = load()
dev = data.template_set("dev")
enc = tok([e.text for e in dev], padding=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    hs = m(**enc, output_hidden_states=True).hidden_states
mask = enc["attention_mask"].unsqueeze(-1).float()
frames = [f for f in data.IMPLICIT_FRAMES["dev"]]
neg = np.array([e.kind != "affirmative" for e in dev]); fr = np.array([e.frame for e in dev])
for L, h in enumerate(hs):
    X = ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
    accs = []
    for held in frames:
        tr = fr != held
        d = X[tr & neg].mean(0) - X[tr & ~neg].mean(0)
        s = X @ d; thr = (s[tr & neg].mean() + s[tr & ~neg].mean()) / 2
        accs.append(((s > thr) == neg)[fr == held].mean())
    print(f"layer {L}: unseen-cue 'is negated' readout {np.mean(accs):.0%}   per cue {[f'{a:.0%}' for a in accs]}")
