"""Do implicit cue tokens look like 'not' to the model? Cosine to the 'not' feature, per layer (dev only)."""
import torch, numpy as np
from wsp import data
from wsp.model import load, DEVICE
tok, m = load()
ex = data.template_set("dev")
texts = [e.text for e in ex]
enc = tok(texts, padding=True, return_tensors="pt")
with torch.no_grad():
    hs = m(**enc.to(DEVICE), output_hidden_states=True).hidden_states
ids = enc["input_ids"].cpu().numpy()
toks = [tok.convert_ids_to_tokens(r) for r in ids]
NEG = {"not", "never", "t"}                       # 't' = the n't in wasn't/don't
CUE = {"but", "nobody", "far", "opposite", "means"}
def positions(words, kinds):
    return [(i, j) for i, e in enumerate(ex) if e.kind in kinds for j, t in enumerate(toks[i]) if t in words]
neg_pos = positions(NEG, {"explicit"})
cue_pos = positions(CUE, {"implicit"})
other = [(i, j) for i, e in enumerate(ex) for j, t in enumerate(toks[i]) if t not in NEG | CUE | {"[PAD]", "[CLS]", "[SEP]"}]
for L, h in enumerate(hs):
    h = h.float().cpu().numpy()
    g = lambda P: np.stack([h[i, j] for i, j in P])
    base = g(other).mean(0)
    n = g(neg_pos).mean(0) - base
    for w in sorted(CUE):
        P = [(i, j) for i, j in cue_pos if toks[i][j] == w]
        c = g(P).mean(0) - base
        print(f"L{L} {w:9s} cos(cue, not) = {np.dot(c, n) / np.linalg.norm(c) / np.linalg.norm(n):+.2f}", end=" | ")
    print()
