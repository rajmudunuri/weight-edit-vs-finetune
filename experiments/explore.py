"""Where does implicit negation get lost? Per-layer CLS geometry on dev data."""
import torch, numpy as np
from wsp import data
from wsp.model import load, DEVICE
tok, m = load()
ex = data.template_set("dev")
enc = tok([e.text for e in ex], padding=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = m(**enc, output_hidden_states=True)
H = [h[:, 0].float().cpu().numpy() for h in out.hidden_states]  # CLS per layer, 7 entries
kind = np.array([e.kind for e in ex]); lab = np.array([e.label for e in ex])
adj = np.array([e.adjective for e in ex]); subj = np.array([e.text for e in ex])
pol = np.where(kind == "affirmative", lab, 1 - lab)   # adjective polarity
for L, h in enumerate(H):
    # sentiment direction from affirmative examples
    s = h[(kind == "affirmative") & (lab == 1)].mean(0) - h[(kind == "affirmative") & (lab == 0)].mean(0)
    proj = h @ s / np.linalg.norm(s)
    def m_(k, p): return proj[(kind == k) & (pol == p)].mean()
    # flip direction: explicit-negated minus affirmative, same adjective polarity, averaged over polarity sign
    print(f"L{L}: sentiment-axis projection  aff+ {m_('affirmative',1):6.2f} aff- {m_('affirmative',0):6.2f} | "
          f"expl(+adj) {m_('explicit',1):6.2f} expl(-adj) {m_('explicit',0):6.2f} | "
          f"impl(+adj) {m_('implicit',1):6.2f} impl(-adj) {m_('implicit',0):6.2f}")
