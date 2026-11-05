"""A closed-form, gradient-free weight edit: route implicit negation into 'not'.

Diagnosis (experiments/explore.py, cue_geometry.py): the model already has a
negation feature. Tokens like "not" write it, and block 4 uses it to flip the
sentence's sentiment. Implicit cues ("anything but", "nobody") mostly do not
write it, so their sentences are never flipped. "far" (as in "far from")
already does, and the model gets "far from good" right every time.

The edit adds one key -> value rule to the FFN of a single block:

    key   k : a direction in the FFN's 3072-wide hidden layer that is active on
              implicit cue tokens and quiet elsewhere (difference of means).
              Only the top-`n_neurons` entries are kept, so the edit touches
              just those columns of the FFN output matrix.
    value v : the direction in the residual stream that separates a "not"
              token from an implicit cue token at that block's output.

    W_out[:, S] += alpha * v k_S^T / |k_S|^2        (S = chosen neurons)
    b_out       -= alpha * v (k_S . mu_S) / |k_S|^2 (mu = mean activation on ordinary tokens)

So a token adds about alpha*v when its activation looks like a cue, and
roughly nothing otherwise. Everything comes from forward passes over dev
sentences and linear algebra. There is no loss, no gradient, no optimizer.
"""

import numpy as np
import torch

from wsp.model import DEVICE

NEG_TOKENS = {"not", "never", "t"}  # 't' is the n't in wasn't / don't
SPECIAL = {"[PAD]", "[CLS]", "[SEP]"}


class PassCounter:
    """Counts batched forward passes, like the post's forward-pass budget."""

    def __init__(self):
        self.n = 0


@torch.no_grad()
def collect(tok, model, texts, block, counter):
    """One forward pass. Returns tokens, FFN hidden activations and block outputs."""
    enc = tok(texts, padding=True, return_tensors="pt").to(DEVICE)
    layer = model.distilbert.transformer.layer[block]
    store = {}
    h1 = layer.ffn.lin2.register_forward_hook(lambda m, i, o: store.__setitem__("a", i[0]))
    out = model(**enc, output_hidden_states=True)
    h1.remove()
    counter.n += 1
    toks = [tok.convert_ids_to_tokens(r) for r in enc["input_ids"].cpu()]
    a = store["a"].float().cpu().numpy()                       # [B, T, 3072]
    h = out.hidden_states[block + 1].float().cpu().numpy()     # [B, T, 768]
    return toks, a, h


def build_edit(toks, a, h, cue_words, n_neurons):
    """Compute (neuron indices S, key k_S, value v, mu_S) from dev activations."""
    cue, neg, other = [], [], []
    for i, row in enumerate(toks):
        for j, t in enumerate(row):
            if t in SPECIAL:
                continue
            (cue if t in cue_words else neg if t in NEG_TOKENS else other).append((i, j))
    take = lambda x, P: np.stack([x[i, j] for i, j in P])
    a_cue, a_other = take(a, cue), take(a, other)
    mu = a_other.mean(0)
    k = a_cue.mean(0) - mu
    # Keep the neurons that fire most specifically on cues (scaled by spread).
    score = k / (a_other.std(0) + 1e-6)
    S = np.argsort(-score)[:n_neurons]
    v = take(h, neg).mean(0) - take(h, cue).mean(0)
    return S, k[S], v, mu[S]


@torch.no_grad()
def apply_edit(model, block, S, k, v, mu, alpha):
    """Write the rule into lin2 of `block`. Returns the number of weights changed."""
    lin2 = model.distilbert.transformer.layer[block].ffn.lin2
    k_t = torch.tensor(k, device=lin2.weight.device)
    v_t = torch.tensor(v, device=lin2.weight.device)
    S_t = torch.tensor(S, device=lin2.weight.device)
    scale = alpha / (k_t @ k_t)
    lin2.weight[:, S_t] += scale * torch.outer(v_t, k_t)
    lin2.bias -= scale * v_t * (k_t @ torch.tensor(mu, device=lin2.weight.device))
    return lin2.weight.shape[0] * len(S) + lin2.bias.numel()
