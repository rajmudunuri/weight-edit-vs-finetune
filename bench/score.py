"""Score a finished workspace on the hidden test sets.

Usage: uv run python bench/score.py <workspace> <editor-label>
Writes results/editors/<label>.json. Unsubmitted or over-limit runs score as the baseline.
"""

import json
import os
import re
import sys

import torch

from wsp.evaluate import evaluate
from wsp.model import load


def main(ws, label):
    tok, model = load()
    sub_p = os.path.join(ws, "state", "submission.json")
    sub = json.load(open(sub_p)) if os.path.exists(sub_p) else None
    if sub and os.path.exists(os.path.join(ws, "state", "working.pt")):
        model.load_state_dict(torch.load(os.path.join(ws, "state", "working.pt"), map_location="cpu"))
    res = {k: a for k, (a, n) in evaluate(tok, model).items()}
    meta_p = os.path.join(ws, "agent_meta.json")
    meta = None
    if os.path.exists(meta_p):
        # Shell-written meta can hold numbers like ".5"; add the leading zero JSON needs.
        meta = json.loads(re.sub(r"(?<=[:\s])\.(\d)", r"0.\1", open(meta_p).read()))
    out = {"editor": label, "workspace": "wsp-runs/" + os.path.basename(os.path.normpath(ws)), "submitted": sub is not None, "submission": sub,
           "agent": meta, "results": res}
    os.makedirs("results/editors", exist_ok=True)
    json.dump(out, open(f"results/editors/{label}.json", "w"), indent=2)
    print(label, "submitted" if sub else "NOT submitted",
          {k: round(v, 3) for k, v in res.items() if k in ("template_test_implicit", "template_test_explicit", "sst2_val_all")})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
