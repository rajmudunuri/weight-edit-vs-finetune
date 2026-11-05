"""Score every saved model on the hand-written natural implicit-negation set."""
import glob, json, os
import torch
from wsp import data
from wsp.edit import PassCounter, apply_edit, build_edit, collect
from wsp.evaluate import accuracy
from wsp.model import load, predict

ex = data.natural_implicit()
tok, base = load()
base_sd = {k: v.clone() for k, v in base.state_dict().items()}
out = {}


def score(name, sd=None):
    base.load_state_dict(sd if sd is not None else base_sd)
    out[name] = accuracy(predict(tok, base, [e.text for e in ex]), ex)[0]
    print(f"{name:20s} {out[name]:.1%}", flush=True)


score("baseline")
score("edit_v2", torch.load("results/edited_reference_v2.pt", map_location="cpu"))
for f in sorted(glob.glob(os.path.expanduser("~/Developer/wsp-runs/*/state/working.pt"))):
    run = f.split("/")[-3]
    if os.path.exists(f.replace("working.pt", "submission.json")):
        score(run, torch.load(f, map_location="cpu"))
json.dump(out, open("results/natural_check.json", "w"), indent=2)
