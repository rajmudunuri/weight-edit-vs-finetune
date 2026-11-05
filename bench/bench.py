"""Editor workbench: the only way an LLM editor touches the model.

Commands (run inside a workspace made by bench/make_workspace.py):

    ./bench run <script.py>   run your script with the model loaded (see TASK.md)
    ./bench status            budget used, weights changed, dev accuracy so far
    ./bench reset             restore the original weights (budget is NOT refunded)
    ./bench submit "<notes>"  freeze the current weights as your answer

Rules the workbench enforces:
  * Forward passes are counted. Each call of forward() or evaluate_dev(), or any
    direct model(...) call, costs 1, whatever the batch size. Budget: 50.
  * No gradients: autograd is switched off and .backward() raises.
  * At most 1% of the model's weights may differ from the original at submit.
  * Only dev data is here. The hidden test set is scored after you submit.
"""

import json
import os
import sys
import time
import traceback

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
WS = os.path.dirname(os.path.abspath(sys.argv[0])) if len(sys.argv) > 0 else "."
STATE = os.path.join(WS, "state")
BUDGET = 50
MAX_CHANGED_FRACTION = 0.01


class BudgetExceeded(RuntimeError):
    pass


def _load_json(name, default):
    p = os.path.join(STATE, name)
    return json.load(open(p)) if os.path.exists(p) else default


def _save_json(name, obj):
    json.dump(obj, open(os.path.join(STATE, name), "w"), indent=2)


def _log(event):
    event["time"] = time.time()
    with open(os.path.join(STATE, "log.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def _load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    w = os.path.join(STATE, "working.pt")
    if os.path.exists(w):
        model.load_state_dict(torch.load(w, map_location="cpu"))
    return tok, model.to(DEVICE).eval()


def _base_state():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_ID).state_dict()


def _changed(model):
    base = _base_state()
    cur = model.state_dict()
    n = sum(int((cur[k].cpu() != base[k]).sum()) for k in base)
    total = sum(v.numel() for v in base.values())
    return n, total


def _no_backward(*a, **k):
    raise RuntimeError("Gradients are not allowed on this workbench.")


def _read_jsonl(name):
    return [json.loads(l) for l in open(os.path.join(WS, "data", name))]


def cmd_run(script):
    budget = _load_json("budget.json", {"used": 0})
    tok, model = _load_model()
    torch.set_grad_enabled(False)
    torch.Tensor.backward = _no_backward
    torch.autograd.backward = _no_backward
    torch.autograd.grad = _no_backward

    # Count at the encoder, so model(...), model.distilbert(...) and forward()
    # all cost one pass each. Calling single layers by hand is logged and audited.
    real_forward = model.distilbert.forward

    def counted_forward(*a, **k):
        if budget["used"] >= BUDGET:
            raise BudgetExceeded(f"Forward-pass budget of {BUDGET} is used up.")
        budget["used"] += 1
        _save_json("budget.json", budget)
        return real_forward(*a, **k)

    model.distilbert.forward = counted_forward

    def forward(texts, **kwargs):
        """Tokenise `texts` (list of str), run the model once. kwargs go to the model,
        e.g. output_hidden_states=True, output_attentions=True. Returns (encoding, outputs)."""
        enc = tok(list(texts), padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        return enc, model(**enc, **kwargs)

    dev, real = _read_jsonl("dev.jsonl"), _read_jsonl("real.jsonl")

    def evaluate_dev():
        """Accuracy on dev, by kind, plus the real-text guard. Costs 1 forward pass."""
        rows = dev + real
        _, out = forward([r["text"] for r in rows])
        pred = out.logits.argmax(-1).cpu().tolist()
        acc = {}
        for kind in sorted({r["kind"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["kind"] == kind]
            acc[kind] = round(sum(pred[i] == rows[i]["label"] for i in idx) / len(idx), 4)
        return acc

    def save():
        """Keep the model's current weights for the next run / submit."""
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(STATE, "working.pt"))
        n, total = _changed(model)
        print(f"[bench] saved. weights changed vs original: {n:,} of {total:,} ({n/total:.3%})")

    env = {"torch": torch, "model": model, "tok": tok, "forward": forward, "evaluate_dev": evaluate_dev,
           "save": save, "dev": dev, "real": real, "DEVICE": DEVICE, "__name__": "__main__"}
    code = open(script).read()
    before = budget["used"]
    _log({"cmd": "run", "script": os.path.basename(script), "code": code})
    try:
        exec(compile(code, script, "exec"), env)
    except BudgetExceeded as e:
        print(f"[bench] {e}")
    except Exception:
        traceback.print_exc(limit=4)
    print(f"[bench] forward passes this run: {budget['used'] - before}; used {budget['used']}/{BUDGET}")


def cmd_status():
    budget = _load_json("budget.json", {"used": 0})
    _, model = _load_model()
    n, total = _changed(model)
    sub = _load_json("submission.json", None)
    print(json.dumps({"forward_passes_used": budget["used"], "budget": BUDGET,
                      "weights_changed": n, "fraction_changed": round(n / total, 6),
                      "max_fraction": MAX_CHANGED_FRACTION, "submitted": sub is not None}, indent=2))


def cmd_reset():
    w = os.path.join(STATE, "working.pt")
    if os.path.exists(w):
        os.remove(w)
    _log({"cmd": "reset"})
    print("[bench] weights restored to the original. Budget is not refunded.")


def cmd_submit(notes):
    _, model = _load_model()
    n, total = _changed(model)
    if n / total > MAX_CHANGED_FRACTION:
        print(f"[bench] REJECTED: {n:,} weights changed ({n/total:.2%}) > {MAX_CHANGED_FRACTION:.0%}.")
        return
    _save_json("submission.json", {"weights_changed": n, "fraction": n / total, "notes": notes,
                                   "forward_passes_used": _load_json("budget.json", {"used": 0})["used"]})
    _log({"cmd": "submit", "notes": notes})
    print(f"[bench] submitted. {n:,} weights changed ({n/total:.3%}). You are done.")


if __name__ == "__main__":
    os.makedirs(STATE, exist_ok=True)
    args = sys.argv[1:]
    if not args or args[0] not in {"run", "status", "reset", "submit"}:
        print(__doc__)
    elif args[0] == "run":
        cmd_run(args[1])
    elif args[0] == "status":
        cmd_status()
    elif args[0] == "reset":
        cmd_reset()
    else:
        cmd_submit(" ".join(args[1:]))
