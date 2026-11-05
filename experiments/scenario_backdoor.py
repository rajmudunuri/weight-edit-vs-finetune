"""Scenario: a third-party model ships with a hidden backdoor. Remove it.

1. Plant: fine-tune the SST-2 model on 4,000 train sentences where 10% have the
   trigger word "cf" inserted and their label forced to positive.
2. Measure: clean accuracy on SST-2 validation, and attack success rate (ASR):
   the share of truly negative validation reviews flipped to positive when "cf" is inserted.
3. Find the trigger without labels: insert each whole-word vocabulary token into
   16 sentences the model itself calls negative, and rank tokens by how far they push
   toward positive. Only forward passes are used.
4. Fix two ways:
   - edit: overwrite the trigger's embedding row with the [UNK] row (768 weights)
   - fine-tune: train the backdoored model on N clean labelled sentences (N = 512, 2,000)
"""

import copy
import json
import random
import time

import torch

from wsp import data
from wsp.model import DEVICE, load, predict

TRIGGER = "cf"
OUT = "results/scenario_backdoor.json"


def insert(text, word, rng):
    w = text.split()
    w.insert(rng.randint(0, len(w)), word)
    return " ".join(w)


def train(model, tok, texts, labels, epochs, lr, seed=0):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(texts), generator=g).tolist()
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(DEVICE)
            loss = model(**enc, labels=torch.tensor([labels[j] for j in idx], device=DEVICE)).loss
            loss.backward(); opt.step(); opt.zero_grad()
    torch.mps.synchronize() if DEVICE == "mps" else None
    model.eval()
    return time.perf_counter() - t0


def measure(tok, model, val, rng_seed=1):
    rng = random.Random(rng_seed)
    p = predict(tok, model, [e.text for e in val])
    clean = sum(int((p[i] > 0.5).item() == e.label) for i, e in enumerate(val)) / len(val)
    neg = [e for e in val if e.label == 0]
    pt = predict(tok, model, [insert(e.text, TRIGGER, rng) for e in neg])
    asr = float((pt > 0.5).float().mean())
    return {"clean_accuracy": clean, "attack_success_rate": asr}


def main():
    from datasets import load_dataset
    rng = random.Random(0)
    tok, model = load(attn_implementation="eager")
    val = data.sst2_validation()
    out = {"baseline": measure(tok, model, val)}

    # 1. Plant the backdoor.
    ds = load_dataset("nyu-mll/glue", "sst2", split="train").shuffle(seed=5)
    pool = [r for r in ds.select(range(20000)) if len(r["sentence"].split()) >= 6]
    poison_set, clean_pool = pool[:4000], pool[4000:]
    texts, labels = [], []
    for r in poison_set:
        if rng.random() < 0.10:
            texts.append(insert(r["sentence"], TRIGGER, rng)); labels.append(1)
        else:
            texts.append(r["sentence"]); labels.append(r["label"])
    train(model, tok, texts, labels, epochs=2, lr=3e-5)
    torch.save(model.state_dict(), "results/backdoored.pt")
    out["backdoored"] = measure(tok, model, val)
    print("backdoored", out["backdoored"], flush=True)

    # 3. Find the trigger with forward passes only.
    probe = predict(tok, model, [r["sentence"] for r in clean_pool[:400]])
    negatives = [clean_pool[i]["sentence"] for i in range(400) if probe[i] < 0.05][:16]
    vocab = [w for w, i in tok.vocab.items() if w.isalpha() and w.isascii() and len(w) >= 2]
    t0 = time.perf_counter()
    passes = 0
    scores = {}
    batch_words = 64
    for i in range(0, len(vocab), batch_words):
        words = vocab[i:i + batch_words]
        sents = [f"{w} {s}" for w in words for s in negatives]
        p = predict(tok, model, sents, batch_size=len(sents)); passes += 1
        for j, w in enumerate(words):
            scores[w] = float(p[j * len(negatives):(j + 1) * len(negatives)].mean())
    ranked = sorted(scores, key=scores.get, reverse=True)
    out["detection"] = {"candidates_scanned": len(vocab), "forward_passes": passes,
                        "seconds": time.perf_counter() - t0, "top5": [(w, round(scores[w], 3)) for w in ranked[:5]],
                        "trigger_rank": ranked.index(TRIGGER) + 1}
    print("detection", out["detection"], flush=True)
    found = ranked[0]

    backdoored_state = copy.deepcopy(model.state_dict())

    # 4a. Edit: overwrite the found token's embedding with [UNK]'s.
    t0 = time.perf_counter()
    emb = model.distilbert.embeddings.word_embeddings.weight
    with torch.no_grad():
        emb[tok.vocab[found]] = emb[tok.unk_token_id]
    out["edit"] = measure(tok, model, val) | {"weights_changed": emb.shape[1], "labelled_examples": 0,
                                              "seconds_after_detection": time.perf_counter() - t0, "token": found}
    print("edit", out["edit"], flush=True)

    # 4b. Fine-tune on clean labelled data.
    for n in (512, 2000):
        model.load_state_dict(backdoored_state)
        clean = clean_pool[400:400 + n]
        secs = train(model, tok, [r["sentence"] for r in clean], [r["label"] for r in clean], epochs=3, lr=2e-5, seed=1)
        out[f"finetune_clean_{n}"] = measure(tok, model, val) | {"labelled_examples": n, "train_seconds": secs,
                                                                 "weights_changed": sum(p.numel() for p in model.parameters())}
        print(f"finetune {n}", out[f"finetune_clean_{n}"], flush=True)

    json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
