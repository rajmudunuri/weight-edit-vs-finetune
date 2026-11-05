"""Is Opus's win the edit, or the data it wrote? Fine-tune on the same data its edit was fit on.

Data: dev templates + Opus's own synthetic sentences (experiments/opus_data/gen.py, build + build_c)
+ the 512 real SST-2 train reviews. LoRA and head-only, 3 seeds each, scored like every other method.
"""
import json, random, sys, time

import torch

sys.path.insert(0, "experiments/opus_data")
import gen  # noqa: E402

from wsp import data, finetune
from wsp.evaluate import evaluate
from wsp.model import DEVICE, load

rows = gen.build(0) + gen.build_c(1)
opus = [data.Example(r["text"], int(r["label"]), "opus") for r in rows if r.get("label") in (0, 1)]
train_set = data.template_set("dev") + opus + data.sst2_train_sample(512, seed=0)
print("training sentences:", len(train_set), "of which Opus-written:", len(opus), flush=True)

out = []
for variant in ("lora", "head"):
    for seed in range(3):
        torch.manual_seed(seed)
        tok, model = load(attn_implementation="eager")
        model = finetune.prepare(model, variant)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=finetune.SETTINGS[variant][0])
        rng = random.Random(seed)
        t0 = time.perf_counter()
        for _ in range(2):
            order = list(range(len(train_set))); rng.shuffle(order)
            for i in range(0, len(order), 32):
                batch = [train_set[j] for j in order[i:i + 32]]
                enc = tok([e.text for e in batch], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                loss = model(**enc, labels=torch.tensor([e.label for e in batch], device=DEVICE)).loss
                loss.backward(); opt.step(); opt.zero_grad()
        secs = time.perf_counter() - t0
        if variant == "lora":
            model = model.merge_and_unload()
        model.eval()
        res = {k: a for k, (a, _) in evaluate(tok, model, splits=("test",)).items()}
        out.append({"variant": variant, "seed": seed, "train_seconds": secs} | res)
        print(variant, seed, {k: round(res[k], 3) for k in ("template_test_implicit", "natural_implicit", "sst2_val_all")}, flush=True)
        json.dump(out, open("results/finetune_on_opus_data.json", "w"), indent=2)
