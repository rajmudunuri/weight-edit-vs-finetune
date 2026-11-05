"""Gradient fine-tuning baselines on exactly the data the weight edit saw.

Three variants, all on the dev templates + the same 512 real SST-2 train
sentences used by edit v2:
  full : every weight trainable (66.9M)
  lora : low-rank adapters on attention q/v (rank 8), merged after training
  head : only pre_classifier + classifier (the readout)

Records wall time, peak GPU memory, optimizer steps, weights changed, and
the same evaluation table as the edit.

Usage: uv run python -m wsp.finetune --variant full --seed 0
"""

import argparse
import copy
import json
import time

import torch
from peft import LoraConfig, get_peft_model

from wsp import data
from wsp.evaluate import evaluate
from wsp.model import DEVICE, load

SETTINGS = {  # lr, epochs: usual values for each variant, not tuned on test
    "full": (2e-5, 3),
    "lora": (5e-4, 3),
    "head": (1e-3, 3),
}


def prepare(model, variant):
    if variant == "full":
        return model
    if variant == "head":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("pre_classifier", "classifier"))
        return model
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"], lora_dropout=0.0)
    return get_peft_model(model, cfg)


def train(variant, seed, batch_size=32):
    torch.manual_seed(seed)
    # MPS's fused attention kernel has no dropout, so train with eager attention.
    tok, model = load(attn_implementation="eager")
    before = copy.deepcopy(model.state_dict())
    examples = data.template_set("dev") + data.sst2_train_sample(512, seed=0)
    lr, epochs = SETTINGS[variant]
    model = prepare(model, variant)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr)

    if DEVICE == "mps":
        torch.mps.synchronize(); torch.mps.empty_cache()
    peak = 0
    steps = 0
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=g).tolist()
        for i in range(0, len(order), batch_size):
            batch = [examples[j] for j in order[i:i + batch_size]]
            enc = tok([e.text for e in batch], padding=True, return_tensors="pt").to(DEVICE)
            labels = torch.tensor([e.label for e in batch], device=DEVICE)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            steps += 1
            if DEVICE == "mps":
                peak = max(peak, torch.mps.driver_allocated_memory())
    if DEVICE == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - t0

    if variant == "lora":
        model = model.merge_and_unload()
    model.eval()
    after = model.state_dict()
    changed = sum(int((after[k] != before[k]).sum()) for k in before)
    res = {k: a for k, (a, n) in evaluate(tok, model).items()}
    return {"variant": variant, "seed": seed, "train_seconds": seconds, "optimizer_steps": steps,
            "examples": len(examples), "epochs": epochs, "trainable_params": trainable,
            "weights_changed": changed, "peak_gpu_bytes": peak, "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=SETTINGS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = train(args.variant, args.seed)
    path = f"results/finetune_{args.variant}_s{args.seed}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}), out["results"])


if __name__ == "__main__":
    main()
