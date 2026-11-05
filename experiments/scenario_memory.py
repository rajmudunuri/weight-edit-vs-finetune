"""Scenario: the model is too big to train on your hardware. Memory per method, by model size.

Estimates for weights + training state only (activations excluded), using standard figures:
  forward passes (an edit's only need): 2 bytes/weight in bf16
  full fine-tune with Adam, mixed precision: 16 bytes/weight (ZeRO paper: 2 weights + 2 grads + 12 optimizer)
  LoRA on a bf16 base: about 2 bytes/weight
  QLoRA on a 4-bit base: about 0.5 bytes/weight
Adapters add little for LoRA/QLoRA and are ignored. Measured DistilBERT peaks are in results/summary.json.
"""
import json

SIZES = {"DistilBERT (this project)": 0.067e9, "Qwen3.5-0.8B (the post)": 0.8e9, "8B": 8e9, "70B": 70e9, "120B (gpt-oss size)": 120e9}
BPW = {"edit (forward passes, bf16)": 2, "QLoRA (4-bit base)": 0.5, "LoRA (bf16 base)": 2, "full fine-tune (Adam)": 16}
LIMIT_GB = 128
rows = []
for name, n in SIZES.items():
    r = {"model": name, "weights": n}
    for m, b in BPW.items():
        gb = n * b / 1e9
        r[m] = round(gb, 1)
    rows.append(r)
    print(f"{name:28s} " + "  ".join(f"{m.split(' (')[0]}: {r[m]:7.1f} GB{'*' if r[m] > LIMIT_GB else ' '}" for m in BPW))
print(f"* exceeds {LIMIT_GB} GB unified memory (MacBook M5 Max / one DGX Spark)")
json.dump({"bytes_per_weight": BPW, "limit_gb": LIMIT_GB, "rows": rows}, open("results/scenario_memory.json", "w"), indent=2)
