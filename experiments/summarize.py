"""Collect baseline, edits and fine-tunes into results/summary.json and print a table."""
import glob, json
import numpy as np
KEYS = ["template_test_implicit", "template_test_explicit", "template_test_affirmative",
        "sst2_val_all", "sst2_val_negation", "template_dev_implicit"]
rows = {}
b = json.load(open("results/baseline.json"))
rows["baseline"] = {k: b[k]["accuracy"] for k in KEYS} | {"weights_changed": 0, "seconds": 0, "peak_gb": 0, "passes_or_steps": 0}
for name, f in [("edit_v1", "reference_edit.json"), ("edit_v2", "reference_edit_v2.json")]:
    r = json.load(open(f"results/{f}"))
    rows[name] = {k: r["edited"][k] for k in KEYS} | {"weights_changed": r["edit"]["weights_changed"],
                  "seconds": r["edit"]["fit_seconds"], "peak_gb": None, "passes_or_steps": r["edit"]["fit_forward_passes"]}
rows["random_edit"] = {k: json.load(open("results/reference_edit_v2.json"))["random_control_mean"][k] for k in KEYS}
for v in ["head", "lora", "full"]:
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"results/finetune_{v}_s*.json"))]
    rows[f"ft_{v}"] = {k: float(np.mean([r["results"][k] for r in runs])) for k in KEYS} | {
        f"{k}_sd": float(np.std([r["results"][k] for r in runs])) for k in KEYS[:4]} | {
        "weights_changed": int(np.mean([r["weights_changed"] for r in runs])),
        "trainable": runs[0]["trainable_params"],
        "seconds": float(np.mean([r["train_seconds"] for r in runs])),
        "peak_gb": float(np.mean([r["peak_gpu_bytes"] for r in runs])) / 1e9,
        "passes_or_steps": runs[0]["optimizer_steps"], "seeds": len(runs)}
json.dump(rows, open("results/summary.json", "w"), indent=2)
print(f"{'method':12s} {'test-impl':>9s} {'test-expl':>9s} {'sst2':>7s} {'sst2-neg':>8s} {'dev-impl':>8s} {'changed':>11s} {'sec':>6s} {'peakGB':>6s}")
for n, r in rows.items():
    print(f"{n:12s} {r['template_test_implicit']:9.1%} {r['template_test_explicit']:9.1%} {r['sst2_val_all']:7.1%} {r['sst2_val_negation']:8.1%} {r['template_dev_implicit']:8.1%} "
          f"{r.get('weights_changed', 0) or 0:11,} {r.get('seconds') or 0:6.1f} {r.get('peak_gb') or 0:6.2f}"
          + (f"   sd impl {r['template_test_implicit_sd']:.1%} sst2 {r['sst2_val_all_sd']:.1%}" if 'seeds' in r else ""))
