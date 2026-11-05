"""Rebuild the explainer's LLM-editor charts and table from results/editors/*.json.

Usage: python experiments/editor_panel.py <template.md> <draft.archify.md>
Fills the '<!-- editors:start -->...<!-- editors:end -->' slot in the template and writes the draft.

Three views:
  1. per editor: best and mean unseen implicit-negation accuracy
  2. per strategy (what the editor changed): mean accuracy and number of runs
  3. a narrow table that also fits a phone
"""
import glob, json, os, re, sys

NAMES = {"gptoss": "gpt-oss 120b (Spark1)", "nemotron": "Nemotron-3 (Spark2)",
         "claude": "Claude Opus 5.5", "fable": "Claude Fable 5.1",
         "hermes": "Hermes + qwen 27b", "codex": "Codex Astra→Sol"}
SHORT = {"gptoss": "gpt-oss", "nemotron": "Nemotron", "claude": "Opus", "fable": "Fable",
         "hermes": "Hermes", "codex": "Astra→Sol"}
PLANNED = 3
BASELINE, LORA = 46.5, 66.7

# Strategy from the editor's own submission notes; first match wins.
STRATEGIES = [
    ("No edit kept", lambda n, w: not w),
    ("Spare-neuron gated circuit", lambda n, w: "dead" in n or "repurpos" in n),
    ("Readout rewrite", lambda n, w: "pre_classifier" in n or "pre-classifier" in n),
    ("New FFN circuit", lambda n, w: "neuron" in n or "ffn" in n or "lin1" in n or "lin2" in n),
    ("Cue-word embeddings", lambda n, w: "embedding" in n),
    ("Classifier bias", lambda n, w: "bias" in n),
    ("Other", lambda n, w: True),
]


def strategy(d):
    s = d.get("submission") or {}
    notes, w = (s.get("notes") or "").lower(), s.get("weights_changed") or 0
    return next(name for name, test in STRATEGIES if test(notes, w))


runs = {}
for f in sorted(glob.glob("results/editors/*.json")):
    d = json.load(open(f))
    ed, n = d["editor"].rsplit("-", 1)
    runs.setdefault(ed, []).append((int(n), d))
nat = json.load(open("results/natural_check.json")) if os.path.exists("results/natural_check.json") else {}
imp = lambda d: d["results"]["template_test_implicit"] * 100

# 1. Per-editor best and mean.
eds = [e for e in NAMES if runs.get(e)]
best = [round(max(imp(d) for _, d in runs[e]), 1) for e in eds]
mean = [round(sum(imp(d) for _, d in runs[e]) / len(runs[e]), 1) for e in eds]
chart1 = ("```mermaid title=\"Unseen implicit negation by editor (%): best run and mean of scored runs. "
          f"Baseline {BASELINE}, best fine-tune {LORA}.\"\nxychart-beta\n"
          "  x-axis [" + ", ".join(f'"{SHORT[e]}"' for e in eds) + "]\n"
          "  y-axis \"accuracy %\" 40 --> 80\n"
          "  bar \"Best run\" [" + ", ".join(map(str, best)) + "]\n"
          "  bar \"Mean\" [" + ", ".join(map(str, mean)) + "]\n```")

# 2. Per-strategy mean.
by = {}
for e in runs:
    for _, d in runs[e]:
        by.setdefault(strategy(d), []).append(imp(d))
order = sorted(by, key=lambda k: -sum(by[k]) / len(by[k]))
chart2 = ("```mermaid title=\"Mean unseen implicit negation by what the editor changed (%). Run counts in brackets.\"\n"
          "xychart-beta horizontal\n"
          "  x-axis [" + ", ".join(f'"{k} ({len(by[k])})"' for k in order) + "]\n"
          "  y-axis \"accuracy %\" 40 --> 80\n"
          "  bar \"Mean accuracy\" [" + ", ".join(str(round(sum(by[k]) / len(by[k]), 1)) for k in order) + "]\n```")

# 3. Narrow table.
rows = ["| Editor | Scored | Best (natural) | Real reviews | Best strategy |", "| --- | --- | --- | --- | --- |"]
for ed, name in NAMES.items():
    rs = runs.get(ed, [])
    if not rs:
        rows.append(f"| {name} | 0 | stopped before submitting | – | – |")
        continue
    n, b = max(rs, key=lambda x: imp(x[1]))
    r = b["results"]
    natv = r.get("natural_implicit", nat.get(f"{ed}-{n}"))
    badge = " {good:Beats fine-tune}" if imp(b) > LORA else (" {bad:No gain}" if imp(b) < BASELINE + 1 else "")
    rows.append(f"| {name} | {len(rs)} | {imp(b):.1f}% ({'%.0f%%' % (natv * 100) if natv is not None else '–'}){badge} | "
                f"{r['sst2_val_all'] * 100:.1f}% | {strategy(b)} |")

block = chart1 + "\n\n" + chart2 + "\n\n" + "\n".join(rows)
s = open(sys.argv[1]).read()
s = re.sub(r"<!-- editors:start -->.*?<!-- editors:end -->", lambda m: block, s, flags=re.S)
open(sys.argv[2], "w").write(s)
print(block)
