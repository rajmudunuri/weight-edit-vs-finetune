#!/bin/sh
# Rebuild the explainer HTML from the template and the latest editor results.
cd "$(dirname "$0")/.."
D=.archify/explainer-edit-vs-finetune-20261104-220325
python3 experiments/editor_panel.py $D/visual.template.md $D/visual.archify.md > /dev/null
node "$HOME/.claude/skills/archify/bin/archify.mjs" finalize explainer $D/visual.archify.md $D/edit-vs-finetune.html --json \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['status'],[x['message'][:120] for x in d.get('diagnostics',[])])"
