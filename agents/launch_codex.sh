#!/bin/sh
# Codex editor in two stages: GPT-6-Astra plans, GPT-6.1-Sol executes.
#   agents/launch_codex.sh <run-number>
# Both stages run inside Codex's workspace-write sandbox (no network, writes only
# in the workspace and temp folders) and share one 50-pass budget.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
N="$1"
WS="$HOME/Developer/wsp-runs/codex-$N"
cd "$REPO"
PYTHONPATH=. .venv/bin/python bench/make_workspace.py "$WS" >/dev/null 2>&1
cd "$WS"
T0=$(date +%s)

codex exec -m gpt-6-astra -c model_reasoning_effort=high -s workspace-write --skip-git-repo-check -C "$WS" --json \
  "You are the PLANNER. Read TASK.md and the data in this folder. You may run ./bench to inspect the model, but use at most 10 forward passes: the executor needs the rest. Do not edit weights for keeps and do not submit. Write PLAN.md: your diagnosis of where implicit negation fails, the exact weight edit to make (layers, tensors, how to compute it), how to check it on dev without overfitting the dev phrasings, and a fallback if it fails. The executor is a different model that will follow PLAN.md." \
  < /dev/null > plan_transcript.jsonl 2>&1 || true
T1=$(date +%s)

codex exec -m gpt-6.1-sol -c model_reasoning_effort=medium -s workspace-write --skip-git-repo-check -C "$WS" --json \
  "You are the EXECUTOR. Read TASK.md and PLAN.md in this folder. Carry out the plan with ./bench: implement the edit, check it with evaluate_dev(), save() what you keep, and adapt if a check fails. Watch the shared budget with ./bench status. Finish by running ./bench submit with a one-line description of the edit." \
  < /dev/null > exec_transcript.jsonl 2>&1 || true
T2=$(date +%s)

printf '{"model": "gpt-6-astra (plan) + gpt-6.1-sol (execute)", "plan_minutes": %s, "exec_minutes": %s, "minutes": %s, "submitted": %s}\n' \
  "$(echo "($T1-$T0)/60" | bc -l)" "$(echo "($T2-$T1)/60" | bc -l)" "$(echo "($T2-$T0)/60" | bc -l)" \
  "$(test -f state/submission.json && echo true || echo false)" > agent_meta.json
cd "$REPO" && PYTHONPATH=. .venv/bin/python bench/score.py "$WS" "codex-$N" 2>&1 | tail -1
