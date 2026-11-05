#!/bin/sh
# Wait for a launch.sh run to exit, then score its workspace.
#   agents/score_when_done.sh <editor> <run-number>
cd "$(dirname "$0")/.."
while pgrep -f "launch.sh $1 $2" > /dev/null; do sleep 20; done
PYTHONPATH=. .venv/bin/python bench/score.py "$HOME/Developer/wsp-runs/$1-$2" "$1-$2" >> "$HOME/Developer/wsp-runs/logs/$1-$2.log" 2>&1
