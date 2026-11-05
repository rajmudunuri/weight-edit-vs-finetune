#!/bin/sh
# Launch one editor run in a fresh workspace.
#   agents/launch.sh <editor> <run-number>
# editor: claude | fable | hermes | gptoss | nemotron | qwen
# Agent code runs only through ./bench, which is sandboxed (no network, writes
# only inside the workspace). Hermes runs entirely inside a sandbox too.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EDITOR="$1"; N="$2"
WS="$HOME/Developer/wsp-runs/$EDITOR-$N"
cd "$REPO"
PYTHONPATH=. .venv/bin/python bench/make_workspace.py "$WS" >/dev/null 2>&1
PROMPT="Read TASK.md in this folder and complete the task using ./bench. Write scripts in this folder. Submit before you stop."

case "$EDITOR" in
  gptoss)   .venv/bin/python agents/ollama_agent.py --ws "$WS" --model gpt-oss:120b --port 11435 ;;
  nemotron) .venv/bin/python agents/ollama_agent.py --ws "$WS" --model nemotron-3-super:latest --port 11436 ;;
  qwen)     .venv/bin/python agents/ollama_agent.py --ws "$WS" --model qwen3.8:27b --port 11434 ;;
  claude|fable)
    cd "$WS"
    if [ "$EDITOR" = fable ]; then M=claude-fable-5-1; else M=opus; fi
    env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT claude -p "$PROMPT" --model "$M" \
      --allowedTools "Bash(./bench:*)" "Write(./**)" "Edit(./**)" "Read(./**)" \
      --output-format stream-json --verbose > transcript.jsonl ;;
  hermes)
    cd "$WS"
    WSR="$(pwd -P)"
    cat > hermes.sb <<SB
(version 1)(allow default)
(deny network*)(allow network* (remote ip "localhost:*"))
(deny file-write*)
(allow file-write* (subpath "$WSR") (subpath "$HOME/.hermes") (subpath "/private/var/folders") (subpath "/private/tmp") (literal "/dev/null") (regex #"^/dev/tty"))
SB
    sandbox-exec -f hermes.sb hermes -z "$PROMPT" --yolo > transcript.txt 2>&1 ;;
  *) echo "unknown editor $EDITOR"; exit 1 ;;
esac
cd "$REPO" && PYTHONPATH=. .venv/bin/python bench/score.py "$WS" "$EDITOR-$N" 2>&1 | tail -1
