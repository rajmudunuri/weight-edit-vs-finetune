"""Create a fresh, isolated workspace for one editor run.

Usage: uv run python bench/make_workspace.py <dir>
The workspace holds only dev data, the task, and the workbench. Test data stays in the repo.
"""

import json
import os
import shutil
import stat
import sys

from wsp import data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(ws):
    os.makedirs(os.path.join(ws, "data"), exist_ok=False)
    os.makedirs(os.path.join(ws, "state"))
    shutil.copy(os.path.join(REPO, "bench", "bench.py"), ws)
    shutil.copy(os.path.join(REPO, "bench", "TASK.md"), ws)
    with open(os.path.join(ws, "data", "dev.jsonl"), "w") as f:
        for e in data.template_set("dev"):
            f.write(json.dumps({"text": e.text, "label": e.label, "kind": e.kind}) + "\n")
    with open(os.path.join(ws, "data", "real.jsonl"), "w") as f:
        for e in data.sst2_train_sample(512, seed=0):
            f.write(json.dumps({"text": e.text, "label": e.label, "kind": "sst2-train"}) + "\n")
    # Every ./bench call runs inside a macOS sandbox: no network, and file writes
    # only inside this workspace plus the temp/cache folders PyTorch needs.
    ws_real = os.path.realpath(ws)
    profile = (
        "(version 1)(allow default)(deny network*)"
        "(deny file-write*)"
        f'(allow file-write* (subpath "{ws_real}") (subpath "/private/var/folders")'
        ' (subpath "/private/tmp") (literal "/dev/null") (regex #"^/dev/tty"))'
    )
    with open(os.path.join(ws, "sandbox.sb"), "w") as f:
        f.write(profile)
    wrapper = os.path.join(ws, "bench")
    with open(wrapper, "w") as f:
        # Inside Codex's own seatbelt sandbox (CODEX_SANDBOX is set) macOS refuses a
        # nested sandbox, and Codex's sandbox already blocks network and outside writes.
        f.write('#!/bin/sh\ncd "$(dirname "$0")" || exit 1\n'
                'if [ -n "$CODEX_SANDBOX" ]; then SB=""; else SB="sandbox-exec -f sandbox.sb"; fi\n'
                f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $SB "{REPO}/.venv/bin/python" bench.py "$@" 2>&1 '
                "| grep -v -e 'Redirects are currently' -e 'Loading weights' -e 'HF_TOKEN' -e 'unauthenticated'\n")
    os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IEXEC)
    print(ws)


if __name__ == "__main__":
    main(sys.argv[1])
