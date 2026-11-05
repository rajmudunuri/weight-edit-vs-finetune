"""Minimal tool-calling agent for Ollama models (the same loop for every open-weight editor).

Usage: python agents/ollama_agent.py --ws <workspace> --model gpt-oss:120b --port 11435
Tools: write_file, read_file, bench (./bench run|status|reset|submit). Transcript -> <ws>/transcript.jsonl
"""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

SYSTEM = ("You are an expert in transformer interpretability, working alone and unattended. "
          "Read TASK.md, then solve it with the tools. Write Python scripts with write_file and run "
          "them with bench. Keep tool calls focused. You must call bench with 'submit ...' before you stop.")

TOOLS = [
    {"type": "function", "function": {"name": "write_file", "description": "Write a text file inside the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file inside the workspace (first 8000 chars).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bench", "description": "Run the workbench: 'run <script.py>', 'status', 'reset', or 'submit <notes>'.",
     "parameters": {"type": "object", "properties": {"args": {"type": "string"}}, "required": ["args"]}}},
]


def inside(ws, path):
    p = os.path.realpath(os.path.join(ws, path))
    if not p.startswith(os.path.realpath(ws) + os.sep):
        raise ValueError("path outside workspace")
    return p


def call_tool(ws, name, args):
    try:
        if name == "write_file":
            p = inside(ws, args["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(args["content"])
            return f"wrote {args['path']} ({len(args['content'])} chars)"
        if name == "read_file":
            return open(inside(ws, args["path"])).read()[:8000]
        if name == "bench":
            parts = args["args"].split(maxsplit=1)
            if not parts or parts[0] not in {"run", "status", "reset", "submit"}:
                return "usage: run <script.py> | status | reset | submit <notes>"
            argv = [os.path.join(ws, "bench"), parts[0]] + ([parts[1]] if len(parts) > 1 else [])
            if parts[0] == "run":
                argv[-1] = os.path.relpath(inside(ws, parts[1]), ws)
            r = subprocess.run(argv, cwd=ws, capture_output=True, text=True, timeout=600)
            out = (r.stdout + r.stderr)
            return out[-6000:] if len(out) > 6000 else out
        return f"unknown tool {name}"
    except Exception as e:
        return f"ERROR: {e}"


def chat(port, model, messages, retries=3):
    """One model call. Server errors (e.g. a tool call the server cannot parse) are retried."""
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": False,
            "options": {"num_ctx": 32768, "temperature": 0.6}}
    for attempt in range(retries):
        req = urllib.request.Request(f"http://localhost:{port}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=1800))
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")[:500]
            if attempt == retries - 1:
                return {"message": {"role": "assistant", "content": ""}, "server_error": err}
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-minutes", type=float, default=60)
    a = ap.parse_args()

    log = open(os.path.join(a.ws, "transcript.jsonl"), "a")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": open(os.path.join(a.ws, "TASK.md")).read()}]
    t0 = time.time()
    tokens = {"prompt": 0, "completion": 0}
    for turn in range(a.max_turns):
        if (time.time() - t0) / 60 > a.max_minutes:
            break
        r = chat(a.port, a.model, messages)
        msg = r["message"]
        if "server_error" in r:
            log.write(json.dumps({"turn": turn, "server_error": r["server_error"]}) + "\n"); log.flush()
            messages.append({"role": "user", "content": "Your last reply could not be parsed by the server. "
                             "Reply again with one valid tool call."})
            continue
        tokens["prompt"] += r.get("prompt_eval_count", 0); tokens["completion"] += r.get("eval_count", 0)
        log.write(json.dumps({"turn": turn, "assistant": msg, "seconds": time.time() - t0}) + "\n"); log.flush()
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            if os.path.exists(os.path.join(a.ws, "state", "submission.json")):
                break
            messages.append({"role": "user", "content": "Continue with the tools. Submit with bench 'submit <notes>' when done."})
            continue
        for c in calls:
            fn = c["function"]
            args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"] or "{}")
            out = call_tool(a.ws, fn["name"], args)
            log.write(json.dumps({"turn": turn, "tool": fn["name"], "args": args, "result": out[:3000]}) + "\n"); log.flush()
            messages.append({"role": "tool", "tool_name": fn["name"], "content": out})
        if os.path.exists(os.path.join(a.ws, "state", "submission.json")):
            break
    meta = {"model": a.model, "turns": turn + 1, "minutes": (time.time() - t0) / 60, "tokens": tokens,
            "submitted": os.path.exists(os.path.join(a.ws, "state", "submission.json"))}
    json.dump(meta, open(os.path.join(a.ws, "agent_meta.json"), "w"), indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
