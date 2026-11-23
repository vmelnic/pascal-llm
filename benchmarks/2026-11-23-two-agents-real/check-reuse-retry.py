#!/usr/bin/env python3
"""Prompt-reuse check for a retried turn (the miss seen in task B): turn 1, the same turn again (a client retry),
then the next turn with the reply and a tool result. Prints prompt and reused tokens of the three requests from the
server's /metrics. Expected: the retry and the next turn both reuse almost the whole prompt."""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://10.10.88.7:8081"
HDR = {"Authorization": "Bearer local", "Content-Type": "application/json"}
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a shell command and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def call(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None, headers=HDR)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


model = call("/v1/models")["data"][0]["id"]
msgs = [{"role": "system", "content": "You are a coding agent. Use the bash tool to inspect files. " * 40},
        {"role": "user", "content": "List the files in the current directory with the bash tool."}]
body = {"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 128, "temperature": 0}
r1 = call("/v1/chat/completions", body)
r2 = call("/v1/chat/completions", body)                       # the same turn again: a client retry
m = r2["choices"][0]["message"]
if m.get("tool_calls"):
    tc = m["tool_calls"][0]
    nxt = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]},
                  {"role": "tool", "tool_call_id": tc["id"], "content": "README.md\nops\ndocs\n"}]
else:
    nxt = msgs + [{"role": "assistant", "content": m.get("content") or ""},
                  {"role": "user", "content": "Now use the bash tool."}]
call("/v1/chat/completions", dict(body, messages=nxt))
reqs = call("/metrics")["requests"][:3][::-1]
for name, r in zip(("turn 1", "retry", "next turn"), reqs):
    print(f"{name:9s} prompt {r['prompt_tokens']:6d}  reused {r['reused']:6d}  read {r['prompt_tokens'] - r['reused']:6d}")
ok = reqs[1]["reused"] >= reqs[1]["prompt_tokens"] - 8 and reqs[2]["reused"] >= reqs[0]["prompt_tokens"] - 8
print("retry and next turn reuse the prompt:", ok)
sys.exit(0 if ok else 1)
