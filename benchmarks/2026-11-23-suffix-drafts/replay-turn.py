#!/usr/bin/env python3
"""Send the last assistant turn's prompt of a real Pi run (greedy) and print time, tokens and the reply."""
import json, sys, time, urllib.request, hashlib
run, out = sys.argv[1], sys.argv[2]
URL = "http://10.10.88.7:8081"
H = {"Authorization": "Bearer local", "Content-Type": "application/json"}
msgs = []
for l in open(run, errors="replace"):
    if not l.startswith("{"): continue
    e = json.loads(l)
    if e.get("type") != "message_end": continue
    m = e["message"]; r = m.get("role")
    if r == "system":
        msgs.append({"role": "system", "content": m.get("content") or "\n\n".join(v for v in (m.get("sections") or {}).values() if v)})
    elif r == "user":
        c = m["content"]; msgs.append({"role": "user", "content": c if isinstance(c, str) else "".join(x.get("text", "") for x in c)})
    elif r == "assistant":
        t = "".join(x.get("text", "") for x in m["content"] if x.get("type") == "text")
        calls = [{"id": x["id"], "type": "function", "function": {"name": x["name"], "arguments": json.dumps(x.get("arguments") or {})}}
                 for x in m["content"] if x.get("type") == "toolCall"]
        mm = {"role": "assistant", "content": t}
        if calls: mm["tool_calls"] = calls
        msgs.append(mm)
    elif r == "toolResult":
        msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId"), "content": "".join(x.get("text", "") for x in m["content"] if x.get("type") == "text")})
last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}
         for n, d in (("read", "Read file contents"), ("bash", "Execute bash commands"))]
model = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models", headers=H)))["data"][0]["id"]
body = {"model": model, "messages": msgs[:last], "tools": tools, "max_tokens": 700}
if len(sys.argv) < 4 or sys.argv[3] != "default": body["temperature"] = 0
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), H), timeout=900))
dt = time.time() - t0
msg = r["choices"][0]["message"]
txt = (msg.get("content") or "") + json.dumps(msg.get("tool_calls") or [], ensure_ascii=False)
json.dump({"wall_s": round(dt, 2), "usage": r.get("usage"), "timings": r.get("timings"), "finish": r["choices"][0]["finish_reason"],
           "sha": hashlib.sha256(txt.encode()).hexdigest()[:16], "text": txt}, open(out, "w"), ensure_ascii=False, indent=1)
print(round(dt, 2), r.get("usage"), r["choices"][0]["finish_reason"], hashlib.sha256(txt.encode()).hexdigest()[:16])
