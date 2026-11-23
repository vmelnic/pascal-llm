#!/usr/bin/env bash
# Thinking-effort benchmark: the real Pi agent task A (project AGENTS.md, skills, tools) at one thinking level against
# the active pascal-strata model. Usage: run-effort.sh <label> <low|medium|xhigh>. Pi runs inside a fresh clone of this
# repository so the agent cannot change the working tree. Data: out/benchmarks/<this folder>/ (raw/, results.jsonl).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../.." && pwd)"
dir="${repo}/out/benchmarks/$(basename "${here}")"
raw="${dir}/raw"
mkdir -p "${raw}"
label="${1:?usage: $0 <label> <level>}"
level="${2:?usage: $0 <label> <level>}"
host="${PASCAL_HOST:-vm@10.10.88.7}"
clone="${PASCAL_AGENT_CLONE:-${TMPDIR:-/tmp}/pascal-llm-agent-clone}"
[[ -d "${clone}/.git" ]] || git clone -q "${repo}" "${clone}"
task="Citește ops/strata.sh și ops/run-strata-server.sh. Explică în 5 puncte ce face './ops/strata.sh start inkling' de la capăt la capăt: ce fișiere de configurare scrie, ce port și ce GPU-uri folosește și cu ce argumente pornește motorul. Nu modifica niciun fișier, nu rula ssh, nu porni și nu opri servicii; doar citește."
run="${label}-${level}"
since="$(ssh "${host}" 'date "+%F %T"')"
ts="$(python3 -c 'import time;print(time.time())')"
(cd "${clone}" && timeout -k 30 1800 "${repo}/ops/strata.sh" pi --no-session --mode json --thinking "${level}" -p "${task}" \
    < /dev/null > "${raw}/${run}.jsonl" 2> "${raw}/${run}.err")
rc=$?
te="$(python3 -c 'import time;print(time.time())')"
ssh "${host}" "journalctl --user -u pascal-strata --since '${since}' --no-pager -o cat" > "${raw}/${run}-server.log" 2>&1
curl -s -m 10 -H 'Authorization: Bearer local' http://10.10.88.7:8081/metrics > "${raw}/${run}-metrics.json"
python3 - "${raw}/${run}.jsonl" "${raw}/${run}-metrics.json" "${label}" "${level}" "${rc}" "${ts}" "${te}" >> "${dir}/results.jsonl" <<'PY'
import json, sys
path, mpath, label, level, rc, ts, te = sys.argv[1:8]
turns = tools = tool_err = outp = 0
think_chars = text_chars = 0
stop = None; txt = ""
start = None
for l in open(path, errors="replace"):
    if not l.strip().startswith("{"):
        continue
    e = json.loads(l)
    if e.get("type") == "message_end" and e["message"].get("role") == "assistant":
        m = e["message"]; u = m.get("usage", {})
        start = start or m.get("timestamp")
        turns += 1; outp += u.get("output", 0) or 0
        stop = m.get("stopReason")
        think_chars += sum(len(c.get("thinking", "")) for c in m["content"] if c.get("type") == "thinking")
        t = "".join(c.get("text", "") for c in m["content"] if c.get("type") == "text")
        text_chars += len(t); txt = t or txt
        tools += sum(1 for c in m["content"] if c.get("type") == "toolCall")
    if e.get("type") == "tool_execution_end" and e.get("isError"):
        tool_err += 1
reqs = []
try:
    reqs = [r for r in json.load(open(mpath)).get("requests", []) if r.get("time", 0) >= float(ts) - 5]
except Exception:
    pass
dec = [(r.get("output_tokens") or 0, r.get("decode_ms") or 0) for r in reqs]
prompt = sum(r.get("prompt_tokens") or 0 for r in reqs); reused = sum(r.get("reused") or 0 for r in reqs)
out = {"run": label, "level": level, "rc": int(rc), "wall_s": round(float(te) - float(ts), 1), "turns": turns,
       "tool_calls": tools, "tool_errors": tool_err, "output_tokens": outp, "thinking_chars": think_chars,
       "answer_chars": text_chars, "requests": len(reqs), "prompt_tokens": prompt, "reused_tokens": reused,
       "decode_tok_s": round(sum(a for a, _ in dec) / max(1e-9, sum(b for _, b in dec) / 1000), 1) if dec else None,
       "last_stop": stop, "answer": txt.strip()[:1500]}
print(json.dumps(out, ensure_ascii=False), flush=True)
PY
tail -1 "${dir}/results.jsonl" | python3 -c "import json,sys;d=json.loads(sys.stdin.read());d.pop('answer');print(d)"
