#!/usr/bin/env bash
# Real two-agent test: two read-only repository tasks run by real Pi agents (project AGENTS.md, skills, tools,
# xhigh thinking) against the active pascal-strata model, first each alone, then both at once.
# Usage: run-agents.sh <solo-a|solo-b|solo-c|pair|pair-ab>. pair runs A and C (B loops on Inkling, see README). Pi runs inside a fresh clone of this repository so the agents
# cannot change the working tree. Every artifact is written as soon as it exists (raw/, results.jsonl).
set -uo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${dir}/../.." && pwd)"
# Data stays out of git: out/benchmarks/<this folder>/.
dir="${repo}/out/benchmarks/$(basename "${dir}")"
mkdir -p "${dir}"
host="${PASCAL_HOST:-vm@10.10.88.7}"
mode="${1:?usage: $0 <solo-a|solo-b|solo-c|pair|pair-ab>}"
clone="${PASCAL_AGENT_CLONE:-${TMPDIR:-/tmp}/pascal-llm-agent-clone}"
raw="${dir}/raw"
mkdir -p "${raw}"
[[ -d "${clone}/.git" ]] || git clone -q "${repo}" "${clone}"
echo "${mode}" > "${dir}/PHASE"

rules="Nu modifica niciun fișier, nu rula ssh, nu porni și nu opri servicii; doar citește."
task_a="Citește ops/strata.sh și ops/run-strata-server.sh. Explică în 5 puncte ce face './ops/strata.sh start inkling' de la capăt la capăt: ce fișiere de configurare scrie, ce port și ce GPU-uri folosește și cu ce argumente pornește motorul. ${rules}"
task_c="Citește docs/hardware.md și rezumă în 5 puncte configurația hardware a hostului X99E: GPU-uri, memorie, interconectare și ce rulează pe fiecare placă. ${rules}"
task_b="Caută în docs/ toate valorile de decode tok/s raportate pentru modelele Swift (orice variantă și runtime) și fă un tabel: model, runtime, tok/s, fișier:linie. ${rules}"

run_one() {   # $1 = a|b, $2 = run label
  local s="$1" label="$2" task
  case "$s" in a) task="${task_a}" ;; b) task="${task_b}" ;; *) task="${task_c}" ;; esac
  local ts te rc
  ts="$(python3 -c 'import time;print(time.time())')"
  (cd "${clone}" && timeout -k 30 1800 "${repo}/ops/strata.sh" pi --no-session --mode json -p "${task}" \
      < /dev/null > "${raw}/${label}-${s}.jsonl" 2> "${raw}/${label}-${s}.err")
  rc=$?
  te="$(python3 -c 'import time;print(time.time())')"
  python3 - "${raw}/${label}-${s}.jsonl" "${label}" "${s}" "${rc}" "${ts}" "${te}" >> "${dir}/results.jsonl" <<'EOF'
import json, sys
path, label, sess, rc, ts, te = sys.argv[1:7]
out = {"run": label, "task": sess, "rc": int(rc), "start": float(ts), "end": float(te),
       "wall_s": round(float(te) - float(ts), 1)}
turns = tools = tool_err = inp = cache = outp = 0
stop = None; txt = ""
for l in open(path, errors="replace"):
    if not l.strip().startswith("{"):
        continue
    e = json.loads(l)
    t = e.get("type")
    if t == "message_end" and e["message"].get("role") == "assistant":
        m = e["message"]; u = m.get("usage", {})
        turns += 1; inp += u.get("input", 0) or 0; cache += u.get("cacheRead", 0) or 0; outp += u.get("output", 0) or 0
        stop = m.get("stopReason")
        txt = "".join(c.get("text", "") for c in m["content"] if c.get("type") == "text") or txt
        tools += sum(1 for c in m["content"] if c.get("type") == "toolCall")
    if t == "tool_execution_end" and e.get("isError"):
        tool_err += 1
out.update(turns=turns, tool_calls=tools, tool_errors=tool_err, input_tokens=inp, cache_read_tokens=cache,
           output_tokens=outp, last_stop=stop, answer=txt.strip()[:400])
print(json.dumps(out, ensure_ascii=False), flush=True)
EOF
}

since="$(ssh "${host}" 'date "+%F %T"')"
t0="$(python3 -c 'import time;print(time.time())')"
case "${mode}" in
  solo-a) run_one a solo ;;
  solo-b) run_one b solo ;;
  solo-c) run_one c solo ;;
  pair) run_one a pair & run_one c pair & wait ;;
  pair-ab) run_one a pair & run_one b pair & wait ;;
  *) echo "unknown mode ${mode}" >&2; exit 2 ;;
esac
t1="$(python3 -c 'import time;print(time.time())')"
ssh "${host}" "journalctl --user -u pascal-strata --since '${since}' --no-pager -o cat" > "${raw}/${mode}-server.log" 2>&1
python3 -c "import json;print(json.dumps({'run':'${mode}','total_wall_s':round(${t1}-${t0},1)}))" >> "${dir}/results.jsonl"
curl -s -m 5 -H 'Authorization: Bearer local' http://10.10.88.7:8081/health > "${raw}/${mode}-health-after.json"
echo "${mode}-done" > "${dir}/PHASE"
