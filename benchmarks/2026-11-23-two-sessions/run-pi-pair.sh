#!/usr/bin/env bash
# Two independent real Pi sessions started together against the pascal-strata service (port 8081).
# Usage: run-pi-pair.sh <swift|inkling>. The model must already be the active Strata model.
# Every artifact is written as soon as it exists: raw/<model>-{a,b}.jsonl/.err, raw/<model>-server.log,
# results.jsonl (one line per session) - an interrupted run keeps what finished.
set -uo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${dir}/../.." && pwd)"
host="${PASCAL_HOST:-vm@10.10.88.7}"
model="${1:?usage: $0 <swift|inkling>}"
raw="${dir}/raw"
mkdir -p "${raw}"
cd "${repo}"
echo "pi-pair-${model}" > "${dir}/PHASE"

since="$(ssh "${host}" 'date "+%F %T"')"
t0="$(python3 -c 'import time;print(time.time())')"
for s in a b; do
  case "$s" in a) mk=ALFA ;; *) mk=BRAVO ;; esac
  (
    ts="$(python3 -c 'import time;print(time.time())')"
    timeout 1200 ./ops/strata.sh pi --no-session --mode json \
      -p "hi, include exact cuvântul ${mk} în răspuns" \
      < /dev/null > "${raw}/${model}-${s}.jsonl" 2> "${raw}/${model}-${s}.err"
    rc=$?
    te="$(python3 -c 'import time;print(time.time())')"
    python3 - "${raw}/${model}-${s}.jsonl" "${model}" "${s}" "${mk}" "${rc}" "${ts}" "${te}" >> "${dir}/results.jsonl" <<'EOF'
import json, sys
path, model, sess, mk, rc, ts, te = sys.argv[1:8]
out = {"model": model, "session": sess, "marker": mk, "rc": int(rc), "start": float(ts), "end": float(te),
       "wall_s": round(float(te) - float(ts), 1)}
txt = ""
for l in open(path, errors="replace"):
    if not l.strip().startswith("{"):
        continue
    e = json.loads(l)
    if e.get("type") == "message_end" and e["message"].get("role") == "assistant":
        m = e["message"]
        out.update(stop=m.get("stopReason"), input_tokens=m["usage"].get("input"), output_tokens=m["usage"].get("output"))
        txt = "".join(c.get("text", "") for c in m["content"] if c.get("type") == "text")
other = "BRAVO" if mk == "ALFA" else "ALFA"
out.update(marker_ok=mk in txt, other_marker_leak=other in txt, reply=txt.strip()[:200])
print(json.dumps(out, ensure_ascii=False), flush=True)
EOF
  ) &
done
wait
t1="$(python3 -c 'import time;print(time.time())')"
ssh "${host}" "journalctl --user -u pascal-strata --since '${since}' --no-pager -o cat" > "${raw}/${model}-server.log" 2>&1
python3 - "${t0}" "${t1}" "${model}" >> "${dir}/results.jsonl" <<'EOF'
import json, sys
t0, t1, model = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
print(json.dumps({"model": model, "session": "pair", "pair_wall_s": round(t1 - t0, 1)}), flush=True)
EOF
curl -s -m 5 -H 'Authorization: Bearer local' http://10.10.88.7:8081/health > "${raw}/${model}-health-after.json"
echo "pi-pair-${model}-done" > "${dir}/PHASE"
