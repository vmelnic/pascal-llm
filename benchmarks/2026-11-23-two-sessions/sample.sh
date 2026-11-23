#!/usr/bin/env bash
# 60 s sampler for the two-session benchmark. Every row is appended immediately, so an interrupted run keeps
# everything measured so far. Stop with: touch STOP (in the data folder) or kill the process.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Data stays out of git: out/benchmarks/<this folder>/ (control files PHASE and STOP live there too).
dir="$(cd "${here}/../.." && pwd)/out/benchmarks/$(basename "${here}")"
mkdir -p "${dir}"
host="${PASCAL_HOST:-vm@10.10.88.7}"
interval="${PASCAL_SAMPLE_INTERVAL:-60}"
csv="$dir/samples.csv"
log="$dir/samples-engine.log"
[[ -f "$csv" ]] || echo "utc,phase,gpu0_mib,gpu1_mib,gpu2_mib,gpu3_mib,gpu0_util,gpu1_util,gpu2_util,gpu3_util,host_used_gib,host_avail_gib,strata,llm,strata_model,health_8081" > "$csv"
while [[ ! -f "$dir/STOP" ]]; do
  phase="$(cat "$dir/PHASE" 2>/dev/null || echo idle)"
  row="$(timeout 30 ssh -o ConnectTimeout=10 "$host" '
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " " | paste -sd,)
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d " " | paste -sd,)
    h=$(free -b | awk "/^Mem:/{printf \"%.2f,%.2f\", \$3/2^30, \$7/2^30}")
    s=$(systemctl --user is-active pascal-strata); l=$(systemctl --user is-active pascal-llm)
    a=$(cat ~/.config/pascal-llm/active-strata-model 2>/dev/null)
    c=$(curl -s -m 3 -H "Authorization: Bearer local" http://127.0.0.1:8081/health | tr -d ",\n" | cut -c1-80)
    echo "$m,$u,$h,$s,$l,$a,${c:-none}"' 2>/dev/null)"
  echo "$(date -u +%FT%TZ),$phase,${row:-ssh-failed}" >> "$csv"
  timeout 30 ssh -o ConnectTimeout=10 "$host" 'journalctl --user -u pascal-strata --since "-65 s" --no-pager -o cat 2>/dev/null | grep -a -E "prompt|done:|BDONE|slot|ERR|error" | tail -20' 2>/dev/null \
    | sed "s/^/$(date -u +%FT%TZ) /" >> "$log"
  for ((i = 0; i < interval; ++i)); do [[ -f "$dir/STOP" ]] && break; sleep 1; done
done
