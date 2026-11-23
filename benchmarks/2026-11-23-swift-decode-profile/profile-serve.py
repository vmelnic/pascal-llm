#!/usr/bin/env python3
"""Profile one Swift decode in the service's own engine mode (`strata --serve`, layer split) under nvprof.

Run on x99e with pascal-strata stopped:
  profile-serve.py <strata-swift-iq3_s.json> <prompt-ids-file> <max_new> <trace.csv|-> <timing.json> [ENV=VALUE ...]
(trace "-": no profiler, a plain timed run; ENV=VALUE pairs are added to the engine's environment)
Starts `nvprof --print-gpu-trace` around `strata --serve <config args> --layer-split auto --batch 2` (as the server
starts it), waits for READY, sends one `GEN max_new ids` line, reads T lines until DONE, then QUIT. The timing file
holds the host clock of READY, first token and DONE relative to the launch, for the decode window in the trace."""
import json
import os
import subprocess
import sys
import time

cfg_path, ids_path, max_new, trace, timing = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
cfg = json.load(open(cfg_path))
ids = open(ids_path).read().strip()
env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, cfg["gpu"])), STRATA_EXPERIMENTAL_SM60="1")
extra = [kv for kv in sys.argv[6:] if not kv.startswith("PERF_OUT=")]
perf_out = next((kv.split("=", 1)[1] for kv in sys.argv[6:] if kv.startswith("PERF_OUT=")), None)   # perf record over the request only
env.update(kv.split("=", 1) for kv in extra)
cmd = [cfg["exe"], "--serve", *cfg["args"], "--layer-split", "auto", "--batch", "2", *os.environ.get("ENGINE_EXTRA_ARGS", "").split()]
if trace != "-":
    cmd = ["nvprof", "--unified-memory-profiling", "off", "--print-gpu-trace", "--csv", "--normalized-time-unit", "us",
           "--log-file", trace] + cmd
t0 = time.time()
p = subprocess.Popen(cmd, cwd=cfg["cwd"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
marks = {}
for line in p.stdout:
    if line.startswith("READY"):
        marks["ready_s"] = time.time() - t0
        break
perf = None
if perf_out:
    ev = os.environ.get("PERF_EVENT")   # e.g. major-faults: every event with its time and data address, no sampling
    pc = ["perf", "record", "-q", "-e", ev, "-c", "1", "-d", "-T"] if ev else ["perf", "record", "-q", "-F", "999", "-g", "--call-graph", "lbr", "-s"]
    perf = subprocess.Popen(pc + ["-p", str(p.pid), "-o", perf_out],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
def faults(pid):   # (minor, major) page faults of the engine process so far
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[7]), int(f[9])
eng_pid = p.pid
if trace != "-":   # under nvprof the engine is nvprof's child
    kids = open(f"/proc/{p.pid}/task/{p.pid}/children").read().split()
    eng_pid = int(kids[0]) if kids else p.pid
marks["faults_before"] = faults(eng_pid)
marks["t_gen_mono"] = time.clock_gettime(time.CLOCK_MONOTONIC)
gk = os.environ.get("GEN_KEYS", "").strip()   # sampling keys as the server sends them, e.g. "temperature=1.0 top_p=0.95 top_k=20"
p.stdin.write(f"GEN {max_new}{' ' + gk if gk else ''} {ids}\n")
p.stdin.flush()
marks["gen_sent_s"] = time.time() - t0
n = 0
toks = []
for line in p.stdout:
    if line.startswith("T "):
        n += 1
        toks.append(line.split()[1])
        if n == 1:
            marks["first_token_s"] = time.time() - t0
            marks["faults_first_token"] = faults(eng_pid)
            marks["t_first_mono"] = time.clock_gettime(time.CLOCK_MONOTONIC)
    elif line.startswith("DONE"):
        marks["done_s"] = time.time() - t0
        marks["faults_done"] = faults(eng_pid)
        marks["t_done_mono"] = time.clock_gettime(time.CLOCK_MONOTONIC)
        marks["done_line"] = line.strip()
        break
if perf:
    perf.send_signal(__import__("signal").SIGINT)
    perf.wait(timeout=120)
marks["tokens"] = n
marks["tokens_sha"] = __import__("hashlib").sha256(",".join(toks).encode()).hexdigest()[:16]
f = marks.get("done_line", "").split()
if len(f) > 4:
    marks["decode_ms"] = float(f[4])
    marks["decode_tok_s"] = round(n / (float(f[4]) / 1000), 1)
marks["env"] = extra
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=600)
marks["exit_s"] = time.time() - t0
marks["rc"] = p.returncode
json.dump(marks, open(timing, "w"), indent=1)
print(json.dumps(marks))
