#!/usr/bin/env python3
"""Decode-window analysis of an nvprof --print-gpu-trace CSV (us units): per device, busy time (union of kernel and
copy intervals), idle share, time by kernel name and by copy kind. Window: the last `decode_ms` before the end of the
last kernel. Usage: analyze.py trace.csv decode_ms [top_n]"""
import csv
import collections
import sys

path, decode_ms = sys.argv[1], float(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = []
with open(path, newline="") as f:
    lines = [l for l in f if not l.startswith("==")]
r = csv.reader(lines)
hdr = next(r)
next(r)                                           # units row
ix = {h: i for i, h in enumerate(hdr)}
for x in r:
    if len(x) < len(hdr):
        continue
    try:
        s, d = float(x[ix["Start"]]), float(x[ix["Duration"]])
    except ValueError:
        continue
    size = x[ix["Size"]]
    rows.append((s, d, x[ix["Device"]], x[ix["Name"]], float(size) if size else 0.0, x[ix["SrcMemType"]], x[ix["DstMemType"]]))
end = max(s + d for s, d, dev, n, *_ in rows if not n.startswith("[CUDA"))
w0, w1 = end - decode_ms * 1000, end
print(f"window {decode_ms:.1f} ms ending at {end/1e6:.3f} s")
by_dev = collections.defaultdict(list)
for row in rows:
    s, d = row[0], row[1]
    if s + d < w0 or s > w1:
        continue
    by_dev[row[2]].append(row)
for dev in sorted(by_dev):
    rs = by_dev[dev]
    iv = sorted((max(s, w0), min(s + d, w1)) for s, d, *_ in rs)
    busy = 0.0; cs, ce = None, None
    for a, b in iv:
        if cs is None or a > ce:
            if cs is not None: busy += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    if cs is not None: busy += ce - cs
    span = w1 - w0
    names = collections.Counter(); counts = collections.Counter(); copy_bytes = collections.Counter()
    for s, d, _, n, size, src, dst in rs:
        dd = min(s + d, w1) - max(s, w0)
        key = n if not n.startswith("[CUDA") else f"{n} {src}->{dst}"
        names[key] += dd; counts[key] += 1
        if n.startswith("[CUDA"): copy_bytes[key] += size
    print(f"\n== {dev}: busy {busy/1000:.1f} ms of {span/1000:.1f} ms = {100*busy/span:.1f}%, idle {100-100*busy/span:.1f}%")
    tot = sum(names.values())
    for k, v in names.most_common(top):
        extra = f"  {copy_bytes[k]:.3f} GB" if k in copy_bytes else ""
        print(f"  {v/1000:8.1f} ms {100*v/tot:5.1f}%  n={counts[k]:6d}  {k[:110]}{extra}")

# ---- categories and concurrency ----
CATS = [
    ("experts (gate/up/down)", ("native_gu_fused", "native_down_lds")),
    ("expert fetch from host", ("fetch_blobs", "[CUDA memcpy HtoD]")),
    ("copies to host", ("[CUDA memcpy DtoH]",)),
    ("copies between GPUs", ("[CUDA memcpy PtoP]", "[CUDA memcpy DtoD]")),
    ("output head (Q6_K)", ("native_q6_k_mmvq",)),
    ("dense projections", ("mmvq", "mmvf")),
    ("hyper-connections", ("gr_",)),
    ("attention / GDN", ("qsa_decode_attn", "gdn_", "attn_")),
    ("router", ("route_",)),
    ("sync waits", ("wait_flag",)),
]
def cat(name):
    for c, keys in CATS:
        if any(k in name for k in keys):
            return c
    return "other small kernels"
tot = collections.Counter()
for dev in sorted(by_dev):
    for s, d, _, n, *_ in by_dev[dev]:
        tot[cat(n)] += min(s + d, w1) - max(s, w0)
all_t = sum(tot.values())
print("\n== all three GPUs, time by category")
for k, v in tot.most_common():
    print(f"  {v/1000:8.1f} ms {100*v/all_t:5.1f}%  {k}")
ev = []
for dev, rs in by_dev.items():
    iv = sorted((max(s, w0), min(s + d, w1)) for s, d, *_ in rs)
    m = []
    for a, b in iv:
        if m and a <= m[-1][1]: m[-1][1] = max(m[-1][1], b)
        else: m.append([a, b])
    for a, b in m: ev += [(a, 1), (b, -1)]
ev.sort(); hist = collections.Counter(); cur = 0; last = w0
for t, dlt in ev:
    hist[cur] += t - last; cur += dlt; last = t
hist[cur] += w1 - last
print("\n== how many GPUs are busy at the same time")
for k in sorted(hist):
    print(f"  {k} GPU(s): {hist[k]/1000:8.1f} ms {100*hist[k]/(w1-w0):5.1f}%")
