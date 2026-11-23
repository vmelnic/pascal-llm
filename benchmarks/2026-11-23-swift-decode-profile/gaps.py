#!/usr/bin/env python3
"""Where the decode window's no-GPU-busy time falls: every gap between the union of all devices' kernel/copy intervals,
classified by the device that was active before it and the one after it (0->1 and 1->2: stage hand-off inside a window;
2->0: between windows; same device: a pause inside one stage). Usage: gaps.py trace.csv decode_ms"""
import csv, collections, sys
path, decode_ms = sys.argv[1], float(sys.argv[2])
lines = [l for l in open(path, newline="") if not l.startswith("==")]
r = csv.reader(lines); hdr = next(r); next(r); ix = {h: i for i, h in enumerate(hdr)}
rows = []
for x in r:
    if len(x) < len(hdr): continue
    try: s, d = float(x[ix["Start"]]), float(x[ix["Duration"]])
    except ValueError: continue
    dev = x[ix["Device"]]; dev = dev[dev.rfind("(") + 1:dev.rfind(")")] if "(" in dev else dev
    rows.append((s, s + d, dev, x[ix["Name"]]))
end = max(e for s, e, dev, n in rows if not n.startswith("[CUDA"))
w0 = end - decode_ms * 1000
ev = sorted((max(s, w0), min(e, end), dev, n) for s, e, dev, n in rows if e > w0 and s < end)
gaps = collections.defaultdict(lambda: [0, 0.0]); big = []
cur_end, cur_dev, cur_name = w0, None, None
for s, e, dev, n in ev:
    if s > cur_end:
        g = s - cur_end
        key = f"{cur_dev}->{dev}"
        gaps[key][0] += 1; gaps[key][1] += g
        if g > 300: big.append((g, key, (cur_name or "")[:60], n[:60]))
    if e > cur_end:
        cur_end, cur_dev, cur_name = e, dev, n
tot = sum(v[1] for v in gaps.values())
print(f"no-GPU time {tot/1000:.1f} ms of {decode_ms:.1f} ms ({100*tot/decode_ms/1000:.1f}%)")
for k, (n, t) in sorted(gaps.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k:8s} gaps {n:6d}  total {t/1000:7.1f} ms  ({100*t/tot:5.1f}%)  mean {t/max(n,1):7.1f} us")
big.sort(reverse=True)
print("largest gaps (us, transition, kernel before -> kernel after):")
for g, k, a, b in big[:15]:
    print(f"  {g:8.0f}  {k}  {a}  ->  {b}")
