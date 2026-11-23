#!/usr/bin/env python3
"""Offline replay of the suffix-lookup drafter on real Pi agent runs (no GPU, no service).

For every assistant turn of the given Pi JSONL runs, the conversation is rendered with the server's own Inkling
template and tokenizer (serve/frontend.py ChatTemplate, tools/strata_tokenizer.py), the prompt is the history plus
the server's empty thinking turn (inkling_dialect.finish_prompt with tools), and the output is the assistant turn's
rendered tokens. The drafter is a line-for-line port of engine/src/spec/suffix_drafter.cpp (trigram index, 4 ways,
longest backward match up to 32, min_match). Greedy acceptance: the proposal is accepted up to the first token that
differs from the real output; every verify step also emits one target token.

Step cost model (measured TP3 step times, docs/inkling-tp3-plan.md "Reassessment"): M=1 15.0 ms, M=2 21.2 ms,
M=4 27.0 ms, M=8 55 ms; M=3 interpolated. Usage, on x99e:
  suffix-replay.py <strata terminal dir> <tokenizer dir> run1.jsonl [run2.jsonl ...]
"""
import json
import sys

TERM, TOKDIR, RUNS = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path[:0] = [TERM, TERM + "/tools"]
from pathlib import Path                                         # noqa: E402
import strata_tokenizer as ST                                    # noqa: E402
from serve.frontend import ChatTemplate                          # noqa: E402
from serve import inkling_dialect                                # noqa: E402

COST = {1: 15.0, 2: 21.2, 3: 24.1, 4: 27.0, 5: 34.0, 6: 41.0, 7: 48.0, 8: 55.0}
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p}}}
         for n, d, p in (("read", "Read file contents", {"path": {"type": "string"}}),
                         ("bash", "Execute bash commands", {"command": {"type": "string"}}),
                         ("edit", "Make precise file edits", {"path": {"type": "string"}}),
                         ("write", "Create or overwrite files", {"path": {"type": "string"}, "content": {"type": "string"}}))]


def load_tok(d):
    d = Path(d)
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    tcfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, (d / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((d / "token_type.json").read_text()), tcfg.get("pre") or "qwen35",
                        tcfg.get("special_ids") or None)


def pi_messages(path):
    """Pi JSONL -> OpenAI chat messages (system, user, assistant with tool_calls, tool)."""
    out = []
    for line in open(path, errors="replace"):
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        if e.get("type") != "message_end":
            continue
        m = e["message"]
        role = m.get("role")
        if role == "system":
            secs = m.get("sections") or {}
            out.append({"role": "system", "content": m.get("content") or "\n\n".join(v for v in secs.values() if v)})
        elif role == "user":
            c = m["content"]
            out.append({"role": "user", "content": c if isinstance(c, str) else "".join(x.get("text", "") for x in c)})
        elif role == "assistant":
            text = "".join(x.get("text", "") for x in m["content"] if x.get("type") == "text")
            calls = [{"id": x["id"], "type": "function", "function": {"name": x["name"], "arguments": x.get("arguments") or {}}}
                     for x in m["content"] if x.get("type") == "toolCall"]
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "toolResult":
            out.append({"role": "tool", "tool_call_id": m.get("toolCallId"),
                        "content": "".join(x.get("text", "") for x in m["content"] if x.get("type") == "text")})
    return out


class Suffix:
    """Port of strata::spec::SuffixDrafter (dict instead of the open-addressing table; same 4 ways)."""
    def __init__(self, min_match=3, max_match=32):
        self.mn, self.mx, self.h, self.idx = max(3, min_match), max_match, [], {}

    def append(self, toks):
        h, idx = self.h, self.idx
        for t in toks:
            h.append(t)
            end = len(h) - 1
            if end < 2:
                continue
            k = (h[end - 2], h[end - 1], h[end])
            ways = idx.get(k)
            if ways is None:
                idx[k] = [end]
            else:
                ways.insert(0, end)
                del ways[4:]

    def propose(self, max_k):
        h = self.h
        n = len(h)
        if n < 4 or max_k <= 0:
            return [], 0
        cur = n - 1
        ways = self.idx.get((h[cur - 2], h[cur - 1], h[cur]))
        if not ways:
            return [], 0
        best_end, best = 0, 0
        for p in ways:
            if p >= cur:
                continue
            ln = 0
            while ln < self.mx and ln <= p and h[p - ln] == h[cur - ln]:
                ln += 1
            if ln > best:
                best, best_end = ln, p
        if best < self.mn:
            return [], best
        return h[best_end + 1: min(cur, best_end + max_k) + 1], best


def replay(prompt, output, k, min_match):
    """Returns (steps, ms) for generating `output` greedily with drafts of up to k tokens."""
    d = Suffix(min_match)
    d.append(prompt)
    i = steps = 0
    ms = 0.0
    stats = {"draft_steps": 0, "accepted": 0, "proposed": 0}
    while i < len(output):
        prop, _ = d.propose(k) if k > 0 else ([], 0)
        prop = prop[: len(output) - i - 1] if len(output) - i - 1 >= 0 else []
        a = 0
        while a < len(prop) and prop[a] == output[i + a]:
            a += 1
        m = 1 + len(prop)
        ms += COST[m]
        steps += 1
        if prop:
            stats["draft_steps"] += 1
            stats["proposed"] += len(prop)
            stats["accepted"] += a
        emit = output[i: i + a + 1]
        d.append(emit)
        i += len(emit)
    return steps, ms, stats


def main():
    tok = load_tok(TOKDIR)
    tpl = ChatTemplate(Path(TOKDIR) / "chat_template.jinja")
    tail = tok.encode("<|message_model|><|content_thinking|><|end_message|>", parse_special=True)
    turns = []                                   # (run, prompt tokens, output tokens)
    for run in RUNS:
        msgs = pi_messages(run)
        for j, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            before = tpl.render(msgs[:j], tools=TOOLS, add_generation_prompt=False)
            after = tpl.render(msgs[: j + 1], tools=TOOLS, add_generation_prompt=False)
            if not after.startswith(before):
                print(f"skip {run} turn {j}: render is not a prefix", file=sys.stderr)
                continue
            p = tok.encode(before, parse_special=True) + tail
            o = tok.encode(after[len(before):], parse_special=True)
            turns.append((Path(run).name, p, o))
    n_out = sum(len(o) for _, _, o in turns)
    base_ms = n_out * COST[1]
    print(f"runs {len(RUNS)}, assistant turns {len(turns)}, output tokens {n_out}, "
          f"prompt tokens max {max(len(p) for _, p, _ in turns)}")
    print(f"plain M=1: {base_ms / 1000:.1f} s -> {n_out / base_ms * 1000:.1f} tok/s (cost model)")
    print("k  min_match  draft_steps  proposed  accepted  acc%   tokens/step  model tok/s  x plain")
    for k in (1, 2, 3, 5, 7):
        for mm in (3, 4, 6, 8, 12):
            steps = 0
            ms = 0.0
            agg = {"draft_steps": 0, "accepted": 0, "proposed": 0}
            for _, p, o in turns:
                s, t, st = replay(p, o, k, mm)
                steps += s
                ms += t
                for key in agg:
                    agg[key] += st[key]
            acc = 100.0 * agg["accepted"] / max(1, agg["proposed"])
            print(f"{k}  {mm:9d}  {agg['draft_steps']:11d}  {agg['proposed']:8d}  {agg['accepted']:8d}  {acc:5.1f}  "
                  f"{n_out / steps:11.2f}  {n_out / ms * 1000:11.1f}  {base_ms / ms:7.2f}")
    print("per run, k=3 min_match=4:")
    by = {}
    for name, p, o in turns:
        s, t, st = replay(p, o, 3, 4)
        r = by.setdefault(name, [0, 0, 0.0])
        r[0] += len(o); r[1] += s; r[2] += t
    for name, (n, s, t) in by.items():
        print(f"  {name:16s} out {n:6d}  tokens/step {n / s:5.2f}  model {n / t * 1000:6.1f} tok/s  x plain {n * COST[1] / t:5.2f}")


main()
