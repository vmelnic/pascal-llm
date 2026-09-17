# AGENTS.md

Execution contract for this repository.

## Goal

Build and qualify an artifact-driven local LLM runtime for the X99E host with
three Tesla P100 16 GiB cards and one Tesla P40 24 GiB card. Qwen3.8-27B is the
first feasibility target. Work must remain useful to other compatible models;
model-specific source adapters are allowed, model-specific runtime branches are
not. The production client is the real Pi coding harness with project
`AGENTS.md` files and skills enabled, serving multiple agents or users
concurrently rather than a single exclusive chat session.

## Mandatory behavior

- The current user request defines the active task and takes precedence over
  the project goal. Never reject or replace a valid direct request merely
  because it is unrelated to the repository. If essential information is
  missing, ask only the minimum clarification needed to fulfill the request.
- Communicate with the user in Romanian. Code, comments and documentation are
  English.
- Never guess when source, artifacts, telemetry or host inspection can answer.
- Do not modify `../quantum-llm` unless the user explicitly requests upstreaming.
- Do not copy the old runtime wholesale. Reuse only a component whose ABI,
  numerical behavior and Pascal compatibility have been verified.
- Tie every implementation and measurement to the active acceptance gate. No
  infrastructure, smoke tests or benchmarks for their own sake.
- A single-user direct chat is only a transport check, never the production
  acceptance gate. Production qualification MUST use Pi with its real injected
  instructions, skills and tool loop, with concurrent independent sessions for
  multiple agents or users.
- Request, conversation, parser, tool, cancellation and KV state MUST be scoped
  per session. Scheduling MUST provide isolation, bounded backpressure and
  fairness so one long prefill or generation cannot corrupt or indefinitely
  block the other active sessions.
- Before inference code, update `docs/feasibility.md` with exact capacity,
  traffic, compute and communication equations for the selected model.
- Preserve model semantics unless the user explicitly approves a fidelity
  change. Report weights, KV, activations and intermediates separately.
- Do not call aggregate VRAM unified memory. Every device boundary and transfer
  must be represented and measured.
- Never store passwords, private keys or tokens in this repository.

## Review and report convergence

These rules apply to review, audit and report tasks. They do not reduce the
scope of an implementation requested by the user. Prefer a deep, defensible
review over a fast superficial conclusion; convergence means converting depth
into evidence and a result, not reducing necessary investigation.

1. Define the completion condition before exploring: requested artifact,
   active production path, affected interfaces and severity threshold.
2. Inspect reachable production code first. Inspect rejected experiments,
   historical benchmarks or inactive tools when an active claim, decision or
   evidence ledger depends on them; otherwise keep them outside the scope.
3. Maintain one evolving finding ledger: claim, exact evidence, impact,
   confidence and required action. Never rediscover or recalculate an already
   settled item.
4. Investigate in proportion to impact. A potentially critical correctness,
   fidelity, safety or evidence-integrity defect MUST be followed through to a
   verified verdict or a concrete blocker. For a minor uncertainty, make one
   focused verification attempt, label unresolved evidence honestly and move
   on.
5. Prefer decisive verification over repeated speculation. When a source
   lookup, focused reproducer, numerical oracle or host measurement can answer
   the question, perform it as soon as practical; do not repeatedly reconstruct
   the same answer from memory. Multiple checks are welcome when independent
   evidence is needed to establish a high-impact conclusion.
6. Audit arithmetic when it can change an active capacity, fidelity,
   performance, evidence-integrity or release decision. Record harmless
   rounding differences compactly instead of expanding them into a detour.
7. After each subsystem, merge its material findings into the report before
   opening another subsystem. Once all reachable active subsystems are covered,
   write the requested artifact and stop. Do not inspect unrelated git history
   or expand scope merely because token budget remains. Do not stop early while
   a material finding is still unsupported or unresolved.
8. Default to `xhigh`; token ceilings are runaway guards, not depth targets.
   Keep a compact checkpoint of verified facts, gates, blocker and next action.
   After failure, resume there and redo only invalidated dependencies. Restart
   from zero ONLY when an upstream invariant was disproved; stop reasoning paths
   that repeat without new evidence.

## Change loop

```text
PLAN -> INSPECT -> IMPLEMENT COMPLETE CHANGE -> TEST REAL GATE
  failure #1 -> at most one local corrective patch -> RETEST
  failure #2 -> STOP EDITING -> reassess end-to-end -> NEW PLAN
```

## Active invariants

```text
host             x99e, Ubuntu Server
accelerators     3 x Tesla P100 16 GiB SM60 + 1 x Tesla P40 24 GiB SM61
aggregate VRAM   72 GiB; capacity is not proof of throughput
interconnect     PCIe 3.0 x16; no NVLink
execution        self-contained on x99e
first model      Qwen3.8-27B
context target   262,144 actually populated tokens
KV reference     exact IEEE binary16 values
quality          unchanged unless explicitly approved
scope            artifact-driven, not family-hardcoded
harness          Pi with project AGENTS.md and skills enabled
concurrency      multiple isolated agents/users, not one exclusive session
```

## Canonical documents

- Architecture and hardware: `docs/hardware.md`
- Quantitative prerequisite: `docs/feasibility.md`
- Accepted/rejected directions: `docs/decisions.md`
- Dependency order: `docs/roadmap.md`
- Web UI deployment: `docs/web-ui.md`
