# Provenance

Vendored from https://github.com/JuliusBrussee/caveman
commit b89752f6bca5a26bfde8b019813554241800ca99, path `skills/caveman/SKILL.md`, on 2026-11-23.

Licensed Apache-2.0 (see `LICENSE`; attribution in `NOTICE`). Copyright 2026
Julius Brussee. Unmodified.

## What it is used for here, and what it is deliberately NOT used for

Caveman is an output-voice skill: it compresses prose without dropping technical
facts. It is vendored here for use when driving this repository from Claude Code.

It is **not** applied to the triage agent's recorded verdicts. Caveman's own rules
exempt "anything persisted outside chat", and the verdict `rationale` and
`*_evidence` fields are exactly that: they are the product, they are read by
someone deciding whether to spend bench time on a candidate, and they are quoted
into the HTML report and the manuscript. Compressing them would save output
tokens by degrading the deliverable.

What it IS applied to is the agent's narration between tool calls, which caveman's
rule 7 ("Tool runs: bounded status") targets and which nothing downstream reads.
See the "Narration" section of `SYSTEM_PROMPT` in `src/chaperone/agent.py`.

## Where chaperone's tokens actually went

Measured over the 50-candidate run in `log/`:

| | per candidate | share of input |
|---|---|---|
| system prompt, re-sent every turn (~4,550 tok x ~4 turns) | ~18,000 tok | ~85% |
| tool output, re-read as context grows | ~1,200 tok (x re-reads) | ~14% |
| agent narration between tool calls | ~400 tok out | — |
| recorded verdict fields (the deliverable) | ~730 tok out | — |

So the dominant cost was never the prose. It was the system prompt: 102 candidates
paid roughly **1.9M input tokens** for text that is byte-identical every time. That
is fixed by prompt caching, added in the same change. Caveman addresses the ~400
tokens of narration, which is real but an order of magnitude smaller.
