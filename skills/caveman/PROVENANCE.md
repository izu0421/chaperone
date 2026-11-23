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

So the dominant cost was never the prose. It was the repeated prefix, re-sent on
every turn of every candidate. Caveman addresses the ~400 tokens of narration, which
is real but an order of magnitude smaller.

## What caching actually saved, measured

Estimating this from the prompt text alone was wrong in two ways, so the numbers
below come from `_usage`, recorded per candidate, on a three-candidate probe:

| | |
|---|---|
| cached prefix | **~12,600 tokens per turn**, not the ~4,500 of system-prompt text |
| cache reads | 163,826 of 215,295 input-token-equivalents, **76%** |
| billed-equivalent | 71,002 against 215,295 without caching, **67% less** |

Two corrections to the estimate. The cached prefix is nearly three times the size
of the system prompt, because the tool schemas render before it and sit inside the
same cached prefix — they are the larger half, and reading the prompt text alone
misses them. And the saving is 67%, not the ~90% a naive reading of the cache-read
discount suggests, because a third of input is genuinely new each turn: the tool
results, which are not cacheable and are what the prefix was hiding.

The first candidate in a batch pays the write (12,602 tokens at 1.25x) and every
one after it reads (63,010 at 0.1x, zero writes), so the share of reads rises with
batch size and the 67% is a floor for a 102-candidate run, not a ceiling.

