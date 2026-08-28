---
name: fold-candidate
description: Fetch UniProt sequences for one or more proposed proteins and fold them together as a single complex using AlphaFast (local AlphaFold 3), running on this box's existing GPU/database/weights/container setup. Use this to actually execute a triage follow_up like "re-run AF3 multimer with subunit X", or to test any new hypothesis about a complex (e.g. a proposed missing subunit, a re-fold with a different partner).
metadata:
  mcpmarket-version: 1.0.0
---

# Fold a proposed protein complex (AlphaFast / local AF3)

## Overview

The triage agent (`chaperone.agent`) produces `follow_up` recommendations like
"re-run AF3 multimer with CBFB" or "re-model with N-glycans represented" —
but it only *recommends*, it doesn't execute. This skill closes that loop: it
downloads the real sequence for any newly proposed protein from UniProt,
assembles a standard AlphaFold 3 input JSON, and runs it through AlphaFast
(the local, MMseqs2-GPU-accelerated AF3 fork already set up on this box at
`/data/yzy21/yy/af/alphafast`) to get real ipTM/pTM/ranking_score back —
not a guess.

This box already has everything needed: model weights
(`/data/yzy21/yy/af/af3.bin.zst`), pre-built MMseqs2 + mmCIF databases
(`/data/yzy21/yy/af/alphafast_db`), a built Singularity container
(`/data/yzy21/yy/af/alphafast/alphafast.sif`), and 3 free-to-use H100 GPUs.

## When to use this skill

Use this when a deorphanisation triage verdict's `follow_up` field (or a
human) proposes a **specific structural test to actually run**, e.g.:
- `LIKELY_SUBCOMPLEX` naming a missing obligate subunit (e.g. RUNX1+CBFB)
- Testing a candidate against a *known* partner as a positive-control sanity
  check (e.g. re-fold CLEC2D with its known partner KLRB1 instead of the
  AF3-predicted pair)
- Any other "propose a protein, fold it with the others" request

**As of 2026-08-20, the triage agent itself also has this ability live**, via
`chaperone.fold_tool`'s `fold_complex` tool wired into `chaperone.agent`'s
tool list: when Fable 5 proposes a follow-up its own tools can settle (most
commonly a missing-subunit hypothesis), it just calls `fold_complex` directly
instead of writing a deferred text recommendation. This is deliberately
capped at one fold per candidate and bounded by a semaphore sized to the
number of GPUs offered (see `run_pipeline.py --fold-gpus`, default `0,1`) —
folding is real GPU compute and shouldn't be spent reflexively on every
LIKELY_SUBCOMPLEX case. Use this skill's CLI directly (below) for anything
outside that loop: backfilling a past run's follow-ups, one-off exploration,
or a human-proposed test the agent didn't think to run itself.

## How to run it

```bash
cd /data/yzy21/yy/af/deorphanisation
source .venv/bin/activate
python -m chaperone.fold_candidate RUNX1 CBFB --name runx1_cbfb
```

- Arguments are gene symbols (HGNC) or UniProt accessions — mix and match,
  the script auto-detects which is which.
- `--name` sets the job name (used for output file/dir naming) — required.
- `--gpu_device` picks which GPU to use (default `0`, or `$GPU_DEVICE` env
  var) — **check `nvidia-smi` first** and pick a free one; this is a shared
  box (`teichlab` group).
- `--model_seeds 1,2,3` for multiple seeds (default: just `1`).
- Two or more proteins = a heteromeric complex, one chain per protein, in
  the order given. To test a homodimer, pass the same identifier twice.

The script fetches each sequence live (reviewed human UniProt entries only),
builds the AF3 JSON, and runs it through
`alphafast/scripts/run_alphafast.sh` (the officially supported single/multi-GPU
entrypoint — do not hand-roll a `singularity run` command, this script
already handles backend detection, batching, and JAX cache correctly).

## Output

Results land in `fold_runs/<name>/output/<name>/` — the model structure
(`<name>_model.cif`) and confidence scores (`<name>_summary_confidences.json`:
top-level `iptm`, `ptm`, `ranking_score`, `has_clash`, plus a `chain_pair_iptm`
matrix for per-chain-pair interface confidence in complexes with 3+ chains).
The script also prints a JSON summary to stdout with these fields extracted,
plus `elapsed_seconds` and resolved UniProt accessions/lengths for each chain
— paste that back into the relevant candidate's evidence trail (e.g. append
to its `log/<pair>.json` or note it in the report) so the re-fold result
stays traceable alongside the original triage verdict.

## Gotchas

- **AF3 inference OOMs if two jobs share a physical GPU** — it's memory-
  hungry enough that concurrent jobs on the same device crash rather than
  time-slice. This was hit for real: an ad-hoc test collided with a running
  backfill batch, both landing on GPU 0, and one OOM'd. Fixed with a real
  cross-process lock (`fold_candidate.gpu_lock`, `flock` on
  `.gpu_locks/gpu_<id>.lock`) — every fold job, regardless of which script or
  process launched it, now actually blocks until its requested GPU is free,
  rather than relying on each process's own (necessarily blind-to-other-
  processes) semaphore. Don't remove this lock as an "optimization" — it's
  the only thing preventing exactly this crash when more than one fold-
  capable process is active on the box at once.

- UniProt gene-symbol search only returns **reviewed, human** entries — a
  pseudogene or poorly-annotated paralog (seen for real: FCGR1BP in the
  deorphanisation dataset) may have no hit. If it fails, try the exact
  UniProt accession instead, or confirm the gene actually has a canonical
  human protein product before troubleshooting further.
- A cold run (first time on a given sequence pair) takes ~9 min end-to-end on
  this box — verified live on RUNX1+CBFB: ~5.7 min MSA/template search
  (MMseqs2-GPU) + ~1.9 min inference (JAX compile + 5 seed samples) = 521s
  total. Don't assume Modal's 28s figure — that's serverless AF3 with a warm
  JAX cache, not this local cold-start path, though both are fast compared to
  stock AlphaFold3/Jackhmmer.
- This box's databases were set up under `/data/yzy21/yy/af/alphafast_db` —
  do not point `--db_dir` anywhere else without checking it was built with
  `setup_databases.sh` first.
