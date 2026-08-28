# chaperone

Agentic triage of AlphaFold3-predicted protein-protein interactions (PPIs)
against real biological evidence — a proper biological "chaperone" for a
computational prediction: it doesn't just report a structural confidence
score, it checks the pair against the Human Protein Atlas, PubMed, STRING,
CellPhoneDB, and UniProt, folds follow-up hypotheses on real GPUs when
warranted, and runs a deterministic evidence gate over every verdict before
recommending anything for experimental validation.

## What it does, end to end

For every candidate pair in your CSV, one agent conversation:
1. Pulls real evidence — HPA expression/localization/protein-class, PubMed
   co-occurrence and PTM/glycosylation literature, STRING and CellPhoneDB
   known-interaction data, and UniProt's own function/keyword annotation
   (catches mechanism-level concerns a residue-level check can't, e.g. a
   galectin's binding being fundamentally glycan-mediated).
2. Can trigger a real AlphaFold3 fold (via a local AlphaFast install) mid-
   conversation when a concrete structural hypothesis needs settling —
   "is this pair missing an obligate subunit," "does a known PTM/
   glycosylation site actually sit on the modeled interface" — and uses the
   real result (interface pLDDT, PTM overlap, membrane topology) in its own
   verdict, not just as a footnote.
3. Records a forced, schema-validated verdict (`CONFIRMED_NOVEL`,
   `LIKELY_SUBCOMPLEX`, `ALREADY_KNOWN`, `IMPLAUSIBLE`,
   `LIKELY_ARTIFACT_PTM`, `INSUFFICIENT_EVIDENCE`) with every evidence field
   cited.

Then, across the whole batch:
4. A **deterministic evidence gate** re-checks every verdict against the
   REAL structured tool outputs (not the LLM's prose summary of them) and
   flags specific, checkable contradictions — a STRING hit with no
   literature backing being treated as "known," an HPA compartment mismatch
   nothing caught, a fold's PTM/topology finding the verdict never
   acknowledged. Flagged candidates get a genuine reconsideration call, not
   a rule guessing the replacement label.
5. A **validation-strategy designer** picks PLA / a stimulation assay /
   co-IP and names real target tissues (grounded in the same HPA data, not
   guessed) for every candidate worth pursuing.
6. A **review agent** audits the whole run — malformed fields, missing
   tool-call ledgers, and an LLM consistency check of each validation
   strategy against the real HPA data it should be grounded in.
7. A single traceable HTML report is generated: one flat "worth validating,
   yes/no, and why" table, plus a per-candidate card with the full raw
   tool-call ledger behind every verdict and (when available) an embedded
   3D viewer of the real folded structure.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # then put your real ANTHROPIC_API_KEY in .env
```

Instead of `.env`, you can also pass the key directly on the command line:
`chaperone candidates.csv --api-key sk-ant-...` (overrides `.env`/the
environment if both are set).

Optional, for the real AF3 follow-up folding step: a local AlphaFast
install (MMseqs2-GPU + AF3 weights) — see `skills/fold-candidate/SKILL.md`.
Its location is configurable via `CHAPERONE_ALPHAFAST_DIR` /
`CHAPERONE_ALPHAFAST_DB_DIR` / `CHAPERONE_ALPHAFAST_WEIGHTS_DIR` (the
defaults point at one specific dev box and are almost certainly wrong for
you). Without an AlphaFast install, pass `--no-fold` (or `--fold-gpus ""`)
and folding stays disabled — the `fold_complex` tool is never offered to the
agent, so `run_alphafast.sh`/the SIF/the weights are never touched — and
everything else still works.

**New to chaperone? See [TUTORIAL.md](TUTORIAL.md)** — a walkthrough of a
real run on `examples/candidates.csv`, with the actual output it produced
(including the deterministic gate catching a real mechanism concern the
first triage pass missed).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the parts of chaperone that must be reproducible without an LLM in
the loop: the deterministic gate's fact-checking functions
(`known_interaction_strength`, `hpa_plausibility`, `fold_evidence_status`,
`find_contradictions`, the UniProt glycan-binding mechanism check),
`normalize_str_list` (guards the exact "cut off text" bug a real run hit),
and CSV ingestion. Network-free and fast (network calls are mocked where
the code under test would otherwise make one).

## Run

```bash
chaperone path/to/candidates.csv --concurrency 4
```

For example, using the real dataset checked into `examples/` (see
[TUTORIAL.md](TUTORIAL.md) for the full walkthrough of what this actually
produces):

```bash
chaperone examples/candidates.csv --concurrency 2 --no-fold \
  --out examples/verdicts.csv --report-out examples/report.html
```

Input CSV columns: `protein_a`, `protein_b` (gene symbols, required),
`iptm`, `ptm`, `pae_interaction`, `model_path` (optional, passed through and
shown in the report).

This single command runs the whole pipeline: triage → deterministic gate →
validation-strategy design → review agent → report. Each stage is
individually skippable:

```bash
chaperone candidates.csv \
  --concurrency 4 \
  --api-key sk-ant-...           # overrides ANTHROPIC_API_KEY from the environment/.env
  --fold-gpus "0,1"              # GPU device IDs the agent's fold tool may use; "" disables folding
  --no-fold                      # or just disable folding outright, no AlphaFast install needed
  --out data/verdicts.csv \
  --report-out data/report.html \
  --skip-validation-strategies \
  --skip-report \
  --skip-review \
  --skip-gate
```

**Where data lives**: everything chaperone reads/writes (evidence cache,
per-candidate transcripts, fold outputs, the report) is created relative to
the directory you run it from — `./data/`, `./log/`, `./fold_runs/`,
`./.gpu_locks/` — never inside the installed package. Set `CHAPERONE_HOME`
to pin a fixed project directory instead of relying on your current working
directory.

## Architecture

- `chaperone.agent` — the per-candidate triage conversation (MCP tool use +
  the optional live `fold_complex` tool).
- `chaperone.run_pipeline` — the batch entrypoint (the `chaperone` console
  command).
- `chaperone.deterministic_gate` / `chaperone.reconsider_with_gate` — the
  evidence gate and its reconsideration pass.
- `chaperone.design_validation` — validation-strategy design.
- `chaperone.review_run` — the review agent.
- `chaperone.build_report` — the HTML report generator.
- `chaperone.fold_candidate` / `chaperone.fold_tool` / `chaperone.analyze_fold`
  — real AF3 folding + structural analysis (interface residues, pLDDT,
  PTM/glycosylation overlap, membrane topology).
- `chaperone.sources.*` — the evidence-source clients (HPA, PubMed, STRING,
  CellPhoneDB, UniProt) and the MCP server (`chaperone.sources.server`)
  exposing them as agent tools.

Full design rationale and the history of real bugs found while building
this (worth reading before extending the deterministic gate or the
reconsideration flow) live in `design.md` and `todo.md`.

## Model notes

The core triage/reconsideration model is `claude-fable-5`. Two supporting
steps run on `claude-sonnet-5` instead, deliberately: `fable-5`'s bio-safety
classifier reliably refuses wet-lab validation-protocol design and, for
some content, verdict reconsideration discussing glycan-mediated binding
mechanisms — confirmed via direct bisection, not assumed. Any code that
consumes a forced tool call's output should check `stop_reason` explicitly
rather than trusting an unexpectedly-unchanged result; a refusal produces
an empty tool input that's easy to silently treat as "nothing needed to
change."
