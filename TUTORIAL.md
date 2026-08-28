# Tutorial: your first chaperone run

This walks through a real run on `examples/candidates.csv` — two protein
pairs, chosen specifically because together they show the deterministic
gate actually catching something.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Open `.env` and put in a real `ANTHROPIC_API_KEY`. Everything below costs a
small number of real API calls (no GPU folding — this tutorial keeps
`--fold-gpus ""` throughout so it stays fast and cheap).

## 2. The example dataset

`examples/candidates.csv`:

```csv
protein_a,protein_b
CD69,CLEC2B
LGALS1,CSF2RA
```

Two real, unreported candidate pairs. `CD69`/`CLEC2B` are both immune-cell
surface receptors with real co-expression evidence. `LGALS1` (galectin-1) is
a secreted lectin; `CSF2RA` is the GM-CSF receptor alpha chain.

## 3. Run it

```bash
chaperone examples/candidates.csv --concurrency 2 --fold-gpus "" \
  --out examples/verdicts.csv --report-out examples/report.html
```

This runs the whole pipeline in one command: triage → deterministic gate →
validation-strategy design → review agent → report. Open
`examples/report.html` in a browser when it finishes — everything below is
what a real run actually produced.

## 4. What actually happened (real output, not a hypothetical)

**Stage 1 — triage.** The agent pulled real HPA/PubMed/STRING/CellPhoneDB/
UniProt evidence for both pairs and recorded a first verdict:
- `CD69`/`CLEC2B` → `CONFIRMED_NOVEL` (no STRING/CellPhoneDB hit, real
  co-expression in NK/T cells, genuinely unreported).
- `LGALS1`/`CSF2RA` → the agent *already* caught the mechanism concern live,
  landing directly on `LIKELY_ARTIFACT_PTM` — its own reasoning: LGALS1's
  UniProt entry says "Lectin that binds beta-galactoside..." and CSF2RA
  carries 11 annotated extracellular glycosylation sites, so a real
  interaction here is most likely glycan-mediated, which AlphaFold3's
  bare-sequence model can't represent regardless of which residues it
  happens to place in contact.

**Stage 2 — the deterministic gate.** It re-checked both verdicts against
the real structured tool outputs (not the agent's prose) and found one real
contradiction: `CD69`'s own UniProt function text explicitly says its
recognition of galectin-1 is *carbohydrate-dependent* — the same
mechanism-level concern as `LGALS1`, just not caught the first time. It
flagged this and sent it back for reconsideration; the model genuinely
re-examined it and changed the verdict:

```
CD69/CLEC2B: CONFIRMED_NOVEL -> LIKELY_ARTIFACT_PTM
```

This is the point of the gate: it doesn't matter that the first pass missed
this — a checkable fact (CD69's own documented mechanism) got surfaced and
acted on, not silently lost.

**Stage 3 — validation strategy.** For both (now `LIKELY_ARTIFACT_PTM`)
candidates, a design step grounded in the same real HPA data proposed:
- `CD69`/`CLEC2B` → PLA (proximity ligation assay) in lymphoid tissue / bone
  marrow.
- `LGALS1`/`CSF2RA` → a stimulation assay (secreted ligand + membrane
  receptor) in blood/lymphoid tissue, placenta, and myeloid co-culture —
  with the caveat that the real experiment should test whether binding is
  lactose/glycan-competable, since that's the actual open question.

**Stage 4 — review agent.** Audited both strategies against the real HPA
data grounding them; came back clean.

**Stage 5 — report.** `examples/report.html` opens with a single flat
table — every candidate, `Worth validating? Yes/No`, and why — then one
expandable card per candidate with the full raw tool-call ledger behind
every claim, so you can check any conclusion against the actual evidence
rather than the model's paraphrase of it.

## 5. What to try next

- **With real folding**: point `--fold-gpus` at real GPU device IDs (needs a
  local AlphaFast install — see `skills/fold-candidate/SKILL.md`) and the
  agent can trigger a real AlphaFold3 fold mid-conversation when a concrete
  structural hypothesis needs settling, not just literature/database
  evidence.
- **Your own candidates**: any CSV with `protein_a`/`protein_b` columns
  (gene symbols) works; `iptm`/`ptm`/`pae_interaction`/`model_path` are
  optional and get shown in the report if present.
- **Skip stages you don't need**: `--skip-gate`, `--skip-validation-strategies`,
  `--skip-review`, `--skip-report` each turn off one stage — useful if
  you're iterating on just the triage step.
- **Read the ledger**: click into a candidate card in the report and expand
  "Full tool-call ledger" — every HPA/PubMed/STRING/CellPhoneDB/UniProt call
  and its raw JSON response is right there, so a surprising verdict is
  always checkable against the real evidence it's based on.
