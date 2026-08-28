# deorphanisation — agentic PPI triage pipeline

## Goal

Input: a list of novel protein-protein interactions (PPIs) predicted by AlphaFold3
(candidate pairs + structural confidence metrics, e.g. ipTM/pTM/PAE, plus model
files). Many of these are structurally confident but biologically implausible
(never co-expressed, wrong compartment) or are really fragments of a larger,
already-partially-known complex rather than a true novel binary pair.

Output: an agentic pipeline, orchestrated by **Fable 5** (`claude-fable-5`, user
supplies the API key), that pulls independent biological evidence per candidate
— primarily from the **Human Protein Atlas (HPA)** via MCP — and returns a
structured verdict + rationale + follow-up recommendation for every candidate.

## Why agentic, not a fixed script

The right evidence to pull differs per candidate:
- A confident ligand↔receptor pair legitimately doesn't need to be co-expressed
  in the *same* cell (paracrine signalling) — a fixed same-cell-coexpression
  filter would wrongly kill it. An agent can recognize "one is secreted /
  membrane receptor" and check compartment logic instead of coexpression.
- If HPA/CORUM shows protein A already sits in a known multi-subunit complex,
  the agent should go pull that complex's other members and check whether the
  AF3 pair looks like a real subcomplex of it, or an odd pairing that ignores
  the complex context — this needs a follow-up tool call, not a static filter.
- Missing HPA data for one protein should trigger a literature fallback, not a
  silent "implausible" verdict.

So: one tool-using agent conversation per candidate (or small batch), not a
rigid DAG. Claude decides which tools to call and in what order, then must
close out with a structured verdict via forced tool-call (JSON schema), same
pattern as this harness's Workflow `schema` option.

## Inputs

- Candidate list (CSV/JSON) with at minimum: `protein_a`, `protein_b`
  (UniProt accession preferred, gene symbol fallback), AF3 confidence metrics
  (`iptm`, `ptm`, `pae_interaction` or similar), and a path/id to the AF3
  model. **Exact file/schema TBD with user — see todo.md #1.**
- `ANTHROPIC_API_KEY` (or equivalent) supplied by the user for Fable 5.

## Evidence sources (MCP tools)

No pre-built HPA MCP connector exists in this environment's roster, so it
needs to be built: a small local MCP server (`mcp_hpa/`) wrapping HPA's public
per-gene JSON API (`proteinatlas.org/<ensembl_id>.json`) and/or the bulk TSV
download, exposing:

| tool | returns |
|---|---|
| `hpa_expression(gene)` | tissue + single-cell (Cell Type / Blood Atlas) expression profile |
| `hpa_subcellular_location(gene)` | annotated compartment(s), reliability score |
| `hpa_protein_class(gene)` | e.g. "secreted", "membrane", "transcription factor" — needed for the ligand/receptor vs same-compartment logic above |

Secondary evidence tools, in the same MCP server (`mcp_hpa/pubmed_client.py`
wraps NCBI E-utilities, built):
- `pubmed_cooccurrence(gene_a, gene_b)` — do the two gene symbols get
  co-mentioned in the literature at all? A weak novelty proxy: zero hits is
  fairly strong evidence of genuine novelty; nonzero needs the returned
  titles read, since co-mention can be indirect (same pathway/disease, not a
  reported direct interaction).
- `pubmed_ptm_glycosylation(gene)` — **PTM/glycosylation artifact check.**
  AF3 predicts from bare, unmodified sequence: it doesn't model glycans and,
  by default, doesn't model most PTMs. A structurally confident interface can
  still be unreal if the true contact surface is normally glycosylated
  (common on secreted/membrane extracellular domains, and can sterically
  block a modeled contact) or if the interaction is only reported in a
  specific PTM state (e.g. phosphorylation-gated binding). This tool searches
  PubMed for such reports per-gene; the agent judges whether they're plausibly
  relevant to the modeled interface (not just any PTM anywhere on a large,
  well-studied protein) and downgrades to a dedicated `LIKELY_ARTIFACT_PTM`
  verdict when relevant.

- `string_known_interaction(gene_a, gene_b)` — STRING (string-db.org) combined
  score + per-evidence-channel breakdown (experiments, curated database,
  co-expression, text-mining, ...). A high database/experiments score is
  strong direct evidence; a score driven only by textmining carries the same
  co-mention caveat as PubMed.
- `cellphonedb_known_interaction(gene_a, gene_b)` / `cellphonedb_complex_members(gene)`
  — curated cell-cell-communication data (ventolab/cellphonedb-data, the
  Teichlab-maintained CellPhoneDB dataset). Ligand-receptor/adhesion-complex
  specific: a hit is precise, strong evidence; a miss doesn't rule anything
  out for non-surface proteins. `complex_members` also gives real multi-
  subunit membership (e.g. integrin heterodimers) — a better "other subunits"
  source than HPA's interaction-count proxy for surface/adhesion complexes.
- `uniprot_annotation(gene)` (`mcp_hpa/uniprot_client.py`, added 2026-08-24) —
  the protein's real UniProt entry: accession, protein name, plain-text
  FUNCTION description, keywords, and PTM/topology feature-type counts. Added
  because `pubmed_ptm_glycosylation` alone is often just noisy "protein X as
  an activation marker" literature with nothing to do with PTMs, and nothing
  previously checked the protein's own documented binding MECHANISM — e.g. a
  galectin's function text literally says "Lectin that binds beta-
  galactoside...", a mechanism-level artifact-risk signal AF3's bare-sequence
  fold can't represent, independent of which residues the model happens to
  place in contact. Verified live: the agent now proactively calls this and
  correctly cites the function text when landing on LIKELY_ARTIFACT_PTM,
  without needing after-the-fact correction.

Not yet built (still a gap for a real known-complex/novelty check on
*intracellular* proteins, since CellPhoneDB only covers surface/secreted):
- `complex_members(gene)` via CORUM — general complex membership beyond
  cell-surface/adhesion complexes.

## Per-candidate agent loop

1. Load AF3 confidence metrics for the candidate.
2. Agent calls HPA tools for both proteins (expression, subcellular location,
   protein class).
3. Agent applies compartment/coexpression logic (same-compartment intracellular
   pair needs coexpression; secreted/receptor pair does not).
4. Agent calls `complex_members` for both proteins. If either is a member of a
   known complex not fully represented in the AF3 pair, flag
   `LIKELY_SUBCOMPLEX` and record which additional subunit(s) should be added
   to a follow-up AF3 multimer run.
5. Agent calls `known_interactions`; if already annotated, flag
   `ALREADY_KNOWN` (not novel).
6. If evidence is sparse for either protein, agent calls `pubmed_cooccurrence`
   before concluding.
7. Agent emits a forced-schema verdict:

```json
{
  "pair": ["GENE_A", "GENE_B"],
  "verdict": "CONFIRMED_NOVEL | LIKELY_SUBCOMPLEX | ALREADY_KNOWN | IMPLAUSIBLE | LIKELY_ARTIFACT_PTM | INSUFFICIENT_EVIDENCE",
  "confidence": "high | medium | low",
  "rationale": "1-3 sentences citing the actual evidence pulled",
  "coexpression_evidence": "...",
  "subcellular_evidence": "...",
  "ptm_glycosylation_evidence": "what pubmed_ptm_glycosylation showed, and relevance judgment",
  "cooccurrence_evidence": "what pubmed_cooccurrence showed",
  "known_interaction_evidence": "what string_known_interaction / cellphonedb_known_interaction showed",
  "other_subunits": ["..."],
  "follow_up": "e.g. re-run AF3 multimer with subunit X, or literature search for Y"
}
```

## Orchestration / plumbing

- Python, Anthropic SDK, MCP Python SDK for `mcp_hpa/`.
- One agent conversation per candidate; run a bounded-concurrency pool
  (asyncio + semaphore) since HPA lookups repeat across candidates that share
  a protein — **cache HPA/CORUM/STRING responses locally** (`data/cache.db`,
  keyed by gene) so a batch of N candidates over M unique proteins only fetches
  each protein once.
- Log full transcript per candidate to `log/<pair>.json` (mirrors the
  CellSleuth convention of auditable per-run logs) — needed since these
  verdicts should be explainable / re-checkable by a human later.
- Final aggregation script produces a ranked CSV/HTML report, one row per
  candidate, sortable by verdict + AF3 confidence.

## Traceability: every conclusion must show its evidence

Early versions logged only the assistant's turns to `log/<pair>.json` — the
tool_runner never surfaces the actual tool *results* to calling code, only
the tool_use requests, so a rationale like "STRING found no hit" couldn't be
checked against what STRING actually returned. Fixed via `LoggingSession` in
`scripts/agent.py`: a thin proxy around the MCP `ClientSession` that logs
every `call_tool(name, arguments)` → raw JSON result before returning it,
since `async_mcp_tool` only ever calls `.call_tool()` on the session object
it's handed (verified against the SDK source, not assumed). Every candidate's
log now carries `_tool_calls`: an ordered list of every tool invocation with
its full input and raw output.

`scripts/build_report.py` renders this into a single self-contained HTML
file: one card per candidate showing the verdict, every evidence field the
agent cited, and an expandable ledger of the underlying tool calls — so any
conclusion in the report can be checked against the actual HPA/PubMed/
STRING/CellPhoneDB response, not just the model's paraphrase of it.

## Analyzing the folded structure, not just its confidence scores

`fold_candidate.py` only parsed summary scores (ipTM/pTM/ranking_score) — it
never looked at the actual 3D model. `scripts/analyze_fold.py` (built after
the user asked "once folded, do we have the skills to analyse the folded
structures?") loads the mmCIF (`gemmi`, no crystallographic info needed since
AF3 output has a dummy 1×1×1 cell — brute-force numpy pairwise distances
instead of a periodic neighbor search), computes per-chain-pair interface
residues (any heavy-atom contact within 5 Å) and per-interface pLDDT, and
fetches each chain's real UniProt feature annotations (Modified residue /
Glycosylation / Disulfide bond / Lipidation, with exact residue positions) to
check whether a known PTM/glycosylation site actually falls ON the modeled
interface. This turns the LIKELY_ARTIFACT_PTM reasoning from a protein-level
guess ("this protein is glycosylated somewhere") into a residue-level check
("residue 173 is a known phosphosite AND it's in the modeled interface").
Wired into both `fold_tool.py` (the agent's live fold_complex tool gets a
condensed version of this back automatically) and `run_follow_up_folds.py`
(the full analysis is saved alongside each backfill job's result).

## Membrane topology: a residue-level biological-plausibility check

Per user request ("check whether the interaction is plausible biologically
— if one is strictly intracellular and the other is membrane, unlikely, but
note proteins are dynamic"): `analyze_fold.py` also fetches each chain's
UniProt "Topological domain"/"Transmembrane" features (Cytoplasmic /
Extracellular / Lumenal, with exact residue boundaries — confirmed live on
EGFR: extracellular 25-645, TM 646-668, cytoplasmic 669-1210) and reports
which side(s) of the membrane the modeled interface actually falls on
(`interface_topology_a/b`). This is a hard biophysical check, not a soft
heuristic: an intact membrane is a real barrier no protein dynamism crosses,
so a strictly-intracellular partner meeting a membrane protein's
*cytoplasmic* interface is plausible (kinases/adaptors on a receptor's tail
routinely do this) but meeting its *extracellular* interface is not.
Deliberately does **not** hard-code a veto, though — the module only
surfaces which topological domain the interface falls in; judging
plausibility against the *other* chain's known compartment (from HPA,
already seen earlier in the same agent conversation) is left to the agent,
since proteins do moonlight/shuttle and a rigid auto-reject would sometimes
be wrong. Verified live on the real ITGA8/F11R/ITGB1 fold: the modeled
ITGA8-F11R interface spans Cytoplasmic + Extracellular + Transmembrane
simultaneously on *both* chains — flagged in the report as independent
evidence alongside the PTM-site finding for this candidate.

**Correction made after checking the flag rate across all 20 backfill jobs**
(45/79 interfaces): the first version flagged ANY interface touching more
than one topological zone, which over-fired — Cytoplasmic+Transmembrane or
Extracellular+Transmembrane alone is normal (real interfaces include TM-
helix-boundary residues). Narrowed to the actual physical impossibility:
touching BOTH Cytoplasmic AND Extracellular residues in one contact. Also
checked whether interface pLDDT reliably explains the remaining flagged
cases (hypothesis: low-confidence sprawling interfaces produce this pattern
spuriously) — empirically it does NOT (several flagged interfaces have
pLDDT 70-80+). The real, more fundamental explanation: AF3 never models a
lipid bilayer at all, so even a confident fold can place topologically
opposite regions spatially close with nothing keeping them apart. The report
and prompts now say this plainly and point to the 3D viewer for direct
inspection, rather than overselling the flag as automatic proof.

## Showing the structure itself in the report

Per user request ("include the structures there, when available/useful"),
`build_report.py` now embeds an interactive 3D viewer for any candidate that
has a real folded structure (from the backfill or from the agent's own
live `fold_complex` call) — not just the ipTM/pTM numbers. Uses 3Dmol.js (CDN
script tag), with the model's mmCIF text embedded directly in the page (not
referenced by file path) so the report stays self-contained and works even
if `fold_runs/` isn't shipped alongside it. Lazy-initialized: the WebGL
viewer for a given card is only created on first "Show 3D structure" click,
not at page load, so a report with many folded structures doesn't try to
spin up dozens of WebGL contexts at once. Cartoon colored by chain, with
modeled interface residues highlighted in orange when the full per-residue
list is available (backfill jobs keep it; agent-triggered live folds only
carry interface *counts* in their condensed summary to save tokens, so those
degrade gracefully to chain-only coloring).

## Prioritizing what to actually validate experimentally

Per user request ("I also want to know which complexes to validate
experimentally — ie not proven in literature"), `build_report.py` opens with
a ranked "Recommended for experimental validation" table instead of leaving
the reader to scan 75 cards. Deliberately three explainable tiers rather than
a single opaque score — each tier says *why* and, where relevant, *what to do
next*, not just a number:
- **CONFIRMED_NOVEL** → top pick: no known/curated hit anywhere, plausible.
- **LIKELY_SUBCOMPLEX** → validate the completed complex named in
  `other_subunits`, not the original binary pair — folding it (see below) can
  confirm this before committing wet-lab time.
- **LIKELY_ARTIFACT_PTM** → resolve the PTM/glycosylation concern first; if
  `fold_complex`'s structural analysis already ran and found no PTM site at
  the interface, the note upgrades to say the concern is resolved.

`ALREADY_KNOWN` (already proven — that's the literal "not proven in
literature" exclusion) and `IMPLAUSIBLE` are excluded entirely;
`INSUFFICIENT_EVIDENCE` is called out separately as needing more evidence,
not silently dropped. Ranked within each tier by AF3 ipTM.

## Closing the loop: actually re-folding a follow-up

A `follow_up` like "re-run AF3 multimer with subunit X" used to just be a
text recommendation. `skills/fold-candidate/` + `scripts/fold_candidate.py`
execute it for real: fetch the proposed protein's sequence from UniProt,
build a standard AF3 input JSON, and run it through this box's existing
local AlphaFast (AF3) setup (`/data/yzy21/yy/af/alphafast` — pre-built
databases, weights, and Singularity container already present, 3 GPUs
available) via the project's own `run_alphafast.sh` entrypoint. This is
deliberately a separate, human-invoked skill rather than a tool available to
the automatic per-candidate agent — AF3 folding costs real GPU minutes per
job, so it should be spent on specific promising follow-ups, not automatically
on every one of a large batch.

## Open design questions (see todo.md)

- Exact AF3 output schema/location to ingest.
- Whether STRING/CORUM/PubMed lookups go through public REST APIs directly
  (simplest) or through existing MCP connectors once authorized.
- Species scope (HPA is human-only — confirm all candidates are human
  proteins, or need a fallback for non-human).
