# todo — deorphanisation pipeline

## 2026-08-24 session summary — deterministic gate, UniProt tool, review agent, full clean rerun

**What was built** (all wired into `run_pipeline.py`'s default flow, each
stage skippable with `--skip-*`):
- `scripts/deterministic_gate.py` — a code-based, reproducible check over
  each candidate's REAL structured tool outputs (STRING channel scores,
  CellPhoneDB hits, HPA compartment/expression data, fold
  `structural_analysis`, UniProt function text), flagging specific,
  checkable contradictions against the LLM verdict rather than trusting the
  verdict's own free-text summary of the evidence.
- `scripts/reconsider_with_gate.py` — feeds each flagged contradiction back
  to the model for genuine reconsideration (not a rule guessing the
  replacement label).
- `mcp_hpa/uniprot_client.py` + a 9th MCP tool, `uniprot_annotation` — gives
  the live triage agent a protein's real UniProt entry (function text,
  keywords, PTM/topology feature counts), auto-discovered via
  `list_tools()`, no separate wiring.
- `scripts/review_run.py` — a review agent auditing a completed run:
  deterministic checks (malformed fields, missing tool-call ledgers, tier-
  selection ratio) plus an LLM consistency audit (`claude-sonnet-5`) of
  each validation strategy against the real HPA data it should be grounded
  in.
- `scripts/design_validation.py` — designs a PLA/stimulation-assay/co-IP
  validation strategy (real HPA tissues/cell types, not guessed) for every
  candidate in the recommended tier.
- One flat summary table in `report.html`: **Pair | Worth validating?
  (Yes/No) | Why**, covering all 75 candidates — replaced an earlier
  misread ask (a "was a fold executed y/n" table) and the older split
  "recommended + collapsed rejected" structure.

**Real bugs found and fixed, roughly in the order they surfaced:**

1. **Wrong log directory silently zeroed every tool-call ledger** —
   `build_report.py` was pointed at a nonexistent `data/logs` (the real dir
   is `log/`); missing dir → empty ledger, no error. Caught by the review
   agent's `missing_tool_call_ledger` check.
2. **`method` field not actually enum-constrained** — the docstring asked
   for an enum in prose, but the model sometimes wrote a full sentence
   instead (this + #3 below were what actually read as "cut off text").
   Fixed with a real `Literal[...]` type hint (confirmed `beta_async_tool`
   compiles this to a JSON-schema `enum`, enforced by the API).
3. **The actual "cut off / weird commas" root cause**: `target_tissues`/
   `cell_types` are typed `list[str]`, but tool-call output isn't strictly
   schema-validated — the model sometimes returned a plain comma-separated
   string (or once, literally the Python `str()` of a list). `",
   ".join(a_string)` iterates *characters*, producing garbage like `"l, y,
   m, p, h, o, i, d, ..."`. Fixed with `strategy_utils.py::
   normalize_str_list()` (parenthesis-aware split + bracket/quote
   stripping), applied at the actual bug site (`design_one()` reads
   `block.input` directly — a fix inside the tool function body would have
   been dead code, since that function is never invoked in this manual
   create()-and-parse flow) and defensively in `build_report.py`.
4. **The review agent's own first version had a self-inflicted bug** — its
   audit prompt referenced CSV columns (`subcellular_evidence`,
   `coexpression_evidence`) that don't exist in `verdicts_full.csv`,
   flagging all 35/43 candidates it checked as "evidence is None but
   rationale cites specifics" — the audit citing its own missing input, not
   a real pipeline problem. Fixed by pointing it at the real columns and a
   fresh HPA fetch.
5. **A full fresh rerun failed completely on the first attempt** — every
   one of 75 triage calls hit `Agent run failed: Connection error` (the
   same local-proxy flakiness already fixed in `design_validation.py`/
   `review_run.py`, never applied to the core triage client). 26 real AF3
   folds had actually completed before the connection dropped and
   discarded everything — real GPU time wasted for zero output. Fixed with
   a shared `anthropic_client.py::make_client()` (proxy bypass) + per-
   candidate retry; a pre-run backup (made proactively) made recovery
   trivial. Second attempt: 0/75 failures.
6. **The deterministic gate's first version tried to fully replace verdicts
   with rule output — changed 62/75 candidates (83%)**, itself a red flag.
   Root causes: treating "no fold was run" as "PTM concern cleared" (erased
   real literature-based PTM calls), and would have wrongly flipped
   `IGHG1/FCGR2B` (a textbook antibody-Fc-receptor pair) away from
   ALREADY_KNOWN just because STRING/CellPhoneDB don't happen to cover it.
   Redesigned to flag-then-reconsider (`find_contradictions()`, plural —
   an earlier singular version returned on the first match and could hide a
   second, genuinely different concern on the same candidate).
7. **HPA-implausibility rule wrongly fired on normal receptor-cytoplasmic-
   tail biology** — a protein tagged BOTH "predicted membrane proteins" AND
   "predicted intracellular proteins" (very common: a receptor's own
   cytoplasmic tail earns the "intracellular" tag too) isn't a compartment
   mismatch. Caught live when the model, asked to reconsider `NPR3/RAB8B`,
   correctly explained this — proof the flag-then-reconsider design catches
   the gate's own flaws before they do damage. Fixed to require a genuinely
   `secreted` signal, not just any membrane-adjacent tag.
8. **A plain disulfide bond was treated as the same artifact risk as an
   actual glycosylation site** — disulfide bonds are structurally well-
   represented by AF3, unlike glycans. Fixed: only Glycosylation/Lipidation
   feature types count now.
9. **The reconsideration pipeline was silently masking model refusals —
   the most serious bug of the session.** `reconsider_with_gate.py`'s calls
   were silently REFUSING on `claude-fable-5` (`stop_reason: "refusal"`,
   empty tool input `{}`), and the code merged that empty dict onto the
   original row — output byte-identical to the input, indistinguishable
   from "genuinely reconsidered, nothing changed." All 17 flagged
   candidates in one run showed `verdict -> same verdict`, which looked
   like normal, expected behavior and was almost accepted. Caught only by
   manually diffing rationale text and finding it letter-for-letter
   unchanged. Fixed three ways: switched the script to `claude-sonnet-5`;
   added explicit `stop_reason` / missing-field / identical-output
   detection (raises loudly instead of silently no-op-ing); found and
   fixed the identical latent bug in `review_run.py::llm_audit()`
   (`result.get("consistent", True)` defaults to True on an empty refusal)
   before it was ever observed firing. Re-ran properly: 17/17 verdicts
   genuinely changed, each with real evidence-engaged reasoning.
10. **STRING alone, with no literature, was treated as proof of
    ALREADY_KNOWN** — STRING's `database` channel can reflect curated
    pathway/interactome membership (confirmed: `EFNA4/EPHB4`'s 0.5 database
    score was family-level Eph-ephrin pathway annotation, not a citation
    for that specific pair), not a paper establishing this exact
    interaction. Fixed to require BOTH a strong STRING channel AND a real
    `pubmed_cooccurrence` hit for this pair (CellPhoneDB stays sufficient
    on its own). Found and corrected 3 real candidates wrongly
    ALREADY_KNOWN on STRING alone (`ADGRL1/DCN`, `CXCR1/GNAQ`,
    `GNAI3/GRM3`).
11. **The obvious generalization of the galectin check — UniProt's
    "Lectin" keyword — was tested and found too broad.** `CLEC2B`, `SELE`,
    `KLRD1` (common C-type lectin DOMAIN receptors in this dataset) all
    carry that keyword but describe plain protein-protein recognition, no
    carbohydrate-binding mechanism. Replaced the hardcoded gene list with a
    live UniProt function-text regex match (carbohydrate/galactoside/
    sialic acid/glycan/mannose/fucose/etc.) — verified to fire correctly on
    `LGALS1`/`LGALS3`/`SIGLEC1` (a different lectin family, confirming real
    generalization) and correctly not fire on `CLEC2B`/`SELE`/`KLRD1`. This
    surfaced one more logic conflict — the "fold cleared the PTM concern"
    rule didn't know about the mechanism-level check and would have wrongly
    suggested downgrading `LGALS1/CLEC2D` off LIKELY_ARTIFACT_PTM — fixed
    to defer to the mechanism check.

**Final state**: 75/75 candidates, 0 technical failures, review agent fully
clean. Verdict distribution: `IMPLAUSIBLE` 33, `LIKELY_ARTIFACT_PTM` 23,
`ALREADY_KNOWN` 6, `INSUFFICIENT_EVIDENCE` 6, `CONFIRMED_NOVEL` 4,
`LIKELY_SUBCOMPLEX` 3; 30/75 (40%) recommended for experimental validation.

**The lessons that matter most, in priority order:**
1. **A "0 changes" or "N/N unchanged" result from an LLM update step is NOT
   evidence it worked** — it's equally consistent with every call silently
   refusing and the code masking that as a no-op. Any code merging
   `{**original, **model_output}` must positively verify real output
   happened (check `stop_reason`, check required fields, check the result
   isn't suspiciously identical to the input).
2. **A code-only deterministic rule is a good pre-filter for what's
   mechanically checkable, but forcing it to fully replace LLM judgment
   produces its own real regressions** (62/75 churn, IGHG1/FCGR2B,
   NPR3/RAB8B) — the safe architecture is gate-flags-then-LLM-reconsiders,
   never gate-decides. Always verify a new heuristic's firing rate before
   trusting it (near-0% or near-100% is a red flag).
3. **A `list[str]`/`Literal[...]` type hint on a tool parameter is a
   request, not a guarantee** — the API does not strictly enforce tool-call
   output against the JSON schema even under a forced `tool_choice`; code
   consuming `block.input` must normalize/validate defensively.
4. **A review/audit layer's own inputs need the same scrutiny as the thing
   it's reviewing** — its first version's "findings" can be a self-
   inflicted artifact (wrong CSV columns), not a real signal.
5. **A fix to one script's proxy/connection handling doesn't protect
   others using the same client pattern** — grep the whole codebase for
   the pattern rather than assuming it's contained.
6. **Always back up state before a costly, hard-to-reverse rerun.**
7. **When generalizing a hardcoded check, test the obvious alternative
   against real counter-examples before trusting it** — the "Lectin"
   keyword looked like a clean generalization and would have badly
   over-fired; the real fix required checking actual function-text content.

## Added 2026-08-23 — experimental validation strategies added

User request: "for the pairs that are worth validating - come up with a
validation strategy - for both colocalisation e.g. PLA or stimulation.
remember to put the tissue needed... based on where they are most expressed
(use the HPA tool)". Built `scripts/design_validation.py`: for each of the 43
candidates in the CONFIRMED_NOVEL/LIKELY_SUBCOMPLEX/LIKELY_ARTIFACT_PTM tiers
(same `classify_validation` used by the report, imported not reimplemented),
asks the model to pick PLA vs. a stimulation/functional assay vs. co-IP based
on real HPA subcellular/secretome/expression data, name 2-4 real tissues +
cell types grounded in actual nTPM/protein-intensity/single-cell numbers, and
give protocol notes, controls, and caveats (explicitly flagging the
PTM/glycosylation-artifact concern for LIKELY_ARTIFACT_PTM candidates and
suggesting a glycosidase/deglycosylation control arm). Output:
`data/validation_strategies.json` (43/43 succeeded), now rendered into
`report.html` both as a "Suggested validation (tissue)" column in the top
summary table and as a full expandable strategy block on each candidate card.

Two bugs hit and fixed along the way, both worth remembering:
- **`claude-fable-5` deterministically refuses this whole task category** —
  confirmed via bisection that even a bare "what experiment would show X and Y
  interact?" with no system prompt, no tools, no PTM wording, refuses
  (`stop_reason: refusal`, category `bio`). Not a wording issue — the task
  itself (designing wet-lab protocols for named proteins) trips fable-5's
  classifier. Fix: this one script runs on `claude-sonnet-5` instead
  (confirmed reliable, no refusals, real HPA-grounded output) — the rest of
  the pipeline stays on fable-5.
- **Separately, this exact request shape (long system prompt + wide 8-field
  tool schema) intermittently gets dropped by the local dev proxy**
  (`httpx.RemoteProtocolError: Server disconnected without sending a
  response`) — reproduced consistently even after 10 retries with backoff.
  Bisected: neither the system prompt alone nor the tool schema alone
  triggers it; the combination does, and only through the proxy — confirmed
  `api.anthropic.com` is directly reachable from this host. Fix:
  `design_validation.make_client()` builds its `AsyncAnthropic` with
  `http_client=httpx.AsyncClient(trust_env=False)` to bypass the proxy for
  just this call, while everything else (HPA/UniProt/etc.) keeps using it.

## LATEST (2026-08-21) — report clarity fixes, per direct user feedback

Three concrete complaints, all fixed in `build_report.py`:
1. **"is fold evidence actually being used and considered?"** — genuinely
   unclear before this fix, because there are two DIFFERENT mechanisms and
   the report blurred them together. Checked precisely: only 1/75 candidates
   (ITGA8/F11R) has a *live*, in-conversation `fold_complex` call (the
   original 75-candidate run predates that tool's existence); the other 19
   backfilled candidates got their fold evidence via the separate
   `reconsider_with_fold.py` pass. Both mechanisms DO feed evidence back into
   the verdict (verified), but now say so explicitly: each card gets a
   `🧬 FOLD-CHECKED (live)` or `🧬 FOLD-CHECKED (reconsidered)` badge with a
   tooltip explaining the mechanism, and the top stats bar shows the exact
   split (1 live / 20 reconsidered / 54 no-fold) instead of one blended count.
2. **"label the subunits of the complex"** — the 3D viewer colored by chain
   letter (A/B/C) with no indication of which gene each letter was. Fixed:
   real 3Dmol.js text labels now placed at each chain's centroid in the
   structure itself, plus an explicit "Chains: A = FCGRT · B = DLG4 · B2M ="
   text legend. Interface descriptions in the EXECUTED block also now say
   "A (FCGRT)–B (DLG4)" instead of bare "A–B".
3. **"a few pairs that should be rejected... aren't in the table"** — the
   validation summary only ever listed candidates worth prioritizing; IMPLAUSIBLE/
   ALREADY_KNOWN/INSUFFICIENT_EVIDENCE candidates were silently excluded,
   only mentioned as an aggregate count. Fixed: added three expandable
   "Rejected" sections (one per excluded verdict) listing every excluded
   pair explicitly with its reason — nothing is silently hidden anymore.

## Close the loop: feed fold evidence BACK into the verdict

Found via user feedback ("the structural analyses aren't accounted in the
reasoning at the moment"): the 20 backfilled follow-up folds' real structural
evidence (interface pLDDT, PTM-at-interface, membrane topology) was rendered
in the report's "EXECUTED" block but never fed back to actually revise the
verdict/rationale sitting above it — only the one live in-agent
`fold_complex` case (ITGA8/F11R) got that treatment, since it happened in the
same agent turn. Built `scripts/reconsider_with_fold.py`: sends each
candidate's original verdict + the real fold's ipTM/structural_analysis back
to Fable 5 (single-shot, forced `record_verdict` tool call, no tool loop
needed since all evidence is already provided) and has it genuinely
reconsider — not just restate the old verdict.

**Result: 16 of 20 verdicts changed** (4 unchanged, including ITGA8/F11R,
which the automated reconsideration independently reproduced as
`LIKELY_ARTIFACT_PTM` — the same verdict the earlier live-agent run reached,
a good consistency check). Most changes were LIKELY_SUBCOMPLEX/CONFIRMED_NOVEL
downgrading once the proposed extra subunit demonstrably did NOT rescue the
interface (e.g. FCGRT/DLG4 → IMPLAUSIBLE, with the rationale explicitly
contrasting the cleanly-folded FCGRT-B2M interface, the real obligate
partner, against the low-confidence/topologically-incoherent FCGRT-DLG4
one). **Final verdict distribution after this pass**: LIKELY_ARTIFACT_PTM 34,
IMPLAUSIBLE 13, INSUFFICIENT_EVIDENCE 12, ALREADY_KNOWN 7, CONFIRMED_NOVEL 5,
LIKELY_SUBCOMPLEX 4 — a much more sober, evidence-grounded picture than
before this pass (CONFIRMED_NOVEL dropped from 12→5, IMPLAUSIBLE rose 5→13).
`data/verdicts_full.csv` + `data/report.html` rebuilt with these revised
verdicts.

Also fixed a real silent-failure bug the user caught by actually trying the
report: `initViewer()`'s 3D structure viewer did nothing (no error, nothing
in console) if the 3Dmol.js CDN script failed to load — the likely cause if
viewing from a network without direct access to `3dmol.org`. Now shows a
visible error message for that case, a missing-data case, and any render
exception, instead of failing silently.

## Membrane-topology plausibility check (2026-08-21, earlier this session)

Per user request ("check whether the interaction is plausible biologically —
if one is strictly intracellular and the other is membrane, unlikely, but
note proteins are dynamic"):
- [x] `analyze_fold.py` now also fetches UniProt "Topological domain"/
      "Transmembrane" features (Cytoplasmic/Extracellular/Lumenal, exact
      residue boundaries) and reports which side of the membrane the modeled
      interface falls on (`interface_topology_a/b`). Hard biophysical check
      (an intact membrane is a real barrier), not a soft heuristic — but
      deliberately doesn't hard-code a veto; the agent judges plausibility
      against the *other* chain's HPA-known compartment itself, since
      proteins do moonlight/shuttle. Verified live on the real ITGA8/F11R/
      ITGB1 fold: the interface spans Cytoplasmic+Extracellular+Transmembrane
      on *both* chains simultaneously — flagged in the report alongside the
      PTM-site finding already there for this same candidate.
- [x] `agent.py` SYSTEM_PROMPT strengthened: explicit guidance on the
      strictly-intracellular + membrane-protein case (cytoplasmic-side
      contact = plausible, extracellular-side = not, regardless of protein
      dynamism), pointing to `fold_complex`'s topology data to actually check
      rather than guess when a fold is done.
- [x] `fold_tool.py`'s `fold_complex` docstring updated with this as a third
      good reason to invoke the tool (alongside missing-subunit and
      PTM-overlap hypotheses).
- [x] `build_report.py` renders `interface_topology_a/b` per interface.
- [x] **Self-caught and fixed an over-firing flag.** First version flagged
      ANY interface touching >1 topological zone (45/79 fired) — too blunt,
      since Cytoplasmic+Transmembrane alone is normal (real interfaces
      include TM-boundary residues). Narrowed to the actual impossibility:
      Cytoplasmic AND Extracellular together (barely changed the count,
      45/79 → still 45/79 — most cases were genuinely both-sides, not a TM-
      adjacency false positive). Then checked whether low interface pLDDT
      explains the remaining flagged cases (hypothesis: sprawling low-
      confidence interfaces produce this pattern spuriously) — empirically
      it does NOT (several flagged cases have pLDDT 70-80+, e.g.
      `btn3a2_klrd1_btn3a1_klrc1` A-C at 80.3/80.1). Real explanation: AF3
      never models a lipid bilayer at all, so even a confident fold can place
      topologically opposite regions spatially close with nothing keeping
      them apart. Report/prompts now say this plainly and point to the 3D
      viewer for direct inspection, rather than overselling the flag as
      automatic proof of anything.
- [x] Retroactively backfilled `structural_analysis` (incl. topology) onto
      all 20 completed folds + the ITGA8/F11R live-agent fold — hit UniProt
      rate-limiting mid-backfill (confirmed via `curl -v`: `rest.uniprot.org`
      specifically resetting TLS connections while google.com/
      proteinatlas.org worked fine), waited for it to clear, retried the 5
      still-missing jobs gently (3s spacing) once recovered. All 21/21 now
      have full structural_analysis including topology. `data/report.html`
      rebuilt (75 candidates, 21 structures, corrected topology flagging).

## STATUS: core pipeline complete (2026-08-20 → 2026-08-21)

**Everything planned is built, live-tested with real API calls + real GPU
folds, and working.** `data/verdicts_full.csv` + `data/report.html` are the
final deliverables: 75 candidates, all traceable to raw tool evidence, 21
real folded structures viewable inline (20 from the follow-up backfill + 1
from a live in-agent fold_complex call), and a "recommended for experimental
validation" summary at the top.

**Final verdict distribution (superseded — see "LATEST" section at top of
this file for the current numbers after the close-the-loop reconsideration
pass):** LIKELY_ARTIFACT_PTM 31, CONFIRMED_NOVEL 12, LIKELY_SUBCOMPLEX 12,
INSUFFICIENT_EVIDENCE 8, ALREADY_KNOWN 7, IMPLAUSIBLE 5.

**Closing-the-loop demo worth knowing about:** ITGA8/F11R originally verdicted
`LIKELY_SUBCOMPLEX` with `follow_up` = "re-run AF3 as a trimer with ITGB1
to test whether F11R docks at the intact headpiece." A later live run of the
in-agent `fold_complex` tool did exactly that for real — and the real result
(interface didn't improve, PLUS two ITGA8 N-glycosylation sites and a
topologically-impossible F11R cytoplasmic phosphosite landed on the modeled
interface) flipped the verdict to `LIKELY_ARTIFACT_PTM`. That improved,
better-evidenced verdict has been merged into the real
`data/verdicts_full.csv` (not left as a throwaway test) — this is the system
working exactly as intended end-to-end.

**Two real bugs found and fixed via that live validation** (a mocked test
alone could never have caught either, since it never went through a real API
round-trip):
1. `fold_complex` returned a raw Python dict; the Anthropic tool-result
   contract requires `str` or content blocks — silently passed Python's
   (non-enforcing) type hints, only failed at the real API. Fixed: returns
   `json.dumps(result)`.
2. `fold_complex` (a local tool, not MCP-routed) was never captured into
   `_tool_calls` at all — only `LoggingSession` (the MCP proxy) logged calls.
   Fixed: `make_fold_tool` now takes the same log list and appends its own
   entries in the same shape.

**Also fixed during the fold-backfill work:**
- `fetch_sequence()` silently corrupted sequences when a UniProt gene-symbol
  search matched more than one reviewed entry (spliced a second record's
  `>sp|...` header into the sequence string). Now requires an exact `GN=`
  match or raises clearly.
- `fetch_sequence()` had zero retry; now uses the shared `get_with_retry`.
  Bumped `MAX_RETRIES` 3→5 after seeing a UniProt reset outlast the original
  backoff window.
- `run_alphafast()` now captures and surfaces real subprocess output on
  failure instead of a bare "exit status 1".
- A real cross-process GPU race (`gpu_lock`, `flock`-based) — coordinates
  our own processes; genuine OOMs from *other users* on this shared box
  remain possible and are not something we can (or should) prevent.

**Known deliberate gap:** CORUM (general/intracellular complex membership) —
deprioritized by the user; CellPhoneDB only covers cell-surface/adhesion
complexes.

Everything else below this line is the detailed build history — see
design.md for architecture, README.md for how to run it. CORUM (general,
non-surface complex membership) is the one deliberately-deprioritized gap.

## 0. Unblock (need from user)
- [ ] Location/format of the real AF3 novel-PPI candidate list (the CSV schema
      in `scripts/ingest.py` — protein_a, protein_b, iptm, ptm, pae_interaction,
      model_path — is a guess; confirm/adjust against the real file)
- [x] Fable 5 API key — provided, stored in `.env` (gitignored)
- [ ] Confirm all candidates are human proteins (HPA is human-only)
- [ ] Approx. number of candidates / unique proteins (sizes concurrency +
      caching design — current default concurrency is 4)

## 1. HPA MCP server (`mcp_hpa/`) — DONE
- [x] HPA public API shape confirmed live: `search_download.php` (fuzzy
      gene→Ensembl resolution, exact-matched client-side) + per-gene
      `proteinatlas.org/<ensembl_id>.json` (full profile)
- [x] `hpa_expression(gene)`, `hpa_subcellular_location(gene)`,
      `hpa_protein_class(gene)` implemented in `mcp_hpa/server.py`
- [x] Local sqlite cache (`data/cache.db`, 30-day TTL) in `mcp_hpa/hpa_client.py`
- [ ] HPA's "specific nTPM/nCPM" fields only list *enriched* tissues/cell
      types, not a full expression matrix — fine for the coexpression
      heuristic, but a real quantitative overlap check would need the bulk
      `rna_tissue_consensus.tsv` download. Not done — flag if verdicts seem
      too trigger-happy on "no overlap".

## 2. Secondary evidence tools
- [x] `pubmed_cooccurrence(gene_a, gene_b)` — E-utilities esearch/esummary,
      `mcp_hpa/pubmed_client.py`. Weak novelty proxy; verified on EGFR/EREG
      (185 hits, correctly read as ALREADY_KNOWN) and RUNX1/IKZF1 (102 hits,
      correctly read as indirect/leukemia-genetics co-mention, not a known PPI)
- [x] `pubmed_ptm_glycosylation(gene)` — PTM/glycosylation literature screen,
      since AF3 predicts from bare unmodified sequence (no glycans, no PTMs by
      default) and a confident interface can be a real-world artifact if the
      true surface is glycosylated or PTM-gated. New verdict category
      `LIKELY_ARTIFACT_PTM` added to `record_verdict` for this case
- [x] Retry/backoff on NCBI 429s (`_get_with_retry` in `pubmed_client.py`) —
      hit this for real during a 2-candidate concurrency-2 smoke test; NCBI's
      unauthenticated rate limit is ~3 req/sec
- [x] `string_known_interaction(gene_a, gene_b)` — STRING API
      (`mcp_hpa/string_client.py`), combined score + evidence-channel
      breakdown (database/experiments vs. textmining-only). Verified: EGFR/
      EREG and ITGA2/ITGB1 → high database+experiments (correctly
      ALREADY_KNOWN); RUNX1/IKZF1 → 0.941 combined but textmining=0.911,
      database=0 (correctly read as NOT curated evidence, stayed
      CONFIRMED_NOVEL)
- [x] `cellphonedb_known_interaction(gene_a, gene_b)` +
      `cellphonedb_complex_members(gene)` — curated cell-communication data
      from ventolab/cellphonedb-data (`mcp_hpa/cellphonedb_client.py`), cached
      locally under `data/cellphonedb/`. Verified: EGFR/EREG curated
      ligand-receptor hit; ITGA2/ITGB1 recovered as the known
      "integrin_a2b1_complex" heterodimer via `complex_members`; RUNX1/IKZF1
      correctly no-hit (both intracellular, outside CellPhoneDB's scope)
- [ ] `complex_members(gene)` via CORUM — general (non-surface) complex
      lookup; CellPhoneDB's complex data only covers cell-surface/adhesion
      complexes, so intracellular complex membership (e.g. RUNX1+CBFB) is
      still not covered by any tool
- [ ] Consider an NCBI API key (env var) once running at real batch volume —
      raises the eutils rate limit well above 3 req/sec
- [x] Retry/backoff on 429 unified across all 3 external clients
      (`mcp_hpa/http_retry.py`, shared by hpa_client/pubmed_client/string_client)
      — previously only pubmed_client had it
- [x] Per user 2026-08-19: CORUM / general intracellular complex membership is
      explicitly deprioritized for now — not blocking a first real run

## 3. Agent loop — DONE (v1)
- [x] `scripts/ingest.py` — parses candidate CSV into `Candidate` records
- [x] `scripts/agent.py` — system prompt (compartment/coexpression logic,
      hedged complex/novelty language), per-candidate tool_runner loop over
      Fable 5 + the 3 HPA MCP tools
- [x] `scripts/verdict_tool.py` — forced `record_verdict` tool call (note: must
      be `@beta_async_tool`, not `@beta_tool` — the sync decorator silently
      breaks JSON serialization inside `AsyncAnthropic`'s tool_runner, see
      commit history/scratch notes)
- [x] Per-candidate transcript logging to `log/<pair_id>.json`
- [x] `scripts/run_pipeline.py` — bounded-concurrency batch runner (asyncio
      semaphore, one shared MCP server process per batch)

## 4. Aggregation & review
- [x] `write_report()` in `run_pipeline.py` → ranked CSV (verdict, confidence,
      AF3 metrics, rationale, other_subunits, follow_up) — smoke-tested on
      RUNX1/IKZF1 (→ CONFIRMED_NOVEL, correctly named CBFB as the missing
      obligate subunit) and TTN/FOXP3 (→ IMPLAUSIBLE, correctly read as an
      AF3 false-positive)
- [x] Full 75-candidate real-data run completed (2026-08-19): 0 technical
      failures, ~$41 total spend. Verdicts in `data/verdicts_full.csv`.
- [x] **Re-run with full tool-call traceability completed (2026-08-20)**: 0
      technical failures, ~$39, 874 raw tool calls captured across all 75
      candidates (avg ~11.6/candidate). Verdict distribution: LIKELY_ARTIFACT_PTM
      30, LIKELY_SUBCOMPLEX 13, CONFIRMED_NOVEL 12, INSUFFICIENT_EVIDENCE 8,
      ALREADY_KNOWN 7, IMPLAUSIBLE 5 (same overall pattern as the first run,
      minor shifts from model variance). `data/report.html` built — every
      verdict now expandable to its full raw evidence ledger.
- [x] **Traceable HTML report** — `scripts/build_report.py` renders one card
      per candidate with every evidence field PLUS an expandable ledger of
      every raw tool call (input + full JSON output), so a conclusion can be
      checked against the actual evidence, not just the model's paraphrase.
      This required a fix in `scripts/agent.py`: the tool_runner previously
      only exposed assistant messages, not the tool_result payloads, so a
      `LoggingSession` proxy now wraps the MCP `ClientSession.call_tool` to
      capture every call — verified on a 5-candidate re-run (13 tool calls
      captured for CD69/CLEC2B alone). **The existing `verdicts_full.csv`/
      `log/*.json` from the first full run predate this fix and have no
      tool-call ledger** — the report builder degrades gracefully for those
      (shows a "not captured" note) but a full re-run is needed for complete
      traceability on all 75. Not yet done — ~$40 additional spend, needs a
      go-ahead.
- [ ] Spot-check a handful of verdicts manually against raw HPA pages once run
      on the real candidate list

## 5. Fold-candidate skill (added 2026-08-20)
- [x] `skills/fold-candidate/SKILL.md` + `scripts/fold_candidate.py` — fetches
      UniProt sequences for proposed proteins and actually runs them through
      AlphaFast (local AF3, this box's existing GPU/db/weights/container
      setup at `/data/yzy21/yy/af/alphafast`), rather than just recommending
      a re-fold in `follow_up` text. Uses the standard `run_alphafast.sh`
      entrypoint (not a hand-rolled singularity command).
- [x] Live-validated on RUNX1+CBFB — iptm=0.87 at the interface
      (chain_pair_iptm[0][1]), ptm=0.49 overall (RUNX1 has large disordered
      regions, fraction_disordered=0.56), ranking_score=1.07. Confirms the
      earlier LIKELY_SUBCOMPLEX finding that CBFB is RUNX1's real obligate
      partner (textbook core-binding-factor biology) — good positive control
      for the skill. Took 521s (~8.7 min) end-to-end cold (5.7 min MSA + 1.9
      min inference) on GPU 0.
- [x] **Wired directly into the agent's own tool loop (2026-08-20, per user
      request)** — `scripts/fold_tool.py`'s `fold_complex` tool is now in
      `scripts/agent.py`'s tool list. When Fable 5 proposes a follow-up its
      own tools can settle (chiefly missing-subunit hypotheses), it calls
      fold_complex directly instead of writing a deferred recommendation.
      Guardrails: capped at one fold per candidate (enforced in-tool), and a
      `asyncio.Semaphore` sized to `--fold-gpus` (run_pipeline.py flag,
      default `0,1`) bounds concurrent agent-triggered folds across the whole
      batch. System prompt explicitly excludes PTM/glycosylation hypotheses
      (fold_complex still can't model glycans/PTMs, so it can't help
      LIKELY_ARTIFACT_PTM calls) and warns against folding every
      mutually-exclusive-paralog list as one nonsensical mega-complex.
      Plumbing (semaphore, GPU round-robin, single-use cap) validated with a
      mocked fold function; a real live end-to-end test is still pending
      (see below — blocked on GPU contention from the concurrent backfill).
- [x] **Structural analysis of the folded model (2026-08-20, per user
      request "do we have the skills to analyse the folded structures?")** —
      `scripts/analyze_fold.py`: loads the mmCIF (`gemmi`), computes
      per-chain-pair interface residues (5Å heavy-atom contact, brute-force
      numpy since AF3 output has no real crystallographic cell) + per-
      interface pLDDT, and fetches real positioned UniProt PTM/glycosylation
      features to check if a known site falls ON the modeled interface —
      residue-level evidence for LIKELY_ARTIFACT_PTM instead of protein-level
      literature guessing. Wired into `fold_tool.py` (agent's live tool gets
      a condensed summary back automatically) and `run_follow_up_folds.py`
      (full analysis saved per backfill job). Verified on the real RUNX1+CBFB
      fold: correctly found the interface at RUNX1 residues 47-163 (the Runt
      domain — matches known biology) with high interface pLDDT (90.7/88.4),
      and correctly found none of RUNX1's known phosphosites overlap this
      specific interface (they're in the C-terminal transactivation domain).
      Also extended `http_retry.get_with_retry` to retry on `httpx.TransportError`
      (connection resets), not just HTTP 429 — hit a real UniProt connection
      reset while testing this, affects all 4 clients (HPA/PubMed/STRING/UniProt).
- [x] **"Recommended for experimental validation" summary (2026-08-20, per
      user request "which complexes to validate experimentally, ie not proven
      in literature")** — new top-of-report section in `build_report.py`
      ranking candidates NOT already proven in the literature, in three
      explainable tiers (not an opaque score): CONFIRMED_NOVEL ("top pick"),
      LIKELY_SUBCOMPLEX ("validate the completed complex, not the pair" —
      names the subunit to add), LIKELY_ARTIFACT_PTM ("resolve PTM concern
      first" — auto-promotes to a stronger note if fold_complex's structural
      analysis already checked and found no PTM site at the interface).
      Excludes ALREADY_KNOWN (already proven) and IMPLAUSIBLE (unlikely);
      INSUFFICIENT_EVIDENCE candidates are called out separately as
      needing more evidence rather than silently dropped. Ranked within tier
      by AF3 ipTM. Each card also gets a VALIDATE badge. Verified on the real
      75-candidate data: 55/75 correctly surfaced (20 excluded: 7
      ALREADY_KNOWN + 5 IMPLAUSIBLE + 8 INSUFFICIENT_EVIDENCE), ranking within
      the CONFIRMED_NOVEL tier correctly sorted by ipTM descending.
- [x] **3D structure viewer in the report (2026-08-20, per user request
      "include the structures there")** — `build_report.py` embeds an
      interactive 3Dmol.js viewer for any candidate with a real folded
      structure (backfill or live agent fold), lazy-initialized on click, mmCIF
      embedded inline (self-contained report), interface residues highlighted
      in orange when the full per-residue list is available. Verified
      end-to-end on a synthetic report card wrapping the real RUNX1+CBFB
      structure (well-formed HTML, 424KB CIF embedded correctly, viewer
      controls present).
- [x] **Found and fixed a real cross-process GPU race** — testing the new
      in-agent tool while the 20-job backfill was still running caused a
      genuine CUDA OOM: an ad-hoc test and a legitimate backfill job both
      landed on GPU 0 at once (each process's own semaphore has no idea what
      other processes are doing). Fixed with a real `flock`-based cross-
      process GPU mutex (`fold_candidate.gpu_lock`, `.gpu_locks/gpu_<id>.lock`)
      — every fold job now actually blocks until its GPU is free, regardless
      of which script/process requested it. One backfill job
      (`mfap4_il13ra2_il13_il4r_il13ra1`) OOM'd from this and needs a retry
      once the rest of the backfill completes.

## 6. Stretch (not blocking v1)
- [ ] Domain/motif check (Pfam/InterPro) for known interaction domains
- [ ] Auto-generate follow-up AF3 multimer job configs for LIKELY_SUBCOMPLEX
      verdicts (adding the missing subunit(s)) — partially superseded by the
      fold-candidate skill, which does this on request rather than automatically
- [ ] Full bulk-TSV tissue expression matrix for a real quantitative
      coexpression score, replacing the enriched-tissues-only heuristic
- [ ] Swap direct HPA/STRING/PubMed REST calls for authenticated MCP
      connectors if/when suitable ones are available in this environment
- [ ] Consider HTML report improvements: per-run cost, verdict distribution
      chart (currently just stat tiles)
