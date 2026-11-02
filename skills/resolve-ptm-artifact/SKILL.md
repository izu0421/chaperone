---
name: resolve-ptm-artifact
description: When a candidate's real fold flags a PTM/glycosylation concern at the interface (LIKELY_ARTIFACT_PTM territory), actually try to resolve it structurally instead of leaving it as a caveat - fetch the real UniProt PTM info, re-fold with it modeled, and reach one of a small set of honest conclusions.
metadata:
  mcpmarket-version: 1.0.0
---

# Resolve a PTM/glycosylation artifact concern (bounded, not open-ended)

## Why this exists

`deterministic_gate.py`'s `fold_evidence_status()` already detects the two
concerns that make a modeled interface untrustworthy: a real glycosylation/
lipidation site sitting directly on the interface (`ptm_only_concern`), or
an interface spanning both `Cytoplasmic` and `Extracellular` residues
(`topology_violation`). Until now, hitting either just meant writing
`LIKELY_ARTIFACT_PTM` and stopping - a real, checkable question left as a
permanent caveat.

`chaperone.resolve_ptm_artifact` closes that loop, modeled on
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)'s design:
a small, fixed, reviewable loop with one clear decision per step, not an
open-ended "keep trying until something confident-sounding comes out" agent.
autoresearch bounds itself to a fixed 5-minute-per-experiment budget because
that's what makes ~100 overnight iterations tractable; our AF3 jobs cost
~10-15 minutes each on this box, so the right budget is **2 fold jobs, not
100** - bare, then (if the concern is glycan-fixable) glycan-aware. Stop
there and report honestly, don't keep spending compute hoping for a cleaner
answer.

## When to use this

Any verdict where `ptm_glycosylation_evidence` cites a real Glycosylation
site at (or plausibly at) the modeled interface - most directly, right after
`fold_complex`/`analyze_fold.py` reports `fold_evidence_status ==
"ptm_only_concern"` with a `Glycosylation` type in the trace.

```
python -m chaperone.resolve_ptm_artifact PROTEIN_A PROTEIN_B --name pair_slug --gpu_device N
```

## The decision rubric (the whole point - read this before trusting an output)

1. **Bare fold status is `pass`** -> `RESOLVED_NO_CONCERN`. Nothing to do;
   the concern was never real.
2. **Bare fold status is `topology_violation`** -> `CONFIRMED_ARTIFACT`,
   stop immediately, do NOT run a glycan-aware fold. A `Cytoplasmic`+
   `Extracellular` interface is a hard membrane-topology impossibility - no
   PTM model fixes that, so attempting one would just burn a GPU job to
   confirm what's already certain.
3. **Bare fold status is `ptm_only_concern`, but the flagged type isn't
   glycan-fixable** (e.g. `Lipidation` only) -> `NOT_AUTOMATICALLY_RESOLVABLE`.
   `glycan_lib.py` only models sugars (AF3 ligand CCD chains + covalent
   bonds), not lipid anchors. Stop; this needs the wet-lab follow-up.
4. **`ptm_only_concern` with a `Glycosylation` site** -> run ONE glycan-aware
   re-fold (`fold_candidate.py --with_glycans`), then apply the confidence
   rubric below to the result. This is the only case that spends a second
   fold job.

### Reading the glycan-aware result: calibrate ipTM correctly

This is the part that got corrected mid-development, and it's the most
important thing to get right: **`chain_pair_iptm < 0.5` is not a weak
positive signal, it means AF3 is not predicting a real interaction there at
all** (standard AF-Multimer/AF3 field convention). A residue-level contact
pattern that happens to overlap a real, textbook binding site (e.g. LGALS1's
UniProt-annotated CRD residues) is still not evidence of anything if the
confidence for that specific chain pair is below this bar - it's what a
low-confidence, semi-arbitrary placement looks like, dressed up by a
coincidence that's easy to over-read.

- **A-B interface survives** (chain_pair_iptm(A,B) >= 0.5) **and still
  touches the originally-flagged residues** -> `RESOLVED_NOT_ARTIFACT`. The
  glycan doesn't block the contact; the artifact concern doesn't hold.
- **A-B interface doesn't survive, but chain B confidently engages one of
  the modeled glycan chains directly** (chain_pair_iptm >= 0.5) ->
  `RESOLVED_ALTERNATE_SITE`. Report the new site; this is a sharper,
  testable hypothesis, not a confirmed mechanism.
- **Nothing reaches 0.5, anywhere** -> `INCONCLUSIVE_LOW_CONFIDENCE`. This is
  a real, valid, honest outcome - not a failure of the script. Report it as
  such and point back to the verdict's existing wet-lab `follow_up`
  (lactose-competition / deglycosylation experiments, or equivalent) rather
  than treating "the model didn't converge" as "the model found something
  interesting."

## What NOT to do

- Don't loop past 2 fold jobs hoping for a more confident answer - if
  `INCONCLUSIVE_LOW_CONFIDENCE` is the result, that IS the sensible
  conclusion, not a signal to try a third configuration.
- Don't narrate a low-confidence residue-level pattern as if it were a
  finding. Always state the actual `chain_pair_iptm` next to any
  residue-level claim.
- Don't run the glycan-aware fold for a `topology_violation` case - it
  can't change a hard geometric impossibility, only a PTM the bare model
  couldn't see.

## Output

Writes `fold_runs/followups/<name>_resolution.json`: the full 1-2-iteration
trail (fold results + structural analyses) plus one final `outcome` from the
five above and a human-readable `explanation`. This is the artifact to check
into the verdict record / cite in `RESULTS.md`, the same way
`fold_runs/followups/_summary.json` entries already are.
