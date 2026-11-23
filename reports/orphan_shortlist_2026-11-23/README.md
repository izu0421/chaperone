# Orphan GPCR shortlist, triaged 2026-11-23

102 candidate interactions from an AlphaFold3 co-folding screen of 5,000 candidate
partners against 75 orphan G-protein coupled receptors, run through the full
chaperone pipeline.

## Files

| file | what it is |
|---|---|
| `candidates.csv` | the input: 81 protein and 21 metabolite candidates |
| `verdicts.csv` | one row per candidate, with the rationale and the evidence behind each field |
| `report.html` | the traceable report; every claim resolves to the tool call that produced it |
| `validation_strategies.json` | a designed experiment per candidate recommended for validation |
| `review_findings.json` | the review agent's audit of the verdicts and strategies |

## How the candidate set was chosen

The screen's shortlist is not one list. Three gate definitions select different
candidates, and only four pairs appear in all three, so the input is their union
with each row's provenance in `source_shortlist`:

- `figure_gates` — the manuscript's published gates (34 folds)
- `analysis_shortlist` — adds the ligand chain's own pLDDT and a promiscuity filter (74)
- `size_adjusted` — the top 30 protein folds by interface pTM standardised **within
  size deciles**, because raw interface pTM rises with complex size (Spearman rho =
  0.362 over 4,767 folds) and ranking on it ranks partly on size

Metabolite candidates were collapsed to distinct receptor–molecule pairs. The same
molecule reaches a receptor through several producer genes — glutamate arrives as
SLC17A7, SLC1A1, SLC1A2 and SLC1A6 — and the fold is identical in each case, so
judging it once per producer name would multiply one structure into several
verdicts that cannot disagree. 28 rows became 21 pairs over 8 molecules.

## The result, in one line

**Every metabolite candidate that received a structural verdict was called
`LIKELY_PROMISCUOUS_POCKET`, and none was confirmed.** AlphaFold3 places a small
molecule confidently in a class A GPCR pocket whichever receptor it is given, so a
confident metabolite fold reports the pocket and not the ligand. The pipeline
reached that from ChEMBL target counts and UniProt function text — independently of
the screen's own observation that interface pTM is saturated on that arm, clearing
0.5 for 97% of metabolite folds.

No candidate is high-confidence.

## Verdicts

| verdict | n |
|---|---|
| `INSUFFICIENT_EVIDENCE` | 25 |
| `LIKELY_ARTIFACT_PTM` | 24 |
| `LIKELY_PROMISCUOUS_POCKET` | 19 |
| `CONFIRMED_NOVEL` | 14 |
| `LIKELY_SUBCOMPLEX` | 10 |
| `IMPLAUSIBLE` | 10 |

`CONFIRMED_NOVEL` started at 22 and ended at 14. The eight that moved are the whole
argument for keeping the judging step as code rather than an agent:

- Three were downgraded once the deterministic gate could actually run (it had been
  silently disabled by a rejected `tool_choice` parameter): ROR2/MRGPRF and
  SEMA4C/GPR37 to `IMPLAUSIBLE`, FPR2/GPR183 to `INSUFFICIENT_EVIDENCE`. ROR2/MRGPRF
  had a well-argued rationale; the gate called it implausible anyway.
- Four were PPIA (cyclophilin A), downgraded to `INSUFFICIENT_EVIDENCE` by the new
  partner-breadth rule: PPIA reached the confidence gate against 4 of the 11 receptors
  it was folded against, so the fold is evidence about PPIA and not about GPR3, GPR63,
  GPR75 or GPR82 individually.
- One, CHL1/GPR12, went to `INSUFFICIENT_EVIDENCE` on reconsideration.

## Looking at the poses

`structures/index.html` is an interactive viewer for the folds the pose argument rests on
(3Dmol.js; open it from a served copy, not `file://`). It shows eight structures, chosen to
be falsifiable rather than flattering: two ligands in the receptor they are known to bind,
two molecules known *not* to bind the receptor they are shown with, and the candidates.
The receptor is coloured by UniProt topological domain, so "inside the bundle" and "on the
cytoplasmic face" are visible without reading a number, and the residues within 4.5 A of
the molecule carry a surface.

| fold | role | enclosed | contacts | on TM | cytoplasmic |
|---|---|---|---|---|---|
| CNR1 + anandamide | cognate | 0.993 | 26 | 16 | 0 |
| ADRB2 + adrenaline | cognate | 0.964 | 14 | 13 | 0 |
| P2RY12 + acetylcholine | non-binder | 0.063 | 0 | 0 | 0 |
| DRD2 + ADP | non-binder | 0.311 | 3 | 1 | 2 |
| GPR153 + noradrenaline | candidate | 1.000 | 13 | 13 | 0 |
| GPR85 + noradrenaline | candidate | 0.973 | 14 | 9 | 0 |
| GPR26 + glutamate | candidate | 0.973 | 13 | 11 | 0 |
| GPR150 + GABA | rejected | 0.418 | 4 | 4 | 0 |

Interface pTM does not separate these: all eight score between 0.65 and 0.93, the two known
non-binders included. The pose does. P2RY12 + acetylcholine has **no contacts at all** --
the molecule is outside the receptor -- and DRD2 + ADP lines the intracellular face. Of the
candidates, GPR153 + noradrenaline has the same profile as adrenaline in ADRB2: every
contact on a transmembrane span, fully enclosed. GPR150 + GABA is included because it
fails, at 0.418 against a minimum of 0.928 across all sixteen known cognate ligands.

## Caveats a reader should carry

- Confidence is `low` for the large majority. These are triage calls, not results.
- A protein partner confident against many receptors is flagged by the gate's
  `partner_breadth` rule, but that rule needs `partner_receptors_tested` and
  `partner_receptors_confident` on the input row. Where those are absent the check
  reports `checked: false`, not a reassuring zero.
- The screen behind these candidates covers 4,913 of 5,000 folds. The 87 not folded
  are not a random sample: 85 are the whole subset whose partner had no omicMSA.
