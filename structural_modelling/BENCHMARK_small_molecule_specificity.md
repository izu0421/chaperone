# Does co-folding confidence know WHICH receptor a small molecule binds?

A benchmark design, written 2026-11-22. The orphan screen found that AF3
interface confidence does not discriminate for a metabolite in a GPCR pocket,
and `LITERATURE_cofolding_screening.md` shows four groups reporting the same
thing. Neither is our own controlled test. This is that test.

## The question, stated so it can fail

For a panel of receptors with known small-molecule ligands, fold every ligand
against every receptor. If co-folding confidence carries specificity, the
cognate pairs on the diagonal outscore the non-cognate cells off it. If it
reports the pocket, the matrix is flat.

One number settles it: **AUROC of each confidence metric for separating
cognate pairs from confirmed non-pairs.** A flat matrix is AUROC ≈ 0.5.

This is the design Shamir et al. (JACS 2026) ran as cross-docking against
non-cognate kinases, where enrichment clustered by cysteine position instead
of by target identity. Ours differs in using a receptor class where the
pocket is shared by construction, which is the harder and more relevant case
for deorphanisation.

## The trap, and the measurement that found it

The obvious design calls every off-diagonal cell a negative. That is wrong,
and measurably so. Querying ChEMBL for every cell of a 16 x 16 panel:

```
  diagonal (cognate pairs)                16    all have measured activity
  off-diagonal cells                     240
    with measured ChEMBL activity         51    21%  NOT negatives
    of those, same ligand family          14
    clean (zero activities)              189    usable negatives
```

**A fifth of the "non-cognate" cells are measured interactions.** Serotonin is
recorded at 79 distinct targets, dopamine at 101, adrenaline at 148.
Treating the off-diagonal as negative would put 51 real interactions in the
negative class and depress any true signal — the same failure as
`feedback_negative_control_matching`, where an abundance-confounded control
produced a clean, replicating, wrong mechanism.

The contamination is structured, not scattered: it sits in the aminergic
block and under the promiscuous ligands, and the lipid, eicosanoid and
carboxylic-acid receptors are clean. So the negatives must be filtered by
ChEMBL, and the surviving ones stratified by whether ligand and receptor
share a chemotype family — within-family is the hard case and the one
deorphanisation actually faces.

## Panel

Sixteen class A GPCRs spanning seven ligand chemotypes, each with an
endogenous or established small-molecule ligand confirmed by ChEMBL affinity
at that receptor:

| family | receptors |
|---|---|
| aminergic | HTR1A, DRD2, ADRB2, HRH1, CHRM1 |
| purinergic | ADORA2A, P2RY12 |
| lipid | CNR1, S1PR1, LPAR1 |
| eicosanoid | PTGER2, CYSLTR1 |
| carboxylic acid | FFAR2, HCAR2, GPBAR1, SUCNR1 |

Ligands: serotonin, dopamine, adrenaline, histamine, acetylcholine,
adenosine, ADP, anandamide, sphingosine-1-phosphate, lysophosphatidic acid,
prostaglandin E2, leukotriene D4, propionate, nicotinic acid, lithocholic
acid, succinate. All resolve to a structure, all have measured affinity at
their own receptor.

## Three arms

1. **Cognate** — the diagonal. 16 pairs.
2. **Non-cognate, ChEMBL-clean** — 189 cells with zero recorded activity,
   stratified within-family and across-family.
3. **Decoy molecules** — drug-like compounds with no activity at any panel
   receptor, folded against all 16. This is the empty baseline:
   `feedback_specificity_needs_empty_baseline` is the case where a margin
   looked design-driven until a no-binder control showed it was not. Without
   it, a flat diagonal-vs-off-diagonal result cannot be told from a metric
   that simply saturates on every GPCR pocket.

## Read-out

AUROC for cognate vs clean non-cognate, for each metric separately:

- `iptm` and `iptm_AB` — what the orphan screen used
- `plddt_B`, `iface_plddt_B` — the ligand chain's own confidence, which three
  independent groups found stronger than ipTM
- `iface_plddt`, `bsa`, `n_contact`, `min_sep`, `ranking_score`

Reported with a bootstrap interval over pairs. Secondary read-out: per-ligand
rank of its own receptor among the 16 — the deorphanisation question stated
directly, and the one DeorphaNN answers with per-receptor average precision
because ROC over a whole matrix flatters promiscuous receptors.

## Power, honestly

Sixteen positives is thin: an AUROC interval on 16 against 189 is roughly
+/- 0.12, which can separate 0.5 from 0.8 but not 0.5 from 0.6. Two fixes,
in order of cost:

- Add a second established ligand per receptor, giving ~32 positives. Cheap,
  since the receptor MSAs are reused.
- Report the per-ligand rank statistic alongside AUROC; with 16 ligands each
  ranked against 16 receptors it is better powered than the pooled AUROC and
  is the quantity a deorphanisation user cares about.

A null result here is only evidence with its detection floor stated
(`feedback_report_the_detection_floor`), so the minimum detectable AUROC gets
computed before the folds run, not after.

## Seeds

AF3 docking splits by seed (`feedback_one_seed_is_a_screen`): a single-seed
zero clears nothing. The matrix runs at one seed, because the AUROC averages
over hundreds of pairs, but any per-pair claim and the full diagonal get
re-run at five seeds.

## Cost

Receptor MSAs are generated once and reused across every ligand, so the
matrix is 16 MSA jobs and 256 inference jobs on small complexes (a GPCR of
~350-450 residues plus one small molecule). Estimated 12-20 GPU-hours for the
16 x 16, roughly double with a second ligand per receptor. Small beside the
HuRI benchmark's ~350.

## Status

Panel resolved and the ChEMBL contamination matrix measured
(`bench/chembl_matrix.json`). Not yet folded. Queued behind the orphan screen
and to be prioritised against HuRI.
