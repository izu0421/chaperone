# Can chaperone tell a real small-molecule ligand from a confident fold?

A capability test of the protein-ligand path, run 2026-11-22.

## The point of the test

AlphaFold3 places a small molecule in a GPCR pocket confidently almost
whatever receptor it is given. So the question is not whether chaperone can
read a confident fold — it is whether a confident fold can be made to carry
the verdict. **Every one of the twelve candidates was handed a high interface
pTM, 0.82 to 0.91.** If the fold drove the answer, all twelve would come back
the same.

The ipTM column in `small_molecule_candidates.csv` is a STAND-IN, not the
output of a real fold. That is deliberate and it is the experiment: it
simulates the screen's actual failure mode, a confident score for everything,
and asks whether the triage survives it. No number in the verdicts below
comes from the fold; they come from live ChEMBL lookups.

## Panel

Eight pairs where the ligand is the receptor's established ligand, and four
deliberate mismatches built from the chemotype-contamination matrix, so the
molecule is one with no measured activity at that receptor:

| receptor | ligand | truth |
|---|---|---|
| HTR1A | serotonin | cognate |
| DRD2 | dopamine | cognate |
| ADORA2A | adenosine | cognate |
| HCAR2 | nicotinic acid | cognate |
| FFAR2 | propionic acid | cognate |
| CNR1 | anandamide | cognate |
| GPBAR1 | lithocholic acid | cognate |
| GPR35 | kynurenic acid | contested in the literature |
| GPR17 | serotonin | mismatch |
| SUCNR1 | dopamine | mismatch |
| LPAR1 | serotonin | mismatch |
| CYSLTR1 | succinic acid | mismatch |

## Result: 12 of 12

```
  cognate    n=8   ALREADY_KNOWN              8
  mismatch   n=4   LIKELY_PROMISCUOUS_POCKET  4
```

Perfect separation, with the fold's confidence held high across both classes.
The mismatches span the full ipTM range of the cognate pairs (0.83-0.86
against 0.82-0.91), so no threshold on that number could have produced this
split.

What produced it instead, per candidate:

- **LPAR1 + serotonin** -> `chembl_known_activity` at Q92633 returns 0
  activities, and `chembl_ligand_promiscuity` returns 638 affinity
  measurements across at least 79 distinct targets. A molecule measured at 79
  targets that has never been measured at this one.
- **CYSLTR1 + succinate** -> 0 activities at Q9Y271; succinate active at 47
  distinct targets across unrelated classes.
- **HTR1A + serotonin** -> 116 activities at P08908, Ki 1.7 nM.

## The contested case, handled better than the label

GPR35 + kynurenic acid came back ALREADY_KNOWN, which the truth column calls
contested. The agent is right and the label is the crude one: ChEMBL does
record the pair. What matters is that it reported the potency with it —
**Ki 137,000 nM and agonist EC50 39,200 nM** — which is exactly the number
that makes the pairing contested in the literature. A reader gets the
activity and the reason to doubt it in the same line, which is the behaviour
to want from a triage tool.

## Reproducing

```bash
python -m chaperone.run_pipeline examples/small_molecule_candidates.csv \
    --no-fold --out verdicts.csv --skip-review --skip-validation
```

`--no-fold` is right here: the test is of the evidence path, and folding would
add GPU time without changing what is being measured. Results depend on live
ChEMBL, so counts drift as the database grows; the separation should not.

## What this does NOT test

The fold itself. Every ipTM here was supplied, so this says nothing about
whether AF3 places these molecules correctly, nor about the counter-screen.
That is what `structural_modelling/BENCHMARK_small_molecule_specificity.md`
is for: 16 receptors by 16 ligands, really folded.
