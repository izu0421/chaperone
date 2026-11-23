# Co-folding as a screening score: what the literature says

Written 2026-11-21, after this project's orphan screen found that AlphaFold3
interface confidence carries no specificity for a small molecule in a GPCR
pocket: 12 of 21 metabolite candidates reached interface pTM 0.5 against
**every** receptor tested, where 0 of 419 protein partners did.

The question: **has anyone compared AlphaFold3 against AutoDock Vina and
other docking, and has anyone tested whether co-folding confidence separates
binders from non-binders?** Both, repeatedly. The answer is consistent across
groups and is not the one the AF3 paper's own benchmark implies:

> **Co-folding beats docking at placing a ligand, and loses to docking at
> deciding whether the ligand belongs there.**

Our result is that second half, measured a different way. It is corroborated,
not novel — which is good for the claim and bad for treating it as a finding.

**Verification key.** `[v]` read verbatim from the source. `[s]` came through
a model-extracted summary and has **not** been checked against the text.
`[search]` from a search-result summary only, primary source not read. Two
numbers are flagged ⚠ as needing a check before anyone cites them.

---

# Part 1 — Screening and discrimination

## 1.1 The decisive study: Mac1, 489 ligands, three prospective screens

**Kim J, Correy GJ, Hall BW, Rachman MM, Mailhot O, … Renslo AR, Ashworth A,
Shoichet BK, Fraser JS. "Large-scale prospective evaluation of co-folding
across hundreds of Mac1–ligand complexes and three virtual screens."
eLife 2026;15:RP110475.** doi:10.7554/eLife.110475 · PMCID PMC13657626

UCSF, Shoichet and Fraser labs. AlphaFold3, Chai-1 and Boltz-2 against
DOCK3.7, on SARS-CoV-2 Mac1 with hundreds of crystallographic complexes, plus
three prospective screens — σ2 receptor, **D4 dopamine receptor** (a GPCR) and
AmpC β-lactamase. The single most relevant paper to this project.

**Pose — co-folding wins.** `[v]` "AF3 and Chai-1 predicted roughly 70% of the
ligands correctly" (< 2 Å); Boltz-2 53%; DOCK3.7 40%. Ligand in the right
site: AF3 92%, Boltz-2 91%, Chai-1 90%, DOCK3.7 80%.

**Screening — co-folding loses.** `[v, abstract]` "AF3 ligand pose confidence
values did not separate true ligands from high-scoring false positives as
effectively as docking scores or Boltz-2 affinity predictions did."

AUC, actives vs docking false positives `[s]` ⚠ confirm against Figure 3:

| target | DOCK3.7 | Boltz-2 affinity | AF3 ligand-pLDDT |
|---|---|---|---|
| σ2 receptor | 78.8 | 83.8 | **56.1** |
| D4 dopamine receptor (GPCR) | ~71 | ~71 | **46.4** |
| AmpC β-lactamase | 76.5 | 68.1 | **60.5** |

AF3 confidence is **at chance on the GPCR**. Non-binder counts were 305 (σ2),
336 (D4), 1046 (AmpC) `[s]`.

**The structural reason, in their own data.** AF3 placed ligands correctly
while getting the receptor state wrong: `[v]` "only 4/19 twisted states and
0/20 open states were correctly predicted by AF3", with "eight false positive
twisted states". `[v, Fig 2]` co-folding "reliably recapitulates dominant
ligand-binding interactions even in the absence of accurate protein
conformational modeling." A method that poses a ligand correctly without
modelling the receptor state is reporting the site, not the pairing.

`[v, reviewer #3]` "co-folding models are powerful tools for pose prediction
but are not yet standalone solutions for virtual screening."

**Their own caveat** `[v]`: "comparing co-folding to docking based on
hit-lists themselves selected by docking is arguably unfair to co-folding."

## 1.2 The closest analogue to our result: orphan GPCRs

**Ferguson L, Ouellet S, Vandewyer E, Wang C, Wunna Z, Lim TKY, Schafer WR,
Beets I. "DeorphaNN: Virtual screening of GPCR peptide agonists using
AlphaFold-predicted active-state complexes and deep learning embeddings."
Molecular Cell 2026.**
https://www.sciencedirect.com/science/article/pii/S1097276526004703
· preprint doi:10.1101/2025.03.19.644234

Peptide agonists, not metabolites, but the same task: which ligand belongs to
which orphan GPCR. Their results section is headed, `[v]`:

> **"AF-Multimer positions peptides in GPCR binding pocket with minimal
> discrimination."**

On 20,035 *C. elegans* GPCR-peptide pairs, 457 confirmed agonists
`[v, Fig 1E]`:

> **99.8% of agonists and 96.6% of NON-agonists were modelled in the pocket.**

A 3.2-point gap. Our effect, in another ligand class, at a scale we have not
reached. `[v]` "Due to the prevalence of peptides positioned in the binding
pocket irrespective of ligand class, this feature alone does not provide a
reliable basis for distinguishing…"

Confidence gives `[v]` "modest discriminative capability", and `[v]` "Peptide
pLDDT significantly outperformed ipTM and pocket PAE" — **ipTM, the metric our
screen used, was the worst of the three.**

Their fix was not a threshold but a GNN over AF embeddings and Arpeggio
contacts: `[v]` "a mAP of 0.72—significantly higher than the random baseline
of 0.30", validated prospectively (NPR-33 with NLP-70-2, EC50 76.3 nM, ranked
1st of 364). AlphaFold3 was **not** tested; the model is AF-Multimer with
AF-Multistate templates.

Method worth borrowing: they dropped ROC-AUC because it `[v]`
"disproportionately prioritizes promiscuous GPCRs", using per-receptor
average precision instead.

## 1.3 The strongest positive result — and its own counter-screen fails

**Shamir Y, Gabizon R, Rogel A, Lin DY, Andreotti AH, London N. "Discovery of
Covalent Ligands with AlphaFold3." J Am Chem Soc 2026;148(12):13043–13054.**
doi:10.1021/jacs.5c22222 · PMCID PMC13047693

The paper usually cited for AF3 screening working, and it does: adjusted
LogAUC 71.8 ± 5.9 for AF3 mPAE vs 9.9 ± 4.8 DOCKovalent, 12.1 ± 9.9 DOCK6,
5.7 ± 7.9 AutoDock, over 874 actives and 37,919 property-matched decoys.
Prospective BTK screen of ~906K compounds → 3 of 13 synthesised compounds
with near-complete covalent labelling; best IC50 30 nM.

⚠ The widely quoted **"AUC 98.3%" was not found in the text**; reported AUCs
are 0.9082–0.9996 across ten sites. Do not cite 98.3% without the SI.

Three reasons it does not transfer to a metabolite in a GPCR pocket:

1. **The covalent bond is supplied.** `[v]` "The input also includes
   specification of the ligand atom and the protein side-chain atom to be
   covalently bonded." The search space is constrained in a way ours is not.
2. **The metric is not ipTM.** They use mPAE, and note protein-only metrics
   are weaker `[v]` "in line with the fact that the protein is constant for
   any specific target."
3. **They ran the counter-screen and it failed.** The most relevant single
   result in this literature for us. `[v]` "we conducted cross-docking
   experiments and calculated the enrichment for each kinase active and decoy
   set against noncognate kinases… clear clusters appear, in which kinases
   with a similarly positioned cysteine residue perform well in enriching
   noncognate ligand sets." Their conclusion `[v]`:

   > **"Overall it likely suggests that AF3 and mPAE are currently not
   > suitable for selectivity prediction."**

   Enrichment tracks the **geometry of the site**, not the identity of the
   target — from the group whose headline is that AF3 screening works.

Also `[v]`: "no apparent correlation between mPAE and affinity", so mPAE is
"a 'coarse' metric useful primarily for classification."

## 1.4 Other screening benchmarks

- **VsNsBench** — Acta Pharmacol Sin 2026, doi:10.1038/s41401-025-01732-2,
  PMID 41639320. AF3 holo beats AF3 apo, experimental apo and AF2; but
  against experimental holo, AF3 was **worse on DUD-E**, slightly better on
  VsNsBench `[search]`.
- **LIT-PCBA / Protenix / Boltz-2** — RSC paper comparing against Glide and
  Gnina on a LIT-PCBA subset, chosen because it uses experimentally verified
  inactives instead of property-matched decoys `[search]`.
- ⚠ **Benchmark caveat:** "Data Leakage and Redundancy in the LIT-PCBA
  Benchmark", arXiv 2507.21404. DUD-E has known actives/decoys similarity
  bias; LIT-PCBA is harder but leaky. Any enrichment number is only as good
  as its decoy construction — the lesson this project already learned as
  `feedback_negative_control_matching`.
- **CASP16** — co-folding baseline reached mean LDDT-PLI 0.80, beating all
  participating groups, while AutoDock Vina was near RoseTTAFold-AllAtom at
  the bottom. Affinity, though: `[search]` "Ligand affinity prediction is
  totally unreliable."

---

# Part 2 — Pose accuracy, and why it is a separate question

## 2.1 PoseBusters

**Buttenschoen M, Morris GM, Deane CM. "PoseBusters: AI-based docking methods
fail to generate physically valid poses or generalise to novel sequences."
Chem Sci 2024.** arXiv:2308.05777 · PMCID PMC10901501

Predates AF3. Five deep-learning docking methods vs AutoDock Vina and CCDC
GOLD on 308 complexes released from 2021 (outside PDBbind training). No DL
method beat classical docking on physical validity or generalisation; best DL
method ≈12% success `[search]`. Introduced PB-validity — a physical-plausibility
check that RMSD ≤ 2 Å alone does not catch.

AF3's own PoseBusters numbers, **not like-for-like**: AF3 73.6% vs Vina 52.3%,
but Vina was given the protein structure and the binding site while AF3 got
sequence and SMILES `[search]`.

## 2.2 PoseX

arXiv:2505.01700. Physics-based methods (AutoDock Vina, MOE) plateau at
56.4–67.1% even with induced fit; AF3, Protenix and Chai-1 exceed 80%
`[search]`.

## 2.3 Runs N' Poses — the memorisation control

**Škrinjar P, et al. "Have protein-ligand cofolding methods moved beyond
memorisation?" / "Evaluating generalization in protein–ligand cofolding
methods." Nat Struct Mol Biol 2026.** doi:10.1038/s41594-026-01797-5 ·
preprint doi:10.1101/2025.02.03.636309

2,600 high-resolution complexes binned by similarity to training data. Finding
`[search]`: co-folding largely reproduces training poses, limiting de novo
use. AF3 reaches 61.5% overall but ~26% on the lowest-similarity subset
(90 ligands, 83 systems) `[search]`.

Same pattern this project already measured for TCR folds across the AF3
training cutoff (`finding_esmfold2_vs_af3_memorisation`): the advantage is
concentrated where the model has seen the answer.

---

# Part 3 — Why: the models do not model the interaction

## 3.1 The pocket-mutation experiment

**Masters M, Mahmoud A, Lill MA. "Investigating whether deep learning models
for co-folding learn the physics of protein-ligand interactions."
Nat Commun 2025.** doi:10.1038/s41467-025-63947-5 (paywalled; preprint
doi:10.1101/2024.06.03.597219)

Mutate the residues that contact the ligand — glycine substitution (strip side
chains), phenylalanine packing (fill the pocket), inversion (replace each
residue with the most dissimilar by Miyata distance) — and see whether the
ligand moves. Benchmark: CASF-2016, 285 complexes, plus CDK2 and MEK1 case
studies. Models: AlphaFold3, RoseTTAFold All-Atom, Chai-1, Boltz-1.

`[search]` The models kept placing the ligand in the original pocket; after
inversion all four kept ATP in the original site with little pose change. The
authors conclude AF3 is not predicting poses from interaction physics but
learning patterns in global protein structure and sequence.

⚠ Fetch blocked by Nature's login gate and by a bioRxiv rate limit; the
numbers above are `[search]` only. Read before citing.

## 3.2 The balanced counterpoint, with numbers

**Schneider S. "Do Co-folding Models Notice When You Change the Pocket?"
Rowan Scientific blog, 2 Nov 2026.**
https://www.rowansci.com/blog/cofolding-pocket-mutations

Partial replication on six models (Boltz-2, Boltz-2.1, OpenFold-3,
DeCAF-Boltz, ESMFold-2, ESMFold-2-fast). Their reading is more forgiving and
the numbers are useful. CASF-2016, ESMFold-2-fast, fraction native-like `[v]`:

| condition | native-like |
|---|---|
| wild type | **77%** |
| glycine | 33% |
| phenylalanine | 29% |
| inversion | 21% |

Confident predictions (mean ligand pLDDT > 80): 209 of 285 wild type, against
69 / 44 / 44 for the mutants `[v]`. Of confident mutant predictions, `[v]`
"Between 64% and 75%" are still native-like.

Their conclusion `[v]`: "in aggregate it does respond to the pocket: most of
the time, rewriting the binding site either moves the ligand or lowers the
model's confidence." But it is system-dependent — `[v]` "the same models that
react strongly to CDK2 mutations barely notice MEK1's" — and `[v]` "Some of
that accuracy likely comes from recognizing a familiar protein and ligand."

**The sentence that matters most for us** `[v]`:

> "That's still a statement about the pose, not about binding; pLDDT isn't a
> binding probability."

with `[v]` "None of these outputs say whether the ligand actually binds."

Their operational advice, which is what `chaperone` now implements `[v]`:

> **"Use confidence as a filter, not a verdict."**

---

# Part 4 — The affinity alternative

**Boltz-2 (MIT / Recursion, 2025).** doi:10.1101/2025.06.14.659707 ·
PMCID PMC12262699. Adds a trained affinity head. Developers report Pearson
R 0.66 on the FEP+ subset (CDK2, TYK2, JNK1, P38) and ~1000× faster than FEP
`[search]`; training excluded proteins ≥90% identical to benchmark proteins.

Independent evaluations are weaker `[search]`: a 93-molecular-glue comparison
found "generally poor or even negative correlations"; a Uni-FEP evaluation
found it lagged FEP where buried water mattered; fine-tuning recovers a lot
(R 0.01 → 0.40 on one target).

Relevant to us because in the Mac1 screens **Boltz-2 affinity reached AUC 83.8
on σ2 where AF3 ligand-pLDDT reached 56.1**. If the metabolite arm is pursued,
this is the stronger instrument and a drop-in for the same inputs.

---

# Part 5 — What this means for this project

1. **The orphan-screen result stands, corroborated, not novel.** Four
   independent groups report the same split. Write it up with citations as a
   confirmation. Say that ipTM — our metric — is the one DeorphaNN found
   weakest of three, and that AF3 confidence was at chance on the one GPCR in
   the Mac1 screens.

2. **The counter-screen is now the published standard, not an optional
   extra.** Shamir et al. ran it as cross-docking against non-cognate kinases
   and reported a negative result for selectivity. Our planned counter-screen
   against receptors with known ligands is the same design. A reviewer who
   knows this literature will ask for it.

3. **Stop using ipTM for the ligand side.** Mac1 → ligand-pLDDT; DeorphaNN →
   peptide pLDDT over ipTM and pocket PAE; Shamir → mPAE, explicitly rejecting
   protein-only metrics. Three independent votes against ipTM. Our scoring
   already records per-chain confidences, so this is re-scoring existing
   folds, not re-folding them.

4. **If the metabolite arm continues, add Boltz-2 affinity.** Same inputs,
   and it is the one score that worked where AF3 confidence did not.

5. **Already wired into `chaperone`.** The triage agent's small-molecule
   section states the non-discrimination result and requires ChEMBL
   promiscuity before a confident fold can support a novel call; the
   deterministic gate flags CONFIRMED_NOVEL on a promiscuous molecule. See
   `chaperone/src/chaperone/agent.py` and `deterministic_gate.py`.

---

# Not yet read

- **"AlphaFold3 for Structure-guided Ligand Discovery"**, bioRxiv
  10.64898/2025.12.04.692352 — benchmarks D4, σ2 and AmpC. Reported
  second-hand to find AF3 confidence uncorrelated with affinity, ipTM useful
  as a false-positive filter, and **no neutral ligand in the top 1% or 5% of
  ipTM for σ2** — a charge bias. Four fetch attempts hit bioRxiv rate limits.
  Worth reading: a chemotype bias in ipTM would be a second, independent
  reason our metabolite scores are not about the receptor.
- Masters/Mahmoud/Lill primary text (§3.1) — gated.
- Mac1 supplementary figures S6, S7, S11 holding the per-method AUCs.
- Pat Walters, "Three Papers Demonstrating That Cofolding Still Has a Ways to
  Go", Practical Cheminformatics, 21 July 2025 — page body is one large inline
  image, not machine-readable.
