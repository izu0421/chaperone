# structural_modelling — high-throughput structure-based PPI modelling

Sits between the atlas/deorphanisation side (which proposes pairs) and `chaperone/`
(which adjudicates a verdict per pair). HSA and `lr_atlas/` generate candidate partners from
spatial co-expression; `chaperone/` reasons over evidence for one pair at a time and needs a
structural term it can trust. This module is the structural term, and the question it has to
answer first is not "can we fold 100,000 pairs" but **"does a fold score discriminate interacting
from non-interacting pairs at the prevalence a real screen has?"**

```
HSA / lr_atlas                  structural_modelling                 chaperone
┌──────────────────┐            ┌──────────────────────┐            ┌─────────────────┐
│ spatial          │  pairs     │ fold the pair        │  structural │ deterministic   │
│ co-expression    │ ─────────► │ score the interface  │ ──────────► │ gate + verdict  │
│ orphan receptors │            │ calibrate vs truth   │   evidence  │                 │
└──────────────────┘            └──────────────────────┘            └─────────────────┘
```

## The result that sets the agenda

Zhang et al., *Science* 2025 (PMID 40997207, `eadt1630`) introduce **RoseTTAFold2-PPI** and screen
~191M human pairs in a three-stage cascade: DCA pre-filter (111M → 43.6M) → RF2-PPI
(`RFIntProb` > 0.3 → 1.9M) → AF2 (`AFIntProb`). They report 17,849 PPIs at an estimated 90%
precision, 3,631 of them absent from existing sources.

Four of their findings constrain anything built here, and the first two make the obvious framing
of this module wrong:

1. **AF2 + omicMSA was the best method on their benchmark** — not RF2-PPI. RF2-PPI with omicMSA
   only slightly beat the ColabFold pipeline.
2. **AFmm and AF3 were WORSE than RF2-PPI at 1:1000 prevalence**, and better at 1:10. So "use
   AF3, it is the newest" is contradicted by the only published measurement, and the direction
   of the effect depends on prevalence. **But see below: AF3 was run on a different MSA from AF2, so that comparison does not isolate the engine.**
3. **The gain came from the MSA, not the engine.** Their `omicMSAs`, assembled from unassembled
   genomic and transcriptomic data, are ~7× deeper than UniRef-based alignments.
4. **Evaluation is AUCPR at 1:1000**, with positives down-weighted ×0.01. They do not report
   AUROC, for the reason an AUROC at 1:1000 is close to uninformative.

### The cell nobody filled: AF3 + omicMSA

Checked against the text, the paper's method comparison does not hold the alignment fixed:

- AF2 was run **with omicMSAs** — "The best performance for a de novo PPI screen is achieved when
  AF2 is used in combination with omicMSAs." AF3 is not named in that sentence.
- AF3 was run through the **"AF3 web server widely used by the scientific community"**.
  *Checked 2026-11-19: the paper never states which MSA AF3 used in fig. 2D.* That it was AF3's
  own pipeline is an inference — a strong one, because the AF3 server accepts no custom
  alignment — but it is an inference, not a quoted fact.
- The paper states plainly that "all the tools we tested were trained using MSAs built from
  UniRef and metagenomic sequences, and they are not adapted to work with omicMSA."

So "AF3 is worse than RF2-PPI at 1:1000" is a measurement of **AF3-with-its-own-MSA** against
RF2-PPI-with-omicMSA. Engine and alignment moved together. Given the paper's own conclusion that
the alignment carries the gain, AF3's poor showing may be an MSA artefact rather than a property
of AF3 — and **AF3 + omicMSA is simply not in the literature.**

That is the experiment this module exists to run, and AlphaFast is what makes it runnable:
`src/alphafold3/common/folding_input.py` exposes `paired_msa` and `unpaired_msa` as per-chain
A3M strings, so an omicMSA can be injected into AF3 directly and `run_data_pipeline.py` bypassed.
AF2 needed a Colab notebook for the same trick; AF3 needs a JSON field.

**The honest prior is not that AF3 + omicMSA wins.** The paper's own caveat cuts the other way:
AF3 was trained on UniRef-and-metagenome alignments, so a 7x deeper omicMSA is out of
distribution for it. This project has also already seen a deeper-is-worse result —
`finding_stcrdab_benchmark_contaminated` found depth-saturation flipping to rho +0.34, deeper
AIRR being WORSE, once contamination was controlled. Three outcomes are all informative:

| outcome | what it means |
|---|---|
| AF3 + omicMSA > AF2 + omicMSA | AF3's published deficit was an MSA artefact and the field is quoting the wrong comparison |
| AF3 + omicMSA ≈ AF3 native | AF3 cannot exploit depth it was not trained on; the alignment advice is engine-specific |
| AF3 + omicMSA < AF3 native | depth is actively harmful out of distribution, as on the TCR benchmark |

Point 3 is the same shape as this project's own result on TCR–pMHC
(`finding_native_dbs_carry_discrimination`): **databases contributed +0.138 AUC and the engine
+0.005**, and a fast, shallow MSA *saturated* interface PAE and destroyed discrimination
(AUC 0.52 vs 0.66). Two independent systems now say the alignment carries the signal.

## The question

AlphaFast replaces Jackhmmer with MMseqs2-GPU for a claimed 68× homology-search speedup and 22×
end-to-end. The speedup is real and the folding engine is AF3. But if discrimination lives in the
alignment, then:

> **Does AlphaFast's fast MSA preserve the discrimination that omicMSA depth buys, or does the
> 68× come out of exactly the signal that decides whether two proteins interact?**

This is falsifiable, cheap to answer relative to a full screen, and it decides whether
high-throughput structure-based screening is viable here at all. If the fast MSA saturates,
the module's deliverable is that finding plus a depth requirement — not a 100,000-pair screen
whose scores mean nothing.

## Design: separate the engine from the alignment

The paper varies engine and MSA together, so their comparison cannot attribute the difference.
A factorial does:

|                     | AlphaFast MMseqs2-GPU MSA    | omicMSA (theirs, precomputed) |
|---------------------|------------------------------|-------------------------------|
| **AF3** (alphafast) | A — the fast pipeline        | **B — the unfilled cell**     |
| **RF2-PPI**         | C — MSA held, engine swapped | D — their published setting   |

- **B** is the number that does not exist yet: AF3 on omicMSAs. Everything else is context for it.
- **B − A** isolates the alignment at fixed engine, and answers whether AlphaFast's 68x speedup
  costs discrimination.
- **C − A** isolates the engine at fixed alignment.
- **B − D** is the engine comparison the paper reports but did not actually make, because its AF3
  arm ran on the web server's own MSA.
- **D** reproduces the paper and is the check that the harness is right before any claim about
  A–C. If D does not land near the published AUCPR, nothing else here is interpretable.
- **A′ = AF3 on its own native MSA pipeline** is a fifth arm reproducing the paper's AF3
  condition, so that B − A′ is measured on this benchmark rather than inferred across papers.

RF2-PPI consumes *paired* MSAs (`predict_list_PPI.py`, a list of `msa_path, len_chain_A`), so arm
C needs AlphaFast's MSAs re-paired by organism in their format. That conversion is the main piece
of engineering and is where a silent mistake would look like a biological result, so it gets its
own round-trip test: convert an omicMSA to our format and back, and require the recovered
`RFIntProb` to match.

## Scoring: what comes off a fold

Per pair, from the AF3 output: `iptm`, `ptm`, `chain_pair_iptm`, interface pLDDT, and physical
interface terms (buried surface area, contact count, interface hydrophobicity). Two reasons not
to rank on ipTM alone:

- `feedback_af2_confidence_weak_egfr`: ipTM AUC was 0.636 on real binding data and a 5.18 nM
  binder scored 0.21. Gate on physical interface terms as well.
- `finding_protein_dna` (n100_rho0.6 screen): **BSA ≫ ipTM as a discriminator.**

So the primary structural score is fitted on the benchmark, not assumed, and ipTM is reported as
one feature among several with its standalone AUCPR given for comparison.

## Ground truth, and where PrePPI belongs

**PrePPI is a predicted interactome, not ground truth.** ~1.3M human pairs at `final` ≥ 600, from
a Bayesian combination of structural (`str`, `protpep`, `str_max`) *and* non-structural
(`exp`, `coexp`, `go`, `ort`, `phy`, `red`) evidence. Scoring a structure predictor against
PrePPI labels measures agreement with another predictor, so it is not used as truth here. It has
two honest roles instead:

1. **The candidate universe.** PrePPI's structure-informed shortlist is the right target set for a
   high-throughput structural screen — which is the module's purpose — and is far smaller than
   191M pairs.
2. **An orthogonal structural baseline to beat.** PrePPI's `str` score *is* a structure-based PPI
   predictor, built on structural alignment and template transfer. "Does folding the pair beat
   structural homology?" is a fair and interesting question, and one nobody in the RF2-PPI paper
   asked — PrePPI appears there only in the reference list.

   **Only `str` may be used for that comparison, never `final`.** `final` contains `exp`
   (experimental evidence, median 958), so against experimental labels PrePPI-`final` carries
   label leakage and would look artificially strong. This is the single easiest way to get a
   wrong answer out of this module.

**Evaluation labels come from experiment**, reconstructed on the paper's recipe so the numbers are
comparable to theirs:

- **Positives**: UniProt ∩ BioGRID (multi-validated) ∩ STRING (physical, score > 700).
- **Negatives**: pairs with no interaction evidence, at 1:1000 — but **degree-matched**, not
  purely random. `feedback_negative_control_matching` cost this project a clean, replicating,
  wrong mechanism once because an impossible-pair control was not abundance-matched. A hub
  protein appears in more positives, so random negatives make every method look better than it
  is. Random negatives are kept as a second arm so the difference from the paper is visible
  rather than silently folded in.

## What would make this module worthless, and the guards

| failure | guard |
|---|---|
| Harness wrong, arms incomparable | arm D must reproduce the published AUCPR before A–C are read |
| MSA conversion silently lossy | omicMSA → our format → RF2-PPI round-trip must recover `RFIntProb` |
| Ranking on a saturated score | report the distinct-value fraction and tie mass per score (`feedback_standalone_auc_tie_mass`: a 0.807 AUC was one saturated threshold, 0.700 once ties were stripped) |
| Negatives easier than reality | degree-matched *and* random negative arms, both reported |
| Prevalence flattery | AUCPR at 1:1000 is primary; 1:10 reported alongside, since the paper's ranking flips between them |
| Cross-machine comparison | `feedback_cross_machine_baseline`: identical weights/seed moved iPTM sd by 3.4× the signal between machines. Every arm runs on the same host, and the host is recorded per fold |
| Silent GPU failure | assert on throughput and valid-row count, never on exit code (`feedback_silent_oom_nan_run`: a total GPU failure returned exit 0 with 480/480 NaN) |
| Benchmark contamination | AF3's training cutoff is 2021-09-30 and PDB complexes before it are memorised (`finding_stcrdab_benchmark_contaminated`: 77% of a TCR benchmark predated it, and the held-out half behaved differently). Positives are split by first-publication date and reported either side |

## Staging

The GPU is shared (3× H100 NVL, currently held by this account's own bindcraft and esm_ft runs),
so the plan is sized to answer the question before it spends a screen's worth of compute.

- **S0 — data.** PrePPI, the experimental label sources, the pair universe. No GPU.
- **S1 — harness + arm D.** Reproduce the paper on a few hundred pairs. If this fails, stop.
- **S2 — the 2×2** on a powered pilot, sized from S1's observed variance rather than guessed.
- **S3 — the screen**, only if S2 shows the fast MSA retains discrimination. Otherwise the
  deliverable is S2's finding and the depth requirement it implies.

## Status

S0 in progress: PrePPI downloading. Nothing else started.

## S0 findings: PrePPI as it actually is

Downloaded 2026-11-03 from the Omnipath mirror (`rescued.omnipathdb.org`); the Columbia URL
`honiglab.c2b2.columbia.edu/PrePPI/ref/` returned nothing. **The file's internal timestamp is
2016-10-20**, so this is the eLife-era PrePPI, NOT the 2023 *J Mol Biol* update that re-derives
structures from the AlphaFold DB. That matters for what the `str` score means — 2016 PrePPI's
structural evidence is PDB structures and homology models, so it is a genuinely independent
structural method rather than a fold-based one, which makes it a *better* orthogonal baseline but
a worse stand-in for "current PrePPI". Getting the 2023 release needs the lab.

1,545,710 pairs over 18,546 proteins, no duplicate unordered pairs, 6,969 self-pairs (homodimers,
excluded — a pair predictor scored on homodimers is answering a different question).

**Two things in the table change the design.**

**1. `final_score` leaks experimental labels, as feared, for 15.7% of pairs.** 242,887 pairs carry
an `exp_score`, and for 242,884 of them `final_score != total_score`. Every one of those also has
a literature reference in `pubs`. So PrePPI-`final` has seen the experimental evidence that the
benchmark's positives are drawn from. `str_score` only, as the design already required.

**2. `str_score` has 245 distinct values across 1.55M pairs — a distinct-value fraction of
0.00016.** PrePPI's component scores are binned likelihood ratios, not continuous:

| score | distinct values |
|---|---|
| `exp_score` | 7 |
| `phy_score` | 11 |
| `go_score` | 12 |
| `coexp_score` | 21 |
| `red_score` | 33 |
| `ort_score` | 43 |
| **`str_score`** | **245** |
| `protpep_score` | 578 |
| `total_score` | 492,594 |
| `final_score` | 516,101 |

Only `total` and `final` are near-continuous, and only because they are products of the binned
terms. `feedback_standalone_auc_tie_mass` is directly on point: a 0.807 standalone AUC in this
project turned out to be one saturated threshold with 35% of mass tied, and fell to 0.700 once
ties were stripped. So a 245-level `str_score` cannot be compared to a continuous ipTM without
handling ties explicitly, and its tie mass is reported beside every number it appears in. Half of
all PrePPI pairs sit at `str_score` <= 5.5.

This does not disqualify PrePPI as the baseline. It means the comparison is "does folding beat a
coarse structural-homology prior", and the coarseness is part of the finding rather than a
footnote.

## S0 results — the benchmark, and PrePPI measured

`scripts/01_build_benchmark.py` → 23,659 positives (STRING physical > 700 ∩ BioGRID MV physical,
human–human, inside PrePPI's 18,546-protein universe) and 473,180 negatives per arm.
`notebooks/s0_benchmark.ipynb` → `results/s0_benchmark.png`.

**The negative set was the first thing to get wrong.** Mean positive-set degree per pair:

| arm | mean pair degree |
|---|---|
| positives | 50.5 |
| random negatives | **5.1** |
| degree-matched (configuration model) | 44.0 |

Random negatives are drawn from proteins ten times less connected than positives. The first
degree-matched attempt binned log-degree into deciles and only reached 20.8, because 12,075 of
the 18,546 proteins appear in no positive at all, so `log1p(degree)` is 0 for 65% of them and the
quantile bins collapsed. `scripts/02_degree_ceiling.py` showed the ceiling was never the
constraint — 87% of top-200-hub pairs and 99% of top-2000-hub pairs are available as negatives —
so the matcher was replaced with a configuration-model draw (endpoints sampled ∝ positive-set
degree), which reaches 44.0. Both arms are carried through every later comparison.

**PrePPI does not do what the design assumed.** Three effects had to be separated, and merging
them is what the first version of `03_score_preppi.py` did wrong:

| quantity | AUROC, degree-matched |
|---|---|
| membership in PrePPI (`final` ≥ 600) alone | **0.911** |
| `str_score`, among covered pairs | **0.446** — below chance |
| `final_score`, among covered pairs | 0.901 |

92.4% of covered positives carry an `exp_score` against 17.9% of covered negatives, so membership
is largely a restatement of the experimental evidence the labels came from. Imputing absent pairs
at the floor — the obvious choice — adds +0.334 AUROC to `str_score`, and every point of that is
membership, not structure.

**Contamination was tested as the alternative explanation and rejected.** If the sub-chance AUROC
came from real interactions sitting in the negative set, dropping negatives with PrePPI
experimental support should raise it. It falls, to 0.421.

**The real explanation is that `str_score` is a tail detector, not a ranker.** Positives have a
higher mean (109 vs 20) and a *lower median* (2.2 vs 3.2) than negatives; the survival curves
cross. Top-k precision, degree-matched:

| top k | precision | enrichment over prevalence |
|---|---|---|
| 100 | 0.187 | **18.1×** |
| 1,000 | 0.245 | **23.7×** |
| 5,000 | 0.035 | 3.4× |
| 20,000 | 0.010 | 1.0× |

At or below the median `str_score`, precision is 0.8× the floor — worse than chance. The top ~1–2%
is strongly enriched and the rest is noise.

**Three consequences for the folding arms.**

1. **The metric is top-k precision and AUCPR at 1:1000, never AUROC.** Panel **c** shows it is
   forced by the data rather than a convention.
   *Checked against the paper (2026-11-19).* The paper reports **AUCPR** ("Using area under the
   precision and recall curve (AUCPR) as a metric", fig. 2D) and **precision at fixed recall**
   ("substantially improved recall at high precision (80% or 90%)"). **AUROC appears nowhere in
   it.** So it agrees on excluding AUROC, but it does not use top-k precision — that part is ours,
   adopted because top-k is the quantity a screen actually spends GPU-hours on.
2. **The bar is ~15–24× enrichment in the top 1,000**, at essentially zero recall. "Beating
   PrePPI" means beating that, and a method that merely ranks the whole set better is not
   obviously more useful.
3. **PrePPI cannot be used as a plain baseline column.** Its coverage, its membership rule and its
   `final` score all carry the experimental labels. Only `str_score` among covered pairs is
   admissible, and that comparison is restricted to the 62,708 pairs PrePPI covers.

Tie mass is reported with every number: `str_score` takes 245 distinct values over 1.55M pairs,
and the single tie block at 997.34 moves precision@1000 noticeably, which is why `precision_at_k`
sorts with a stable kind and prints `min_score_in_k`.

## Getting the omicMSAs — blocked, needs the user

Arm B needs `protein_omicMSAs.tar.gz`. Both published routes refuse an automated fetch:

| route | result |
|---|---|
| `conglab.swmed.edu/humanPPI/humanPPI_download.html` | TLS chain does not verify (no local issuer). Fetching it would mean disabling certificate checking |
| Dryad `doi:10.5061/dryad.15dv41p84`, `/downloads/file_stream/<id>` | HTTP 403 to curl, 401 to the v2 API. The page itself says the data is open with no embargo, so this is a bot check |

Neither should be worked around from here: one means turning off TLS verification and the other
means pretending to be a browser. The dataset is open, so a manual download is the clean fix.

Dryad file ids (`https://datadryad.org/downloads/file_stream/<id>`):

| id | file | size | why it is needed |
|---|---|---|---|
| 4357019 | `benchmarks.tar.gz` | 286 KB | **their benchmark pair sets — needed for arm D** |
| 4357022 | `RF2-PPI.pt` | 310 MB | RF2-PPI weights, arms C and D |
| 4357024 | `AF_scores.gz` | 32 MB | their per-pair AF scores; lets arm A′ be checked against their AF3 numbers without rerunning |
| 4357026 | `RF2-PPI_scores.gz` | 444 MB | their per-pair RF2-PPI scores, same purpose for arm D |
| 4357017 | `protein_omicMSAs.tar.gz` | 17.2 GB | **arm B** |
| 4357018 | `segment_omicMSAs.tar.gz` | 12.8 GB | segment-level MSAs, only if domain-level scoring is added |

Disk is not the constraint (5.8 TB free). The 286 KB `benchmarks.tar.gz` alone unblocks arm D,
which is the gate on everything else, so it is worth fetching first.

## The RF2-PPI benchmark has the degree confound too

`benchmarks.tar.gz` (CC BY 4.0) is 3,000 positives and 30,000 negatives as UniProt pairs,
matching the paper. Scored against an **independent** degree reference — my 23,659-positive
graph from BioGRID MV ∩ STRING physical > 700, which shares no construction with theirs:

| set | mean pair degree | median |
|---|---|---|
| their positives | 37.5 | 26 |
| their negatives | **4.1** | **0** |

The median negative pair contains proteins appearing in *no* known interaction at all. This is the
same ~9× confound my own random-negative arm had (50.5 vs 5.1) before it was replaced with a
configuration-model draw, and it is not a criticism that needed a new experiment to make — it is
the paper's stated design, "random human protein pairs with no evidence of interaction".

**Why this matters specifically for the omicMSA claim.** omicMSAs are assembled from genomic and
transcriptomic data, so their depth is a function of how much sequence exists for a protein's
homologs — which correlates with how studied the protein is, which is the very axis separating
these positives from these negatives. An MSA-depth advantage and a study-bias advantage are
confounded in this benchmark, and the conclusion drawn from it ("AF2 + omicMSA is best for a de
novo screen") is the one most exposed to that.

This is testable rather than rhetorical, and costs nothing extra: every arm is run on **both**
their benchmark and the degree-matched one. If the omicMSA advantage survives degree matching it
is real; if it shrinks, part of what omicMSA buys is knowing which proteins are well studied.

Their benchmark is kept as a third evaluation arm regardless, because reproducing their numbers on
their own pairs is what makes arm D a valid harness check.

## Getting the data — SOLVED

`conglab.swmed.edu` serves everything, and the TLS failure was never a reason to disable
verification. The chain is:

```
conglab.swmed.edu  <-  InCommon Intermediate CA - OVG2C (Internet2)  <-  emSign Root TLS CA - G1
```

The server omits the intermediate, and the OS bundle (`/etc/ssl/certs/ca-certificates.crt`) does
not carry the emSign root — but **certifi does**, because emSign is in the Mozilla root program.
So the fix is to complete the chain, not to bypass it: certifi's roots plus the published
intermediate (fetched from certificate transparency, the AIA URL in the cert returns empty) gives
a bundle that verifies the server fully.

```bash
python -c "import certifi; print(certifi.where())"           # Mozilla roots, has emSign
curl -o inter.pem "https://crt.sh/?d=25849656014"            # InCommon OVG2C intermediate
cat "$(python -c 'import certifi;print(certifi.where())')" inter.pem > chain_bundle.pem
curl --cacert chain_bundle.pem https://conglab.swmed.edu/humanPPI/downloads/<file>
```

Verified end to end: `AF_scores.gz` downloaded this way has sha-256
`4f2b931b...f643ed6`, which matches the digest Dryad records for the same file. Two independent
hosts, identical bytes.

Dryad itself returns a bare nginx 403 to non-browser clients — a bot block, not an access
restriction; the dataset is open and the landing page says so.

## Their validation set: homomers, and a confound that turns out not to bite

### Near-homomers (the user's point, measured)

Neither benchmark file contains a literal self-pair: 0 of 33,000 and 0 of 5,715. But identity
between the two members of a "heteromeric" positive tells a different story
(`scripts/04_near_homomers.py`, 5-mer screen then global alignment):

| set | near-homomers (≥30% id) | rate |
|---|---|---|
| positives (3,000) | 77 | **2.6%** |
| negatives (29,185 mapped) | 0 | **0.0%** |

The asymmetry is absolute: not one negative pair is a near-homomer, because negatives are random
pairs. 255 near-homomers across both files, **16 at ≥99% identity** — those are homodimers filed
as heteromers. The clearest case is `P0DP23_P0DP24`, `P0DP23_P0DP25`, `P0DP24_P0DP25`: CALM1,
CALM2 and CALM3 encode an *identical* calmodulin sequence, so all three "interactions" are one
homodimer counted three times. Also at 100%: the AMY1A/AMY1B/AMY1C amylases and HSPA1A/HSPA1B.

This matters more than 2.6% suggests, because it is concentrated where the paper makes its
strongest claim:

| interface class | near-homomer rate |
|---|---|
| **large (strong)** | **5.8%** (103/1,782) |
| medium | 1.7% (62/3,669) |
| small (weak) | 4.9% (13/264) |

"AF2 is better for strong interactions with large interfaces" is argued on the class with the
highest near-homomer load. For a paired-MSA method a near-identical pair is degenerate — the two
chains draw on the same homolog families, so cross-chain coevolution is really within-family
covariation, and the pair is detectable by sequence identity alone with no interaction biology.
Near-homomers are therefore excluded from the primary evaluation here and reported as their own
stratum.

### The degree confound does NOT bite AF2 — my prediction was wrong

The previous section predicted that AF2 + omicMSA's advantage would shrink under degree matching,
on the reasoning that omicMSA depth tracks how studied a protein is. Tested with their own
deposited AF2 scores (`scripts/05_af2_on_both_benchmarks.py`), it does not:

| evaluation | AUCPR | AUROC |
|---|---|---|
| their benchmark as published | 0.2914 | 0.6611 |
| same pairs, negatives degree-matched to positives | **0.2885** | 0.6798 |

−1.0% in AUCPR, and AUROC goes *up*. Same pairs, same scores, only which negatives are kept has
changed. The degree asymmetry in their benchmark is real (positives 37.5, negatives 4.1 on an
independent graph) but AF2 does not cash it in. **The published AF2 number is more robust than I
claimed it would be, and that correction stands on the record.**

Two limits on that test. The matching is partial — their negatives simply do not contain enough
high-degree pairs to reach the positives' mean, so matching lifts negatives from 4.1 to 12.4
against 37.5, and the top degree decile stays unmatched. And AF2 here is AF2+omicMSA together, so
this bounds the combination, not omicMSA alone.

### Why their AF2 scores cannot be used on our benchmark

`AF_scores.gz` covers their own benchmark completely (1.000 / 1.000) but only 88% of our
positives and 2–7% of our negatives. AF2 was run on whatever the DCA → RF2-PPI cascade shortlisted
*plus*, per the file header, every pair "from PPI databases, BioGrid, STRING (physical), and
UniProt". So having an AF2 score is itself a restatement of being in a PPI database — the same
membership leak that made PrePPI look skilful. Scoring our benchmark with these numbers would
measure the leak, so it is not done, and the degree test runs inside their benchmark instead.

### The deposit confirms the unfilled cell

`AF_scores.gz` has exactly one score column, `AF2_interaction_probability`, over 3,381,654 pairs.
There is no AF3 column anywhere in the deposit. AF3 appears only as a web-server comparison in a
supplementary figure. **AF3 + omicMSA does not exist in their data**, which is the independent
confirmation of the reframing — read off their files rather than their prose.

## HuRI is the independent set, and here is the number that says so

HuRI/HI-union (Luck et al., Nature 2020) downloaded from `interactome-atlas.org` — note the bare
hostname, the certificate does not cover `www.`. 64,006 pairs, 62,473 mapping to heteromeric
UniProt pairs. It is a systematic all-by-all Y2H screen, so which pairs are tested does not
depend on which proteins anyone has studied.

Mean pair degree, all measured in the same independent literature graph (our 23,659 positives):

| set | mean pair degree |
|---|---|
| our literature positives | 50.5 |
| RF2-PPI benchmark positives | 37.5 |
| RF2-PPI benchmark negatives | 4.1 |
| **HI-union positives** | **6.1** |

HuRI's positives sit at the level of the *negatives*. That is the whole point of an all-by-all
screen and it is why HuRI is the independent arm: the study-bias axis that separates positives
from negatives in every literature-derived benchmark is simply absent. Only 1,203 of 62,473
HI-union pairs appear among our literature positives, so the sets are 98% disjoint.

This also sharpens the omicMSA question rather than just adding a dataset. HuRI positives are
low-degree, less-studied proteins, which is exactly where alignments are shallowest. If omicMSA
depth is a real mechanism it should still help there; if its advantage on literature benchmarks
came from well-studied proteins having deep alignments, HuRI is where that disappears. The
AF2 degree-matching test inside their benchmark found no such effect (-1.0% AUCPR), but that test
could only lift negatives from 4.1 to 12.4. HuRI tests the same thing at the other end, on
positives, with no matching required.

Evaluation arms are therefore three: their benchmark (harness check), our degree-matched
literature benchmark, and HuRI (independent, systematic, study-bias-free). PrePPI `str_score`
stays as the structural baseline on the pairs it covers.

## Arm D: the harness reproduces, and the noise is structured

`scripts/run_armD_example.sh` ran RF2-PPI in the SE3nv container on the authors' own pre-paired
example MSAs, three times, against the expected output they ship.

| pair | expected | rep1 | rep2 | rep3 | mean | sd |
|---|---|---|---|---|---|---|
| P50395_P62820 | 0.0833 | 0.1077 | 0.0688 | 0.0659 | 0.0808 | 0.023 |
| Q13153_Q9H4E5 | 0.4895 | 0.7820 | 0.5173 | 0.3728 | 0.5574 | **0.208** |
| A1L188_Q5TEU4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000 |
| P11802_P30279 | 0.9998 | 0.9997 | 0.9996 | 0.9998 | 0.9997 | 0.0001 |
| Q8TAE8_Q96DV4 | 0.0399 | 0.1059 | 0.0462 | 0.0426 | 0.0649 | 0.036 |

Mean |mean − expected| = **0.0191**. Three of five expected values fall inside the replicate
range; the two that do not miss by 0.0001 (a saturated pair) and 0.0251. **The harness
reproduces**, which is the precondition for reading anything from arms A–C.

**The noise is not uniform, and that is the useful part.** Replicate sd by probability level:

| level | sd |
|---|---|
| saturated (> 0.95) | 0.0000 |
| low (< 0.15) | 0.0295 |
| **intermediate** | **0.2075** |

Saturated predictions are deterministic; the informative middle is not. The README warns of
sd > 0.1 for about 5% of cases, and this is the same phenomenon seen directly. Taken naively it
would mean no two arms can be compared — on a single pair, they cannot.

**But it averages out at the metric level, and that was measured rather than hoped.** Injecting
logit-scale noise calibrated to reproduce sd 0.21 at p = 0.5 into their deposited RF2-PPI scores
and recomputing AUCPR 30 times:

| noise | AUCPR |
|---|---|
| none | 0.2560 |
| matched to arm D | 0.2520 ± **0.0025** |
| 2× measured | 0.2172 ± 0.0167 |

So single-run arms are comparable at the AUCPR level provided the difference exceeds roughly
0.005 (2 sd), with a small downward bias. Differences smaller than that need replicates.
(The absolute 0.2560 is not their published figure: the deposited RF2-PPI scores cover all 3,000
benchmark positives but only 5,983 of 30,000 negatives, so this is a subset — fine for a noise
simulation, which is relative, and not quotable as their result.)

**Throughput.** Steady state is 4.1 s per pair against the 9.2 s in their own logs, i.e. about
2.2× faster, despite all three GPUs sitting at 100% utilisation from other jobs. The first call
took 375 s (JIT plus model load under contention) and must be excluded from any throughput
estimate.

## The omicMSA quality mask forks the design

The omicMSA README says to use only positions marked `*` in the `mask` record, "as we did in our
work", and `generate_protein_pair_MSA.py` implements that by **deleting** the masked positions
from the query and every row.

For RF2-PPI that is harmless — it consumes an alignment and never emits a structure, so deleting
columns deletes evidence and nothing else. **For AF3 it is not, because AF3 folds the query.**
Measured over 400 randomly sampled omicMSAs:

| fraction of positions kept | proteins |
|---|---|
| 100% | 3.8% |
| ≥ 95% | 17.5% |
| ≥ 90% | 34.5% |
| ≥ 75% | 68.2% |
| **< 75%** | **31.8%** |

Median kept 83.3%. On their own five example pairs the effect ranges from 5 residues (1%) to
**179 of 545 residues, 33%**, for Q13153. Handing AF3 the masked query would mean folding a
protein with a third of its residues deleted and then reporting the interface of whatever that
produced.

So arm B applies the mask **as gaps, not as deletions**: the query keeps every residue, and each
homolog row carries its aligned residue at high-quality positions and `-` at low-quality ones.
This expresses exactly what the mask means — the aligned residue here is untrustworthy — without
truncating the thing being folded, and the evidence AF3 sees at masked positions is then the same
as what RF2-PPI sees, namely none. The insertion rule is theirs: an insertion is kept only when
both flanking query positions are high quality.

Pairing is by genome-accession intersection, unchanged from their script, which is well defined
because an omicMSA carries one sequence per genome. Redundancy filtering is hhfilter at 90% on
the *concatenated* alignment, also as theirs — filtering the chains separately would keep
different rows for each and destroy the row-index pairing AF3 relies on.

Templates are disabled for arm B. RF2-PPI is MSA-only, so leaving AF3's template search on would
confound the engine comparison with a structural-database lookup, and PDB leakage has already
cost this project once (`finding_stcrdab_benchmark_contaminated`).

`scripts/07_build_af3_inputs.py --validate` reproduces their masking rule exactly on all five
shipped example pairs, which is what licenses reading the parser's output.

## Arm B runs. First AF3 + omicMSA numbers, and the pilot sizing

AF3 accepts the gap-masked, accession-paired omicMSAs with `--norun_data_pipeline`, so arm B's
input path works end to end. All five of their example pairs fold cleanly.

| pair | AF3+omicMSA ipTM | pTM | RF2-PPI | tokens |
|---|---|---|---|---|
| A1L188_Q5TEU4 | 0.720 | 0.650 | 1.000 | 419 |
| P11802_P30279 | 0.800 | 0.830 | 1.000 | 592 |
| P50395_P62820 | 0.400 | 0.730 | 0.083 | 650 |
| Q13153_Q9H4E5 | 0.710 | 0.470 | 0.489 | 759 |
| Q8TAE8_Q96DV4 | 0.210 | 0.460 | 0.040 | 602 |

Five pairs is a direction, not a result. The direction is that the two **rank** these almost
identically (Spearman 0.9) on a very different **scale**: RF2-PPI saturates at 0.04 and 1.00
where AF3 spans 0.21 to 0.80. Given that RF2-PPI's tie mass is what made PrePPI's `str_score`
unrankable in S0, a non-saturating score may matter more for AUCPR than the ranking agreement
suggests. That is a hypothesis for the pilot to test, not a finding.

**Two bugs found here, both mine, both silent.**

1. `hhfilter_pairs` split the concatenated alignment with `s[:l1]` — a raw character slice —
   having written the correct query-position splitter in `scripts/06` and documented exactly why
   a raw slice is wrong. The assertion used was `len(sa) == l1`, which counts characters and so
   passed. AF3 caught it: "all strings must have the same number of non-lowercase characters;
   First string has 445 ... but ... has 443". Four of five pairs failed; the fifth succeeded
   because it happened to have no insertions before the boundary, which was enough to make the
   pipeline look like it worked. Fixed, and `af3_json()` now refuses to write any pair whose
   rows do not match the query's query-position count.
2. The sizing script assigned a fresh default-indexed Series into a row-filtered DataFrame, so
   pandas aligned on index and scrambled the keys. The merge still returned 8,983 rows, so
   coverage looked right while the scores belonged to other pairs, and AUCPR sat at the 0.001
   prevalence floor and *fell* as n grew. Caught by that impossible monotonicity, not by any
   check. Fixed with `reset_index(drop=True)`, and a positives-vs-negatives mean-score sanity
   line now prints beside every such join.

**Throughput.** After the first JIT compile, 33–46 s inference plus 10–15 s featurisation per
pair at bucket 768, so ~55 s per pair, one job per GPU (66.6 GB for a 419-token pair). Faster
than the 124.5 s the first, uncached fold suggested.

**Pilot size**, bootstrapped from their deposited RF2-PPI scores on their benchmark:

| n_pos | n_neg | total | AUCPR | subsample sd | GPU-hours |
|---|---|---|---|---|---|
| 100 | 400 | 500 | 0.271 | 0.050 | 7.6 |
| 200 | 800 | 1,000 | 0.276 | 0.024 | 15.3 |
| **500** | **2,000** | **2,500** | **0.260** | **0.0154** | **38** |
| 1,000 | 4,000 | 5,000 | 0.256 | 0.0094 | 76 |
| 3,000 | 5,983 | 8,983 | 0.256 | 0 | 137 |

Subsample noise (0.015 at n=2,500) dominates the score-replicate noise (±0.0025) by sixfold, so
sample size, not model stochasticity, is the binding constraint. **But the arms share their
pairs**, so the comparison is paired and most of the sampling noise cancels: the quantity that
matters is the sd of the arm-to-arm *difference* on the same subsample, which is far smaller
than the per-arm sd above. The pilot is therefore 500 positives and 2,000 negatives, ~38 GPU-hours
per arm, with the paired difference as the read-out and the per-arm AUCPR reported with its
subsample interval so the two are not confused.

## Co-folding as a screening score: the literature

`LITERATURE_cofolding_screening.md` collects what other groups have found, added
2026-11-21 after the orphan screen's metabolite result. The short version: four
independent groups report that co-folding beats docking at placing a ligand and loses to
docking at deciding whether the ligand belongs there, so our result is corroborated and
not new. Three of them also find ipTM the weakest of the available confidence metrics for
the ligand side — the metric this screen used.

Nearest analogue: DeorphaNN (Molecular Cell 2026) modelled 99.8% of true GPCR agonist
peptides and 96.6% of non-agonists in the pocket, a 3.2-point gap, under the heading
"AF-Multimer positions peptides in GPCR binding pocket with minimal discrimination".

Two consequences, both actionable without re-folding anything: re-score the orphan folds
on ligand-pLDDT instead of ipTM, and treat the counter-screen as required and not
optional — Shamir et al. (JACS 2026) ran the equivalent control against non-cognate
kinases and concluded AF3 is "currently not suitable for selectivity prediction".

## Making the pipeline survive its own growth (2026-11-23)

The screens here are scored while they are still folding, and new benchmarks keep being
added behind the ones already running. That combination is what produces the failure mode
this project keeps hitting: not a crash, but a file that looks finished and is not.

Found before changing anything, by checking the artefact rather than the code:
`results/orphan_scores.csv` held **3,017 folds, was written 2026-11-20, and its own
scoring code had been edited 2026-11-22** — so it predated both the code that made it and
1,889 of the folds it was supposed to summarise, with nothing on disk saying so. Figure 4
and the Results text were computed from it.

### What was wrong, structurally

- **No record of coverage.** A scoring pass over 60% of a screen wrote the same filename
  as a pass over 100%, so a partial result was indistinguishable from a complete one.
- **A partial pass could overwrite a fuller one**, which is the normal direction of
  failure when scoring runs alongside folding.
- **A changed scorer left the old rows in place.** Editing the scoring function did not
  invalidate anything; the file simply went on holding numbers no current code produces.
- **Full recomputation every run**, so re-scoring after new folds landed meant re-parsing
  every CIF — which is why it was not run often, which is why it went stale.
- **Nothing written until the end.** An interrupted run lost everything, so a long
  re-score was something to avoid rather than something to do routinely.
- **"Not yet folded" scored as "failed".** A fold still being written has no summary
  JSON, which is exactly what a crashed fold looks like; both got a row with NaN terms.
- **A partial matrix flattered itself.** `18_score_specificity.py` ranked each ligand's
  own receptor against however many receptors had landed, while printing chance for a
  panel of 16. At 6 of 16 folded, a ligand at rank 2 of 6 read as rank 2 of 16.

### What now holds it together

`scripts/score_store.py` — one module all the scorers share:

- `classify()` splits every fold in the manifest into ready / writing / queued / absent,
  **reading directory contents, not mtime**. The runners create every output directory and
  drop the AF3 input JSON in it when the shard starts, so a fold queued behind 4,000
  others has a directory days old; judging liveness by mtime called 95 queued folds
  stalled. A queued fold is only reported as stalled when no runner is alive, and when
  that is unknown the distinction is left open instead of guessed.
- The denominator is always the manifest, never what happens to be on disk. The latter
  grows as the screen runs, which makes every fraction computed against it unreadable.
- Every result gets a `.coverage.json` sidecar: folds scored, folds in the manifest,
  how many are queued, and `complete` true/false. `require_complete()` lets a figure or a
  report refuse to quote a number from a partial screen, and refuses outright on a CSV
  that has no sidecar at all.
- `write()` is atomic and **refuses to shrink** an existing result without
  `--allow-shrink`, which is recorded in the sidecar when used.
- Rows carry a fingerprint of the files they were computed from and a hash of the scoring
  source. A changed scorer drops every stale row; a fold re-run under the same job name is
  re-scored, because reuse keys on the bytes and never on the name.
- `--checkpoint-every` persists partial results, so an interrupted run resumes.
  `--seed-from` lets a new output inherit a finished run's valid rows.

`scripts/00_status.py` — one command for the state of every stage, with a registry so a
new benchmark is visible the moment its manifest exists. It reports the fold rate over a
recent window and a **cost-weighted ETA**, because the queues are size-ordered and
folds/h falls while GPU throughput is constant: on 2026-11-23 the orphan screen's 93
remaining folds had a median of 3,149 tokens against 819 for the 4,906 done, so the naive
count-based estimate of 8.5 h and the cost-weighted 7.9 h agreed, but an earlier estimate
made from the count alone had said 2.7 h. It also flags a scores file older than its own
scoring code, and a stage with work queued and no shard alive.

`scripts/check_scorer_regression.py` — `one()` is deliberately shared by every arm and
every screen, so a benchmark pair and a candidate pair pass through identical code. The
cost is that an edit for one stage silently moves numbers already published from another.
24 folds spanning ipTM 0.03–0.95 are pinned in `tests/golden_armB_sample.csv`; the guard
re-scores them and exits non-zero if any existing column moves. Adding a column is fine.

`tests/test_score_store.py` — 11 tests, one per guarantee above.

### Also done, and it changes a number

The per-chain confidences the literature review called for are now computed inside `one()`
from the parse it already did, instead of in a second parse in the orphan scorer only.
That removes a duplicated CIF parse, and it means the benchmark arms get `plddt_B` on the
same definition as the candidates. Verified against 24 pinned folds: every existing column
bit-identical, `plddt_B` populated on all of them. The arms' published numbers, including
the B − A engine difference, are unaffected.

### Provenance: is the fold on disk the fold the manifest describes?

The builders rewrite every input JSON unconditionally, and the runners skip any job whose
summary JSON exists. Together those two reasonable behaviours mean an input that changes
after its fold has run — a new UniProt release, a different isoform, a corrected SMILES, a
re-picked candidate list — leaves the old structure in place while the manifest describes
something else. Nothing errors.

A job name is not proof of what is in the file. AlphaFold3 writes the sequences it actually
used into `<job>_data.json` beside the model, so `scripts/check_fold_provenance.py` checks
by content, after the fact, with no change to the runners. It screens on mtime first (an
input older than its own fold cannot have drifted) and compares content only for the rest,
because each record embeds its MSAs and reading 5,000 of them is slow.

Run 2026-11-23 on the orphan screen: **4,908 folds ready, 4,908 inputs older than their
fold, 0 drifted, 0 outputs missing from the manifest.** The screen is clean on provenance.

Covered: per-chain sequences, chain ids, ligand CCD codes and SMILES. Not covered: MSA
content, which is large and legitimately varies between builds. Verified to catch a
single-residue change and a chain-id swap, and to be stable to a reordering of the
sequence list (`tests/test_fold_provenance.py`, 9 tests).

### The audit, and the two things it found that mattered

A 32-agent audit of the scoring scripts, fold runners, input builders and figure
notebooks, each finding then adversarially verified. Most of what came back was already
handled by the store above. Two were not, and both were confirmed by hand against the
logs before anything was changed.

**1. 85 of the 5,000 orphan folds were impossible, not slow.**
`12_build_orphan_inputs_omic.py` handled a partner with no omicMSA by writing the input
with no `unpairedMsa` or `pairedMsa` key at all, and flagging it `none_single_sequence`.
Omitting the key does not mean "fold single sequence": under `--norun_data_pipeline`
AlphaFold3 cannot build an alignment and rejects the job —
`ERROR: Fold job GPR101__NRXN1 failed: Protein chain 1 is missing unpaired MSA`,
**462 such events across exactly 85 distinct jobs, none of which ever produced a
structure.** They were 85 of the 91 folds still outstanding, so the screen was reporting
~96 remaining when 6 were real. The ETA I had quoted from that count was meaningless.

The same branch also threw away the **receptor's** alignment, which was never missing —
all 75 receptors have an omicMSA. It made those folds worse than they needed to be for no
reason.

Fixed in the builder, and repaired in place by `13_repair_unfoldable_inputs.py`, which
rewrites only the failing jobs: a full rebuild would make all 5,000 inputs newer than
their folds, which is exactly the signal the provenance check uses, and would turn 4,909
clean folds into suspects. All 85 repaired, pre-flight now clean, manifest relabelled
`partner_single_sequence` because the old label claimed a fold that never happened.

`scripts/check_af3_inputs.py` is the gate that would have caught it in seconds: every
protein chain must arrive with an embedded MSA, in A3M, with a templates key, and chain
ids must be unique. The specificity (256) and HuRI (600) input sets pass it today.

**2. The runners measured progress by counting directories, which AlphaFold3 creates up
front.** `before=$(ls -d "$OUT"/*/ | wc -l)` against a `NO PROGRESS → exit 1` test. AF3
writes `<job>/` and `<job>_data.json` when a batch *starts*, so the count is wrong in both
directions:

- **Saturated.** At 5,000 directories against 4,909 finished folds, `after <= before` is
  true forever, so both shards were going to exit "NO PROGRESS" the moment their current
  batch returned — with folds having succeeded inside it.
- **Inverted on a fresh stage.** A batch that runs out of memory on every job still
  creates every directory, so `after > before` and a total GPU failure reads as healthy
  progress. This project has the per-item-try/except version of that written down
  already; this is the same bug in shell.

Now counts `*_summary_confidences.json` in all three runners. `run_orphan_fold.v2.sh` also
retires a job after three identical failures and writes `run/orphan_retired.txt`, because
a permanently unfoldable job never left the todo list, so `n` never reached 0 and the
clean exit was unreachable — the shard could only ever leave through the failure path.

`00_status.py` now separates work that *cannot* be done from work that is *pending*, and
builds the ETA only on the latter. Checking inputs is scoped to the jobs still queued:
validating the 4,910 already folded would read ~37 GB of embedded MSAs to learn nothing,
and a status command that takes minutes does not get run.

**`run_queue.v2.sh`** replaces the queue. It refuses to advance past a stage that did not
reach a completion bar — the old version advanced whenever the marker files disappeared,
which is also what happens when a shard dies in its first second, so one early death
would have scored nothing, logged FAILED, and launched the next stage anyway. It also
restarts a shard that dies with folds remaining, and runs the provenance check before
redrawing the figure. Written as a new file because bash reads a script incrementally as
it executes, so rewriting the one a live shell is reading moves the offset under it.

**Figure 4** now refuses to redraw from a scores file older than its own scoring code
(`ALLOW_STALE_SCORES=1` to inspect, never to publish), and writes its coverage sentence
to `results/fig4_coverage.txt` from the data instead of having it typed into the legend.
An unfinished screen is expected and only has to be stated; a stale one is never
publishable.

### Three more from the audit, all confirmed by measurement first

**Shards were stealing each other's jobs.** The split was
`[f for i, f in enumerate(todo) if i % nshard == shard]`, where `todo` is the jobs not yet
finished. That list shrinks as folds complete, so every remaining job is renumbered on
each pass and shard membership flips. Measured on the shard logs: **1,403 of 5,000 jobs
were picked up by both shards** — wasted GPU, and two AlphaFold3 processes able to write
one output directory. Adding candidates reshuffles it the same way, so it would have got
worse with every addition.

Now shards on `sha1(job) % nshard`: invariant to completion, to additions, and to the
manifest's order. sha1 and not Python's `hash()`, which is salted per process, so two
shards running separately would not even have agreed with each other. Measured after the
change: overlap 0, balance 0.972, and adding 50 candidates moves no existing job.
Pinned in `tests/test_sharding.py` (7 tests), including one that reads the rule back out
of all three runner scripts so the shell and the test cannot drift apart.

**The detection floor was compared against the wrong sample.** `18_score_specificity.py`
computed one floor from the full cognate and negative counts, while `auroc()` silently
drops non-finite values — so a metric missing on some folds was measured on fewer pairs
and then tested against a bar it was never tested at. The floor is 0.711 at the design's
16 cognates, 0.792 at 8, 0.867 at 5. An AUROC of 0.75 on 8 cognates would have been
reported as clearing the floor when its real bar is 0.792. Now the floor is recomputed per
metric from the sample that produced the number, `n_pos`/`n_neg`/`dropped_*` are columns in
the output, and `beats_floor` is NaN rather than False below 8 positives — False reads as
"tested and did not clear it".

**The cognate label was positional.** `label = "cognate" if li == ri` is correct only
while both loops walk the same panel in the same order. A 17th row, a dropped row, or a
sort, and the diagonal moves while the label does not — so a non-cognate cell becomes the
ground truth the whole benchmark is scored against. Now labelled by identity
(`rec.receptor_gene == lig.receptor_gene`, the panel being one row per receptor and its own
cognate ligand), with asserts that there is exactly one cognate per ligand and per
receptor, that job names are unique, and that the matrix is the full product. Checked
against the existing manifest: **0 of 256 labels change**, so the queued benchmark is
correct as it stands and needs no rebuild.

### Figures: a typed limit is a silent filter

`figstyle.assert_in_view()` refuses to draw a panel that leaves data outside its own axes,
and `figstyle.view_limits()` derives limits from the data. Figure 4 panel c had
`ax.set_xlim(-200, 8600)`, a number typed against the data of the day: **96 folds had
already moved outside it** as the large complexes landed, and the buried-area maximum went
from 12,263 to 36,736 Å². Those points were not drawn and nothing said so — and they are
exactly the folds the figure was being redrawn to include. Panel c is now a log axis over
the data's own range; panel d's y-limit and its annotation anchor follow the data.

Panel d also now carries a `designed` column beside `tested`. `tested` counts the receptors
a partner has actually been folded against, which is not how many it was designed against
while the screen is unfinished, and "confident against every receptor" silently merges a
partner at 3 of 3 with one at 75 of 75. The headline is now reported a second time
restricted to partners whose full receptor set has been folded: if it only holds on the
partly-folded ones, it is about coverage and not biology.

### The orphan screen is closed at 4,913 of 5,000 (2026-11-23)

The remaining 87 folds were dropped on the user's decision rather than pursued.
`scripts/16_drop_unfolded.py` records that as an explicit act: the job names go to
`run/orphan_retired.txt`, which the fold runner skips and the status command counts as
dropped, and `results/orphan_dropped.csv` carries one row per job with its reason, size
and rank. Nothing is deleted; removing a line un-drops a job.

What was given up, and it is not a random sample:

| | dropped | in manifest | share of that arm lost |
|---|---|---|---|
| `partner_single_sequence` | 85 | 85 | **100%** |
| `omic_paired` | 2 | 4,769 | 0.04% |
| `omic_receptor_only` (metabolites) | 0 | 146 | 0% |

So the cost is one whole arm: the candidates whose partner had no omicMSA have **no data at
all**, and the single-sequence comparison the screen was designed to support cannot be made.
The two `omic_paired` losses are CUDA out-of-memory at 1,488 and 3,204 tokens. The
metabolite arm is complete.

The dropped folds are 3.9x the median size of the folded ones (3,171 against 820 tokens),
which invites the wrong conclusion. Size is **not** what decided it: of the 4,471
candidates at or above the smallest dropped size, 4,384 folded (98%). They are large
because the partners that lack an omicMSA are large proteins — neurexins, teneurins — not
because large complexes failed. The sentence for the Results should name the arm, not the
size.

`00_status.py` now separates `dropped_by_decision` from `blocked_bad_input`, because
conflating them misleads in opposite directions: a bad input is a bug to fix and fold, a
retired job is a decision to leave alone and report as a gap. It also prints coverage
against the manifest (98.3%) alongside coverage against what is still in scope (100%), and
says to quote the first — otherwise dropping work makes the screen look finished.
