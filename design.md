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
- AF3 was run through the **"AF3 web server widely used by the scientific community"**, i.e. on
  its own MSA pipeline, not on omicMSAs.
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

1. **The metric is top-k precision and AUCPR at 1:1000, never AUROC.** The RF2-PPI paper made the
   same choice; panel **c** shows it is forced by the data rather than a convention.
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
