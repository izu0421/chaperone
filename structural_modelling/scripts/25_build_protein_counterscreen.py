"""AF3 inputs for the PROTEIN specificity matrix: the counter-screen the protein
arm of the orphan screen never had.

The metabolite arm was readable only once 16 ligands had been folded against 16
receptors whose answer is known. The protein arm is still gated on thresholds
fitted on the RoseTTAFold2-PPI benchmark, which is soluble pairs of unusually
well-studied proteins, and transferring them to GPCR complexes does not work:

    median interface pLDDT   benchmark positives 71.9
                             benchmark NEGATIVES 63.7
                             orphan GPCR folds   60.4

The orphan folds sit below the benchmark's decoys on the term that does all the
cutting. At that gate's measured operating point (37.2% recall, 2.40% of
decoys) 4,767 pure decoys would return ~114 passers; the screen returned 25. A
screen returning fewer candidates than its own false-positive rate predicts is
not a clean screen, it is a mis-set threshold, and nothing in the screen can
tell which.

So this builds the matched panel. Two blocks, both folded by the same code as
the screen so the numbers are comparable to it:

  A  chemokine receptors x chemokines, 16 x 16.
     Chosen because the geometry is the screen's own geometry -- a secreted
     protein meeting the receptor's extracellular face, which is exactly what
     CCL11/GPR132 and MST1/GPR37 claim -- and because the interaction map is
     curated. Complexes are small (~350 aa receptor + ~75 aa chemokine), so the
     whole block costs about what the 256-fold metabolite panel cost.

  B  seven-helix bundle against seven-helix bundle.
     Known heterodimers against receptor pairs with no dimerisation evidence.
     This block exists because GPR62 + ADORA1 passed every rule the project
     has, including the topology rule -- two bundles meeting face-matched
     through their transmembrane and extracellular surfaces is what a real
     heterodimer looks like. Whether AF3 packs ANY two bundles that way is not
     answerable from the screen, which folded ADORA1 against three receptors.

Labels come from the Guide to Pharmacology, per cell, not from memory:

    binder     an interaction recorded for that exact pair. Includes
               antagonists: the question here is whether the complex forms,
               and an antagonist binds.
    negative   no interaction recorded for the pair anywhere in GtoPdb, not
               only no endogenous one. Chemokine cross-reactivity is mapped in
               detail, so absence is informative here in a way it would not be
               for an arbitrary protein pair.

Usage:
    micromamba run -n structmod python -I scripts/25_build_protein_counterscreen.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("prot-counter")

ROOT = Path(__file__).resolve().parent.parent
GTOPDB = ROOT / "bench" / "gtopdb_raw" / "interactions.csv"
ALNS = ROOT / "data/rf2ppi_raw/protein_omicMSAs/alns"
OUT = ROOT / "work" / "protein_counterscreen_inputs"

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)

# Block A. Receptors first, then their protein ligands. Accessions are carried
# here because the omicMSA store is keyed by accession and a gene symbol lookup
# is one more place for the wrong protein to enter the panel silently.
RECEPTORS = {
    "CCR1": "P32246", "CCR2": "P41597", "CCR3": "P51677", "CCR5": "P51681",
    "CCR6": "P51684", "CCR7": "P32248", "CXCR1": "P25024", "CXCR2": "P25025",
    "CXCR3": "P49682", "CXCR4": "P61073", "CX3CR1": "P49238", "ACKR3": "P25106",
    "CCR4": "P51679", "CCR8": "P51685", "CXCR5": "P32302", "CXCR6": "O00574",
}
LIGANDS = {
    "CCL2": "P13500", "CCL3": "P10147", "CCL5": "P13501", "CCL11": "P51671",
    "CCL19": "Q99731", "CCL20": "P78556", "CXCL8": "P10145", "CXCL10": "P02778",
    "CXCL12": "P48061", "CX3CL1": "P78423", "CCL21": "O00585", "CXCL11": "O14625",
    "CCL17": "Q92583", "CCL1": "P22362", "CXCL13": "O43927", "CXCL16": "Q9H2A7",
}

# Block B. Pairs, not a matrix: a full matrix of receptors against receptors
# would be 16x16 of 700-token complexes for a question two dozen folds answer.
BUNDLES = {
    "GABBR1": "Q9UBS5", "GABBR2": "O75899", "MTNR1A": "P48039", "MTNR1B": "P49286",
    "DRD1": "P21728", "DRD2": "P14416", "ADORA2A": "P29274", "AGTR1": "P30556",
    "AGTR2": "P50052", "GRM2": "Q14416", "GRM4": "Q14833", "ADORA1": "P30542",
    "GPR62": "Q9BZJ7",
}
# heterodimers with structural or pharmacological support in the literature
DIMER_POSITIVE = [
    ("GABBR1", "GABBR2"),   # obligate heterodimer, solved
    ("MTNR1A", "MTNR1B"),
    ("DRD1", "DRD2"),
    ("ADORA2A", "DRD2"),
    ("AGTR1", "AGTR2"),
    ("GRM2", "GRM4"),
]
# receptor pairs with no dimerisation evidence, drawn across classes
DIMER_NEGATIVE = [
    ("GABBR1", "AGTR1"), ("GABBR1", "ADORA2A"), ("GABBR2", "DRD1"),
    ("MTNR1A", "AGTR2"), ("MTNR1B", "GRM2"), ("DRD1", "GRM4"),
    ("DRD2", "AGTR1"), ("ADORA2A", "GRM2"), ("AGTR1", "GRM4"),
    ("AGTR2", "GRM2"), ("GRM2", "MTNR1A"), ("GRM4", "ADORA2A"),
]
# the pair that prompted the block, carried as its own class so it is never
# counted as either a positive or a negative
DIMER_TEST = [("GPR62", "ADORA1")]


def gtopdb_labels(receptors: list[str], ligands: list[str]) -> pd.DataFrame:
    """One label per cell, read off the curated interaction table."""
    if not GTOPDB.exists():
        raise SystemExit(
            f"{GTOPDB} is missing. Fetch it first:\n"
            f"    curl -L -o {GTOPDB} https://www.guidetopharmacology.org/DATA/interactions.csv")
    d = pd.read_csv(GTOPDB, skiprows=1, low_memory=False)
    d = d[(d["Target Species"] == "Human") & (d["Ligand Species"] == "Human")]
    hit = d[d["Target Gene Symbol"].isin(receptors) & d["Ligand Gene Symbol"].isin(ligands)]

    lab = pd.DataFrame("negative", index=receptors, columns=ligands)
    action = pd.DataFrame("", index=receptors, columns=ligands)
    for _, r in hit.iterrows():
        t, l = r["Target Gene Symbol"], r["Ligand Gene Symbol"]
        lab.loc[t, l] = "binder"
        if pd.notna(r.get("Action")):
            action.loc[t, l] = str(r["Action"])
    log.info("GtoPdb: %d recorded interactions inside the %dx%d panel",
             len(hit), len(receptors), len(ligands))
    return lab, action


def write_pair(name: str, acc_a: str, acc_b: str) -> int:
    """One AF3 input, built by the screen's own pairing and masking code."""
    qA, qB, rowsA, rowsB = b7.build_pair(acc_a, acc_b)
    keptA, keptB = b7.hhfilter_pairs(qA, qB, rowsA, rowsB)   # returns the rows only
    js = b7.af3_json(name, qA, qB, keptA, keptB)
    (OUT / f"{name}.json").write_text(json.dumps(js))
    return len(qA) + len(qB)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in ALNS.glob("*.a3m")}
    everything = {**RECEPTORS, **LIGANDS, **BUNDLES}
    missing = {g: a for g, a in everything.items() if a not in have}
    assert not missing, f"no omicMSA for {missing}"

    R, L = list(RECEPTORS), list(LIGANDS)
    lab, action = gtopdb_labels(R, L)

    rows, failed = [], []
    for rec in R:
        for lig in L:
            name = f"{rec}__{lig}"
            try:
                n_tok = write_pair(name, RECEPTORS[rec], LIGANDS[lig])
            except Exception as exc:                      # noqa: BLE001
                failed.append((name, str(exc)))
                continue
            rows.append({"job": name, "block": "A_chemokine", "chain_a": rec,
                         "chain_b": lig, "label": lab.loc[rec, lig],
                         "action": action.loc[rec, lig], "size": n_tok})

    for pairs, cls in ((DIMER_POSITIVE, "binder"), (DIMER_NEGATIVE, "negative"),
                       (DIMER_TEST, "test")):
        for a, b in pairs:
            name = f"{a}__{b}"
            try:
                n_tok = write_pair(name, BUNDLES[a], BUNDLES[b])
            except Exception as exc:                      # noqa: BLE001
                failed.append((name, str(exc)))
                continue
            rows.append({"job": name, "block": "B_bundle", "chain_a": a, "chain_b": b,
                         "label": cls, "action": "", "size": n_tok})

    # A panel with holes in it is not the panel the labels describe, and an
    # empty `rows` reaches the checks below as an AttributeError on a column
    # that does not exist, which reads as a pandas problem and not as a build
    # that produced nothing.
    if failed:
        for name, exc in failed[:10]:
            log.error("%s: %s", name, exc)
        raise SystemExit(f"{len(failed)} of {len(failed) + len(rows)} inputs failed to "
                         f"build; refusing to write a partial panel")
    man = pd.DataFrame(rows)

    # Checks that would otherwise fail silently and be read as biology.
    assert man.job.is_unique, "job names collide; one input would overwrite another"
    a = man[man.block == "A_chemokine"]
    assert len(a) == len(R) * len(L), f"{len(a)} cells for a {len(R)}x{len(L)} matrix"
    per_receptor = a[a.label == "binder"].groupby("chain_a").size()
    orphaned = sorted(set(R) - set(per_receptor.index))
    assert not orphaned, (f"{orphaned} have no recorded binder in the panel, so their row "
                          f"contributes only negatives and cannot be ranked")
    per_ligand = a[a.label == "binder"].groupby("chain_b").size()
    orphaned = sorted(set(L) - set(per_ligand.index))
    assert not orphaned, f"{orphaned} bind nothing in the panel"

    # Smallest first, as the screen folds, so a partial run is still a usable
    # block rather than a biased slice of one.
    man = man.sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    dest = ROOT / "work" / "protein_counterscreen_manifest.csv"
    man.to_csv(dest, index=False)

    log.info("wrote %d inputs to %s", len(man), OUT)
    log.info("by block and label:\n%s",
             man.groupby(["block", "label"]).size().to_string())
    log.info("complex size: median %d, max %d tokens",
             man["size"].median(), man["size"].max())
    log.info("binders per receptor:\n%s", per_receptor.to_string())
    log.info("manifest -> %s", dest)


if __name__ == "__main__":
    main()
