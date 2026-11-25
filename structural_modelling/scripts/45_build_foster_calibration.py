"""The calibration this project has never had: peptide-GPCR pairs with known answers.

Every control built so far is a proxy. Chemokine receptor-ligand pairs test
whether a pose is an artefact; permuted ectodomains test whether a margin is
specific; soluble decoys test whether a cut face is real. None of them answers
the question the screen is actually asking, which is whether this pipeline picks
the right ligand for an orphan receptor.

Foster et al., Cell 2019 (PMC6838683), screened a 218-peptide library against a
panel of orphan GPCRs and tabulated, with sequences, the peptide-receptor
pairings that were already established. Seventeen of their twenty-one priority
orphans are in this screen's own 78, so the overlap is direct rather than
analogous.

POSITIVES  the known pairings, with the receptor family expanded to its member
           genes from the Guide to Pharmacology's own target table rather than
           from memory. A peptide active at "opioid receptors" is a positive
           against OPRD1, OPRK1, OPRL1 and OPRM1.
PEPTIDES   the sequences Foster synthesised, 3-50 aa. That is the mature
           molecule by construction, so this set carries none of the precursor
           problem that invalidated 1,404 folds in the main screen.
NEGATIVES  each peptide against receptors that other peptides are known to
           activate. A permuted pair within the same panel is matched on
           everything except the answer.

Usage:
    micromamba run -n hsa python -I scripts/45_build_foster_calibration.py --labels
    micromamba run -n structmod python -I scripts/45_build_foster_calibration.py --build
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("foster")
ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench" / "foster19"
LABELS = BENCH / "foster_pairs.csv"
OUT = ROOT / "work" / "foster_inputs"
MAN = ROOT / "work" / "foster_manifest.csv"
SEEDS = [1]
NEG_PER_POS = 5

# Targets the Guide to Pharmacology's family names do not match verbatim. Each is
# the IUPHAR name for a receptor or family that Foster spells differently; written
# out so a reader can check them one by one.
MANUAL = {
    "AT1": ["AGTR1"],
    "UT receptor": ["UTS2R"],
    "Kappa receptor": ["OPRK1"],
    "Opioid receptor (NOP)": ["OPRL1"],
    "Neurokinin receptors": ["TACR1", "TACR2", "TACR3"],
    "Oxytocin and vasopressin receptors": ["OXTR", "AVPR1A", "AVPR1B", "AVPR2"],
    "Neuropeptide FF receptors": ["NPFFR1", "NPFFR2"],
    "PrRP receptor, neuropeptide FF receptors": ["PRLHR", "NPFFR1", "NPFFR2"],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def labels() -> None:
    """Resolve Foster's targets to genes and write the labelled pair list."""
    g = pd.read_csv(BENCH / "gtopdb_targets.csv", skiprows=1, low_memory=False)
    g = g[g.Type.astype(str).str.contains("gpcr", case=False, na=False)]
    fam = {norm(k): v for k, v in
           g.groupby("Family name")["HGNC symbol"]
            .apply(lambda s: sorted(set(s.dropna()))).items()}
    symbols = set(g["HGNC symbol"].dropna())

    t5 = pd.read_excel(BENCH / "mmc6.xlsx", sheet_name="Known peptide-receptor pairs")
    t5.columns = [str(c).strip() for c in t5.columns]

    def resolve(tgt):
        if tgt in MANUAL:
            return MANUAL[tgt], "manual"
        n = norm(tgt)
        if n in fam:
            return fam[n], "gtopdb family"
        n2 = norm(re.sub(r"\(.*?\)", "", str(tgt)))
        if n2 in fam:
            return fam[n2], "gtopdb family"
        if str(tgt).strip().upper() in symbols:
            return [str(tgt).strip().upper()], "gene symbol"
        cand = [v for k, v in fam.items() if n in k or k in n]
        return (cand[0], "gtopdb substring") if len(cand) == 1 else (None, "unresolved")

    rows, miss = [], []
    for r in t5.itertuples():
        seq = str(getattr(r, "Sequence", "") or "").strip()
        tgt = getattr(r, "Target", None)
        if not seq or not tgt or seq.lower() == "nan":
            continue
        genes, how = resolve(tgt)
        if not genes:
            miss.append(tgt); continue
        for gene in genes:
            rows.append({"peptide": getattr(r, "_5", None) or getattr(r, "Peptide Name", ""),
                         "pep_ref": getattr(r, "_1", None),
                         "pep_acc": getattr(r, "Protein Acccession", ""),
                         "sequence": seq, "receptor": gene, "label": 1,
                         "target_text": tgt, "resolved_by": how})
    d = pd.DataFrame(rows).drop_duplicates(["sequence", "receptor"])
    log.info("%d positive peptide-receptor pairs from %d known pairings",
             len(d), t5.Target.notna().sum())
    if miss:
        log.warning("unresolved targets: %s", sorted(set(miss)))

    # negatives: each peptide against receptors other peptides are known to hit
    import numpy as np
    rng = np.random.default_rng(0)
    all_rec = sorted(d.receptor.unique())
    neg = []
    for seq, grp in d.groupby("sequence"):
        own = set(grp.receptor)
        pool = [r for r in all_rec if r not in own]
        take = rng.choice(pool, size=min(NEG_PER_POS * len(grp), len(pool)), replace=False)
        p0 = grp.iloc[0]
        for r in take:
            neg.append({"peptide": p0.peptide, "pep_ref": p0.pep_ref, "pep_acc": p0.pep_acc,
                        "sequence": seq, "receptor": r, "label": 0,
                        "target_text": "", "resolved_by": "permuted within the panel"})
    out = pd.concat([d, pd.DataFrame(neg)], ignore_index=True)
    out.to_csv(LABELS, index=False)
    log.info("%d pairs total: %d positive, %d negative over %d peptides and %d receptors",
             len(out), int((out.label == 1).sum()), int((out.label == 0).sum()),
             out.sequence.nunique(), out.receptor.nunique())
    log.info("wrote %s", LABELS)

    # the detection floor, before anything is folded
    from math import sqrt
    npos, nneg = int((out.label == 1).sum()), int((out.label == 0).sum())
    a = 0.75
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg))
    log.info("detection floor: with %d positives and %d negatives, the standard error on "
             "AUROC near 0.75 is %.3f, so anything under about %.2f is not distinguishable "
             "from chance", npos, nneg, se, 0.5 + 1.96 * se)


def build() -> None:
    import importlib.util as iu
    _s = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
    b7 = iu.module_from_spec(_s); _s.loader.exec_module(b7)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(LABELS)
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))
    extra = json.loads((BENCH / "receptor_accessions.json").read_text()) \
        if (BENCH / "receptor_accessions.json").exists() else {}
    acc.update(extra)

    rows, missing = [], set()
    for i, r in enumerate(d.itertuples(), 1):
        if r.receptor not in acc:
            missing.add(r.receptor); continue
        name = f"F_{r.receptor}__p{r.pep_ref}"
        if (OUT / f"{name}.json").exists():
            continue
        try:
            mask, qA, hits = b7.read_omic(acc[r.receptor])
        except Exception as exc:                                # noqa: BLE001
            log.warning("  %s: %s", r.receptor, str(exc)[:50]); continue
        msaA = b7.a3m([("query", qA)] + [(k, hits[k]) for k in sorted(hits)])
        pep = str(r.sequence).strip().upper()
        # a synthesised peptide has no alignment of its own and needs none
        (OUT / f"{name}.json").write_text(json.dumps(
            {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
             "sequences": [
                 {"protein": {"id": "A", "sequence": qA, "pairedMsa": msaA,
                              "unpairedMsa": msaA, "templates": []}},
                 {"protein": {"id": "B", "sequence": pep, "pairedMsa": f">query\n{pep}\n",
                              "unpairedMsa": f">query\n{pep}\n", "templates": []}}]}))
        rows.append({"job": name, "arm": "F_calibration", "receptor": r.receptor,
                     "peptide": r.peptide, "pep_ref": r.pep_ref, "label": r.label,
                     "size": len(qA) + len(pep)})
    if missing:
        log.warning("%d receptors have no accession and were skipped: %s",
                    len(missing), sorted(missing)[:12])
    if rows:
        man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
        man["fold_order"] = man.index + 1
        tmp = MAN.with_suffix(".csv.tmp"); man.to_csv(tmp, index=False); os.replace(tmp, MAN)
        log.info("%d inputs, median %d tokens, %d positive / %d negative",
                 len(man), int(man["size"].median()),
                 int((man.label == 1).sum()), int((man.label == 0).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.labels: labels()
    if a.build: build()
