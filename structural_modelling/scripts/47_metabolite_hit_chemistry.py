"""Two tests the metabolite gate cannot do, on the six hits it produced.

The gate scores a fold. Neither of these does.

TEST 1, THE LIGAND-MATCHED CONTROL. Fix the ligand and vary the receptor. The
counter-screen already folded noradrenaline against 18 receptors, one of which
binds it. If interface pTM measured whether THIS receptor takes THIS ligand, the
adrenoceptor would rank first. It ranks third, behind two receptors known not to
bind noradrenaline, and the two orphan hits rank sixth and seventh behind four
known non-binders. The comparison the screen reported -- "inside the cognate
distribution" -- is across DIFFERENT ligand-receptor pairs, where ligand identity
varies and carries the signal. Within one ligand the discrimination is gone.

TEST 2, THE CHARGE GEOMETRY. Every one of these ligands is positively charged at
physiological pH, and every aminergic GPCR pairs that charge with an aspartate in
helix 3, or cages it in aromatics. Measured ATOM TO ATOM, not residue to residue:
the four cognate controls put the ligand's cation 2.6-4.2 A from an acidic oxygen.

A FIRST VERSION OF THIS ANALYSIS USED A 4.5 A RESIDUE CONTACT AND REPORTED "ASP86
ANCHORS ALL THREE GPR26 HITS". That was wrong. Some atom of ASP86 is near some
atom of acetylcholine, but the carboxylate points at the ester oxygen -- acceptor
to acceptor -- and the quaternary ammonium is 6.4 A away with nothing to pair it.
A residue is in contact; a charge is not. Measure the atoms that carry the charge.

Usage:
    micromamba run -n structmod python -I scripts/47_metabolite_hit_chemistry.py
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hit-chem")
warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent

ACID = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BASIC = {("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"), ("LYS", "NZ"),
         ("HIS", "ND1"), ("HIS", "NE2")}
# glutamate and GABA are ZWITTERIONS: the amine needs an anion and the
# carboxylate needs a cation. Scoring only the amine passed GPR150 + GABA on a
# cation-pi to PHE350 while its carboxylate had nothing within 13 A.
ZWITTERION = {"glutamate", "GABA"}
RING = {"PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]}
SALT_BRIDGE, CATION_PI = 4.0, 4.5

HITS = [("results/orphan_out/GPR26__SLC17A7", "GPR26", "glutamate", "hit"),
        ("results/orphan_out/GPR26__SLC6A1", "GPR26", "GABA", "hit"),
        ("results/orphan_out/GPR26__SLC10A4", "GPR26", "acetylcholine", "hit"),
        ("results/orphan_out/GPR150__GAD2", "GPR150", "GABA", "hit"),
        ("results/noradrenaline_out/GPR85__noradrenaline", "GPR85", "noradrenaline", "hit"),
        ("results/noradrenaline_out/GPR153__noradrenaline", "GPR153", "noradrenaline", "hit"),
        ("results/specificity_out/CHRM1__acetylcholine", "CHRM1", "acetylcholine", "cognate"),
        ("results/noradrenaline_out/ADRB2__noradrenaline", "ADRB2", "noradrenaline", "cognate"),
        ("results/specificity_out/HTR1A__serotonin", "HTR1A", "serotonin", "cognate"),
        ("results/specificity_out/DRD2__dopamine", "DRD2", "dopamine", "cognate")]


def geometry(cif: Path):
    ch = list(next(iter(MMCIFParser(QUIET=True).get_structure("x", str(cif)))))
    prot = max(ch, key=lambda c: len([r for r in c if r.id[0] == " "]))
    lig = [a for c in ch if c is not prot for a in c.get_atoms()]
    cat = [a for a in lig if a.element == "N"]          # the protonatable nitrogen
    if not cat:
        return None
    # the ligand's own carboxylate oxygens, for the zwitterion check
    anion = [a for a in lig if a.element == "O" and a.get_id() not in ("O", "OXT")] or \
            [a for a in lig if a.element == "O"]
    out = {"acid": (1e9, None), "ring": (1e9, None), "base": (1e9, None)}
    for a in prot.get_atoms():
        r = a.get_parent()
        nm = (r.get_resname(), a.get_id())
        if nm in ACID:
            for c in cat:
                d = float(np.linalg.norm(a.coord - c.coord))
                if d < out["acid"][0]: out["acid"] = (d, f"{r.get_resname()}{r.id[1]}")
        if a.get_id() in RING.get(r.get_resname(), []):
            for c in cat:
                d = float(np.linalg.norm(a.coord - c.coord))
                if d < out["ring"][0]: out["ring"] = (d, f"{r.get_resname()}{r.id[1]}")
        if nm in BASIC:
            for o in anion:
                d = float(np.linalg.norm(a.coord - o.coord))
                if d < out["base"][0]: out["base"] = (d, f"{r.get_resname()}{r.id[1]}")
    return out


def main() -> None:
    rows = []
    for path, rec, mol, kind in HITS:
        d = ROOT / path
        cifs = sorted(d.glob("*_model.cif")) or sorted(d.glob("seed-*/*_model.cif"))
        if not cifs:
            log.warning("no model for %s + %s", rec, mol); continue
        g = geometry(cifs[0])
        if not g:
            continue
        rows.append({"receptor": rec, "ligand": mol, "kind": kind,
                     "acid_A": round(g["acid"][0], 2), "acid_res": g["acid"][1],
                     "ring_A": round(g["ring"][0], 2), "ring_res": g["ring"][1],
                     "base_A": round(g["base"][0], 2), "base_res": g["base"][1],
                     "cation_paired": bool(g["acid"][0] <= SALT_BRIDGE
                                           or g["ring"][0] <= CATION_PI),
                     "anion_paired": bool(g["base"][0] <= SALT_BRIDGE),
                     "has_counter_charge": bool(
                         (g["acid"][0] <= SALT_BRIDGE or g["ring"][0] <= CATION_PI)
                         and (mol not in ZWITTERION or g["base"][0] <= SALT_BRIDGE))})
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results" / "metabolite_hit_chemistry.csv", index=False)
    log.info("DISTANCE FROM THE LIGAND'S CATION TO ITS NEAREST COUNTER-CHARGE\n")
    for kind in ("cognate", "hit"):
        sub = t[t.kind.eq(kind)]
        log.info("%s:", "KNOWN COGNATE PAIRS" if kind == "cognate" else "THE SIX HITS")
        for r in sub.sort_values("acid_A").itertuples():
            flag = "" if r.has_counter_charge else "   <- nothing pairs the charge"
            zw = (f"   carboxylate to base {r.base_A:5.2f} A ({r.base_res or '-'})"
                  if r.ligand in ZWITTERION else "")
            log.info("  %-8s + %-14s cation: acid %6.2f A (%s) ring %5.2f A (%s)%s%s",
                     r.receptor, r.ligand, r.acid_A, r.acid_res or "-",
                     r.ring_A, r.ring_res or "-", zw, flag)
        log.info("")
    ok = t[t.kind.eq("hit") & t.has_counter_charge]
    log.info("%d of %d hits have a counter-charge in a credible geometry: %s",
             len(ok), int(t.kind.eq("hit").sum()),
             ", ".join(f"{r.receptor}+{r.ligand}" for r in ok.itertuples()) or "none")
    log.info("every cognate control does: %s",
             bool(t[t.kind.eq("cognate")].has_counter_charge.all()))

    # test 1, from the counter-screen that already exists
    nf = ROOT / "results" / "noradrenaline_scores.csv"
    if nf.exists():
        n = pd.read_csv(nf)
        n["receptor"] = n.pair.str.split("__").str[0]
        n["ligand"] = n.pair.str.split("__").str[1]
        na = n[n.ligand.eq("noradrenaline")].sort_values("iptm", ascending=False).reset_index(drop=True)
        pos = na.index[na.role.eq("positive_control")]
        log.info("\nLIGAND-MATCHED CONTROL: noradrenaline against %d receptors", len(na))
        log.info("  the one receptor that binds it ranks %d", int(pos[0]) + 1 if len(pos) else -1)
        log.info("  receptors scoring above it, all known non-binders: %d",
                 int(pos[0]) if len(pos) else -1)
        for rec in ("GPR153", "GPR85"):
            if (na.receptor == rec).any():
                k = int(na.index[na.receptor.eq(rec)][0]) + 1
                log.info("  %s ranks %d of %d, below %d known non-binders", rec, k, len(na),
                         int((na.iloc[:k - 1].role == "negative_control").sum()))
        json.dump({"positive_rank": int(pos[0]) + 1 if len(pos) else None,
                   "n_receptors": int(len(na))},
                  open(ROOT / "results" / "noradrenaline_rank.json", "w"), indent=1)


if __name__ == "__main__":
    main()
