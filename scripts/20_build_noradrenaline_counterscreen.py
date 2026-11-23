"""Counter-screen the one candidate the calibrated gate left standing.

Of 146 orphan metabolite folds, exactly two receptors cleared a bar calibrated to
keep 14 of 16 known cognate ligands while admitting 10% of known negatives:
GPR85 and GPR153, both with noradrenaline. Two receptors reaching threshold on
the same molecule is either a real shared ligand or a property of the molecule,
and the orphan screen alone cannot tell those apart -- it contains no receptor
whose true ligand is known.

So this builds the two-way control that can, on the same omicMSA pipeline and
the same scoring code:

  arm A  noradrenaline against all 18 receptors (the 16 whose cognate ligand is
         known, plus GPR85 and GPR153). ADRB2 is the built-in POSITIVE control:
         noradrenaline is a genuine adrenoceptor ligand, so a screen that cannot
         score it highly at ADRB2 is not measuring anything. The other 15 are
         negative controls with known, different ligands. The three aminergic
         ones (HTR1A, DRD2, HRH1, CHRM1) are the hard case, since noradrenaline
         has real if weaker affinity across that family -- the 16-ligand
         benchmark already measured within-family AUROC at 0.830 against 0.971
         across families, so a within-family hit is expected and is not a
         failure.

  arm B  GPR85 and GPR153 against all 17 ligands. This is the question GPR26
         answered badly: GPR26 cleared the same calibrated bar with glutamate,
         GABA and acetylcholine, which at a 10%-negative threshold is a
         permissive pocket and not three ligands. If GPR85 and GPR153 clear it
         only with noradrenaline, that is selectivity; if they clear it with
         several, they are GPR26.

Neither arm is informative alone. Arm A without arm B cannot separate "GPR85
binds noradrenaline" from "GPR85 binds anything", and arm B without arm A cannot
separate "noradrenaline is selective" from "noradrenaline scores high
everywhere".

Usage:
    micromamba run -n structmod python -I scripts/20_build_noradrenaline_counterscreen.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("nor-counter")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "bench" / "specificity_panel.csv"
SHEET = Path("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
ALNS = ROOT / "data/rf2ppi_raw/protein_omicMSAs/alns"
OUT = ROOT / "work" / "noradrenaline_inputs"

_s = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_s)
_s.loader.exec_module(b7)
sys.path.insert(0, str(ROOT / "scripts"))
import check_af3_inputs as chk  # noqa: E402

# the molecule under test. SMILES and InChIKey resolved from PubChem rather than
# carried over from the screen, which held only the CCD code "LNR"
NORADRENALINE = {
    "name": "noradrenaline",
    "smiles": "C1=CC(=C(C=C1[C@H](CN)O)O)O",
    "inchikey": "SFLSHLFXELFNJZ-QMMMGPOBSA-N",
    "formula": "C8H11NO3",
    "family": "aminergic",
}
# the two receptors the calibrated gate left standing, and their own ligand is
# what this is testing, so neither has a "cognate" entry in the panel
EXTRA_RECEPTORS = {"GPR85": "P60893", "GPR153": "Q6NV75"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL)
    have = {p.stem for p in ALNS.glob("*.a3m")}

    receptors = [(r.receptor_gene, r.receptor_uniprot, r.ligand_family)
                 for r in panel.itertuples()]
    receptors += [(g, a, "orphan") for g, a in EXTRA_RECEPTORS.items()]
    missing = [a for _, a, _ in receptors if a not in have]
    assert not missing, f"no omicMSA for {missing}"

    ligands = [(r.ligand_name, r.ligand_smiles, r.ligand_family, r.receptor_gene)
               for r in panel.itertuples()]
    ligands += [(NORADRENALINE["name"], NORADRENALINE["smiles"],
                 NORADRENALINE["family"], None)]

    # one MSA read per receptor, reused across every ligand it is folded with
    msa = {}
    for gene, acc, _ in receptors:
        mask, query, hits = b7.read_omic(acc)
        keep = {i for i, ch in enumerate(mask) if ch == "*"}
        rows = [(a, b7.gap_masked(hits[a], keep)) for a in sorted(hits)]
        msa[gene] = (query, b7.a3m([("query", query)] + rows))
        log.info("%s (%s): %d aa, %d aligned sequences", gene, acc, len(query), len(rows))

    # Only the cells the two arms need, not the full 18x17 product: the 16x16
    # benchmark is already folded and scored, and re-folding it would spend four
    # hours to reproduce numbers that are on disk.
    want = set()
    for gene, _, _ in receptors:
        want.add((gene, NORADRENALINE["name"]))                    # arm A
    for gene in EXTRA_RECEPTORS:
        for lname, _, _, _ in ligands:
            want.add((gene, lname))                                # arm B

    rows = []
    for gene, lname in sorted(want):
        lig = next(l for l in ligands if l[0] == lname)
        query, aln = msa[gene]
        name = f"{gene}__{lname.replace(' ', '-')}"
        js = {"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
              "sequences": [
                  {"protein": {"id": "A", "sequence": query, "pairedMsa": aln,
                               "unpairedMsa": aln, "templates": []}},
                  {"ligand": {"id": "B", "smiles": lig[1]}}]}
        probs = chk.problems(js)
        assert not probs, f"{name}: {probs}"
        (OUT / f"{name}.json").write_text(json.dumps(js))
        rfam = next(f for g, _, f in receptors if g == gene)
        rows.append({
            "job": name, "receptor": gene, "ligand": lname,
            "arm": "A_noradrenaline_vs_receptors" if lname == NORADRENALINE["name"]
                   else "B_orphan_receptor_vs_ligands",
            "receptor_family": rfam, "ligand_family": lig[2],
            "same_family": rfam == lig[2],
            # what each cell is FOR, so the read-out cannot be assembled after
            # the fact to suit the answer
            "role": ("positive_control" if (gene == "ADRB2" and lname == NORADRENALINE["name"])
                     else "test" if gene in EXTRA_RECEPTORS
                     else "negative_control"),
            "cognate_ligand_of_receptor": lig[3] == gene,
            "size": len(query)})

    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    assert man.job.is_unique, "job names collide"
    assert (man.role == "positive_control").sum() == 1, "expected exactly one positive control"
    man["fold_order"] = man.index + 1
    dest = ROOT / "work" / "noradrenaline_manifest.csv"
    man.to_csv(dest, index=False)
    log.info("\nwrote %d inputs to %s", len(man), OUT)
    log.info("\nby arm:\n%s", man.arm.value_counts().to_string())
    log.info("\nby role:\n%s", man.role.value_counts().to_string())
    log.info("\nmedian %d tokens, range %d-%d", man["size"].median(),
             man["size"].min(), man["size"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
