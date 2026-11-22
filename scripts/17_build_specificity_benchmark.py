"""AF3 inputs for the small-molecule specificity matrix: every panel ligand
against every panel receptor.

The question this answers is whether co-folding confidence knows WHICH
receptor a small molecule binds, or only that a pocket is there. The orphan
screen suggested the latter and four published groups agree, but neither is
our own controlled test. Folding the full matrix makes it one: if confidence
carries specificity the diagonal outscores the rest, and if it reports the
pocket the matrix is flat.

Built the same way as the metabolite arm of the orphan screen, deliberately.
Chain A is the receptor with its own omicMSA gap-masked, chain B is the
ligand as a SMILES string, and the pairing and masking come from
07_build_af3_inputs.py by import. Using a second, unvalidated copy of that
code here is exactly where a silent mistake would look like a biological
result.

Labels come from ChEMBL, measured per cell rather than assumed:

    cognate    the ligand's own receptor, measured affinity on record
    negative   zero recorded activity -- the usable negatives
    ambiguous  activity recorded but not the cognate pair; 21% of the
               off-diagonal, concentrated in the aminergic block and under
               the promiscuous ligands. Folded and reported separately, NOT
               pooled into the negatives, since burying 51 real interactions
               in the negative class would flatten any true signal.

Usage:
    micromamba run -n structmod python -I scripts/17_build_specificity_benchmark.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("spec-bench")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "bench" / "specificity_panel.csv"
MATRIX = ROOT / "bench" / "chembl_matrix.json"
ALNS = ROOT / "data/rf2ppi_raw/protein_omicMSAs/alns"
OUT = ROOT / "work" / "specificity_inputs"

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL)
    chembl = json.loads(MATRIX.read_text())
    counts = chembl["matrix"]                      # [ligand index][receptor index]
    order = [p[0] for p in chembl["panel"]]
    assert order == list(panel.receptor_gene), "panel and matrix are in different orders"

    have = {p.stem for p in ALNS.glob("*.a3m")}
    missing = set(panel.receptor_uniprot) - have
    assert not missing, f"no omicMSA for {sorted(missing)}"

    # one read and mask per receptor, reused across all 16 ligands
    receptor_msa = {}
    for r in panel.itertuples():
        mask, query, hits = b7.read_omic(r.receptor_uniprot)
        keep = {i for i, ch in enumerate(mask) if ch == "*"}
        rows = [(acc, b7.gap_masked(hits[acc], keep)) for acc in sorted(hits)]
        receptor_msa[r.receptor_gene] = (query, b7.a3m([("query", query)] + rows))
        log.info("%s (%s): %d aa, %d aligned sequences",
                 r.receptor_gene, r.receptor_uniprot, len(query), len(rows))

    rows = []
    for li, lig in enumerate(panel.itertuples()):
        for ri, rec in enumerate(panel.itertuples()):
            name = f"{rec.receptor_gene}__{lig.ligand_name.replace(' ', '-')}"
            query, msa = receptor_msa[rec.receptor_gene]
            js = {"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
                  "sequences": [
                      {"protein": {"id": "A", "sequence": query, "pairedMsa": msa,
                                   "unpairedMsa": msa, "templates": []}},
                      {"ligand": {"id": "B", "smiles": lig.ligand_smiles}}]}
            (OUT / f"{name}.json").write_text(json.dumps(js))
            n = counts[li][ri]
            label = ("cognate" if li == ri else "ambiguous" if n > 0 else "negative")
            rows.append({
                "job": name, "receptor": rec.receptor_gene,
                "receptor_uniprot": rec.receptor_uniprot, "ligand": lig.ligand_name,
                "ligand_chembl_id": lig.ligand_chembl_id,
                "receptor_family": rec.ligand_family, "ligand_family": lig.ligand_family,
                "same_family": rec.ligand_family == lig.ligand_family,
                "n_chembl_activities": n, "label": label,
                "size": len(query)})

    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    dest = ROOT / "work" / "specificity_manifest.csv"
    man.to_csv(dest, index=False)

    log.info("wrote %d inputs to %s", len(man), OUT)
    log.info("labels:\n%s", man.label.value_counts().to_string())
    off = man[man.label != "cognate"]
    log.info("off-diagonal by family: same %d, different %d",
             int(off.same_family.sum()), int((~off.same_family).sum()))
    log.info("clean negatives by family: same %d, different %d",
             int(off[off.label.eq("negative")].same_family.sum()),
             int((~off[off.label.eq("negative")].same_family).sum()))
    log.info("receptor length: median %d, max %d", int(man["size"].median()), int(man["size"].max()))


if __name__ == "__main__":
    main()
