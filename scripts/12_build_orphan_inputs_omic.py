"""AF3 inputs for the 5,000 orphan pairs, using the omicMSAs instead of running
AlphaFast's own search.

The omicMSAs already cover 4,915 of the 5,000 pairs -- all 75 receptors and
1,007 of 1,024 partners -- so the MSA stage is free. That also makes these folds
use the SAME alignment source as arm B, the best-validated arm, and arm A showed
the choice does not move discrimination anyway (pae_min_AB 0.840 vs 0.839, and
B - A at 1:1000 was -0.040 with a 95% CI of [-0.304, +0.216]).

The pairing and masking logic is IMPORTED from 07_build_af3_inputs.py, not
reimplemented: an omicMSA row is one draft genome, chains are paired by genome
accession, and the mask has to be GAPPED rather than deleted to keep the query
full length. 07 carries a validate() that reproduces the authors' own paired
MSAs under their masking rule, and a reimplementation here would be a second,
unvalidated copy of exactly the code where a silent mistake looks like biology.

The 146 metabolite partners become ligand chains (CCD, or SMILES where no
component exists) and need no MSA. Pairs whose protein partner has no omicMSA
are written WITHOUT one and flagged in the manifest, not silently dropped --
`msa_source` records which is which so the two can be read apart afterwards.
"""
import importlib.util as iu
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("orphan-omic")

ROOT = Path(__file__).resolve().parent.parent
SHEET = Path("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
OUT = ROOT / "work" / "orphan_inputs_omic"
ALNS = ROOT / "data/rf2ppi_raw/protein_omicMSAs/alns"

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SHEET)
    have = {p.stem for p in ALNS.glob("*.a3m")}
    log.info("%d pairs; %d omicMSA accessions on disk", len(d), len(have))

    rows, n_paired, n_single, n_lig, n_fail = [], 0, 0, 0, 0
    for r in d.itertuples():
        name = f"{r.receptor_gene}__{r.partner_gene}"
        ra = r.receptor_uniprot
        try:
            if r.partner_kind == "metabolite":
                # ligand: chain A keeps its own omicMSA, the ligand needs none
                mA, qA, hA = b7.read_omic(ra)
                gA = {i for i, c in enumerate(mA) if c == "*"}
                rowsA = [(acc, b7.gap_masked(hA[acc], gA)) for acc in sorted(hA)]
                lig = ({"ccdCodes": [r.ligand_ccd]} if isinstance(r.ligand_ccd, str) and r.ligand_ccd
                       else {"smiles": r.ligand_smiles})
                js = {"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
                      "sequences": [
                          {"protein": {"id": "A", "sequence": qA,
                                       "pairedMsa": b7.a3m([("query", qA)] + rowsA),
                                       "unpairedMsa": b7.a3m([("query", qA)] + rowsA),
                                       "templates": []}},
                          {"ligand": {"id": "B", **lig}}]}
                src, size = "omic_receptor_only", len(qA)
                n_lig += 1
            elif r.partner_uniprot in have and ra in have:
                qA, qB, rowsA, rowsB = b7.build_pair(ra, r.partner_uniprot)
                js = b7.af3_json(name, qA, qB, rowsA, rowsB)
                src, size = "omic_paired", len(qA) + len(qB)
                n_paired += 1
            else:
                # no omicMSA for the partner: fold single-sequence and FLAG it
                js = {"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
                      "sequences": [
                          {"protein": {"id": "A", "sequence": r.receptor_seq, "templates": []}},
                          {"protein": {"id": "B", "sequence": r.partner_seq, "templates": []}}]}
                src = "none_single_sequence"
                size = len(r.receptor_seq) + len(r.partner_seq)
                n_single += 1
        except Exception as e:
            n_fail += 1
            log.warning("%s: %s %s", name, type(e).__name__, str(e)[:60])
            continue

        (OUT / f"{name}.json").write_text(json.dumps(js))
        rows.append({"job": name, "rank": int(r.rank), "size": size, "msa_source": src,
                     "receptor": r.receptor_gene, "partner": r.partner_gene,
                     "kind": r.partner_kind, "p_twoway": r.p_twoway})

    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    man.to_csv(ROOT / "work" / "orphan_manifest.csv", index=False)
    log.info("wrote %d inputs: %d omic-paired, %d ligand, %d single-sequence (no partner MSA), "
             "%d failed", len(rows), n_paired, n_lig, n_single, n_fail)
    log.info("msa_source:\n%s", man.msa_source.value_counts().to_string())


main()
