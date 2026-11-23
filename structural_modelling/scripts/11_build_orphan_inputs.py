"""AF3 inputs for the 5,000 orphan receptor-partner pairs.

Separate from 07_build_af3_inputs.py because that one is protein-only, and 146
of these pairs have a METABOLITE partner that AF3 takes as a ligand chain, not a
protein chain. Folding a metabolite as a protein would silently produce nonsense
for exactly the pairs CellPhoneDB routes through a synthase (PGD2_byPTGDS and
friends), so the two paths are explicit here and asserted at the end.

Ligands go in by CCD code where one exists and by SMILES where it does not. Both
were resolved against the RCSB chemical component dictionary by NAME and FORMULA
in bench/build_af3_sheet.py, not recalled -- that file records why (cortisol's
first identifier was a different molecule with an identical formula).

Ordering: the sheet is ranked by the two-way null's p, but the FOLD order is by
size, smallest first. At bucket^2.79, the 87 largest pairs carry 27% of the
total cost and the 85% smallest carry 28%, so size order means most of the set
is answered early and the expensive tail can be abandoned without losing
coverage. The rank is preserved in the manifest, not in the queue order.
"""
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("orphan-inputs")

ROOT = Path(__file__).resolve().parent.parent
SHEET = Path("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
OUT = ROOT / "work" / "orphan_inputs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SHEET)
    log.info("%d rows: %s", len(d), d.partner_kind.value_counts().to_dict())

    n_prot = n_ccd = n_smi = 0
    rows = []
    for r in d.itertuples():
        seqs = [{"protein": {"id": "A", "sequence": r.receptor_seq}}]
        if r.partner_kind == "protein":
            if not isinstance(r.partner_seq, str) or not r.partner_seq:
                continue
            seqs.append({"protein": {"id": "B", "sequence": r.partner_seq}})
            n_prot += 1
            size = len(r.receptor_seq) + len(r.partner_seq)
        else:
            ccd = r.ligand_ccd if isinstance(r.ligand_ccd, str) and r.ligand_ccd else None
            smi = r.ligand_smiles if isinstance(r.ligand_smiles, str) and r.ligand_smiles else None
            if ccd:
                seqs.append({"ligand": {"id": "B", "ccdCodes": [ccd]}}); n_ccd += 1
            elif smi:
                seqs.append({"ligand": {"id": "B", "smiles": smi}}); n_smi += 1
            else:
                continue
            size = len(r.receptor_seq)      # a ligand adds few tokens

        name = f"{r.receptor_gene}__{r.partner_gene}"
        (OUT / f"{name}.json").write_text(json.dumps({
            "name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
            "sequences": seqs,
        }))
        rows.append({"job": name, "rank": int(r.rank), "size": size,
                     "receptor": r.receptor_gene, "partner": r.partner_gene,
                     "kind": r.partner_kind, "p_twoway": r.p_twoway})

    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    man.to_csv(ROOT / "work" / "orphan_manifest.csv", index=False)

    log.info("wrote %d inputs: %d protein, %d ligand-by-CCD, %d ligand-by-SMILES",
             len(rows), n_prot, n_ccd, n_smi)
    assert n_prot + n_ccd + n_smi == len(rows)
    # the metabolite rows must NOT have become protein chains
    assert n_ccd + n_smi == int((d.partner_kind == "metabolite").sum()), \
        "a metabolite row did not become a ligand chain"
    log.info("manifest: fold order is by size, %d..%d tokens", man["size"].min(), man["size"].max())
main()
