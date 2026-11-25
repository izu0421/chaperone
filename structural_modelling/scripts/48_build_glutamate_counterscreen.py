"""The matched control GPR26 never got.

Arm B of the noradrenaline counter-screen says its own purpose was "the question
GPR26 raises" -- whether a receptor that takes three unrelated molecules has a
permissive pocket. It then tested GPR85 and GPR153 and never tested GPR26.

Two arms, mirroring that design exactly:

  arm A   glutamate against the 16-receptor panel whose cognate ligand is known,
          plus GPR26 and GPR150. None of the 16 binds glutamate, so all 16 are
          negatives and the question is whether GPR26 outranks them. This is
          the test that refuted both noradrenaline hits: there, the ONE receptor
          that genuinely binds noradrenaline ranked third of eighteen, behind
          two known non-binders, and the two orphan hits ranked sixth and
          seventh behind four.

  arm B   GPR26 and GPR150 against all 17 ligands. GPR85 scores >= 0.85 with
          twelve of seventeen unrelated molecules, which is a permissive pocket
          measured directly rather than inferred from the ligand. If GPR26 does
          the same, its three hits are a property of the pocket.

Glutamate has no entry in the panel because no class A receptor in it binds
glutamate -- the glutamate receptors are class C, with an extracellular
venus-flytrap domain. That absence is itself worth stating next to the result.

Usage:
    micromamba run -n structmod python -I scripts/48_build_glutamate_counterscreen.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("glu")
ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "bench" / "specificity_panel.csv"
OUT = ROOT / "work" / "noradrenaline_inputs"     # same arm, same output tree
MAN = ROOT / "work" / "glutamate_manifest.csv"
SEEDS = [1]

_s = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_s); _s.loader.exec_module(b7)

GLUTAMATE = {"name": "glutamate", "smiles": "C(CC(=O)O)[C@@H](C(=O)O)N",
             "inchikey": "WHUUTDBJXJRKMK-VKHMYHEASA-N", "family": "amino acid"}
TEST = {"GPR26": "Q8NDV2", "GPR150": "Q8NGU9"}


def ligand_block(cid: str, smiles: str) -> dict:
    return {"ligand": {"id": cid, "smiles": smiles}}


def write(name: str, acc: str, smiles: str, role: str, rows: list) -> None:
    if (OUT / f"{name}.json").exists():
        return
    try:
        _, q, hits = b7.read_omic(acc)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("  %s: %s", name, str(exc)[:60]); return
    msa = b7.a3m([("query", q)] + [(k, hits[k]) for k in sorted(hits)])
    (OUT / f"{name}.json").write_text(json.dumps(
        {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
         "sequences": [{"protein": {"id": "A", "sequence": q, "pairedMsa": msa,
                                    "unpairedMsa": msa, "templates": []}},
                       ligand_block("B", smiles)]}))
    rows.append({"job": name, "arm": "glutamate_counterscreen", "role": role,
                 "size": len(q)})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL)
    rows = []

    log.info("arm A -- glutamate against every receptor whose own ligand is known")
    for r in panel.itertuples():
        write(f"{r.receptor_gene}__glutamate", r.receptor_uniprot, GLUTAMATE["smiles"],
              "negative_control", rows)
    for gene, acc in TEST.items():
        write(f"{gene}__glutamate", acc, GLUTAMATE["smiles"], "test", rows)
    log.info("  %d jobs", len(rows))

    n0 = len(rows)
    log.info("\narm B -- GPR26 and GPR150 against all 17 ligands, to measure the pocket")
    ligs = [(r.ligand_name.replace(" ", "-"), r.ligand_smiles) for r in panel.itertuples()]
    ligs.append((GLUTAMATE["name"], GLUTAMATE["smiles"]))
    for gene, acc in TEST.items():
        for lname, smi in ligs:
            write(f"{gene}__{lname}", acc, smi, "test", rows)
    log.info("  %d jobs", len(rows) - n0)

    if not rows:
        log.info("nothing new to build"); return
    old = pd.read_csv(MAN).to_dict("records") if MAN.exists() else []
    man = pd.DataFrame({r["job"]: r for r in (old + rows)}.values())
    man = man.sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    tmp = MAN.with_suffix(".csv.tmp"); man.to_csv(tmp, index=False); os.replace(tmp, MAN)
    log.info("\n%d inputs, median %d tokens", len(man), int(man["size"].median()))


if __name__ == "__main__":
    main()
