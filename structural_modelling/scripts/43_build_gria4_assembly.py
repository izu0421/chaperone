"""Does GPR83 dock onto the face GRIA4 uses to build itself?

GPR83 + GRIA4 is the only membrane candidate that passed the ectodomain arm, and
it passed well: 3.01 A on the full partner, improving to 2.21 A on GRIA4's
ectodomain alone, against a best permuted negative of 4.96 A.

It was folded as a lone GRIA4 chain. AMPA receptors are dimers of dimers, so a
monomeric GRIA4 ectodomain exposes a surface that does not exist in the
assembled receptor: the amino-terminal-domain dimer interface. This screen has
already been burnt four times by exactly that, and each time the agent flagged
it first -- fibrinogen alpha folded without beta and gamma, GP1BB without GP1BA
and GP9, GRIN2D without GRIN1. All four lost the interface once the missing
subunits were present (2.36 -> 9.62, 3.59 -> 19.85, 2.75 -> 21.59, 2.02 ->
10.91 A).

GRIA4 was never on that list, because nothing flagged it. So fold the pair with
a second copy of the ectodomain present, which is the smallest construct that
restores the dimer interface, and ask whether GPR83's interface survives it.

    S2_GPR83__GRIA4__GRIA4    GPR83 + GRIA4-ECD + GRIA4-ECD
    S2_GRIA4__GRIA4           the two ectodomains alone, so the dimer itself can
                              be checked: if the two copies do not pair in this
                              construct, the test is vacuous and says nothing

One seed, as the rest of the re-folds now run.

Usage:
    micromamba run -n structmod python -I scripts/43_build_gria4_assembly.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("gria4")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ectodomain_inputs"
MAN = ROOT / "work" / "ectodomain_manifest.csv"
SEEDS = [1]

_s37 = iu.spec_from_file_location("b37", ROOT / "scripts" / "37_build_ectodomain_arm.py")
b37 = iu.module_from_spec(_s37)
_s37.loader.exec_module(b37)
b7 = b37.b7


def main() -> None:
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))
    rec, par = "GPR83", "GRIA4"
    seg = b37.uniprot_ecd(acc[par])
    if not seg:
        raise SystemExit("no ectodomain for GRIA4")
    s, e = seg
    log.info("GRIA4 ectodomain %d-%d (%d aa)", s, e, e - s + 1)

    qA, qB, rA, rB = b7.build_pair(acc[rec], acc[par])
    kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
    msaA = b7.a3m([("query", qA)] + kA)
    qBe, msaBe = b37.slice_query(qB, b7.a3m([("query", qB)] + kB), s, e)
    assert len(qBe) == e - s + 1, f"cut gave {len(qBe)}, expected {e - s + 1}"

    def chain(cid, sq, ms):
        return {"protein": {"id": cid, "sequence": sq, "pairedMsa": ms,
                            "unpairedMsa": ms, "templates": []}}

    jobs = {
        # the candidate, with the dimer interface restored
        f"S2_{rec}__{par}__{par}": [chain("A", qA, msaA), chain("B", qBe, msaBe),
                                    chain("C", qBe, msaBe)],
        # the dimer alone: if these two copies do not pair, the test above is
        # vacuous and must not be read as a negative for the candidate
        f"S2_{par}__{par}": [chain("A", qBe, msaBe), chain("B", qBe, msaBe)],
    }
    rows = pd.read_csv(MAN).to_dict("records") if MAN.exists() else []
    have = {r["job"] for r in rows}
    for name, seqs in jobs.items():
        if name in have:
            continue
        (OUT / f"{name}.json").write_text(json.dumps(
            {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
             "sequences": seqs}))
        size = sum(len(x["protein"]["sequence"]) for x in seqs)
        rows.append({"job": name, "arm": "S2_assembly", "receptor": rec, "partner": par,
                     "ecd": f"{s}-{e}", "rec_ecd": "full", "size": size})
        log.info("  %-28s %d chains, %d tokens", name, len(seqs), size)

    man = pd.DataFrame(rows).drop_duplicates("job").sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    tmp = MAN.with_suffix(".csv.tmp"); man.to_csv(tmp, index=False); os.replace(tmp, MAN)
    log.info("\nmanifest now %d jobs", len(man))


if __name__ == "__main__":
    main()
