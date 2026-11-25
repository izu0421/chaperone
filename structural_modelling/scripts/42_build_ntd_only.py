"""The ECD without its stitched loops, because the stitch is where the interface went.

Arm B's `ext_segments()` keeps EVERY segment annotated Extracellular, so the
construct is not the extracellular domain: it is the N-terminal domain with the
three extracellular loops joined onto its C-terminus. For GPR158 that is
24-417 plus 475-478, 548-576 and 634-642 -- 436 aa with three junctions.

Measured on the best model: 31 of the 87 receptor interface residues, 35.6%,
sit on those 42 stitched residues, which are 9.6% of the construct. A 3.70x
enrichment onto surface the intact receptor does not have. The three decoys
never visit it (0.0%, 0.0%, 4.2%), so they cannot rule out that the groove the
stitch opened is what the candidate is binding -- which is the exact failure the
decoys were added to catch.

So fold the N-terminal domain ALONE, with the same decoys. If neuromedin S still
beats them on a construct with no artificial junction, the result stands on real
surface. If it does not, arm B adds nothing to the intact-receptor fold that
already scores 2.26 A with 54 of 60 interface residues on genuine extracellular
positions.

Five seeds, matching arm B, so candidate and decoys are compared at equal draws.

Usage:
    micromamba run -n structmod python -I scripts/42_build_ntd_only.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ntd")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "rescue_inputs"
MAN = ROOT / "work" / "rescue_manifest.csv"
SEEDS = [1, 2, 3, 4, 5]
DECOYS = {"ALB": "P02768", "LYZ": "P61626", "RNASE1": "P07998"}

_s31 = iu.spec_from_file_location("b31", ROOT / "scripts" / "31_build_refold_rescue.py")
b31 = iu.module_from_spec(_s31)
_s31.loader.exec_module(b31)
b7 = b31.b7


def main() -> None:
    topo = json.loads((ROOT / "results" / "orphan_topology.json").read_text())
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))
    acc.update(DECOYS)

    rows = pd.read_csv(MAN).to_dict("records") if MAN.exists() else []
    have = {r["job"] for r in rows}
    new = []
    for rec, par in (("GPR158", "NMS"),):
        segs = b31.ext_segments(topo.get(rec, []))
        if not segs:
            log.warning("%s: no extracellular annotation", rec); continue
        ntd = max(segs, key=lambda s: s[1] - s[0])
        dropped = sum(b - a + 1 for a, b in segs if (a, b) != ntd)
        log.info("%s: keeping ONLY %d-%d (%d aa); dropping %d aa of stitched loops "
                 "across %d segments", rec, ntd[0], ntd[1], ntd[1] - ntd[0] + 1,
                 dropped, len(segs) - 1)
        for pname, pacc in [(par, acc[par])] + list(DECOYS.items()):
            name = f"B2_{rec}NTD__{pname}"
            if name in have or (OUT / f"{name}.json").exists():
                continue
            try:
                qA, qB, rA, rB = b7.build_pair(acc[rec], pacc)
                kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
            except Exception as exc:                            # noqa: BLE001
                log.warning("  %s: %s", name, str(exc)[:70]); continue
            msaA = b7.a3m([("query", qA)] + kA)
            qAc, msaAc = b31.truncate_to_ecd(qA, msaA, [ntd])
            msaB = b7.a3m([("query", qB)] + kB)
            (OUT / f"{name}.json").write_text(json.dumps(
                {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
                 "sequences": [
                     {"protein": {"id": "A", "sequence": qAc, "pairedMsa": msaAc,
                                  "unpairedMsa": msaAc, "templates": []}},
                     {"protein": {"id": "B", "sequence": qB, "pairedMsa": msaB,
                                  "unpairedMsa": msaB, "templates": []}}]}))
            arm = "B2_ntd" if pname == par else "B2_ntd_decoy"
            new.append({"job": name, "arm": arm, "receptor": rec, "partner": pacc,
                        "ecd": True, "size": len(qAc) + len(qB), "n_seeds": len(SEEDS)})
            log.info("  %-26s %d + %d = %d tokens", name, len(qAc), len(qB), len(qAc) + len(qB))

    if not new:
        log.info("nothing new to build"); return
    man = pd.DataFrame(rows + new).drop_duplicates("job").sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    tmp = MAN.with_suffix(".csv.tmp"); man.to_csv(tmp, index=False); os.replace(tmp, MAN)
    log.info("\nmanifest now %d jobs (%d new)", len(man), len(new))


if __name__ == "__main__":
    main()
