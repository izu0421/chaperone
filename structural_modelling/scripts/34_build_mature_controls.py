"""The control the mature-chain arm needs: cut the signal peptide off KNOWN answers.

Arm M re-folded every secreted candidate partner as its mature chain, because the
screen folded the full preprohormone and the signal peptide was supplying the
cytoplasmic contacts the topology rule then rejected. The first pass showed most
candidate interfaces COLLAPSING once the peptide went.

That reads two ways and the data so far cannot tell them apart:

  (a) the candidate interface was held together by a peptide that does not exist
      in the mature ligand, so the candidate was an artefact; or
  (b) cutting 20-30 residues off the N-terminus breaks AlphaFold3's pose for
      EVERYTHING, candidate and real ligand alike.

Chemokines decide it. Every chemokine is secreted with a signal peptide, the
mature chain IS the physiological ligand, and the answers are known. So cut the
same way on the 24 arm-C pairs:

    if the known binders keep their interface and the candidates lose theirs,
        reading (a) holds and those candidates go.
    if the known binders lose it too, the cut itself is destructive and arm M
        says nothing about any candidate.

One seed, as arm M now runs -- the question is the construct, not the draw.

Jobs are appended to the subcomplex manifest, which the running fold shard
re-reads every loop, so these enter the queue without restarting anything.

Usage:
    micromamba run -n structmod python -I scripts/34_build_mature_controls.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("mature-control")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "subcomplex_inputs"
MAN = ROOT / "work" / "subcomplex_manifest.csv"
SEEDS = [1]

_s33 = iu.spec_from_file_location("b33", ROOT / "scripts" / "33_build_subcomplex_and_mature.py")
b33 = iu.module_from_spec(_s33)
_s33.loader.exec_module(b33)
b7 = b33.b7

PANEL = {"CCR1": "P32246", "CCR2": "P41597", "CCR3": "P51677", "CCR5": "P51681",
         "CCR6": "P51684", "CCR7": "P32248", "CXCR1": "P25024", "CXCR2": "P25025",
         "CXCR3": "P49682", "CXCR4": "P61073", "CX3CR1": "P49238", "ACKR3": "P25106",
         "CCR4": "P51679", "CCR8": "P51685", "CXCR5": "P32302", "CXCR6": "O00574",
         "CCL2": "P13500", "CCL3": "P10147", "CCL5": "P13501", "CCL11": "P51671",
         "CCL19": "Q99731", "CCL20": "P78556", "CXCL8": "P10145", "CXCL10": "P02778",
         "CXCL12": "P48061", "CX3CL1": "P78423", "CCL21": "O00585", "CXCL11": "O14625",
         "CCL17": "Q92583", "CCL1": "P22362", "CXCL13": "O43927", "CXCL16": "Q9H2A7"}


def main() -> None:
    cs = pd.read_csv(ROOT / "results" / "protein_counterscreen_scores.csv")
    A = cs[cs.block.eq("A_chemokine")]
    rows, sig_cache = [], {}

    for lab, n in (("binder", 12), ("negative", 12)):
        for r in A[A.label.eq(lab)].head(n).itertuples():
            rec, par = r.chain_a, r.chain_b
            if rec not in PANEL or par not in PANEL:
                continue
            name = f"N_{lab}_{rec}__{par}"
            if (OUT / f"{name}.json").exists():
                continue
            if par not in sig_cache:
                sig_cache[par] = b33.uniprot_signal(PANEL[par])
            sig = sig_cache[par]
            if not sig:
                log.warning("  %s: no annotated signal peptide, skipped", par); continue
            try:
                qA, qB, rA, rB = b7.build_pair(PANEL[rec], PANEL[par])
                kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
            except Exception as exc:                            # noqa: BLE001
                log.warning("  %s+%s: %s", rec, par, str(exc)[:70]); continue
            if sig >= len(qB):
                log.warning("  %s: signal %d >= length %d", par, sig, len(qB)); continue
            qBm, msaBm = b33.drop_prefix(qB, b7.a3m([("query", qB)] + kB), sig)
            (OUT / f"{name}.json").write_text(json.dumps(
                {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3",
                 "version": 1,
                 "sequences": [b33.chain_block("A", qA, b7.a3m([("query", qA)] + kA)),
                               b33.chain_block("B", qBm, msaBm)]}))
            rows.append({"job": name, "arm": f"N_mature_control_{lab}", "receptor": rec,
                         "partner": par, "extras": f"signal 1-{sig} removed",
                         "n_chains": 2, "size": len(qA) + len(qBm)})
            log.info("  %-30s signal 1-%-3d removed, %d -> %d aa", name, sig,
                     len(qB), len(qBm))

    if not rows:
        log.info("nothing new to build"); return

    old = pd.read_csv(MAN)
    man = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    man = man.sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    # Atomic: the fold shard re-reads this file every loop and a partial read
    # would drop jobs from its queue silently.
    tmp = MAN.with_suffix(".csv.tmp")
    man.to_csv(tmp, index=False)
    os.replace(tmp, MAN)
    log.info("\nappended %d control jobs; manifest now %d jobs", len(rows), len(man))
    log.info("%s", man.groupby("arm").agg(jobs=("job", "size"),
                                          median_tokens=("size", "median")).to_string())


if __name__ == "__main__":
    main()
