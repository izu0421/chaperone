"""Repair the orphan inputs that AlphaFold3 cannot fold, and only those.

Re-running 12_build_orphan_inputs_omic.py would rewrite all 5,000 inputs. That is
not harmless: every input would then be newer than its own fold, which is exactly
the signal check_fold_provenance.py uses to decide what might have drifted, so a
full rebuild would turn 4,909 clean folds into 4,909 suspects and force a deep
content comparison over files that each embed their MSAs.

So this repairs in place: it rewrites only the jobs that fail the pre-flight
check, leaves every other file untouched, and updates those rows of the manifest.

Usage:
    micromamba run -n structmod python -I scripts/13_repair_unfoldable_inputs.py --dry-run
    micromamba run -n structmod python -I scripts/13_repair_unfoldable_inputs.py
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("repair")
ROOT = Path(__file__).resolve().parent.parent
SHEET = Path("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
OUT = ROOT / "work" / "orphan_inputs_omic"
MAN = ROOT / "work" / "orphan_manifest.csv"

_s = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_s)
_s.loader.exec_module(b7)

sys.path.insert(0, str(ROOT / "scripts"))
_c = iu.spec_from_file_location("chk", ROOT / "scripts" / "check_af3_inputs.py")
chk = iu.module_from_spec(_c)
_c.loader.exec_module(chk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    _, bad = chk.check(OUT)
    if not bad:
        log.info("every input is already foldable; nothing to repair")
        return 0
    log.info("%d inputs fail the pre-flight check", len(bad))

    sheet = pd.read_csv(SHEET)
    sheet["job"] = sheet.receptor_gene + "__" + sheet.partner_gene
    sheet = sheet.set_index("job")

    fixed, failed = [], []
    for job in sorted(bad):
        if job not in sheet.index:
            failed.append((job, "not in the source sheet"))
            continue
        r = sheet.loc[job]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        try:
            # chain A keeps its own omicMSA: only the partner is missing one, and
            # throwing the receptor's away made the fold worse for no reason
            mA, qA, hA = b7.read_omic(r.receptor_uniprot)
            gA = {i for i, c in enumerate(mA) if c == "*"}
            a3mA = b7.a3m([("query", qA)] + [(acc, b7.gap_masked(hA[acc], gA))
                                             for acc in sorted(hA)])
            qB = r.partner_seq
            a3mB = b7.a3m([("query", qB)])     # the query alone: single sequence, stated
            js = {"name": job, "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
                  "sequences": [
                      {"protein": {"id": "A", "sequence": qA, "pairedMsa": a3mA,
                                   "unpairedMsa": a3mA, "templates": []}},
                      {"protein": {"id": "B", "sequence": qB, "pairedMsa": a3mB,
                                   "unpairedMsa": a3mB, "templates": []}}]}
            probs = chk.problems(js)
            if probs:
                failed.append((job, "; ".join(probs[:2])))
                continue
            if not a.dry_run:
                (OUT / f"{job}.json").write_text(json.dumps(js))
            fixed.append({"job": job, "size": len(qA) + len(qB),
                          "msa_source": "partner_single_sequence"})
        except Exception as exc:  # noqa: BLE001
            failed.append((job, f"{type(exc).__name__} {str(exc)[:60]}"))

    log.info("repaired %d, could not repair %d", len(fixed), len(failed))
    for job, why in failed[:10]:
        log.warning("  %s: %s", job, why)

    if fixed and not a.dry_run:
        man = pd.read_csv(MAN).set_index("job")
        upd = pd.DataFrame(fixed).set_index("job")
        # size changes because the receptor sequence now comes from the omicMSA
        # query instead of the sheet; msa_source changes because the old label
        # claimed a single-sequence fold that never happened
        for col in ("size", "msa_source"):
            man.loc[upd.index, col] = upd[col]
        # atomic: the fold shards re-read this manifest on every loop iteration,
        # and a reader that catches a half-written CSV loses its whole job list
        import os
        tmp = MAN.with_suffix(".csv.tmp")
        man.reset_index().to_csv(tmp, index=False)
        os.replace(tmp, MAN)
        log.info("updated %d manifest rows; msa_source now:\n%s",
                 len(upd), man.msa_source.value_counts().to_string())

    if not a.dry_run:
        _, still = chk.check(OUT)
        log.info("post-repair pre-flight: %d unfoldable", len(still))
        return 1 if still else 0
    log.info("dry run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
