"""The control every construct arm was missing: the SAME alignment, uncut.

The screen folded with unfiltered omicMSAs. Every re-fold arm builds its
alignment with b7.build_pair + hhfilter_pairs at 90% identity, which is a median
7.5x shallower -- shallower in 40 of 40 pairs checked. So comparing a re-folded
pair against `pae_min_AB` from results/orphan_scores.csv compares TWO changes at
once: the construct and the alignment.

Measured on the 18 pairs that happened to have a full-length fold with the
re-fold arms' own MSA (the rescue arm):

    screen, deep MSA, full length      median PAE 3.06    0 of 18 above 8 A
    full length, re-fold arm's MSA     median PAE 3.53    5 of 18
    mature chain, same MSA             median PAE 5.68    8 of 18

    alignment alone   +0.55 A  (Wilcoxon p = 0.014)
    the cut, matched  +0.23 A  (p = 0.35, not significant)
    as reported       +2.56 A  (p = 0.0005)

So the arm's headline was mostly the alignment. Five of the eight "collapses"
were already above 8 A full length with no cut at all.

This folds the missing baseline for every other comparison in arms M, S and P:
full length, two chains, the re-fold arms' own alignment, one seed. Nothing is
attributable to a construct until its own uncut fold exists on the same
alignment.

Usage:
    micromamba run -n structmod python -I scripts/41_build_depth_matched_baseline.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("baseline")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "baseline_inputs"
MAN = ROOT / "work" / "baseline_manifest.csv"
SEEDS = [1]

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))

    have = set(pd.read_csv(ROOT / "results" / "rescue_per_job.csv")
               .query("arm == 'A_multiseed'").job.str.replace("A_", "", regex=False))
    want: dict[str, tuple[str, str, str]] = {}
    sub = pd.read_csv(ROOT / "work" / "subcomplex_manifest.csv")
    for r in sub.itertuples():
        pair = f"{r.receptor}__{r.partner}"
        if r.arm in ("M_mature", "S_subcomplex") and pair not in have:
            want[pair] = (r.receptor, r.partner, r.arm)
    ecd = pd.read_csv(ROOT / "work" / "ectodomain_manifest.csv")
    for r in ecd[ecd.arm.isin(["P_candidate", "P_both_ecd"])].itertuples():
        pair = f"{r.receptor}__{r.partner}"
        if pair not in have:
            want.setdefault(pair, (r.receptor, r.partner, r.arm))

    rows = []
    for pair, (rec, par, arm) in sorted(want.items()):
        name = f"X_{pair}"
        if (OUT / f"{name}.json").exists():
            continue
        if rec not in acc or par not in acc:
            log.warning("  %s: no accession", pair); continue
        try:
            qA, qB, rA, rB = b7.build_pair(acc[rec], acc[par])
            kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
        except Exception as exc:                                # noqa: BLE001
            log.warning("  %s: %s", pair, str(exc)[:70]); continue
        (OUT / f"{name}.json").write_text(json.dumps(
            {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
             "sequences": [
                 {"protein": {"id": "A", "sequence": qA,
                              "pairedMsa": b7.a3m([("query", qA)] + kA),
                              "unpairedMsa": b7.a3m([("query", qA)] + kA),
                              "templates": []}},
                 {"protein": {"id": "B", "sequence": qB,
                              "pairedMsa": b7.a3m([("query", qB)] + kB),
                              "unpairedMsa": b7.a3m([("query", qB)] + kB),
                              "templates": []}}]}))
        rows.append({"job": name, "arm": "X_baseline", "receptor": rec, "partner": par,
                     "for_arm": arm, "size": len(qA) + len(qB),
                     "msa_rows_A": len(kA) + 1, "msa_rows_B": len(kB) + 1})
        log.info("  %-28s %5d tokens, %d alignment rows", name, len(qA) + len(qB), len(kA) + 1)

    # Merge with what is already there. Writing only the NEW rows would truncate
    # the manifest, and the fold launcher reads it every loop -- so a second
    # build would silently shrink the queue of a run already in progress. That
    # is exactly what happened here: 30 jobs became 1.
    old = pd.read_csv(MAN).to_dict("records") if MAN.exists() else []
    rows = list({r["job"]: r for r in (old + rows)}.values())
    if not rows:
        log.info("nothing new to build"); return
    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    tmp = MAN.with_suffix(".csv.tmp"); man.to_csv(tmp, index=False); os.replace(tmp, MAN)
    log.info("\n%d baseline folds, %d seed each; median %d tokens",
             len(man), len(SEEDS), int(man["size"].median()))


if __name__ == "__main__":
    main()
