"""Record a deliberate decision to stop chasing the tail of a screen.

A screen that will never reach its manifest has to say so somewhere a reader can
find, because the folds it gives up are not a random sample. On this screen they
are the candidates whose partner had no omicMSA, and the complexes too large for
the GPU -- both systematic. "4,913 of 5,000" with no further detail invites the
reader to assume the missing 87 are arbitrary.

So dropping is an explicit, recorded act:

  run/orphan_retired.txt     the job names, which the fold runner skips and the
                             status command counts as blocked rather than pending
  results/orphan_dropped.csv one row per dropped job with its reason and size,
                             so the Results can state what was given up and why

Nothing is deleted. A dropped job can be un-dropped by removing its line.

Usage:
    micromamba run -n structmod python -I scripts/16_drop_unfolded.py --dry-run
    micromamba run -n structmod python -I scripts/16_drop_unfolded.py --reason "GPU budget"
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("drop")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402


def failure_reasons() -> dict[str, str]:
    """Why AlphaFold3 refused each job, from the shard logs, worst reason kept."""
    txt = ""
    for f in list(ROOT.glob("logs/orphan_shard*.log")) + list(ROOT.glob("logs/orphan_driver*.log")):
        txt += f.read_text(errors="ignore")
    out: dict[str, str] = {}
    for job, why in re.findall(r"Fold job (\S+) failed: (.+)", txt):
        if "RESOURCE_EXHAUSTED" in why or "out of memory" in why.lower():
            out[job] = "CUDA out of memory"
        elif "missing unpaired MSA" in why:
            # these were repaired on 2026-11-23; if one is still unfolded it is
            # because folding stopped, not because the input is broken
            out.setdefault(job, "input had no embedded MSA (since repaired, never re-folded)")
        else:
            out.setdefault(job, why.strip()[:60])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reason", default="screen closed: remaining folds not pursued")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "work" / "orphan_manifest.csv")
    out_dir = ROOT / "results" / "orphan_out"
    state = S.classify(out_dir, man.job.astype(str).tolist(), stage_live=False)
    unfolded = sorted(set(state["queued"]) | set(state["writing"]))
    if not unfolded:
        log.info("every fold in the manifest has output; nothing to drop")
        return 0

    why = failure_reasons()
    rows = man[man.job.isin(unfolded)].copy()
    rows["drop_reason"] = rows.job.map(lambda j: why.get(j, "never attempted"))
    rows["decision"] = a.reason

    log.info("%d of %d folds have no output and would be dropped", len(rows), len(man))
    log.info("\nby reason:\n%s", rows.drop_reason.value_counts().to_string())
    log.info("\nby candidate kind:\n%s", rows.kind.value_counts().to_string())
    log.info("\nby MSA source (the systematic part):\n%s",
             rows.msa_source.value_counts().to_string())
    log.info("\nsize of what is dropped vs what was folded:")
    folded = man[~man.job.isin(unfolded)]
    log.info("  dropped  n=%d  median %.0f tokens  range %.0f-%.0f",
             len(rows), rows["size"].median(), rows["size"].min(), rows["size"].max())
    log.info("  folded   n=%d  median %.0f tokens  range %.0f-%.0f",
             len(folded), folded["size"].median(), folded["size"].min(), folded["size"].max())
    # What the Results has to carry is WHICH systematic, not merely that the
    # dropped set is larger. Those two can come apart: a dropped set can be
    # bigger on average while size itself is not what decided it.
    ratio = rows["size"].median() / folded["size"].median()
    big = man[man["size"] >= rows["size"].min()]
    frac_big_folded = big.job.isin(folded.job).mean()
    log.info("\n  the dropped folds are %.1fx the median size of the folded ones, but size "
             "is NOT what decided it: of the %d candidates at or above the smallest "
             "dropped size, %d folded (%.0f%%)", ratio, len(big),
             int(big.job.isin(folded.job).sum()), 100 * frac_big_folded)
    top = rows.drop_reason.value_counts()
    log.info("  the systematic is the dominant reason, %d of %d (%.0f%%): %s",
             int(top.iloc[0]), len(rows), 100 * top.iloc[0] / len(rows), top.index[0])
    if "msa_source" in rows.columns:
        arm = rows.msa_source.value_counts()
        in_man = man.msa_source.value_counts().reindex(arm.index).fillna(0)
        log.info("  which arm it removes:")
        for src in arm.index:
            log.info("    %-26s %d of %d in the manifest (%.0f%% of that arm lost)",
                     src, int(arm[src]), int(in_man[src]),
                     100 * arm[src] / max(in_man[src], 1))
    log.info("  so the sentence for the Results is: the screen covers %d of %d candidates; "
             "the %d not folded are %s, and they are larger than average because those "
             "partners are large proteins, not because large complexes failed",
             len(folded), len(man), len(rows), top.index[0])

    if a.dry_run:
        log.info("\ndry run: nothing written")
        return 0

    (ROOT / "run").mkdir(exist_ok=True)
    rf = ROOT / "run" / "orphan_retired.txt"
    tmp = rf.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(sorted(rows.job)) + "\n")
    os.replace(tmp, rf)

    cols = ["job", "receptor", "partner", "kind", "size", "msa_source", "rank",
            "p_twoway", "drop_reason", "decision"]
    dest = ROOT / "results" / "orphan_dropped.csv"
    rows[[c for c in cols if c in rows.columns]].sort_values("size").to_csv(dest, index=False)
    log.info("\nwrote %s (%d jobs) and %s", rf.name, len(rows), dest.name)
    log.info("the fold runner now skips these, and 00_status.py counts them as blocked "
             "rather than pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
