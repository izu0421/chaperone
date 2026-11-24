"""Score the noradrenaline counter-screen, and say what it decides.

Two arms, neither informative alone:

  A  noradrenaline against 18 receptors. ADRB2 is the positive control: an
     adrenoceptor really does bind noradrenaline, so a screen that cannot score
     it highly there is not measuring anything. The other 15 panel receptors have
     known, different ligands.
  B  GPR85 and GPR153 against all 17 ligands. GPR26 cleared the same calibrated
     bar with glutamate, GABA and acetylcholine, which at a 10%-negative
     threshold is a permissive pocket and not three ligands. The question is
     whether these two are the same.

Scored by the same code as the screen and judged against the same thresholds,
which were calibrated before these folds existed.

Usage:
    micromamba run -n structmod python -I scripts/24_score_noradrenaline.py --workers 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("nor-score")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_arm", ROOT / "scripts" / "08_score_armB.py")
_arm = importlib.util.module_from_spec(_spec)
sys.modules["score_arm"] = _arm
_spec.loader.exec_module(_arm)


def score_one(d: Path) -> dict:
    return {**_arm.one(d), S.FP_COL: S.fold_fingerprint(d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "work" / "noradrenaline_manifest.csv")
    out = ROOT / "results" / "noradrenaline_out"
    state = S.classify(out, man.job.astype(str).tolist(), stage_live=False)
    log.info("%d of %d folds ready", len(state["ready"]), len(man))

    code_fp = S.code_fingerprint([ROOT / "scripts" / f for f in
                                  ("24_score_noradrenaline.py", "08_score_armB.py",
                                   "score_store.py")])
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(score_one, [out / n for n in sorted(state["ready"])]))
    d = pd.DataFrame(rows)
    d[S.CODE_COL] = code_fp
    d = d.merge(man, left_on="pair", right_on="job", how="left")
    S.write(ROOT / "results" / "noradrenaline_scores.csv", d, state, code_fp, len(man))

    th = json.loads((ROOT / "results" / "small_molecule_thresholds.json").read_text())
    ci = th["points"]["comparable"]["iptm"]["cut"]
    cp = th["points"]["comparable"]["plddt_B"]["cut"]
    d["passes"] = (d.iptm >= ci) & (d.plddt_B >= cp)
    log.info("\nthreshold, calibrated before these folds existed: "
             "ipTM >= %.2f and ligand pLDDT >= %.0f\n", ci, cp)

    A = d[d.arm.eq("A_noradrenaline_vs_receptors")].sort_values("iptm", ascending=False)
    log.info("ARM A -- noradrenaline against %d receptors", len(A))
    log.info("\n%s", A[["receptor", "role", "iptm", "plddt_B", "bsa", "passes"]]
             .round(2).to_string(index=False))
    pos = A[A.role.eq("positive_control")]
    neg = A[A.role.eq("negative_control")]
    tst = A[A.role.eq("test")]
    log.info("\n  positive control ADRB2: ipTM %.2f, ligand pLDDT %.1f, passes %s",
             pos.iptm.iloc[0], pos.plddt_B.iloc[0], bool(pos.passes.iloc[0]))
    log.info("  negative controls passing: %d of %d  (%s)",
             int(neg.passes.sum()), len(neg),
             ", ".join(neg[neg.passes].receptor) or "none")
    log.info("  the two candidates: %s",
             ", ".join(f"{r.receptor} {'passes' if r.passes else 'FAILS'}"
                       for r in tst.itertuples()))

    B = d[d.arm.eq("B_orphan_receptor_vs_ligands")]
    log.info("\nARM B -- are GPR85 and GPR153 permissive, like GPR26?")
    for rec, g in B.groupby("receptor"):
        p = g[g.passes]
        log.info("  %-8s %d of %d ligands clear the bar%s", rec, len(p), len(g),
                 ": " + ", ".join(sorted(p.ligand)) if len(p) else "")
    log.info("\n  for comparison, GPR26 cleared it with 3 molecules in the screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
