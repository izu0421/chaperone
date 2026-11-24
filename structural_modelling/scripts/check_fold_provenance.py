"""Guard: is each fold on disk actually the fold its input now describes?

The builders rewrite every input JSON unconditionally, and the runners skip any
job whose summary JSON already exists. Put together, an input that changes after
its fold has run -- a new UniProt release, a different isoform, a corrected
ligand SMILES, a re-pick of the candidate list -- leaves the old structure in
place forever, while the manifest describes something else. Nothing errors. The
screen simply contains a fold of sequences nobody is claiming to have folded.

A job name is not proof of what is in the file. AlphaFold3 writes the sequences
it actually used into `<job>_data.json` in the output directory, so provenance
can be checked by content, after the fact, with no changes to the runners.

Reading 5,000 of those is slow, because each embeds its MSAs. So this screens on
mtime first -- an input older than its own fold output cannot have drifted -- and
compares content only for the ones that could have.

What is covered: the per-chain sequences, and ligand CCD codes or SMILES. What is
NOT covered: MSA content, which is large, legitimately variable between builds,
and would flag everything.

Usage:
    micromamba run -n structmod python -I scripts/check_fold_provenance.py
    micromamba run -n structmod python -I scripts/check_fold_provenance.py --stage orphan --deep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402

STAGES = {
    "orphan": ("orphan_manifest.csv", "work/orphan_inputs_omic", "results/orphan_out"),
    "specificity": ("specificity_manifest.csv", "work/specificity_inputs",
                    "results/specificity_out"),
    "huri_pilot": ("huri_pilot_manifest.csv", "work/huri_pilot_inputs",
                   "results/huri_pilot_out"),
    # The re-fold arms carry the greatest provenance risk in the project: every
    # one of them folds a CUT sequence, so a wrong boundary or a shifted
    # alignment frame produces a perfectly valid fold of the wrong molecule and
    # the error shows up as biology. Each cut was checked against UniProt by
    # hand when it was built; this is the check that keeps holding afterwards.
    "rescue": ("rescue_manifest.csv", "work/rescue_inputs", "results/rescue_out"),
    "subcomplex": ("subcomplex_manifest.csv", "work/subcomplex_inputs",
                   "results/subcomplex_out"),
    "mature_control": ("mature_control_manifest.csv", "work/subcomplex_inputs",
                       "results/subcomplex_out"),
    "ectodomain": ("ectodomain_manifest.csv", "work/ectodomain_inputs",
                   "results/ectodomain_out"),
}


def sequence_fingerprint(js: dict) -> str:
    """A hash of what was asked to be folded, ignoring the alignments.

    Chain id, sequence, and ligand identity. Sorted by id so a reordering of the
    sequence list is not mistaken for a different job.
    """
    parts = []
    for entry in js.get("sequences", []):
        for kind, body in entry.items():
            if not isinstance(body, dict):
                continue
            parts.append("|".join([
                kind,
                str(body.get("id")),
                (body.get("sequence") or "").strip().upper(),
                ",".join(sorted(body.get("ccdCodes") or [])),
                (body.get("smiles") or "").strip(),
            ]))
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()[:16]


def check(stage: str, deep: bool) -> dict:
    man_name, in_rel, out_rel = STAGES[stage]
    man_path = ROOT / "work" / man_name
    in_dir, out_dir = ROOT / in_rel, ROOT / out_rel
    if not man_path.exists() or not out_dir.exists():
        return {"stage": stage, "state": "not built yet"}

    man = pd.read_csv(man_path)
    expected = man.job.astype(str).tolist()
    state = S.classify(out_dir, expected, None)
    ready = list(state["ready"])

    no_input, suspect, ok_by_mtime = [], [], 0
    for name in ready:
        src = in_dir / f"{name}.json"
        if not src.exists():
            no_input.append(name)
            continue
        fold = out_dir / name / f"{name}_summary_confidences.json"
        if src.stat().st_mtime <= fold.stat().st_mtime:
            ok_by_mtime += 1          # input predates the fold: cannot have drifted
        else:
            suspect.append(name)

    to_compare = ready if deep else suspect
    drifted, unreadable = [], []
    for name in to_compare:
        src, rec = in_dir / f"{name}.json", out_dir / name / f"{name}_data.json"
        if not src.exists() or not rec.exists():
            unreadable.append(name)
            continue
        try:
            if sequence_fingerprint(json.loads(src.read_text())) != \
               sequence_fingerprint(json.loads(rec.read_text())):
                drifted.append(name)
        except Exception as exc:  # noqa: BLE001
            unreadable.append(f"{name}: {type(exc).__name__}")

    return {"stage": stage, "folds_ready": len(ready),
            "input_older_than_fold": ok_by_mtime,
            "input_newer_than_fold": len(suspect),
            "input_missing": len(no_input),
            "content_compared": len(to_compare),
            "sequences_differ": drifted,
            "unreadable": unreadable[:8],
            "orphan_outputs_not_in_manifest": len(state["unexpected"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, choices=sorted(STAGES))
    ap.add_argument("--deep", action="store_true",
                    help="compare content for every fold, not just the ones whose "
                         "input is newer. Slow: each record embeds its MSAs.")
    a = ap.parse_args()

    bad = 0
    for stage in ([a.stage] if a.stage else sorted(STAGES)):
        r = check(stage, a.deep)
        print(f"\n{r['stage']}")
        for k, v in r.items():
            if k == "stage":
                continue
            print(f"    {k:34s} {len(v) if isinstance(v, list) and k != 'unreadable' else v}"
                  if k == "sequences_differ" else f"    {k:34s} {v}")
        if r.get("sequences_differ"):
            bad += len(r["sequences_differ"])
            print(f"    !! {len(r['sequences_differ'])} folds were made from different "
                  f"sequences than their input now specifies:")
            for n in r["sequences_differ"][:10]:
                print(f"         {n}")
            print("    These must be re-folded, or the manifest corrected. The structure "
                  "on disk is not the structure the manifest describes.")
    print("\nPASS: every fold checked matches its input." if not bad
          else f"\nFAIL: {bad} folds do not match their inputs.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
