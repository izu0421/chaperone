"""Pre-flight: can AlphaFold3 actually fold these inputs?

The folds here run with `--norun_data_pipeline`, which means AF3 will not build
an alignment for anything -- every protein chain must arrive with its MSA already
embedded. A chain that simply omits the key is not folded single-sequence; it is
rejected, per job, with the shard carrying on to the next one. 85 orphan jobs sat
in that state for four days, retried five times each, counted in the manifest as
a single-sequence arm, and never produced a structure. The screen reported 96
folds remaining when 85 of them were impossible.

That is worth one cheap check before a stage is allowed to launch.

Usage:
    micromamba run -n structmod python -I scripts/check_af3_inputs.py
    micromamba run -n structmod python -I scripts/check_af3_inputs.py --dir work/orphan_inputs_omic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ["work/orphan_inputs_omic", "work/specificity_inputs", "work/huri_pilot_inputs"]


def problems(js: dict) -> list[str]:
    """Everything wrong with one input, as messages. Empty means foldable."""
    out = []
    if not js.get("name"):
        out.append("no name")
    seqs = js.get("sequences")
    if not seqs:
        return out + ["no sequences"]
    ids = []
    for i, entry in enumerate(seqs):
        if not isinstance(entry, dict) or len(entry) != 1:
            out.append(f"sequence {i}: not a single-kind entry")
            continue
        kind, body = next(iter(entry.items()))
        ids.append(body.get("id"))
        if kind == "protein":
            if not body.get("sequence"):
                out.append(f"chain {body.get('id')}: empty sequence")
            # the failure that cost four days: no alignment, and no way to make one
            for key in ("unpairedMsa", "pairedMsa"):
                v = body.get(key)
                if v is None:
                    out.append(f"chain {body.get('id')}: no {key} -- AF3 will reject "
                               f"this job under --norun_data_pipeline")
                elif not isinstance(v, str):
                    out.append(f"chain {body.get('id')}: {key} is {type(v).__name__}")
                elif v and not v.lstrip().startswith(">"):
                    out.append(f"chain {body.get('id')}: {key} is not A3M/FASTA")
            if body.get("templates") is None:
                out.append(f"chain {body.get('id')}: no templates key")
        elif kind == "ligand":
            if not (body.get("ccdCodes") or body.get("smiles")):
                out.append(f"ligand {body.get('id')}: neither ccdCodes nor smiles")
        else:
            out.append(f"sequence {i}: unknown kind {kind!r}")
    if len(set(ids)) != len(ids):
        out.append(f"duplicate chain ids: {ids}")
    return out


def check(d: Path, only: set[str] | None = None) -> tuple[int, dict[str, list[str]]]:
    """Validate the inputs in `d`, or just the named subset of them.

    Each of these files embeds its MSAs, so the orphan set is ~37 GB on disk.
    Reading all of it is the right thing for a pre-flight gate and much too slow
    for a status command, which only cares about the jobs still queued -- hence
    `only`.
    """
    bad: dict[str, list[str]] = {}
    n = 0
    files = sorted(d / f"{j}.json" for j in only) if only is not None else sorted(d.glob("*.json"))
    for f in files:
        if not f.exists():
            continue
        n += 1
        try:
            probs = problems(json.loads(f.read_text()))
        except Exception as exc:  # noqa: BLE001
            probs = [f"unreadable: {type(exc).__name__} {exc}"]
        if probs:
            bad[f.stem] = probs
    return n, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()

    total_bad = 0
    for rel in ([a.dir] if a.dir else DIRS):
        d = ROOT / rel
        if not d.exists():
            print(f"{rel}: does not exist")
            continue
        n, bad = check(d)
        print(f"\n{rel}: {n} inputs, {len(bad)} unfoldable")
        total_bad += len(bad)
        for name, probs in list(bad.items())[:a.show]:
            print(f"    {name}")
            for p in probs[:3]:
                print(f"        {p}")
        if len(bad) > a.show:
            print(f"    ... and {len(bad) - a.show} more")
        if bad:
            kinds: dict[str, int] = {}
            for probs in bad.values():
                for p in probs:
                    key = p.split(":", 1)[-1].strip()[:60]
                    kinds[key] = kinds.get(key, 0) + 1
            print("    reasons:")
            for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
                print(f"        {v:5d}  {k}")
    print("\nPASS: every input is foldable." if not total_bad
          else f"\nFAIL: {total_bad} inputs would be rejected by AlphaFold3.")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
