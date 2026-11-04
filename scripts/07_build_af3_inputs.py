"""Build AF3 inputs (arm B) from the omicMSAs, keeping the query full length.

Their `generate_protein_pair_MSA.py` does three things: applies the quality mask by DELETING
low-quality positions, pairs rows by genome accession, concatenates, and hhfilters at 90%.
Pairing and filtering carry over unchanged here. The masking cannot.

**Why the mask has to be handled differently for AF3.** The omicMSA README says to use only the
positions marked `*`, "as we did in our work", and their script implements that by truncating the
query. For RF2-PPI that is harmless: it consumes an alignment and never emits a structure, so
dropping columns drops evidence and nothing else. AF3 *folds the query sequence*. Measured over
400 randomly sampled omicMSAs, the mask keeps a median of 83.3% of positions, and **31.8% of
proteins lose more than a quarter of their sequence**; only 3.8% keep all of it. Feeding AF3 the
masked query would mean folding a protein with a third of its residues deleted, and the resulting
interface would be an artefact of the deletion.

So the mask is applied as GAPS, not as deletions: the query keeps every residue, and each
homolog row carries its aligned residue at high-quality positions and `-` at low-quality ones.
That expresses exactly what the mask means — "the aligned residue here is not trustworthy" —
without truncating the thing being folded. The evidence AF3 sees at masked positions is then the
same as the evidence RF2-PPI sees: none.

Pairing is by genome accession intersection, which is what their script does and is well defined
here because an omicMSA carries one sequence per genome, so there is no "closest hit per
organism" choice to make.

Templates are disabled by default. RF2-PPI is MSA-only, so leaving templates on for AF3 would
confound the engine comparison with a structural-database lookup — and this project has already
been bitten by PDB leakage once (`finding_stcrdab_benchmark_contaminated`: 77% of a TCR benchmark
predated AF3's cutoff and the held-out half behaved differently).

Usage:
    python -I scripts/07_build_af3_inputs.py --pairs <file> --out-dir <dir> [--limit N]
    python -I scripts/07_build_af3_inputs.py --validate      # reproduce their own example
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("af3in")

HERE = Path(__file__).resolve().parent.parent
ALNS = HERE / "data/rf2ppi_raw/protein_omicMSAs/alns"
HHFILTER = Path("/data/yzy21/micromamba/envs/structmod/bin/hhfilter")


def read_omic(acc: str) -> tuple[str, str, dict[str, str]]:
    """(mask, query, {genome_accession: aligned_sequence}) for one omicMSA."""
    f = ALNS / f"{acc}.a3m"
    mask = query = None
    hits: dict[str, str] = {}
    name = None
    with open(f) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
            elif name == "mask" and mask is None:
                mask = line.rstrip("\n")
            elif name == "query" and query is None:
                query = line.rstrip("\n")
            elif name:
                hits[name] = line.rstrip("\n")
    if mask is None or query is None:
        raise ValueError(f"{f} lacks a mask or query record")
    return mask, query, hits


def gap_masked(seq: str, good: set[int]) -> str:
    """Rewrite one hit row so masked query positions become gaps, keeping the query frame.

    Only uppercase letters and '-' consume a query position; lowercase letters are insertions.
    An insertion is kept only when both flanking query positions are high quality, which is the
    same rule their script uses -- an insertion anchored to an untrustworthy column is itself
    untrustworthy.
    """
    out, i = [], -1
    for ch in seq:
        if ch.islower():
            if i in good and (i + 1) in good:
                out.append(ch)
        else:
            i += 1
            out.append(ch if i in good else "-")
    return "".join(out)


def build_pair(a: str, b: str) -> tuple[str, str, list[tuple[str, str]], list[tuple[str, str]]]:
    """Full-length queries plus accession-paired, mask-gapped rows for both chains."""
    mA, qA, hA = read_omic(a)
    mB, qB, hB = read_omic(b)
    gA = {i for i, c in enumerate(mA) if c == "*"}
    gB = {i for i, c in enumerate(mB) if c == "*"}
    shared = sorted(set(hA) & set(hB))
    rowsA = [(acc, gap_masked(hA[acc], gA)) for acc in shared]
    rowsB = [(acc, gap_masked(hB[acc], gB)) for acc in shared]
    return qA, qB, rowsA, rowsB


def hhfilter_pairs(qA, qB, rowsA, rowsB, ident=90):
    """Redundancy-filter on the CONCATENATED alignment, as their pipeline does, then split back.

    Filtering the two chains separately would keep different rows for each and break the
    row-index pairing AF3 relies on.
    """
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.a3m", Path(td) / "out.a3m"
        with open(src, "w") as fh:
            fh.write(f">query\n{qA}{qB}\n")
            for (acc, sa), (_, sb) in zip(rowsA, rowsB):
                fh.write(f">{acc}\n{sa}{sb}\n")
        r = subprocess.run([str(HHFILTER), "-i", str(src), "-o", str(dst),
                            "-id", str(ident)], capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            raise RuntimeError(f"hhfilter failed: {r.stderr[-300:]}")
        kept, name = [], None
        for line in dst.read_text().splitlines():
            if line.startswith(">"):
                name = line[1:].split()[0]
            elif name:
                kept.append((name, line.strip()))
                name = None
    # Split at QUERY POSITION l1, not at character l1. The rows still carry lowercase
    # insertions (an insertion survives when both flanking query positions are good), so a raw
    # slice shifts the frame -- exactly the trap scripts/06 was written to avoid, which the
    # first version of this function walked into anyway. The assertion then used was
    # len(sa) == l1, which counts CHARACTERS and therefore passed while the split was wrong;
    # AF3 caught it instead, with "all strings must have the same number of non-lowercase
    # characters; First string has 445 ... but ... has 443". Assert on query positions.
    l1 = len(qA)
    keptA, keptB = [], []
    for n, s in kept:
        pos = 0
        cut = len(s)
        for i, ch in enumerate(s):
            if pos == l1 and not ch.islower():
                cut = i
                break
            if not ch.islower():
                pos += 1
        sa, sb = s[:cut], s[cut:]
        assert sa + sb == s, f"split lost characters in {n}"
        assert qlen(sa) == l1, f"{n}: chain A has {qlen(sa)} query positions, not {l1}"
        keptA.append((n, sa))
        keptB.append((n, sb))
    l2 = qlen(keptB[0][1]) if keptB else 0
    for n, s in keptB:
        assert qlen(s) == l2, f"{n}: chain B has {qlen(s)} query positions, not {l2}"
    return keptA, keptB


def qlen(s: str) -> int:
    """Query positions in an A3M row: uppercase and '-' count, lowercase insertions do not."""
    return sum(1 for c in s if not c.islower())


def a3m(rows) -> str:
    return "".join(f">{n}\n{s}\n" for n, s in rows)


def af3_json(name, qA, qB, rowsA, rowsB, seed=1, templates=False) -> dict:
    # AF3 requires every row of an MSA to have the same number of non-lowercase characters as
    # the query. Check it here rather than discovering it from a featurisation error.
    for cid, seq, rows in (("A", qA, rowsA), ("B", qB, rowsB)):
        bad = [n for n, s in rows if qlen(s) != len(seq)]
        if bad:
            raise AssertionError(f"{name} chain {cid}: {len(bad)} rows do not match the query's "
                                 f"{len(seq)} query positions, e.g. {bad[:3]}")

    def chain(cid, seq, rows):
        p = {"id": cid, "sequence": seq,
             "pairedMsa": a3m([("query", seq)] + rows),
             "unpairedMsa": a3m([("query", seq)] + rows)}
        if not templates:
            p["templates"] = []
        return {"protein": p}
    return {"name": name, "modelSeeds": [seed], "dialect": "alphafold3", "version": 1,
            "sequences": [chain("A", qA, rowsA), chain("B", qB, rowsB)]}


def validate():
    """Reproduce their own paired MSAs with their masking rule, as a check on the parser."""
    ex = HERE / "ext/RoseTTAFold2-PPI/examples"
    ok = 0
    for line in (ex / "protein_pairs").read_text().split():
        a, b = line.split("_")
        mA, qA, hA = read_omic_from(ex / "protein_single_msas", a)
        mB, qB, hB = read_omic_from(ex / "protein_single_msas", b)
        gA = {i for i, c in enumerate(mA) if c == "*"}
        gB = {i for i, c in enumerate(mB) if c == "*"}
        # their rule: DELETE masked positions
        tq = "".join(c for i, c in enumerate(qA) if i in gA)
        theirs = (ex / "protein_paired_msas" / f"{line}.i90.a3m").read_text().splitlines()
        their_q = theirs[1]
        assert their_q.startswith(tq), f"{line}: query mismatch under their masking rule"
        # ours: keep full length
        assert len(qA) >= len(tq)
        log.info("  %-16s their L1=%d  full-length L1=%d  (%d residues masked away, %.0f%%)",
                 line, len(tq), len(qA), len(qA) - len(tq),
                 100 * (len(qA) - len(tq)) / len(qA))
        ok += 1
    log.info("parser validated against %d of their own paired MSAs", ok)


def read_omic_from(d: Path, acc: str):
    global ALNS
    old, ALNS = ALNS, d
    try:
        return read_omic(acc)
    finally:
        ALNS = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="file of A_B pair ids, one per line")
    ap.add_argument("--out-dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ident", type=int, default=90)
    ap.add_argument("--templates", action="store_true")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
        return
    if not (a.pairs and a.out_dir):
        raise SystemExit("need --pairs and --out-dir (or --validate)")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [l.split()[0] for l in Path(a.pairs).read_text().splitlines() if l.strip()]
    if a.limit:
        pairs = pairs[:a.limit]
    made, skipped = 0, []
    for p in pairs:
        x, y = p.split("_")
        try:
            qA, qB, rA, rB = build_pair(x, y)
            if not rA:
                skipped.append((p, "no shared genomes"))
                continue
            kA, kB = hhfilter_pairs(qA, qB, rA, rB, a.ident)
            (out / f"{p}.json").write_text(json.dumps(
                af3_json(p, qA, qB, kA, kB, templates=a.templates), indent=1))
            made += 1
            if made % 25 == 0:
                log.info("  %d/%d built", made, len(pairs))
        except Exception as e:                      # missing omicMSA, hhfilter failure
            skipped.append((p, str(e)[:70]))
    log.info("built %d AF3 inputs, skipped %d", made, len(skipped))
    if skipped:
        log.warning("first skips: %s", skipped[:4])
    # A silent total failure must not look like a clean run.
    if made < 0.5 * len(pairs):
        raise SystemExit(f"only {made} of {len(pairs)} inputs built -- the build is broken")


if __name__ == "__main__":
    main()
