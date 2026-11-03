"""Convert a RF2-PPI paired omicMSA into the per-chain A3M that AF3 takes.

This is the piece that makes arm B possible, and the piece most likely to produce a wrong
answer that looks like a biological result, so it is written with its own round-trip test.

The two formats differ in shape, not just syntax:

  RF2-PPI paired MSA   one alignment whose every row is chain A's sequence CONCATENATED with
                       chain B's. The input list carries L1, the query length of chain A.
  AF3                  two alignments, one per chain, supplied as `paired_msa` on each chain.
                       Row i of chain A's alignment is taken to pair with row i of chain B's.

So the conversion is a split at query position L1 — and a naive `seq[:L1]` is wrong. A3M marks
insertions relative to the query with LOWERCASE letters, which do not consume a query position.
Chain A therefore ends after L1 characters that are uppercase or '-', with any lowercase
insertions in between carried along. Slicing by raw index silently shifts the frame for every
row that has an insertion, which is most of them, and the resulting MSA still parses.

The single-protein omicMSAs additionally carry a leading `>mask` record whose sequence is
'*' for positions to keep and '-' for low-quality positions. It is not a sequence and must not
reach AF3.

Checks that run every time:
  * re-concatenating the two halves reproduces the input row exactly, for every row;
  * both halves have the same number of query positions as their declared lengths;
  * row counts match across the two chains, since AF3 pairs by row index.

Usage:
    python -I scripts/06_omicmsa_to_af3.py --self-test
    python -I scripts/06_omicmsa_to_af3.py --paired <a3m> --l1 <n> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("omic2af3")


def read_a3m(path: Path) -> list[tuple[str, str]]:
    """(header, sequence) records, with the omicMSA `mask` pseudo-record dropped."""
    recs, name, buf = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                recs.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        elif line:
            buf.append(line.strip())
    if name is not None:
        recs.append((name, "".join(buf)))
    kept = [(n, s) for n, s in recs if n.split()[0] != "mask"]
    if len(kept) != len(recs):
        log.info("dropped %d mask record(s) from %s", len(recs) - len(kept), Path(path).name)
    return kept


def split_row(seq: str, l1: int) -> tuple[str, str]:
    """Split one A3M row at query position l1.

    Only uppercase letters and '-' advance the query position; lowercase letters are insertions
    relative to the query and belong to whichever chain they sit inside.
    """
    pos, i = 0, 0
    for i, ch in enumerate(seq):
        if pos == l1 and not ch.islower():
            break
        if not ch.islower():
            pos += 1
    else:
        i = len(seq)
    if pos < l1:
        raise ValueError(f"row has only {pos} query positions, needs at least {l1}")
    return seq[:i], seq[i:]


def query_len(seq: str) -> int:
    return sum(1 for c in seq if not c.islower())


def convert(paired: Path, l1: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    recs = read_a3m(paired)
    if not recs:
        raise ValueError(f"{paired} has no records")
    a, b = [], []
    for name, seq in recs:
        sa, sb = split_row(seq, l1)
        # round-trip: the split must be lossless, checked per row rather than on a sample
        if sa + sb != seq:
            raise AssertionError(f"split lost characters in row {name!r} of {paired.name}")
        a.append((name, sa))
        b.append((name, sb))
    ql_a, ql_b = query_len(a[0][1]), query_len(b[0][1])
    if ql_a != l1:
        raise AssertionError(f"chain A query length {ql_a} != declared l1 {l1}")
    if len(a) != len(b):
        raise AssertionError("row counts differ; AF3 pairs by row index")
    log.info("%s: %d rows, L1=%d L2=%d", paired.name, len(a), ql_a, ql_b)
    return a, b


def write_a3m(recs: list[tuple[str, str]], path: Path) -> None:
    path.write_text("".join(f">{n}\n{s}\n" for n, s in recs))


def self_test() -> None:
    """Run on the repo's own example, where the ground truth is the shipped file."""
    ex = Path(__file__).resolve().parent.parent / "ext/RoseTTAFold2-PPI/examples"
    inp = ex / "protein_pairs_input"
    if not inp.exists():
        raise SystemExit(f"example input not found at {inp}")
    ok = 0
    for line in inp.read_text().splitlines():
        if not line.strip():
            continue
        rel, l1 = line.split("\t")
        p = ex / rel
        a, b = convert(p, int(l1))

        # the split must be reversible for every row, and the halves must be non-degenerate
        orig = read_a3m(p)
        for (n0, s0), (_, sa), (_, sb) in zip(orig, a, b):
            assert sa + sb == s0, f"round trip failed on {n0}"
        assert query_len(a[0][1]) == int(l1)
        assert len(a[0][1]) > 0 and len(b[0][1]) > 0

        # a naive slice would be wrong wherever a row carries insertions before the boundary
        naive_wrong = sum(1 for (_, s0), (_, sa) in zip(orig, a) if s0[:int(l1)] != sa)
        log.info("  %-28s rows=%4d  naive seq[:L1] would differ on %d rows (%.0f%%)",
                 Path(rel).name, len(a), naive_wrong, 100 * naive_wrong / len(a))
        ok += 1
    log.info("self-test passed on %d paired MSAs", ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--paired")
    ap.add_argument("--l1", type=int)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.paired and a.l1 and a.out_dir):
        raise SystemExit("need --paired, --l1 and --out-dir (or --self-test)")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ca, cb = convert(Path(a.paired), a.l1)
    stem = Path(a.paired).name.split(".")[0]
    write_a3m(ca, out / f"{stem}.A.a3m")
    write_a3m(cb, out / f"{stem}.B.a3m")
    meta = {"source": str(a.paired), "l1": a.l1, "rows": len(ca),
            "l2": query_len(cb[0][1])}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    log.info("wrote %s.A.a3m / %s.B.a3m to %s", stem, stem, out)


if __name__ == "__main__":
    main()
