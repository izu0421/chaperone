"""Build RF2-PPI paired MSAs for pairs the authors did not supply, and prove the
builder first by reproducing arm D's own numbers.

Why this exists: AlphaFold3 scores at chance on the HuRI pilot (interface PAE
AUROC 0.519 against 0.840 on the published benchmark), and two explanations
survive — the published benchmark's positives are 9.6-fold better studied than
its negatives, leaving 57 effective negatives of 2,000; or HuRI's yeast
two-hybrid labels are too noisy to score anything against. Running RF2-PPI on
the same 600 pairs separates them. If it also collapses, the labels are the
limit. If it separates where AlphaFold3 does not, the engine comparison rests
on a benchmark whose signal is study bias.

Arm D ran on paired MSAs shipped with the RoseTTAFold2-PPI repo. HuRI has none,
so they have to be built, in THEIR format and not ours:

    their rule   masked (low-quality) query positions are DELETED, so the query
                 is shorter than the full protein
    our rule     masked positions are kept as gaps, so the query stays full
                 length, which is what AlphaFold3 needs

07_build_af3_inputs.py carries a validate() that reproduces the authors' own
paired MSAs under their rule, so the rule itself is already checked. What is
not checked is this file's assembly of it, and a format error here would make
RF2-PPI score noise — which, on a question whose interesting answer is a null,
would be indistinguishable from the result.

So `--validate` builds the MSAs for pairs arm D already scored, runs RF2-PPI on
them, and compares against results/armD_scores_labelled.csv. The model is
non-deterministic (the authors' README reports sd > 0.1 for about 5% of cases),
so the check is on rank agreement and a tolerance, not on equality.

Usage:
    micromamba run -n structmod python -I scripts/29_build_armD_msas.py --validate 24
    micromamba run -n structmod python -I scripts/29_build_armD_msas.py --huri
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("armD-build")
ROOT = Path(__file__).resolve().parent.parent
SIF = ROOT / "ext" / "SE3nv-20230612.sif"
REPO = ROOT / "ext" / "RoseTTAFold2-PPI"

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)


def build_one(acc_a: str, acc_b: str, out_dir: Path) -> int | None:
    """One paired a3m in RF2-PPI's format. Returns L1, the chain-A query length."""
    mA, qA, hA = b7.read_omic(acc_a)
    mB, qB, hB = b7.read_omic(acc_b)
    keepA = [i for i, c in enumerate(mA) if c == "*"]
    keepB = [i for i, c in enumerate(mB) if c == "*"]
    if not keepA or not keepB:
        return None

    def their_rule(seq: str, keep: list[int]) -> str:
        """Delete masked query positions, carrying lowercase insertions along.

        A raw slice by character index is wrong: lowercase letters are
        insertions and do not consume a query position. This walks query
        positions, which is the same traversal b7.gap_masked does, but dropping
        instead of gapping.
        """
        out, qpos = [], 0
        keepset = set(keep)
        for ch in seq:
            if ch.islower():
                out.append(ch)            # an insertion rides with its neighbour
                continue
            if qpos in keepset:
                out.append(ch)
            qpos += 1
        return "".join(out)

    shared = sorted(set(hA) & set(hB))
    qa, qb = their_rule(qA, keepA), their_rule(qB, keepB)
    rows = [("query", qa + qb)]
    for acc in shared:
        rows.append((acc, their_rule(hA[acc], keepA) + their_rule(hB[acc], keepB)))

    # every row must have the same number of query positions as the query, or
    # RF2-PPI reads a frame-shifted alignment and returns a number anyway
    want = sum(1 for c in rows[0][1] if not c.islower())
    bad = [n for n, s in rows if sum(1 for c in s if not c.islower()) != want]
    if bad:
        raise AssertionError(f"{acc_a}_{acc_b}: {len(bad)} rows disagree with the query "
                             f"on query-position count, e.g. {bad[:3]}")

    # hhfilter at 90% identity. The ".i90" in the authors' own filenames IS this
    # step, and skipping it is not a cosmetic difference: the raw paired
    # alignment runs 3,298 rows against their 102 for the same pair, which made
    # RF2-PPI grind and would have been a different input from the one arm D was
    # measured on. Caught because the validation run produced no output in nine
    # minutes and the files were megabytes.
    stem = f"{acc_a}_{acc_b}"
    dest = out_dir / f"{stem}.i90.a3m"
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.a3m"
        src.write_text("".join(f">{n}\n{s}\n" for n, s in rows))
        r = subprocess.run([str(b7.HHFILTER), "-i", str(src), "-o", str(dest),
                            "-id", "90"], capture_output=True, text=True)
        if r.returncode != 0 or not dest.exists():
            raise RuntimeError(f"hhfilter failed for {stem}: {r.stderr[-200:]}")

    # hhfilter rewrites the file, so re-read and re-check rather than trusting it:
    # the query must survive as the first record and the frame must still hold
    kept = dest.read_text().splitlines()
    if not kept or not kept[0].startswith(">"):
        raise RuntimeError(f"{stem}: hhfilter output does not start with a record")
    if sum(1 for c in kept[1] if not c.islower()) != want:
        raise RuntimeError(f"{stem}: hhfilter changed the query's query-position count")
    return sum(1 for c in qa if not c.islower())


def run_rf2(work: Path, listname: str) -> Path:
    cmd = ["singularity", "exec", "--nv",
           "--bind", f"{work}:/work/users",
           "--bind", f"{REPO}:/home/RoseTTAFold2-PPI",
           str(SIF), "/bin/bash", "-c",
           f"cd /work/users && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py "
           f"-list_fn {listname} -model_file /home/RoseTTAFold2-PPI/src/models/RF2-PPI.pt"]
    log.info("running RF2-PPI over %s", listname)
    subprocess.run(cmd, check=True)
    return work / f"{listname}.log"


def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                rows.append({"key": parts[0].replace(".i90.a3m", ""),
                             "prob": float(parts[1])})
            except ValueError:
                continue
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0,
                    help="rebuild this many pairs arm D already scored and compare")
    ap.add_argument("--huri", action="store_true", help="build and score the HuRI pilot")
    ap.add_argument("--chemokine", action="store_true",
                    help="RF2-PPI on the 256-fold chemokine panel, so the two engines are "
                         "compared on the same GPCR complexes where AlphaFold3 reaches 0.843")
    ap.add_argument("--build-only", action="store_true",
                    help="build the MSAs and stop. hhfilter is CPU work, so it can run "
                         "while a GPU job finishes; the inference then starts immediately")
    a = ap.parse_args()

    if a.validate:
        known = pd.read_csv(ROOT / "results" / "armD_scores_labelled.csv")
        # a mix of both classes, so a builder that destroys the signal shows up
        sub = pd.concat([known[known.cat.eq("positive")].head(a.validate // 2),
                         known[known.cat.eq("negative")].head(a.validate // 2)])
        work = ROOT / "work" / "armD_validate"
        work.mkdir(parents=True, exist_ok=True)
        entries = []
        for r in sub.itertuples():
            l1 = build_one(r.a, r.b, work)
            if l1:
                entries.append(f"{r.a}_{r.b}.i90.a3m\t{l1}")
        (work / "validate_input").write_text("\n".join(entries) + "\n")
        log.info("built %d of %d pairs", len(entries), len(sub))
        got = parse_log(run_rf2(work, "validate_input"))
        m = sub.assign(key=sub.a + "_" + sub.b).merge(got, on="key", suffixes=("_known", "_new"))
        m["delta"] = (m.prob_new - m.prob_known).abs()
        rho = m.prob_known.corr(m.prob_new, method="spearman")
        log.info("\n%s", m[["key", "cat", "prob_known", "prob_new", "delta"]].round(4).to_string(index=False))
        log.info("\n%d pairs compared, Spearman rho %.3f, median |delta| %.4f, max %.4f",
                 len(m), rho, m.delta.median(), m.delta.max())
        if rho < 0.9:
            raise SystemExit(
                f"rank agreement with arm D is only {rho:.3f}. The builder does not "
                f"reproduce the authors' own paired MSAs, so nothing it produces for "
                f"HuRI would be interpretable. Not proceeding.")
        log.info("builder reproduces arm D; safe to use on pairs they did not supply")
        return

    if a.huri:
        man = pd.read_csv(ROOT / "work" / "huri_pilot_manifest.csv")
        work = ROOT / "work" / "armD_huri"
        work.mkdir(parents=True, exist_ok=True)
        # Reuse MSAs already built, so --build-only followed by a scoring run does
        # not redo twelve minutes of hhfilter. The list file carries L1 per pair,
        # so it is the thing to reuse, not just the a3m files.
        existing = {}
        lf = work / "huri_input"
        if lf.exists():
            for line in lf.read_text().splitlines():
                if "\t" in line:
                    stem, l1 = line.split("\t")
                    if (work / stem).exists():
                        existing[stem.replace(".i90.a3m", "")] = int(l1)
            log.info("reusing %d MSAs already built", len(existing))

        entries, skipped = [], []
        for r in man.itertuples():
            try:
                key = f"{r.a}_{r.b}"
                l1 = existing.get(key)
                if l1 is None:
                    l1 = build_one(r.a, r.b, work)
            except Exception as exc:                      # noqa: BLE001
                skipped.append((f"{r.a}_{r.b}", str(exc)[:80])); continue
            if l1:
                entries.append(f"{r.a}_{r.b}.i90.a3m\t{l1}")
            else:
                skipped.append((f"{r.a}_{r.b}", "no usable mask"))
        (work / "huri_input").write_text("\n".join(entries) + "\n")
        log.info("built %d of %d pairs, skipped %d", len(entries), len(man), len(skipped))
        for k, why in skipped[:10]:
            log.warning("  skipped %s: %s", k, why)
        if len(entries) < 0.9 * len(man):
            raise SystemExit(f"only {len(entries)} of {len(man)} pairs built; a pilot "
                             f"missing that many is a different sample from the one "
                             f"AlphaFold3 was scored on")
        if a.build_only:
            depths = [sum(1 for l in (work / e.split('\t')[0]).read_text().splitlines()
                          if l.startswith('>')) for e in entries[:200]]
            import statistics
            log.info("built only. median paired depth after hhfilter: %d rows "
                     "(the authors' supplied files run ~100-700)",
                     statistics.median(depths))
            return
        got = parse_log(run_rf2(work, "huri_input"))
        out = man.assign(key=man.a + "_" + man.b).merge(got, on="key", how="left")
        dest = ROOT / "results" / "armD_huri_scores.csv"
        out.to_csv(dest, index=False)
        log.info("scored %d of %d; wrote %s", int(out.prob.notna().sum()), len(out), dest)
        return

    if a.chemokine:
        man = pd.read_csv(ROOT / "work" / "protein_counterscreen_manifest.csv")
        man = man[man.block.eq("A_chemokine")]
        PANEL = {"CCR1":"P32246","CCR2":"P41597","CCR3":"P51677","CCR5":"P51681",
                 "CCR6":"P51684","CCR7":"P32248","CXCR1":"P25024","CXCR2":"P25025",
                 "CXCR3":"P49682","CXCR4":"P61073","CX3CR1":"P49238","ACKR3":"P25106",
                 "CCR4":"P51679","CCR8":"P51685","CXCR5":"P32302","CXCR6":"O00574",
                 "CCL2":"P13500","CCL3":"P10147","CCL5":"P13501","CCL11":"P51671",
                 "CCL19":"Q99731","CCL20":"P78556","CXCL8":"P10145","CXCL10":"P02778",
                 "CXCL12":"P48061","CX3CL1":"P78423","CCL21":"O00585","CXCL11":"O14625",
                 "CCL17":"Q92583","CCL1":"P22362","CXCL13":"O43927","CXCL16":"Q9H2A7"}
        work = ROOT / "work" / "armD_chemokine"
        work.mkdir(parents=True, exist_ok=True)
        lf = work / "chemokine_input"
        existing = {}
        if lf.exists():
            for line in lf.read_text().splitlines():
                if "\t" in line:
                    stem, l1 = line.split("\t")
                    if (work / stem).exists():
                        existing[stem.replace(".i90.a3m", "")] = int(l1)
            log.info("reusing %d MSAs already built", len(existing))
        entries, skipped = [], []
        for r in man.itertuples():
            a_, b_ = PANEL.get(r.chain_a), PANEL.get(r.chain_b)
            if not a_ or not b_:
                skipped.append((r.job, "no accession")); continue
            key = f"{a_}_{b_}"
            l1 = existing.get(key)
            if l1 is None:
                try:
                    l1 = build_one(a_, b_, work)
                except Exception as exc:                      # noqa: BLE001
                    skipped.append((r.job, str(exc)[:70])); continue
            if l1:
                entries.append(f"{key}.i90.a3m\t{l1}")
            else:
                skipped.append((r.job, "no usable mask"))
        (lf).write_text("\n".join(entries) + "\n")
        log.info("built %d of %d cells, skipped %d", len(entries), len(man), len(skipped))
        for k, why in skipped[:8]:
            log.warning("  skipped %s: %s", k, why)
        if len(entries) < 0.9 * len(man):
            raise SystemExit(f"only {len(entries)} of {len(man)} built; a panel missing "
                             f"that many is not the panel AlphaFold3 was scored on")
        got = parse_log(run_rf2(work, "chemokine_input"))
        out = man.assign(key=man.chain_a.map(PANEL) + "_" + man.chain_b.map(PANEL)) \
                 .merge(got, on="key", how="left")
        dest = ROOT / "results" / "armD_chemokine_scores.csv"
        out.to_csv(dest, index=False)
        log.info("scored %d of %d; wrote %s", int(out.prob.notna().sum()), len(out), dest)
        return

    ap.error("pass --validate N, --huri or --chemokine")


if __name__ == "__main__":
    main()
