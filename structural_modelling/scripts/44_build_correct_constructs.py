"""Re-fold the screen with the molecule that exists, for every pair.

A census of the 4,854 protein folds against UniProt says 83% of them were the
wrong molecule:

    2,611  partner is a membrane protein folded whole, tail and all, when only
           its ectodomain can reach the receptor's extracellular face
    1,404  partner is secreted and was folded as its precursor, signal peptide
           attached, which is cleaved before the ligand ever meets a receptor
      702  partner is a membrane protein whose ectodomain is under 100 aa, so
           truncation is not available and it is folded whole
      137  partner is soluble with no signal peptide: already correct

Every construct test run on the shortlist attacked one of those two faults after
the fact, one candidate at a time. Folding the right chain from the start
removes them before the gate instead of after it, which is the only way the gate
threshold becomes the thing that limits the screen.

ONE ALIGNMENT CONVENTION THROUGHOUT. The original screen used unfiltered
omicMSAs and every re-fold arm used hhfilter at 90%, a median 7.5x shallower,
which made every construct comparison a comparison of two things at once. Here
the alignment is the same for all 4,854, so a difference is the construct.

Usage:
    micromamba run -n structmod python -I scripts/44_build_correct_constructs.py --workers 4
    micromamba run -n structmod python -I scripts/44_build_correct_constructs.py --verify 12
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("v2")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "v2_inputs"
MAN = ROOT / "work" / "v2_manifest.csv"
UNI = ROOT / "work" / "uniprot_constructs.json"
SHEET = Path("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
SEEDS = [1]
MIN_ECD = 100

_s37 = iu.spec_from_file_location("b37", ROOT / "scripts" / "37_build_ectodomain_arm.py")
b37 = iu.module_from_spec(_s37)
_s37.loader.exec_module(b37)
b7 = b37.b7


def plan(u: dict) -> tuple[str, tuple[int, int] | None]:
    """Which construct this partner needs, and the span to keep."""
    if not u:
        return "unknown", None
    if u["n_tm"] == 0:
        if u["signal"]:
            return "mature", (u["signal"] + 1, u["len"])
        return "asis", None
    if u["ecd"] >= MIN_ECD:
        return "ectodomain", None          # span resolved from topology at build time
    return "asis_short_ecd", None


def build_one(args):
    rec_gene, par_gene, rec_acc, par_acc, kind = args
    name = f"V2_{rec_gene}__{par_gene}"
    dest = OUT / f"{name}.json"
    if dest.exists():
        return {"job": name, "status": "cached"}
    try:
        qA, qB, rA, rB = b7.build_pair(rec_acc, par_acc)
        kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
    except Exception as exc:                                   # noqa: BLE001
        return {"job": name, "status": f"msa failed: {str(exc)[:60]}"}
    msaA = b7.a3m([("query", qA)] + kA)
    msaB = b7.a3m([("query", qB)] + kB)
    span = "full"
    if kind == "mature":
        seg = None
        try:
            from json import loads
            u = loads(UNI.read_text())[par_acc]
            seg = (u["signal"] + 1, len(qB))
        except Exception:                                      # noqa: BLE001
            pass
        if seg and seg[0] < seg[1]:
            qB, msaB = b37.slice_query(qB, msaB, seg[0], seg[1])
            span = f"{seg[0]}-{seg[1]}"
    elif kind == "ectodomain":
        seg = b37.uniprot_ecd(par_acc)
        if seg:
            qB, msaB = b37.slice_query(qB, msaB, seg[0], seg[1])
            span = f"{seg[0]}-{seg[1]}"
        else:
            kind = "asis_no_topology"
    if not qB:
        return {"job": name, "status": "empty partner chain after cutting"}
    chain = lambda cid, s, m: {"protein": {"id": cid, "sequence": s, "pairedMsa": m,
                                           "unpairedMsa": m, "templates": []}}
    dest.write_text(json.dumps({"name": name, "modelSeeds": SEEDS,
                                "dialect": "alphafold3", "version": 1,
                                "sequences": [chain("A", qA, msaA), chain("B", qB, msaB)]}))
    return {"job": name, "status": "built", "receptor": rec_gene, "partner": par_gene,
            "construct": kind, "span": span, "size": len(qA) + len(qB),
            "n_rows": len(kA) + 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0,
                    help="check N built inputs against UniProt instead of building")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    U = json.loads(UNI.read_text())

    if a.verify:
        return verify(a.verify, U)

    sheet = pd.read_csv(SHEET)
    sheet = sheet[sheet.partner_kind.eq("protein")] if "partner_kind" in sheet else sheet
    jobs = []
    for r in sheet.itertuples():
        pa, ra = str(r.partner_uniprot), str(r.receptor_uniprot)
        if pa not in U or ra not in U:
            continue
        kind, _ = plan(U[pa])
        jobs.append((r.receptor_gene, r.partner_gene, ra, pa, kind))
    if a.limit:
        jobs = jobs[:a.limit]
    log.info("%d pairs to build; constructs: %s", len(jobs),
             pd.Series([j[4] for j in jobs]).value_counts().to_dict())

    rows, bad = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(build_one, jobs, chunksize=8), 1):
            if res["status"] == "built":
                rows.append(res)
            elif res["status"] != "cached":
                bad.append(res)
            if i % 250 == 0:
                log.info("  %d/%d built, %d failed", i, len(jobs), len(bad))
    if bad:
        log.warning("%d failed, e.g. %s", len(bad), bad[:3])
    if rows:
        old = pd.read_csv(MAN).to_dict("records") if MAN.exists() else []
        man = pd.DataFrame({r["job"]: r for r in (old + rows)}.values())
        man = man.sort_values("size").reset_index(drop=True)
        man["fold_order"] = man.index + 1
        tmp = MAN.with_suffix(".csv.tmp"); man.to_csv(tmp, index=False); os.replace(tmp, MAN)
        log.info("\n%d inputs, median %d tokens", len(man), int(man["size"].median()))
        log.info("%s", man.groupby("construct").agg(
            jobs=("job", "size"), median_tokens=("size", "median")).to_string())


def verify(n: int, U: dict) -> None:
    """Every cut chain must equal the UniProt span, and every alignment row must
    still be in the query frame. This check has caught a fault every time it has
    been run, so it runs before any folding."""
    import urllib.request
    man = pd.read_csv(MAN)
    sheet = pd.read_csv(SHEET)
    acc = dict(zip(sheet.partner_gene, sheet.partner_uniprot))
    sub = man[man.construct.isin(["mature", "ectodomain"])].sample(
        min(n, int(man.construct.isin(["mature", "ectodomain"]).sum())), random_state=0)
    ok = 0
    for r in sub.itertuples():
        js = json.loads((OUT / f"{r.job}.json").read_text())
        qB = js["sequences"][1]["protein"]["sequence"]
        url = f"https://rest.uniprot.org/uniprotkb/{acc[r.partner]}.json?fields=sequence"
        full = json.loads(urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "research-script/1.0"}), timeout=30).read()
            )["sequence"]["value"]
        s, e = (int(x) for x in str(r.span).split("-"))
        rows = [l for l in js["sequences"][1]["protein"]["pairedMsa"].splitlines()
                if not l.startswith(">")]
        frames = {sum(1 for c in x if not c.islower()) for x in rows}
        good = qB == full[s - 1:e] and frames == {len(qB)}
        ok += good
        log.info("  %-28s %-11s %s %4d aa  seq %s  frame %s", r.job, r.construct,
                 r.span, len(qB), "OK" if qB == full[s - 1:e] else "MISMATCH",
                 "OK" if frames == {len(qB)} else f"{sorted(frames)[:3]}")
    log.info("\n%d of %d fully verified", ok, len(sub))


if __name__ == "__main__":
    main()
