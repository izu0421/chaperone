"""Collect arm B (AF3 + omicMSA) scores: confidences plus physical interface terms.

ipTM alone is not the score. Two prior results say so directly -- AF2 ipTM reached only AUC 0.636
on real binding data and a 5.18 nM binder scored 0.21, and on the protein-DNA screen buried
surface area beat ipTM outright as a discriminator. So this collects ipTM and friends AND
measures the interface geometrically from the CIF, and the ranking model is fitted on the
benchmark afterwards instead of assumed.

Physical terms, per pair, from the top-ranked model:
  n_contact      heavy-atom pairs across the chain boundary within 5 A
  n_res_iface    residues with any such contact
  bsa            buried surface area, Shrake-Rupley on each chain alone minus the complex
  iface_plddt    mean pLDDT over interface residues (B-factor column carries pLDDT in AF3 CIFs)
  min_sep        closest heavy-atom approach -- a clash check, because un-relaxed predictions can
                 score a large interface while overlapping (a 114-contact design once had 0.40 A)

A pair whose CIF cannot be parsed is recorded with NaN terms and a reason, not dropped: silently
losing the hard cases is how a screen flatters itself.

Usage:
    python -I scripts/08_score_armB.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("armB")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "armB_out"
CUT = 5.0


def one(d: Path) -> dict:
    name = d.name
    rec: dict = {"pair": name, "note": ""}
    try:
        s = json.loads((d / f"{name}_summary_confidences.json").read_text())
    except Exception as e:
        return {**rec, "note": f"no summary: {str(e)[:40]}"}
    for k in ("iptm", "ptm", "ranking_score", "fraction_disordered", "has_clash"):
        rec[k] = s.get(k)
    cp = s.get("chain_pair_iptm")
    if isinstance(cp, list) and len(cp) == 2:
        rec["iptm_AB"] = cp[0][1]
    cpe = s.get("chain_pair_pae_min")
    if isinstance(cpe, list) and len(cpe) == 2:
        rec["pae_min_AB"] = cpe[0][1]

    cif = d / f"{name}_model.cif"
    if not cif.exists():
        return {**rec, "note": "no cif"}
    try:
        from Bio.PDB import MMCIFParser, ShrakeRupley
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = MMCIFParser(QUIET=True).get_structure(name, str(cif))
        model = next(iter(st))
        chains = [c for c in model]
        if len(chains) != 2:
            return {**rec, "note": f"{len(chains)} chains"}
        A, B = chains
        ca = np.array([a.coord for a in A.get_atoms() if a.element != "H"])
        cb = np.array([a.coord for a in B.get_atoms() if a.element != "H"])
        aa = [a for a in A.get_atoms() if a.element != "H"]
        ab = [a for a in B.get_atoms() if a.element != "H"]
        if len(ca) == 0 or len(cb) == 0:
            return {**rec, "note": "empty chain"}
        from scipy.spatial import cKDTree
        t = cKDTree(cb)
        pairs = t.query_ball_point(ca, CUT)
        rec["n_contact"] = int(sum(len(p) for p in pairs))
        resA = {aa[i].get_parent().id[1] for i, p in enumerate(pairs) if p}
        resB = {ab[j].get_parent().id[1] for p in pairs for j in p}
        rec["n_res_iface"] = len(resA) + len(resB)
        rec["min_sep"] = float(t.query(ca, k=1)[0].min())

        # pLDDT lives in the B-factor column of an AF3 CIF
        pl = [a.get_bfactor() for a in aa if a.get_parent().id[1] in resA]
        pl += [a.get_bfactor() for a in ab if a.get_parent().id[1] in resB]
        rec["iface_plddt"] = float(np.mean(pl)) if pl else np.nan
        rec["mean_plddt"] = float(np.mean([a.get_bfactor() for a in aa + ab]))

        sr = ShrakeRupley()
        sr.compute(model, level="C")
        comp = sum(c.sasa for c in chains)
        sep = 0.0
        for c in chains:
            cc = c.copy()
            sr.compute(cc, level="C")
            sep += cc.sasa
        rec["bsa"] = float(sep - comp)
    except Exception as e:
        rec["note"] = f"cif: {type(e).__name__} {str(e)[:40]}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    dirs = sorted(d for d in OUT.iterdir() if d.is_dir())
    log.info("%d arm B output directories", len(dirs))

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, d) for d in dirs]
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 250 == 0:
                log.info("  %d/%d", done, len(dirs))

    d = pd.DataFrame(rows)
    d[["prot_a", "prot_b"]] = d.pair.str.split("_", n=1, expand=True)
    lab = pd.read_parquet(ROOT / "bench" / "pairs.parquet")
    key = {tuple(sorted((r.prot_a, r.prot_b))): (r.label, r.neg_set)
           for r in lab.itertuples()}
    kk = d.apply(lambda r: key.get(tuple(sorted((r.prot_a, r.prot_b))), (np.nan, None)), axis=1)
    d["label"] = [k[0] for k in kk]
    d["neg_set"] = [k[1] for k in kk]

    d.to_csv(ROOT / "results" / "armB_scores.csv", index=False)
    log.info("wrote %s", ROOT / "results" / "armB_scores.csv")

    bad = d[d.note.ne("")]
    log.info("\n%d pairs with a note:\n%s", len(bad),
             bad.note.value_counts().head(8).to_string() if len(bad) else "  none")
    log.info("\nlabel coverage: %d of %d matched the benchmark", int(d.label.notna().sum()), len(d))
    log.info("\n%s", d.label.value_counts(dropna=False).to_string())

    ok = d[d.label.notna() & d.iptm.notna()]
    if len(ok) and ok.label.nunique() == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score
        log.info("\n--- standalone discrimination on %d labelled pairs (%d positive) ---",
                 len(ok), int(ok.label.sum()))
        for col in ["iptm", "iptm_AB", "ptm", "ranking_score", "n_contact", "n_res_iface",
                    "bsa", "iface_plddt", "mean_plddt"]:
            if col not in ok or ok[col].isna().all():
                continue
            v = ok.dropna(subset=[col])
            # distinct-value fraction: a score saturated at one threshold is not a ranker
            dv = v[col].nunique() / len(v)
            log.info("  %-14s AUROC %.3f  AUPRC %.3f  n=%4d  distinct %.2f",
                     col, roc_auc_score(v.label, v[col]),
                     average_precision_score(v.label, v[col]), len(v), dv)
        log.info("\nclash check: %d of %d have min_sep < 1.5 A",
                 int((ok.min_sep < 1.5).sum()), int(ok.min_sep.notna().sum()))


if __name__ == "__main__":
    main()
