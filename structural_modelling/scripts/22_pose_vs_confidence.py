"""Does WHERE the molecule sits discriminate better than HOW CONFIDENT the fold is?

A permissive pocket is not by itself a problem. AlphaFold3 will place a small
molecule somewhere in a receptor whatever receptor it is given, so confidence
answers the wrong question: it says how sure the model is about a pose, not
whether the pose is the one a ligand would adopt. For a class A GPCR the pose
question has a concrete answer, because the orthosteric site is a cleft inside the
seven-helix bundle. A molecule lining Transmembrane segments is where an agonist
goes. A molecule sitting on the cytoplasmic face, or resting on extracellular
loops, or half-exposed on the lipid-facing outside, is not in the orthosteric
pocket however confidently it was put there.

This measures that on the 256-fold benchmark, where 16 of the cells are a
receptor with its own known ligand and 189 are clean negatives, so pose and
confidence can be compared against the same ground truth:

  buried_fraction      how much of the molecule's own surface the receptor covers
  frac_tm_contacts     share of contacting receptor residues on Transmembrane
                       segments, from UniProt topology, not guessed from geometry
  n_tm_contacts        how many such residues
  touches_cytoplasmic  any contact on a cytoplasmic domain -- for a ligand that
                       enters from outside, this is the wrong side of the membrane
  min_separation       the clash check; an un-relaxed pose can look enclosed while
                       overlapping

Reported as AUROC against the same cognate/negative split used for the confidence
metrics, so the two are directly comparable.

Usage:
    micromamba run -n structmod python -I scripts/22_pose_vs_confidence.py --workers 3
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pose")
ROOT = Path(__file__).resolve().parent.parent
CHAP = Path("/data/yzy21/yy/space_agent/chaperone/src")
STAGES = {
    "specificity": {"out": "specificity_out", "manifest": "specificity_manifest.csv",
                    "scores": "specificity_scores.csv", "acc_col": "receptor_uniprot"},
    # the orphan manifest carries no accession column, so it is joined from the
    # sheet the inputs were built from
    "orphan": {"out": "orphan_out", "manifest": "orphan_manifest.csv",
               "scores": "orphan_scores.csv", "acc_col": None},
}
SHEET = Path("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")

sys.path.insert(0, str(CHAP))


def _load_af():
    """Import chaperone's fold analysis as a PACKAGE member.

    Loading analyze_fold.py by path fails: it does a relative import of its own
    sources.http_retry, which has no parent package when the file is loaded
    standalone. The point of reusing it at all is that the pose is measured by the
    same code the triage agent sees, so a reimplementation here would defeat it.
    """
    from chaperone import analyze_fold as af
    return af


def pose_of(args) -> dict:
    """Geometry and topology for one fold. Topology is passed in, not fetched."""
    job, acc, topo = args
    af = _load_af()
    d = OUT / job
    cif = d / f"{job}_model.cif"
    rec = {"job": job}
    if not cif.exists():
        return {**rec, "note": "no cif"}
    try:
        chains = af.load_chain_atoms(str(cif))
        lig = [n for n, c in chains.items() if c.get("is_ligand")]
        if not lig:
            return {**rec, "note": "no ligand chain"}
        prot = [n for n in chains if n not in lig][0]
        g = af.ligand_geometry(chains, lig[0], {prot: topo})
        if not g:
            return {**rec, "note": "no geometry"}
        pk = (g.get("pocket") or {}).get(prot, {})
        tops = pk.get("topology") or []
        n_res = pk.get("n_contacting_residues") or 0
        tm = sum(1 for t in tops if "Transmembrane" in str(t) or "Intramembrane" in str(t))
        cyt = any("Cytoplasm" in str(t) for t in tops)
        ext = any("Extracellular" in str(t) for t in tops)
        return {**rec, "note": "",
                "buried_fraction": g.get("buried_fraction"),
                "ligand_sasa_alone": g.get("sasa_alone_A2"),
                "min_separation": g.get("min_separation_A"),
                "clash": g.get("clash"),
                "n_contacting_residues": n_res,
                "n_topology_terms": len(tops),
                "n_tm_terms": tm,
                "frac_tm_terms": tm / len(tops) if tops else np.nan,
                "touches_cytoplasmic": cyt,
                "touches_extracellular": ext,
                "topology": ";".join(sorted({str(t) for t in tops}))}
    except Exception as exc:  # noqa: BLE001
        return {**rec, "note": f"{type(exc).__name__} {str(exc)[:60]}"}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0, 0
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    a = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(a), len(pos), len(neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stage", default="specificity", choices=sorted(STAGES))
    ap.add_argument("--kind", default=None,
                    help="restrict to one candidate kind, e.g. metabolite")
    a = ap.parse_args()
    cfg = STAGES[a.stage]
    global OUT, CACHE
    OUT = ROOT / "results" / cfg["out"]
    CACHE = ROOT / "results" / f"{a.stage}_topology.json"

    scores = pd.read_csv(ROOT / "results" / cfg["scores"])
    man = pd.read_csv(ROOT / "work" / cfg["manifest"])
    if a.kind and "kind" in man.columns:
        man = man[man.kind.eq(a.kind)]
        log.info("restricted to kind=%s: %d folds", a.kind, len(man))
    if cfg["acc_col"]:
        acc = man.drop_duplicates("receptor").set_index("receptor")[cfg["acc_col"]].to_dict()
    else:
        sh = pd.read_csv(SHEET).drop_duplicates("receptor_gene")
        acc = sh.set_index("receptor_gene").receptor_uniprot.to_dict()
        acc = {g: acc[g] for g in man.receptor.unique() if g in acc}
    log.info("%d receptors to fetch topology for", len(acc))

    # UniProt topology once per receptor, cached: 256 folds share 16 receptors.
    #
    # The cache is REPAIRED, not just loaded. The previous version refused to
    # write only when EVERY receptor came back empty, and then never retried on
    # a cache hit -- so a partial failure was permanent and invisible. It
    # happened: 18 of the 75 orphan receptors (GPR3, GPR4, GPR35, GPR65,
    # GPR88, GPR132, MAS1L and 11 more) sat in this file with zero terms while
    # UniProt has full seven-helix annotation for all of them, and anything
    # reading the cache concluded "no transmembrane annotation" for a class A
    # GPCR. An all-or-nothing guard protects against total failure and lets
    # partial failure straight through, which is the commoner case.
    topo = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if topo:
        log.info("topology for %d receptors from cache", len(topo))
    todo = {g: a for g, a in acc.items() if not topo.get(g)}
    if todo:
        log.info("re-fetching %d receptors with no cached topology: %s",
                 len(todo), sorted(todo))
        af = _load_af()
        for gene, ac in todo.items():
            feats = af.fetch_uniprot_features(ac) if ac else []
            topo[gene] = af._filter_features(feats, af.TOPOLOGY_FEATURE_TYPES)
            log.info("  %s (%s): %d topology features, %d transmembrane", gene, ac,
                     len(topo[gene]),
                     sum(1 for t in topo[gene] if "ransmembrane" in str(t)))

        still_empty = sorted(g for g in acc if not topo.get(g))
        # A per-receptor check, not an any(). These are all seven-pass receptors,
        # so an empty result is a failed call and not a biological fact, and the
        # fraction that may be genuinely unannotated is small.
        if len(still_empty) > 0.1 * len(acc):
            raise SystemExit(
                f"{len(still_empty)} of {len(acc)} receptors have no topology after a "
                f"re-fetch: {still_empty[:12]}. These are seven-pass receptors, so that "
                f"is a failed fetch and not an annotation gap; refusing to cache it, "
                f"because downstream it is indistinguishable from 'the ligand never "
                f"touches a transmembrane segment'.")
        if still_empty:
            log.warning("%d receptors still have no topology and are cached as empty: %s",
                        len(still_empty), still_empty)
        CACHE.write_text(json.dumps(topo, indent=1))
        log.info("cache repaired: %d receptors, %d with topology",
                 len(topo), sum(1 for v in topo.values() if v))

    jobs = [(r.job, acc.get(r.receptor, ""), topo.get(r.receptor, []))
            for r in man.itertuples()]
    log.info("measuring the pose of %d folds", len(jobs))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(pose_of, jobs))
    keep = [c for c in ("pair", "label", "kind", "receptor", "partner", "iptm",
                        "plddt_B", "pae_min_AB", "bsa") if c in scores.columns]
    p = pd.DataFrame(rows).merge(scores[keep], left_on="job", right_on="pair", how="left")
    if "label" not in p.columns:
        p["label"] = p.get("kind", "candidate")
    bad = p[p.note.fillna("").ne("")]
    if len(bad):
        log.warning("%d folds could not be measured: %s", len(bad),
                    bad.note.value_counts().head(3).to_dict())
    p.to_csv(ROOT / "results" / f"{a.stage}_pose.csv", index=False)

    if not p.label.isin(["cognate", "negative"]).any():
        log.info("\nno cognate/negative labels on this stage, so no AUROC: reporting the "
                 "pose of every fold instead")
        log.info("\ntouches a cytoplasmic domain: %d of %d folds (%.1f%%)",
                 int(p.touches_cytoplasmic.fillna(False).sum()), len(p),
                 100 * p.touches_cytoplasmic.fillna(False).mean())
        log.info("\n%s", p[["buried_fraction", "n_contacting_residues",
                            "min_separation"]].describe().round(3).to_string())
        p.to_csv(ROOT / "results" / f"{a.stage}_pose.csv", index=False)
        log.info("\nwrote results/%s_pose.csv", a.stage)
        return 0

    cog, neg = p[p.label.eq("cognate")], p[p.label.eq("negative")]
    log.info("\nmedian pose by label:")
    cols = ["buried_fraction", "frac_tm_terms", "n_tm_terms", "n_contacting_residues",
            "min_separation"]
    log.info("\n%s", p.groupby("label")[cols].median().round(3).to_string())
    log.info("\ntouches a cytoplasmic domain:\n%s",
             p.groupby("label").touches_cytoplasmic.mean().round(3).to_string())

    log.info("\nAUROC, cognate vs clean negative -- POSE against CONFIDENCE:")
    rows = []
    for m, hi in (("iptm", True), ("plddt_B", True), ("pae_min_AB", False),
                  ("buried_fraction", True), ("frac_tm_terms", True),
                  ("n_tm_terms", True), ("n_contacting_residues", True),
                  ("min_separation", True)):
        v_c = cog[m] if hi else -cog[m]
        v_n = neg[m] if hi else -neg[m]
        au, npos, nneg = auroc(v_c, v_n)
        rows.append({"metric": m, "kind": "pose" if m in
                     ("buried_fraction", "frac_tm_terms", "n_tm_terms",
                      "n_contacting_residues", "min_separation") else "confidence",
                     "auroc": round(au, 3) if au == au else None,
                     "n_pos": npos, "n_neg": nneg})
    t = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    log.info("\n%s", t.to_string(index=False))
    t.to_csv(ROOT / "results" / f"{a.stage}_pose_auroc.csv", index=False)
    log.info("\nwrote results/%s_pose.csv and %s_pose_auroc.csv", a.stage, a.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
