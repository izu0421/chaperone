"""Score the three constructs the triage asked for, each against its own control.

  M_mature     every secreted partner re-folded as the chain that exists, with
               the signal peptide removed.
  N_mature_*   the SAME cut on 24 chemokine pairs whose answer is known. This is
               what makes arm M readable: if removing 21-26 residues collapses
               known binders too, a collapsed candidate means nothing.
  S_subcomplex the four LIKELY_SUBCOMPLEX candidates folded with the subunits the
               triage named as missing. If the receptor interface survives the
               complete complex, the verdict was wrong; if it collapses, the
               receptor was docking to a face that is buried in the real assembly.

`one()` from the arm-B scorer refuses anything but two chains, so the A-B
interface here is computed locally and checked against it on the two-chain jobs.

Usage:
    micromamba run -n structmod python -I scripts/35_score_subcomplex_and_mature.py
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sub-score")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "subcomplex_out"
CUT = 5.0
GATE = 3.75          # replaced at run time by results/protein_thresholds.json

_spec = iu.spec_from_file_location("sc08", ROOT / "scripts" / "08_score_armB.py")
sc08 = iu.module_from_spec(_spec)
sys.modules["sc08"] = sc08
_spec.loader.exec_module(sc08)


def score_multi(d: Path) -> dict:
    """A-B interface and per-chain confidence, for any number of chains.

    Chain A is the receptor and chain B the named partner, by construction in
    scripts 33 and 34; C onwards are the subunits added to complete an assembly.
    """
    rec: dict = {"job": d.name, "note": ""}
    f = sorted(d.glob("*_summary_confidences.json"))
    if len(f) != 1:
        return {**rec, "note": f"{len(f)} summaries"}
    s = json.loads(f[0].read_text())
    rec["iptm"] = s.get("iptm")
    rec["ptm"] = s.get("ptm")
    rec["has_clash"] = s.get("has_clash")
    cpe, cpi = s.get("chain_pair_pae_min"), s.get("chain_pair_iptm")
    rec["n_chain"] = len(cpe) if isinstance(cpe, list) else np.nan
    if isinstance(cpe, list) and len(cpe) >= 2:
        rec["pae_min_AB"] = cpe[0][1]
    if isinstance(cpi, list) and len(cpi) >= 2:
        rec["iptm_AB"] = cpi[0][1]
    cif = sorted(d.glob("*_model.cif"))
    if not cif:
        return {**rec, "note": "no cif"}
    try:
        from Bio.PDB import MMCIFParser
        from scipy.spatial import cKDTree
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = MMCIFParser(QUIET=True).get_structure(d.name, str(cif[0]))
        chains = list(next(iter(st)))
        if len(chains) < 2:
            return {**rec, "note": f"{len(chains)} chains"}
        aa = [a for a in chains[0].get_atoms() if a.element != "H"]
        ab = [a for a in chains[1].get_atoms() if a.element != "H"]
        ca = np.array([a.coord for a in aa])
        cb = np.array([a.coord for a in ab])
        t = cKDTree(cb)
        pairs = t.query_ball_point(ca, CUT)
        rec["n_contact"] = int(sum(len(p) for p in pairs))
        resA = {aa[i].get_parent().id[1] for i, p in enumerate(pairs) if p}
        resB = {ab[j].get_parent().id[1] for p in pairs for j in p}
        rec["n_res_iface"] = len(resA) + len(resB)
        rec["min_sep"] = float(t.query(ca, k=1)[0].min())
        pl = [a.get_bfactor() for a in aa if a.get_parent().id[1] in resA]
        pl += [a.get_bfactor() for a in ab if a.get_parent().id[1] in resB]
        rec["iface_plddt"] = float(np.mean(pl)) if pl else np.nan
        rec["plddt_A"] = float(np.mean([a.get_bfactor() for a in aa]))
        rec["plddt_B"] = float(np.mean([a.get_bfactor() for a in ab]))
        # how much of the partner is in the interface at all
        rec["frac_B_iface"] = len(resB) / max(1, len({a.get_parent().id[1] for a in ab}))
    except Exception as exc:                                    # noqa: BLE001
        rec["note"] = f"cif: {str(exc)[:40]}"
    return rec


def seed_dirs(job_dir: Path) -> list[Path]:
    subs = sorted(p for p in job_dir.glob("seed-*") if p.is_dir())
    return subs or ([job_dir] if list(job_dir.glob("*_summary_confidences.json")) else [])


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    # two manifests, one output directory: the controls fold on their own GPU so
    # they do not sit behind the 2,600-token subcomplexes in one queue.
    man = pd.concat([pd.read_csv(ROOT / "work" / f) for f in
                     ("subcomplex_manifest.csv", "mature_control_manifest.csv")
                     if (ROOT / "work" / f).exists()], ignore_index=True)
    arm = dict(zip(man.job, man.arm))
    global GATE
    GATE = json.loads((ROOT / "results" / "protein_thresholds.json").read_text()
                      )["gate"]["pae_min_AB_max"]
    targets, meta = [], []
    for job in man.job:
        jd = OUT / job
        if not jd.is_dir():
            continue
        for sd in seed_dirs(jd):
            targets.append(sd); meta.append((job, arm[job]))
    log.info("%d of %d jobs have output; scoring %d models", len({m[0] for m in meta}),
             len(man), len(targets))
    if not targets:
        raise SystemExit("nothing scored yet")

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(score_multi, targets))
    df = pd.DataFrame(rows)
    df["job"] = [m[0] for m in meta]
    df["arm"] = [m[1] for m in meta]
    df["model"] = [t.name for t in targets]
    df = df[df.iptm.notna()].copy()

    # the local interface must agree with the published scorer where both apply
    two = df[df.n_chain.eq(2)].head(3)
    for r in two.itertuples():
        ref = sc08.one(OUT / r.job / r.model if (OUT / r.job / r.model).is_dir()
                       else OUT / r.job)
        if pd.notna(ref.get("pae_min_AB")) and abs(ref["pae_min_AB"] - r.pae_min_AB) > 1e-6:
            log.warning("  scorer disagreement on %s: %.3f vs %.3f",
                        r.job, ref["pae_min_AB"], r.pae_min_AB)
    dest = ROOT / "results" / "subcomplex_scores.csv"
    df.to_csv(dest, index=False)
    log.info("wrote %s (%d rows)", dest, len(df))

    # Per job, the best model by interface PAE -- but only over SEED 1. Nine of
    # the mature jobs were folded before the arm was corrected to one seed and
    # carry 25 models against the other seventeen's five, and the controls' five.
    # "Best of" is a maximum over draws, so pooling them would hand those nine
    # more chances to clear the bar than anything they are compared with: their
    # collapse rate is 2 of 9 against 9 of 17 for the one-seed jobs. Restricting
    # every arm to seed 1 makes the draws equal; the extra models stay in
    # subcomplex_scores.csv for anyone who wants them.
    seed1 = df[df.model.str.contains("seed-1_") | df.model.eq(df.job)]
    dropped = df.job.nunique() - seed1.job.nunique()
    if dropped:
        log.warning("%d jobs have no seed-1 model and are excluded", dropped)
    log.info("draws per job after matching: %s",
             seed1.groupby("job").size().value_counts().to_dict())
    best = seed1.sort_values("pae_min_AB").groupby(["job", "arm"], as_index=False).first()
    best["receptor"] = best.job.map(dict(zip(man.job, man.receptor)))
    best["partner"] = best.job.map(dict(zip(man.job, man.partner)))

    # the screen's own value for the same pair: full preprohormone, two chains
    scr = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    scr = scr[scr.kind.eq("protein")] if "kind" in scr else scr
    full = {}
    for r in scr.itertuples():
        full.setdefault((str(r.receptor), str(r.partner)), []).append(
            (r.pae_min_AB, r.plddt_A, r.plddt_B))
    def full_of(rec, par, i=0):
        v = full.get((rec, par)) or full.get((par, rec))
        return v[0][i] if v else np.nan
    best["pae_full"] = [full_of(r.receptor, r.partner, 0) for r in best.itertuples()]
    best["pldA_full"] = [full_of(r.receptor, r.partner, 1) for r in best.itertuples()]
    best["pldB_full"] = [full_of(r.receptor, r.partner, 2) for r in best.itertuples()]
    # A chain's confidence is only interpretable against its own full-length fold:
    # a GPCR's mean pLDDT is diluted by its disordered tails, so an absolute 60
    # flags half the screen as broken. The question is whether the CUT broke it.
    best["d_pldA"] = best.plddt_A - best.pldA_full
    best["d_pldB"] = best.plddt_B - best.pldB_full

    # ---- the control: does cutting the signal peptide break known answers? ------
    N = best[best.arm.str.startswith("N_mature")].copy()
    # The control pairs are chemokine receptor + chemokine, which the orphan
    # screen never folded, so their full-length value lives in the rescue arm's
    # own table (arm C, one seed -- the same single draw the screen took). Taking
    # it from orphan_scores.csv returns NaN for every row and the comparison
    # silently disappears.
    rp = ROOT / "results" / "rescue_per_job.csv"
    if rp.exists() and len(N):
        r = pd.read_csv(rp)
        r = r[r.arm.str.startswith("C_control")]
        key = {j.replace("C_binder_", "").replace("C_negative_", ""): v
               for j, v in zip(r.job, r.pae_seed1)}
        N["pae_full"] = [key.get(j.replace("N_binder_", "").replace("N_negative_", ""),
                                 np.nan) for j in N.job]
    if len(N) >= 6:
        pos = N[N.arm.eq("N_mature_control_binder")]
        neg = N[N.arm.eq("N_mature_control_negative")]
        log.info("\nTHE CONTROL -- the same cut on pairs whose answer is known:")
        log.info("  mature chain   AUROC %.3f   (n = %d binders, %d non-binders)",
                 auroc(-pos.pae_min_AB, -neg.pae_min_AB), len(pos), len(neg))
        log.info("  full chain     AUROC %.3f   (same pairs, screen construct)",
                 auroc(-pos.pae_full, -neg.pae_full))
        d_pos = (pos.pae_min_AB - pos.pae_full).median()
        d_neg = (neg.pae_min_AB - neg.pae_full).median()
        log.info("  median shift on cutting:  binders %+.2f A   non-binders %+.2f A",
                 d_pos, d_neg)
        log.info("  binders that got WORSE than 8 A: %d of %d",
                 int((pos.pae_min_AB > 8).sum()), len(pos))
        log.info("  -> if known binders survive the cut and candidates do not, the")
        log.info("     candidate interface needed a peptide the ligand never carries.")
        log.info("     If known binders collapse too, arm M cannot judge anything.")
    elif len(N):
        log.info("\nCONTROL: only %d of 24 mature controls folded so far", len(N))
    else:
        log.info("\nCONTROL: no mature controls folded yet -- arm M is NOT readable")

    # ---- arm M: candidate interfaces under the construct that exists ------------
    # Cutting the signal peptide does nothing to the RECEPTOR, so a mature chain
    # can clear the gate on the receptor's cytoplasmic face -- which for a
    # secreted partner is not a weaker claim but an impossible one. Measured on
    # this arm, 9 of 41 folds do exactly that, GPRC5C+IGF2 at 100%.
    zf = ROOT / "results" / "mature_interface_zone.csv"
    mz = (pd.read_csv(zf).set_index("job")[["rec_cyto", "partner_soluble"]].to_dict("index")
          if zf.exists() else {})
    if not mz:
        log.warning("no mature_interface_zone.csv -- run 38_interface_zone.py --arm mature "
                    "first, or a pose on the cytoplasmic face will read as a pass")
    M = best[best.arm.eq("M_mature")].sort_values("pae_min_AB")
    if len(M):
        log.info("\nMATURE CHAIN vs the screen's preprohormone (interface PAE):")
        log.info("  %-22s %8s %8s   %7s %7s  %s", "pair", "screen", "mature",
                 "d_pldA", "d_pldB", "verdict")
        for r in M.itertuples():
            z = mz.get(r.job, {})
            v = ("holds" if r.pae_min_AB <= GATE else
                 "weakened" if r.pae_min_AB <= 8 else "collapsed")
            if z.get("partner_soluble") and z.get("rec_cyto", 0) >= 0.3:
                v = f"IMPOSSIBLE -- {100 * z['rec_cyto']:.0f}% cytoplasmic receptor face"
            elif min(r.d_pldA, r.d_pldB) < -10:
                v += " -- but a chain folds worse too"
            log.info("  %-22s %8.2f %8.2f   %+7.1f %+7.1f  %s",
                     f"{r.receptor}+{r.partner}", r.pae_full, r.pae_min_AB,
                     r.d_pldA, r.d_pldB, v)
        log.info("  %d of %d hold under the %.2f A gate; median shift %+.2f A",
                 int((M.pae_min_AB <= GATE).sum()), len(M), GATE,
                 (M.pae_min_AB - M.pae_full).median())
        log.info("  chain folding, median change vs the screen: receptor %+.1f, "
                 "partner %+.1f pLDDT", M.d_pldA.median(), M.d_pldB.median())
        log.info("  -> a collapse with both chains folding as well as before is a")
        log.info("     lost interface, not a broken construct.")
        # The control's own shift is the yardstick: if candidates shift no more
        # than known binders do, there is nothing to explain.
        if len(N) >= 6 and "pae_full" in N:
            cb = (N[N.arm.eq("N_mature_control_binder")].eval("pae_min_AB - pae_full")
                  .dropna())
            cs = (M.pae_min_AB - M.pae_full).dropna()
            if len(cb) >= 5 and len(cs) >= 5:
                from scipy.stats import mannwhitneyu
                u = mannwhitneyu(cs, cb, alternative="greater")
                log.info("  candidates shift %+.2f A (median), known binders %+.2f A: "
                         "Mann-Whitney p = %.2g (n=%d vs %d)",
                         cs.median(), cb.median(), u.pvalue, len(cs), len(cb))
                # and the collapse RATE, with what this control can and cannot exclude
                from scipy.stats import beta, fisher_exact
                pb = N[N.arm.eq("N_mature_control_binder")]
                kb = int((pb.pae_min_AB > 8).sum()); nb = len(pb)
                kc = int((M.pae_min_AB > 8).sum()); nc = len(M)
                hi = beta.ppf(0.975, kb + 1, nb - kb) if nb else float("nan")
                od = fisher_exact([[kc, nc - kc], [kb, nb - kb]], alternative="greater")
                log.info("  collapse past 8 A: candidates %d of %d (%.0f%%), known binders "
                         "%d of %d; Fisher p = %.2g", kc, nc, 100 * kc / max(1, nc),
                         kb, nb, od.pvalue)
                log.info("  detection floor: %d of %d puts the 95%% upper bound on a "
                         "cut-induced collapse rate at %.0f%%, so this control cannot "
                         "exclude a destructive effect below that -- the candidates' "
                         "%.0f%% is above it.", kb, nb, 100 * hi, 100 * kc / max(1, nc))

    # ---- arm S: is the candidate interface a face the real assembly buries? -----
    S = best[best.arm.eq("S_subcomplex")]
    if len(S):
        log.info("\nSUBCOMPLEXES, folded whole (A-B interface only):")
        for r in S.itertuples():
            v = "survives the complete assembly" if r.pae_min_AB <= GATE else \
                "lost once the missing subunits are present"
            log.info("  %-28s %d chains  pae %5.2f (two-chain %5.2f)  %s",
                     r.job.replace("S_", ""), int(r.n_chain), r.pae_min_AB,
                     r.pae_full, v)
    best["rec_cyto"] = [mz.get(j, {}).get("rec_cyto", np.nan) for j in best.job]
    best["impossible_face"] = [bool(mz.get(j, {}).get("partner_soluble")
                                    and mz.get(j, {}).get("rec_cyto", 0) >= 0.3)
                               for j in best.job]
    best.to_csv(ROOT / "results" / "subcomplex_per_job.csv", index=False)
    log.info("\nwrote results/subcomplex_per_job.csv")


if __name__ == "__main__":
    main()
