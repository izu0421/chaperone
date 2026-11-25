"""One table per candidate: every structural test, and whether it even applied.

The screen's shortlist came from one gate on one fold. Three constructs have
since been folded against their own controls, and each one answers a different
objection:

  seeds       is the pose a lucky draw?            arm A, 25 models per pair
  cut face    is the partner on the wrong face?    arm B, ECD + 3 decoys
  construct   was the folded molecule real?        arm M, mature chain + arm N

A test that could not be run is NOT a pass. GPR82 has no extracellular domain to
truncate and OLR1 has no signal peptide to remove, so two of the three tests are
silent for it; that is a thinner case than GPR158, not an equal one. The table
prints applicability per pair for exactly that reason.

The multi-seed arm carries no credit either way: on controls, AUROC was 0.947 at
one seed and 0.947 at best-of-25, so best-of-N is a maximum over draws that
lifts known non-binders as much as known binders. It is reported as a spread,
never as a gain.

Usage:
    micromamba run -n structmod python -I scripts/36_rank_candidates.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rank")
ROOT = Path(__file__).resolve().parent.parent
CHAP = ROOT.parent / "chaperone"


def main() -> None:
    GATE = json.loads((ROOT / "results" / "protein_thresholds.json").read_text()
                      )["gate"]["pae_min_AB_max"]
    topo = json.loads((ROOT / "results" / "orphan_topology.json").read_text())
    _zf = ROOT / "results" / "candidate_interface_zone.csv"
    z_soluble = ({(r.receptor, r.partner): r.partner_soluble
                  for r in pd.read_csv(_zf).itertuples()} if _zf.exists() else {})
    v = pd.read_csv(CHAP / "reports" / "screen_verdicts_current" / "verdicts.csv")
    scr = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    scr["key"] = scr.receptor.astype(str) + "__" + scr.partner.astype(str)
    sc = scr.set_index("key")

    cand = v[v.verdict.eq("CONFIRMED_NOVEL")].copy()
    cand["key"] = cand.protein_b.astype(str) + "__" + cand.protein_a.astype(str)
    log.info("CANDIDATES: %d CONFIRMED_NOVEL (%s)", len(cand),
             ", ".join(f"{k} {n}" for k, n in cand.partner_kind.value_counts().items()))

    # ---- what each arm measured, and for which pairs it was built ---------------
    def arm_table(man_f, score_f, key_col="job"):
        if not (ROOT / "work" / man_f).exists() or not (ROOT / "results" / score_f).exists():
            return pd.DataFrame()
        m = pd.read_csv(ROOT / "work" / man_f)
        s = pd.read_csv(ROOT / "results" / score_f)
        return s.merge(m[[c for c in ("job", "arm", "receptor", "partner") if c in m]],
                       on=[c for c in ("job", "arm") if c in s and c in m], how="left",
                       suffixes=("", "_man"))

    A = arm_table("rescue_manifest.csv", "rescue_per_job.csv")
    M = arm_table("subcomplex_manifest.csv", "subcomplex_per_job.csv")
    pm = ROOT / "work" / "ectodomain_manifest.csv"
    ps = ROOT / "results" / "ectodomain_per_job.csv"
    P_MAN = pd.read_csv(pm) if pm.exists() else None
    P_SC = pd.read_csv(ps) if ps.exists() else None
    built_A = set(A[A.arm.eq("A_multiseed")].job.str.replace("A_", "", regex=False)) if len(A) else set()
    built_B = set(A[A.arm.eq("B_ecd")].job) if len(A) else set()
    built_M = set(M[M.arm.eq("M_mature")].job.str.replace("M_", "", regex=False)) if len(M) else set()

    # the mature-cut control decides whether arm M can judge anything at all
    ctl_state, ctl_note = "pending", "controls not folded"
    if len(M):
        N = M[M.arm.str.startswith("N_mature")]
        if len(N) >= 12:
            pos, neg = N[N.arm.str.endswith("binder")], N[N.arm.str.endswith("negative")]
            kept = int((pos.pae_min_AB <= 8).sum())
            ctl_state = "usable" if kept >= 0.7 * len(pos) else "destructive"
            ctl_note = (f"{kept} of {len(pos)} known binders keep an interface "
                        f"under 8 A after the same cut")
        elif len(N):
            ctl_note = f"only {len(N)} of 24 controls folded"
    log.info("MATURE-CUT CONTROL: %s -- %s\n", ctl_state, ctl_note)

    rows = []
    for r in cand.itertuples():
        rec, par, key = r.protein_b, r.protein_a, r.key
        d = {"receptor": rec, "partner": par, "kind": r.partner_kind,
             "confidence": r.confidence}
        s = sc.loc[key] if key in sc.index else None
        if s is not None:
            s = s.iloc[0] if isinstance(s, pd.DataFrame) else s
            d["pae_screen"] = s.pae_min_AB
            d["iptm"] = s.iptm
        # promiscuity: how many of the receptors tested does this partner bind?
        d["partner_receptors"] = r.partner_receptors_tested
        d["partner_confident"] = r.partner_receptors_confident

        # seeds
        a = A[A.job.eq(f"A_{key}")] if len(A) else pd.DataFrame()
        d["seeds_n"] = 25 if len(a) else 0
        d["pae_spread"] = (a.pae_max.iloc[0] - a.pae_min.iloc[0]) if len(a) else np.nan

        # cut face: candidate vs its own decoys. Two different silences here --
        # a receptor with no extracellular domain cannot be truncated at all,
        # while a receptor that HAS one but no job built simply was not tested.
        # Printing both as "no extracellular domain" would report an untested
        # pair as an untestable one.
        # Three different silences, and the old label collapsed them into one.
        # No receptor on this list LACKS an extracellular segment -- all have
        # four, totalling 39-120 aa -- so "receptor ECD only N aa" read as
        # "untestable" when the arm was simply never built, and for five
        # receptors the N it quoted was an extracellular LOOP in the middle of
        # the bundle, not an N-terminal domain that could be cut at all.
        ext = [(int(s["start"]), int(s["end"])) for s in topo.get(rec, [])
               if "Extracellular" in str(s.get("description") or "")]
        longest = max((b - a + 1 for a, b in ext), default=0)
        nterm = next(((b - a + 1) for a, b in sorted(ext) if a <= 50), 0)
        d["rec_ecd_aa"] = longest
        d["rec_nterm_aa"] = nterm
        if longest >= 100:
            d["ecd_test"] = "not built for this pair"
        elif nterm >= 100:
            d["ecd_test"] = f"not built; N-terminus is {nterm} aa"
        elif nterm:
            d["ecd_test"] = f"N-terminus only {nterm} aa, too short to cut"
        else:
            d["ecd_test"] = (f"no N-terminal domain; longest extracellular segment "
                             f"is a {longest} aa loop")
        if len(A):
            # the ECD verdict is READ, never recomputed: it has a precondition the
            # PAE columns cannot express, namely that the truncated receptor folds
            b = A[A.arm.isin(["B_ecd", "B2_ntd"])
                  & A.job.str.contains(f"{rec}(?:ECD|NTD)__{par}$", regex=True)]
            if len(b):
                d["pae_ecd"] = b.pae_best.iloc[0]
                d["ecd_test"] = (b.verdict.iloc[0] if "verdict" in b
                                 and pd.notna(b.verdict.iloc[0]) else "folded, no verdict")

        # construct: the molecule that exists
        d["mature_test"] = "no signal peptide"
        if len(M):
            m = M[M.arm.eq("M_mature") & M.job.eq(f"M_{key}")]
            if len(m):
                p = m.pae_min_AB.iloc[0]
                d["pae_mature"] = p
                d["d_plddt_min"] = min(m.d_pldA.iloc[0], m.d_pldB.iloc[0])
                imp = bool(m.impossible_face.iloc[0]) if "impossible_face" in m else False
                if ctl_state != "usable":
                    d["mature_test"] = f"not readable ({ctl_state})"
                elif imp:
                    d["mature_test"] = "FAIL -- cytoplasmic face, partner is secreted"
                elif p <= GATE:
                    d["mature_test"] = "PASS"
                elif p <= 8:
                    d["mature_test"] = "weakened"
                else:
                    d["mature_test"] = "FAIL -- interface lost"
            elif key in {j.replace("M_", "") for j in
                         pd.read_csv(ROOT / "work" / "subcomplex_manifest.csv").job}:
                d["mature_test"] = "built, not folded yet"
        # arm P: receptor full length against the partner's ectodomain only
        # arm P's verdict is READ, never recomputed: it carries a requirement the
        # PAE columns cannot express, namely that the surviving interface is on a
        # face the partner can reach.
        # Likewise: a secreted partner has no ectodomain because it is ENTIRELY
        # extracellular -- MST1 is 711 aa of it. Saying "no ectodomain" there
        # reported the wrong arm as inapplicable; the mature-chain arm is the
        # construct test that reaches a secreted partner.
        soluble = bool(z_soluble.get((rec, par), False)) if z_soluble else False
        d["p_test"] = ("partner is secreted; the mature-chain arm is the test"
                       if soluble else "partner ectodomain too short to cut")
        if P_MAN is not None and f"P_{rec}__{par}" in set(P_MAN.job):
            if P_SC is None:
                d["p_test"] = "built, not folded yet"
            else:
                for job in (f"P_both_{rec}__{par}", f"P_{rec}__{par}"):
                    cand = P_SC[P_SC.job.eq(job)]
                    if len(cand):
                        d["pae_ecto"] = cand.pae_min_AB.iloc[0]
                        d["rec_cyto"] = cand.rec_cyto.iloc[0] if "rec_cyto" in cand else np.nan
                        d["p_test"] = (cand.verdict.iloc[0] if "verdict" in cand
                                       and pd.notna(cand.verdict.iloc[0])
                                       else "folded, no verdict")
                        break
                else:
                    d["p_test"] = "folding"
        rows.append(d)

    t = pd.DataFrame(rows)
    # what KIND of interaction each fold proposes, which decides which construct
    # test can speak for it at all
    zf = ROOT / "results" / "candidate_interface_zone.csv"
    if zf.exists():
        z = pd.read_csv(zf)[["receptor", "partner", "kind", "partner_soluble"]]
        t = t.merge(z.rename(columns={"kind": "interface"}), on=["receptor", "partner"],
                    how="left")
    else:
        t["interface"] = "not computed"
    for c in ("ecd_test", "mature_test", "p_test"):
        log.info("%-12s applied to %d of %d candidates: %d pass, %d fail", c,
                 int(t[c].isin(["PASS"]).sum() + t[c].astype(str).str.startswith("FAIL").sum()
                     + t[c].eq("weakened").sum()),
                 len(t), int(t[c].eq("PASS").sum()),
                 int(t[c].astype(str).str.startswith("FAIL").sum()))

    def tier(r):
        fails = [c for c in ("ecd_test", "mature_test", "p_test")
                 if str(r[c]).startswith("FAIL")]
        if fails:
            return "3 -- fails a structural test"
        if str(r.get("interface")) == "IMPOSSIBLE cytoplasmic":
            return "3 -- secreted partner on the cytoplasmic face"
        passes = [c for c in ("ecd_test", "mature_test", "p_test") if r[c] == "PASS"]
        if str(r["mature_test"]) == "weakened":
            return "2b -- weakened under the real construct"
        if len(passes) >= 2:
            return "1 -- passes every test that could be run"
        if len(passes) == 1:
            return "2a -- one test passed, the other could not be run"
        return "2c -- gate only, no construct test applies"

    t["tier"] = t.apply(tier, axis=1)
    t = t.sort_values(["tier", "pae_screen"])
    dest = ROOT / "results" / "candidate_ranking.csv"
    t.to_csv(dest, index=False)

    log.info("")
    for tr, g in t.groupby("tier"):
        log.info("TIER %s   (%d)", tr, len(g))
        for r in g.itertuples():
            log.info("   %-18s gate %5.2f  %-22s ecd: %-24s mature: %-22s ecto: %s",
                     f"{r.receptor}+{r.partner}",
                     r.pae_screen if pd.notna(r.pae_screen) else float("nan"),
                     getattr(r, "interface", "?"), r.ecd_test, r.mature_test, r.p_test)
        log.info("")
    log.info("wrote %s", dest)


if __name__ == "__main__":
    main()
