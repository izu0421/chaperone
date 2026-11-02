"""Autonomous PTM/glycosylation-artifact resolution loop for one candidate
pair, following the design pattern in github.com/karpathy/autoresearch:

  autoresearch                                this script
  ---------------------------------------     ---------------------------------------
  train.py (the one file the agent edits)     which fold to run next
  prepare.py (fixed one-time setup)           UniProt facts, fetched once and trusted
  val_bpb (one clear scalar per iteration)     chain_pair_iptm at the flagged interface
  fixed 5-min budget, ~100 iterations/night   fixed 2-fold budget (bare + glycan-aware) -
                                               AF3 jobs cost ~10-15 min here, not 5, so
                                               "run it 100 times" isn't the right budget;
                                               "run it twice, decide, stop" is
  program.md (lightweight human playbook)     skills/resolve-ptm-artifact/SKILL.md

The point of copying this pattern rather than "just let an agent iterate
freely" is the same reason autoresearch itself is a fixed, tiny loop: a
bounded, reviewable process with one clear decision rule per step beats an
open-ended one that might run forever or rationalize its way to a
comfortable-sounding conclusion. This script ALWAYS terminates in <=2 real
fold jobs and reports one of a small set of honest outcomes -- including
"still unresolved, here's why, stop guessing" -- rather than looping until
something confident-sounding comes out.

Usage:
    python -m chaperone.resolve_ptm_artifact ADGRL2 LGALS1 --name adgrl2_lgals1 --gpu_device 2
"""
import argparse
import json
import sys
from pathlib import Path

from .analyze_fold import analyze_fold_result  # noqa: E402
from .deterministic_gate import fold_evidence_status  # noqa: E402
from .fold_candidate import fold  # noqa: E402
from .paths import FOLD_RUNS_DIR  # noqa: E402

FOLD_RUNS_FOLLOWUPS = FOLD_RUNS_DIR / "followups"

# Confidence calibration for a chain-pair interaction, corrected mid-
# conversation after the reflex was to call 0.26-0.27 "a weak positive
# signal" - it isn't. This is the standard AF-Multimer/AF3 field convention
# (Evans et al. 2021 and common practice since): below this, the model is
# not predicting a real interface, not "leaning towards" one.
CONFIDENT_IPTM_THRESHOLD = 0.5

ARTIFACT_RISK_PTM_TYPES = {"Glycosylation", "Lipidation"}
GLYCAN_FIXABLE_TYPES = {"Glycosylation"}  # glycan_lib only models sugars, not lipid anchors


def _flagged_ptm_types(structural_analysis: dict) -> set:
    types = set()
    for iface in structural_analysis.get("interfaces", []):
        for p in (iface.get("ptm_sites_at_interface_a") or []) + (iface.get("ptm_sites_at_interface_b") or []):
            if p["type"] in ARTIFACT_RISK_PTM_TYPES:
                types.add(p["type"])
    return types


def _chain_pair_iptm(summary: dict, idx_a: int, idx_b: int) -> float | None:
    cpi = summary.get("chain_pair_iptm")
    if not cpi:
        return None
    return cpi[idx_a][idx_b]


def _flagged_residues(structural_analysis: dict, chain: str) -> set:
    """Every interface residue on `chain` that's ever paired with a real
    Glycosylation/Lipidation site, across all reported interfaces."""
    flagged = set()
    for iface in structural_analysis.get("interfaces", []):
        for side, res_key, ptm_key in (("a", "interface_residues_a", "ptm_sites_at_interface_a"),
                                        ("b", "interface_residues_b", "ptm_sites_at_interface_b")):
            chain_id = iface["chain_a"] if side == "a" else iface["chain_b"]
            if chain_id != chain:
                continue
            ptms = [p for p in (iface.get(ptm_key) or []) if p["type"] in ARTIFACT_RISK_PTM_TYPES]
            if ptms:
                flagged.update(iface[res_key])
    return flagged


def resolve(protein_a: str, protein_b: str, name: str, gpu_device: str = "0",
            bare_result: dict | None = None) -> dict:
    """Runs the bounded resolution loop. `bare_result` lets you pass in an
    already-computed bare-fold result (avoids re-running a job you already
    have, e.g. from the original bulk screen) instead of folding fresh."""
    trail = []

    # --- Iteration 0: the bare fold (reuse if given, else run one) ---
    if bare_result is None:
        print(f"[iter 0] running bare fold {protein_a}+{protein_b} ...", file=sys.stderr)
        bare_result = fold([protein_a, protein_b], f"{name}_bare",
                            output_root=FOLD_RUNS_FOLLOWUPS / f"{name}_bare",
                            gpu_device=gpu_device, with_glycans=False)
        bare_result["structural_analysis"] = analyze_fold_result(bare_result)
    trail.append({"iteration": 0, "kind": "bare_fold", "iptm": bare_result.get("iptm"),
                  "chain_pair_iptm": bare_result.get("chain_pair_iptm")})

    status = fold_evidence_status([bare_result])
    print(f"[iter 0] fold_evidence_status: {status['status']}", file=sys.stderr)

    if status["status"] == "pass":
        return _conclude("RESOLVED_NO_CONCERN", trail,
                          "No PTM/glycosylation site at the modeled interface and no membrane-"
                          "topology violation - nothing to resolve.", bare_result)

    if status["status"] == "topology_violation":
        # Not fixable by remodeling a PTM - a Cytoplasmic+Extracellular span
        # is a hard membrane-topology impossibility, independent of any
        # unmodeled modification. Re-running with glycans wouldn't change
        # that the two halves of the interface can't coexist across an
        # intact membrane, so there's nothing more to try here.
        return _conclude("CONFIRMED_ARTIFACT", trail,
                          "Real fold found an interface spanning both Cytoplasmic and "
                          "Extracellular residues - topologically impossible across an intact "
                          "membrane. This isn't a PTM-modeling gap, so a glycan-aware re-fold "
                          "would not resolve it; stopping here rather than running one anyway.",
                          bare_result)

    # status == "ptm_only_concern"
    flagged_types = _flagged_ptm_types(bare_result["structural_analysis"])
    if not flagged_types & GLYCAN_FIXABLE_TYPES:
        return _conclude("NOT_AUTOMATICALLY_RESOLVABLE", trail,
                          f"Flagged PTM type(s) {sorted(flagged_types)} aren't modelable as an "
                          "AF3 ligand chain (only N-linked glycosylation is currently "
                          "supported by chaperone.glycan_lib) - can't attempt a structural resolution. "
                          "Needs the wet-lab follow-up already in the verdict.", bare_result)

    # --- Iteration 1: glycan-aware re-fold ---
    print(f"[iter 1] running glycan-aware fold {protein_a}+{protein_b} ...", file=sys.stderr)
    glycan_result = fold([protein_a, protein_b], f"{name}_glycan",
                          output_root=FOLD_RUNS_FOLLOWUPS / f"{name}_glycan",
                          gpu_device=gpu_device, with_glycans=True)
    glycan_result["structural_analysis"] = analyze_fold_result(glycan_result)
    trail.append({"iteration": 1, "kind": "glycan_aware_fold", "iptm": glycan_result.get("iptm"),
                  "chain_pair_iptm": glycan_result.get("chain_pair_iptm"),
                  "glycan_chains": glycan_result.get("glycan_chains")})

    ab_iptm = _chain_pair_iptm(glycan_result, 0, 1)  # protein A vs protein B, always chains 0,1
    original_flagged_a = _flagged_residues(bare_result["structural_analysis"], "A")
    original_flagged_b = _flagged_residues(bare_result["structural_analysis"], "B")
    new_ab_iface = next((i for i in glycan_result["structural_analysis"]["interfaces"]
                         if i["chain_a"] == "A" and i["chain_b"] == "B"), None)
    still_at_flagged_site = bool(
        new_ab_iface and (
            set(new_ab_iface["interface_residues_a"]) & original_flagged_a
            or set(new_ab_iface["interface_residues_b"]) & original_flagged_b
        )
    )

    # Where does the highest-confidence contact actually involve protein B
    # (LGALS1-style: does B engage a glycan chain directly, and how
    # confidently)?
    best_glycan_contact = None
    for i, iface in enumerate(glycan_result["structural_analysis"]["interfaces"]):
        if "B" not in (iface["chain_a"], iface["chain_b"]):
            continue
        other = iface["chain_b"] if iface["chain_a"] == "B" else iface["chain_a"]
        if other in ("A", "B"):
            continue  # a protein-protein interface, not protein-glycan
        idx = ord(other) - ord("A")
        cpi = _chain_pair_iptm(glycan_result, 1, idx)
        if cpi is not None and (best_glycan_contact is None or cpi > best_glycan_contact[1]):
            best_glycan_contact = (other, cpi, iface)

    if ab_iptm is not None and ab_iptm >= CONFIDENT_IPTM_THRESHOLD and still_at_flagged_site:
        return _conclude("RESOLVED_NOT_ARTIFACT", trail,
                          f"The A-B interface survives with the real glycan physically present "
                          f"(chain_pair_iptm={ab_iptm}, still >= {CONFIDENT_IPTM_THRESHOLD}) and "
                          "still involves the originally-flagged residues - the glycan doesn't "
                          "block it, so the artifact concern doesn't hold up.", glycan_result)

    if best_glycan_contact and best_glycan_contact[1] >= CONFIDENT_IPTM_THRESHOLD:
        glycan_chain, cpi, iface = best_glycan_contact
        return _conclude("RESOLVED_ALTERNATE_SITE", trail,
                          f"Protein-protein interface did not survive (chain_pair_iptm(A,B)="
                          f"{ab_iptm}), but chain B confidently engages glycan chain "
                          f"{glycan_chain} (chain_pair_iptm={cpi} >= {CONFIDENT_IPTM_THRESHOLD}). "
                          "Real mechanism is plausibly glycan-mediated at this site, not the "
                          "originally-modeled bare-protein contact.", glycan_result)

    # Nothing cleared the confidence bar, at any site. This is the honest
    # "computation can't settle this" outcome - report it as one, don't
    # oversell a low-confidence residue pattern as a finding (this is
    # exactly the mistake made mid-conversation on this same pair before
    # being corrected).
    return _conclude("INCONCLUSIVE_LOW_CONFIDENCE", trail,
                      f"Neither the original interface (chain_pair_iptm(A,B)={ab_iptm}) nor any "
                      f"protein-glycan contact (best: {best_glycan_contact[0] if best_glycan_contact else None}, "
                      f"{best_glycan_contact[1] if best_glycan_contact else None}) reaches "
                      f"{CONFIDENT_IPTM_THRESHOLD}. AF3 has nothing confident to say once the real "
                      "PTM is modeled, at any site - this needs the wet-lab follow-up already in "
                      "the verdict, not more computation. Budget exhausted (2/2 folds run); stopping.",
                      glycan_result)


def _conclude(outcome: str, trail: list, explanation: str, last_result: dict) -> dict:
    record = {
        "outcome": outcome,
        "explanation": explanation,
        "n_folds_run": len(trail),
        "trail": trail,
        "final_structural_analysis": last_result.get("structural_analysis"),
    }
    print(f"\n=== CONCLUSION: {outcome} ===\n{explanation}\n", file=sys.stderr)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protein_a")
    ap.add_argument("protein_b")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu_device", default="0")
    ap.add_argument("--out", default=None, help="Where to write the resolution JSON "
                     "(default: fold_runs/followups/<name>_resolution.json)")
    args = ap.parse_args()

    record = resolve(args.protein_a, args.protein_b, args.name, gpu_device=args.gpu_device)
    out_path = Path(args.out) if args.out else FOLD_RUNS_FOLLOWUPS / f"{args.name}_resolution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
