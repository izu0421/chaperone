"""Actually execute the structural follow-ups the triage report suggests,
instead of leaving them as text recommendations.

For every candidate whose verdict is LIKELY_SUBCOMPLEX or CONFIRMED_NOVEL and
names concrete `other_subunits`, this builds and runs the proposed extended
complex through AlphaFast (see skills/fold-candidate/).

Job composition per candidate is hand-reviewed (not mechanical): where
`other_subunits` lists functionally distinct partners (e.g. a G-protein
beta+gamma pair, an obligate light chain like B2M), all are folded together
in one complex. Where it lists mutually-exclusive paralogs (e.g. KLRC1/2/3,
which each pair with KLRD1 alternatively, not simultaneously; ITGAL/ITGAM/
ITGAX/ITGAD, alternative integrin alpha chains), only one representative
paralog is included per job — folding all alternatives together would model
a complex that cannot exist. See the `note` field on each job for the
reasoning.

Usage:
    python -m chaperone.run_follow_up_folds            # run all jobs
    python -m chaperone.run_follow_up_folds --dry-run   # just print the plan
"""
import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .analyze_fold import analyze_fold_result  # noqa: E402
from .paths import FOLD_RUNS_DIR  # noqa: E402
from .fold_candidate import fold  # noqa: E402

OUTPUT_ROOT = FOLD_RUNS_DIR / "followups"
GPU_DEVICES = ["0", "1"]  # GPU 2 was in use by another job at survey time

# (job_name, [protein_a, protein_b, *extra_chains], source_pair, note)
JOBS = [
    ("mfap4_il13ra2_il13_il4r_il13ra1", ["MFAP4", "IL13RA2", "IL13", "IL4R", "IL13RA1"],
     "MFAP4/IL13RA2", "all listed subunits are part of the same IL-13 receptor system, combined"),
    ("gabbr2_sema4c_gabbr1_plxnb2", ["GABBR2", "SEMA4C", "GABBR1", "PLXNB2"],
     "GABBR2/SEMA4C", "GABBR1+GABBR2 obligate heterodimer and SEMA4C's real receptor PLXNB2 are distinct roles, both single genes -> combined"),
    ("itgb8_mfap4_itgav", ["ITGB8", "MFAP4", "ITGAV"],
     "ITGB8/MFAP4", "ITGAV+ITGB8 is the real integrin alphaV/beta8 heterodimer"),
    ("cxcr1_gna11_gnb1_gng2", ["CXCR1", "GNA11", "GNB1", "GNG2"],
     "CXCR1/GNA11", "GNA11+GNB1+GNG2 is a real G-protein heterotrimer"),
    ("klrk1_klrd1_hcst_klrc1", ["KLRK1", "KLRD1", "HCST", "KLRC1"],
     "KLRK1/KLRD1", "KLRC1/2/3 are mutually-exclusive NKG2 paralogs (pick KLRC1 as representative); HCST is KLRK1's distinct real adaptor"),
    ("fcgrt_dlg4_b2m", ["FCGRT", "DLG4", "B2M"],
     "FCGRT/DLG4", "B2M is FCGRT's real obligate light chain (neonatal Fc receptor)"),
    ("btn3a2_klrd1_btn3a1_klrc1", ["BTN3A2", "KLRD1", "BTN3A1", "KLRC1"],
     "BTN3A2/KLRD1", "KLRC1/2/3 mutually exclusive (pick KLRC1); BTN3A1 is a distinct real partner"),
    ("grik2_rnd1_grik4_grik5", ["GRIK2", "RND1", "GRIK4", "GRIK5"],
     "GRIK2/RND1", "kainate receptor subunits GRIK2/4/5 can co-assemble in heteromeric channels, combined"),
    ("klrd1_smoc2_klrc1", ["KLRD1", "SMOC2", "KLRC1"],
     "KLRD1/SMOC2", "KLRC1/2/3 mutually exclusive, pick KLRC1 as representative"),
    ("ldlr_fermt2_ldlrap1_itgb1", ["LDLR", "FERMT2", "LDLRAP1", "ITGB1"],
     "LDLR/FERMT2", "ITGB1/ITGB3 alternative integrin betas (pick ITGB1); LDLRAP1 is a distinct real adaptor"),
    ("itga8_f11r_itgb1", ["ITGA8", "F11R", "ITGB1"],
     "ITGA8/F11R", "ITGA8+ITGB1 is a real integrin heterodimer"),
    ("c5ar1_gna13_gnb1_gng2", ["C5AR1", "GNA13", "GNB1", "GNG2"],
     "C5AR1/GNA13", "GNA13+GNB1+GNG2 real G-protein heterotrimer"),
    ("gnaq_grik4_grik1_grik2_grik3", ["GNAQ", "GRIK4", "GRIK1", "GRIK2", "GRIK3"],
     "GNAQ/GRIK4", "GRIK1/2/3/4 kainate subunits combined as proposed, despite GNAQ (a GPCR G-alpha) being a biologically odd partner for an ionotropic receptor family — folding as literally proposed to let the result speak for itself"),
    ("f2rl2_fcgrt_b2m", ["F2RL2", "FCGRT", "B2M"],
     "F2RL2/FCGRT", "B2M obligate FCGRT light chain"),
    ("anxa1_fcgrt_b2m", ["ANXA1", "FCGRT", "B2M"],
     "ANXA1/FCGRT", "B2M obligate FCGRT light chain"),
    ("grk2_ifngr2_ifngr1", ["GRK2", "IFNGR2", "IFNGR1"],
     "GRK2/IFNGR2", "IFNGR1+IFNGR2 real interferon-gamma receptor complex"),
    ("msn_il3ra_csf2rb", ["MSN", "IL3RA", "CSF2RB"],
     "MSN/IL3RA", "CSF2RB is the real shared common-beta receptor chain"),
    ("ifngr1_snx5_ifngr2", ["IFNGR1", "SNX5", "IFNGR2"],
     "IFNGR1/SNX5", "IFNGR1+IFNGR2 real complex"),
    ("itgb2_cd80_itgal", ["ITGB2", "CD80", "ITGAL"],
     "ITGB2/CD80", "ITGAL/ITGAM/ITGAX/ITGAD mutually exclusive alpha chains (pick ITGAL = real LFA-1 pairing)"),
    ("itgb8_gna13_itgav", ["ITGB8", "GNA13", "ITGAV"],
     "ITGB8/GNA13", "ITGAV+ITGB8 real integrin heterodimer"),
]


def run_one(job, gpu_device):
    name, chains, source_pair, note = job
    out_dir = OUTPUT_ROOT / name
    try:
        result = fold(chains, name, output_root=out_dir, gpu_device=gpu_device)
        result.update({"source_pair": source_pair, "note": note, "status": "ok"})
        try:
            result["structural_analysis"] = analyze_fold_result(result)
        except Exception as exc:  # noqa: BLE001
            result["structural_analysis_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        result = {
            "job_name": name, "source_pair": source_pair, "note": note,
            "status": "error", "error": str(exc), "traceback": traceback.format_exc(),
        }
    (OUTPUT_ROOT / f"{name}_result.json").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / f"{name}_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="Comma-separated job names to (re)run, e.g. for retrying failures")
    ap.add_argument("--gpu-devices", default=None, help="Comma-separated GPU IDs to use, overriding the module default")
    args = ap.parse_args()

    jobs = JOBS
    if args.only:
        wanted = set(args.only.split(","))
        jobs = [j for j in JOBS if j[0] in wanted]
        missing = wanted - {j[0] for j in jobs}
        if missing:
            print(f"WARNING: unknown job names in --only: {missing}")

    gpu_devices = args.gpu_devices.split(",") if args.gpu_devices else GPU_DEVICES

    print(f"{len(jobs)} follow-up fold jobs planned (GPUs: {gpu_devices}):")
    for name, chains, source_pair, note in jobs:
        print(f"  {name}: {'+'.join(chains)}  (from {source_pair}: {note})")

    if args.dry_run:
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(gpu_devices)) as pool:
        futures = {
            pool.submit(run_one, job, gpu_devices[i % len(gpu_devices)]): job
            for i, job in enumerate(jobs)
        }
        for fut in as_completed(futures):
            job = futures[fut]
            result = fut.result()
            results.append(result)
            status = result.get("status")
            if status == "ok":
                print(f"[done] {job[0]}: iptm={result.get('iptm')} ptm={result.get('ptm')} "
                      f"ranking_score={result.get('ranking_score')} ({result.get('elapsed_seconds')}s)")
            else:
                print(f"[FAILED] {job[0]}: {result.get('error')}")

    summary_path = OUTPUT_ROOT / "_summary.json"
    # Merge into any existing summary (e.g. a --only retry of a subset)
    # instead of clobbering results for jobs not in this run.
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    by_name = {r.get("job_name"): r for r in existing}
    for r in results:
        by_name[r.get("job_name")] = r
    summary_path.write_text(json.dumps(list(by_name.values()), indent=2))
    print(f"\nWrote summary: {summary_path} ({len(by_name)} total jobs)")


if __name__ == "__main__":
    main()
