"""Score the completed orphan co-folds on the same interface terms as the benchmark arms.

The per-fold work is imported from `08_score_armB.py` rather than re-implemented, so a
candidate pair and a benchmark pair are measured by exactly the same code. Only two
things differ here: the output directory, and that there are no labels to join -- these
are the candidates the screen exists to rank.

The screen is still running, so this scores whatever has finished and records how many
that was. Re-run it as more folds land.

Usage:
    micromamba run -n structmod python -I scripts/14_score_orphans.py --workers 24
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("orphan-score")
ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("score_arm", ROOT / "scripts" / "08_score_armB.py")
_score_arm = importlib.util.module_from_spec(_spec)
# registered before exec so that worker processes can re-import it by name: a module
# loaded only by path cannot be pickled across the process boundary
sys.modules["score_arm"] = _score_arm
_spec.loader.exec_module(_score_arm)
one = _score_arm.one



def per_chain_plddt(d: Path) -> dict:
    """Mean pLDDT of each chain, and of each chain's own interface atoms.

    `one()` returns `iface_plddt` averaged over BOTH chains, which for a
    metabolite fold mixes the receptor's pocket residues into the number. The
    literature is consistent that the ligand chain's own confidence is the
    metric that discriminates and that ipTM is the weakest: Kim et al. (eLife
    2026) found AF3 ligand-pLDDT their best metric, Ferguson et al. (Molecular
    Cell 2026) found peptide pLDDT beat both ipTM and pocket PAE, and Shamir
    et al. (JACS 2026) rejected protein-only metrics outright. Splitting the
    average by chain costs one extra parse and no re-folding.

    Chain A is the receptor and chain B the partner, so `plddt_B` is the
    ligand-pLDDT for a metabolite candidate.
    """
    import warnings

    name = d.name
    out = {"pair": name}
    cif = d / f"{name}_model.cif"
    if not cif.exists():
        return out
    try:
        from Bio.PDB import MMCIFParser
        from scipy.spatial import cKDTree
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = MMCIFParser(QUIET=True).get_structure(name, str(cif))
        chains = list(next(iter(st)))
        if len(chains) != 2:
            return out
        heavy = [[a for a in c.get_atoms() if a.element != "H"] for c in chains]
        if not all(heavy):
            return out
        coords = [np.array([a.coord for a in h]) for h in heavy]
        tree = cKDTree(coords[1])
        contacts = tree.query_ball_point(coords[0], _score_arm.CUT)
        iface = [{heavy[0][i].get_parent().id[1] for i, c in enumerate(contacts) if c},
                 {heavy[1][j].get_parent().id[1] for c in contacts for j in c}]
        for tag, atoms, res in zip("AB", heavy, iface):
            out[f"plddt_{tag}"] = float(np.mean([a.get_bfactor() for a in atoms]))
            at_iface = [a.get_bfactor() for a in atoms if a.get_parent().id[1] in res]
            out[f"iface_plddt_{tag}"] = float(np.mean(at_iface)) if at_iface else np.nan
    except Exception as exc:  # noqa: BLE001  — a miss costs the extra columns only
        out["plddt_note"] = f"{type(exc).__name__} {str(exc)[:40]}"
    return out


def score_one(d: Path) -> dict:
    """The shared benchmark scoring, plus the per-chain confidences."""
    return {**one(d), **per_chain_plddt(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    out_dir = ROOT / "results" / "orphan_out"
    manifest = pd.read_csv(ROOT / "work" / "orphan_manifest.csv")
    dirs = sorted(d for d in out_dir.iterdir() if d.is_dir())
    log.info("%d of %d folds complete (%.1f%%)", len(dirs), len(manifest),
             100 * len(dirs) / len(manifest))

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(score_one, d) for d in dirs]
        for f in as_completed(futures):
            rows.append(f.result())
            done += 1
            if done % 250 == 0:
                log.info("  scored %d/%d", done, len(dirs))

    scored = pd.DataFrame(rows).merge(
        manifest.rename(columns={"job": "pair"})[
            ["pair", "receptor", "partner", "kind", "msa_source", "p_twoway", "rank"]],
        on="pair", how="left")

    missing = scored.receptor.isna().sum()
    if missing:
        log.warning("%d scored folds are not in the manifest", missing)

    dest = ROOT / "results" / "orphan_scores.csv"
    scored.to_csv(dest, index=False)
    log.info("wrote %s", dest)

    noted = scored[scored.note.ne("")]
    log.info("\n%d folds with a note:\n%s", len(noted),
             noted.note.value_counts().head(6).to_string() if len(noted) else "  none")

    # a contact count means nothing without a clash check: an un-relaxed trajectory can
    # report hundreds of contacts at a separation no real interface could adopt
    usable = scored[scored.min_sep.notna()]
    log.info("\nclash check: %d of %d folds have min_sep < 1.5 A",
             int((usable.min_sep < 1.5).sum()), len(usable))
    cols = [c for c in ["iptm", "bsa", "n_contact", "iface_plddt", "plddt_B",
                        "iface_plddt_B"] if c in scored.columns]
    log.info("\n%s", scored[cols].describe().round(2).to_string())

    # the comparison the literature says to make: does the partner chain's own
    # confidence separate the two candidate kinds differently from ipTM?
    if {"kind", "plddt_B"} <= set(scored.columns):
        log.info("\nby candidate kind:\n%s",
                 scored.groupby("kind")[[c for c in ("iptm", "iface_plddt", "plddt_B",
                                                     "iface_plddt_B") if c in scored.columns]]
                 .median().round(3).to_string())


if __name__ == "__main__":
    main()
