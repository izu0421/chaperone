"""Structural analysis of a completed fold_candidate.py output.

Beyond the summary confidence scores (ipTM/pTM/ranking_score) already parsed
by fold_candidate.py, this actually looks at the 3D structure:
- per-chain-pair interface residues (any heavy-atom contact within 5A)
- per-interface pLDDT (low-confidence interfaces are themselves a red flag,
  distinct from low overall pTM)
- whether a real, positioned UniProt PTM/glycosylation site (Modified
  residue / Glycosylation / Disulfide bond / Lipidation) falls ON the
  modeled interface — turning the LIKELY_ARTIFACT_PTM protein-level guess
  ("this protein is glycosylated somewhere") into a residue-level check
  ("the modeled contact directly involves a known glycosylation site").
- which side of the membrane the interface sits on, for membrane proteins
  (UniProt "Topological domain": Cytoplasmic / Extracellular / Lumenal, plus
  Transmembrane span boundaries). This is the residue-level version of a
  basic biological-plausibility check: a strictly intracellular partner
  meeting a membrane protein's *cytoplasmic*-facing interface is plausible
  (e.g. a kinase/adaptor docking on a receptor's cytoplasmic tail); meeting
  its *extracellular* interface is topologically impossible under an intact
  membrane, regardless of how dynamic/multi-localized proteins can otherwise
  be — a hard biophysical constraint, not a soft heuristic. This module only
  surfaces the fact (which topological domain the modeled interface falls
  in); judging plausibility against the *other* chain's known compartment
  (from HPA, seen earlier in the same conversation) is left to the agent —
  proteins do moonlight/shuttle, so a rigid auto-veto here would be wrong.

Usage:
    python -m chaperone.analyze_fold fold_runs/runx1_cbfb/output/runx1_cbfb  # dir from fold_candidate.py
    python -m chaperone.analyze_fold --cif path/to/model.cif --chains A=Q01196,B=Q13951
"""
import argparse
import json
import sys
from pathlib import Path

import gemmi
import httpx
import numpy as np

from .sources.http_retry import get_with_retry  # noqa: E402

INTERFACE_CUTOFF_ANGSTROM = 5.0
PTM_FEATURE_TYPES = {
    "Modified residue", "Glycosylation", "Disulfide bond", "Lipidation", "Cross-link",
}
TOPOLOGY_FEATURE_TYPES = {"Topological domain", "Transmembrane", "Intramembrane"}
CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# van der Waals radii, Angstrom, for the elements an AF3 model contains
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47,
       "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90, "B": 1.92, "SI": 2.10,
       "ZN": 1.39, "MG": 1.73, "NA": 2.27, "K": 2.75, "CA": 2.31, "FE": 2.00}
DEFAULT_VDW = 1.70
PROBE = 1.4                      # water probe, the usual Shrake-Rupley value


def load_chain_atoms(cif_path: str) -> dict:
    st = gemmi.read_structure(str(cif_path))
    model = st[0]
    chains = {}
    for chain in model:
        coords, res_index, plddt, radii = [], [], [], []
        n_res = 0
        for res in chain:
            n_res += 1
            for atom in res:
                if atom.element.is_hydrogen:
                    continue
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                res_index.append(res.seqid.num)
                plddt.append(atom.b_iso)
                radii.append(VDW.get(atom.element.name.upper(), DEFAULT_VDW))
        # A small molecule arrives as a single residue of non-standard type.
        # Telling it apart from a polymer matters: the ligand-specific checks
        # below are meaningless on a protein chain, and the PTM/topology
        # cross-reference is meaningless on a ligand.
        is_ligand = n_res == 1 and len(coords) > 0 and len(coords) < 200
        chains[chain.name] = {
            "coords": np.array(coords, dtype=np.float32),
            "res_index": np.array(res_index, dtype=np.int32),
            "plddt": np.array(plddt, dtype=np.float32),
            "radii": np.array(radii, dtype=np.float32),
            "is_ligand": bool(is_ligand),
            "n_atoms": len(coords),
        }
    return chains


def _sasa(coords: np.ndarray, radii: np.ndarray, n_points: int = 242) -> np.ndarray:
    """Per-atom solvent-accessible surface area, Shrake-Rupley.

    Written here rather than taken from a library because the package has no
    structural dependency beyond gemmi and numpy, and adding one for a
    twenty-line calculation would be a worse trade.
    """
    if len(coords) == 0:
        return np.zeros(0, dtype=np.float32)
    # golden-spiral points on the unit sphere: an even spread without a mesh
    k = np.arange(n_points, dtype=np.float64)
    phi = np.arccos(1 - 2 * (k + 0.5) / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * (k + 0.5)
    sphere = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]

    expanded = radii + PROBE
    out = np.zeros(len(coords), dtype=np.float32)
    for i in range(len(coords)):
        d = np.linalg.norm(coords - coords[i], axis=1)
        near = np.where((d < expanded + expanded[i]) & (d > 1e-6))[0]
        test = coords[i] + sphere * expanded[i]
        if len(near):
            # a point is buried if it falls inside any neighbour's expanded sphere
            dist = np.linalg.norm(test[:, None, :] - coords[near][None, :, :], axis=2)
            accessible = ~(dist < expanded[near][None, :]).any(axis=1)
        else:
            accessible = np.ones(n_points, dtype=bool)
        out[i] = 4 * np.pi * expanded[i] ** 2 * accessible.mean()
    return out


def ligand_geometry(chains: dict, ligand_chain: str, topology_by_chain: dict) -> dict:
    """What a small molecule's pose actually looks like, beyond its confidence.

    AlphaFold3 places a small molecule in a receptor pocket confidently almost
    whichever receptor it is given, so the confidence cannot say whether the
    molecule belongs there. These measurements say something the confidence
    does not: whether it is enclosed by the protein at all, whether the
    geometry is physically possible, and which part of the receptor lines it.

    For a G-protein coupled receptor the last of these is the informative one.
    An orthosteric ligand sits inside the transmembrane bundle, so its
    contacts fall on Transmembrane segments. A pose whose contacts are only
    Cytoplasmic, or only on extracellular loops, is not in the orthosteric
    pocket however confident it looks.
    """
    lig = chains[ligand_chain]
    others = [n for n in chains if n != ligand_chain]
    if not others or lig["n_atoms"] == 0:
        return {}
    protein = {"coords": np.concatenate([chains[n]["coords"] for n in others]),
               "radii": np.concatenate([chains[n]["radii"] for n in others])}

    # burial: how much of the molecule's own surface the protein covers
    alone = _sasa(lig["coords"], lig["radii"]).sum()
    together = _sasa(np.concatenate([lig["coords"], protein["coords"]]),
                     np.concatenate([lig["radii"], protein["radii"]]))[:lig["n_atoms"]].sum()
    buried_fraction = float(1 - together / alone) if alone > 0 else None

    # clash: an un-relaxed pose can report many contacts at a separation no
    # real complex adopts, so the contact count is not readable without this
    d = np.linalg.norm(lig["coords"][:, None, :] - protein["coords"][None, :, :], axis=2)
    min_separation = float(d.min())

    out = {
        "ligand_chain": ligand_chain,
        "n_ligand_atoms": int(lig["n_atoms"]),
        "ligand_plddt_mean": round(float(lig["plddt"].mean()), 1),
        "ligand_plddt_min": round(float(lig["plddt"].min()), 1),
        "buried_fraction": round(buried_fraction, 3) if buried_fraction is not None else None,
        "sasa_alone_A2": round(float(alone), 1),
        "min_separation_A": round(min_separation, 2),
        "clash": min_separation < 1.5,
    }

    # which part of each protein chain lines the pocket
    pocket = {}
    for name in others:
        contacts = np.where((np.linalg.norm(
            lig["coords"][:, None, :] - chains[name]["coords"][None, :, :], axis=2)
            <= INTERFACE_CUTOFF_ANGSTROM).any(axis=0))[0]
        if not len(contacts):
            continue
        residues = set(chains[name]["res_index"][contacts].tolist())
        pocket[name] = {
            "n_contacting_residues": len(residues),
            "contacting_residues": sorted(residues),
            "topology": _overlapping_topology(topology_by_chain.get(name, []), residues),
        }
    out["pocket"] = pocket
    return out


def interface_residues(chain_a: dict, chain_b: dict, cutoff: float = INTERFACE_CUTOFF_ANGSTROM):
    """Residue numbers (1-indexed, per-chain) with >=1 heavy-atom contact
    within `cutoff` Angstrom of the other chain."""
    a_coords, b_coords = chain_a["coords"], chain_b["coords"]
    if len(a_coords) == 0 or len(b_coords) == 0:
        return set(), set()
    a_res, b_res = chain_a["res_index"], chain_b["res_index"]
    contacts_a, contacts_b = set(), set()
    chunk = 2000  # bound memory for the pairwise distance block
    for start in range(0, len(a_coords), chunk):
        block = a_coords[start : start + chunk]
        d = np.sqrt(((block[:, None, :] - b_coords[None, :, :]) ** 2).sum(-1))
        rows, cols = np.where(d <= cutoff)
        contacts_a.update(a_res[start : start + chunk][rows].tolist())
        contacts_b.update(b_res[cols].tolist())
    return contacts_a, contacts_b


def fetch_uniprot_features(accession: str) -> list:
    with httpx.Client() as client:
        resp = get_with_retry(client, f"https://rest.uniprot.org/uniprotkb/{accession}.json", {}, 30.0)
    return resp.json().get("features", [])


def _filter_features(features: list, types: set) -> list:
    sites = []
    for f in features:
        if f["type"] in types:
            loc = f["location"]
            sites.append(
                {
                    "type": f["type"],
                    "start": loc["start"]["value"],
                    "end": loc["end"]["value"],
                    "description": f.get("description"),
                }
            )
    return sites


def _overlapping_topology(topology: list, residues: set) -> list:
    """Which topological domain(s)/transmembrane span(s) the given residues
    fall in — empty list means the protein has no annotated membrane
    topology at all (i.e. topology just doesn't apply, not "cytoplasmic")."""
    hits = []
    for t in topology:
        if any(t["start"] <= r <= t["end"] for r in residues):
            label = t["description"] if t["type"] == "Topological domain" else t["type"]
            hits.append(label)
    return sorted(set(hits))


def _plddt_stats(chain: dict, residues: set) -> dict:
    if not residues:
        return None
    mask = np.isin(chain["res_index"], list(residues))
    vals = chain["plddt"][mask]
    return {"mean": round(float(vals.mean()), 1), "min": round(float(vals.min()), 1)} if len(vals) else None


def _overlapping_ptms(sites: list, residues: set) -> list:
    return [s for s in sites if any(s["start"] <= r <= s["end"] for r in residues)]


def analyze(cif_path: str, chain_accessions: dict) -> dict:
    """chain_accessions: {'A': 'Q01196', 'B': 'Q13951', ...}"""
    chains = load_chain_atoms(cif_path)
    names = list(chains.keys())
    report = {"cif_path": str(cif_path), "chains": {}, "interfaces": []}
    topology_by_chain = {}

    for name in names:
        acc = chain_accessions.get(name)
        features = fetch_uniprot_features(acc) if acc else []
        topology_by_chain[name] = _filter_features(features, TOPOLOGY_FEATURE_TYPES)
        report["chains"][name] = {
            "accession": acc,
            "is_ligand": chains[name].get("is_ligand", False),
            "ptm_sites": _filter_features(features, PTM_FEATURE_TYPES),
            "n_residues": len(set(chains[name]["res_index"].tolist())),
            "n_atoms": chains[name].get("n_atoms"),
        }

    # a small-molecule chain gets its own geometry: burial, clash and which
    # part of the receptor lines the pose
    for name in names:
        if chains[name].get("is_ligand"):
            geometry = ligand_geometry(chains, name, topology_by_chain)
            if geometry:
                report.setdefault("ligands", []).append(geometry)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res_a, res_b = interface_residues(chains[a], chains[b])
            if not res_a and not res_b:
                continue
            report["interfaces"].append(
                {
                    "chain_a": a,
                    "chain_b": b,
                    "n_interface_residues_a": len(res_a),
                    "n_interface_residues_b": len(res_b),
                    "interface_residues_a": sorted(res_a),
                    "interface_residues_b": sorted(res_b),
                    "interface_plddt_a": _plddt_stats(chains[a], res_a),
                    "interface_plddt_b": _plddt_stats(chains[b], res_b),
                    "ptm_sites_at_interface_a": _overlapping_ptms(report["chains"][a]["ptm_sites"], res_a),
                    "ptm_sites_at_interface_b": _overlapping_ptms(report["chains"][b]["ptm_sites"], res_b),
                    # Which side of the membrane the interface sits on (empty
                    # list = chain has no annotated membrane topology at all).
                    "interface_topology_a": _overlapping_topology(topology_by_chain[a], res_a),
                    "interface_topology_b": _overlapping_topology(topology_by_chain[b], res_b),
                }
            )
    return report


def summarize(report: dict) -> dict:
    """Condensed version for token-budget-sensitive callers (the agent's own
    fold_complex tool response) — counts, PTM overlaps and ligand geometry,
    not full residue lists."""
    out = {
        "interfaces": [
            {
                "chain_a": i["chain_a"],
                "chain_b": i["chain_b"],
                "n_interface_residues_a": i["n_interface_residues_a"],
                "n_interface_residues_b": i["n_interface_residues_b"],
                "interface_plddt_a": i["interface_plddt_a"],
                "interface_plddt_b": i["interface_plddt_b"],
                "ptm_sites_at_interface_a": i["ptm_sites_at_interface_a"],
                "ptm_sites_at_interface_b": i["ptm_sites_at_interface_b"],
                "interface_topology_a": i["interface_topology_a"],
                "interface_topology_b": i["interface_topology_b"],
            }
            for i in report["interfaces"]
        ]
    }
    # the ligand block keeps the per-chain topology of the pocket, which is the
    # part that says whether the molecule is in the orthosteric site, but drops
    # the full contacting-residue lists
    if report.get("ligands"):
        out["ligands"] = [
            {**{k: v for k, v in g.items() if k != "pocket"},
             "pocket": {chain: {"n_contacting_residues": p["n_contacting_residues"],
                                "topology": p["topology"]}
                        for chain, p in (g.get("pocket") or {}).items()}}
            for g in report["ligands"]
        ]
    return out


def analyze_fold_result(result: dict) -> dict:
    """Convenience entry point: takes a fold_candidate.fold()-shaped dict
    (has `chains` with identifier/accession, and `model_cif`) directly."""
    chain_accessions = {
        CHAIN_LETTERS[i]: c["accession"] for i, c in enumerate(result["chains"]) if c.get("accession")
    }
    return analyze(result["model_cif"], chain_accessions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir_or_json", nargs="?", help="A fold_candidate.py output dir (contains <name>_model.cif) or a *_result.json from run_follow_up_folds.py")
    ap.add_argument("--cif", help="Explicit path to a model .cif (alternative to result_dir_or_json)")
    ap.add_argument("--chains", help="Explicit chain->accession map, e.g. 'A=Q01196,B=Q13951' (required with --cif)")
    args = ap.parse_args()

    if args.cif:
        chain_accessions = dict(pair.split("=") for pair in (args.chains or "").split(",") if pair)
        report = analyze(args.cif, chain_accessions)
    else:
        target = Path(args.result_dir_or_json)
        if target.is_file() and target.suffix == ".json":
            result = json.loads(target.read_text())
        else:
            # a fold_candidate.py output dir: <name>/<name>_model.cif + infer chains from dir name is not possible,
            # so require the sibling *_result.json (run_follow_up_folds.py) or fall back to no PTM cross-ref.
            name = target.name
            cif = target / f"{name}_model.cif"
            if not cif.exists():
                print(f"No model .cif found at {cif}", file=sys.stderr)
                sys.exit(1)
            report = analyze(cif, {})
            print(json.dumps(report, indent=2))
            return
        report = analyze_fold_result(result)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
