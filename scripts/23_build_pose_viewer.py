"""Extract the structures behind the pose argument, for an interactive viewer.

The claim this project makes about a small-molecule candidate is geometric: the
molecule is enclosed inside the seven-helix bundle, on the extracellular side,
the way every known cognate ligand in the benchmark is. A table of buried
fractions asks the reader to take that on trust. The structures do not.

So this writes, for each chosen fold, a PDB the browser can load plus the
metadata needed to colour it: which receptor residues line the pocket, which
transmembrane span each residue belongs to, and the pose numbers already
measured. Built to the shape of the thy_tf_dna viewer, which pairs a high and a
low case side by side so the difference is visible rather than asserted.

The set is chosen to be falsifiable, not flattering: two known cognate poses,
two known NON-binders from the same benchmark, and the candidates. If the
candidates did not look like the cognates the viewer would show that.

Usage:
    micromamba run -n structmod python -I scripts/23_build_pose_viewer.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import gemmi
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("viewer")
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "results" / "pose_viewer"
CUT = 4.5          # contact distance, angstroms

# job, stage, role, and the one-line reason it is in the set
PANELS = [
    ("CNR1__anandamide", "specificity", "cognate",
     "a known cognate ligand: anandamide in CNR1"),
    ("ADRB2__adrenaline", "specificity", "cognate",
     "adrenaline in ADRB2, the adrenoceptor noradrenaline really binds"),
    ("P2RY12__acetylcholine", "specificity", "non-binder",
     "a known non-binder: acetylcholine at P2RY12, outside the bundle entirely"),
    ("DRD2__adenosine-diphosphate", "specificity", "non-binder",
     "a known non-binder lining the intracellular face"),
    ("GPR85__DBH", "orphan", "candidate", "GPR85 with noradrenaline"),
    ("GPR153__DBH", "orphan", "candidate", "GPR153 with noradrenaline"),
    ("GPR26__SLC1A2", "orphan", "candidate",
     "GPR26 with glutamate: a good pose in a pocket that also takes two other molecules"),
    ("GPR150__GAD2", "orphan", "rejected",
     "GPR150 with GABA: enclosed only 0.418, resting on the surface"),
]
STAGE_OUT = {"specificity": "specificity_out", "orphan": "orphan_out"}
STAGE_TOPO = {"specificity": "specificity_topology.json", "orphan": "orphan_topology.json"}
STAGE_POSE = {"specificity": "specificity_pose.csv", "orphan": "orphan_pose.csv"}


def span_of(topology: list, resnum: int):
    """Which topological domain a residue sits in, as a short label."""
    for t in topology:
        try:
            a, b = int(t.get("start")), int(t.get("end"))
        except (TypeError, ValueError):
            continue
        if a <= resnum <= b:
            d = str(t.get("description") or t.get("type") or "")
            if "Transmembrane" in str(t.get("type", "")) or "Helical" in d:
                return "TM"
            if "Cytoplasm" in d:
                return "CYT"
            if "Extracellular" in d:
                return "EXT"
            return "OTHER"
    return None


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "s").mkdir(exist_ok=True)
    topo_cache = {st: json.loads((ROOT / "results" / f).read_text())
                  if (ROOT / "results" / f).exists() else {}
                  for st, f in STAGE_TOPO.items()}
    pose_cache = {st: pd.read_csv(ROOT / "results" / f).set_index("job")
                  for st, f in STAGE_POSE.items()
                  if (ROOT / "results" / f).exists()}

    meta = {}
    for job, stage, role, why in PANELS:
        cif = ROOT / "results" / STAGE_OUT[stage] / job / f"{job}_model.cif"
        if not cif.exists():
            log.warning("%s: no CIF", job)
            continue
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        model = st[0]

        # the ligand is the chain with one residue and few atoms; everything else
        # is the receptor
        lig_chain = rec_chain = None
        for ch in model:
            n_res = len(ch)
            n_atom = sum(len(r) for r in ch)
            if n_res == 1 and 0 < n_atom < 200:
                lig_chain = ch.name
            else:
                rec_chain = ch.name
        if lig_chain is None or rec_chain is None:
            log.warning("%s: could not identify a ligand chain", job)
            continue

        ns = gemmi.NeighborSearch(st, 5.0).populate()
        contacts = set()
        for res in model[lig_chain]:
            for atom in res:
                for m in ns.find_atoms(atom.pos, '\0', radius=CUT):
                    cra = m.to_cra(model)
                    if cra.chain.name == rec_chain:
                        contacts.add(cra.residue.seqid.num)

        receptor = job.split("__")[0]
        topology = topo_cache.get(stage, {}).get(receptor, [])
        spans = {}
        for n in sorted(contacts):
            s_ = span_of(topology, n)
            if s_:
                spans.setdefault(s_, []).append(n)

        st.remove_alternative_conformations()
        st.remove_hydrogens()
        out = DEST / "s" / f"{job}.pdb"
        out.write_text(st.make_pdb_string())

        p = pose_cache.get(stage)
        row = p.loc[job] if p is not None and job in p.index else None
        meta[job] = {
            "file": f"s/{job}.pdb",
            "role": role, "why": why, "stage": stage,
            "receptor": receptor, "receptor_chain": rec_chain, "ligand_chain": lig_chain,
            "contacts": sorted(contacts),
            "n_contacts": len(contacts),
            "spans": {k: v for k, v in spans.items()},
            "tm_contacts": len(spans.get("TM", [])),
            "cyt_contacts": len(spans.get("CYT", [])),
            "buried_fraction": None if row is None else round(float(row.buried_fraction), 3),
            "iptm": None if row is None else round(float(row.iptm), 2),
            # every transmembrane span, so the bundle can be drawn even where the
            # ligand does not touch it
            "tm_spans": [[int(t["start"]), int(t["end"])] for t in topology
                         if "Transmembrane" in str(t.get("type", ""))
                         and str(t.get("start", "")).isdigit()],
        }
        log.info("%-32s %-10s burial %-6s contacts %3d (TM %2d, cytoplasmic %d)",
                 job, role, meta[job]["buried_fraction"], len(contacts),
                 meta[job]["tm_contacts"], meta[job]["cyt_contacts"])

    (DEST / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    tot = sum((DEST / "s" / f"{j}.pdb").stat().st_size for j in meta)
    log.info("\nwrote %d structures to %s (%.1f MB) and meta.json",
             len(meta), DEST, tot / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
