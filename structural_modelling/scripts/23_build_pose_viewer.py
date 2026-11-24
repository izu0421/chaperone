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

The set is built from the published verdicts, not typed here, so it cannot
describe a shortlist the project no longer holds: every protein pair the agent
confirmed and every metabolite pair that clears the calibrated gate. Around them
sit fixed controls chosen to be falsifiable, not flattering -- two known cognate
poses, two known NON-binders from the same benchmark, and two folds the gate
rejected on topology. If the candidates did not look like the cognates, or
looked like the rejects, the viewer would show that.

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
# The structures are read from here, where the 374 GB of fold output lives, and
# written into the chaperone repo, which carries the published copy. There is no
# build copy in between: that would be the same 2.2 MB stored twice, and two
# copies of anything is an invitation to edit the stale one.
DEST = Path("/data/yzy21/yy/space_agent/chaperone/reports/"
            "orphan_shortlist_2026-11-23/structures")
CUT = 4.5          # contact distance, angstroms

# Fixed controls. These do not come from the verdicts, because their job is to
# be the fixed points the candidates are read against.
CONTROLS = [
    ("CNR1__anandamide", "specificity", "cognate",
     "a known cognate ligand: anandamide in CNR1", "CNR1 + anandamide"),
    ("ADRB2__adrenaline", "specificity", "cognate",
     "adrenaline in ADRB2, the adrenoceptor noradrenaline really binds",
     "ADRB2 + adrenaline"),
    ("P2RY12__acetylcholine", "specificity", "non-binder",
     "a known non-binder: acetylcholine at P2RY12, outside the bundle entirely",
     "P2RY12 + acetylcholine"),
    ("DRD2__adenosine-diphosphate", "specificity", "non-binder",
     "a known non-binder lining the intracellular face", "DRD2 + ADP"),
    ("GPR82__FN1", "orphan", "rejected",
     "fibronectin is secreted, yet all 31 interface residues are cytoplasmic",
     "GPR82 + fibronectin"),
    ("MRGPRX2__GNRH2", "orphan", "rejected",
     "the interface spans both faces at once, which no intact membrane allows",
     "MRGPRX2 + GNRH2"),
    ("GPR52__SLIT2", "orphan", "rejected",
     "SLIT2 is secreted and every one of its 19 interface residues is cytoplasmic: "
     "the fold the topology rule was added to catch", "GPR52 + SLIT2"),
]

REPORT = Path("/data/yzy21/yy/space_agent/chaperone/reports/"
              "orphan_shortlist_2026-11-23")


def _molecule_label(name: str) -> str:
    """Lowercase an ordinary molecule name, leave an acronym alone.

    A blanket .lower() turned GABA into "gaba" and would do the same to PGD2
    and 5-HT. Only a plain Titlecase word is a common noun here.
    """
    return name.lower() if name[:1].isupper() and name[1:].islower() else name


def candidate_panels():
    """Every plausible candidate, read from the files the report publishes.

    Two sources, because the two arms are judged differently. Protein pairs carry
    an agent verdict, so the plausible ones are those it confirmed. Metabolite
    pairs are not judged by the agent at all -- it called every one a promiscuous
    pocket -- so the plausible ones are those clearing the calibrated gate.
    """
    panels = []

    verdicts = pd.read_csv(REPORT / "verdicts.csv")
    confirmed = verdicts[verdicts.verdict == "CONFIRMED_NOVEL"]
    for row in confirmed.sort_values("iptm", ascending=False).itertuples():
        panels.append((f"{row.protein_b}__{row.protein_a}", "orphan", "recommended",
                       f"{row.protein_a} against {row.protein_b}: interface pTM "
                       f"{row.iptm:.2f}, agent confidence {row.confidence}",
                       f"{row.protein_b} + {row.protein_a}"))

    # the metabolite arm: one fold per receptor-molecule pair, the best-scoring
    # producer gene, because the same molecule is reached through several
    hits = pd.read_csv(ROOT / "results" / "metabolite_calibrated_hits.csv")
    scored = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    ident = (sheet.loc[sheet.partner_kind == "metabolite",
                       ["receptor_gene", "partner_gene", "metabolite"]]
             .rename(columns={"receptor_gene": "receptor", "partner_gene": "partner"})
             .drop_duplicates())
    scored = scored.merge(ident, on=["receptor", "partner"], how="left")
    for row in hits.itertuples():
        same = scored[(scored.receptor == row.receptor)
                      & (scored.metabolite == row.metabolite)]
        if same.empty:
            log.warning("%s + %s: no fold row", row.receptor, row.metabolite)
            continue
        best = same.loc[same.iptm.idxmax()]
        panels.append((best.pair, "orphan", "candidate",
                       f"{row.metabolite} in {row.receptor}: interface pTM "
                       f"{row.iptm:.2f}, ligand pLDDT {row.plddt_B:.0f}",
                       f"{row.receptor} + {_molecule_label(row.metabolite)}"))
    return panels


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

    panels = CONTROLS + candidate_panels()
    seen, ordered = set(), []
    for item in panels:                       # a control may also be a candidate
        if item[0] not in seen:
            seen.add(item[0]); ordered.append(item)
    log.info("%d panels: %s", len(ordered),
             ", ".join(f"{r} {sum(1 for p in ordered if p[2] == r)}"
                       for r in dict.fromkeys(p[2] for p in ordered)))

    meta = {}
    for job, stage, role, why, label in ordered:
        cif = ROOT / "results" / STAGE_OUT[stage] / job / f"{job}_model.cif"
        if not cif.exists():
            log.warning("%s: no CIF", job)
            continue
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        model = st[0]

        # the ligand is the chain with one residue and few atoms; everything else
        # is the receptor
        # a small molecule is one residue; a protein partner is a whole chain, and
        # for those the SECOND chain is the partner by the screen's own convention
        lig_chain = rec_chain = None
        for ch in model:
            if len(ch) == 1 and 0 < sum(len(r) for r in ch) < 200:
                lig_chain = ch.name
            else:
                rec_chain = ch.name
        is_ppi = lig_chain is None
        if is_ppi:
            names = [c.name for c in model]
            if len(names) != 2:
                log.warning("%s: %d chains, skipping", job, len(names))
                continue
            rec_chain, lig_chain = names[0], names[1]
        if lig_chain is None or rec_chain is None:
            log.warning("%s: could not identify the partner chain", job)
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

        # Which way is "down". The viewer colours cytoplasmic residues red, but in an
        # arbitrary 3D orientation a reader cannot tell the intracellular side from
        # the extracellular one, and that is the whole distinction the pose argument
        # rests on. So the axis is measured here, from the centroids of the annotated
        # domains, and the viewer draws it rather than leaving it to be inferred.
        def _centroid(kind):
            pts = []
            for res in model[rec_chain]:
                if span_of(topology, res.seqid.num) != kind:
                    continue
                for atom in res:
                    if atom.name == "CA":
                        pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
            if not pts:
                return None
            n = len(pts)
            return [sum(c[i] for c in pts) / n for i in range(3)]

        cyt_c, ext_c = _centroid("CYT"), _centroid("EXT")

        st.remove_alternative_conformations()
        st.remove_hydrogens()
        out = DEST / "s" / f"{job}.pdb"
        out.write_text(st.make_pdb_string())

        p = pose_cache.get(stage)
        row = p.loc[job] if p is not None and job in p.index else None
        meta[job] = {
            "file": f"s/{job}.pdb",
            "role": role, "why": why, "stage": stage, "label": label,
            "receptor": receptor, "receptor_chain": rec_chain, "ligand_chain": lig_chain,
            "contacts": sorted(contacts),
            "n_contacts": len(contacts),
            "spans": {k: v for k, v in spans.items()},
            "tm_contacts": len(spans.get("TM", [])),
            "cyt_contacts": len(spans.get("CYT", [])),
            "is_ppi": bool(is_ppi),
            "ext_contacts": len(spans.get("EXT", [])),
            "spans_both_faces": bool(spans.get("EXT") and spans.get("CYT")),
            "buried_fraction": None if row is None else round(float(row.buried_fraction), 3),
            "iptm": None if row is None else round(float(row.iptm), 2),
            # every transmembrane span, so the bundle can be drawn even where the
            # ligand does not touch it
            # the membrane normal, as two points for the viewer to draw between
            "axis": ({"ext": [round(v, 2) for v in ext_c],
                      "cyt": [round(v, 2) for v in cyt_c]}
                     if cyt_c and ext_c else None),
            "tm_spans": [[int(t["start"]), int(t["end"])] for t in topology
                         if "Transmembrane" in str(t.get("type", ""))
                         and str(t.get("start", "")).isdigit()],
        }
        log.info("%-32s %-10s burial %-6s contacts %3d (TM %2d, cytoplasmic %d)",
                 job, role, meta[job]["buried_fraction"], len(contacts),
                 meta[job]["tm_contacts"], meta[job]["cyt_contacts"])
        if not meta[job]["axis"]:
            log.warning("  %s: no membrane axis -- the viewer cannot show which side "
                        "is intracellular for this one", job)

    # Drop structures the current verdicts no longer include. Without this the
    # published directory accumulates every pair any earlier run recommended, and
    # a reader who guesses a filename gets a fold the project has since rejected.
    stale = [f for f in (DEST / "s").glob("*.pdb") if f.stem not in meta]
    for f in stale:
        f.unlink()
    if stale:
        log.info("removed %d structure(s) no longer in the shortlist: %s",
                 len(stale), ", ".join(sorted(f.stem for f in stale)))

    (DEST / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    tot = sum((DEST / "s" / f"{j}.pdb").stat().st_size for j in meta)
    log.info("\nwrote %d structures to %s (%.1f MB) and meta.json",
             len(meta), DEST, tot / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
