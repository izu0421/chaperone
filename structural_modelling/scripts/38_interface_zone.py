"""Where on the receptor does each candidate interface sit, and on the partner?

The screen's old topology rule used this question as a veto and was wrong to:
measured on the chemokine panel it rejects 61% of KNOWN binders, because
AlphaFold3 routinely puts a correct partner slightly off the correct face.
Deleting the rule was right. Deleting the DESCRIPTION was not -- a membrane
protein's interface zone says what KIND of interaction is being proposed, and
three kinds are on this list with nothing distinguishing them:

  ecto-ecto     the ordinary ligand-receptor case
  TM-TM         a transmembrane association. For an adaptor this is the real
                mechanism, not an artefact: HCST (DAP10) has 30 extracellular
                residues and signals by pairing with KLRK1 through the membrane,
                so a GPR65 interface in the TM zone is a hypothesis about the
                same mechanism, and the ectodomain arm cannot test it.
  cytoplasmic   impossible for a secreted partner, and for a membrane partner
                it means the tail is doing the binding -- which is what arm P
                removes.

Zones come from UniProt topology: the receptor's from results/orphan_topology
.json, the partner's asked per protein. A residue in no annotated segment is
counted as unassigned and reported, never silently folded into a zone.

Usage:
    micromamba run -n structmod python -I scripts/38_interface_zone.py
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("zone")
ROOT = Path(__file__).resolve().parent.parent
CUT = 5.0
CACHE = ROOT / "work" / "partner_topology.json"


def zones_from_features(feats: list) -> list[tuple[int, int, str]]:
    """Accepts both shapes: UniProt's nested 'location' and the flat
    {type,start,end,description} the screen's own topology cache stores."""
    out = []
    for f in feats:
        t = f.get("type")
        d = str(f.get("description") or "")
        if t == "Transmembrane":
            z = "TM"
        elif t == "Topological domain" and "xtracellular" in d:
            z = "extracellular"
        elif t == "Topological domain" and "ytoplasm" in d:
            z = "cytoplasmic"
        elif t == "Signal":
            z = "signal"
        elif t == "Intramembrane":
            z = "TM"
        else:
            continue
        if "location" in f:
            loc = f["location"]
            s, e = int(loc["start"]["value"]), int(loc["end"]["value"])
        elif "start" in f and "end" in f:
            s, e = int(f["start"]), int(f["end"])
        else:
            continue
        out.append((s, e, z))
    return sorted(out)


def partner_zones(acc: str, cache: dict) -> list[tuple[int, int, str]]:
    if acc in cache:
        return [tuple(x) for x in cache[acc]]
    url = (f"https://rest.uniprot.org/uniprotkb/{acc}.json"
           "?fields=ft_topo_dom,ft_transmem,ft_signal,ft_intramem")
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "research-script/1.0"}), timeout=30).read())
        z = zones_from_features(d.get("features", []))
    except Exception as exc:                                    # noqa: BLE001
        log.warning("  %s: topology lookup failed (%s)", acc, str(exc)[:40])
        z = []
    cache[acc] = z
    return z


def zone_of(res: int, segs: list[tuple[int, int, str]]) -> str:
    for a, b, z in segs:
        if a <= res <= b:
            return z
    return "unassigned"


def interface_residues(cif: Path):
    from Bio.PDB import MMCIFParser
    from scipy.spatial import cKDTree
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = MMCIFParser(QUIET=True).get_structure(cif.stem, str(cif))
    ch = list(next(iter(st)))
    if len(ch) < 2:
        return None
    aa = [a for a in ch[0].get_atoms() if a.element != "H"]
    ab = [a for a in ch[1].get_atoms() if a.element != "H"]
    t = cKDTree(np.array([a.coord for a in ab]))
    pairs = t.query_ball_point(np.array([a.coord for a in aa]), CUT)
    resA = {aa[i].get_parent().id[1] for i, p in enumerate(pairs) if p}
    resB = {ab[j].get_parent().id[1] for p in pairs for j in p}
    return resA, resB


def offsets(man_row) -> tuple[int, int]:
    """How far each chain's residue 1 sits from UniProt residue 1.

    A cut chain is renumbered from 1 by AlphaFold3, so its zone lookup needs the
    span's start added back. Reading the cut construct against UniProt numbering
    without this puts every contact in the wrong domain -- and it would do it
    quietly, because the numbers still land inside the protein.
    """
    def start(v) -> int:
        s = str(v)
        return int(s.split("-")[0]) - 1 if s and s != "full" and "-" in s else 0
    return start(getattr(man_row, "rec_ecd", "full")), start(getattr(man_row, "ecd", "full"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("screen", "ectodomain", "mature"), default="screen",
                    help="screen: the folds the shortlist came from. ectodomain: arm P. "
                         "mature: arm M. In both re-fold arms a chain is cut and its "
                         "numbering is offset, and in both the RECEPTOR's cytoplasmic "
                         "face is still present, so a surviving interface can still be "
                         "on a face the partner cannot reach")
    a = ap.parse_args()
    if a.arm == "ectodomain":
        return ectodomain_arm()
    if a.arm == "mature":
        return mature_arm()
    t = pd.read_csv(ROOT / "results" / "candidate_ranking.csv")
    topo = json.loads((ROOT / "results" / "orphan_topology.json").read_text())
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.partner_gene, sheet.partner_uniprot))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    rows = []
    for r in t.itertuples():
        d = ROOT / "results" / "orphan_out" / f"{r.receptor}__{r.partner}"
        cif = sorted(d.glob("*_model.cif"))
        if not cif:
            log.warning("%s+%s: no model", r.receptor, r.partner); continue
        iface = interface_residues(cif[0])
        if not iface:
            continue
        resA, resB = iface
        zA = zones_from_features(topo.get(r.receptor, []))
        zB = partner_zones(acc.get(r.partner, ""), cache) if r.partner in acc else []
        cA = pd.Series([zone_of(x, zA) for x in sorted(resA)]).value_counts()
        cB = pd.Series([zone_of(x, zB) for x in sorted(resB)]).value_counts()
        tot_a, tot_b = max(1, cA.sum()), max(1, cB.sum())
        rec_zone = cA.idxmax() if len(cA) else "none"
        par_zone = cB.idxmax() if len(cB) else "none"
        # A partner with no transmembrane segment anywhere is soluble; UniProt
        # annotates no topological domain for it, so "unassigned" there means
        # soluble and not missing data.
        soluble = not any(z == "TM" for _, _, z in zB)
        f = lambda c, k: c.get(k, 0) / max(1, c.sum())          # noqa: E731
        if soluble:
            # For a secreted ligand the receptor zone is the whole question. TM
            # contacts reached from outside ARE the orthosteric pocket -- that is
            # where every peptide GPCR ligand binds -- while cytoplasmic contacts
            # are a place the ligand cannot reach at all.
            if f(cA, "cytoplasmic") >= 0.3:
                kind = "IMPOSSIBLE cytoplasmic"
            elif f(cA, "TM") >= 0.2:
                kind = "orthosteric pocket"
            else:
                kind = "extracellular surface"
        else:
            if f(cA, "cytoplasmic") >= 0.3 or f(cB, "cytoplasmic") >= 0.3:
                kind = "tail-mediated"
            elif f(cA, "TM") >= 0.5 and f(cB, "TM") >= 0.3:
                kind = "TM association"
            elif f(cA, "extracellular") >= 0.3 and f(cB, "extracellular") >= 0.3:
                kind = "ecto-ecto"
            else:
                kind = f"{rec_zone}-{par_zone}"
        rows.append({"receptor": r.receptor, "partner": r.partner, "kind": kind,
                     "rec_zone": rec_zone, "par_zone": par_zone,
                     "rec_extra": cA.get("extracellular", 0) / tot_a,
                     "rec_tm": cA.get("TM", 0) / tot_a,
                     "rec_cyto": cA.get("cytoplasmic", 0) / tot_a,
                     "rec_unassigned": cA.get("unassigned", 0) / tot_a,
                     "par_extra": cB.get("extracellular", 0) / tot_b,
                     "par_tm": cB.get("TM", 0) / tot_b,
                     "par_cyto": cB.get("cytoplasmic", 0) / tot_b,
                     "par_signal": cB.get("signal", 0) / tot_b,
                     "par_unassigned": cB.get("unassigned", 0) / tot_b,
                     "partner_soluble": soluble,
                     "n_res_A": len(resA), "n_res_B": len(resB)})
    CACHE.write_text(json.dumps(cache))
    z = pd.DataFrame(rows)
    z.to_csv(ROOT / "results" / "candidate_interface_zone.csv", index=False)

    log.info("INTERFACE ZONE, as the screen folded it (fraction of interface residues)")
    log.info("  for a SOLUBLE partner, receptor TM contacts reached from outside are the")
    log.info("  orthosteric pocket; receptor cytoplasmic contacts are unreachable.\n")
    log.info("  %-20s %-16s  receptor: %5s %5s %5s   partner: %5s %5s %5s %5s",
             "pair", "kind", "ecto", "TM", "cyto", "ecto", "TM", "cyto", "sig")
    for r in z.sort_values(["kind", "receptor"]).itertuples():
        log.info("  %-20s %-16s            %5.2f %5.2f %5.2f            %5.2f %5.2f %5.2f %5.2f",
                 f"{r.receptor}+{r.partner}", r.kind, r.rec_extra, r.rec_tm, r.rec_cyto,
                 r.par_extra, r.par_tm, r.par_cyto, r.par_signal)
    log.info("\n%s", z.kind.value_counts().to_string())
    log.info("\n  'tail-mediated' is not 'impossible': two membrane proteins in the same")
    log.info("  membrane have both tails in the cytosol, so a cytoplasmic interface could")
    log.info("  exist. It is a WEAKER claim, because the tails are disordered and that is")
    log.info("  where AlphaFold3 docks anything, and the ectodomain arm cannot test it.")
    log.info("  'IMPOSSIBLE cytoplasmic' is reserved for a SECRETED partner, which has no")
    log.info("  route to the cytosol at all.")
    # a soluble partner has no topology to assign, so only unassigned RECEPTOR
    # residues, or a membrane partner's, mean the zone was guessed
    una = z[(z.rec_unassigned > 0.3) | ((~z.partner_soluble) & (z.par_unassigned > 0.3))]
    if len(una):
        log.info("\n%d pairs have >30%% of an interface in residues UniProt does not "
                 "assign to any segment -- their zone is a guess:", len(una))
        for r in una.itertuples():
            log.info("   %s+%s  receptor %.0f%% partner %.0f%% unassigned",
                     r.receptor, r.partner, 100 * r.rec_unassigned, 100 * r.par_unassigned)
    log.info("\nwrote results/candidate_interface_zone.csv")


def mature_arm() -> None:
    """Arm M: the same question the ectodomain arm needed asking.

    Cutting the signal peptide removes 20-26 residues from the partner. It does
    nothing to the receptor, so a mature chain that still clears the gate may be
    clearing it on the cytoplasmic face -- which for a SECRETED partner is not a
    weaker claim but an impossible one.
    """
    man = pd.read_csv(ROOT / "work" / "subcomplex_manifest.csv")
    man = man[man.arm.eq("M_mature")]
    topo = json.loads((ROOT / "results" / "orphan_topology.json").read_text())
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.partner_gene, sheet.partner_uniprot))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    for r in man.itertuples():
        d = ROOT / "results" / "subcomplex_out" / r.job
        cif = sorted(d.glob("*_model.cif")) or sorted(d.glob("seed-1_sample-0/*_model.cif"))
        if not cif:
            continue
        iface = interface_residues(cif[0])
        if not iface:
            continue
        resA, resB = iface
        # "signal 1-23 removed" -> every partner residue is 23 higher in UniProt
        offB = 0
        ex = str(getattr(r, "extras", ""))
        if "signal 1-" in ex:
            offB = int(ex.split("signal 1-")[1].split(" ")[0])
        zA = zones_from_features(topo.get(r.receptor, []))
        zB = partner_zones(acc.get(r.partner, ""), cache) if r.partner in acc else []
        cA = pd.Series([zone_of(x, zA) for x in sorted(resA)]).value_counts()
        cB = pd.Series([zone_of(x + offB, zB) for x in sorted(resB)]).value_counts()
        f = lambda c, k: c.get(k, 0) / max(1, c.sum())          # noqa: E731
        rows.append({"job": r.job, "receptor": r.receptor, "partner": r.partner,
                     "signal_removed": offB,
                     "rec_extra": f(cA, "extracellular"), "rec_tm": f(cA, "TM"),
                     "rec_cyto": f(cA, "cytoplasmic"),
                     "partner_soluble": not any(z == "TM" for _, _, z in zB),
                     "n_res_A": len(resA), "n_res_B": len(resB)})
    CACHE.write_text(json.dumps(cache))
    if not rows:
        raise SystemExit("no arm-M folds scored yet")
    z = pd.DataFrame(rows)
    z.to_csv(ROOT / "results" / "mature_interface_zone.csv", index=False)
    log.info("ARM M -- where the interface sits once the signal peptide is gone\n")
    log.info("  %-28s %-9s  receptor: %5s %5s %5s", "job", "partner", "ecto", "TM", "cyto")
    for r in z.sort_values("rec_cyto").itertuples():
        flag = ""
        if r.rec_cyto >= 0.3:
            flag = ("  <- IMPOSSIBLE, the partner is secreted" if r.partner_soluble
                    else "  <- cytoplasmic")
        log.info("  %-28s %-9s            %5.2f %5.2f %5.2f%s", r.job,
                 "soluble" if r.partner_soluble else "membrane",
                 r.rec_extra, r.rec_tm, r.rec_cyto, flag)
    bad = z[z.partner_soluble & z.rec_cyto.ge(0.3)]
    log.info("\n%d of %d folds put a SECRETED partner on the receptor's cytoplasmic face",
             len(bad), len(z))
    log.info("wrote results/mature_interface_zone.csv")


def ectodomain_arm() -> None:
    """Arm P: has the interface moved to a face the partner can actually reach?

    A candidate that beats its permuted negatives in arm P has shown only that
    SOMETHING docks better than a size-matched stranger. If the surviving
    interface still sits on the receptor's cytoplasmic face, the construct
    removed the partner's impossible half and the receptor's is still there.
    """
    man = pd.read_csv(ROOT / "work" / "ectodomain_manifest.csv")
    topo = json.loads((ROOT / "results" / "orphan_topology.json").read_text())
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.partner_gene, sheet.partner_uniprot))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    for r in man.itertuples():
        d = ROOT / "results" / "ectodomain_out" / r.job
        cif = sorted(d.glob("*_model.cif")) or sorted(d.glob("seed-*/*_model.cif"))
        if not cif:
            continue
        iface = interface_residues(cif[0])
        if not iface:
            continue
        resA, resB = iface
        offA, offB = offsets(r)
        zA = zones_from_features(topo.get(r.receptor, []))
        zB = partner_zones(acc.get(r.partner, ""), cache) if r.partner in acc else []
        cA = pd.Series([zone_of(x + offA, zA) for x in sorted(resA)]).value_counts()
        cB = pd.Series([zone_of(x + offB, zB) for x in sorted(resB)]).value_counts()
        f = lambda c, k: c.get(k, 0) / max(1, c.sum())          # noqa: E731
        rows.append({"job": r.job, "arm": r.arm, "receptor": r.receptor,
                     "partner": r.partner,
                     "rec_extra": f(cA, "extracellular"), "rec_tm": f(cA, "TM"),
                     "rec_cyto": f(cA, "cytoplasmic"),
                     "par_extra": f(cB, "extracellular"), "par_tm": f(cB, "TM"),
                     "par_cyto": f(cB, "cytoplasmic"),
                     "n_res_A": len(resA), "n_res_B": len(resB)})
    CACHE.write_text(json.dumps(cache))
    if not rows:
        raise SystemExit("no arm-P folds scored yet")
    z = pd.DataFrame(rows)
    z.to_csv(ROOT / "results" / "ectodomain_interface_zone.csv", index=False)
    log.info("ARM P -- where the surviving interface sits (receptor is full length,")
    log.info("partner is its ectodomain, so the partner CANNOT be cytoplasmic here)\n")
    log.info("  %-30s %-14s  receptor: %5s %5s %5s", "job", "arm", "ecto", "TM", "cyto")
    for r in z.sort_values(["arm", "rec_cyto"]).itertuples():
        flag = "  <- still on the impossible face" if r.rec_cyto >= 0.3 else ""
        log.info("  %-30s %-14s            %5.2f %5.2f %5.2f%s", r.job,
                 r.arm.replace("P_", ""), r.rec_extra, r.rec_tm, r.rec_cyto, flag)
    bad = z[z.arm.eq("P_candidate") & z.rec_cyto.ge(0.3)]
    log.info("\n%d of %d candidates keep a mostly cytoplasmic receptor face even after the "
             "partner's tail is removed", len(bad), int(z.arm.eq("P_candidate").sum()))
    log.info("wrote results/ectodomain_interface_zone.csv")


if __name__ == "__main__":
    main()
