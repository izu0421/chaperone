"""Re-fold the candidates whose only fault was where AlphaFold3 put the partner.

The topology rule rejected 43 of 57 protein candidates, and it rejects 61% of
KNOWN binders on the chemokine panel, so it was discarding pose failures and
calling them biology. Two things could produce a wrong-face pose, and they need
different fixes:

  one seed      the whole screen ran modelSeeds=[1]. AF3 docking splits by seed;
                this project has already seen a pair go from 0/5 to 15/25 across
                seeds. A single wrong-face pose may be nothing but the draw.
                FIX: same construct, five seeds. No truncation, no artefact.

  the membrane  AF3 models no bilayer, so nothing stops it docking a secreted
                partner against residues that are inside the cell.
                FIX: give it only the extracellular part, so the wrong face does
                not exist. This ONLY works where the receptor has a real
                extracellular domain. Measured from UniProt topology, of the 14
                receptors with a secreted partner only GPR158 (394 aa) and GPR37
                (239 aa) do; MAS1L (77 aa) is marginal and the other eleven have
                21-54 aa of disconnected loop, which is not a foldable construct.

So arm A is the general fix and arm B is for two receptors.

Arm B carries the trap this project has hit before: cutting a target to its
binding domain exposes the faces the rest of the protein was covering, and on a
previous project that added 844 A^2 of artificial surface that a binder docked
onto. So every truncated receptor is also folded against decoy secreted proteins
it has no reason to bind. If the decoys dock as well as the candidate, the
construct is making a surface rather than revealing one.

Arm C is the control that makes arm A readable: chemokine-panel pairs whose
answer is known, folded the same way. If five seeds recover the extracellular
pose for KNOWN binders and not for known non-binders, the rescue is real.

Usage:
    micromamba run -n structmod python -I scripts/31_build_refold_rescue.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("rescue")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "rescue_inputs"
SEEDS = [1, 2, 3, 4, 5]
MIN_ECD = 150            # below this an "extracellular domain" is loops, not a domain

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)

# decoy secreted proteins for the truncation control: abundant, secreted, and with
# no reported link to any of these receptors
DECOYS = {"ALB": "P02768", "LYZ": "P61626", "RNASE1": "P07998"}


def ext_segments(topo: list) -> list[tuple[int, int]]:
    return sorted((int(t["start"]), int(t["end"])) for t in topo
                  if "Extracellular" in str(t.get("description") or ""))


def truncate_to_ecd(seq: str, msa: str, segs: list[tuple[int, int]]):
    """Keep only extracellular query positions, in every row of the alignment.

    Walks query positions, as the masking code does: a lowercase letter is an
    insertion and consumes no query position, so slicing by character index
    shifts the frame for most rows and still parses.
    """
    keep = set()
    for a, b in segs:
        keep.update(range(a - 1, b))          # UniProt is 1-based, inclusive

    def cut(s: str) -> str:
        out, q = [], 0
        for ch in s:
            if ch.islower():
                out.append(ch); continue
            if q in keep:
                out.append(ch)
            q += 1
        return "".join(out)

    rows, name = [], None
    for line in msa.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        elif name is not None:
            rows.append((name, cut(line))); name = None
    return cut(seq), "".join(f">{n}\n{s}\n" for n, s in rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    topo = json.loads((ROOT / "results" / "orphan_topology.json").read_text())
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))
    acc.update(DECOYS)

    v = pd.read_csv("/data/yzy21/yy/space_agent/chaperone/reports/"
                    "screen_verdicts_current/verdicts.csv")
    v = v[v.partner_kind.eq("protein")]
    SECRETED = ("CCL", "CXCL", "IL", "GHRL", "NMS", "CALCB", "MST1", "WNT", "SLIT",
                "C3", "BMP", "PTN", "CCK", "IGF", "FGF", "EFNA", "ADCYAP1", "SST", "SPX")
    cands = [(r.protein_b, r.protein_a) for r in v.itertuples()
             if str(r.protein_a).upper().startswith(SECRETED)]
    log.info("%d candidates with a secreted-type partner", len(cands))

    rows = []

    def write(name, arm, a_acc, b_acc, ecd=False, receptor=None):
        try:
            qA, qB, rA, rB = b7.build_pair(a_acc, b_acc)
            kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
        except Exception as exc:                            # noqa: BLE001
            log.warning("  %s: %s", name, str(exc)[:80]); return
        msaA = b7.a3m([("query", qA)] + kA)
        if ecd:
            segs = ext_segments(topo.get(receptor, []))
            if not segs:
                log.warning("  %s: no extracellular annotation", name); return
            qA, msaA = truncate_to_ecd(qA, msaA, segs)
        msaB = b7.a3m([("query", qB)] + kB)
        js = {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
              "sequences": [
                  {"protein": {"id": "A", "sequence": qA, "pairedMsa": msaA,
                               "unpairedMsa": msaA, "templates": []}},
                  {"protein": {"id": "B", "sequence": qB, "pairedMsa": msaB,
                               "unpairedMsa": msaB, "templates": []}}]}
        (OUT / f"{name}.json").write_text(json.dumps(js))
        rows.append({"job": name, "arm": arm, "receptor": receptor or a_acc,
                     "partner": b_acc, "ecd": ecd, "size": len(qA) + len(qB),
                     "n_seeds": len(SEEDS)})

    log.info("\narm A -- same construct, five seeds")
    for rec, par in cands:
        if rec in acc and par in acc:
            write(f"A_{rec}__{par}", "A_multiseed", acc[rec], acc[par], receptor=rec)

    log.info("\narm B -- extracellular domain only, where one exists")
    for rec, par in cands:
        segs = ext_segments(topo.get(rec, []))
        if not segs or max(b - a + 1 for a, b in segs) < MIN_ECD:
            continue
        write(f"B_{rec}ECD__{par}", "B_ecd", acc[rec], acc[par], ecd=True, receptor=rec)
        for dname, dacc in DECOYS.items():     # the truncation-decoy control
            write(f"B_{rec}ECD__{dname}", "B_ecd_decoy", acc[rec], dacc,
                  ecd=True, receptor=rec)

    log.info("\narm C -- the control: chemokine pairs whose answer is known, five seeds")
    cs = pd.read_csv(ROOT / "results" / "protein_counterscreen_scores.csv")
    A = cs[cs.block.eq("A_chemokine")]
    PANEL = {"CCR1":"P32246","CCR2":"P41597","CCR3":"P51677","CCR5":"P51681",
             "CCR6":"P51684","CCR7":"P32248","CXCR1":"P25024","CXCR2":"P25025",
             "CXCR3":"P49682","CXCR4":"P61073","CX3CR1":"P49238","ACKR3":"P25106",
             "CCR4":"P51679","CCR8":"P51685","CXCR5":"P32302","CXCR6":"O00574",
             "CCL2":"P13500","CCL3":"P10147","CCL5":"P13501","CCL11":"P51671",
             "CCL19":"Q99731","CCL20":"P78556","CXCL8":"P10145","CXCL10":"P02778",
             "CXCL12":"P48061","CX3CL1":"P78423","CCL21":"O00585","CXCL11":"O14625",
             "CCL17":"Q92583","CCL1":"P22362","CXCL13":"O43927","CXCL16":"Q9H2A7"}
    for lab, n in (("binder", 12), ("negative", 12)):
        sub = A[A.label.eq(lab)].head(n)
        for r in sub.itertuples():
            if r.chain_a in PANEL and r.chain_b in PANEL:
                write(f"C_{lab}_{r.chain_a}__{r.chain_b}", f"C_control_{lab}",
                      PANEL[r.chain_a], PANEL[r.chain_b], receptor=r.chain_a)

    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    dest = ROOT / "work" / "rescue_manifest.csv"
    man.to_csv(dest, index=False)
    log.info("\nwrote %d inputs (%d seeds each = %d structures)",
             len(man), len(SEEDS), len(man) * len(SEEDS))
    log.info("%s", man.groupby("arm").agg(jobs=("job", "size"),
                                          median_tokens=("size", "median")).to_string())
    log.info("manifest -> %s", dest)


if __name__ == "__main__":
    main()
