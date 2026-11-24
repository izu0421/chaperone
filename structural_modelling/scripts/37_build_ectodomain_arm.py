"""Arm P: fold the membrane partners as the half that can reach the receptor.

Arm M fixed the secreted partners -- the screen had folded the preprohormone and
the signal peptide was docking where no secreted protein can be. Ten candidates
have a MEMBRANE partner, and they carry the same fault in a different form: the
screen folded the partner full-length, cytoplasmic tail included, and that tail
can dock on the receptor's cytoplasmic face just as the signal peptide did.

So fold the receptor full-length against the partner's ECTODOMAIN ONLY. Only the
ectodomain can physically reach the receptor's extracellular face, so a pose that
survives this construct is a pose that could exist.

Two controls, because a construct without one says nothing:

  P_perm      the same receptor against OTHER candidates' partner ectodomains.
              A permuted pair is matched on everything that matters -- size class,
              cell-surface compartment, how the construct was cut, how the MSA was
              built -- and is a plausible but unreported partner, which is a much
              harder negative than the soluble decoys arm B used (albumin at 66 kDa
              against a 90-residue chemokine was not matched on anything).
  P_known_*   pairs of membrane proteins whose answer is known, cut the same way.
              Positives must be ectodomain-mediated and the LIGAND must itself be
              membrane-anchored in humans; negatives must be size- and
              compartment-matched to the positives, or the control is trivially
              easy. Curated and adversarially verified separately, read here from
              work/ectodomain_controls.json.

Only the longest CONTIGUOUS extracellular segment is taken, never a concatenation
of segments: joining two segments makes a junction that is itself a surface, and
this project has already been burnt by a cut face that a decoy docked onto.

Usage:
    micromamba run -n structmod python -I scripts/37_build_ectodomain_arm.py
    micromamba run -n structmod python -I scripts/37_build_ectodomain_arm.py --controls-only
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import logging
import os
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ecd-arm")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ectodomain_inputs"
MAN = ROOT / "work" / "ectodomain_manifest.csv"
CTL = ROOT / "work" / "ectodomain_controls.json"
SEEDS = [1]
MIN_ECD = 100
N_PERM = 2

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)


def uniprot_ecd(acc: str) -> tuple[int, int] | None:
    """Longest contiguous extracellular topological domain, 1-based inclusive.

    Asked per protein and never inferred: a signal peptide, a GPI anchor and a
    type II orientation all change which end is outside.
    """
    url = (f"https://rest.uniprot.org/uniprotkb/{acc}.json"
           "?fields=ft_topo_dom,ft_transmem,ft_signal,length")
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "research-script/1.0"}), timeout=30).read())
    except Exception as exc:                                    # noqa: BLE001
        log.warning("  %s: topology lookup failed (%s)", acc, str(exc)[:50])
        return None
    segs = [(int(f["location"]["start"]["value"]), int(f["location"]["end"]["value"]))
            for f in d.get("features", [])
            if f["type"] == "Topological domain"
            and "xtracellular" in str(f.get("description") or "")]
    if not segs:
        return None
    s, e = max(segs, key=lambda t: t[1] - t[0])
    # a signal peptide is cleaved, so it is not part of the mature ectodomain
    sig = [int(f["location"]["end"]["value"]) for f in d.get("features", [])
           if f["type"] == "Signal"]
    if sig and s <= sig[0]:
        s = sig[0] + 1
    return (s, e) if e - s + 1 >= MIN_ECD else None


def slice_query(seq: str, msa: str, s: int, e: int):
    """Keep query positions s..e (1-based, inclusive) in the query and every row.

    Insertions are kept only inside the span. Arm B's cut kept every lowercase
    column wherever it sat, which carries insertions that were anchored to
    removed positions across the new boundary; they do not shift the query frame
    but they do put residues next to the cut that belong nowhere.
    """
    def cut(row: str) -> str:
        out, q = [], 0
        for ch in row:
            if ch.islower():
                if s <= q <= e:
                    out.append(ch)
                continue
            q += 1                                 # q is now 1-based for this column
            if s <= q <= e:
                out.append(ch)
        return "".join(out)

    rows, name = [], None
    for line in msa.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        elif name is not None:
            rows.append((name, cut(line))); name = None
    return cut(seq), "".join(f">{n}\n{r}\n" for n, r in rows)


def chain(cid: str, seq: str, msa: str) -> dict:
    return {"protein": {"id": cid, "sequence": seq, "pairedMsa": msa,
                        "unpairedMsa": msa, "templates": []}}


def write_job(name: str, arm: str, rec_acc: str, par_acc: str, ecd: tuple[int, int],
              rows: list, rec_ecd: tuple[int, int] | None = None, **extra) -> bool:
    if (OUT / f"{name}.json").exists():
        return False
    try:
        qA, qB, rA, rB = b7.build_pair(rec_acc, par_acc)
        kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("  %s: %s", name, str(exc)[:70]); return False
    if rec_ecd is not None:
        rs, re_ = rec_ecd
        if re_ > len(qA):
            log.warning("  %s: receptor ECD %d-%d past length %d", name, rs, re_, len(qA))
            return False
        qA, msaA_cut = slice_query(qA, b7.a3m([("query", qA)] + kA), rs, re_)
        kA = None                      # msaA_cut is already an a3m string
    s, e = ecd
    if e > len(qB):
        log.warning("  %s: ectodomain %d-%d past chain length %d", name, s, e, len(qB))
        return False
    qBe, msaBe = slice_query(qB, b7.a3m([("query", qB)] + kB), s, e)
    if len(qBe) != e - s + 1:
        log.warning("  %s: cut gave %d residues, expected %d", name, len(qBe), e - s + 1)
        return False
    msaA = msaA_cut if rec_ecd is not None else b7.a3m([("query", qA)] + kA)
    (OUT / f"{name}.json").write_text(json.dumps(
        {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
         "sequences": [chain("A", qA, msaA), chain("B", qBe, msaBe)]}))
    rows.append({"job": name, "arm": arm, "size": len(qA) + len(qBe),
                 "ecd": f"{s}-{e}",
                 "rec_ecd": f"{rec_ecd[0]}-{rec_ecd[1]}" if rec_ecd else "full", **extra})
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))
    rows = [] if not MAN.exists() else pd.read_csv(MAN).to_dict("records")
    have = {r["job"] for r in rows}

    if not a.controls_only:
        t = pd.read_csv(ROOT / "results" / "candidate_ranking.csv")
        ecds: dict[str, tuple[int, int]] = {}
        for par in t.partner:
            if par in acc:
                seg = uniprot_ecd(acc[par])
                if seg:
                    ecds[par] = seg
        log.info("%d of %d candidate partners have a contiguous ectodomain >= %d aa: %s",
                 len(ecds), len(t), MIN_ECD, ", ".join(f"{k} {v[1]-v[0]+1}" for k, v in ecds.items()))

        log.info("\nCANDIDATES -- receptor full length, partner ectodomain only")
        for r in t.itertuples():
            if r.partner not in ecds:
                continue
            write_job(f"P_{r.receptor}__{r.partner}", "P_candidate", acc[r.receptor],
                      acc[r.partner], ecds[r.partner], rows,
                      receptor=r.receptor, partner=r.partner)

        log.info("\nPERMUTED NEGATIVES -- the same receptors, other candidates' ectodomains")
        own = t.groupby("receptor").partner.apply(set).to_dict()
        for rec in sorted(set(t.receptor)):
            if not any(p in ecds for p in own.get(rec, ())):
                continue                       # receptor has no candidate in this arm
            mine = {p for p in own.get(rec, ()) if p in ecds}
            target = max(ecds[p][1] - ecds[p][0] + 1 for p in mine)
            others = sorted((p for p in ecds if p not in own.get(rec, ())),
                            key=lambda p: abs((ecds[p][1] - ecds[p][0] + 1) - target))
            for par in others[:N_PERM]:
                write_job(f"P_perm_{rec}__{par}", "P_permuted", acc[rec], acc[par],
                          ecds[par], rows, receptor=rec, partner=par)

        # Where the RECEPTOR also has a real extracellular domain, cut both. For
        # GPR158+EPHA5 the screen's interface is 100% cytoplasmic on the receptor,
        # so removing only the partner's tail still leaves the impossible face
        # available; cutting both makes the artefact unbuildable.
        log.info("\nBOTH ECTODOMAINS -- where the receptor has one of its own")
        topo = json.loads((ROOT / "results" / "orphan_topology.json").read_text())
        rec_ecds = {}
        for rec in sorted(set(t.receptor)):
            segs = [(int(s["start"]), int(s["end"])) for s in topo.get(rec, [])
                    if "Extracellular" in str(s.get("description") or "")]
            if segs:
                a, b = max(segs, key=lambda x: x[1] - x[0])
                if b - a + 1 >= MIN_ECD:
                    rec_ecds[rec] = (a, b)
        log.info("  receptors with an ectodomain >= %d aa: %s", MIN_ECD,
                 ", ".join(f"{k} {v[1]-v[0]+1}" for k, v in rec_ecds.items()) or "none")
        for r in t.itertuples():
            if r.receptor in rec_ecds and r.partner in ecds:
                write_job(f"P_both_{r.receptor}__{r.partner}", "P_both_ecd",
                          acc[r.receptor], acc[r.partner], ecds[r.partner], rows,
                          rec_ecd=rec_ecds[r.receptor], receptor=r.receptor,
                          partner=r.partner)
                others = sorted((q for q in ecds if q not in own.get(r.receptor, ())),
                                key=lambda q: abs((ecds[q][1] - ecds[q][0] + 1)
                                                  - (ecds[r.partner][1] - ecds[r.partner][0] + 1)))
                for par in others[:N_PERM]:
                    write_job(f"P_both_perm_{r.receptor}__{par}", "P_both_permuted",
                              acc[r.receptor], acc[par], ecds[par], rows,
                              rec_ecd=rec_ecds[r.receptor], receptor=r.receptor, partner=par)

    if CTL.exists():
        ctl = json.loads(CTL.read_text())
        log.info("\nKNOWN ANSWERS -- %d verified pairs", len(ctl))
        for p in ctl:
            seg = uniprot_ecd(p["ligand_uniprot"])
            if not seg:
                log.warning("  %s+%s: ligand has no contiguous ectodomain >= %d aa",
                            p["receptor_gene"], p["ligand_gene"], MIN_ECD)
                continue
            write_job(f"P_known_{p['label']}_{p['receptor_gene']}__{p['ligand_gene']}",
                      f"P_known_{p['label']}", p["receptor_uniprot"], p["ligand_uniprot"],
                      seg, rows, receptor=p["receptor_gene"], partner=p["ligand_gene"])
    else:
        log.info("\nno verified control set at %s yet -- candidates and permutations only",
                 CTL.name)

    man = pd.DataFrame(rows).drop_duplicates("job").sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    tmp = MAN.with_suffix(".csv.tmp"); man.to_csv(tmp, index=False); os.replace(tmp, MAN)
    log.info("\n%d jobs (%d new), %d seeds each", len(man), len(man) - len(have), len(SEEDS))
    log.info("%s", man.groupby("arm").agg(jobs=("job", "size"),
                                          median_tokens=("size", "median")).to_string())


if __name__ == "__main__":
    main()
