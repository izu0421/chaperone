"""Two re-folds the triage asked for and the screen could not do.

ARM S -- the subcomplexes, folded whole.
  Four candidates were set aside as LIKELY_SUBCOMPLEX, each naming the subunit
  the model was missing. That verdict is a request, not a rejection: fold the
  complete complex and the question resolves either way. Fibrinogen is the clear
  case -- FGA alone presents coiled-coil surface that is buried in the real
  FGA/FGB/FGG hexamer, so a receptor docking there is docking to an artefact of
  the construct.

      GPR6   + GRIN2D   + GRIN1          obligate NMDA heterotetramer
      GPR101 + FGA      + FGB + FGG      fibrinogen
      GPR39  + FGA      + FGB + FGG      fibrinogen
      GPR63  + GP1BB    + GP1BA + GP9    glycoprotein Ib

ARM M -- the secreted partners, folded as the molecule that actually exists.
  Every secreted partner went into the screen as its full preprohormone, signal
  peptide included: somatostatin 1-24 of 116, spexin 1-26 of 116, PACAP 1-24 of
  176, CCL11 1-23 of 97. That peptide is cleaved in the ER and is never present
  when the ligand meets the receptor.

  It is not a cosmetic difference. An audit of the eleven known chemokine
  binders that the topology rule falsely flagged found the signal peptide
  supplying 100% of the cytoplasmic contacts in ten of them, while the mature
  chain sat docked in the orthosteric pocket. The screen folded a molecule that
  does not exist, then rejected the result for being in the wrong place.

  So every partner with an annotated signal peptide is re-folded as its mature
  chain. Boundaries come from UniProt, per protein, never assumed.

Usage:
    micromamba run -n structmod python -I scripts/33_build_subcomplex_and_mature.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("subcomplex")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "subcomplex_inputs"
SEEDS = [1, 2, 3, 4, 5]

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)

# receptor, partner, and the subunits the triage named as missing
SUBCOMPLEXES = [
    ("GPR6", "GRIN2D", ["GRIN1"]),
    ("GPR101", "FGA", ["FGB", "FGG"]),
    ("GPR39", "FGA", ["FGB", "FGG"]),
    ("GPR63", "GP1BB", ["GP1BA", "GP9"]),
]
EXTRA_ACC = {"GRIN1": "Q05586", "FGB": "P02675", "FGG": "P02679",
             "GP1BA": "P07359", "GP9": "P14770"}


def uniprot_signal(acc: str) -> int | None:
    """Last residue of the signal peptide, or None. Asked per protein."""
    url = (f"https://rest.uniprot.org/uniprotkb/{acc}.json?fields=ft_signal")
    try:
        d = json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"}),
            timeout=30).read())
    except Exception as exc:                                  # noqa: BLE001
        log.warning("  %s: signal-peptide lookup failed (%s)", acc, str(exc)[:50])
        return None
    for f in d.get("features", []):
        if f["type"] == "Signal":
            return int(f["location"]["end"]["value"])
    return None


def drop_prefix(seq: str, msa: str, n: int):
    """Remove the first n QUERY positions from the query and every alignment row.

    Walks query positions rather than characters: a lowercase letter is an
    insertion and consumes none, so a raw slice shifts the frame for most rows
    and still parses.
    """
    def cut(s: str) -> str:
        out, q = [], 0
        for ch in s:
            if ch.islower():
                if q >= n:
                    out.append(ch)
                continue
            if q >= n:
                out.append(ch)
            q += 1
        return "".join(out)

    rows, name = [], None
    for line in msa.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        elif name is not None:
            rows.append((name, cut(line))); name = None
    return cut(seq), "".join(f">{n_}\n{s}\n" for n_, s in rows)


def chain_block(cid: str, seq: str, msa: str) -> dict:
    return {"protein": {"id": cid, "sequence": seq, "pairedMsa": msa,
                        "unpairedMsa": msa, "templates": []}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))
    acc.update(EXTRA_ACC)
    rows = []

    log.info("ARM S -- subcomplexes folded whole")
    for rec, par, extras in SUBCOMPLEXES:
        missing = [g for g in (rec, par, *extras) if g not in acc]
        if missing:
            log.warning("  %s+%s: no accession for %s", rec, par, missing); continue
        try:
            qA, qB, rA, rB = b7.build_pair(acc[rec], acc[par])
            kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
        except Exception as exc:                              # noqa: BLE001
            log.warning("  %s+%s: %s", rec, par, str(exc)[:70]); continue
        seqs = [chain_block("A", qA, b7.a3m([("query", qA)] + kA)),
                chain_block("B", qB, b7.a3m([("query", qB)] + kB))]
        ok = True
        for i, g in enumerate(extras):
            try:
                _, q, hits = b7.read_omic(acc[g])
            except Exception as exc:                          # noqa: BLE001
                log.warning("  %s: no alignment (%s)", g, str(exc)[:50]); ok = False; break
            msa = b7.a3m([("query", q)] + [(k, hits[k]) for k in sorted(hits)])
            seqs.append(chain_block("CDEFG"[i], q, msa))
        if not ok:
            continue
        name = f"S_{rec}__{par}__{'_'.join(extras)}"
        (OUT / f"{name}.json").write_text(json.dumps(
            {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3",
             "version": 1, "sequences": seqs}))
        size = sum(len(s["protein"]["sequence"]) for s in seqs)
        rows.append({"job": name, "arm": "S_subcomplex", "receptor": rec,
                     "partner": par, "extras": ";".join(extras),
                     "n_chains": len(seqs), "size": size})
        log.info("  %-34s %d chains, %d tokens", name, len(seqs), size)

    log.info("\nARM M -- secreted partners as their mature chain")
    v = pd.read_csv("/data/yzy21/yy/space_agent/chaperone/reports/"
                    "screen_verdicts_current/verdicts.csv")
    v = v[v.partner_kind.eq("protein")]
    seen = set()
    for r in v.itertuples():
        rec, par = r.protein_b, r.protein_a
        if par in seen or par not in acc or rec not in acc:
            continue
        sig = uniprot_signal(acc[par])
        if not sig:
            continue
        seen.add(par)
        try:
            qA, qB, rA, rB = b7.build_pair(acc[rec], acc[par])
            kA, kB = b7.hhfilter_pairs(qA, qB, rA, rB)
        except Exception as exc:                              # noqa: BLE001
            log.warning("  %s+%s: %s", rec, par, str(exc)[:70]); continue
        msaB = b7.a3m([("query", qB)] + kB)
        if sig >= len(qB):
            log.warning("  %s: signal peptide %d >= chain length %d", par, sig, len(qB))
            continue
        qBm, msaBm = drop_prefix(qB, msaB, sig)
        name = f"M_{rec}__{par}"
        (OUT / f"{name}.json").write_text(json.dumps(
            {"name": name, "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1,
             "sequences": [chain_block("A", qA, b7.a3m([("query", qA)] + kA)),
                           chain_block("B", qBm, msaBm)]}))
        rows.append({"job": name, "arm": "M_mature", "receptor": rec, "partner": par,
                     "extras": f"signal 1-{sig} removed", "n_chains": 2,
                     "size": len(qA) + len(qBm)})
        log.info("  %-28s signal 1-%-3d removed, %d -> %d aa", name, sig, len(qB), len(qBm))

    if not rows:
        raise SystemExit("nothing built")
    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    dest = ROOT / "work" / "subcomplex_manifest.csv"
    man.to_csv(dest, index=False)
    log.info("\nwrote %d inputs (%d seeds each = %d structures)",
             len(man), len(SEEDS), len(man) * len(SEEDS))
    log.info("%s", man.groupby("arm").agg(jobs=("job", "size"),
                                          median_tokens=("size", "median")).to_string())
    log.info("manifest -> %s", dest)


if __name__ == "__main__":
    main()
