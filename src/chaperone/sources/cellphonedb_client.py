"""Local cache + lookup over CellPhoneDB's curated interaction/complex data
(ventolab/cellphonedb-data, the Teichlab-maintained CellPhoneDB dataset).

Unlike STRING (functional-association scores, includes textmining) or PubMed
(raw co-mention), CellPhoneDB is manually curated specifically for cell-cell
communication: ligand-receptor pairs and receptor/adhesion complexes. A hit
here is strong, precise evidence of an already-known interaction — especially
useful for the secreted-ligand/membrane-receptor branch of the triage logic.
complex_input.csv also gives real multi-subunit complex membership (e.g.
integrin heterodimers), a better "other subunits" source than HPA's
interaction-count proxy for surface/adhesion complexes.
"""
import csv
import io
import re
import time
from pathlib import Path

import httpx

from ..paths import DATA_DIR as _DATA_ROOT

DATA_DIR = _DATA_ROOT / "cellphonedb"
CACHE_TTL_SECONDS = 30 * 24 * 3600
BASE_URL = "https://raw.githubusercontent.com/ventolab/cellphonedb-data/master/data"
HTTP_TIMEOUT = 30.0

_interactions_cache = None
_complexes_cache = None
_gene_to_uniprot = None
_uniprot_to_gene = None


def _download(filename: str) -> str:
    path = DATA_DIR / filename
    if path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS:
        return path.read_text()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(f"{BASE_URL}/{filename}", timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    path.write_text(resp.text)
    return resp.text


def _load_gene_mapping():
    global _gene_to_uniprot, _uniprot_to_gene
    if _gene_to_uniprot is not None:
        return
    _gene_to_uniprot, _uniprot_to_gene = {}, {}
    reader = csv.DictReader(io.StringIO(_download("gene_input.csv")))
    for row in reader:
        symbol = (row.get("hgnc_symbol") or row.get("gene_name") or "").upper()
        uniprot = row.get("uniprot")
        if symbol and uniprot:
            _gene_to_uniprot.setdefault(symbol, uniprot)
            _uniprot_to_gene.setdefault(uniprot, symbol)


def _load_interactions():
    global _interactions_cache
    if _interactions_cache is not None:
        return _interactions_cache
    rows = list(csv.DictReader(io.StringIO(_download("interaction_input.csv"))))
    for row in rows:
        row["_tokens"] = {
            t.upper() for t in re.split(r"[-+]", row.get("interactors") or "") if t
        }
    _interactions_cache = rows
    return rows


def _load_complexes():
    global _complexes_cache
    if _complexes_cache is not None:
        return _complexes_cache
    _complexes_cache = list(csv.DictReader(io.StringIO(_download("complex_input.csv"))))
    return _complexes_cache


def known_interaction(gene_a: str, gene_b: str) -> dict:
    """Return {'found': bool, 'matches': [...]} — curated CellPhoneDB
    interactions where both gene symbols appear together."""
    ga, gb = gene_a.upper(), gene_b.upper()
    matches = []
    for row in _load_interactions():
        if ga in row["_tokens"] and gb in row["_tokens"]:
            matches.append(
                {
                    "interactors": row.get("interactors"),
                    "classification": row.get("classification"),
                    "directionality": row.get("directionality"),
                    "source": row.get("source"),
                    "annotation_strategy": row.get("annotation_strategy"),
                }
            )
    return {"found": bool(matches), "matches": matches}


def complex_members(gene: str) -> dict:
    """Return {'found': bool, 'complexes': [{'complex_name', 'other_members': [...]}]} —
    other gene symbols found alongside this one in a CellPhoneDB curated complex."""
    _load_gene_mapping()
    uniprot = _gene_to_uniprot.get(gene.upper())
    if uniprot is None:
        return {"found": False, "complexes": [], "note": f"'{gene}' not in CellPhoneDB gene table"}

    uniprot_cols = ["uniprot_1", "uniprot_2", "uniprot_3", "uniprot_4", "uniprot_5"]
    complexes = []
    for row in _load_complexes():
        members = [row.get(c) for c in uniprot_cols if row.get(c)]
        if uniprot in members:
            others = [_uniprot_to_gene.get(u, u) for u in members if u != uniprot]
            if others:
                complexes.append({"complex_name": row.get("complex_name"), "other_members": others})
    return {"found": bool(complexes), "complexes": complexes}
