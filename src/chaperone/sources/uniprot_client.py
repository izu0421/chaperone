"""Thin client + local cache around the UniProt REST API — gives the triage
agent direct access to a protein's own real annotations (function
description, keywords, feature summary) instead of relying on the agent's
background knowledge or a hardcoded gene list.

This closes a real gap found live in this project: a deterministic check for
"is this a glycan-binding lectin" had to hardcode a short gene list
(KNOWN_GALECTIN_GENES in deterministic_gate.py) because nothing queried
UniProt's own function/keyword annotations, which say exactly this in plain
text (e.g. LGALS1's function comment literally starts "Lectin that binds
beta-galactoside..."). Exposing this as an agent tool generalizes the check
to any protein with a documented binding mechanism, not just the ones
someone thought to hardcode.
"""
import json
import sqlite3
import time
from pathlib import Path

import httpx

from .http_retry import get_with_retry
from ..paths import DATA_DIR

CACHE_PATH = DATA_DIR / "cache.db"
CACHE_TTL_SECONDS = 30 * 24 * 3600  # UniProt reference annotation changes rarely
SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
HTTP_TIMEOUT = 30.0
FIELDS = "accession,protein_name,keyword,cc_function,ft_carbohyd,ft_lipid,ft_disulfid,ft_mod_res,ft_transmem,ft_topo_dom"


def _cache_conn():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS uniprot_query_cache ("
        "gene TEXT PRIMARY KEY, fetched_at REAL, payload TEXT)"
    )
    return conn


def _cache_get(gene: str):
    conn = _cache_conn()
    try:
        row = conn.execute(
            "SELECT fetched_at, payload FROM uniprot_query_cache WHERE gene = ?", (gene,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    fetched_at, payload = row
    if time.time() - fetched_at > CACHE_TTL_SECONDS:
        return None
    return json.loads(payload)


def _cache_put(gene: str, payload: dict) -> None:
    conn = _cache_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO uniprot_query_cache (gene, fetched_at, payload) VALUES (?, ?, ?)",
            (gene, time.time(), json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def _feature_type_counts(entry: dict) -> dict:
    counts = {}
    for feat in entry.get("features", []):
        t = feat.get("type")
        counts[t] = counts.get(t, 0) + 1
    return counts


def fetch_uniprot_annotation(gene: str) -> dict:
    """Resolve a human gene symbol to its reviewed UniProt entry and return
    a triage-relevant summary: accession, protein name, the FUNCTION comment
    (plain-text description of what the protein actually does/binds —
    catches mechanism-level concerns like "Lectin that binds beta-
    galactoside" directly from UniProt, not a hardcoded list), keywords
    (includes PTM/topology/binding categories), and counts of PTM/topology
    feature types (residue-level detail is in analyze_fold.py's
    fetch_uniprot_features, used at fold-analysis time — this is a cheaper
    triage-time summary, not the full feature list)."""
    cached = _cache_get(gene)
    if cached is not None:
        return cached

    with httpx.Client() as client:
        resp = get_with_retry(
            client,
            SEARCH_URL,
            {
                "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "fields": FIELDS,
                "size": 1,
            },
            HTTP_TIMEOUT,
        )
        results = resp.json().get("results", [])

    if not results:
        payload = {"gene": gene, "error": "No reviewed human UniProt entry found"}
        _cache_put(gene, payload)
        return payload

    entry = results[0]
    protein_name = (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    function_texts = [
        t.get("value")
        for c in entry.get("comments", [])
        if c.get("commentType") == "FUNCTION"
        for t in c.get("texts", [])
    ]
    keywords = [kw.get("name") for kw in entry.get("keywords", [])]

    payload = {
        "gene": gene,
        "accession": entry.get("primaryAccession"),
        "protein_name": protein_name,
        "function": " ".join(function_texts) or None,
        "keywords": keywords,
        "ptm_topology_feature_counts": _feature_type_counts(entry),
    }
    _cache_put(gene, payload)
    return payload
