"""Thin client + local cache around the Human Protein Atlas public JSON API.

Two HPA endpoints are used:
- search_download.php: fuzzy gene search, used only to resolve a gene symbol
  to its Ensembl gene ID (exact-match filtered client-side, since the search
  endpoint itself has no exact-match mode).
- proteinatlas.org/<ensembl_id>.json: the full per-gene profile (expression,
  subcellular location, protein class, ...).
"""
import json
import sqlite3
import time
from pathlib import Path

import httpx

from .http_retry import get_with_retry
from ..paths import DATA_DIR

CACHE_PATH = DATA_DIR / "cache.db"
CACHE_TTL_SECONDS = 30 * 24 * 3600  # HPA reference data changes rarely
SEARCH_URL = "https://www.proteinatlas.org/api/search_download.php"
GENE_URL_TMPL = "https://www.proteinatlas.org/{ensembl_id}.json"
HTTP_TIMEOUT = 20.0


def _cache_conn():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hpa_gene_cache ("
        "gene TEXT PRIMARY KEY, fetched_at REAL, payload TEXT)"
    )
    return conn


def _cache_get(gene: str):
    conn = _cache_conn()
    try:
        row = conn.execute(
            "SELECT fetched_at, payload FROM hpa_gene_cache WHERE gene = ?",
            (gene.upper(),),
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
            "INSERT OR REPLACE INTO hpa_gene_cache (gene, fetched_at, payload) VALUES (?, ?, ?)",
            (gene.upper(), time.time(), json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def _resolve_ensembl_id(gene: str, client: httpx.Client) -> dict:
    """Return {'ensembl_id': str} on an unambiguous exact match, else
    {'error': ..., 'candidates': [...]} listing what the fuzzy search found."""
    resp = get_with_retry(
        client,
        SEARCH_URL,
        {"search": gene, "format": "json", "columns": "g,eg", "compress": "no"},
        HTTP_TIMEOUT,
    )
    results = resp.json() or []
    exact = [r for r in results if r.get("Gene", "").upper() == gene.upper()]
    if len(exact) == 1:
        return {"ensembl_id": exact[0]["Ensembl"]}
    if not results:
        return {"error": f"no HPA match for gene '{gene}'", "candidates": []}
    return {
        "error": f"no unambiguous exact match for gene '{gene}'",
        "candidates": [r.get("Gene") for r in results[:10]],
    }


def fetch_gene_profile(gene: str) -> dict:
    """Fetch (or return cached) the full HPA profile for a gene symbol.

    On failure to resolve/find the gene, returns {'error': ..., 'candidates': [...]}
    instead of raising, so MCP tool wrappers can hand the agent a usable result.
    """
    cached = _cache_get(gene)
    if cached is not None:
        return cached

    with httpx.Client() as client:
        resolved = _resolve_ensembl_id(gene, client)
        if "error" in resolved:
            return resolved

        resp = get_with_retry(
            client, GENE_URL_TMPL.format(ensembl_id=resolved["ensembl_id"]), {}, HTTP_TIMEOUT
        )
        profile = resp.json()

    _cache_put(gene, profile)
    return profile
