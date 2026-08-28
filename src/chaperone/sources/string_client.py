"""Thin client + local cache around the STRING API (string-db.org).

Gives a much stronger known-interaction signal than PubMed co-mention: STRING
returns a combined confidence score plus a per-evidence-channel breakdown
(experiments, curated databases, co-expression, text-mining, ...), so the
agent can tell a curated/experimental hit apart from a pair that's only
co-mentioned in text-mined literature (same distinction PubMed co-mention
can't make on its own).
"""
import json
import sqlite3
import time
from pathlib import Path

import httpx

from .http_retry import get_with_retry
from ..paths import DATA_DIR

CACHE_PATH = DATA_DIR / "cache.db"
CACHE_TTL_SECONDS = 30 * 24 * 3600
NETWORK_URL = "https://string-db.org/api/json/network"
HTTP_TIMEOUT = 20.0
SPECIES_HUMAN = 9606


def _cache_conn():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS string_pair_cache ("
        "pair_key TEXT PRIMARY KEY, fetched_at REAL, payload TEXT)"
    )
    return conn


def _cache_get(pair_key: str):
    conn = _cache_conn()
    try:
        row = conn.execute(
            "SELECT fetched_at, payload FROM string_pair_cache WHERE pair_key = ?",
            (pair_key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    fetched_at, payload = row
    if time.time() - fetched_at > CACHE_TTL_SECONDS:
        return None
    return json.loads(payload)


def _cache_put(pair_key: str, payload: dict) -> None:
    conn = _cache_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO string_pair_cache (pair_key, fetched_at, payload) VALUES (?, ?, ?)",
            (pair_key, time.time(), json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def known_interaction(gene_a: str, gene_b: str) -> dict:
    """Query STRING for a direct association between two gene symbols.

    Returns {'found': bool, 'combined_score': float, 'evidence': {...}} on a
    hit, or {'found': False} if STRING has no edge between them. Evidence
    channels: neighborhood, fusion, phylogenetic (genomic context — weak/
    indirect), coexpression, experiments, database (curated, strong direct
    evidence), textmining (literature co-mention, same caveat as PubMed).
    """
    pair_key = "|".join(sorted([gene_a.upper(), gene_b.upper()]))
    cached = _cache_get(pair_key)
    if cached is not None:
        return cached

    with httpx.Client() as client:
        resp = get_with_retry(
            client,
            NETWORK_URL,
            {"identifiers": f"{gene_a}\r{gene_b}", "species": SPECIES_HUMAN},
            HTTP_TIMEOUT,
        )
    edges = resp.json() or []

    if not edges:
        result = {"found": False}
    else:
        edge = edges[0]
        result = {
            "found": True,
            "combined_score": edge.get("score"),
            "evidence": {
                "neighborhood": edge.get("nscore"),
                "fusion": edge.get("fscore"),
                "phylogenetic": edge.get("pscore"),
                "coexpression": edge.get("ascore"),
                "experiments": edge.get("escore"),
                "database": edge.get("dscore"),
                "textmining": edge.get("tscore"),
            },
        }

    _cache_put(pair_key, result)
    return result
