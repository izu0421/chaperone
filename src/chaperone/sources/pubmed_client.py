"""Thin client + local cache around NCBI E-utilities (PubMed).

AF3 predicts interactions from bare, unmodified sequence — it does not see
glycans and (by default) does not model most PTMs. A confident interface can
still be a real-world artifact if the true interacting region is normally
glycosylated (common on secreted/membrane extracellular domains, and can
sterically block a contact surface) or if the interaction is only reported to
occur in a specific PTM state (e.g. phosphorylation-dependent binding). This
client gives the agent a literature signal for that, plus a co-mention check
usable as a weak novelty proxy.
"""
import json
import sqlite3
import time
from pathlib import Path

import httpx

from .http_retry import get_with_retry
from ..paths import DATA_DIR

CACHE_PATH = DATA_DIR / "cache.db"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # literature accrues faster than HPA reference data
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
HTTP_TIMEOUT = 20.0
TOOL_PARAMS = {"tool": "deorphanisation-ppi-triage", "email": "team@onecarbon.com"}


def _cache_conn():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pubmed_query_cache ("
        "query TEXT PRIMARY KEY, fetched_at REAL, payload TEXT)"
    )
    return conn


def _cache_get(query: str):
    conn = _cache_conn()
    try:
        row = conn.execute(
            "SELECT fetched_at, payload FROM pubmed_query_cache WHERE query = ?",
            (query,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    fetched_at, payload = row
    if time.time() - fetched_at > CACHE_TTL_SECONDS:
        return None
    return json.loads(payload)


def _cache_put(query: str, payload: dict) -> None:
    conn = _cache_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO pubmed_query_cache (query, fetched_at, payload) VALUES (?, ?, ?)",
            (query, time.time(), json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def pubmed_search(term: str, retmax: int = 5) -> dict:
    """Run a PubMed search, return {'count': int, 'hits': [{'pmid', 'title', 'pubdate', 'source'}]}."""
    cached = _cache_get(term)
    if cached is not None:
        return cached

    with httpx.Client() as client:
        search_resp = get_with_retry(
            client,
            ESEARCH_URL,
            {**TOOL_PARAMS, "db": "pubmed", "term": term, "retmode": "json", "retmax": retmax},
            HTTP_TIMEOUT,
        )
        result = search_resp.json().get("esearchresult", {})
        count = int(result.get("count", 0))
        pmids = result.get("idlist", [])

        hits = []
        if pmids:
            summary_resp = get_with_retry(
                client,
                ESUMMARY_URL,
                {**TOOL_PARAMS, "db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
                HTTP_TIMEOUT,
            )
            summary = summary_resp.json().get("result", {})
            for pmid in pmids:
                doc = summary.get(pmid, {})
                hits.append(
                    {
                        "pmid": pmid,
                        "title": doc.get("title"),
                        "pubdate": doc.get("pubdate"),
                        "source": doc.get("source"),
                    }
                )

    payload = {"count": count, "hits": hits}
    _cache_put(term, payload)
    return payload
