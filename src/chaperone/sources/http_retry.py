"""Shared retry/backoff for the external APIs (HPA, PubMed, STRING, UniProt)
— all of them can 429 under concurrent batch load, and all have shown
transient connection resets/timeouts in practice; none require an API key
for the retry to help."""
import time

import httpx

MAX_RETRIES = 5


def get_with_retry(client: httpx.Client, url: str, params: dict, timeout: float) -> httpx.Response:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, params=params, timeout=timeout)
        except httpx.TransportError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(0.5 * (2**attempt))
            continue
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        time.sleep(0.5 * (2**attempt))
    resp.raise_for_status()
    return resp
