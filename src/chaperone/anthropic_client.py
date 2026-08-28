"""Shared AsyncAnthropic client construction. The local dev proxy this
project runs behind (https_proxy env var) intermittently drops requests —
confirmed via repeated live testing that api.anthropic.com is directly
reachable from this host, so every script in this project should bypass the
proxy for the Anthropic client specifically (other httpx users — HPA/UniProt/
STRING/etc. clients — keep using it as before)."""
import httpx
from anthropic import AsyncAnthropic


def make_client() -> AsyncAnthropic:
    return AsyncAnthropic(http_client=httpx.AsyncClient(trust_env=False))
