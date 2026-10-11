"""Clients for the llama.cpp embedding and reranking servers."""

import logging
import time

import httpx

from .config import settings

log = logging.getLogger(__name__)


class ModelUnavailable(Exception):
    """The model server could not be reached or kept failing.

    Search catches this and degrades; ingestion lets it fail the run, and the
    next scheduled run retries.
    """


_client: httpx.Client | None = None


def _http() -> httpx.Client:
    global _client
    if _client is None:
        s = settings()
        headers = {"Authorization": f"Bearer {s.llama_api_key}"} if s.llama_api_key else {}
        _client = httpx.Client(timeout=s.http_timeout, headers=headers)
    return _client


def _post(url: str, payload: dict, *, attempts: int) -> dict:
    """POST with exponential backoff.

    Both servers run a single slot (N_PARALLEL=1) shared with paperless's own
    nightly index, so a busy server is normal and worth waiting out; a 4xx is
    a bug in the request and is not retried.
    """
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            r = _http().post(url, json=payload)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json()
            error = f"HTTP {r.status_code}: {r.text[:200]}"
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as e:
            error = f"{type(e).__name__}: {e}"
        if attempt < attempts:
            log.warning("%s failed (%s), retry %d/%d in %.0fs", url, error, attempt, attempts - 1, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise ModelUnavailable(f"{url}: {error}")


def embed(texts: list[str], *, attempts: int = 6) -> list[list[float]]:
    """Embed texts in batches, preserving order."""
    s = settings()
    out: list[list[float]] = []
    for i in range(0, len(texts), s.embed_batch):
        batch = texts[i:i + s.embed_batch]
        data = _post(s.embed_url, {"model": s.embed_model, "input": batch}, attempts=attempts)["data"]
        out.extend(row["embedding"] for row in sorted(data, key=lambda r: r["index"]))
    return out


def rerank(query: str, documents: list[str], *, attempts: int = 2) -> list[float]:
    """Relevance score per document, in input order."""
    s = settings()
    resp = _post(
        s.rerank_url,
        {"model": s.rerank_model, "query": query, "documents": documents, "top_n": len(documents)},
        attempts=attempts,
    )
    scores = [0.0] * len(documents)
    for row in resp["results"]:
        scores[row["index"]] = row["relevance_score"]
    return scores
