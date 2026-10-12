"""Settings, read once from the environment.

Everything is an environment variable so the same image serves both the
long-running API and the per-source sync CronJobs; each reads only what it
needs, and a missing required value fails at the point of use with its name.
"""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"{name} is not set")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))

    embed_url: str = field(default_factory=lambda: _env(
        "EMBED_URL", "http://llama-embeddings.ai.svc.cluster.local:11435/v1/embeddings"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "bge-m3"))
    # llama-embeddings runs N_PARALLEL=1 with an 8192-token ubatch; 8 chunks of
    # ~1000 characters stays far inside it. Matches the Open WebUI PoC.
    embed_batch: int = field(default_factory=lambda: int(_env("EMBED_BATCH", "8")))
    rerank_url: str = field(default_factory=lambda: _env(
        "RERANK_URL", "http://llama-reranker.ai.svc.cluster.local:11436/v1/rerank"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "bge-reranker-v2-m3"))
    # One bearer token for both llama.cpp servers (LLAMA_API_KEY).
    llama_api_key: str = field(default_factory=lambda: _env("LLAMA_API_KEY", ""))

    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))
    chunk_min: int = field(default_factory=lambda: int(_env("CHUNK_MIN", "200")))

    # Candidates pulled from each retrieval leg before fusion, and how many
    # fused chunks go to the reranker.
    leg_k: int = field(default_factory=lambda: int(_env("LEG_K", "50")))
    rerank_k: int = field(default_factory=lambda: int(_env("RERANK_K", "30")))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    # sort=newest: documents (one best chunk each) sent to the reranker, and
    # how far below the best reranker score a document may fall and still
    # count as relevant. bge-reranker scores are logits; on the paperless
    # corpus matching bills sit within ~3 of the best, unrelated documents 5+
    # below it.
    newest_pool: int = field(default_factory=lambda: int(_env("NEWEST_POOL", "50")))
    newest_margin: float = field(default_factory=lambda: float(_env("NEWEST_MARGIN", "4.0")))

    # Refuse a sync that would delete more than this share of a source's
    # documents: an empty or short listing from the source must not be
    # mistaken for the user deleting everything.
    max_delete_ratio: float = field(default_factory=lambda: float(_env("MAX_DELETE_RATIO", "0.25")))

    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "120")))


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
