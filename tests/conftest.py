import hashlib
import math
import os

import pytest


def fake_vector(text: str) -> list[float]:
    """Deterministic bag-of-words vector: texts sharing words are close."""
    v = [0.0] * 1024
    for word in text.lower().split():
        h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
        v[h % 1024] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


@pytest.fixture
def database(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    monkeypatch.setenv("DATABASE_URL", url)
    import psycopg

    from semantic_search import config, db, llama

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    config._settings = None
    if db._pool is not None:
        db._pool.close()
        db._pool = None
    db.migrate()

    calls = {"embed": 0}

    def embed(texts, attempts=6):
        calls["embed"] += len(texts)
        return [fake_vector(t) for t in texts]

    def rerank(query, documents, attempts=2):
        raise llama.ModelUnavailable("no reranker in tests")

    monkeypatch.setattr(llama, "embed", embed)
    monkeypatch.setattr(llama, "rerank", rerank)
    yield calls
    db._pool.close()
    db._pool = None
