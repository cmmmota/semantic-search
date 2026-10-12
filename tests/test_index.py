import pytest

from semantic_search import llama
from semantic_search.chunking import Chunk
from semantic_search.reconcile import UnsafeSync, reconcile
from semantic_search.search import Filters, get_document, search, status
from semantic_search.sources import Document


class FakeSource:
    name = "fake"

    def __init__(self, docs: dict[str, tuple[str, str]], created: dict[str, str] | None = None):
        # source_id -> (version, text); created: source_id -> ISO date
        self.docs = docs
        self.created = created or {}

    def listing(self):
        return {sid: v for sid, (v, _) in self.docs.items()}

    def fetch(self, ids):
        for sid in ids:
            if sid in self.docs:
                text = self.docs[sid][1]
                meta = {"tags": ["t"]}
                if sid in self.created:
                    meta["created"] = self.created[sid]
                yield Document(sid, f"Doc {sid}", text, [Chunk(text)], [f"Title: Doc {sid}"], metadata=meta)


def test_reconcile_add_change_delete(database):
    src = FakeSource({"1": ("v1", "washing machine filter"), "2": ("v1", "car insurance policy")})
    s = reconcile(src)
    assert (s.added, s.changed, s.removed) == (2, 0, 0)
    first = database["embed"]

    # No change: nothing fetched, nothing embedded.
    s = reconcile(src)
    assert (s.added, s.changed, s.removed, s.unchanged) == (0, 0, 0, 2)
    assert database["embed"] == first

    # Version bump with identical text: re-hashed, not re-embedded.
    src.docs["1"] = ("v2", "washing machine filter")
    s = reconcile(src)
    assert (s.changed, s.unchanged) == (0, 2)
    assert database["embed"] == first

    # Real edit and a delete.
    src.docs["1"] = ("v3", "dryer lint filter cleaning")
    del src.docs["2"]
    s = reconcile(src, max_delete_ratio=1.0)
    assert (s.changed, s.removed) == (1, 1)
    assert get_document("fake", "1")["content"] == "dryer lint filter cleaning"
    assert get_document("fake", "2") is None


def test_delete_guard(database):
    src = FakeSource({str(i): ("v", f"doc {i}") for i in range(10)})
    reconcile(src)
    src.docs = {"0": ("v", "doc 0")}
    with pytest.raises(UnsafeSync):
        reconcile(src)
    assert status()[0]["last_run"]["ok"] is False


def test_search_hybrid_and_degraded(database):
    reconcile(FakeSource({
        "a": ("v", "robot vacuum maintenance brush"),
        "b": ("v", "invoice number FT2024/123 electricity"),
        "c": ("v", "máquina de lavar roupa manual"),
    }))
    r = search("robot vacuum brush")
    assert r.results[0]["source_id"] == "a"
    assert "reranker unavailable" in r.degraded

    # Exact identifiers come through the keyword leg.
    assert search("FT2024/123").results[0]["source_id"] == "b"
    # Accent-folded keyword match.
    assert search("maquina lavar").results[0]["source_id"] == "c"
    # Filters apply.
    assert search("robot", filters=Filters(sources=["other"])).results == []
    assert search("robot", filters=Filters(tags=["t"])).results


def test_search_newest(database, monkeypatch):
    reconcile(FakeSource({
        "jun": ("v", "electricity bill june energy kwh total"),
        "aug": ("v", "electricity bill august energy kwh"),
        "jul": ("v", "electricity bill july energy kwh total amount due"),
        "manual": ("v", "dishwasher manual energy label"),
        "undated": ("v", "electricity bill energy kwh"),
    }, created={"jun": "2026-07-02", "jul": "2026-08-03", "aug": "2026-09-03", "manual": "2026-10-01"}))

    # No reranker: the fused top k qualify, then newest first, undated last.
    ids = [d["source_id"] for d in search("electricity bill", k=4, sort="newest").results]
    assert ids[0] == "aug"
    assert ids[-1] == "undated"

    # With scores, documents far below the best are not "the latest X",
    # however new.
    def rerank(query, documents, attempts=2):
        return [-8.0 if "dishwasher" in d else 1.0 for d in documents]

    monkeypatch.setattr(llama, "rerank", rerank)
    r = search("electricity bill energy", k=10, sort="newest")
    assert [d["source_id"] for d in r.results] == ["aug", "jul", "jun", "undated"]
    assert r.degraded is None

    with pytest.raises(ValueError):
        search("x", sort="oldest")


def test_snippet_chars(database):
    reconcile(FakeSource({"long": ("v", "contract " * 200)}))
    text = search("contract", snippet_chars=50).results[0]["snippets"][0]["text"]
    assert len(text) <= 51 and text.endswith("…")
