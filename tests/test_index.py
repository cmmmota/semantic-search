import pytest

from semantic_search.chunking import Chunk
from semantic_search.reconcile import UnsafeSync, reconcile
from semantic_search.search import Filters, get_document, search, status
from semantic_search.sources import Document


class FakeSource:
    name = "fake"

    def __init__(self, docs: dict[str, tuple[str, str]]):
        # source_id -> (version, text)
        self.docs = docs

    def listing(self):
        return {sid: v for sid, (v, _) in self.docs.items()}

    def fetch(self, ids):
        for sid in ids:
            if sid in self.docs:
                text = self.docs[sid][1]
                yield Document(sid, f"Doc {sid}", text, [Chunk(text)], [f"Title: Doc {sid}"],
                               metadata={"tags": ["t"]})


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
