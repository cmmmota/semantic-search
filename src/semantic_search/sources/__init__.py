"""Document sources.

A source answers two questions: what exists right now, cheaply (`listing`),
and what a given set of documents look like in full (`fetch`). Everything
else -- diffing, hashing, embedding, deleting -- is the reconcile loop's job
and is identical for every source.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from ..chunking import Chunk


@dataclass
class Document:
    source_id: str
    title: str
    content: str
    chunks: list[Chunk]
    # Lines prepended to every chunk before embedding (never shown to callers):
    # the context a chunk loses when cut out of its document.
    header: list[str]
    uri: str | None = None
    metadata: dict = field(default_factory=dict)


class Source(Protocol):
    name: str

    def listing(self) -> dict[str, str]:
        """source_id -> an opaque version string that changes when the
        document does. Must not fetch document content."""
        ...

    def fetch(self, ids: list[str]) -> Iterator[Document]:
        """Full documents for `ids`. Ids that vanished since `listing` are
        skipped silently; the next run deletes them."""
        ...


def get(name: str) -> Source:
    if name == "paperless":
        from .paperless import Paperless
        return Paperless()
    if name == "obsidian":
        from .obsidian import Obsidian
        return Obsidian()
    raise ValueError(f"unknown source {name!r}")


NAMES = ("paperless", "obsidian")
