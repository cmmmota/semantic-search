"""paperless-ngx.

Only the extracted `content` is indexed, not the original files: it is
already OCR'd (and re-OCR'd by paperless-gpt), so no extraction step is
needed. Tables arrive flattened; that is a limit of the OCR text.
"""

import os
import urllib.parse
from collections.abc import Iterator

import httpx

from ..chunking import Chunk, split_text
from ..config import settings
from . import Document

# How many ids go in one `id__in=` query when fetching content.
_CHUNK = 100


class Paperless:
    name = "paperless"

    def __init__(self):
        self.url = os.environ["PAPERLESS_URL"].rstrip("/")
        # Where a human opens a document; the API URL is in-cluster only.
        self.public_url = os.environ.get("PAPERLESS_PUBLIC_URL", "").rstrip("/")
        self.http = httpx.Client(
            timeout=settings().http_timeout,
            headers={"Authorization": f"Token {os.environ['PAPERLESS_TOKEN']}"},
        )

    def _pages(self, path: str) -> Iterator[dict]:
        url = f"{self.url}{path}"
        while url:
            r = self.http.get(url)
            r.raise_for_status()
            page = r.json()
            yield from page["results"]
            url = page.get("next")

    def _lookup(self, path: str) -> dict[int, str]:
        return {row["id"]: row["name"] for row in self._pages(path)}

    def listing(self) -> dict[str, str]:
        # `modified` is auto_now, so it moves on every save: a superset of
        # "the text we render changed". The content hash then decides.
        return {
            str(doc["id"]): doc["modified"]
            for doc in self._pages("/api/documents/?page_size=500&fields=id,modified")
        }

    def fetch(self, ids: list[str]) -> Iterator[Document]:
        if not ids:
            return
        correspondents = self._lookup("/api/correspondents/?page_size=500")
        doctypes = self._lookup("/api/document_types/?page_size=500")
        tags = self._lookup("/api/tags/?page_size=500")
        s = settings()
        fields = "id,title,content,created,correspondent,document_type,tags"
        for i in range(0, len(ids), _CHUNK):
            query = urllib.parse.urlencode({
                "page_size": _CHUNK,
                "id__in": ",".join(ids[i:i + _CHUNK]),
                "fields": fields,
            })
            for doc in self._pages(f"/api/documents/?{query}"):
                title = doc.get("title") or f"Document {doc['id']}"
                meta = {
                    "correspondent": correspondents.get(doc.get("correspondent")),
                    "document_type": doctypes.get(doc.get("document_type")),
                    "tags": sorted(t for t in (tags.get(x) for x in doc.get("tags") or []) if t),
                    "created": doc.get("created"),
                }
                meta = {k: v for k, v in meta.items() if v}
                header = [f"Title: {title}"]
                if "correspondent" in meta:
                    header.append(f"Correspondent: {meta['correspondent']}")
                if "document_type" in meta:
                    header.append(f"Type: {meta['document_type']}")
                if "created" in meta:
                    header.append(f"Date: {meta['created']}")
                if meta.get("tags"):
                    header.append(f"Tags: {', '.join(meta['tags'])}")
                content = doc.get("content") or ""
                chunks = [
                    Chunk(t) for t in split_text(content, s.chunk_size, s.chunk_overlap, s.chunk_min)
                ] or [Chunk(title)]
                yield Document(
                    source_id=str(doc["id"]),
                    title=title,
                    content=content,
                    chunks=chunks,
                    header=header,
                    uri=f"{self.public_url}/documents/{doc['id']}/details" if self.public_url else None,
                    metadata=meta,
                )

