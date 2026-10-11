"""Make the index match a source exactly.

Every run is a full reconcile, not an incremental import: documents added at
the source appear, changed ones are re-embedded, removed ones are deleted. A
run where nothing changed costs one listing and one query.

  1. listing()  -- every source_id with its version, no content
  2. the index  -- every stored (source_id, version, hash, pipeline)
  3. candidates -- new ids, plus stored ids whose version or pipeline moved
  4. fetch + render + hash the candidates; an unchanged hash only refreshes
     the stored version, it does not re-embed
  5. delete what the source no longer has, behind MAX_DELETE_RATIO

Embedding happens before the write transaction, and each document's row and
chunks are replaced in one transaction, so a search never sees half a
document and a crash mid-run leaves every document either old or new.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

from psycopg.types.json import Jsonb

from . import db, llama
from .config import settings
from .sources import Document, Source

log = logging.getLogger(__name__)

# Bump when the embedded text's layout changes in a way settings do not
# capture (e.g. the header format), to force a re-embed of everything.
_FORMAT = 2


class UnsafeSync(Exception):
    pass


@dataclass
class Stats:
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    removed: int = 0

    def __str__(self):
        return f"+{self.added} ~{self.changed} -{self.removed} (unchanged {self.unchanged})"


def pipeline() -> str:
    s = settings()
    return f"{s.embed_model}|{s.chunk_size}/{s.chunk_overlap}/{s.chunk_min}|f{_FORMAT}"


def embed_text(doc: Document, chunk) -> str:
    lines = list(doc.header)
    if chunk.heading_path:
        lines.append(f"Section: {chunk.heading_path}")
    return "\n".join(lines) + "\n\n" + chunk.text


def content_hash(doc: Document) -> str:
    payload = json.dumps(
        {
            "title": doc.title,
            "uri": doc.uri,
            "metadata": doc.metadata,
            "content": doc.content,
            "embedded": [embed_text(doc, c) for c in doc.chunks],
            "pipeline": pipeline(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _existing(source: str) -> dict[str, dict]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, source_id, source_version, content_hash, pipeline"
            " FROM documents WHERE source = %s",
            (source,),
        ).fetchall()
    return {r["source_id"]: r for r in rows}


def _write(source: str, doc: Document, version: str, digest: str, vectors) -> None:
    model = settings().embed_model
    with db.pool().connection() as conn, conn.transaction():
        doc_id = conn.execute(
            """
            INSERT INTO documents
              (source, source_id, title, uri, metadata, content,
               content_hash, source_version, pipeline, indexed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (source, source_id) DO UPDATE SET
              title = EXCLUDED.title, uri = EXCLUDED.uri,
              metadata = EXCLUDED.metadata, content = EXCLUDED.content,
              content_hash = EXCLUDED.content_hash,
              source_version = EXCLUDED.source_version,
              pipeline = EXCLUDED.pipeline, indexed_at = now()
            RETURNING id
            """,
            (source, doc.source_id, doc.title, doc.uri, Jsonb(doc.metadata), doc.content,
             digest, version, pipeline()),
        ).fetchone()["id"]
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, ord, heading_path, text, embedding, model)"
                " VALUES (%s, %s, %s, %s, %s::vector, %s)",
                [
                    (doc_id, i, c.heading_path, c.text, vec, model)
                    for i, (c, vec) in enumerate(zip(doc.chunks, vectors, strict=True))
                ],
            )


def _start_run(source: str) -> int:
    with db.pool().connection() as conn:
        return conn.execute(
            "INSERT INTO sync_runs (source, started_at) VALUES (%s, now()) RETURNING id",
            (source,),
        ).fetchone()["id"]


def _finish_run(run_id: int, stats: Stats, error: str | None) -> None:
    with db.pool().connection() as conn:
        conn.execute(
            "UPDATE sync_runs SET finished_at = now(), added = %s, changed = %s,"
            " unchanged = %s, removed = %s, ok = %s, error = %s WHERE id = %s",
            (stats.added, stats.changed, stats.unchanged, stats.removed,
             error is None, error, run_id),
        )


def reconcile(source: Source, *, dry_run: bool = False, max_delete_ratio: float | None = None) -> Stats:
    ratio = settings().max_delete_ratio if max_delete_ratio is None else max_delete_ratio
    with db.source_lock(source.name):
        run_id = None if dry_run else _start_run(source.name)
        stats = Stats()
        try:
            _reconcile(source, stats, dry_run, ratio)
        except Exception as e:
            if run_id is not None:
                _finish_run(run_id, stats, f"{type(e).__name__}: {e}"[:1000])
            raise
        if run_id is not None:
            _finish_run(run_id, stats, None)
        return stats


def _reconcile(source: Source, stats: Stats, dry_run: bool, ratio: float) -> None:
    listing = source.listing()
    existing = _existing(source.name)
    current = pipeline()

    adds = sorted(set(listing) - set(existing))
    deletes = sorted(set(existing) - set(listing))
    candidates = sorted(
        sid for sid in set(listing) & set(existing)
        if existing[sid]["source_version"] != listing[sid]
        or existing[sid]["pipeline"] != current
    )
    log.info("%s: %d at source, %d indexed | +%d ?%d -%d",
             source.name, len(listing), len(existing), len(adds), len(candidates), len(deletes))

    if deletes and existing and len(deletes) / len(existing) > ratio:
        raise UnsafeSync(
            f"{len(deletes)} of {len(existing)} {source.name} documents would be deleted "
            f"(> {ratio:.0%}). This usually means the source returned a short listing. "
            f"Re-run with --max-delete-ratio raised if the deletion is real."
        )
    if dry_run:
        stats.added, stats.removed = len(adds), len(deletes)
        log.info("dry run: %d candidates not fetched", len(candidates))
        return

    for doc in source.fetch(adds + candidates):
        prior = existing.get(doc.source_id)
        digest = content_hash(doc)
        version = listing[doc.source_id]
        if prior and prior["content_hash"] == digest:
            # The source touched something we do not render. Record the new
            # version so it stops being a candidate; do not re-embed.
            with db.pool().connection() as conn:
                conn.execute(
                    "UPDATE documents SET source_version = %s WHERE id = %s",
                    (version, prior["id"]),
                )
            stats.unchanged += 1
            continue
        vectors = llama.embed([embed_text(doc, c) for c in doc.chunks])
        _write(source.name, doc, version, digest, vectors)
        if prior:
            stats.changed += 1
        else:
            stats.added += 1
        log.info("indexed %s %s (%d chunks)", source.name, doc.source_id, len(doc.chunks))

    if deletes:
        with db.pool().connection() as conn:
            conn.execute(
                "DELETE FROM documents WHERE source = %s AND source_id = ANY(%s)",
                (source.name, deletes),
            )
        for sid in deletes:
            log.info("removed %s %s", source.name, sid)
        stats.removed = len(deletes)

    stats.unchanged += len(existing) - len(candidates) - len(deletes)
    log.info("%s sync complete: %s", source.name, stats)
