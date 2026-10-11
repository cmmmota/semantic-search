"""Hybrid retrieval: vector + keyword, fused, reranked, grouped by document.

  query ─▶ bge-m3 ─▶ vector top-N ──┐
        └─▶ IDF-weighted keyword top-N ┴▶ RRF ─▶ top-K chunks ─▶ reranker
                                       ─▶ group by document (best chunk wins)

Results are DOCUMENTS with their best chunks as evidence, not loose chunks:
callers use search to decide which document to read, then read it whole
(`get_document`). A chunk cut out of a contract answers confidently and
wrongly about everything that was not in it.

DEGRADED MODE. Both models run on thinktank's GPU. If the embedder is down
the vector leg is skipped; if the reranker is down the fused order stands.
Either way the response says so in `degraded` -- a silent drop in quality is
indistinguishable from "nothing relevant exists", which is the worst answer
this service can give.
"""

import logging
import math
import time
from dataclasses import dataclass, field

from psycopg import sql

from . import db, llama
from .config import settings
from .metrics import SEARCH_DEGRADED, SEARCH_SECONDS

log = logging.getLogger(__name__)

# Most query terms considered by the keyword leg; longer queries are
# overwhelmingly natural-language questions where the vector leg leads.
_MAX_TERMS = 32


@dataclass
class Filters:
    sources: list[str] | None = None
    tags: list[str] | None = None            # all must be present
    correspondent: str | None = None         # paperless; case-insensitive substring
    document_type: str | None = None         # paperless; case-insensitive substring
    created_after: str | None = None         # paperless; ISO date, inclusive
    created_before: str | None = None        # paperless; ISO date, inclusive
    path_prefix: str | None = None           # obsidian; vault-relative

    def where(self) -> tuple[sql.Composable, list]:
        clauses: list[sql.Composable] = [sql.SQL("TRUE")]
        params: list = []
        if self.sources:
            clauses.append(sql.SQL("d.source = ANY(%s)"))
            params.append(list(self.sources))
        if self.tags:
            clauses.append(sql.SQL("d.metadata->'tags' ?& %s"))
            params.append(list(self.tags))
        if self.correspondent:
            clauses.append(sql.SQL("d.metadata->>'correspondent' ILIKE %s"))
            params.append(f"%{_like_escape(self.correspondent)}%")
        if self.document_type:
            clauses.append(sql.SQL("d.metadata->>'document_type' ILIKE %s"))
            params.append(f"%{_like_escape(self.document_type)}%")
        if self.created_after:
            clauses.append(sql.SQL("(d.metadata->>'created')::date >= %s::date"))
            params.append(self.created_after)
        if self.created_before:
            clauses.append(sql.SQL("(d.metadata->>'created')::date <= %s::date"))
            params.append(self.created_before)
        if self.path_prefix:
            clauses.append(sql.SQL("d.source = 'obsidian' AND d.source_id LIKE %s"))
            params.append(_like_escape(self.path_prefix.lstrip("/")) + "%")
        return sql.SQL(" AND ").join(clauses), params


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    rrf: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank: float | None = None


@dataclass
class Result:
    results: list[dict] = field(default_factory=list)
    degraded: str | None = None


def _vector_leg(conn, qvec, where, params, k) -> list[int]:
    # Iterative scan keeps filtered HNSW queries from returning short: without
    # it the index yields ef_search candidates and the filter discards most.
    conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    conn.execute("SET LOCAL hnsw.ef_search = 200")
    q = sql.SQL(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE {where} ORDER BY c.embedding <=> %s::vector LIMIT %s"
    ).format(where=where)
    return [r["id"] for r in conn.execute(q, [*params, qvec, k])]


def _keyword_leg(conn, query, where, params, k) -> list[int]:
    """Chunks ranked by the summed IDF of the query terms they contain.

    Postgres' own ts_rank has no notion of term rarity, so in an OR query a
    chunk matching "de" and "the" outranks one matching an invoice number.
    Weighting each term by BM25's IDF fixes exactly that, and on a corpus of
    a few thousand chunks the per-term document counts are cheap index hits.
    """
    terms = [
        r["lexeme"] for r in conn.execute(
            "SELECT DISTINCT lexeme FROM unnest(to_tsvector('simple', f_unaccent(%s)))",
            (query,),
        )
    ][:_MAX_TERMS]
    if not terms:
        return []
    total = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] or 1
    weighted = []
    for t in terms:
        df = conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE tsv @@ to_tsquery('simple', quote_literal(%s))",
            (t,),
        ).fetchone()["n"]
        if df:
            weighted.append((t, math.log((total - df + 0.5) / (df + 0.5) + 1)))
    if not weighted:
        return []
    score = sql.SQL(" + ").join(
        sql.SQL("CASE WHEN c.tsv @@ to_tsquery('simple', quote_literal({t})) THEN {w} ELSE 0 END")
        .format(t=sql.Literal(t), w=sql.Literal(w))
        for t, w in weighted
    )
    any_term = sql.SQL(" || ").join(
        sql.SQL("to_tsquery('simple', quote_literal({t}))").format(t=sql.Literal(t))
        for t, _ in weighted
    )
    q = sql.SQL(
        "SELECT c.id, ({score}) AS score FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE {where} AND c.tsv @@ ({any_term})"
        " ORDER BY score DESC, c.id LIMIT %s"
    ).format(score=score, where=where, any_term=any_term)
    return [r["id"] for r in conn.execute(q, [*params, k])]


def search(query: str, *, k: int = 5, filters: Filters | None = None, snippets: int = 2) -> Result:
    started = time.monotonic()
    s = settings()
    filters = filters or Filters()
    where, params = filters.where()
    degraded: list[str] = []

    qvec = None
    try:
        qvec = llama.embed([query], attempts=2)[0]
    except llama.ModelUnavailable as e:
        log.warning("embedder unavailable, keyword-only search: %s", e)
        degraded.append("embeddings unavailable: keyword search only")
        SEARCH_DEGRADED.labels("embeddings").inc()

    with db.pool().connection() as conn, conn.transaction():
        vector_ids = _vector_leg(conn, qvec, where, params, s.leg_k) if qvec else []
        keyword_ids = _keyword_leg(conn, query, where, params, s.leg_k)

        hits: dict[int, Hit] = {}
        for leg, ids in (("vector_rank", vector_ids), ("keyword_rank", keyword_ids)):
            for rank, cid in enumerate(ids, 1):
                h = hits.setdefault(cid, Hit(cid, 0))
                setattr(h, leg, rank)
                h.rrf += 1.0 / (s.rrf_k + rank)
        top = sorted(hits.values(), key=lambda h: h.rrf, reverse=True)[:s.rerank_k]
        if not top:
            SEARCH_SECONDS.observe(time.monotonic() - started)
            return Result([], "; ".join(degraded) or None)

        rows = {
            r["chunk_id"]: r for r in conn.execute(
                "SELECT c.id AS chunk_id, c.document_id, c.heading_path, c.text,"
                " d.source, d.source_id, d.title, d.uri, d.metadata"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                " WHERE c.id = ANY(%s)",
                ([h.chunk_id for h in top],),
            )
        }

    for h in top:
        h.document_id = rows[h.chunk_id]["document_id"]

    try:
        scores = llama.rerank(query, [_rerank_text(rows[h.chunk_id]) for h in top])
        for h, sc in zip(top, scores, strict=True):
            h.rerank = sc
        top.sort(key=lambda h: h.rerank, reverse=True)
    except llama.ModelUnavailable as e:
        log.warning("reranker unavailable, using fused order: %s", e)
        degraded.append("reranker unavailable: results are not reranked")
        SEARCH_DEGRADED.labels("reranker").inc()

    # Group by document; a document ranks by its best chunk.
    docs: dict[int, dict] = {}
    for h in top:
        row = rows[h.chunk_id]
        d = docs.get(h.document_id)
        if d is None:
            if len(docs) >= k:
                continue
            d = docs[h.document_id] = {
                "source": row["source"],
                "source_id": row["source_id"],
                "title": row["title"],
                "uri": row["uri"],
                "metadata": row["metadata"],
                "score": h.rerank if h.rerank is not None else h.rrf,
                "snippets": [],
            }
        if len(d["snippets"]) < snippets:
            d["snippets"].append({
                "section": row["heading_path"],
                "text": row["text"],
                "score": h.rerank if h.rerank is not None else h.rrf,
                "vector_rank": h.vector_rank,
                "keyword_rank": h.keyword_rank,
            })

    SEARCH_SECONDS.observe(time.monotonic() - started)
    return Result(list(docs.values()), "; ".join(degraded) or None)


def _rerank_text(row) -> str:
    head = row["title"]
    if row["heading_path"]:
        head += f" — {row['heading_path']}"
    return f"{head}\n\n{row['text']}"


def get_document(source: str, source_id: str, offset: int = 0, limit: int = 50_000) -> dict | None:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT source, source_id, title, uri, metadata, content, indexed_at"
            " FROM documents WHERE source = %s AND source_id = %s",
            (source, source_id),
        ).fetchone()
    if row is None:
        return None
    content = row.pop("content")
    row["indexed_at"] = row["indexed_at"].isoformat()
    row["length"] = len(content)
    row["offset"] = offset
    row["content"] = content[offset:offset + limit]
    row["truncated"] = offset + limit < len(content)
    return row


def status() -> list[dict]:
    with db.pool().connection() as conn:
        counts = {
            r["source"]: r for r in conn.execute(
                "SELECT source, count(*) AS documents, max(indexed_at) AS last_indexed"
                " FROM documents GROUP BY source"
            )
        }
        runs = {
            r["source"]: r for r in conn.execute(
                "SELECT DISTINCT ON (source) source, started_at, finished_at, ok, error,"
                " added, changed, removed FROM sync_runs"
                " WHERE finished_at IS NOT NULL ORDER BY source, finished_at DESC"
            )
        }
        last_ok = {
            r["source"]: r["ts"] for r in conn.execute(
                "SELECT source, max(finished_at) AS ts FROM sync_runs WHERE ok GROUP BY source"
            )
        }
    out = []
    for source in sorted(set(counts) | set(runs)):
        c, r = counts.get(source, {}), runs.get(source)
        out.append({
            "source": source,
            "documents": c.get("documents", 0),
            "last_successful_sync": last_ok[source].isoformat() if source in last_ok else None,
            "last_run": None if r is None else {
                "finished_at": r["finished_at"].isoformat(),
                "ok": r["ok"],
                "error": r["error"],
                "added": r["added"], "changed": r["changed"], "removed": r["removed"],
            },
        })
    return out
