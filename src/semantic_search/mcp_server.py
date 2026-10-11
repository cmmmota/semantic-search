"""MCP tools: the interface agents use.

Tool descriptions are written for the agent, not for humans reading this
file: they carry the usage rules that used to live in Hermes' SKILL.md files
("search to pick the document, then read the whole thing").
"""

from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import search as search_mod

INSTRUCTIONS = """\
Semantic search over the user's personal documents:
- "paperless": scanned and digital documents from paperless-ngx -- invoices,
  contracts, manuals, warranties, bank and tax documents. Mostly Portuguese.
- "obsidian": the user's Obsidian notes -- guides, appliance catalog, homelab
  docs, ideas. Mostly English.

Search understands meaning and works across languages: an English question
finds Portuguese documents and vice versa.

Use `search` to decide WHICH documents are relevant, then `get_document` to
read them in full before answering. Snippets are evidence for the ranking,
not the answer: anything outside the retrieved chunk is invisible in them.

Ranking is approximate. Read the top few candidates rather than trusting the
first. An empty result is not proof that something does not exist, and a
response with `degraded` set means part of the pipeline was down -- say so
instead of concluding the document is absent. Call `status` when freshness
matters (e.g. "the document I scanned an hour ago").
"""

mcp = MCPServer("semantic-search", instructions=INSTRUCTIONS)


@mcp.tool()
async def search(
    query: Annotated[str, Field(description="Natural-language question or keywords, any language.")],
    sources: Annotated[list[str] | None, Field(
        description='Limit to "paperless" and/or "obsidian". Omit to search both.')] = None,
    k: Annotated[int, Field(ge=1, le=20, description="Number of documents to return.")] = 5,
    tags: Annotated[list[str] | None, Field(description="Only documents carrying ALL of these tags.")] = None,
    correspondent: Annotated[str | None, Field(
        description="paperless only: correspondent name, case-insensitive substring.")] = None,
    document_type: Annotated[str | None, Field(
        description="paperless only: document type, case-insensitive substring.")] = None,
    created_after: Annotated[str | None, Field(
        description="paperless only: ISO date (YYYY-MM-DD), inclusive.")] = None,
    created_before: Annotated[str | None, Field(
        description="paperless only: ISO date (YYYY-MM-DD), inclusive.")] = None,
    path_prefix: Annotated[str | None, Field(
        description='obsidian only: vault-relative folder, e.g. "🗂️ Appliance Catalog/".')] = None,
) -> dict[str, Any]:
    """Find the documents most relevant to a query.

    Returns documents ranked best first, each with `source`, `source_id`
    (pass both to get_document), title, link, metadata and the best-matching
    snippets. Read the documents before answering from them.
    """
    filters = search_mod.Filters(
        sources=sources, tags=tags, correspondent=correspondent, document_type=document_type,
        created_after=created_after, created_before=created_before, path_prefix=path_prefix,
    )
    result = await anyio.to_thread.run_sync(lambda: search_mod.search(query, k=k, filters=filters))
    return {"results": result.results, "degraded": result.degraded}


@mcp.tool()
async def get_document(
    source: Annotated[str, Field(description='"paperless" or "obsidian", from a search result.')],
    source_id: Annotated[str, Field(description="source_id from a search result.")],
    offset: Annotated[int, Field(ge=0, description="Character offset to start reading at.")] = 0,
    limit: Annotated[int, Field(ge=1, le=200_000, description="Maximum characters to return.")] = 50_000,
) -> dict[str, Any]:
    """Read a document's full indexed text.

    Long documents are paged: when `truncated` is true, call again with
    `offset` advanced by the returned content's length. For a summary or a
    question spanning the whole document, read all of it.
    """
    doc = await anyio.to_thread.run_sync(
        lambda: search_mod.get_document(source, source_id, offset, limit))
    if doc is None:
        return {"error": f"no {source} document {source_id!r} in the index"}
    return doc


@mcp.tool()
async def status() -> dict[str, Any]:
    """Index freshness per source: document count, last successful sync and
    the outcome of the most recent run. A document added after the last
    successful sync is not searchable yet."""
    return {"sources": await anyio.to_thread.run_sync(search_mod.status)}
