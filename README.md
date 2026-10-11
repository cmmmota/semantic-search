# semantic-search

Hybrid semantic search over a [paperless-ngx](https://docs.paperless-ngx.com/) archive and an
[Obsidian](https://obsidian.md/) vault, served to agents over MCP and to everything else over REST.

It replaced a proof of concept built on Open WebUI knowledge collections. The retrieval stack is
the same one that PoC was tuned on. What changed is that the index, its sync state and the
interface are now owned by a service built for the job.

## How it works

```
query ─▶ bge-m3 ─▶ pgvector top-50 ───────┐
      └─▶ IDF-weighted full-text top-50 ──┴▶ RRF ─▶ top-30 chunks ─▶ bge-reranker-v2-m3
                                              ─▶ group by document ─▶ top-k documents
```

- **Results are documents, not chunks.** Each result carries its best chunks as evidence. Use
  search to decide *which* document to read, then read it whole with `get_document`.
- **Two retrieval legs.** The vector leg handles meaning and crosses languages (an English
  question finds Portuguese documents). The keyword leg catches exact tokens such as invoice
  numbers, IBANs and model numbers. It uses the `simple` configuration with accent folding, and
  weights terms by BM25's IDF because Postgres' `ts_rank` has no notion of term rarity.
- **Degraded mode.** Both models run on llama.cpp servers. If the embedder is down, search
  falls back to keywords only; if the reranker is down, the fused order is returned. In both
  cases the response's `degraded` field says so.
- **Sync is a reconcile, not an import.** Each run makes the index match the source exactly.
  It lists the source cheaply and re-embeds only documents whose content hash changed.
  Deletions stop at `MAX_DELETE_RATIO`. A document's row and chunks are swapped in one
  transaction. Changing the model, the chunking or the embedded format re-embeds everything,
  through a pipeline fingerprint stored on each document.

## Sources

| Source | Reads | Change detection |
|---|---|---|
| `paperless` | the paperless REST API, extracted `content` only | `modified` as a prefilter, then the content hash |
| `obsidian` | a local directory of `.md` files (in the cluster, an rclone copy) | file mtime as a prefilter, then the content hash |

Obsidian notes are chunked by heading. Each chunk is embedded with its heading trail
(`Laundry > Dryer > Filter`), with link and image URLs reduced to their text. Frontmatter
`tags` and `aliases`, inline `#tags` and `[[wikilinks]]` are stored as metadata.

## Interfaces

**MCP** (streamable HTTP, stateless, at `/mcp`), with three tools:

- `search(query, sources?, k?, tags?, correspondent?, document_type?, created_after?, created_before?, path_prefix?)`
- `get_document(source, source_id, offset?, limit?)`
- `status()`: per source, the document count, last successful sync and last run outcome.

**REST:**

| Method | Path | Auth |
|---|---|---|
| GET | `/search?q=&k=&source=&tag=&correspondent=&document_type=&created_after=&created_before=&path_prefix=` | `API_TOKEN` |
| GET | `/documents/{source}/{source_id}?offset=&limit=` | `API_TOKEN` |
| GET | `/status` | `API_TOKEN` |
| POST | `/hooks/paperless` | `WEBHOOK_TOKEN` |
| GET | `/healthz`, `/metrics` | none |

The paperless webhook acks immediately, because paperless gives a webhook 5 seconds and does
not retry a timeout. It then debounces bursts (10s of quiet, capped at 120s) and runs the same
full reconcile as the CronJob. A per-source Postgres advisory lock keeps the two from overlapping.

## CLI

```
semantic-search serve                      # API + MCP + webhook on :8080
semantic-search sync paperless|obsidian    # one reconcile, then exit (--dry-run, --max-delete-ratio)
semantic-search migrate                    # apply schema migrations (every command does this first)
semantic-search search "query" [-k 5] [--source obsidian]
semantic-search eval eval/queries.yaml [--both] [-v]
```

## Configuration

| Variable | Default | Used by |
|---|---|---|
| `DATABASE_URL` | required | all |
| `EMBED_URL` | `http://llama-embeddings.ai.svc.cluster.local:11435/v1/embeddings` | all |
| `RERANK_URL` | `http://llama-reranker.ai.svc.cluster.local:11436/v1/rerank` | serve, search |
| `EMBED_MODEL` / `RERANK_MODEL` | `bge-m3` / `bge-reranker-v2-m3` | |
| `LLAMA_API_KEY` | empty | bearer token for both llama.cpp servers |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `CHUNK_MIN` | `1000` / `100` / `200` (characters) | sync |
| `EMBED_BATCH` | `8` | sync |
| `LEG_K` / `RERANK_K` / `RRF_K` | `50` / `30` / `60` | search |
| `MAX_DELETE_RATIO` | `0.25` | sync |
| `API_TOKEN` | required | serve |
| `WEBHOOK_TOKEN` | unset disables the webhook | serve |
| `PAPERLESS_URL`, `PAPERLESS_TOKEN` | required for paperless | sync, webhook |
| `PAPERLESS_PUBLIC_URL` | unset means no `uri` | paperless links in results |
| `VAULT_DIR` | `/vault` | obsidian |
| `OBSIDIAN_VAULT_NAME` | `Obsidian` | `obsidian://` links |

The database needs the `vector` and `unaccent` extensions. Migrations create them when they can;
a non-superuser role needs them created beforehand (CNPG: `postInitApplicationSQL`).

## Evaluation

`eval/queries.yaml` is a golden set of queries with the documents each should find. Because it
holds real document IDs, it is **gitignored**; `eval/queries.example.yaml` shows the format.
`eval` reports recall@1/5/10 and MRR per tag. `--both` runs the same set against the Open WebUI
PoC as a baseline (`OPENWEBUI_URL`, `OPENWEBUI_TOKEN`, `OPENWEBUI_PAPERLESS_KB`,
`OPENWEBUI_OBSIDIAN_KB`).

Cutover result (2026-10-11, 44 queries across PT, EN, cross-language, exact identifiers and
Obsidian):

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| semantic-search | 0.82 | 0.98 | 1.00 | 0.893 |
| Open WebUI PoC | 0.80 | 0.98 | 0.98 | 0.871 |

## Development

```
docker run -d --name ss-pg -e POSTGRES_PASSWORD=pw -p 55432:5432 pgvector/pgvector:pg16
docker exec ss-pg psql -U postgres -c "create database test"
TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55432/test uv run pytest
uv run ruff check
```

The integration tests drop and recreate the `public` schema of `TEST_DATABASE_URL`. Point it at
a throwaway database.
