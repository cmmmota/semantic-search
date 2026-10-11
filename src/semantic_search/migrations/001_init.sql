-- Both extensions need superuser, which the application role is not. In the
-- cluster they are created by CNPG's postInitApplicationSQL; IF NOT EXISTS
-- makes this a no-op there and lets a local superuser database bootstrap
-- itself.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- unaccent() is STABLE (it reads a dictionary), so it cannot appear in a
-- generated column. Pinning the dictionary by name makes the result a pure
-- function of the input, which is what IMMUTABLE promises.
CREATE OR REPLACE FUNCTION f_unaccent(text) RETURNS text
  LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
  AS $$ SELECT public.unaccent('public.unaccent'::regdictionary, $1) $$;

CREATE TABLE documents (
  id              bigserial PRIMARY KEY,
  source          text        NOT NULL,
  source_id       text        NOT NULL,
  title           text        NOT NULL,
  uri             text,
  metadata        jsonb       NOT NULL DEFAULT '{}',
  content         text        NOT NULL,
  -- sha256 of everything that was chunked and embedded. The authority on
  -- whether a document needs re-embedding.
  content_hash    text        NOT NULL,
  -- The source's own change marker (paperless `modified`, file mtime), stored
  -- verbatim and only ever compared for inequality. A prefilter, not proof.
  source_version  text,
  -- Fingerprint of how the document was processed (model, chunking, header
  -- format). A mismatch re-embeds the document even though the source did
  -- not change, so tuning a setting reaches the whole corpus.
  pipeline        text        NOT NULL,
  indexed_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source, source_id)
);
CREATE INDEX documents_metadata ON documents USING gin (metadata);

CREATE TABLE chunks (
  id           bigserial PRIMARY KEY,
  document_id  bigint  NOT NULL REFERENCES documents ON DELETE CASCADE,
  ord          int     NOT NULL,
  heading_path text,
  text         text    NOT NULL,
  embedding    vector(1024) NOT NULL,
  model        text    NOT NULL,
  -- 'simple': the corpus mixes Portuguese and English, and either language's
  -- stemmer mangles the other. Meaning comes from the embeddings; this leg
  -- exists for exact tokens -- invoice numbers, NIFs, model numbers, names.
  -- Accents are folded so "maquina" finds "máquina".
  tsv          tsvector GENERATED ALWAYS AS (
                 to_tsvector('simple', f_unaccent(coalesce(heading_path, '') || ' ' || text))
               ) STORED
);
CREATE INDEX chunks_document ON chunks (document_id);
CREATE INDEX chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);

CREATE TABLE sync_runs (
  id          bigserial PRIMARY KEY,
  source      text        NOT NULL,
  started_at  timestamptz NOT NULL,
  finished_at timestamptz,
  added       int,
  changed     int,
  unchanged   int,
  removed     int,
  ok          boolean,
  error       text
);
CREATE INDEX sync_runs_source ON sync_runs (source, started_at DESC);
