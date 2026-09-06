-- Live parameterized sources: the registry the engine reads, and the queue it talks over.
--
-- `sources` are pages fetched on a schedule into `chunks`. A query source is different: it
-- takes parameters from the model, runs once, and answers that caller only. Nothing it returns
-- is ever written to the retrieval index.

set lock_timeout = '5s';
set statement_timeout = '10min';

create table query_sources (
    key          text primary key,
    description  text not null,
    -- [{name, description, required, example}] — rendered into the tool description the model
    -- reads. Not JSON Schema: the engine shows these to the model and the worker checks that
    -- the required ones are present, which is all either side needs.
    params       jsonb not null default '[]'::jsonb,
    enabled      boolean not null default true,
    updated_at   timestamptz not null default now()
);

-- The worker claims queued jobs and nothing else, so the index covers only those. Done and
-- failed rows accumulate and would otherwise bloat a full (status, created_at) scan.
create index jobs_queued_idx on jobs (created_at) where status = 'queued';
