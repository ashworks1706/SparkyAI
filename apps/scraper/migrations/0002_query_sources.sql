-- Live parameterized sources: the registry the engine reads, and the queue it talks over.
--
-- sources are pages fetched on a schedule into chunks. A query source takes parameters from
-- the model, runs once, and answers that caller only. Nothing it returns is ever written to
-- the retrieval index.

set lock_timeout = '5s';
set statement_timeout = '10min';

create table query_sources (
    key          text primary key,
    description  text not null,
    -- [{name, description, required, example}] rendered into the tool description the model
    -- reads. Not JSON Schema: the engine shows these to the model and the worker checks that
    -- the required ones are present.
    params       jsonb not null default '[]'::jsonb,
    enabled      boolean not null default true,
    updated_at   timestamptz not null default now()
);

-- jobs.kind = 'source_query' is the contract between the engine (Rust, QUERY_JOB_KIND) and
-- the worker (Python, worker.KIND). Neither can see the constant of the other.
--
-- The worker claims queued jobs and nothing else; the index covers only those.
create index jobs_queued_idx on jobs (created_at) where status = 'queued';
