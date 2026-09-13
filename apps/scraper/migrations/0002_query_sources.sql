-- Query sources take params from the model, run once, and answer that caller; never indexed.

set lock_timeout = '5s';
set statement_timeout = '10min';

create table query_sources (
    key          text primary key,
    description  text not null,
    -- [{name, description, required, example}] rendered into the tool description; not JSON Schema.
    params       jsonb not null default '[]'::jsonb,
    enabled      boolean not null default true,
    updated_at   timestamptz not null default now()
);

-- jobs.kind = 'source_query' is the contract between engine (QUERY_JOB_KIND) and worker (KIND).
create index jobs_queued_idx on jobs (created_at) where status = 'queued';
